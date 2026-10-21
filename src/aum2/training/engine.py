"""A small, model-agnostic training loop.

Callers provide two callables: ``train_batches(epoch)`` yields batches, and
``forward(model, batch)`` returns ``(logits, labels)``. Everything else is shared by every
neural family: AdamW (no decay on biases/norms), linear warm-up then cosine decay per step,
fp16 autocast with gradient scaling, gradient-norm clipping, an exponential moving average of
the weights, per-epoch validation (UAR / WAR / macro-F1) and optional early stopping.

Validation uses the EMA weights when EMA is on, and the EMA weights are what the run keeps.
With ``patience = 0`` (the default) the model trains for the full cosine schedule and the final
weights are kept, so the validation fold is never used to pick an epoch and the out-of-fold
scores carry no selection bias.
"""

from __future__ import annotations

import contextlib
import copy
import math
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from aum2.config import TrainConfig
from aum2.evaluation.metrics import summary_metrics
from aum2.logging_utils import get_logger

logger = get_logger(__name__)

Batch = Any
ForwardFn = Callable[[nn.Module, Batch], tuple[torch.Tensor, torch.Tensor]]


class EMA:
    """Exponential moving average of all floating-point parameters and buffers.

    The effective decay ramps up as ``min(decay, (1 + t) / (10 + t))`` so that early on the
    average is not dominated by the random initialisation.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.steps = 0
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for key, value in model.state_dict().items():
            if key in self.shadow:
                self.shadow[key].lerp_(value.detach(), 1.0 - decay)

    @contextlib.contextmanager
    def applied(self, model: nn.Module) -> Iterator[None]:
        """Temporarily load the averaged weights into ``model``."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=False)


@dataclass
class FitResult:
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (decay if param.ndim >= 2 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _schedule(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


@torch.no_grad()
def predict_logits(
    model: nn.Module, batches: Iterable[Batch], forward: ForwardFn, amp: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated float32 logits and labels over ``batches`` in eval mode."""
    model.eval()
    logits, labels = [], []
    for batch in batches:
        device_type = next(model.parameters()).device.type
        with torch.autocast(
            device_type, dtype=torch.float16, enabled=amp and device_type == "cuda"
        ):
            out, target = forward(model, batch)
        logits.append(out.float().cpu())
        labels.append(target.cpu())
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def fit(
    model: nn.Module,
    *,
    train_batches: Callable[[int], Iterable[Batch]],
    steps_per_epoch: int,
    forward: ForwardFn,
    loss_fn: nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    val_batches: Callable[[], Iterable[Batch]] | None = None,
    log_prefix: str = "",
) -> FitResult:
    """Train ``model`` in place; on return it holds the weights the run keeps."""
    model.to(device)
    loss_fn.to(device)
    use_amp = cfg.amp and device.type == "cuda"
    optimizer = torch.optim.AdamW(
        _param_groups(model, cfg.weight_decay), lr=cfg.learning_rate, fused=device.type == "cuda"
    )
    total_steps = cfg.epochs * steps_per_epoch
    warmup = max(1, round(cfg.warmup_epochs * steps_per_epoch))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _schedule(total_steps, warmup))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    result, best_score, best_state, stale = FitResult(), -math.inf, None, 0

    for epoch in range(cfg.epochs):
        model.train()
        started, running, seen = time.perf_counter(), 0.0, 0
        for batch in train_batches(epoch):
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits, target = forward(model, batch)
                loss = loss_fn(logits.float(), target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            running += loss.item() * len(target)
            seen += len(target)
        record = {
            "epoch": epoch + 1,
            "train_loss": running / max(seen, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }

        if val_batches is not None:
            with ema.applied(model) if ema is not None else contextlib.nullcontext():
                logits_np, labels_np = predict_logits(model, val_batches(), forward, cfg.amp)
                metrics = summary_metrics(labels_np, logits_np)
                if cfg.patience and metrics["uar"] > best_score:
                    best_score = metrics["uar"]
                    best_state = copy.deepcopy(model.state_dict())
                    result.best_epoch = epoch + 1
            with torch.no_grad():
                val_loss = loss_fn(
                    torch.from_numpy(logits_np).to(device), torch.from_numpy(labels_np).to(device)
                ).item()
            record.update({"val_loss": val_loss, **metrics})
            logger.info(
                "%sepoch %3d | loss %.4f | val loss %.4f UAR %.4f WAR %.4f F1 %.4f | %.1fs",
                log_prefix,
                epoch + 1,
                record["train_loss"],
                val_loss,
                record["uar"],
                record["war"],
                record["macro_f1"],
                record["seconds"],
            )
        else:
            logger.info(
                "%sepoch %3d | loss %.4f | %.1fs",
                log_prefix,
                epoch + 1,
                record["train_loss"],
                record["seconds"],
            )
        result.history.append(record)
        if cfg.patience and val_batches is not None:
            stale = 0 if result.best_epoch == epoch + 1 else stale + 1
            if stale >= cfg.patience:
                logger.info(
                    "%searly stop at epoch %d (best %d)", log_prefix, epoch + 1, result.best_epoch
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif ema is not None:
        model.load_state_dict(ema.shadow, strict=False)
        result.best_epoch = len(result.history)
    else:
        result.best_epoch = len(result.history)
    return result
