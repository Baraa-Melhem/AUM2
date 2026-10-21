"""Model families behind one interface: fit on training rows, predict probabilities, save.

Every learner receives row indices into the shared :class:`~aum2.data.dataset.CorpusData`
and fits *everything* that depends on data (normalisation statistics, scalers, class
weights) on the training rows only.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path

import joblib
import numpy as np
import torch
from joblib import Parallel, delayed
from safetensors.torch import save_file
from torch import nn

from aum2.config import ExperimentConfig
from aum2.data.dataset import CorpusData
from aum2.evaluation.metrics import softmax, summary_metrics
from aum2.features.frontend import fit_normalization
from aum2.logging_utils import get_logger
from aum2.models import LogMelClassifier
from aum2.models.legacy import SUBTASKS, LegacyCNN, legacy_features, route, subtask_targets
from aum2.models.ssl_head import LayerWeightedHead
from aum2.training.engine import fit, predict_logits
from aum2.training.losses import build_loss

logger = get_logger(__name__)

WaveBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_EVAL_BUCKET = 8000  # samples (0.5 s at 16 kHz)


class Learner(ABC):
    """Common interface of every model family."""

    def __init__(self, cfg: ExperimentConfig, corpus: CorpusData, device: torch.device) -> None:
        self.cfg, self.corpus, self.device = cfg, corpus, device
        self.classes = corpus.classes
        self.labels = corpus.labels

    @abstractmethod
    def fit(self, train_idx: np.ndarray, val_idx: np.ndarray, log_prefix: str = "") -> list[dict]:
        """Train on ``train_idx``; ``val_idx`` is only monitored. Returns per-epoch history."""

    @abstractmethod
    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        """``(len(idx), n_classes)`` class probabilities in the order of ``idx``."""

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Write everything inference needs into ``directory``."""


# --------------------------------------------------------------------------- waveforms → CRNN


class WaveformBatcher:
    """Assembles padded GPU batches from the int16 waveform store held in RAM."""

    def __init__(self, corpus: CorpusData, device: torch.device) -> None:
        self.samples = np.asarray(corpus.samples)
        self.offsets = corpus.offsets
        self.lengths = corpus.lengths
        self.labels = corpus.labels
        self.device = device

    def batch(
        self, idx: np.ndarray, crop: int | None = None, rng: np.random.Generator | None = None
    ) -> WaveBatch:
        """Training batches (``crop`` set) are random crops padded to exactly ``crop`` samples,
        so every training step has the same shape and cuDNN autotuning pays off. Evaluation
        batches keep whole clips, padded up to a multiple of 0.5 s to limit distinct shapes;
        masking makes the padding invisible to the model."""
        lengths = self.lengths[idx]
        starts = self.offsets[idx].copy()
        if crop is not None:
            assert rng is not None
            slack = np.maximum(lengths - crop, 0)
            starts += (rng.random(len(idx)) * (slack + 1)).astype(np.int64)
            lengths = np.minimum(lengths, crop)
            width = crop
        else:
            width = int(np.ceil(lengths.max() / _EVAL_BUCKET) * _EVAL_BUCKET)
        buffer = np.zeros((len(idx), width), dtype=np.int16)
        for row, (start, length) in enumerate(zip(starts, lengths, strict=True)):
            buffer[row, :length] = self.samples[start : start + length]
        wave = torch.from_numpy(buffer).to(self.device, non_blocking=True).float() / 32767.0
        return (
            wave,
            torch.from_numpy(lengths).to(self.device),
            torch.from_numpy(self.labels[idx]).to(self.device),
        )

    def train_epoch(
        self, idx: np.ndarray, batch_size: int, crop: int, rng: np.random.Generator
    ) -> Iterator[WaveBatch]:
        order = rng.permutation(idx)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            yield self.batch(order[start : start + batch_size], crop, rng)

    def sorted_order(self, idx: np.ndarray) -> np.ndarray:
        """Positions of ``idx`` sorted by length (padding-efficient evaluation)."""
        return np.argsort(self.lengths[idx], kind="stable")

    def eval_batches(self, idx: np.ndarray, batch_size: int) -> Iterator[WaveBatch]:
        ordered = idx[self.sorted_order(idx)]
        for start in range(0, len(ordered), batch_size):
            yield self.batch(ordered[start : start + batch_size])


def _wave_forward(model: nn.Module, batch: WaveBatch) -> tuple[torch.Tensor, torch.Tensor]:
    wave, lengths, labels = batch
    return model(wave, lengths), labels


class CRNNLearner(Learner):
    """Log-mel CRNN trained end-to-end on raw waveforms with GPU augmentation."""

    def __init__(self, cfg: ExperimentConfig, corpus: CorpusData, device: torch.device) -> None:
        super().__init__(cfg, corpus, device)
        self.batcher = WaveformBatcher(corpus, device)
        torch.backends.cudnn.benchmark = not cfg.train.deterministic
        self.model = LogMelClassifier(
            cfg.frontend, cfg.augment, cfg.model.crnn, len(self.classes), corpus.sample_rate
        ).to(device)

    def fit(self, train_idx: np.ndarray, val_idx: np.ndarray, log_prefix: str = "") -> list[dict]:
        cfg, train = self.cfg, self.cfg.train
        stats_batches = (b[:2] for b in self.batcher.eval_batches(train_idx, train.eval_batch_size))
        mean, std = fit_normalization(self.model.frontend, list(stats_batches))
        self.model.frontend.set_normalization(mean, std)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "%sCRNN with %.2f M parameters, %d train / %d val clips",
            log_prefix,
            n_params / 1e6,
            len(train_idx),
            len(val_idx),
        )
        rng = np.random.default_rng(cfg.seed)
        crop = int(cfg.augment.crop_seconds * self.corpus.sample_rate)
        result = fit(
            self.model,
            train_batches=lambda _: self.batcher.train_epoch(
                train_idx, train.batch_size, crop, rng
            ),
            steps_per_epoch=len(train_idx) // train.batch_size,
            forward=_wave_forward,
            loss_fn=build_loss(train, self.labels[train_idx], len(self.classes)),
            cfg=train,
            device=self.device,
            val_batches=lambda: self.batcher.eval_batches(val_idx, train.eval_batch_size),
            log_prefix=log_prefix,
        )
        return result.history

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        batches = self.batcher.eval_batches(idx, self.cfg.train.eval_batch_size)
        logits, _ = predict_logits(self.model, batches, _wave_forward, self.cfg.train.amp)
        probs = np.empty_like(logits)
        probs[self.batcher.sorted_order(idx)] = softmax(logits)
        return probs

    def save(self, directory: Path) -> None:
        save_file(self.model.state_dict(), str(directory / "model.safetensors"))


# --------------------------------------------------------------------------- frozen SSL → head


class SSLHeadLearner(Learner):
    """Layer-weighted MLP head on cached, frozen SSL statistics (the whole fold on the device)."""

    def __init__(
        self, cfg: ExperimentConfig, corpus: CorpusData, device: torch.device, features: np.ndarray
    ) -> None:
        super().__init__(cfg, corpus, device)
        self.features = features
        n_layers, feat_dim = features.shape[1:]
        self.model = LayerWeightedHead(n_layers, feat_dim, len(self.classes), cfg.model.ssl).to(
            device
        )

    def _gpu(self, idx: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.features[np.sort(idx)])).to(self.device)

    def fit(self, train_idx: np.ndarray, val_idx: np.ndarray, log_prefix: str = "") -> list[dict]:
        train = self.cfg.train
        train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)
        x_train, x_val = self._gpu(train_idx), self._gpu(val_idx)
        y_train = torch.from_numpy(self.labels[train_idx]).to(self.device)
        y_val = torch.from_numpy(self.labels[val_idx]).to(self.device)
        total = torch.zeros(x_train.shape[1:], dtype=torch.float64, device=self.device)
        total_sq = torch.zeros_like(total)
        for chunk in x_train.split(1024):
            total += chunk.double().sum(0)
            total_sq += chunk.double().square().sum(0)
        mean = total / len(x_train)
        std = (total_sq / len(x_train) - mean.square()).clamp_min(0).sqrt()
        self.model.set_normalization(mean.float(), std.float())
        generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)

        def train_batches(_: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            order = torch.randperm(len(train_idx), device=self.device, generator=generator)
            for start in range(0, len(order) - train.batch_size + 1, train.batch_size):
                rows = order[start : start + train.batch_size]
                yield x_train[rows], y_train[rows]

        def val_batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for start in range(0, len(val_idx), train.eval_batch_size):
                yield (
                    x_val[start : start + train.eval_batch_size],
                    y_val[start : start + train.eval_batch_size],
                )

        result = fit(
            self.model,
            train_batches=train_batches,
            steps_per_epoch=len(train_idx) // train.batch_size,
            forward=lambda model, batch: (model(batch[0]), batch[1]),
            loss_fn=build_loss(train, self.labels[train_idx], len(self.classes)),
            cfg=train,
            device=self.device,
            val_batches=val_batches,
            log_prefix=log_prefix,
        )
        return result.history

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        self.model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(idx), 512):
                chunk = np.asarray(self.features[idx[start : start + 512]])
                out.append(
                    self.model(torch.from_numpy(chunk).to(self.device)).float().cpu().numpy()
                )
        return softmax(np.concatenate(out))

    def save(self, directory: Path) -> None:
        save_file(self.model.state_dict(), str(directory / "model.safetensors"))
        (directory / "layer_weights.json").write_text(
            json.dumps(self.model.layer_weights.detach().cpu().tolist()), encoding="utf-8"
        )


# --------------------------------------------------------------------------- functionals → sklearn


class HandcraftedLearner(Learner):
    """Standardised acoustic functionals → logistic regression or LightGBM."""

    def __init__(
        self, cfg: ExperimentConfig, corpus: CorpusData, device: torch.device, features: np.ndarray
    ) -> None:
        super().__init__(cfg, corpus, device)
        self.features = features
        self.pipeline = self._build()

    def _build(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        hc = self.cfg.model.handcrafted
        if hc.classifier == "logreg":
            clf = LogisticRegression(C=hc.logreg_c, max_iter=5000, class_weight="balanced")
            return make_pipeline(StandardScaler(), clf)
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=hc.lgbm_estimators,
            learning_rate=hc.lgbm_learning_rate,
            num_leaves=hc.lgbm_num_leaves,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            class_weight="balanced",
            random_state=self.cfg.seed,
            verbose=-1,
            n_jobs=-1,
        )
        return make_pipeline(StandardScaler(), clf)

    def fit(self, train_idx: np.ndarray, val_idx: np.ndarray, log_prefix: str = "") -> list[dict]:
        self.pipeline.fit(self.features[train_idx], self.labels[train_idx])
        metrics = summary_metrics(self.labels[val_idx], self.predict_proba(val_idx))
        logger.info(
            "%sval UAR %.4f WAR %.4f F1 %.4f",
            log_prefix,
            metrics["uar"],
            metrics["war"],
            metrics["macro_f1"],
        )
        return [{"epoch": 1, **metrics}]

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(self.features[idx])

    def save(self, directory: Path) -> None:
        joblib.dump(self.pipeline, directory / "pipeline.joblib")


# --------------------------------------------------------------------------- 2023 cascade


class LegacyCascadeLearner(Learner):
    """The original five-CNN cascade, trained with the original recipe on leak-free splits."""

    def __init__(
        self, cfg: ExperimentConfig, corpus: CorpusData, device: torch.device, features: np.ndarray
    ) -> None:
        super().__init__(cfg, corpus, device)
        self.features = features
        self.emotions = corpus.manifest["emotion"].to_numpy()
        size = cfg.model.legacy.image_size
        self.models = [LegacyCNN(len(groups), size).to(device) for groups in SUBTASKS]

    def fit(self, train_idx: np.ndarray, val_idx: np.ndarray, log_prefix: str = "") -> list[dict]:
        legacy = self.cfg.model.legacy
        rng = np.random.default_rng(self.cfg.seed)
        loss_fn = nn.CrossEntropyLoss()
        for task, model in enumerate(self.models):
            rows, targets = subtask_targets(self.emotions[train_idx], task)
            x = torch.from_numpy(np.asarray(self.features[train_idx[rows]])).to(self.device)
            y = torch.from_numpy(targets).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=legacy.learning_rate, eps=1e-7)
            model.train()
            for epoch in range(legacy.epochs[task]):
                order = torch.from_numpy(rng.permutation(len(rows))).to(self.device)
                total = 0.0
                for start in range(0, len(order), legacy.batch_sizes[task]):
                    batch = order[start : start + legacy.batch_sizes[task]]
                    loss = loss_fn(model(x[batch]), y[batch])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(batch)
                logger.info(
                    "%sM%d epoch %2d | loss %.4f",
                    log_prefix,
                    task + 1,
                    epoch + 1,
                    total / len(rows),
                )
            del x, y
        metrics = summary_metrics(self.labels[val_idx], self.predict_proba(val_idx))
        logger.info(
            "%scascade val UAR %.4f WAR %.4f F1 %.4f",
            log_prefix,
            metrics["uar"],
            metrics["war"],
            metrics["macro_f1"],
        )
        return [{"epoch": 1, **metrics}]

    @torch.no_grad()
    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        decisions = [[] for _ in self.models]
        for model in self.models:
            model.eval()
        for start in range(0, len(idx), 256):
            x = torch.from_numpy(np.asarray(self.features[idx[start : start + 256]])).to(
                self.device
            )
            for k, model in enumerate(self.models):
                decisions[k].append(model(x).argmax(1).cpu().numpy())
        names = route([np.concatenate(d) for d in decisions])
        lookup = {name: i for i, name in enumerate(self.classes)}
        return np.eye(len(self.classes))[[lookup[n] for n in names]]

    def save(self, directory: Path) -> None:
        state = {
            f"m{k + 1}.{name}": value
            for k, model in enumerate(self.models)
            for name, value in model.state_dict().items()
        }
        save_file(state, str(directory / "model.safetensors"))


# --------------------------------------------------------------------------- factory


def _cached(path: Path, build: Callable[[], np.ndarray]) -> np.ndarray:
    if path.exists():
        return np.load(path, mmap_mode="r")
    path.parent.mkdir(parents=True, exist_ok=True)
    array = build()
    np.save(path, array)
    return np.load(path, mmap_mode="r")


def learner_factory(
    cfg: ExperimentConfig, corpus: CorpusData, device: torch.device, n_jobs: int = -1
) -> Callable[[], Learner]:
    """Prepare (and cache) the family's input features once; return a fresh-learner factory."""
    family, cache = cfg.model.name, Path(cfg.data.cache_dir)
    if family == "crnn":
        return lambda: CRNNLearner(cfg, corpus, device)
    if family == "ssl_head":
        from aum2.features.ssl import SSLEncoder, cache_path, load_or_extract

        features = load_or_extract(
            lambda: SSLEncoder(cfg.model.ssl.backbone, device),
            corpus.waveform,
            len(corpus),
            cache_path(cache, cfg.model.ssl.backbone, corpus.key),
        )
        return lambda: SSLHeadLearner(cfg, corpus, device, features)
    if family == "handcrafted":
        from aum2.features.handcrafted import extract_all

        n_mfcc = cfg.model.handcrafted.n_mfcc
        logger.info("Hand-crafted functionals (cached after the first run) ...")
        features = _cached(
            cache / f"handcrafted_mfcc{n_mfcc}_{corpus.key}.npy",
            lambda: extract_all(
                [corpus.waveform(i) for i in range(len(corpus))], corpus.sample_rate, n_mfcc, n_jobs
            ),
        )
        return lambda: HandcraftedLearner(cfg, corpus, device, features)
    if family == "legacy_cascade":
        legacy, raw = cfg.model.legacy, Path(cfg.data.raw_dir)
        feature_params = {
            k: getattr(legacy, k) for k in ("n_mels", "n_fft", "hop_length", "image_size")
        }
        tag = hashlib.sha256(json.dumps(feature_params, sort_keys=True).encode()).hexdigest()[:10]
        logger.info("Legacy 128x128 power-mel images from the raw files (cached) ...")
        features = _cached(
            cache / f"legacy_{tag}_{corpus.key}.npy",
            lambda: np.stack(
                Parallel(n_jobs=n_jobs, batch_size=32)(
                    delayed(legacy_features)(raw / p, legacy) for p in corpus.manifest["path"]
                )
            ).astype(np.float32),
        )
        return lambda: LegacyCascadeLearner(cfg, corpus, device, features)
    raise ValueError(f"unknown model family {family!r}")
