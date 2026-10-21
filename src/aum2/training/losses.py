"""Classification losses with class re-weighting and label smoothing."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from aum2.config import TrainConfig


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights ``N / (C · n_c)``: each class contributes equally to the loss."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"every class needs at least one training example, got counts {counts}")
    return torch.tensor(len(labels) / (n_classes * counts), dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) with optional class weights and smoothing."""

    def __init__(self, gamma: float, weight: torch.Tensor | None, label_smoothing: float) -> None:
        super().__init__()
        self.gamma, self.label_smoothing = gamma, label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits.float(), dim=-1)
        n_classes = logits.shape[-1]
        soft = F.one_hot(target, n_classes).float()
        soft = soft * (1 - self.label_smoothing) + self.label_smoothing / n_classes
        focal = (1 - log_probs.exp()).pow(self.gamma)
        per_class = -(soft * focal * log_probs)
        if self.weight is not None:
            per_class = per_class * self.weight
            return per_class.sum(-1).sum() / self.weight[target].sum()
        return per_class.sum(-1).mean()


def build_loss(cfg: TrainConfig, train_labels: np.ndarray, n_classes: int) -> nn.Module:
    """Loss selected by ``cfg.loss``; weights come from the *training* labels only."""
    weight = class_weights(train_labels, n_classes) if cfg.class_weighting == "balanced" else None
    if cfg.loss == "focal":
        return FocalLoss(cfg.focal_gamma, weight, cfg.label_smoothing)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
