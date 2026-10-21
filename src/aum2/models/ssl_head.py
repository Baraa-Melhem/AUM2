"""Layer-weighted classification head over frozen SSL statistics."""

from __future__ import annotations

import torch
from torch import nn

from aum2.config import SSLConfig


class LayerWeightedHead(nn.Module):
    """``(B, n_layers, D)`` per-layer mean‖std statistics → ``(B, n_classes)`` logits.

    Each layer is standardised with training-fold statistics (buffers ``mean``/``std``), the
    layers are mixed with learned softmax weights, and a small MLP classifies the mixture.
    """

    def __init__(self, n_layers: int, feat_dim: int, n_classes: int, cfg: SSLConfig) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.register_buffer("mean", torch.zeros(n_layers, feat_dim))
        self.register_buffer("std", torch.ones(n_layers, feat_dim))
        self.input_dropout = nn.Dropout(cfg.input_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, n_classes),
        )

    @property
    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.float() - self.mean) / self.std
        mixed = torch.einsum("l,bld->bd", self.layer_weights, self.input_dropout(x))
        return self.mlp(mixed)
