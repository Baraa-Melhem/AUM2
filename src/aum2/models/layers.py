"""Building blocks shared by the neural models."""

from __future__ import annotations

import torch
from torch import nn

from aum2.features.frontend import frame_mask


class AttentiveStatsPool(nn.Module):
    """Attentive statistics pooling (Okabe et al., 2018) over valid frames only.

    ``(B, T, D)`` → ``(B, 2D)``: the attention-weighted mean and standard deviation. Padded
    frames get zero weight, so the result does not depend on how much padding a batch has.
    """

    def __init__(self, dim: int, attention_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = frame_mask(lengths, x.shape[1])
        logits = self.score(x).squeeze(-1).float().masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        x = x.float()
        mean = (weights * x).sum(1)
        var = (weights * x.square()).sum(1) - mean.square()
        return torch.cat([mean, var.clamp_min(1e-6).sqrt()], dim=-1)

    @torch.no_grad()
    def attention(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """The ``(B, T)`` attention weights, for inspecting which frames drive a decision."""
        mask = frame_mask(lengths, x.shape[1])
        logits = self.score(x).squeeze(-1).float().masked_fill(~mask, float("-inf"))
        return torch.softmax(logits, dim=1)


class MaskedResBlock(nn.Module):
    """Two 3×3 conv-BN layers with a residual connection; padded frames are re-zeroed.

    Zeroing after every layer means a frame past a clip's end always looks like the convolution's
    own zero padding, so a clip produces identical activations alone or in a padded batch.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Masking the input matters after an odd-length pooling, whose last pooled frame mixes a
        # valid frame with padding.
        x = x * mask
        h = self.act(self.bn1(self.conv1(x))) * mask
        h = self.bn2(self.conv2(h))
        return self.act(h + self.shortcut(x)) * mask
