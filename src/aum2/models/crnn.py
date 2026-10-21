"""Log-mel CRNN: residual CNN → BiGRU → attentive statistics pooling → classifier.

Design choices, against the 2023 model:

* **Variable length, no resizing.** The old model squeezed every clip to 128 frames, erasing
  speaking rate. Here frames are 10 ms apart for every clip; the convolutions pool time by 8
  (80 ms steps) and the GRU plus masked pooling handle any length.
* **Frequency-local, time-shift-invariant features.** Four residual blocks pool frequency
  80 → 5 bands and time only 8×, instead of one conv layer feeding a
  2-million-weight dense layer tied to absolute positions.
* **Temporal context.** A 2-layer bidirectional GRU models prosodic contours; packed sequences
  keep padding out of both directions.
* **Attentive statistics pooling** lets the model weight the emotionally salient frames
  instead of averaging a whole utterance.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from aum2.config import CRNNConfig
from aum2.features.frontend import frame_mask
from aum2.models.layers import AttentiveStatsPool, MaskedResBlock


class CRNN(nn.Module):
    """``(B, n_mels, T)`` normalised log-mels + valid frame counts → ``(B, n_classes)`` logits."""

    def __init__(self, n_mels: int, n_classes: int, cfg: CRNNConfig) -> None:
        super().__init__()
        self.time_pool = cfg.time_pool
        channels = (1, *cfg.channels)
        self.blocks = nn.ModuleList(
            MaskedResBlock(channels[i], channels[i + 1]) for i in range(len(cfg.channels))
        )
        self.pools = nn.ModuleList(nn.MaxPool2d((2, t), (2, t)) for t in cfg.time_pool)
        freq_out = n_mels // 2 ** len(cfg.channels)
        self.project = nn.Sequential(
            nn.Linear(cfg.channels[-1] * freq_out, cfg.proj_dim),
            nn.LayerNorm(cfg.proj_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.gru = nn.GRU(
            cfg.proj_dim,
            cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        self.pool = AttentiveStatsPool(2 * cfg.gru_hidden, cfg.attention_dim)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(4 * cfg.gru_hidden, n_classes))

    def encode(
        self, feats: torch.Tensor, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Frame-level BiGRU outputs ``(B, T', 2H)`` and their valid lengths."""
        x = feats.unsqueeze(1)
        lengths = frames
        for block, pool, t in zip(self.blocks, self.pools, self.time_pool, strict=True):
            x = block(x, frame_mask(lengths, x.shape[-1])[:, None, None, :].to(x.dtype))
            x = pool(x)
            lengths = torch.div(lengths, t, rounding_mode="floor").clamp_min(1)
        batch, channels, freq, steps = x.shape
        x = self.project(x.permute(0, 3, 1, 2).reshape(batch, steps, channels * freq))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=steps)
        return out, lengths

    def forward(self, feats: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        out, lengths = self.encode(feats, frames)
        return self.head(self.pool(out, lengths))
