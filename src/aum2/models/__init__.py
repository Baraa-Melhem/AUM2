"""Neural network architectures."""

from __future__ import annotations

import torch
from torch import nn

from aum2.config import AugmentConfig, CRNNConfig, FrontendConfig
from aum2.features.augment import SpecAugment, WaveAugment
from aum2.features.frontend import LogMelFrontend
from aum2.models.crnn import CRNN


class LogMelClassifier(nn.Module):
    """Raw 16 kHz waveforms → logits: front end, (training-only) augmentation, CRNN.

    Bundling the front end with the network means the normalisation statistics are saved in
    the same checkpoint as the weights, and inference cannot use the wrong ones.
    """

    def __init__(
        self,
        frontend_cfg: FrontendConfig,
        augment_cfg: AugmentConfig,
        crnn_cfg: CRNNConfig,
        n_classes: int,
        sample_rate: int,
    ) -> None:
        super().__init__()
        self.frontend = LogMelFrontend(frontend_cfg, sample_rate)
        self.wave_augment = WaveAugment(augment_cfg)
        self.spec_augment = SpecAugment(augment_cfg)
        self.net = CRNN(frontend_cfg.n_mels, n_classes, crnn_cfg)

    def forward(self, wave: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        wave = self.wave_augment(wave, lengths)
        feats, frames = self.frontend(wave, lengths)
        feats, frames = self.spec_augment(feats, frames)
        return self.net(feats, frames)
