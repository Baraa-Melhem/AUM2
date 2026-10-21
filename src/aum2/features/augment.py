"""Batched GPU augmentation, used on training batches only.

The 2023 notebook "augmented" by duplicating clips before the train/test split, so copies of
one recording landed on both sides. Here augmentation is a random function applied to each
training batch as it is drawn: validation and test clips are never augmented, and no
augmented copy is ever stored.

* :class:`WaveAugment`: random gain and coloured noise at a random SNR (robustness to
  microphones and rooms).
* :class:`SpecAugment`: tempo perturbation (speaking rate) and vocal-tract-length
  perturbation (formant/pitch shift, which simulates unseen speakers) as per-sample resampling
  of the time and mel axes, followed by SpecAugment frequency and time masks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from aum2.config import AugmentConfig
from aum2.features.frontend import frame_mask


def _uniform(n: int, low: float, high: float, device: torch.device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(low, high)


def _coloured_noise(shape: tuple[int, int], exponent: torch.Tensor) -> torch.Tensor:
    """Noise with a ``1/f**exponent`` power spectrum (0 white, 1 pink, 2 brown), unit RMS."""
    white = torch.randn(shape, device=exponent.device)
    spectrum = torch.fft.rfft(white, dim=-1)
    freqs = torch.arange(spectrum.shape[-1], device=exponent.device).clamp_min(1).float()
    spectrum = spectrum / freqs[None, :] ** (exponent[:, None] / 2)
    noise = torch.fft.irfft(spectrum, n=shape[-1], dim=-1)
    return noise / noise.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


class WaveAugment(nn.Module):
    """Random gain plus additive white/pink/brown noise, restricted to each clip's samples."""

    def __init__(self, cfg: AugmentConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, wave: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.cfg.enabled:
            return wave
        batch, device = wave.shape[0], wave.device
        valid = frame_mask(lengths, wave.shape[-1]).float()
        gain_db = _uniform(batch, -self.cfg.gain_db, self.cfg.gain_db, device)
        wave = wave * (10.0 ** (gain_db / 20.0))[:, None]

        signal_power = (wave.square() * valid).sum(-1) / lengths.clamp_min(1)
        snr_db = _uniform(batch, self.cfg.snr_db[0], self.cfg.snr_db[1], device)
        noise_rms = (signal_power / 10.0 ** (snr_db / 10.0)).sqrt()
        apply = (torch.rand(batch, device=device) < self.cfg.noise_prob).float()
        exponent = torch.randint(0, 3, (batch,), device=device).float()
        noise = _coloured_noise(tuple(wave.shape), exponent)
        return wave + noise * (noise_rms * apply)[:, None] * valid


class SpecAugment(nn.Module):
    """Tempo and VTLP warps followed by frequency/time masking of normalised log-mels.

    Masked regions are set to 0, i.e. the training mean after normalisation. Returns the
    warped features and the new valid frame counts.
    """

    def __init__(self, cfg: AugmentConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self, feats: torch.Tensor, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or not self.cfg.enabled:
            return feats, frames
        cfg, device = self.cfg, feats.device
        batch, n_mels, n_frames = feats.shape

        tempo = torch.where(
            torch.rand(batch, device=device) < cfg.tempo_prob,
            _uniform(batch, *cfg.tempo_range, device),
            torch.ones(batch, device=device),
        )
        alpha = torch.where(
            torch.rand(batch, device=device) < cfg.vtlp_prob,
            _uniform(batch, *cfg.vtlp_range, device),
            torch.ones(batch, device=device),
        )
        # Output frame t reads input frame t * tempo; output bin m reads input bin m * alpha.
        new_frames = (frames.float() / tempo).floor().clamp(1, n_frames).long()
        t_src = torch.arange(n_frames, device=device)[None, :] * tempo[:, None]
        m_src = torch.arange(n_mels, device=device)[None, :] * alpha[:, None]
        grid_x = (2 * t_src / max(n_frames - 1, 1) - 1)[:, None, :].expand(batch, n_mels, n_frames)
        grid_y = (2 * m_src / max(n_mels - 1, 1) - 1)[:, :, None].expand(batch, n_mels, n_frames)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        warped = F.grid_sample(
            feats[:, None], grid, mode="bilinear", padding_mode="border", align_corners=True
        )[:, 0]
        warped = warped * frame_mask(new_frames, n_frames)[:, None, :]

        keep = torch.ones_like(warped)
        bins = torch.arange(n_mels, device=device)[None, :]
        for _ in range(cfg.freq_masks):
            width = torch.randint(0, cfg.freq_mask_width + 1, (batch, 1), device=device)
            start = (torch.rand(batch, 1, device=device) * (n_mels - width)).long()
            keep = keep * ~((bins >= start) & (bins < start + width))[:, :, None]
        steps = torch.arange(n_frames, device=device)[None, :]
        max_width = (new_frames.float() * cfg.time_mask_ratio).long()[:, None]
        for _ in range(cfg.time_masks):
            width = (torch.rand(batch, 1, device=device) * (max_width + 1)).long()
            start = (torch.rand(batch, 1, device=device) * (new_frames[:, None] - width)).long()
            keep = keep * ~((steps >= start) & (steps < start + width))[:, None, :]
        return warped * keep, new_frames
