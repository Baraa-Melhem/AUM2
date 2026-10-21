"""GPU log-mel front end with leak-free global mean/variance normalisation.

The mel filterbank is computed here in NumPy (identical to ``librosa.filters.mel`` with its
Slaney defaults, which a unit test checks) so inference does not depend on librosa/numba.

Padding is handled so that a clip gives the same features alone or inside a padded batch:
the STFT pads with zeros (not reflection) and every frame past a clip's end is set to zero
*after* normalisation, i.e. to the training mean.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from aum2.config import FrontendConfig

_LOG_FLOOR = 1e-6


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray:
    """Slaney mel scale: linear below 1 kHz, logarithmic above."""
    freq = np.asarray(freq, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    linear = freq / f_sp
    log = min_log_mel + np.log(np.maximum(freq, min_log_hz) / min_log_hz) / logstep
    return np.where(freq >= min_log_hz, log, linear)


def mel_to_hz(mels: np.ndarray) -> np.ndarray:
    """Inverse of :func:`hz_to_mel`."""
    mels = np.asarray(mels, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    linear = f_sp * mels
    log = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return np.where(mels >= min_log_mel, log, linear)


def mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float
) -> np.ndarray:
    """Slaney-normalised triangular filters, shape ``(n_mels, n_fft // 2 + 1)``."""
    fft_freqs = np.linspace(0.0, sample_rate / 2, 1 + n_fft // 2)
    mel_points = mel_to_hz(np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2))
    widths = np.diff(mel_points)
    ramps = mel_points[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_points[2:] - mel_points[:-2]))[:, None]
    return weights.astype(np.float32)


def frame_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Boolean ``(B, max_len)`` mask that is True on valid frames."""
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class LogMelFrontend(nn.Module):
    """Waveform ``(B, L)`` → normalised log-mel ``(B, n_mels, T)`` plus valid frame counts.

    ``mean``/``std`` are buffers, so they travel inside every checkpoint; they stay at 0/1
    until :meth:`set_normalization` is called with statistics of the *training* fold.
    """

    def __init__(self, cfg: FrontendConfig, sample_rate: int) -> None:
        super().__init__()
        self.n_fft, self.hop, self.win = cfg.n_fft, cfg.hop_length, cfg.win_length
        self.n_mels = cfg.n_mels
        fb = mel_filterbank(sample_rate, cfg.n_fft, cfg.n_mels, cfg.f_min, cfg.f_max)
        self.register_buffer("mel_fb", torch.from_numpy(fb), persistent=False)
        self.register_buffer("window", torch.hann_window(cfg.win_length), persistent=False)
        self.register_buffer("mean", torch.zeros(cfg.n_mels))
        self.register_buffer("std", torch.ones(cfg.n_mels))

    def frame_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return 1 + torch.div(lengths, self.hop, rounding_mode="floor")

    def log_mel(self, wave: torch.Tensor) -> torch.Tensor:
        """Un-normalised log-mel energies (float32 regardless of autocast)."""
        with torch.autocast(device_type=wave.device.type, enabled=False):
            spec = torch.stft(
                wave.float(),
                self.n_fft,
                self.hop,
                self.win,
                self.window,
                center=True,
                pad_mode="constant",
                return_complex=True,
            )
            power = spec.real.square() + spec.imag.square()
            return torch.log(torch.matmul(self.mel_fb, power) + _LOG_FLOOR)

    def forward(
        self, wave: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.log_mel(wave)
        frames = self.frame_lengths(lengths).clamp(max=feats.shape[-1])
        feats = (feats - self.mean[:, None]) / self.std[:, None]
        return feats * frame_mask(frames, feats.shape[-1])[:, None, :], frames

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-3))


@torch.no_grad()
def fit_normalization(
    frontend: LogMelFrontend, batches: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-mel-bin mean/std over every valid frame of the given (training-only) batches."""
    total = torch.zeros(frontend.n_mels, dtype=torch.float64, device=frontend.mel_fb.device)
    total_sq, count = torch.zeros_like(total), 0
    for wave, lengths in batches:
        feats = frontend.log_mel(wave)
        mask = frame_mask(
            frontend.frame_lengths(lengths).clamp(max=feats.shape[-1]), feats.shape[-1]
        )
        valid = feats.transpose(1, 2)[mask].double()
        total += valid.sum(0)
        total_sq += valid.square().sum(0)
        count += valid.shape[0]
    mean = total / count
    std = (total_sq / count - mean.square()).clamp_min(0).sqrt()
    return mean.float(), std.float()
