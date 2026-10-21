"""Waveform loading and conditioning.

:func:`condition` is the single definition of "what a model sees". It runs when the training
cache is built *and* inside :class:`aum2.inference.EmotionRecognizer`, so a microphone
recording goes through exactly the same steps as a corpus file:

1. down-mix to mono and resample to one rate (the 2023 code kept 16-48 kHz native rates, so the
   mel axis meant something different for every corpus);
2. trim leading/trailing silence (RAVDESS clips carry ~1 s of it);
3. normalise loudness to a fixed RMS level (recording gain differs by corpus and microphone and
   is not an emotion cue), with a peak guard against clipping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from aum2.config import DataConfig


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Read any libsndfile-supported file as mono float32 at ``sample_rate``."""
    samples, native_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(samples.mean(axis=1), native_rate, sample_rate)


def resample(samples: np.ndarray, native_rate: int, sample_rate: int) -> np.ndarray:
    """High-quality (soxr VHQ) resampling; a no-op when the rates already match."""
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if native_rate == sample_rate:
        return samples
    return soxr.resample(samples, native_rate, sample_rate, quality="VHQ").astype(np.float32)


def frame_rms_db(samples: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Per-frame RMS level in dB relative to full scale (frames start every ``hop`` samples)."""
    if len(samples) < frame:
        samples = np.pad(samples, (0, frame - len(samples)))
    n_frames = 1 + (len(samples) - frame) // hop
    strides = (samples.strides[0] * hop, samples.strides[0])
    frames = np.lib.stride_tricks.as_strided(samples, (n_frames, frame), strides, writeable=False)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-10))


def trim_silence(
    samples: np.ndarray, sample_rate: int, top_db: float, margin_ms: float
) -> np.ndarray:
    """Drop leading/trailing audio quieter than ``top_db`` below the loudest 25 ms frame."""
    frame, hop = int(0.025 * sample_rate), int(0.010 * sample_rate)
    if len(samples) < frame:
        return samples
    level = frame_rms_db(samples, frame, hop)
    active = np.flatnonzero(level > level.max() - top_db)
    margin = int(margin_ms * sample_rate / 1000)
    start = max(0, active[0] * hop - margin)
    end = min(len(samples), active[-1] * hop + frame + margin)
    return samples[start:end]


def normalize_loudness(samples: np.ndarray, target_dbfs: float, peak: float = 0.99) -> np.ndarray:
    """Scale to ``target_dbfs`` RMS, then back off if that would push a sample past ``peak``."""
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms < 1e-8:
        return samples
    gain = 10.0 ** (target_dbfs / 20.0) / rms
    max_abs = float(np.max(np.abs(samples)))
    gain = min(gain, peak / max_abs)
    return (samples * gain).astype(np.float32)


def condition(samples: np.ndarray, sample_rate: int, cfg: DataConfig) -> np.ndarray:
    """Trim, loudness-normalise and length-cap a mono waveform already at ``cfg.sample_rate``.

    ``max_seconds`` only caps corpus clips; long recordings at inference are windowed by
    :class:`aum2.inference.EmotionRecognizer` before they reach this function.
    """
    if sample_rate != cfg.sample_rate:
        raise ValueError(f"expected {cfg.sample_rate} Hz audio, got {sample_rate} Hz")
    out = trim_silence(samples, sample_rate, cfg.trim_top_db, cfg.trim_margin_ms)
    if cfg.target_dbfs is not None:
        out = normalize_loudness(out, cfg.target_dbfs)
    return out[: int(cfg.max_seconds * sample_rate)].astype(np.float32)


def load_and_condition(path: str | Path, cfg: DataConfig) -> np.ndarray:
    """Convenience wrapper: :func:`load_audio` followed by :func:`condition`."""
    return condition(load_audio(path, cfg.sample_rate), cfg.sample_rate, cfg)
