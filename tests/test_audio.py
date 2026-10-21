import numpy as np
import pytest

from aum2.config import DataConfig
from aum2.data.audio import condition, normalize_loudness, resample, trim_silence


def _rms_dbfs(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_resample_changes_length_proportionally():
    x = np.random.default_rng(0).normal(size=48000).astype(np.float32)
    assert len(resample(x, 48000, 16000)) == 16000
    assert len(resample(x, 16000, 16000)) == 48000


def test_trim_silence_keeps_speech_and_margin():
    sr = 16000
    tone = np.sin(2 * np.pi * 200 * np.arange(sr) / sr).astype(np.float32) * 0.5
    padded = np.concatenate([np.zeros(sr), tone, np.zeros(sr)]).astype(np.float32)
    trimmed = trim_silence(padded, sr, top_db=35.0, margin_ms=100.0)
    assert sr <= len(trimmed) <= sr + int(0.25 * sr)


def test_normalize_loudness_hits_target_and_respects_peak():
    x = np.random.default_rng(1).normal(scale=0.01, size=16000).astype(np.float32)
    assert _rms_dbfs(normalize_loudness(x, -25.0)) == pytest.approx(-25.0, abs=0.05)
    spiky = np.zeros(16000, dtype=np.float32)
    spiky[100] = 0.5
    assert np.abs(normalize_loudness(spiky, -10.0)).max() <= 0.99 + 1e-6


def test_condition_caps_length_and_rejects_wrong_rate():
    cfg = DataConfig(max_seconds=1.0)
    x = np.random.default_rng(2).normal(scale=0.1, size=3 * 16000).astype(np.float32)
    assert len(condition(x, 16000, cfg)) == 16000
    with pytest.raises(ValueError):
        condition(x, 8000, cfg)
