"""Utterance-level acoustic functionals for the classical baseline.

Frame-level descriptors (25 ms windows, 10 ms hop):

* 40 MFCCs with Δ and ΔΔ (spectral envelope and its dynamics)
* 12 chroma bins, 7 spectral-contrast bands
* spectral centroid, bandwidth, roll-off (85 %), flatness; zero-crossing rate; RMS energy (dB)
* F0 (YIN, 60-500 Hz) on voiced frames only

Each descriptor is summarised by seven functionals (mean, std, min, p10, median, p90, max),
plus the voiced-frame ratio and the clip duration: 1 024 features for 40 MFCCs. Scaling is
*not* done here; the classifier pipeline fits its scaler on the training fold only.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

FUNCTIONALS = ("mean", "std", "min", "p10", "p50", "p90", "max")


def _functionals(frames: np.ndarray) -> np.ndarray:
    """``frames`` is (n_descriptors, n_frames); returns (n_descriptors * 7,)."""
    if frames.shape[1] == 0:
        return np.zeros(frames.shape[0] * len(FUNCTIONALS), dtype=np.float32)
    p10, p50, p90 = np.percentile(frames, [10, 50, 90], axis=1)
    stats = [frames.mean(1), frames.std(1), frames.min(1), p10, p50, p90, frames.max(1)]
    return np.stack(stats, axis=1).reshape(-1).astype(np.float32)


def feature_names(n_mfcc: int = 40) -> list[str]:
    rows = (
        [f"mfcc{i}" for i in range(n_mfcc)]
        + [f"mfcc_d{i}" for i in range(n_mfcc)]
        + [f"mfcc_dd{i}" for i in range(n_mfcc)]
        + [f"chroma{i}" for i in range(12)]
        + [f"contrast{i}" for i in range(7)]
        + ["centroid", "bandwidth", "rolloff", "flatness", "zcr", "rms_db"]
    )
    names = [f"{row}_{stat}" for row in rows for stat in FUNCTIONALS]
    names += [f"f0_{stat}" for stat in FUNCTIONALS] + ["voiced_ratio", "duration"]
    return names


def extract_functionals(samples: np.ndarray, sample_rate: int, n_mfcc: int = 40) -> np.ndarray:
    """Feature vector of one conditioned clip (see the module docstring)."""
    import librosa  # local import: only the classical baseline needs librosa at runtime

    n_fft, hop = int(0.025 * sample_rate), int(0.010 * sample_rate)
    power = np.abs(librosa.stft(samples, n_fft=512, hop_length=hop, win_length=n_fft)) ** 2
    mel = librosa.feature.melspectrogram(S=power, sr=sample_rate, n_mels=64)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)
    width = 9 if mfcc.shape[1] >= 9 else (mfcc.shape[1] // 2) * 2 + 1
    deltas = (
        [
            librosa.feature.delta(mfcc, width=width),
            librosa.feature.delta(mfcc, order=2, width=width),
        ]
        if width >= 3
        else [np.zeros_like(mfcc), np.zeros_like(mfcc)]
    )
    magnitude = np.sqrt(power)
    spectral = np.vstack(
        [
            librosa.feature.chroma_stft(S=power, sr=sample_rate, n_fft=512),
            librosa.feature.spectral_contrast(S=magnitude, sr=sample_rate, n_fft=512, fmin=100.0),
            librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate),
            librosa.feature.spectral_bandwidth(S=magnitude, sr=sample_rate),
            librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate, roll_percent=0.85),
            librosa.feature.spectral_flatness(S=magnitude),
            librosa.feature.zero_crossing_rate(samples, frame_length=n_fft, hop_length=hop),
            librosa.amplitude_to_db(librosa.feature.rms(S=magnitude, frame_length=512), ref=1.0),
        ]
    )
    n = min(mfcc.shape[1], spectral.shape[1])
    frame_rows = np.vstack([mfcc[:, :n], deltas[0][:, :n], deltas[1][:, :n], spectral[:, :n]])

    f0 = librosa.yin(
        samples, fmin=60.0, fmax=500.0, sr=sample_rate, frame_length=1024, hop_length=hop
    )
    rms_db = spectral[-1]
    voiced = rms_db[: len(f0)] > (rms_db.max() - 30.0)
    voiced_f0 = f0[: len(voiced)][voiced][None, :]
    extras = np.array([voiced.mean(), len(samples) / sample_rate], dtype=np.float32)
    return np.concatenate([_functionals(frame_rows), _functionals(voiced_f0), extras])


def extract_all(
    waveforms: list[np.ndarray], sample_rate: int, n_mfcc: int = 40, n_jobs: int = -1
) -> np.ndarray:
    """Feature matrix ``(n_clips, n_features)`` computed in parallel processes."""
    rows = Parallel(n_jobs=n_jobs, batch_size=16)(
        delayed(extract_functionals)(w, sample_rate, n_mfcc) for w in waveforms
    )
    return np.nan_to_num(np.stack(rows), nan=0.0, posinf=0.0, neginf=0.0)
