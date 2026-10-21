"""Faithful PyTorch port of the 2023 notebook's features, CNNs and cascade.

It exists to answer one question: *how good was the original method when it is evaluated
without leakage?* Everything that defined the method is kept:

* features: ``librosa.feature.melspectrogram`` with default settings (128 bands, 2048-point
  FFT, hop 512) at each file's **native** sample rate, no log, resized to 128×128 bilinear;
* network: Conv2D(16, 3×3) → ReLU → MaxPool(2) → Flatten → Dense(32) → ReLU → Dense(k),
  Glorot-uniform initialisation as in Keras;
* training: Adam (lr 1e-3, ε 1e-7), the original epochs and batch sizes, no regularisation;
* routing: the exact ``models_system`` cascade of five models.

What is deliberately *not* kept are the bugs: duplicated corpora, mislabelled SAVEE, augmenting
before splitting, and testing on training clips. The ``time_stretch(rate=1)`` call is dropped
too; it is an identity up to phase-vocoder artefacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn

from aum2.config import LegacyConfig

# Sub-problems of the cascade, as (class names in output order) per model, in the original
# numbering: M1 negative/positive, M2 negative emotions, M3 positive, M4 sad/neutral,
# M5 angry/happy.
NEGATIVE = ("angry", "disgust", "fear", "sad")
POSITIVE = ("happy", "neutral")
SUBTASKS: tuple[tuple[tuple[str, ...], ...], ...] = (
    (NEGATIVE, POSITIVE),
    (("angry",), ("disgust",), ("fear",), ("sad",)),
    (("happy",), ("neutral",)),
    (("sad",), ("neutral",)),
    (("angry",), ("happy",)),
)


def legacy_features(path: str | Path, cfg: LegacyConfig) -> np.ndarray:
    """128×128 resized *power* mel spectrogram of a raw file at its native sample rate."""
    import librosa

    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mel = librosa.feature.melspectrogram(
        y=samples.mean(axis=1),
        sr=sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
    )
    resized = F.interpolate(
        torch.from_numpy(mel)[None, None],
        size=(cfg.image_size, cfg.image_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy()


class LegacyCNN(nn.Module):
    """The single-conv-layer network every sub-model of the cascade used."""

    def __init__(self, n_classes: int, image_size: int = 128) -> None:
        super().__init__()
        side = (image_size - 2) // 2
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * side * side, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )
        for module in self.net:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1))


def subtask_targets(emotions: np.ndarray, task: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows used by sub-model ``task`` and their sub-labels."""
    groups = SUBTASKS[task]
    lookup = {emotion: k for k, group in enumerate(groups) for emotion in group}
    rows = np.flatnonzero(np.isin(emotions, list(lookup)))
    return rows, np.array([lookup[e] for e in emotions[rows]], dtype=np.int64)


def route(decisions: list[np.ndarray]) -> np.ndarray:
    """Replicates ``models_system``: every sub-model's argmax for every clip → emotion names.

    Note the original quirks: the "positive" branch can end in Angry (via M5) or Sad (via M4),
    and the "negative" branch's Angry and Sad are re-decided by M5 and M4.
    """
    m1, m2, m3, m4, m5 = decisions
    angry_happy = np.where(m5 == 0, "angry", "happy")
    sad_neutral = np.where(m4 == 0, "sad", "neutral")
    negative = np.select(
        [m2 == 0, m2 == 1, m2 == 2], [angry_happy, "disgust", "fear"], default=sad_neutral
    )
    positive = np.where(m3 == 0, angry_happy, sad_neutral)
    return np.where(m1 == 0, negative, positive)
