"""Shared fixtures: a miniature copy of all four corpora with their real naming conventions.

Each file is a short synthetic "voice" (a harmonic tone whose pitch depends on the emotion,
with silence on both ends) written at the corpus's native sample rate, so the tests exercise
parsing, resampling, trimming, caching, splitting, training and inference without the 1.5 GB
of real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aum2.config import ExperimentConfig, config_from_dict, to_dict

EMOTION_PITCH = {
    "angry": 220,
    "disgust": 140,
    "fear": 260,
    "happy": 240,
    "neutral": 160,
    "sad": 120,
}


def synthetic_voice(
    emotion: str, sample_rate: int, rng: np.random.Generator, seconds: float | None = None
) -> np.ndarray:
    """Harmonic tone + noise with 0.2 s of near-silence before and after."""
    seconds = seconds or rng.uniform(0.9, 1.5)
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    f0 = EMOTION_PITCH.get(emotion, 180) * rng.uniform(0.95, 1.05)
    voice = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    voice *= np.hanning(len(t)) * 0.3
    pad = np.zeros(int(0.2 * sample_rate))
    signal = np.concatenate([pad, voice, pad]) + rng.normal(0, 1e-4, 2 * len(pad) + len(t))
    return signal.astype(np.float32)


def _write(path: Path, emotion: str, sample_rate: int, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, synthetic_voice(emotion, sample_rate, rng), sample_rate, subtype="PCM_16")


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    codes = {
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad",
    }

    # CREMA-D: 6 actors x 2 sentences x 6 emotions, 16 kHz, plus the demographics table.
    sexes = {
        1001: "Male",
        1002: "Female",
        1003: "Male",
        1004: "Female",
        1005: "Male",
        1006: "Female",
    }
    (root / "cremad").mkdir()
    (root / "cremad" / "VideoDemographics.csv").write_text(
        "ActorID,Age,Sex,Race,Ethnicity\n"
        + "".join(f"{a},30,{s},Caucasian,Not Hispanic\n" for a, s in sexes.items()),
        encoding="utf-8",
    )
    for actor in sexes:
        for sentence in ("DFA", "IEO"):
            for code, emotion in codes.items():
                _write(
                    root / f"cremad/AudioWAV/{actor}_{sentence}_{code}_XX.wav", emotion, 16000, rng
                )

    # RAVDESS: 4 actors x 8 emotions (incl. calm/surprise, which are dropped), 48 kHz,
    # plus one song file that must be ignored.
    ravdess = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprise",
    }
    for actor in range(1, 5):
        for code, emotion in ravdess.items():
            name = f"03-01-{code}-01-01-01-{actor:02d}.wav"
            _write(root / f"ravdess/Actor_{actor:02d}/{name}", emotion, 48000, rng)
    _write(root / "ravdess/Actor_01/03-02-03-01-01-01-01.wav", "happy", 48000, rng)

    # TESS: 2 speakers x 3 words x 7 emotions ("ps" = surprise), 24.414 kHz.
    tess = ["angry", "disgust", "fear", "happy", "neutral", "sad", "ps"]
    for speaker in ("OAF", "YAF"):
        for word in ("back", "bar", "base"):
            for emotion in tess:
                _write(root / f"tess/{speaker}_{word}_{emotion}.wav", emotion, 24414, rng)

    # SAVEE: 4 speakers x 7 emotions x 2 sentences, 44.1 kHz.
    savee = {
        "a": "angry",
        "d": "disgust",
        "f": "fear",
        "h": "happy",
        "n": "neutral",
        "sa": "sad",
        "su": "surprise",
    }
    for speaker in ("DC", "JE", "JK", "KL"):
        for code, emotion in savee.items():
            for idx in (1, 2):
                _write(root / f"savee/AudioData/{speaker}_{code}{idx:02d}.wav", emotion, 44100, rng)
    return root


def tiny_config(raw_dir: Path, tmp_path: Path, family: str, **sections: dict) -> ExperimentConfig:
    """A seconds-fast configuration on the synthetic corpus."""
    tree = to_dict(ExperimentConfig())
    tree.update(name=f"tiny_{family}", output_dir=str(tmp_path / "runs"))
    tree["data"].update(raw_dir=str(raw_dir), cache_dir=str(tmp_path / "cache"))
    tree["split"].update(n_folds=2)
    tree["frontend"].update(n_mels=32)
    tree["augment"].update(crop_seconds=1.0)
    tree["model"]["name"] = family
    tree["model"]["crnn"].update(channels=[8, 8, 8, 8], proj_dim=16, gru_hidden=8, attention_dim=8)
    tree["model"]["ssl"].update(hidden_dim=16)
    tree["model"]["legacy"].update(epochs=[1, 1, 1, 1, 1], image_size=32, n_mels=32)
    tree["train"].update(
        epochs=2, batch_size=8, eval_batch_size=16, warmup_epochs=0.5, device="cpu"
    )
    for section, values in sections.items():
        tree[section].update(values)
    return config_from_dict(tree)
