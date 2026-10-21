"""File / waveform / microphone inference with a trained run's fold ensemble.

The 2023 assistant recorded 20 s and squeezed it into a 128-frame image, an input ten times
more compressed than anything seen in training. Here a recording is conditioned exactly like
the training clips (:func:`aum2.data.audio.condition`). Recordings longer than
``max_single_seconds`` are cut into overlapping windows; silent windows are skipped. The
result is the mean of the window probabilities plus a per-window timeline.

Example::

    from aum2.inference import EmotionRecognizer

    recognizer = EmotionRecognizer("runs/wavlm")
    print(recognizer.predict_file("clip.wav"))
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from aum2.config import ExperimentConfig, load_config
from aum2.data.audio import condition, frame_rms_db, load_audio, resample
from aum2.evaluation.calibration import apply_temperature, load_temperature
from aum2.evaluation.metrics import softmax
from aum2.logging_utils import get_logger
from aum2.reproducibility import resolve_device

logger = get_logger(__name__)

# A backend maps conditioned segments to per-fold probabilities, (n_folds, n_segments, n_classes).
Backend = Callable[[list[np.ndarray]], np.ndarray]


@dataclass(frozen=True)
class SegmentPrediction:
    """Emotion probabilities of one analysis window (times in seconds of the input)."""

    start: float
    end: float
    label: str
    probabilities: dict[str, float]


@dataclass(frozen=True)
class Prediction:
    """Utterance-level result: top label, its probability, the full distribution, the timeline."""

    label: str
    confidence: float
    probabilities: dict[str, float]
    duration: float
    segments: tuple[SegmentPrediction, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        ranked = sorted(self.probabilities.items(), key=lambda item: -item[1])
        bars = "\n".join(f"  {name:<8} {p:6.1%} {'#' * round(30 * p)}" for name, p in ranked)
        return f"{self.label} ({self.confidence:.1%}) over {self.duration:.1f} s\n{bars}"


def _load_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path), device=str(device))


def _crnn_backend(
    cfg: ExperimentConfig, fold_dirs: list[Path], n_classes: int, device: torch.device
) -> Backend:
    from aum2.models import LogMelClassifier

    models = []
    for fold_dir in fold_dirs:
        model = LogMelClassifier(
            cfg.frontend, cfg.augment, cfg.model.crnn, n_classes, cfg.data.sample_rate
        )
        model.load_state_dict(_load_state(fold_dir / "model.safetensors", device), strict=True)
        models.append(model.to(device).eval())

    @torch.no_grad()
    def predict_chunk(segments: list[np.ndarray]) -> np.ndarray:
        lengths = torch.tensor([len(s) for s in segments], device=device)
        wave = torch.zeros(len(segments), int(lengths.max()), device=device)
        for row, segment in enumerate(segments):
            wave[row, : len(segment)] = torch.from_numpy(segment).to(device)
        return np.stack([softmax(m(wave, lengths).float().cpu().numpy()) for m in models])

    def predict(segments: list[np.ndarray]) -> np.ndarray:
        # Bounded batches keep memory flat for hour-long recordings (thousands of windows).
        chunks = [predict_chunk(segments[i : i + 32]) for i in range(0, len(segments), 32)]
        return np.concatenate(chunks, axis=1)

    return predict


def _ssl_backend(
    cfg: ExperimentConfig, fold_dirs: list[Path], n_classes: int, device: torch.device
) -> Backend:
    from aum2.features.ssl import SSLEncoder
    from aum2.models.ssl_head import LayerWeightedHead

    encoder = SSLEncoder(cfg.model.ssl.backbone, device)
    heads = []
    for fold_dir in fold_dirs:
        head = LayerWeightedHead(encoder.n_layers, 2 * encoder.hidden, n_classes, cfg.model.ssl)
        head.load_state_dict(_load_state(fold_dir / "model.safetensors", device), strict=True)
        heads.append(head.to(device).eval())

    @torch.no_grad()
    def predict(segments: list[np.ndarray]) -> np.ndarray:
        stats = torch.from_numpy(np.stack([encoder.encode(s) for s in segments])).to(device)
        # Match training, where the cached statistics were stored in float16.
        stats = stats.half().float()
        return np.stack([softmax(h(stats).float().cpu().numpy()) for h in heads])

    return predict


def _handcrafted_backend(cfg: ExperimentConfig, fold_dirs: list[Path]) -> Backend:
    """sklearn pipelines are pickles: only load ``pipeline.joblib`` files you trained yourself."""
    import joblib

    from aum2.features.handcrafted import extract_functionals

    pipelines = [joblib.load(fold_dir / "pipeline.joblib") for fold_dir in fold_dirs]

    def predict(segments: list[np.ndarray]) -> np.ndarray:
        feats = np.stack(
            [
                extract_functionals(s, cfg.data.sample_rate, cfg.model.handcrafted.n_mfcc)
                for s in segments
            ]
        )
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return np.stack([p.predict_proba(feats) for p in pipelines])

    return predict


class EmotionRecognizer:
    """Loads the fold models of a run folder and averages their probabilities.

    An ensemble run (``evaluate.py --ensemble``) loads its member runs, which live next to it,
    and mixes them with the weights chosen on the dev folds.
    """

    def __init__(
        self, run_dir: str | Path, device: str = "auto", folds: Sequence[int] | None = None
    ) -> None:
        run_dir = Path(run_dir)
        bundle = json.loads((run_dir / "bundle.json").read_text(encoding="utf-8"))
        self.family: str = bundle["family"]
        self.classes: tuple[str, ...] = tuple(bundle["classes"])
        self.device = resolve_device(device)
        if self.family == "ensemble":
            self._init_ensemble(run_dir, bundle, device, folds)
            return
        self.cfg = load_config(run_dir / "config.yaml")
        fold_dirs = [run_dir / f"fold{k}" for k in (folds or bundle["folds"])]
        if not fold_dirs:
            raise ValueError(f"{run_dir} has no trained folds")
        if self.family == "crnn":
            backend = _crnn_backend(self.cfg, fold_dirs, len(self.classes), self.device)
        elif self.family == "ssl_head":
            backend = _ssl_backend(self.cfg, fold_dirs, len(self.classes), self.device)
        elif self.family == "handcrafted":
            backend = _handcrafted_backend(self.cfg, fold_dirs)
        else:
            raise ValueError(f"family {self.family!r} is evaluation-only and has no inference path")
        temperatures = [load_temperature(fold_dir) for fold_dir in fold_dirs]

        def calibrated_mean(segments: list[np.ndarray]) -> np.ndarray:
            per_fold = backend(segments)
            return np.mean(
                [apply_temperature(p, t) for p, t in zip(per_fold, temperatures, strict=True)],
                axis=0,
            )

        self._predict_segments = calibrated_mean
        logger.info(
            "Loaded %s run %s (%d fold models) on %s",
            self.family,
            run_dir.name,
            len(fold_dirs),
            self.device,
        )

    def _init_ensemble(
        self, run_dir: Path, bundle: dict, device: str, folds: Sequence[int] | None
    ) -> None:
        members = [
            EmotionRecognizer(run_dir.parent / name, device, folds) for name in bundle["members"]
        ]
        for member in members:
            if member.cfg.data != members[0].cfg.data or member.classes != self.classes:
                raise ValueError(f"ensemble members of {run_dir} disagree on data or classes")
        self.cfg = members[0].cfg
        weights = bundle["weights"]

        def predict(segments: list[np.ndarray]) -> np.ndarray:
            return sum(
                w * m._predict_segments(segments) for w, m in zip(weights, members, strict=True)
            )

        self._predict_segments = predict

    @property
    def sample_rate(self) -> int:
        return self.cfg.data.sample_rate

    def predict_file(self, path: str | Path, **kwargs: float) -> Prediction:
        return self.predict_waveform(load_audio(path, self.sample_rate), self.sample_rate, **kwargs)

    def predict_waveform(
        self,
        samples: np.ndarray,
        sample_rate: int,
        window_seconds: float = 3.0,
        hop_seconds: float = 1.5,
        max_single_seconds: float = 6.0,
        silence_db: float = 35.0,
        min_level_dbfs: float = -60.0,
    ) -> Prediction:
        """Predict one recording (mono float array; resampled here if needed).

        Windows whose loudest 25 ms frame is more than ``silence_db`` below the recording's
        loudest frame are skipped; a recording that never exceeds ``min_level_dbfs`` is
        rejected as silent.
        """
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = resample(samples, sample_rate, self.sample_rate)
        sr = self.sample_rate
        duration = len(samples) / sr
        if duration <= max_single_seconds:
            windows = [(0, len(samples))]
        else:
            size, hop = int(window_seconds * sr), int(hop_seconds * sr)
            starts = list(range(0, len(samples) - size + 1, hop))
            if starts[-1] + size < len(samples):
                starts.append(len(samples) - size)
            windows = [(s, s + size) for s in starts]

        level = frame_rms_db(samples, int(0.025 * sr), int(0.010 * sr))
        floor = level.max() - silence_db
        kept, segments = [], []
        if level.max() < min_level_dbfs:
            windows = []
        for start, end in windows:
            chunk = samples[start:end]
            peak = frame_rms_db(chunk, int(0.025 * sr), int(0.010 * sr)).max()
            conditioned = condition(chunk, sr, self.cfg.data)
            if peak < floor or len(conditioned) < self.cfg.data.min_seconds * sr:
                continue
            kept.append((start, end))
            segments.append(conditioned)
        if not segments:
            raise ValueError(
                "no speech found: the recording is silent or shorter than "
                f"{self.cfg.data.min_seconds} s after trimming"
            )

        probs = self._predict_segments(segments)
        mean = probs.mean(axis=0)
        timeline = tuple(
            SegmentPrediction(
                start=round(start / sr, 3),
                end=round(end / sr, 3),
                label=self.classes[int(p.argmax())],
                probabilities={c: round(float(v), 4) for c, v in zip(self.classes, p, strict=True)},
            )
            for (start, end), p in zip(kept, probs, strict=True)
        )
        best = int(mean.argmax())
        return Prediction(
            label=self.classes[best],
            confidence=round(float(mean[best]), 4),
            probabilities={c: round(float(v), 4) for c, v in zip(self.classes, mean, strict=True)},
            duration=round(duration, 3),
            segments=timeline if len(timeline) > 1 else (),
        )


def record_microphone(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    """Record mono audio from the default input device (needs the ``mic`` extra)."""
    try:
        import sounddevice as sd
    except ImportError as error:  # pragma: no cover - optional dependency
        raise SystemExit("microphone input needs `pip install sounddevice`") from error
    frames = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return frames[:, 0]
