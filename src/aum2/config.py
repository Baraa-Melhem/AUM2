"""Typed experiment configuration.

Every experiment is described by one frozen :class:`ExperimentConfig`. YAML files map onto it
one-to-one; a file may name a ``base`` file whose values it overrides, and command-line
``--set section.key=value`` overrides are applied last. Unknown keys raise immediately, so a
typo in a config can never be silently ignored.
"""

from __future__ import annotations

import copy
import dataclasses
import types
import typing
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aum2 import EMOTIONS

CORPORA: tuple[str, ...] = ("cremad", "ravdess", "tess", "savee")
MODEL_FAMILIES = frozenset({"crnn", "ssl_head", "handcrafted", "legacy_cascade"})
NEURAL_FAMILIES = frozenset({"crnn", "ssl_head", "legacy_cascade"})


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class DataConfig:
    """Which corpora are used and how every waveform is conditioned before any model sees it.

    The same conditioning runs at inference time (:func:`aum2.data.audio.condition`), so
    training and serving can never drift apart.
    """

    raw_dir: str = "data/raw"
    cache_dir: str = "data/cache"
    corpora: tuple[str, ...] = CORPORA
    classes: tuple[str, ...] = EMOTIONS
    sample_rate: int = 16000
    trim_top_db: float = 35.0
    trim_margin_ms: float = 100.0
    target_dbfs: float | None = -25.0
    max_seconds: float = 8.0
    min_seconds: float = 0.3


@dataclass(frozen=True)
class SplitConfig:
    """Evaluation protocol.

    ``speaker`` holds out ``test_fraction`` of the speakers of every corpus (stratified by
    gender) and splits the remaining speakers into ``n_folds`` speaker-disjoint folds.
    ``random`` reproduces the original notebook's utterance-level split and exists only to
    measure how much that protocol inflates the numbers. The split seed is deliberately
    independent of the training seed so that every experiment sees the same speakers.
    """

    protocol: str = "speaker"
    test_fraction: float = 0.2
    n_folds: int = 5
    seed: int = 1337


@dataclass(frozen=True)
class FrontendConfig:
    """Log-mel front end: 25 ms Hann windows every 10 ms, 80 Slaney mel bands."""

    n_mels: int = 80
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    f_min: float = 20.0
    f_max: float = 8000.0


@dataclass(frozen=True)
class AugmentConfig:
    """On-the-fly augmentation, applied to training batches only (never to val/test)."""

    enabled: bool = True
    crop_seconds: float = 3.0
    gain_db: float = 6.0
    noise_prob: float = 0.5
    snr_db: tuple[float, ...] = (10.0, 35.0)
    tempo_prob: float = 0.5
    tempo_range: tuple[float, ...] = (0.85, 1.15)
    vtlp_prob: float = 0.5
    vtlp_range: tuple[float, ...] = (0.9, 1.1)
    freq_masks: int = 2
    freq_mask_width: int = 12
    time_masks: int = 2
    time_mask_ratio: float = 0.1


@dataclass(frozen=True)
class CRNNConfig:
    """Residual CNN → BiGRU → attentive statistics pooling."""

    channels: tuple[int, ...] = (32, 64, 128, 256)
    time_pool: tuple[int, ...] = (2, 2, 2, 1)
    proj_dim: int = 256
    gru_hidden: int = 128
    gru_layers: int = 2
    attention_dim: int = 128
    dropout: float = 0.3


@dataclass(frozen=True)
class SSLConfig:
    """Frozen self-supervised backbone plus a learned layer-weighted classification head."""

    backbone: str = "microsoft/wavlm-large"
    batch_seconds: float = 64.0
    hidden_dim: int = 256
    dropout: float = 0.3
    input_dropout: float = 0.1


@dataclass(frozen=True)
class HandcraftedConfig:
    """Utterance-level acoustic functionals plus a classical classifier."""

    classifier: str = "logreg"
    n_mfcc: int = 40
    logreg_c: float = 0.05
    lgbm_estimators: int = 800
    lgbm_learning_rate: float = 0.03
    lgbm_num_leaves: int = 31


@dataclass(frozen=True)
class LegacyConfig:
    """The 2023 notebook's features and five-CNN cascade, re-implemented faithfully."""

    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    image_size: int = 128
    epochs: tuple[int, ...] = (10, 7, 10, 10, 10)
    batch_sizes: tuple[int, ...] = (32, 35, 25, 25, 25)
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class ModelConfig:
    """Model family selector plus the typed hyper-parameter block of each family."""

    name: str = "crnn"
    crnn: CRNNConfig = field(default_factory=CRNNConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    handcrafted: HandcraftedConfig = field(default_factory=HandcraftedConfig)
    legacy: LegacyConfig = field(default_factory=LegacyConfig)


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings shared by the neural families."""

    batch_size: int = 64
    eval_batch_size: int = 64
    epochs: int = 60
    learning_rate: float = 2e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 2.0
    loss: str = "ce"
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    class_weighting: str = "balanced"
    grad_clip: float = 5.0
    ema_decay: float = 0.999
    patience: int = 0
    amp: bool = True
    device: str = "auto"
    deterministic: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration of one experiment (one model family under one protocol)."""

    name: str = "crnn"
    seed: int = 0
    output_dir: str = "runs"
    folds: tuple[int, ...] = ()
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    @property
    def fold_ids(self) -> tuple[int, ...]:
        return self.folds or tuple(range(self.split.n_folds))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise ConfigError(f"circular 'base' reference involving {path}")
    with path.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, Mapping):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    base_name = content.pop("base", None)
    if base_name is None:
        return dict(content)
    base = _read_yaml(path.parent / base_name, _seen | {path})
    return _deep_merge(base, content)


def _set_dotted(tree: dict[str, Any], dotted_key: str, value: Any) -> None:
    *parents, leaf = dotted_key.split(".")
    node = tree
    for part in parents:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override '{dotted_key}' descends into a non-mapping value")
    node[leaf] = value


def apply_overrides(tree: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` strings; values are parsed as YAML scalars or lists."""
    result = copy.deepcopy(tree)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override '{item}' must look like section.key=value")
        key, raw_value = item.split("=", 1)
        _set_dotted(result, key.strip(), yaml.safe_load(raw_value))
    return result


def _coerce(value: Any, annotation: Any, location: str) -> Any:
    origin = typing.get_origin(annotation)
    if dataclasses.is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{location} must be a mapping")
        return _build(annotation, value, location)
    if origin in (typing.Union, types.UnionType):
        options = typing.get_args(annotation)
        if value is None and type(None) in options:
            return None
        non_null = [option for option in options if option is not type(None)]
        return _coerce(value, non_null[0], location)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{location} must be a list")
        (item_type, _ellipsis) = typing.get_args(annotation)
        return tuple(_coerce(item, item_type, f"{location}[]") for item in value)
    if annotation is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if annotation is int and isinstance(value, bool):
        raise ConfigError(f"{location} must be int, got {value!r}")
    if annotation in (int, str, bool) and not isinstance(value, annotation):
        raise ConfigError(f"{location} must be {annotation.__name__}, got {value!r}")
    return value


def _build(cls: type, values: Mapping[str, Any], location: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        where = location or "top level"
        raise ConfigError(f"unknown key(s) {sorted(unknown)} at {where}")
    kwargs = {
        name: _coerce(value, hints[name], f"{location}.{name}".lstrip("."))
        for name, value in values.items()
    }
    return cls(**kwargs)


def _check_range(pair: tuple[float, ...], name: str, low: float = 0.0) -> None:
    if len(pair) != 2 or not low < pair[0] <= pair[1]:
        raise ConfigError(f"{name} must be [min, max] with {low} < min <= max")


def validate(cfg: ExperimentConfig) -> ExperimentConfig:
    """Check cross-field invariants that the type system cannot express."""
    data, split, train, aug = cfg.data, cfg.split, cfg.train, cfg.augment
    if cfg.model.name not in MODEL_FAMILIES:
        raise ConfigError(f"model.name must be one of {sorted(MODEL_FAMILIES)}")
    if unknown := set(data.corpora) - set(CORPORA):
        raise ConfigError(f"data.corpora contains unknown corpora {sorted(unknown)}")
    if unknown := set(data.classes) - set(EMOTIONS):
        raise ConfigError(f"data.classes must be a subset of {EMOTIONS}, got {sorted(unknown)}")
    if len(set(data.classes)) < 2 or len(set(data.classes)) != len(data.classes):
        raise ConfigError("data.classes needs at least two distinct emotions")
    if not 0 < data.min_seconds < data.max_seconds:
        raise ConfigError("need 0 < data.min_seconds < data.max_seconds")
    if split.protocol not in {"speaker", "random"}:
        raise ConfigError("split.protocol must be 'speaker' or 'random'")
    if not 0.0 < split.test_fraction < 0.5 or split.n_folds < 2:
        raise ConfigError("split.test_fraction must be in (0, 0.5) and split.n_folds >= 2")
    if bad_folds := [f for f in cfg.folds if not 0 <= f < split.n_folds]:
        raise ConfigError(f"folds {bad_folds} outside [0, {split.n_folds})")
    fe = cfg.frontend
    if fe.win_length > fe.n_fft or fe.hop_length <= 0 or fe.f_max > data.sample_rate / 2:
        raise ConfigError("frontend needs win_length <= n_fft, hop > 0, f_max <= sample_rate/2")
    _check_range(aug.snr_db, "augment.snr_db", low=-100.0)
    _check_range(aug.tempo_range, "augment.tempo_range")
    _check_range(aug.vtlp_range, "augment.vtlp_range")
    if aug.crop_seconds > data.max_seconds:
        raise ConfigError("augment.crop_seconds cannot exceed data.max_seconds")
    if train.loss not in {"ce", "focal"}:
        raise ConfigError("train.loss must be 'ce' or 'focal'")
    if train.class_weighting not in {"balanced", "none"}:
        raise ConfigError("train.class_weighting must be 'balanced' or 'none'")
    if not 0.0 <= train.ema_decay < 1.0 or not 0.0 <= train.label_smoothing < 1.0:
        raise ConfigError("train.ema_decay and train.label_smoothing must be in [0, 1)")
    crnn = cfg.model.crnn
    if len(crnn.channels) != len(crnn.time_pool) or any(p not in (1, 2) for p in crnn.time_pool):
        raise ConfigError("model.crnn.time_pool needs one entry (1 or 2) per channel block")
    if cfg.frontend.n_mels % (2 ** len(crnn.channels)):
        raise ConfigError("frontend.n_mels must be divisible by 2 ** len(model.crnn.channels)")
    if cfg.model.handcrafted.classifier not in {"logreg", "lightgbm"}:
        raise ConfigError("model.handcrafted.classifier must be 'logreg' or 'lightgbm'")
    legacy = cfg.model.legacy
    if len(legacy.epochs) != 5 or len(legacy.batch_sizes) != 5:
        raise ConfigError("model.legacy needs five epochs and five batch_sizes (one per CNN)")
    if cfg.model.name == "legacy_cascade" and set(data.classes) != set(EMOTIONS):
        raise ConfigError("the legacy cascade's routing is hard-wired to all six emotions")
    return cfg


def config_from_dict(values: Mapping[str, Any]) -> ExperimentConfig:
    """Build and validate an :class:`ExperimentConfig` from a nested mapping."""
    return validate(_build(ExperimentConfig, values))


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> ExperimentConfig:
    """Load a YAML experiment file, resolving ``base`` inheritance and CLI overrides."""
    tree = _read_yaml(Path(path))
    return config_from_dict(apply_overrides(tree, overrides))


def to_dict(cfg: Any) -> dict[str, Any]:
    """Serialise a config (or any config section) to plain YAML/JSON-friendly types."""

    def convert(value: Any) -> Any:
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    return convert(dataclasses.asdict(cfg))


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Write the fully-resolved config so every run is reproducible from its output folder."""
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_dict(cfg), handle, sort_keys=False)
