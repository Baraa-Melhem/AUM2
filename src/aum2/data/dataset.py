"""The conditioned corpus: a manifest table plus every waveform, cached as one int16 array.

All ~11 k clips fit in about 1 GB of RAM as 16 kHz int16, so the corpus is decoded, resampled
and conditioned once (in parallel) and memory-mapped afterwards. The cache is keyed by a hash of
the file list and every conditioning parameter; changing either rebuilds it automatically.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from aum2.config import DataConfig, to_dict
from aum2.data.audio import load_and_condition
from aum2.data.corpora import iter_corpus
from aum2.logging_utils import get_logger

logger = get_logger(__name__)

MANIFEST_COLUMNS = (
    "utt_id",
    "path",
    "corpus",
    "speaker",
    "gender",
    "emotion",
    "text_id",
    "intensity",
)
_INT16_SCALE = 32767.0
_CACHE_FORMAT = 2  # bump whenever conditioning or de-duplication logic changes


def build_manifest(
    raw_dir: str | Path, corpora: tuple[str, ...], classes: tuple[str, ...]
) -> pd.DataFrame:
    """Parse every corpus folder and keep the utterances whose emotion is in ``classes``."""
    rows = [u.__dict__ for corpus in corpora for u in iter_corpus(Path(raw_dir), corpus)]
    table = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    if table["utt_id"].duplicated().any():
        dupes = table.loc[table["utt_id"].duplicated(), "utt_id"].tolist()[:5]
        raise ValueError(f"duplicate utterance ids, e.g. {dupes}")
    dropped = table.loc[~table["emotion"].isin(classes)]
    if len(dropped):
        logger.info(
            "Dropping %d clips outside the %d target classes: %s",
            len(dropped),
            len(classes),
            dropped.groupby("corpus")["emotion"].value_counts().to_dict(),
        )
    table = table.loc[table["emotion"].isin(classes)].reset_index(drop=True)
    table["label"] = table["emotion"].map({name: i for i, name in enumerate(classes)})
    return table


def _process(raw_dir: Path, rel_path: str, cfg: DataConfig) -> tuple[np.ndarray | None, str]:
    try:
        samples = load_and_condition(raw_dir / rel_path, cfg)
    except Exception as error:
        return None, f"unreadable ({type(error).__name__}: {error})"
    if len(samples) < cfg.min_seconds * cfg.sample_rate:
        return None, f"too short after trimming ({len(samples) / cfg.sample_rate:.2f} s)"
    if not np.isfinite(samples).all():
        return None, "non-finite samples"
    return np.round(np.clip(samples, -1.0, 1.0) * _INT16_SCALE).astype(np.int16), ""


@dataclass
class CorpusData:
    """Manifest rows aligned 1:1 with the conditioned waveforms in ``samples``."""

    manifest: pd.DataFrame
    samples: np.ndarray
    offsets: np.ndarray
    sample_rate: int
    classes: tuple[str, ...]
    key: str = ""

    def __len__(self) -> int:
        return len(self.manifest)

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def labels(self) -> np.ndarray:
        return self.manifest["label"].to_numpy(dtype=np.int64)

    def waveform(self, index: int) -> np.ndarray:
        """Float32 waveform in [-1, 1] of row ``index``."""
        start, end = self.offsets[index], self.offsets[index + 1]
        return self.samples[start:end].astype(np.float32) / _INT16_SCALE

    def waveform_int16(self, index: int) -> np.ndarray:
        return self.samples[self.offsets[index] : self.offsets[index + 1]]


def _deduplicate(table: pd.DataFrame) -> np.ndarray:
    """Boolean keep-mask over rows that share identical conditioned audio (``audio_sha1``).

    Identical audio under two names would sit on both sides of a split. If the copies carry
    the same emotion, the first is kept; if they disagree (CREMA-D ships three such pairs, e.g.
    ``1006_TIE_HAP_XX`` = ``1006_TIE_NEU_XX``), the true label is unknowable and all copies go.
    """
    keep = np.ones(len(table), dtype=bool)
    for _, group in table.groupby("audio_sha1", sort=False):
        if len(group) == 1:
            continue
        conflicting = group["emotion"].nunique() > 1
        drop = group.index if conflicting else group.index[1:]
        keep[drop] = False
        logger.warning(
            "Identical audio: %s -> %s",
            ", ".join(group["path"]),
            "labels disagree, dropping all" if conflicting else "keeping the first copy",
        )
    return keep


def _cache_key(manifest: pd.DataFrame, cfg: DataConfig) -> str:
    params = {k: v for k, v in to_dict(cfg).items() if k not in {"raw_dir", "cache_dir"}}
    params["cache_format"] = _CACHE_FORMAT
    payload = json.dumps(params, sort_keys=True) + "\n".join(manifest["path"])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_corpus(cfg: DataConfig, n_jobs: int = -1) -> CorpusData:
    """Build (or reuse) the conditioned waveform cache for the corpora/classes in ``cfg``."""
    manifest = build_manifest(cfg.raw_dir, cfg.corpora, cfg.classes)
    key = _cache_key(manifest, cfg)
    cache = Path(cfg.cache_dir) / f"waveforms_{key}"
    if not (cache / "manifest.csv").exists():
        _build_cache(manifest, cfg, cache, n_jobs)
    table = pd.read_csv(cache / "manifest.csv", dtype={"text_id": str, "intensity": str})
    samples = np.load(cache / "samples.npy", mmap_mode="r")
    offsets = np.load(cache / "offsets.npy")
    logger.info(
        "Corpus: %d clips, %.1f h of audio (cache %s)",
        len(table),
        len(samples) / cfg.sample_rate / 3600,
        cache,
    )
    return CorpusData(table, samples, offsets, cfg.sample_rate, cfg.classes, key)


def _build_cache(manifest: pd.DataFrame, cfg: DataConfig, cache: Path, n_jobs: int) -> None:
    logger.info("Conditioning %d clips -> %s (one-off) ...", len(manifest), cache)
    raw_dir = Path(cfg.raw_dir)
    results = Parallel(n_jobs=n_jobs, batch_size=32)(
        delayed(_process)(raw_dir, path, cfg) for path in manifest["path"]
    )
    keep = [samples is not None for samples, _ in results]
    for (_, reason), (_, row) in zip(results, manifest.iterrows(), strict=True):
        if reason:
            logger.warning("Excluding %s: %s", row["path"], reason)
    waves = [samples for samples, _ in results if samples is not None]
    table = manifest.loc[keep].reset_index(drop=True)
    table["num_samples"] = [len(w) for w in waves]
    table["duration"] = table["num_samples"] / cfg.sample_rate
    table["audio_sha1"] = [hashlib.sha1(w.tobytes()).hexdigest() for w in waves]
    keep_mask = _deduplicate(table)
    waves = [w for w, k in zip(waves, keep_mask, strict=True) if k]
    table = table.loc[keep_mask].reset_index(drop=True)
    cache.mkdir(parents=True, exist_ok=True)
    offsets = np.concatenate([[0], np.cumsum([len(w) for w in waves])]).astype(np.int64)
    np.save(cache / "samples.npy", np.concatenate(waves))
    np.save(cache / "offsets.npy", offsets)
    with (cache / "params.json").open("w", encoding="utf-8") as handle:
        json.dump(to_dict(cfg), handle, indent=2)
    table.to_csv(cache / "manifest.csv", index=False)  # written last: marks the cache complete
