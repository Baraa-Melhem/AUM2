"""Evaluation splits.

Speaker-independent protocol (the default and the only one used for reported results):

* **test**: ``test_fraction`` of the speakers of *each* corpus, drawn separately per gender, so
  every corpus and gender is represented and no test speaker is ever seen in training. TESS has
  two speakers, so one of them is held out; SAVEE has four, so one is held out.
* **dev**: the remaining speakers, split into ``n_folds`` speaker-disjoint folds with
  :class:`~sklearn.model_selection.StratifiedGroupKFold` (groups = speaker, strata = corpus ×
  emotion). All model selection happens on these folds; the test speakers are scored once.

The ``random`` protocol reproduces the 2023 notebook's utterance-level split, purely to
measure how much a speaker leak inflates accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from aum2.config import SplitConfig

TEST_FOLD = -1


def _test_speakers(manifest: pd.DataFrame, fraction: float, rng: np.random.Generator) -> set[str]:
    chosen: set[str] = set()
    speakers = manifest.drop_duplicates("speaker")[["corpus", "gender", "speaker"]]
    for corpus, group in speakers.groupby("corpus", sort=True):
        n_corpus = max(1, round(fraction * len(group)))
        by_gender = {g: sorted(s["speaker"]) for g, s in group.groupby("gender", sort=True)}
        # Largest-remainder allocation of the corpus quota across genders.
        exact = {g: n_corpus * len(s) / len(group) for g, s in by_gender.items()}
        quota = {g: int(np.floor(v)) for g, v in exact.items()}
        for g in sorted(exact, key=lambda g: exact[g] - quota[g], reverse=True):
            if sum(quota.values()) >= n_corpus:
                break
            quota[g] += 1
        for gender, names in by_gender.items():
            chosen.update(rng.permutation(names)[: quota[gender]].tolist())
        if not chosen & set(group["speaker"]):
            raise AssertionError(f"no test speaker drawn for {corpus}")
    return chosen


def assign_folds(manifest: pd.DataFrame, cfg: SplitConfig) -> np.ndarray:
    """Return one fold id per row: ``-1`` for test, ``0..n_folds-1`` for the dev folds."""
    folds = np.full(len(manifest), TEST_FOLD, dtype=np.int64)
    strata = (manifest["corpus"] + "/" + manifest["emotion"]).to_numpy()
    if cfg.protocol == "random":
        dev, _ = train_test_split(
            np.arange(len(manifest)),
            test_size=cfg.test_fraction,
            stratify=strata,
            random_state=cfg.seed,
        )
        splitter = StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)
        for k, (_, val) in enumerate(splitter.split(dev, strata[dev])):
            folds[dev[val]] = k
        return folds

    rng = np.random.default_rng(cfg.seed)
    test = manifest["speaker"].isin(_test_speakers(manifest, cfg.test_fraction, rng)).to_numpy()
    dev = np.flatnonzero(~test)
    splitter = StratifiedGroupKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)
    groups = manifest["speaker"].to_numpy()[dev]
    for k, (_, val) in enumerate(splitter.split(dev, strata[dev], groups)):
        folds[dev[val]] = k
    return folds


def check_speaker_disjoint(manifest: pd.DataFrame, folds: np.ndarray) -> None:
    """Raise if any speaker appears in more than one partition (test or a dev fold)."""
    partitions = pd.DataFrame({"speaker": manifest["speaker"].to_numpy(), "fold": folds})
    spread = partitions.groupby("speaker")["fold"].nunique()
    if (leaky := spread[spread > 1]).any():
        raise AssertionError(f"speakers in several partitions: {leaky.index.tolist()[:5]}")


def split_summary(manifest: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """Clips and speakers per corpus and partition (``test`` / ``fold k``)."""
    part = np.where(folds == TEST_FOLD, "test", np.char.add("fold ", folds.astype(str)))
    table = manifest.assign(partition=part)
    return table.groupby(["corpus", "partition"]).agg(
        clips=("utt_id", "size"), speakers=("speaker", "nunique")
    )
