"""Model selection on the dev folds only.

:func:`cross_validate` trains a config on each dev fold and scores the out-of-fold predictions;
it never loads, predicts or reports the held-out test speakers. :func:`grid_search` runs it
over a grid of ``--set`` overrides and returns a table sorted by OOF UAR, so hyper-parameters
are chosen without looking at the test set.
"""

from __future__ import annotations

import gc
import itertools
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from aum2.config import ExperimentConfig, apply_overrides, config_from_dict, to_dict
from aum2.data.dataset import CorpusData, load_corpus
from aum2.data.splits import TEST_FOLD, assign_folds
from aum2.evaluation.metrics import summary_metrics
from aum2.logging_utils import get_logger
from aum2.reproducibility import resolve_device, seed_everything
from aum2.training.learners import learner_factory

logger = get_logger(__name__)


def cross_validate(cfg: ExperimentConfig, corpus: CorpusData | None = None,
                   n_jobs: int = -1) -> dict[str, float]:  # fmt: skip
    """Dev-only K-fold CV: mean OOF WAR / UAR / macro-F1 over ``cfg.fold_ids``."""
    corpus = corpus or load_corpus(cfg.data, n_jobs)
    device = resolve_device(cfg.train.device)
    folds = assign_folds(corpus.manifest, cfg.split)
    factory = learner_factory(cfg, corpus, device, n_jobs)
    labels = corpus.labels
    oof_rows, oof_probs = [], []
    for k in cfg.fold_ids:
        seed_everything(cfg.seed * 100 + k, cfg.train.deterministic)
        train_idx = np.flatnonzero((folds != TEST_FOLD) & (folds != k))
        val_idx = np.flatnonzero(folds == k)
        learner = factory()
        learner.fit(train_idx, val_idx, log_prefix=f"[cv {k}] ")
        oof_rows.append(val_idx)
        oof_probs.append(learner.predict_proba(val_idx))
        del learner
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rows = np.concatenate(oof_rows)
    return summary_metrics(labels[rows], np.concatenate(oof_probs))


def grid_search(base: ExperimentConfig, grid: Mapping[str, Sequence[object]],
                n_jobs: int = -1) -> pd.DataFrame:  # fmt: skip
    """Cross-validate every combination of ``grid`` (dotted keys → candidate values)."""
    corpus = load_corpus(base.data, n_jobs)
    keys = list(grid)
    results = []
    for values in itertools.product(*(grid[k] for k in keys)):
        overrides = [f"{k}={v}" for k, v in zip(keys, values, strict=True)]
        cfg = config_from_dict(apply_overrides(to_dict(base), overrides))
        logger.info("CV %s", " ".join(overrides) or "(base config)")
        metrics = cross_validate(cfg, corpus, n_jobs)
        results.append({**dict(zip(keys, values, strict=True)), **metrics})
        logger.info("-> OOF UAR %.4f WAR %.4f F1 %.4f", metrics["uar"], metrics["war"],
                    metrics["macro_f1"])  # fmt: skip
    return pd.DataFrame(results).sort_values("uar", ascending=False).reset_index(drop=True)
