"""One experiment = one model family under one protocol: K-fold dev CV plus held-out test.

For each dev fold *k* a fresh learner is trained on the other folds, scored on fold *k* (the
out-of-fold, "OOF", predictions used for model selection) and on the held-out test speakers.
The reported test number is the K-model ensemble (mean probability); the mean ± std of the
single fold models is reported next to it.

Each fold writes ``fold{k}/predictions.npz`` last, so an interrupted run resumes at the first
unfinished fold.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch

from aum2 import __version__
from aum2.config import ExperimentConfig, load_config, save_config, to_dict
from aum2.data.dataset import load_corpus
from aum2.data.splits import TEST_FOLD, assign_folds, check_speaker_disjoint, split_summary
from aum2.evaluation.report import evaluate_run
from aum2.logging_utils import get_logger
from aum2.reproducibility import resolve_device, seed_everything
from aum2.training.learners import learner_factory

logger = get_logger(__name__)

MANIFEST_EXPORT = ("utt_id", "path", "corpus", "speaker", "gender", "emotion", "label", "duration")


def _flatten(tree: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _check_resumable(cfg: ExperimentConfig, saved: Path) -> None:
    """Finished folds are only reused if they were trained with the same settings."""
    previous, current = _flatten(to_dict(load_config(saved))), _flatten(to_dict(cfg))
    changed = sorted(k for k in previous.keys() | current.keys()
                     if k != "folds" and previous.get(k) != current.get(k))  # fmt: skip
    if changed:
        raise SystemExit(
            f"{saved.parent} was trained with different settings ({', '.join(changed[:6])}); "
            "choose another --name or delete the folder"
        )


def run_experiment(cfg: ExperimentConfig, n_jobs: int = -1) -> Path:
    """Train every requested fold (skipping finished ones), then evaluate the run."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "config.yaml").exists():
        _check_resumable(cfg, run_dir / "config.yaml")
    save_config(cfg, run_dir / "config.yaml")
    device = resolve_device(cfg.train.device)
    if device.type == "cuda":
        # Never let the WDDM driver silently spill VRAM into system RAM (5-10x slowdowns).
        torch.cuda.set_per_process_memory_fraction(0.9)
    logger.info(
        "Run %s | family %s | protocol %s | device %s",
        cfg.name,
        cfg.model.name,
        cfg.split.protocol,
        device,
    )

    corpus = load_corpus(cfg.data, n_jobs)
    folds = assign_folds(corpus.manifest, cfg.split)
    if cfg.split.protocol == "speaker":
        check_speaker_disjoint(corpus.manifest, folds)
    exported = corpus.manifest[list(MANIFEST_EXPORT)].assign(fold=folds)
    exported.to_csv(run_dir / "manifest.csv", index=False)
    logger.info(
        "Split:\n%s", split_summary(corpus.manifest, folds).unstack(fill_value=0).to_string()
    )

    factory = learner_factory(cfg, corpus, device, n_jobs)
    test_idx = np.flatnonzero(folds == TEST_FOLD)
    for k in cfg.fold_ids:
        fold_dir = run_dir / f"fold{k}"
        if (fold_dir / "predictions.npz").exists():
            logger.info("[fold %d] already finished, skipping", k)
            continue
        seed_everything(cfg.seed * 100 + k, cfg.train.deterministic)
        train_idx = np.flatnonzero((folds != TEST_FOLD) & (folds != k))
        val_idx = np.flatnonzero(folds == k)
        learner = factory()
        history = learner.fit(train_idx, val_idx, log_prefix=f"[fold {k}] ")
        val_probs, test_probs = learner.predict_proba(val_idx), learner.predict_proba(test_idx)
        fold_dir.mkdir(parents=True, exist_ok=True)
        learner.save(fold_dir)
        (fold_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        np.savez_compressed(
            fold_dir / "predictions.npz",
            val_idx=val_idx,
            val_probs=val_probs,
            test_idx=test_idx,
            test_probs=test_probs,
        )
        del learner
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    bundle = {
        "aum2_version": __version__,
        "family": cfg.model.name,
        "classes": list(corpus.classes),
        "folds": sorted(
            int(p.name[4:]) for p in run_dir.glob("fold*") if (p / "predictions.npz").exists()
        ),
        "data": to_dict(cfg.data),
        "corpus_key": corpus.key,
    }
    (run_dir / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    evaluate_run(run_dir)
    return run_dir
