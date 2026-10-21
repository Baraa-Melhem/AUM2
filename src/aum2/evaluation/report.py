"""Turn a run's saved predictions into metrics, figures, ensembles and the benchmark table."""

from __future__ import annotations

import itertools
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aum2.evaluation import plots
from aum2.evaluation.calibration import (
    apply_temperature,
    fit_temperature,
    load_temperature,
    save_temperature,
)
from aum2.evaluation.metrics import (
    full_report,
    paired_speaker_bootstrap,
    speaker_bootstrap,
    summary_metrics,
)
from aum2.logging_utils import get_logger

logger = get_logger(__name__)

# One-hot cascade outputs cannot be rescaled; ensembles mix already-calibrated members.
UNCALIBRATED_FAMILIES = frozenset({"legacy_cascade", "ensemble"})
DISPLAY_NAMES = {
    "ensemble": "WavLM + HuBERT ensemble",
    "wavlm": "WavLM-Large head",
    "hubert": "HuBERT-Large head",
    "crnn": "Log-mel CRNN",
    "handcrafted": "Functionals + LightGBM",
    "legacy": "2023 method, re-run",
    "wavlm_random": "WavLM-Large head, random split",
    "legacy_random": "2023 method, random split",
}
CORPUS_NAMES = {"cremad": "CREMA-D", "ravdess": "RAVDESS", "tess": "TESS", "savee": "SAVEE"}


@dataclass
class RunPredictions:
    """Everything evaluation needs from a run folder."""

    name: str
    family: str
    classes: list[str]
    manifest: pd.DataFrame
    oof: np.ndarray
    test_idx: np.ndarray
    test_by_fold: dict[int, np.ndarray]
    histories: dict[int, list[dict]]

    @property
    def labels(self) -> np.ndarray:
        return self.manifest["label"].to_numpy()

    @property
    def test_ensemble(self) -> np.ndarray:
        return np.mean(list(self.test_by_fold.values()), axis=0)


def load_run(run_dir: str | Path) -> RunPredictions:
    run_dir = Path(run_dir)
    bundle = json.loads((run_dir / "bundle.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(run_dir / "manifest.csv")
    classes = bundle["classes"]
    oof = np.full((len(manifest), len(classes)), np.nan)
    test_by_fold: dict[int, np.ndarray] = {}
    histories: dict[int, list[dict]] = {}
    test_idx = np.flatnonzero(manifest["fold"].to_numpy() == -1)
    for k in bundle["folds"]:
        temperature = load_temperature(run_dir / f"fold{k}")
        with np.load(run_dir / f"fold{k}" / "predictions.npz") as z:
            oof[z["val_idx"]] = apply_temperature(z["val_probs"], temperature)
            if not np.array_equal(z["test_idx"], test_idx):
                raise ValueError(f"fold {k} of {run_dir} was scored on a different test set")
            test_by_fold[k] = apply_temperature(z["test_probs"], temperature)
        history_file = run_dir / f"fold{k}" / "history.json"
        if history_file.exists():
            histories[k] = json.loads(history_file.read_text(encoding="utf-8"))
    return RunPredictions(
        run_dir.name, bundle["family"], classes, manifest, oof, test_idx, test_by_fold, histories
    )


def _per_corpus(
    manifest: pd.DataFrame, rows: np.ndarray, probs: np.ndarray, labels: np.ndarray
) -> dict:
    out = {}
    corpora = manifest["corpus"].to_numpy()[rows]
    for corpus in CORPUS_NAMES:
        mask = corpora == corpus
        if mask.any():
            out[corpus] = {"n": int(mask.sum()), **summary_metrics(labels[rows][mask], probs[mask])}
    return out


def calibrate_run(run_dir: str | Path) -> None:
    """Fit each fold's temperature on its own validation predictions (once; kept on disk)."""
    run_dir = Path(run_dir)
    bundle = json.loads((run_dir / "bundle.json").read_text(encoding="utf-8"))
    if bundle["family"] in UNCALIBRATED_FAMILIES:
        return
    labels = pd.read_csv(run_dir / "manifest.csv")["label"].to_numpy()
    for k in bundle["folds"]:
        fold_dir = run_dir / f"fold{k}"
        if not (fold_dir / "calibration.json").exists():
            with np.load(fold_dir / "predictions.npz") as z:
                temperature = fit_temperature(z["val_probs"], labels[z["val_idx"]])
            save_temperature(fold_dir, temperature)
            logger.info("%s fold %d: temperature %.3f", run_dir.name, k, temperature)


def evaluate_run(run_dir: str | Path, n_boot: int = 1000) -> dict[str, Any]:
    """Compute dev (OOF) and test metrics, write ``metrics.json``, predictions and figures.

    Probabilities are temperature-scaled per fold (:mod:`aum2.evaluation.calibration`); dev ECE
    is therefore in-sample, while test ECE is an honest calibration estimate.
    """
    run_dir = Path(run_dir)
    calibrate_run(run_dir)
    run = load_run(run_dir)
    labels, classes = run.labels, run.classes
    dev_rows = np.flatnonzero(~np.isnan(run.oof).any(axis=1))
    dev_probs = run.oof[dev_rows]
    test_probs = run.test_ensemble
    test_labels = labels[run.test_idx]
    speakers = run.manifest["speaker"].to_numpy()

    single = [summary_metrics(test_labels, p) for p in run.test_by_fold.values()]
    report: dict[str, Any] = {
        "run": run.name,
        "family": run.family,
        "classes": classes,
        "folds": sorted(run.test_by_fold),
        "dev": {
            **full_report(labels[dev_rows], dev_probs, classes),
            "per_corpus": _per_corpus(run.manifest, dev_rows, dev_probs, labels),
        },
        "test": {
            **full_report(test_labels, test_probs, classes),
            "per_corpus": _per_corpus(run.manifest, run.test_idx, test_probs, labels),
            "ci95": speaker_bootstrap(
                test_labels, test_probs.argmax(1), speakers[run.test_idx], n_boot
            ),
            "speakers": len(np.unique(speakers[run.test_idx])),
        },
        "test_single_model": {
            key: {
                "mean": float(np.mean([s[key] for s in single])),
                "std": float(np.std([s[key] for s in single], ddof=1)) if len(single) > 1 else 0.0,
            }
            for key in ("war", "uar", "macro_f1")
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_predictions(run, dev_rows, dev_probs, test_probs, run_dir / "predictions.csv.gz")
    _write_figures(run, report, dev_rows, dev_probs, test_probs, run_dir / "figures")
    t, d = report["test"], report["dev"]
    logger.info(
        "%s | dev OOF UAR %.4f | test UAR %.4f [%.4f, %.4f] WAR %.4f F1 %.4f"
        " | single-model UAR %.4f +/- %.4f",
        run.name,
        d["uar"],
        t["uar"],
        *t["ci95"]["uar"],
        t["war"],
        t["macro_f1"],
        report["test_single_model"]["uar"]["mean"],
        report["test_single_model"]["uar"]["std"],
    )
    return report


def _write_predictions(
    run: RunPredictions,
    dev_rows: np.ndarray,
    dev_probs: np.ndarray,
    test_probs: np.ndarray,
    path: Path,
) -> None:
    frames = []
    for partition, rows, probs in (
        ("dev", dev_rows, dev_probs),
        ("test", run.test_idx, test_probs),
    ):
        table = run.manifest.iloc[rows][["utt_id", "corpus", "speaker", "emotion", "label"]].copy()
        table.insert(0, "partition", partition)
        table["pred"] = np.asarray(run.classes)[probs.argmax(1)]
        for k, name in enumerate(run.classes):
            table[f"p_{name}"] = probs[:, k].round(5)
        frames.append(table)
    pd.concat(frames).to_csv(path, index=False)


def _write_figures(
    run: RunPredictions,
    report: dict,
    dev_rows: np.ndarray,
    dev_probs: np.ndarray,
    test_probs: np.ndarray,
    folder: Path,
) -> None:
    labels, classes = run.labels, run.classes
    test_labels = labels[run.test_idx]
    plots.plot_confusion(
        np.array(report["test"]["confusion_normalized"]),
        classes,
        f"{run.name}: test speakers (UAR {report['test']['uar']:.3f})",
        folder / "confusion_test.png",
    )
    plots.plot_confusion(
        np.array(report["dev"]["confusion_normalized"]),
        classes,
        f"{run.name}: dev out-of-fold (UAR {report['dev']['uar']:.3f})",
        folder / "confusion_dev.png",
    )
    if run.family not in {"legacy_cascade"}:
        plots.plot_roc_pr(test_labels, test_probs, classes, folder / "roc_pr_test.png")
        plots.plot_reliability(test_labels, test_probs, folder / "reliability_test.png")
    per_corpus = {CORPUS_NAMES[c]: m for c, m in report["test"]["per_corpus"].items()}
    plots.plot_per_corpus(
        per_corpus, folder / "per_corpus_test.png", f"{run.name}: test speakers by corpus"
    )
    plots.plot_training_curves(run.histories, folder / "training_curves.png")
    layer_files = sorted(folder.parent.glob("fold*/layer_weights.json"))
    if layer_files:
        stacked = np.array([json.loads(f.read_text(encoding="utf-8")) for f in layer_files])
        config = yaml.safe_load((folder.parent / "config.yaml").read_text(encoding="utf-8"))
        plots.plot_layer_weights(
            stacked, folder / "layer_weights.png", config["model"]["ssl"]["backbone"]
        )


def ensemble_runs(
    run_dirs: list[str | Path],
    name: str,
    output_dir: str | Path = "runs",
    tune_weights: bool = False,
) -> Path:
    """Average the probabilities of several runs that share one split.

    With ``tune_weights`` the mixing weights are chosen on the dev out-of-fold predictions
    (grid over the simplex, step 0.05 for two members, 0.1 otherwise) and then applied,
    frozen, to the test predictions.
    """
    out = Path(output_dir) / name
    for d in run_dirs:  # inference finds members by name next to the ensemble folder
        if Path(d).resolve().parent != out.resolve().parent:
            raise ValueError(f"ensemble members must live in {out.parent}, got {d}")
    runs = [load_run(d) for d in run_dirs]
    ref = runs[0]
    for run in runs[1:]:
        if not run.manifest["utt_id"].equals(ref.manifest["utt_id"]) or not run.manifest[
            "fold"
        ].equals(ref.manifest["fold"]):
            raise ValueError(f"{run.name} does not share the split of {ref.name}")
    folds = sorted(set.intersection(*(set(r.test_by_fold) for r in runs)))
    dev_mask = np.all([~np.isnan(r.oof).any(1) for r in runs], axis=0)
    weights = np.full(len(runs), 1.0 / len(runs))
    if tune_weights:
        step = 0.05 if len(runs) == 2 else 0.1
        grid = [
            w
            for w in itertools.product(np.arange(0, 1 + 1e-9, step), repeat=len(runs))
            if abs(sum(w) - 1) < 1e-6
        ]
        labels = ref.labels[dev_mask]
        scores = [
            summary_metrics(
                labels, sum(wi * r.oof[dev_mask] for wi, r in zip(w, runs, strict=True))
            )["uar"]
            for w in grid
        ]
        weights = np.array(grid[int(np.argmax(scores))])
        logger.info(
            "Ensemble weights tuned on dev OOF (UAR %.4f): %s",
            max(scores),
            dict(zip([r.name for r in runs], weights.round(2), strict=True)),
        )

    out.mkdir(parents=True, exist_ok=True)
    ref.manifest.to_csv(out / "manifest.csv", index=False)
    dev_rows = np.flatnonzero(dev_mask)
    for k in folds:
        val_idx = dev_rows[ref.manifest["fold"].to_numpy()[dev_rows] == k]
        (out / f"fold{k}").mkdir(exist_ok=True)
        np.savez_compressed(
            out / f"fold{k}" / "predictions.npz",
            val_idx=val_idx,
            val_probs=sum(w * r.oof[val_idx] for w, r in zip(weights, runs, strict=True)),
            test_idx=ref.test_idx,
            test_probs=sum(w * r.test_by_fold[k] for w, r in zip(weights, runs, strict=True)),
        )
    bundle = {
        "family": "ensemble",
        "classes": ref.classes,
        "folds": folds,
        "members": [Path(d).name for d in run_dirs],
        "weights": weights.tolist(),
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    evaluate_run(out)
    return out


def benchmark_table(run_dirs: list[str | Path]) -> str:
    """Markdown table comparing runs on the held-out test speakers (and dev OOF UAR)."""
    header = (
        "| Model | Dev OOF UAR | Test UAR [95% CI] | Test WAR | Test macro-F1 |"
        " Single-model test UAR |"
        " CREMA-D | RAVDESS | TESS | SAVEE |\n|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for run_dir in run_dirs:
        m = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
        t, s = m["test"], m["test_single_model"]["uar"]
        corpus = [
            f"{t['per_corpus'][c]['uar']:.3f}" if c in t["per_corpus"] else "-"
            for c in CORPUS_NAMES
        ]
        name = DISPLAY_NAMES.get(m["run"], m["run"])
        lines.append(
            f"| {name} (`{m['run']}`) | {m['dev']['uar']:.3f} | **{t['uar']:.3f}** "
            f"[{t['ci95']['uar'][0]:.3f}, {t['ci95']['uar'][1]:.3f}] | {t['war']:.3f} | "
            f"{t['macro_f1']:.3f} | {s['mean']:.3f} ± {s['std']:.3f} | " + " | ".join(corpus) + " |"
        )
    return "\n".join(lines)


def compare_runs(run_a: str | Path, run_b: str | Path, n_boot: int = 2000) -> dict[str, Any]:
    """Paired speaker-bootstrap test of test-set UAR and WAR: is ``run_a`` really better?"""
    a, b = load_run(run_a), load_run(run_b)
    if not np.array_equal(a.test_idx, b.test_idx):
        raise ValueError("runs were scored on different test sets")
    labels = a.labels[a.test_idx]
    speakers = a.manifest["speaker"].to_numpy()[a.test_idx]
    pred_a, pred_b = a.test_ensemble.argmax(1), b.test_ensemble.argmax(1)
    result = {
        metric: paired_speaker_bootstrap(labels, pred_a, pred_b, speakers, metric, n_boot)
        for metric in ("uar", "war")
    }
    logger.info(
        "%s vs %s: dUAR %+.4f [%+.4f, %+.4f] p=%.4f",
        a.name, b.name, result["uar"]["difference"], result["uar"]["ci_low"],
        result["uar"]["ci_high"], result["uar"]["p_value"],
    )  # fmt: skip
    return {"a": a.name, "b": b.name, **result}


def export_reports(run_dirs: list[str | Path], reports_dir: str | Path, chart: bool = True) -> None:
    """Copy each run's metrics, resolved config and figures into the versioned reports folder
    and draw the benchmark chart (``runs/`` itself is not versioned). The first run is drawn as
    the selected model."""
    reports_dir = Path(reports_dir)
    rows = []
    for run_dir in map(Path, run_dirs):
        target = reports_dir / "runs" / run_dir.name
        (target / "figures").mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "config.yaml", "bundle.json"):
            if (run_dir / name).exists():
                shutil.copy2(run_dir / name, target / name)
        for figure in (run_dir / "figures").glob("*.png"):
            shutil.copy2(figure, target / "figures" / figure.name)
        test = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["test"]
        label = DISPLAY_NAMES.get(run_dir.name, run_dir.name)
        rows.append({"label": label, "uar": test["uar"], "ci": test["ci95"]["uar"]})
    if chart:
        rows[0]["selected"] = True  # by convention the first run is the dev-selected model
        plots.plot_benchmark(rows, reports_dir / "figures" / "benchmark.png")
