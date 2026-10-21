"""Classification metrics for speech emotion recognition.

* **WAR** (weighted average recall) = plain accuracy.
* **UAR** (unweighted average recall) = mean per-class recall, the standard SER headline
  metric because it does not reward ignoring a rare class.
* Macro/weighted F1, per-class precision/recall/F1, row-normalised confusion matrix.
* One-vs-rest ROC-AUC and average precision (PR-AUC), multi-class log-loss, and the expected
  calibration error (ECE).

Confidence intervals resample **speakers**, not clips: clips of one speaker are correlated,
so a clip-level bootstrap would make the intervals look much tighter than they are.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def uar(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> float:
    """Mean recall over the classes present in ``y_true``."""
    classes = np.unique(y_true) if n_classes is None else np.arange(n_classes)
    recalls = [np.mean(y_pred[y_true == c] == c) for c in classes if np.any(y_true == c)]
    return float(np.mean(recalls))


def summary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """WAR, UAR and macro-F1 from labels and logits or probabilities (only argmax is used)."""
    y_pred = scores.argmax(axis=1)
    return {
        "war": float(np.mean(y_pred == y_true)),
        "uar": uar(y_true, y_pred),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: |accuracy − confidence| averaged over equal-width confidence bins."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def full_report(y_true: np.ndarray, probs: np.ndarray, classes: Sequence[str]) -> dict[str, Any]:
    """Every metric above for one set of predictions (``probs`` rows sum to 1)."""
    n_classes = len(classes)
    labels = np.arange(n_classes)
    y_pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore"):
        normalized = counts / counts.sum(axis=1, keepdims=True)
    present = np.unique(y_true)
    report: dict[str, Any] = {
        "n": len(y_true),
        **summary_metrics(y_true, probs),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(classes)
        },
        "confusion": counts.tolist(),
        "confusion_normalized": np.nan_to_num(normalized).round(4).tolist(),
        "ece": expected_calibration_error(y_true, probs),
    }
    if len(present) == n_classes:
        onehot = np.eye(n_classes)[y_true]
        report["roc_auc_macro"] = float(roc_auc_score(onehot, probs, average="macro"))
        report["pr_auc_macro"] = float(average_precision_score(onehot, probs, average="macro"))
        clipped = np.clip(probs, 1e-7, 1.0)
        report["log_loss"] = float(
            log_loss(y_true, clipped / clipped.sum(1, keepdims=True), labels=labels)
        )
    return report


MetricFn = Callable[[np.ndarray, np.ndarray], float]
BOOTSTRAP_METRICS: dict[str, MetricFn] = {
    "war": lambda y, p: float(np.mean(y == p)),
    "uar": uar,
    "macro_f1": lambda y, p: float(f1_score(y, p, average="macro", zero_division=0)),
}


def speaker_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    speakers: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Percentile CIs from resampling whole speakers with replacement."""
    rng = np.random.default_rng(seed)
    unique = np.unique(speakers)
    rows_of = {s: np.flatnonzero(speakers == s) for s in unique}
    draws: dict[str, list[float]] = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[s] for s in rng.choice(unique, size=len(unique))])
        for name, fn in BOOTSTRAP_METRICS.items():
            draws[name].append(fn(y_true[rows], y_pred[rows]))
    tail = (1.0 - level) / 2 * 100
    return {
        name: (float(np.percentile(v, tail)), float(np.percentile(v, 100 - tail)))
        for name, v in draws.items()
    }


def paired_speaker_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    speakers: np.ndarray,
    metric: str = "uar",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Difference ``metric(a) − metric(b)`` with a speaker-level paired bootstrap.

    Returns the observed difference, its 95 % interval and a two-sided p-value for "no
    difference" (share of resamples whose difference has the opposite sign, doubled).
    """
    fn = BOOTSTRAP_METRICS[metric]
    rng = np.random.default_rng(seed)
    unique = np.unique(speakers)
    rows_of = {s: np.flatnonzero(speakers == s) for s in unique}
    observed = fn(y_true, pred_a) - fn(y_true, pred_b)
    diffs = []
    for _ in range(n_boot):
        rows = np.concatenate([rows_of[s] for s in rng.choice(unique, size=len(unique))])
        diffs.append(fn(y_true[rows], pred_a[rows]) - fn(y_true[rows], pred_b[rows]))
    diffs_arr = np.asarray(diffs)
    p_value = 2 * min(np.mean(diffs_arr <= 0), np.mean(diffs_arr >= 0))
    return {
        "difference": float(observed),
        "ci_low": float(np.percentile(diffs_arr, 2.5)),
        "ci_high": float(np.percentile(diffs_arr, 97.5)),
        "p_value": float(min(1.0, p_value)),
    }
