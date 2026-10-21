"""Diagnostic figures (matplotlib, Agg backend, PNG at 150 dpi)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

PALETTE = ("#2a6fdb", "#e4572e", "#29a36a", "#f3a712", "#8e5bd8", "#5c6b73", "#c2185b")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, classes: Sequence[str], title: str, path: Path) -> None:
    """Row-normalised confusion matrix (recall on the diagonal), annotated in percent."""
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{100 * value:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.55 else "#1b1b1b",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def plot_roc_pr(y_true: np.ndarray, probs: np.ndarray, classes: Sequence[str], path: Path) -> None:
    """One-vs-rest ROC and precision-recall curves per emotion."""
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(10, 4.2))
    for k, name in enumerate(classes):
        target = (y_true == k).astype(int)
        fpr, tpr, _ = roc_curve(target, probs[:, k])
        precision, recall, _ = precision_recall_curve(target, probs[:, k])
        color = PALETTE[k % len(PALETTE)]
        roc_ax.plot(fpr, tpr, color=color, lw=1.6, label=name)
        pr_ax.plot(recall, precision, color=color, lw=1.6, label=name)
    roc_ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
    roc_ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC (one-vs-rest)")
    pr_ax.set(xlabel="recall", ylabel="precision", title="Precision-recall", ylim=(0, 1.02))
    roc_ax.legend(fontsize=8, frameon=False)
    _save(fig, path)


def plot_reliability(y_true: np.ndarray, probs: np.ndarray, path: Path, n_bins: int = 10) -> None:
    """Top-label reliability diagram with the share of clips per confidence bin."""
    confidence, correct = probs.max(1), probs.argmax(1) == y_true
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bins = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    accuracy = np.array(
        [correct[bins == b].mean() if (bins == b).any() else np.nan for b in range(n_bins)]
    )
    share = np.bincount(bins, minlength=n_bins) / len(bins)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.bar(
        centers,
        accuracy,
        width=1 / n_bins,
        color=PALETTE[0],
        alpha=0.8,
        edgecolor="white",
        label="accuracy",
    )
    ax.bar(centers, share, width=1 / n_bins, color=PALETTE[5], alpha=0.35, label="share of clips")
    ax.plot([0, 1], [0, 1], color="#333333", lw=0.8, ls="--")
    ax.set(xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1), title="Reliability")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _save(fig, path)


def plot_training_curves(histories: Mapping[int, list[dict]], path: Path) -> None:
    """Train/val loss and validation UAR per epoch for every fold."""
    if not histories or len(next(iter(histories.values()))) < 2:
        return
    fig, (loss_ax, uar_ax) = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, history in sorted(histories.items()):
        epochs = [r["epoch"] for r in history]
        color = PALETTE[k % len(PALETTE)]
        loss_ax.plot(epochs, [r["train_loss"] for r in history], color=color, lw=1.2, ls="--")
        if "val_loss" in history[0]:
            loss_ax.plot(
                epochs, [r["val_loss"] for r in history], color=color, lw=1.4, label=f"fold {k}"
            )
            uar_ax.plot(epochs, [r["uar"] for r in history], color=color, lw=1.4, label=f"fold {k}")
    loss_ax.set(xlabel="epoch", ylabel="loss", title="Loss (dashed: train, solid: val)")
    uar_ax.set(xlabel="epoch", ylabel="UAR", title="Validation UAR")
    uar_ax.legend(fontsize=8, frameon=False)
    _save(fig, path)


def plot_per_corpus(metrics: Mapping[str, Mapping[str, float]], path: Path, title: str) -> None:
    """Grouped bars of UAR / WAR / macro-F1 per corpus."""
    names = list(metrics)
    keys = ("uar", "war", "macro_f1")
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(1.4 * len(names) + 2.5, 3.6))
    for j, key in enumerate(keys):
        values = [metrics[n][key] for n in names]
        bars = ax.bar(
            x + (j - 1) * 0.26,
            values,
            width=0.26,
            color=PALETTE[j],
            label=key.upper().replace("_", "-"),
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(x, names)
    ax.set(ylim=(0, 1.08), ylabel="score", title=title)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower right")
    _save(fig, path)


def plot_layer_weights(weights: np.ndarray, path: Path, backbone: str) -> None:
    """Learned softmax weight of every SSL layer (mean ± std over folds)."""
    mean, std = weights.mean(0), weights.std(0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(np.arange(len(mean)), mean, yerr=std, color=PALETTE[0], alpha=0.85, capsize=2)
    ax.axhline(1 / len(mean), color="#999999", lw=0.8, ls="--")
    ax.set(
        xlabel="layer (0 = CNN encoder output)", ylabel="weight", title=f"Layer weights, {backbone}"
    )
    _save(fig, path)


def plot_benchmark(rows: list[dict], path: Path) -> None:
    """Horizontal bars of test UAR with speaker-bootstrap 95 % intervals, one per run.

    ``rows`` are ``{"label", "uar", "ci", "selected"}``; the selected model (chosen on dev
    folds, not on this chart) is drawn in colour.
    """
    rows = sorted(rows, key=lambda r: r["uar"])
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(rows) + 1.3))
    y = np.arange(len(rows))
    uar = np.array([r["uar"] for r in rows])
    lows = np.array([r["ci"][0] for r in rows])
    highs = np.array([r["ci"][1] for r in rows])
    colors = [PALETTE[0] if r.get("selected") else PALETTE[5] for r in rows]
    ax.barh(y, uar, color=colors, alpha=0.85)
    ax.errorbar(uar, y, xerr=[uar - lows, highs - uar], fmt="none", ecolor="#222222", capsize=3)
    for yi, value, high in zip(y, uar, highs, strict=True):
        ax.text(high + 0.012, yi, f"{value:.3f}", va="center", fontsize=9)
    ax.axvline(1 / 6, color="#999999", lw=0.8, ls="--")
    ax.text(1 / 6 + 0.006, -0.62, "chance (1/6)", color="#777777", fontsize=8, va="center")
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set(xlim=(0, 1.0), ylim=(-0.9, len(rows) - 0.4))
    ax.set_xlabel("UAR on 25 held-out speakers (95% speaker-bootstrap CI)")
    ax.set_title("Speaker-independent test results", fontsize=11)
    _save(fig, path)
