"""Post-hoc temperature scaling (Guo et al., 2017).

Label smoothing and fold averaging both make the predicted probabilities too flat. Each fold
model gets one temperature ``T``, fitted by minimising the negative log-likelihood on *that
fold's validation speakers*, which the model never trained on; it changes no prediction (argmax
is unchanged), only how confident the probabilities are. It is stored next to the weights
(``calibration.json``) and applied the same way in evaluation and in inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-7


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale probabilities as ``softmax(log(p) / T)``."""
    logits = np.log(np.clip(probs, _EPS, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=-1, keepdims=True)


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """Temperature in [0.05, 20] minimising the multi-class NLL of ``probs``."""
    rows = np.arange(len(labels))

    def nll(log_t: float) -> float:
        return float(-np.log(apply_temperature(probs, np.exp(log_t))[rows, labels] + _EPS).mean())

    result = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(result.x))


def save_temperature(fold_dir: Path, temperature: float) -> None:
    (fold_dir / "calibration.json").write_text(
        json.dumps({"temperature": temperature}), encoding="utf-8"
    )


def load_temperature(fold_dir: Path) -> float:
    """The fold's temperature, or 1.0 (no rescaling) if the fold was never calibrated."""
    path = fold_dir / "calibration.json"
    return (
        float(json.loads(path.read_text(encoding="utf-8"))["temperature"]) if path.exists() else 1.0
    )
