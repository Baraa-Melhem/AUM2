import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from aum2.evaluation.metrics import (
    expected_calibration_error,
    full_report,
    paired_speaker_bootstrap,
    speaker_bootstrap,
    summary_metrics,
    uar,
)
from aum2.training.losses import class_weights

CLASSES = ("angry", "disgust", "fear", "happy", "neutral", "sad")


def _predictions(n: int = 600, accuracy: float = 0.7, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 6, n)
    pred = np.where(rng.random(n) < accuracy, y, rng.integers(0, 6, n))
    probs = np.full((n, 6), 0.04)
    probs[np.arange(n), pred] = 0.8
    return y, pred, probs, np.repeat(np.arange(20), n // 20)


def test_uar_equals_balanced_accuracy():
    y, pred, _, _ = _predictions()
    assert uar(y, pred) == pytest.approx(balanced_accuracy_score(y, pred))


def test_summary_and_full_report_are_consistent():
    y, pred, probs, _ = _predictions()
    report = full_report(y, probs, CLASSES)
    assert report["war"] == pytest.approx(np.mean(pred == y))
    assert report["uar"] == pytest.approx(summary_metrics(y, probs)["uar"])
    assert np.sum(report["confusion"]) == len(y)
    assert np.allclose(np.sum(report["confusion_normalized"], axis=1), 1.0, atol=1e-3)
    assert 0.5 < report["roc_auc_macro"] <= 1.0
    recalls = [report["per_class"][c]["recall"] for c in CLASSES]
    assert np.mean(recalls) == pytest.approx(report["uar"])


def test_ece_is_zero_for_perfectly_calibrated_confidence():
    y = np.array([0, 1, 0, 1])
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(y, probs) == pytest.approx(0.0)


def test_speaker_bootstrap_interval_contains_estimate():
    y, pred, _, speakers = _predictions()
    ci = speaker_bootstrap(y, pred, speakers, n_boot=200)
    point = uar(y, pred)
    assert ci["uar"][0] <= point <= ci["uar"][1]


def test_paired_bootstrap_detects_a_real_difference_and_not_a_fake_one():
    y, good, _, speakers = _predictions(accuracy=0.9, seed=1)
    _, bad, _, _ = _predictions(accuracy=0.4, seed=2)
    assert paired_speaker_bootstrap(y, good, bad, speakers, n_boot=300)["p_value"] < 0.01
    assert paired_speaker_bootstrap(y, good, good, speakers, n_boot=300)["difference"] == 0.0


def test_class_weights_balance_the_loss():
    labels = np.array([0] * 30 + [1] * 10)
    weights = class_weights(labels, 2).numpy()
    assert weights[0] * 30 == pytest.approx(weights[1] * 10)


def test_temperature_scaling_sharpens_underconfident_probs_without_changing_argmax():
    from aum2.evaluation.calibration import apply_temperature, fit_temperature

    rng = np.random.default_rng(4)
    logits = rng.normal(scale=3.0, size=(5000, 6))
    calibrated_truth = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(6, p=p) for p in calibrated_truth])  # labels drawn from the model
    flat = np.exp(logits / 4.0)  # the same model with its probabilities flattened (T = 4)
    flat /= flat.sum(1, keepdims=True)
    temperature = fit_temperature(flat, y)
    assert temperature == pytest.approx(0.25, rel=0.15)
    calibrated = apply_temperature(flat, temperature)
    assert np.array_equal(calibrated.argmax(1), flat.argmax(1))
    assert expected_calibration_error(y, calibrated) < expected_calibration_error(y, flat)
    np.testing.assert_allclose(apply_temperature(flat, 1.0), flat, atol=1e-6)
