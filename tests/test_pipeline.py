"""End-to-end: synthetic corpora → cache → CV training → reports → ensemble → inference."""

import json

import numpy as np
import pytest
import soundfile as sf

from aum2.data.dataset import load_corpus
from aum2.evaluation.report import benchmark_table, ensemble_runs
from aum2.inference import EmotionRecognizer
from aum2.training.experiment import run_experiment

from .conftest import synthetic_voice, tiny_config


@pytest.fixture(scope="module")
def runs(raw_dir, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipeline")
    crnn = run_experiment(tiny_config(raw_dir, tmp, "crnn"), n_jobs=1)
    handcrafted = run_experiment(tiny_config(raw_dir, tmp, "handcrafted"), n_jobs=1)
    legacy = run_experiment(tiny_config(raw_dir, tmp, "legacy_cascade"), n_jobs=1)
    return {"tmp": tmp, "crnn": crnn, "handcrafted": handcrafted, "legacy": legacy}


def test_cache_conditions_every_clip_to_16k(raw_dir, tmp_path):
    corpus = load_corpus(tiny_config(raw_dir, tmp_path, "crnn").data, n_jobs=1)
    assert len(corpus) == 72 + 24 + 36 + 48  # six emotions only, RAVDESS song excluded
    assert corpus.sample_rate == 16000
    # 0.2 s silences were trimmed (minus the 0.1 s margin) from ~0.9-1.5 s voices.
    assert corpus.manifest["duration"].between(0.8, 1.6).all()
    # The second call reuses the cache.
    assert load_corpus(tiny_config(raw_dir, tmp_path, "crnn").data, n_jobs=1).key == corpus.key


@pytest.mark.parametrize("family", ["crnn", "handcrafted", "legacy"])
def test_runs_write_complete_reports(runs, family):
    run_dir = runs[family]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["folds"] == [0, 1]
    for part in ("dev", "test"):
        assert 0.0 <= metrics[part]["uar"] <= 1.0
        assert set(metrics[part]["per_class"]) == {
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
        }
    assert metrics["test"]["ci95"]["uar"][0] <= metrics["test"]["ci95"]["uar"][1]
    assert (run_dir / "figures" / "confusion_test.png").exists()
    assert (run_dir / "predictions.csv.gz").exists()


def test_training_resumes_from_finished_folds(runs, raw_dir, caplog):
    cfg = tiny_config(raw_dir, runs["tmp"], "crnn")
    before = (runs["crnn"] / "fold0" / "predictions.npz").stat().st_mtime
    run_experiment(cfg, n_jobs=1)
    assert (runs["crnn"] / "fold0" / "predictions.npz").stat().st_mtime == before


def test_changed_settings_never_reuse_finished_folds(runs, raw_dir):
    cfg = tiny_config(raw_dir, runs["tmp"], "crnn", train={"learning_rate": 0.5})
    with pytest.raises(SystemExit, match=r"train\.learning_rate"):
        run_experiment(cfg, n_jobs=1)


def test_ensemble_and_benchmark(runs):
    out = ensemble_runs(
        [runs["crnn"], runs["handcrafted"]], "ens", runs["tmp"] / "runs", tune_weights=True
    )
    weights = json.loads((out / "bundle.json").read_text())["weights"]
    assert sum(weights) == pytest.approx(1.0)
    table = benchmark_table([runs["crnn"], runs["handcrafted"], out])
    assert table.count("\n") == 4 and "tiny_crnn" in table

    recognizer = EmotionRecognizer(out, device="cpu")
    rng = np.random.default_rng(5)
    prediction = recognizer.predict_waveform(synthetic_voice("angry", 16000, rng), 16000)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("family", ["crnn", "handcrafted"])
def test_recognizer_handles_short_and_long_recordings(runs, family, tmp_path):
    recognizer = EmotionRecognizer(runs[family], device="cpu")
    rng = np.random.default_rng(3)
    short = tmp_path / "short.wav"
    sf.write(short, synthetic_voice("happy", 44100, rng), 44100)
    prediction = recognizer.predict_file(short)
    assert prediction.label in recognizer.classes
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert prediction.segments == ()

    # 10 s: voice, 4 s of silence, voice -> windows, silent ones skipped.
    voice = synthetic_voice("sad", 16000, rng, seconds=2.5)
    long = np.concatenate([voice, np.zeros(4 * 16000, dtype=np.float32), voice, voice])
    result = recognizer.predict_waveform(long, 16000)
    assert len(result.segments) >= 2
    assert all(seg.end - seg.start == pytest.approx(3.0) for seg in result.segments)
    with pytest.raises(ValueError, match="no speech"):
        recognizer.predict_waveform(np.zeros(16000, dtype=np.float32), 16000)


def test_legacy_cascade_is_evaluation_only(runs):
    with pytest.raises(ValueError, match="evaluation-only"):
        EmotionRecognizer(runs["legacy"], device="cpu")
