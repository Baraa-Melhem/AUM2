from pathlib import Path

import pytest

from aum2.config import ConfigError, ExperimentConfig, config_from_dict, load_config, to_dict

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    cfg = load_config(path)
    assert isinstance(cfg, ExperimentConfig)


def test_overrides_are_typed_and_applied():
    cfg = load_config(CONFIGS[0].parent / "crnn.yaml", ["train.epochs=3", "folds=[0, 2]"])
    assert cfg.train.epochs == 3 and cfg.folds == (0, 2) and cfg.fold_ids == (0, 2)


def test_round_trip_through_dict():
    cfg = load_config(CONFIGS[0].parent / "wavlm.yaml")
    assert config_from_dict(to_dict(cfg)) == cfg


@pytest.mark.parametrize(
    "section, values",
    [
        ("model", {"name": "transformer"}),
        ("data", {"classes": ["angry", "bored"]}),
        ("split", {"protocol": "session"}),
        ("augment", {"crop_seconds": 20.0}),
        ("augment", {"tempo_range": [1.2, 0.8]}),
        ("frontend", {"n_mels": 72}),
        ("train", {"loss": "hinge"}),
        ("train", {"epochs": "ten"}),
    ],
)
def test_invalid_values_raise(section, values):
    tree = to_dict(ExperimentConfig())
    tree[section].update(values)
    with pytest.raises(ConfigError):
        config_from_dict(tree)


def test_unknown_keys_raise():
    tree = to_dict(ExperimentConfig())
    tree["train"]["learning_rat"] = 0.1
    with pytest.raises(ConfigError, match="learning_rat"):
        config_from_dict(tree)
