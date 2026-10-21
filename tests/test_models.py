import numpy as np
import torch

from aum2.config import AugmentConfig, CRNNConfig, FrontendConfig, SSLConfig
from aum2.models import LogMelClassifier
from aum2.models.layers import AttentiveStatsPool
from aum2.models.legacy import LegacyCNN, route, subtask_targets
from aum2.models.ssl_head import LayerWeightedHead


def _small_classifier() -> LogMelClassifier:
    torch.manual_seed(0)
    cfg = CRNNConfig(channels=(8, 16, 16, 16), proj_dim=16, gru_hidden=8, attention_dim=8)
    model = LogMelClassifier(FrontendConfig(n_mels=32), AugmentConfig(), cfg, 6, 16000)
    return model.eval()


def test_crnn_output_shape_and_padding_invariance():
    """Masking + packed GRU + masked pooling: padding a clip must not change its logits."""
    model = _small_classifier()
    rng = np.random.default_rng(0)
    clip = torch.from_numpy(rng.normal(scale=0.1, size=13000).astype(np.float32))
    other = torch.from_numpy(rng.normal(scale=0.1, size=24000).astype(np.float32))
    with torch.no_grad():
        alone = model(clip[None], torch.tensor([13000]))
        batch = torch.zeros(2, 24000)
        batch[0, :13000], batch[1] = clip, other
        both = model(batch, torch.tensor([13000, 24000]))
    assert both.shape == (2, 6)
    torch.testing.assert_close(both[0], alone[0], atol=1e-4, rtol=1e-4)


def test_classifier_eval_mode_is_deterministic():
    model = _small_classifier()
    wave, lengths = torch.randn(2, 16000) * 0.1, torch.tensor([16000, 12000])
    with torch.no_grad():
        assert torch.equal(model(wave, lengths), model(wave, lengths))


def test_attentive_pool_ignores_padded_frames():
    pool = AttentiveStatsPool(4, 3)
    x = torch.randn(1, 5, 4)
    padded = torch.cat([x, torch.randn(1, 7, 4) * 100], dim=1)
    torch.testing.assert_close(pool(padded, torch.tensor([5])), pool(x, torch.tensor([5])))


def test_layer_weighted_head_mixes_layers_with_softmax_weights():
    head = LayerWeightedHead(n_layers=5, feat_dim=12, n_classes=6, cfg=SSLConfig(hidden_dim=8))
    assert torch.isclose(head.layer_weights.sum(), torch.tensor(1.0))
    assert head(torch.randn(3, 5, 12)).shape == (3, 6)


def test_legacy_cnn_matches_original_parameter_count():
    # Conv(16) 160 + Dense(63504 -> 32) 2 032 160 + Dense(32 -> 6) 198 for 128x128 inputs.
    model = LegacyCNN(6, 128)
    assert sum(p.numel() for p in model.parameters()) == 160 + 2_032_160 + 198
    assert model(torch.zeros(2, 128, 128)).shape == (2, 6)


def test_cascade_routing_reproduces_models_system():
    #                   M1 M2 M3 M4 M5
    cases = [
        ((0, 0, 0, 0, 0), "angry"),  # negative -> M2 angry -> M5 angry
        ((0, 0, 0, 0, 1), "happy"),  # negative -> M2 angry -> M5 happy (original quirk)
        ((0, 1, 0, 0, 0), "disgust"),
        ((0, 2, 0, 0, 0), "fear"),
        ((0, 3, 0, 0, 0), "sad"),  # negative -> M2 sad -> M4 sad
        ((0, 3, 0, 1, 0), "neutral"),  # negative -> M2 sad -> M4 neutral
        ((1, 0, 0, 0, 0), "angry"),  # positive -> M3 happy -> M5 angry (original quirk)
        ((1, 0, 0, 0, 1), "happy"),
        ((1, 0, 1, 0, 0), "sad"),  # positive -> M3 neutral -> M4 sad (original quirk)
        ((1, 0, 1, 1, 0), "neutral"),
    ]
    decisions = [np.array([c[0][k] for c in cases]) for k in range(5)]
    assert route(decisions).tolist() == [c[1] for c in cases]


def test_subtask_targets_follow_the_original_label_maps():
    emotions = np.array(["angry", "disgust", "fear", "happy", "neutral", "sad"])
    rows, y = subtask_targets(emotions, 0)
    assert y.tolist() == [0, 0, 0, 1, 1, 0]
    rows, y = subtask_targets(emotions, 3)
    assert emotions[rows].tolist() == ["neutral", "sad"] and y.tolist() == [1, 0]
