import numpy as np
import pytest
import torch

from aum2.config import AugmentConfig, FrontendConfig
from aum2.features.augment import SpecAugment, WaveAugment
from aum2.features.frontend import LogMelFrontend, fit_normalization, mel_filterbank


@pytest.mark.parametrize(
    "n_fft, n_mels, f_min, f_max", [(400, 80, 20.0, 8000.0), (2048, 128, 0.0, 8000.0)]
)
def test_mel_filterbank_matches_librosa(n_fft, n_mels, f_min, f_max):
    librosa = pytest.importorskip("librosa")
    ours = mel_filterbank(16000, n_fft, n_mels, f_min, f_max)
    ref = librosa.filters.mel(sr=16000, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max)
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_frontend_is_padding_invariant():
    """A clip yields identical features alone and inside a longer, zero-padded batch."""
    frontend = LogMelFrontend(FrontendConfig(), 16000)
    frontend.set_normalization(torch.full((80,), -5.0), torch.full((80,), 2.0))
    rng = np.random.default_rng(0)
    clip = torch.from_numpy(rng.normal(scale=0.1, size=12345).astype(np.float32))
    alone, frames = frontend(clip[None], torch.tensor([12345]))
    padded = torch.zeros(2, 20000)
    padded[0, :12345] = clip
    padded[1] = torch.from_numpy(rng.normal(scale=0.1, size=20000).astype(np.float32))
    batch, batch_frames = frontend(padded, torch.tensor([12345, 20000]))
    n = int(frames[0])
    assert int(batch_frames[0]) == n
    torch.testing.assert_close(batch[0, :, :n], alone[0, :, :n], atol=1e-4, rtol=1e-4)
    assert torch.all(batch[0, :, n:] == 0)


def test_fit_normalization_standardises_training_frames():
    frontend = LogMelFrontend(FrontendConfig(n_mels=40), 16000)
    rng = np.random.default_rng(1)
    batches = [
        (
            torch.from_numpy(rng.normal(scale=0.2, size=(4, 16000)).astype(np.float32)),
            torch.tensor([16000, 12000, 9000, 16000]),
        )
    ]
    mean, std = fit_normalization(frontend, batches)
    frontend.set_normalization(mean, std)
    feats, frames = frontend(*batches[0])
    valid = torch.cat([feats[i, :, : frames[i]] for i in range(4)], dim=1)
    torch.testing.assert_close(valid.mean(1), torch.zeros(40), atol=1e-3, rtol=0)
    torch.testing.assert_close(valid.std(1, unbiased=False), torch.ones(40), atol=1e-3, rtol=0)


def test_augmentations_are_identity_in_eval_mode_and_random_in_train_mode():
    cfg = AugmentConfig(noise_prob=1.0, tempo_prob=1.0, vtlp_prob=1.0)
    wave_aug, spec_aug = WaveAugment(cfg), SpecAugment(cfg)
    wave, lengths = torch.randn(3, 8000), torch.tensor([8000, 6000, 4000])
    wave[1, 6000:], wave[2, 4000:] = 0, 0  # batches are zero-padded
    feats, frames = torch.randn(3, 80, 51), torch.tensor([51, 38, 26])
    wave_aug.eval(), spec_aug.eval()
    assert torch.equal(wave_aug(wave, lengths), wave)
    out, out_frames = spec_aug(feats, frames)
    assert torch.equal(out, feats) and torch.equal(out_frames, frames)

    wave_aug.train(), spec_aug.train()
    noisy = wave_aug(wave, lengths)
    assert noisy.shape == wave.shape and not torch.equal(noisy, wave)
    assert torch.all(noisy[2, 4000:] == 0)  # padding stays silent
    warped, new_frames = spec_aug(feats, frames)
    assert warped.shape == feats.shape and torch.all(new_frames >= 1)
    for i in range(3):
        assert torch.all(warped[i, :, new_frames[i] :] == 0)
