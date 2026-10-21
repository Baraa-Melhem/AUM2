"""Frozen self-supervised speech representations (WavLM / HuBERT / wav2vec 2.0).

For every clip, each hidden layer of the backbone is summarised by its mean and standard
deviation over time, giving an ``(n_layers, 2 * hidden)`` matrix. The downstream head learns
a softmax weighting over layers (as in the SUPERB benchmark), so it can find the layers that
carry paralinguistic information (typically the middle ones, not the last).

Because the backbone is frozen, these statistics are computed once and cached (~1 GB in
float16 for WavLM-Large), which turns a multi-hour fine-tune into a seconds-long head fit.
Clips are encoded one at a time, so the numbers are exactly what single-file inference sees
(no padding effects).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from aum2.logging_utils import get_logger

logger = get_logger(__name__)


class SSLEncoder:
    """Wraps a Hugging Face speech backbone and returns pooled per-layer statistics."""

    def __init__(self, backbone: str, device: torch.device) -> None:
        from transformers import AutoConfig, AutoFeatureExtractor, AutoModel
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        # Loading a .bin checkpoint otherwise starts a background thread that asks the Hub's
        # conversion bot for a safetensors copy, a network call even for fully cached models.
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")
        self.backbone = backbone
        self.device = device
        try:  # use the local Hugging Face cache without a network round-trip when possible
            config = AutoConfig.from_pretrained(backbone, local_files_only=True)
            offline = True
        except OSError:
            config, offline = AutoConfig.from_pretrained(backbone), False
        self.model = (
            AutoModel.from_pretrained(backbone, config=config, local_files_only=offline)
            .to(device)
            .eval()
        )
        extractor = AutoFeatureExtractor.from_pretrained(backbone, local_files_only=offline)
        self.normalize = bool(getattr(extractor, "do_normalize", False))
        self.sample_rate = int(extractor.sampling_rate)
        self.n_layers = config.num_hidden_layers + 1
        self.hidden = config.hidden_size

    @torch.inference_mode()
    def encode(self, samples: np.ndarray) -> np.ndarray:
        """``(n_layers, 2 * hidden)`` float32 mean‖std statistics of one 16 kHz clip."""
        wave = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).to(self.device)
        if self.normalize:
            wave = (wave - wave.mean()) / (wave.std() + 1e-7)
        # fp16 autocast keeps layer norms and softmax in fp32, unlike model.half().
        with torch.autocast(
            self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
        ):
            out = self.model(wave[None], output_hidden_states=True)
        layers = torch.stack(out.hidden_states, dim=0)[:, 0].float()  # (n_layers, T, hidden)
        stats = torch.cat([layers.mean(1), layers.std(1, unbiased=False)], dim=-1)
        return stats.cpu().numpy()


def cache_path(cache_dir: str | Path, backbone: str, corpus_key: str) -> Path:
    return Path(cache_dir) / f"ssl_{backbone.replace('/', '__')}_{corpus_key}.npy"


def load_or_extract(
    encoder_factory: Callable[[], SSLEncoder],
    waveforms: Sequence[np.ndarray] | Callable[[int], np.ndarray],
    n_clips: int,
    path: Path,
) -> np.ndarray:
    """Return the cached ``(n_clips, n_layers, 2 * hidden)`` float16 array, extracting if needed.

    Extraction writes into a memory-mapped ``.partial`` file and resumes from the last
    completed clip if interrupted (progress is tracked in a small JSON sidecar).
    """
    if path.exists():
        features = np.load(path, mmap_mode="r")
        if features.shape[0] == n_clips:
            return features
        logger.warning(
            "Stale SSL cache %s (%d rows, need %d); rebuilding", path, len(features), n_clips
        )
    get = waveforms if callable(waveforms) else waveforms.__getitem__
    encoder = encoder_factory()
    partial = path.with_suffix(".partial.npy")
    progress = path.with_suffix(".progress.json")
    shape = (n_clips, encoder.n_layers, 2 * encoder.hidden)
    start = 0
    if partial.exists() and progress.exists():
        state = json.loads(progress.read_text())
        if tuple(state["shape"]) == shape:
            start = int(state["done"])
    if start == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        store = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=shape)
    else:
        store = np.lib.format.open_memmap(partial, mode="r+")
        logger.info("Resuming SSL extraction at clip %d / %d", start, n_clips)
    for i in tqdm(
        range(start, n_clips), desc=f"{encoder.backbone} features", initial=start, total=n_clips
    ):
        store[i] = encoder.encode(get(i)).astype(np.float16)
        if (i + 1) % 500 == 0 or i + 1 == n_clips:
            store.flush()
            progress.write_text(json.dumps({"shape": list(shape), "done": i + 1}))
    del store
    partial.replace(path)
    progress.unlink(missing_ok=True)
    return np.load(path, mmap_mode="r")
