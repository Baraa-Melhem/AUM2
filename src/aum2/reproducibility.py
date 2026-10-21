"""Global seeding and device selection."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from aum2.logging_utils import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA), optionally forcing deterministic kernels.

    Deterministic cuDNN/cuBLAS kernels make recurrent and convolutional training several times
    slower, so they are opt-in; seeding alone already fixes initialisation, data order and
    augmentation draws. ``CUBLAS_WORKSPACE_CONFIG`` only takes effect if it is set before the
    CUDA context is created, so call this before any tensor touches the GPU.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global RNG that libraries use
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(preference: str = "auto") -> torch.device:
    """Map ``auto``/``cuda``/``cpu`` to a usable :class:`torch.device`."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return device
