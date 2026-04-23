"""Deterministic seeding across Python, NumPy, and PyTorch.

Call ``set_deterministic_seed(seed)`` at the top of any entry point to
make results reproducible on a given hardware/driver combination. This
does not guarantee bitwise reproducibility across GPU generations.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_deterministic_seed(seed: int) -> None:
    """Seed all relevant RNGs and request deterministic cuDNN/cuBLAS."""
    assert isinstance(seed, int) and seed >= 0, f"bad seed: {seed!r}"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
