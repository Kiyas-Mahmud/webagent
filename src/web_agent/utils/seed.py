"""Global determinism for Q1 reproducibility (IMPLEMENTATION_PLAN 0.3).

Call set_seed(42) at the start of every entry point. Y1 trains on seeds
42, 1, 7 for mean +/- std; other models use 42 unless time permits more.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed python, numpy, torch (CPU + CUDA) and optionally force determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
