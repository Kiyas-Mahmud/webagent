"""VLM adapter — single Linear(D->768) replacing fusion on the VLM path (SPEC 4.5).

Keeps the 4 heads and loss identical; only the front-end differs. This one linear
layer IS the backbone-agnostic claim for unified VLMs.
"""

from __future__ import annotations

import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)
