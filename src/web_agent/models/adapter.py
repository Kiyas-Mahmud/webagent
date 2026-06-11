"""VLM adapter — maps the pooled VLM output into the shared 768-dim space (SPEC 4.5).

Spec: Linear(D->768) -> LayerNorm(768) -> Dropout(0.1). Replaces cross-attention
fusion on the VLM path; the 5 heads + loss stay identical. This small block IS the
backbone-agnostic claim for unified VLMs.
"""

from __future__ import annotations

import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.norm(self.proj(x)))
