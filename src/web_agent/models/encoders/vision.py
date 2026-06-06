"""Vision encoder wrapper -> [B, 768] (SPEC 4.2).

Swappable: SigLIP (Y1/Y2 default), CLIP-ViT-L/14 (Y3), Florence-2 (Y7).
Frozen in Phase 1, unfrozen (lr 1e-5) in Phase 2. Projects to 768 if native dim differs.
"""

from __future__ import annotations

import torch.nn as nn


class VisionEncoder(nn.Module):
    def __init__(self, model_name: str, out_dim: int = 768):
        super().__init__()
        # TODO(Phase 3): load via transformers AutoModel; add projection to out_dim.
        raise NotImplementedError("Build in Phase 3 (see docs/IMPLEMENTATION_PLAN.md §3).")

    def forward(self, pixel_values):
        raise NotImplementedError
