"""Text encoder wrapper -> [B, 768] CLS token (SPEC 4.2).

Default roberta-base (Y1/Y3/Y7), roberta-large for Y2. Same freeze schedule as vision.
"""

from __future__ import annotations

import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, out_dim: int = 768):
        super().__init__()
        # TODO(Phase 3): load AutoModel; take CLS (pooler or last_hidden[:,0]); project.
        raise NotImplementedError("Build in Phase 3 (see docs/IMPLEMENTATION_PLAN.md §3).")

    def forward(self, input_ids, attention_mask):
        raise NotImplementedError
