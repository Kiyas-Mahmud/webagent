"""Cross-Attention Fusion — Pillar 2, the architectural novelty (SPEC 4.2 / ARCH 2.2).

Two transformer cross-attention blocks: vision attends to text (Q=vision, K=V=text)
and text attends to vision (Q=text, K=V=vision). Concat -> LayerNorm ->
Linear(1536->768) -> Dropout(0.1) -> fused [B, 768]. Used by dual-encoder configs
(Y1,Y2,Y3,Y7). VLM configs skip this and use adapter.py instead.

Ablation A3 replaces this with plain concat+Linear to prove cross-attention > concat.
"""

from __future__ import annotations

import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 768, num_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        # TODO(Phase 3): build 2 cross-attention blocks + concat/LayerNorm/Linear(1536->768).
        raise NotImplementedError("Build in Phase 3 (see docs/IMPLEMENTATION_PLAN.md §3.2).")

    def forward(self, vision_emb, text_emb):
        raise NotImplementedError
