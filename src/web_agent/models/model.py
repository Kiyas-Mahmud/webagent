"""WebAgentModel — assembles front-end + FIXED heads into one nn.Module (ARCH 1).

Reads cfg["backbone"]["path"]:
  dual_encoder -> VisionEncoder + TextEncoder -> CrossAttentionFusion -> fused[768]
  vlm          -> VLMEncoder -> Adapter(D->768)                       -> fused[768]
Then the SAME FailureHead + ActionHead + MemoryHead run on fused[768].

forward() returns a dict of all 7 logit/score tensors consumed by CombinedLoss.
Ablations toggle parts via config (A1 vision-only, A2 text-only, A3 concat,
A4 no-memory, A5 no-recovery) without touching this file's structure.
"""

from __future__ import annotations

import torch.nn as nn


class WebAgentModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        # TODO(Phase 3): branch on cfg backbone.path; wire encoders/fusion-or-adapter
        #   + the three heads. Expose encoder/fusion/head param groups for two-phase lr.
        raise NotImplementedError("Build in Phase 3 (see docs/IMPLEMENTATION_PLAN.md §3.1).")

    def forward(self, batch) -> dict:
        """-> {outcome, failure_type, confidence, recovery, action_type, bbox, memory_flag}."""
        raise NotImplementedError

    def param_groups(self) -> list[dict]:
        """Return AdamW param groups with per-component lr for the two-phase schedule."""
        raise NotImplementedError
