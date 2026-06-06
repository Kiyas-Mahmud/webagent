"""The four task heads — FIXED across all 19 models (SPEC 4.2 / ARCH 3-5).

Every head reads the same fused [B, 768] embedding. Do NOT change these when
swapping backbones — that constancy is the backbone-agnostic claim.

  FailureHead (Pillar 1): outcome(2), failure_type(4), confidence(1 sigmoid), recovery(6)
  ActionHead  (Pillar 3): action_type(5), bbox(4)
  MemoryHead  (Pillar 4): memory_flag(1 sigmoid), recovery(6)

Each head trunk: Linear(768->256) -> ReLU -> Dropout(0.3).
"""

from __future__ import annotations

import torch.nn as nn

from web_agent.labels import (
    NUM_ACTION_TYPE,
    NUM_FAILURE_TYPE,
    NUM_OUTCOME,
    NUM_RECOVERY,
)


class FailureHead(nn.Module):
    """Pillar 1 — outcome, failure_type, confidence, recovery."""

    def __init__(self, dim: int = 768, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        # TODO(Phase 3): trunk + 4 output linears (sigmoid on confidence).
        raise NotImplementedError("Build in Phase 3 (see docs/IMPLEMENTATION_PLAN.md §3).")

    def forward(self, fused):
        raise NotImplementedError


class ActionHead(nn.Module):
    """Pillar 3 — action_type (5-way) + bbox (4 floats)."""

    def __init__(self, dim: int = 768, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        raise NotImplementedError("Build in Phase 3.")

    def forward(self, fused):
        raise NotImplementedError


class MemoryHead(nn.Module):
    """Pillar 4 — memory_update_flag (sigmoid) + recovery_strategy (6-way)."""

    def __init__(self, dim: int = 768, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        raise NotImplementedError("Build in Phase 3.")

    def forward(self, fused):
        raise NotImplementedError
