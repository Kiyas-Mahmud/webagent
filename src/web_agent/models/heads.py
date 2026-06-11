"""The four task heads — FIXED across all 19 models (SPEC 4.2 / ARCH 3-5).

Every head reads the same fused [B, 768] embedding. Do NOT change these when
swapping backbones — that constancy is the backbone-agnostic claim.

  FailureHead (Pillar 1): outcome(2), failure_type(4), confidence(1 sigmoid), recovery(6)
  ActionHead  (Pillar 3): action_type(5), bbox(4 sigmoid -> [0,1])
  MemoryHead  (Pillar 4): memory_flag(1 sigmoid), recovery(6)

Each head trunk: Linear(768->256) -> ReLU -> Dropout(0.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from web_agent.labels import (
    NUM_ACTION_TYPE,
    NUM_FAILURE_TYPE,
    NUM_OUTCOME,
    NUM_RECOVERY,
)


def _trunk(dim: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class FailureHead(nn.Module):
    """Pillar 1 — outcome, failure_type, confidence, recovery."""

    def __init__(self, dim: int = 768, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.trunk = _trunk(dim, hidden, dropout)
        self.outcome = nn.Linear(hidden, NUM_OUTCOME)       # 2
        self.failure_type = nn.Linear(hidden, NUM_FAILURE_TYPE)  # 4
        self.confidence = nn.Linear(hidden, 1)              # sigmoid in forward
        self.recovery = nn.Linear(hidden, NUM_RECOVERY)     # 6

    def forward(self, fused: torch.Tensor) -> dict:
        h = self.trunk(fused)
        return {
            "outcome": self.outcome(h),
            "failure_type": self.failure_type(h),
            "confidence": torch.sigmoid(self.confidence(h)),
            "recovery": self.recovery(h),
        }


class ActionHead(nn.Module):
    """Pillar 3 — action_type (5-way) + bbox (4 floats in [0,1])."""

    def __init__(self, dim: int = 768, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.trunk = _trunk(dim, hidden, dropout)
        self.action_type = nn.Linear(hidden, NUM_ACTION_TYPE)  # 5
        self.bbox = nn.Linear(hidden, 4)

    def forward(self, fused: torch.Tensor) -> dict:
        h = self.trunk(fused)
        return {
            "action_type": self.action_type(h),
            "bbox": torch.sigmoid(self.bbox(h)),  # normalized [0,1] like the targets
        }


class MemoryHead(nn.Module):
    """Pillar 4 — memory_update_flag (sigmoid) + recovery_strategy (6-way)."""

    def __init__(self, dim: int = 768, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.trunk = _trunk(dim, hidden, dropout)
        self.memory_flag = nn.Linear(hidden, 1)             # sigmoid in forward
        self.recovery = nn.Linear(hidden, NUM_RECOVERY)     # 6

    def forward(self, fused: torch.Tensor) -> dict:
        h = self.trunk(fused)
        return {
            "memory_flag": self.memory_flag(h),     # LOGIT (BCE-with-logits; autocast-safe)
            "memory_recovery": self.recovery(h),
        }


class RecoveryOutcomeHead(nn.Module):
    """Did the recovery succeed? sigmoid + BCE vs recovery_success."""

    def __init__(self, dim: int = 768, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.trunk = _trunk(dim, hidden, dropout)
        self.recovery_outcome = nn.Linear(hidden, 1)        # sigmoid in forward

    def forward(self, fused: torch.Tensor) -> dict:
        h = self.trunk(fused)
        return {"recovery_outcome": self.recovery_outcome(h)}  # LOGIT (BCE-with-logits)
