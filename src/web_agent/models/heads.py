"""Shared task heads (SPEC 4.2 / ARCH 3-5).

Every head reads the same fused [B, 768] embedding. Do NOT change these when
swapping backbones — that constancy is the backbone-agnostic claim.

  FailureHead (Pillar 1): outcome(2), failure_type(4), confidence(1 sigmoid), recovery(6)
  ActionHead  (Pillar 3): action_type(6), bbox(4 sigmoid -> [0,1])
  MemoryHead  (Pillar 4): memory_flag(1 sigmoid), recovery(6)

Each head trunk: Linear(768->256) -> ReLU -> Dropout(0.3). The registered Gold
recovery-v2 experiment adds a needs-recovery logit and optional image-token
grounding while keeping the legacy path available for controlled comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from web_agent.models.bbox import cxcywh_to_bounded_xywh
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

    def __init__(
        self,
        dim: int = 768,
        hidden: int = 256,
        dropout: float = 0.3,
        needs_recovery: bool = False,
    ):
        super().__init__()
        self.trunk = _trunk(dim, hidden, dropout)
        self.outcome = nn.Linear(hidden, NUM_OUTCOME)       # 2
        self.failure_type = nn.Linear(hidden, NUM_FAILURE_TYPE)  # 4
        self.confidence = nn.Linear(hidden, 1)              # sigmoid in forward
        self.recovery = nn.Linear(hidden, NUM_RECOVERY)     # 6
        self.needs_recovery = nn.Linear(hidden, 1) if needs_recovery else None

    def forward(
        self,
        fused: torch.Tensor,
        confidence_fused: torch.Tensor | None = None,
    ) -> dict:
        """Use post-action features for diagnosis and pre-action features for confidence."""
        h = self.trunk(fused)
        confidence_h = h if confidence_fused is None else self.trunk(confidence_fused)
        out = {
            "outcome": self.outcome(h),
            "failure_type": self.failure_type(h),
            "confidence": torch.sigmoid(self.confidence(confidence_h)),
            "recovery": self.recovery(h),
        }
        if self.needs_recovery is not None:
            out["needs_recovery"] = self.needs_recovery(h)
        return out


class ActionHead(nn.Module):
    """Pillar 3 — action_type (6-way) + bbox (4 floats in [0,1])."""

    def __init__(
        self,
        dim: int = 768,
        hidden: int = 256,
        dropout: float = 0.3,
        spatial_grounding: bool = False,
        bbox_parameterization: str = "legacy_xywh",
    ):
        super().__init__()
        self.spatial_grounding = spatial_grounding
        if bbox_parameterization not in {"legacy_xywh", "cxcywh"}:
            raise ValueError(
                f"unsupported bbox parameterization: {bbox_parameterization!r}"
            )
        self.bbox_parameterization = bbox_parameterization
        self.trunk = _trunk(dim, hidden, dropout)
        self.action_type = nn.Linear(hidden, NUM_ACTION_TYPE)  # 6
        if spatial_grounding:
            self.grounding_attention = nn.MultiheadAttention(
                dim, num_heads=8, dropout=dropout, batch_first=True,
            )
            self.bbox_trunk = _trunk(dim * 2, hidden, dropout)
        self.bbox = nn.Linear(hidden, 4)

    def forward(
        self,
        fused: torch.Tensor,
        bbox_fused: torch.Tensor | None = None,
        spatial_tokens: torch.Tensor | None = None,
        spatial_mask: torch.Tensor | None = None,
    ) -> dict:
        h = self.trunk(fused)
        bbox_h = h
        if self.spatial_grounding:
            if spatial_tokens is None or spatial_mask is None:
                raise ValueError("spatial grounding requires image tokens and a token mask")
            bbox_query = bbox_fused if bbox_fused is not None else fused
            grounded, _ = self.grounding_attention(
                bbox_query.unsqueeze(1),
                spatial_tokens,
                spatial_tokens,
                key_padding_mask=~spatial_mask.bool(),
                need_weights=False,
            )
            bbox_h = self.bbox_trunk(torch.cat([
                bbox_query, grounded.squeeze(1),
            ], dim=-1))
        raw_bbox = torch.sigmoid(self.bbox(bbox_h))
        if self.bbox_parameterization == "cxcywh":
            bbox = cxcywh_to_bounded_xywh(raw_bbox)
        elif self.spatial_grounding:
            # xywh remains normalized and is guaranteed to stay within the image.
            xy = raw_bbox[:, :2]
            bbox = torch.cat([xy, raw_bbox[:, 2:] * (1.0 - xy)], dim=-1)
        else:
            bbox = raw_bbox
        result = {
            "action_type": self.action_type(h),
            "bbox": bbox,
        }
        if self.bbox_parameterization == "cxcywh":
            result["bbox_cxcywh"] = raw_bbox
        return result


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
