"""Shared task heads (SPEC 4.2 / ARCH 3-5).

Every head reads the same fused [B, 768] embedding. Do NOT change these when
swapping backbones — that constancy is the backbone-agnostic claim.

  FailureHead (Pillar 1): outcome(2), failure_type(4), confidence(1 sigmoid), recovery(6)
  ActionHead  (Pillar 3): action_type(6), bbox(4 normalized coordinates)
  MemoryHead  (Pillar 4): memory_flag(1 sigmoid), recovery(6)

Each head trunk: Linear(768->256) -> ReLU -> Dropout(0.3). The registered Gold
recovery-v2 experiment adds a needs-recovery logit and optional image-token
grounding while keeping the legacy path available for controlled comparison.
Recovery-v2.4 optionally predicts width/height in log space to avoid a saturated
sigmoid size branch.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from web_agent.models.bbox import cxcywh_to_bounded_xywh
from web_agent.models.spatial import spatial_soft_argmax
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
        bbox_grounding_mode: str = "content_attention",
        bbox_size_parameterization: str = "sigmoid",
        bbox_log_size_min: float = -9.210340371976184,
        bbox_log_size_max: float = 0.0,
        bbox_attention_dropout: float | None = None,
    ):
        super().__init__()
        self.spatial_grounding = spatial_grounding
        if bbox_parameterization not in {"legacy_xywh", "cxcywh"}:
            raise ValueError(
                f"unsupported bbox parameterization: {bbox_parameterization!r}"
            )
        self.bbox_parameterization = bbox_parameterization
        if bbox_grounding_mode not in {
            "content_attention", "coordinate_softargmax",
        }:
            raise ValueError(f"unsupported bbox grounding mode: {bbox_grounding_mode!r}")
        if bbox_grounding_mode == "coordinate_softargmax" and not spatial_grounding:
            raise ValueError("coordinate_softargmax requires spatial_grounding=true")
        self.bbox_grounding_mode = bbox_grounding_mode
        if bbox_size_parameterization not in {"sigmoid", "log_space"}:
            raise ValueError(
                "unsupported bbox size parameterization: "
                f"{bbox_size_parameterization!r}"
            )
        if (
            bbox_size_parameterization == "log_space"
            and bbox_grounding_mode != "coordinate_softargmax"
        ):
            raise ValueError(
                "log-space bbox size requires coordinate_softargmax grounding"
            )
        if bbox_log_size_min >= bbox_log_size_max:
            raise ValueError("bbox_log_size_min must be smaller than max")
        self.bbox_size_parameterization = bbox_size_parameterization
        self.bbox_log_size_min = float(bbox_log_size_min)
        self.bbox_log_size_max = float(bbox_log_size_max)
        attention_dropout = (
            dropout
            if bbox_attention_dropout is None
            else float(bbox_attention_dropout)
        )
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("bbox_attention_dropout must be in [0, 1)")
        self.trunk = _trunk(dim, hidden, dropout)
        self.action_type = nn.Linear(hidden, NUM_ACTION_TYPE)  # 6
        if spatial_grounding:
            self.grounding_attention = nn.MultiheadAttention(
                dim,
                num_heads=8,
                dropout=attention_dropout,
                batch_first=True,
            )
            self.bbox_trunk = _trunk(dim * 2, hidden, dropout)
            if bbox_grounding_mode == "coordinate_softargmax":
                self.coordinate_projection = nn.Sequential(
                    nn.Linear(2, dim, bias=False),
                    nn.LayerNorm(dim),
                )
        self.bbox = nn.Linear(
            hidden,
            2 if bbox_grounding_mode == "coordinate_softargmax" else 4,
        )

    @torch.no_grad()
    def initialize_bbox_size_prior(self, log_wh) -> None:
        """Initialize the log-size branch from valid training rows only."""
        if self.bbox_size_parameterization != "log_space":
            raise RuntimeError("bbox size prior is only valid in log-space mode")
        prior = torch.as_tensor(
            log_wh,
            dtype=self.bbox.bias.dtype,
            device=self.bbox.bias.device,
        ).view(-1)
        if prior.numel() != 2 or not torch.isfinite(prior).all():
            raise ValueError("bbox log-size prior must contain two finite values")
        self.bbox.weight.zero_()
        self.bbox.bias.copy_(prior)

    def forward(
        self,
        fused: torch.Tensor,
        bbox_fused: torch.Tensor | None = None,
        spatial_tokens: torch.Tensor | None = None,
        spatial_mask: torch.Tensor | None = None,
        spatial_coords: torch.Tensor | None = None,
        force_fp32_bbox: bool = False,
    ) -> dict:
        h = self.trunk(fused)
        bbox_context = (
            torch.autocast("cuda", enabled=False)
            if force_fp32_bbox else contextlib.nullcontext()
        )
        with bbox_context:
            bbox_h = h.float() if force_fp32_bbox else h
            if self.spatial_grounding:
                if spatial_tokens is None or spatial_mask is None:
                    raise ValueError(
                        "spatial grounding requires image tokens and a token mask"
                    )
                bbox_query = bbox_fused if bbox_fused is not None else fused
                attention_tokens = spatial_tokens
                if force_fp32_bbox:
                    bbox_query = bbox_query.float()
                    attention_tokens = attention_tokens.float()
                need_weights = self.bbox_grounding_mode == "coordinate_softargmax"
                if need_weights:
                    if spatial_coords is None:
                        raise ValueError(
                            "coordinate_softargmax requires normalized spatial "
                            "coordinates"
                        )
                    attention_tokens = attention_tokens + self.coordinate_projection(
                        spatial_coords.to(attention_tokens.dtype)
                    )
                grounded, attention_weights = self.grounding_attention(
                    bbox_query.unsqueeze(1),
                    attention_tokens,
                    attention_tokens,
                    key_padding_mask=~spatial_mask.bool(),
                    need_weights=need_weights,
                    average_attn_weights=True,
                )
                bbox_h = self.bbox_trunk(torch.cat([
                    bbox_query, grounded.squeeze(1),
                ], dim=-1))
            attention_entropy = None
            raw_log_wh = None
            if self.bbox_grounding_mode == "coordinate_softargmax":
                centre, attention_entropy = spatial_soft_argmax(
                    attention_weights.squeeze(1),
                    spatial_coords,
                    spatial_mask,
                )
                size_prediction = self.bbox(bbox_h)
                if self.bbox_size_parameterization == "log_space":
                    raw_log_wh = size_prediction
                    size = torch.exp(torch.clamp(
                        raw_log_wh,
                        min=self.bbox_log_size_min,
                        max=self.bbox_log_size_max,
                    ))
                else:
                    size = torch.sigmoid(size_prediction)
                raw_bbox = torch.cat([centre, size], dim=-1)
            else:
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
        if raw_log_wh is not None:
            result["bbox_log_wh"] = raw_log_wh
        if attention_entropy is not None:
            result["bbox_attention_entropy"] = attention_entropy
            result["bbox_attention_weights"] = attention_weights.squeeze(1)
            result["bbox_spatial_coords"] = spatial_coords
            result["bbox_spatial_mask"] = spatial_mask
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
