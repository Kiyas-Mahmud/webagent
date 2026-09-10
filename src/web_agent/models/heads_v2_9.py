"""Recovery v2.9 action head: a zero-initialised sub-cell centre offset.

Why this exists
---------------
``ActionHead`` in ``coordinate_softargmax`` mode places the box centre with a
soft-argmax over the spatial token grid.  At the registered ``max_pixels``
(200,704) a 1280x720 screenshot becomes a 21x12 token grid, so one grid cell
covers roughly 61x60 source pixels.  Measured on the reviewed Gold validation
split, the median target is 160x36 px: horizontally a cell is finer than the
+-80 px the box half-width allows, but vertically a cell is 60 px against a
+-18 px budget -- about 3.3x too coarse.  Raising ``max_pixels`` does not close
this; at 4x the compute the vertical cell is still 30 px.

Soft-argmax interpolates between cell centres, but that interpolation is only
unbiased when the attention mass is symmetric about the true point.  This head
adds an explicit residual correction learned from the query features.

Safety contract
---------------
``centre_offset`` is zero-initialised (weight and bias), and the correction is
``tanh(...) * centre_offset_max``.  ``tanh(0) == 0``, so at initialisation this
head is numerically identical to the v2.8 head it wraps -- see
``tests/test_v2_9.py::test_offset_head_is_identity_at_initialisation``.  That
property is what lets the v2.9 full run use v2.8's epoch-0 bbox metrics as an
abort threshold: the offset can only add signal, never remove it.

This module adds a NEW class.  It never modifies ``heads.py``.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from web_agent.models.bbox import cxcywh_to_bounded_xywh
from web_agent.models.heads import ActionHead

# One vertical grid cell at the registered profile is 1/12 ~= 0.083 of the
# image.  0.10 lets the correction reach just past one cell in either direction,
# which is the whole point, without letting it wander across the page.
DEFAULT_CENTRE_OFFSET_MAX = 0.10


class ActionHeadV29(ActionHead):
    """``ActionHead`` plus a bounded, zero-initialised centre correction.

    Only ``coordinate_softargmax`` + ``cxcywh`` is supported.  Any other
    combination raises rather than silently behaving like the parent, because a
    silent fallback would make the v2.9 run indistinguishable from v2.8.
    """

    def __init__(self, *args, centre_offset_max: float = DEFAULT_CENTRE_OFFSET_MAX, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bbox_grounding_mode != "coordinate_softargmax":
            raise ValueError(
                "ActionHeadV29 requires bbox_grounding_mode='coordinate_softargmax'; "
                f"got {self.bbox_grounding_mode!r}"
            )
        if self.bbox_parameterization != "cxcywh":
            raise ValueError(
                "ActionHeadV29 requires bbox_parameterization='cxcywh'; "
                f"got {self.bbox_parameterization!r}"
            )
        if not 0.0 < float(centre_offset_max) <= 0.5:
            raise ValueError("centre_offset_max must be in (0, 0.5]")
        self.centre_offset_max = float(centre_offset_max)

        dim = self.bbox_trunk[0].in_features // 2  # bbox_trunk consumes [query, grounded]
        self.centre_offset = nn.Linear(dim, 2)
        # Identity at initialisation.  Do not change without updating the abort
        # thresholds in docs/RECOVERY_V2_9_EXPERIMENT.md, which depend on it.
        nn.init.zeros_(self.centre_offset.weight)
        nn.init.zeros_(self.centre_offset.bias)

    @classmethod
    def from_v2_8(
        cls,
        source: ActionHead,
        *,
        centre_offset_max: float = DEFAULT_CENTRE_OFFSET_MAX,
        **kwargs,
    ) -> "ActionHeadV29":
        """Build a v2.9 head carrying over every trained/initialised v2.8 tensor.

        ``build_gold_components`` calls ``initialize_bbox_size_prior`` on the head
        it constructs.  Swapping in a freshly constructed head would discard that
        prior, so the state dict is copied across instead of recomputed.

        Every hyper-parameter is read back off the source instance rather than
        re-derived from config.  ``hidden``, ``dropout`` and especially
        ``bbox_attention_dropout`` carry no parameters, so ``load_state_dict``
        cannot detect a mismatch: ``model.py`` passes
        ``bbox_attention_dropout=0.0`` from the v2.8 config, and reconstructing
        without it would silently give the v2.9 head attention dropout 0.3.
        """
        head = cls(
            source.trunk[0].in_features,
            hidden=source.trunk[0].out_features,
            dropout=source.trunk[2].p,
            bbox_attention_dropout=source.grounding_attention.dropout,
            spatial_grounding=source.spatial_grounding,
            bbox_parameterization=source.bbox_parameterization,
            bbox_grounding_mode=source.bbox_grounding_mode,
            bbox_size_parameterization=source.bbox_size_parameterization,
            bbox_log_size_min=source.bbox_log_size_min,
            bbox_log_size_max=source.bbox_log_size_max,
            centre_offset_max=centre_offset_max,
            **kwargs,
        )
        missing, unexpected = head.load_state_dict(source.state_dict(), strict=False)
        if unexpected:
            raise RuntimeError(f"v2.8 head carried unexpected tensors: {unexpected}")
        if set(missing) != {"centre_offset.weight", "centre_offset.bias"}:
            raise RuntimeError(
                "only the new offset tensors may be missing from the v2.8 head; "
                f"missing={sorted(missing)}"
            )
        return head.to(source.bbox.weight.device)

    def forward(self, fused: torch.Tensor, bbox_fused: torch.Tensor | None = None, **kwargs) -> dict:
        result = super().forward(fused, bbox_fused, **kwargs)
        raw = result.get("bbox_cxcywh")
        if raw is None:  # pragma: no cover - guarded in __init__
            raise RuntimeError("v2.9 head expected a cxcywh box from the parent")

        query = bbox_fused if bbox_fused is not None else fused
        # The parent runs the whole box path outside autocast in fp32 when
        # bbox_fp32_grounding is set. The centre correction is part of that path,
        # so it must use the same precision or it silently reintroduces bf16
        # rounding into a branch the config deliberately keeps in fp32.
        force_fp32 = bool(kwargs.get("force_fp32_bbox", False))
        context = (
            torch.autocast("cuda", enabled=False)
            if force_fp32 else contextlib.nullcontext()
        )
        with context:
            projected = query.float() if force_fp32 else query.to(raw.dtype)
            offset = torch.tanh(self.centre_offset(projected)) * self.centre_offset_max
            centre = (raw[:, :2].to(offset.dtype) + offset).clamp(0.0, 1.0)
            corrected = torch.cat([centre, raw[:, 2:].to(offset.dtype)], dim=-1)
            result["bbox_cxcywh"] = corrected
            result["bbox"] = cxcywh_to_bounded_xywh(corrected)
        result["bbox_centre_offset"] = offset
        return result
