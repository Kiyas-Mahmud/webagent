"""Combined weighted loss — FIXED across all 19 models (SPEC 4.3 / ARCH 7.2).

L_total = 0.25*outcome + 0.20*failtype + 0.18*action + 0.12*bbox(masked)
        + 0.10*memory  + 0.10*recovery + 0.05*confidence

- bbox loss MASKED to rows where bbox exists (bbox_mask).
- action_type CrossEntropy class-weighted (CLICK 83.6%).
- failure_type CrossEntropy class-weighted (LOOP 7.4%).
- outcome/failtype/action/recovery: CrossEntropy ; memory: BCE ; confidence/bbox: MSE.

Weights come from cfg["loss"]; ablations A4/A5 zero out memory/recovery terms.
"""

from __future__ import annotations

import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, cfg: dict, action_class_weights=None, failtype_class_weights=None):
        super().__init__()
        # TODO(Phase 3): store weights + per-term criteria; accept None preds to skip
        #   terms removed by ablations.
        raise NotImplementedError("Build in Phase 3 (see docs/IMPLEMENTATION_PLAN.md §3.3).")

    def forward(self, preds: dict, targets: dict) -> dict:
        """Return {'total': scalar, 'outcome': ..., ...} for logging each term."""
        raise NotImplementedError
