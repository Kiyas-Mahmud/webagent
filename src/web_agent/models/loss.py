"""Combined weighted loss — FIXED across all 19 models (SPEC 4.3 / ARCH 7.2).

L_total = 0.25*outcome + 0.20*failtype + 0.18*action + 0.12*bbox(masked)
        + 0.10*memory  + 0.10*recovery + 0.05*confidence

Per-term criteria:
  outcome/failtype/action  : CrossEntropy (failtype + action class-weighted)
  recovery                 : CrossEntropy, averaged over the Failure head and the
                             Memory head (both predict recovery_strategy)
  bbox                     : MSE, MASKED to rows where a bbox exists
  memory_flag              : BCE
  confidence               : MSE

Weights come from cfg["loss"]. Returns a dict with the scalar 'total' plus each
term for logging. Missing prediction keys (e.g. ablations that drop a head) make
that term contribute 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, cfg: dict, action_class_weights=None, failtype_class_weights=None):
        super().__init__()
        self.w = cfg["loss"]
        # Class weights are buffers so .to(device) moves them with the module.
        self.register_buffer("action_w", action_class_weights, persistent=False)
        self.register_buffer("failtype_w", failtype_class_weights, persistent=False)

    def forward(self, preds: dict, batch: dict) -> dict:
        terms: dict[str, torch.Tensor] = {}
        device = preds["outcome"].device

        terms["outcome"] = F.cross_entropy(preds["outcome"], batch["label_outcome"])
        terms["failtype"] = F.cross_entropy(
            preds["failure_type"], batch["label_failtype"], weight=self.failtype_w,
        )
        terms["action"] = F.cross_entropy(
            preds["action_type"], batch["label_action"], weight=self.action_w,
        )

        # recovery: supervise both heads against the same label, then average.
        rec_targets = batch["label_recovery"]
        rec_losses = [F.cross_entropy(preds["recovery"], rec_targets)]
        if "memory_recovery" in preds:
            rec_losses.append(F.cross_entropy(preds["memory_recovery"], rec_targets))
        terms["recovery"] = torch.stack(rec_losses).mean()

        # bbox: masked MSE (only rows that actually have a bbox).
        mask = batch["bbox_mask"]                       # [B,1]
        sq = (preds["bbox"] - batch["bbox"]) ** 2        # [B,4]
        denom = mask.sum().clamp(min=1.0) * sq.shape[1]
        terms["bbox"] = (sq * mask).sum() / denom

        terms["memory"] = F.binary_cross_entropy(preds["memory_flag"], batch["label_memory"])
        terms["confidence"] = F.mse_loss(preds["confidence"], batch["label_confidence"])

        total = (
            self.w["outcome"] * terms["outcome"]
            + self.w["failure_type"] * terms["failtype"]
            + self.w["action_type"] * terms["action"]
            + self.w["bbox"] * terms["bbox"]
            + self.w["memory_flag"] * terms["memory"]
            + self.w["recovery"] * terms["recovery"]
            + self.w["confidence"] * terms["confidence"]
        )
        terms["total"] = total
        return terms
