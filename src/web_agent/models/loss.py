"""Combined multitask loss with a backward-compatible recovery-v2 extension.

L_total = 0.22 outcome + 0.18 failure_type + 0.15 action + 0.10 bbox
        + 0.10 memory + 0.09 recovery + 0.05 confidence + 0.05 calibration
        + 0.04 contrastive + cfg.loss.recovery_outcome

The base Qwen config uses 0.05 for recovery outcome; Gold 40K overrides it to
0.09. All coefficients remain config-controlled for registered ablations.

Details:
  - CrossEntropy terms use label_smoothing 0.1; failure_type, action, and recovery
    strategy are class-weighted.
  - recovery = mean of CE over the Failure head and the Memory head (both predict it).
  - legacy bbox = masked MSE; recovery-v2 = masked SmoothL1 + GIoU.
  - memory_flag + recovery_outcome = BCE.
  - confidence = MSE after clipping predictions to [0.05, 0.95].
  - calibration = differentiable soft-binned surrogate of ECE (true ECE is logged as a
    metric, not optimized — hard binning has no gradient).
  - contrastive = margin loss over same-task SUCCESS/FAILURE fused embeddings (pulls
    failures of a task together, pushes failure vs success apart).

Class weights come from cfg-time sklearn balanced weights (passed in); see
web_agent.utils.class_weights.balanced_class_weights.

Recovery-v2 adds binary needs-recovery, masks strategy CE to attempted rows,
uses only proper causal transitions for recovery outcome, and can learn Kendall-
style uncertainty weights for active tasks.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, cfg: dict, action_class_weights=None, failtype_class_weights=None,
                 outcome_class_weights=None, recovery_class_weights=None,
                 recovery_success_pos_weight=None):
        super().__init__()
        self.w = cfg["loss"]
        self.smoothing = self.w.get("label_smoothing", 0.0)
        self.conf_lo, self.conf_hi = self.w.get("confidence_clip", [0.05, 0.95])
        self.margin = self.w.get("contrastive_margin", 0.5)
        # "supervised" = SupCon on the outcome label (works in any batch with both
        # classes present, the default). "task_pair" = the original same-task margin
        # loss (needs a task's SUCCESS+FAILURE in one batch — sparse). See plan P1-A.
        self.contrastive_mode = self.w.get("contrastive_mode", "supervised")
        self.contrastive_temp = self.w.get("contrastive_temp", 0.1)
        self.hierarchical_recovery = bool(self.w.get("hierarchical_recovery", False))
        self.bbox_loss = self.w.get("bbox_loss", "mse")
        self.dynamic_weighting = self.w.get("dynamic_weighting", "fixed")
        self.register_buffer("action_w", action_class_weights, persistent=False)
        self.register_buffer("failtype_w", failtype_class_weights, persistent=False)
        # Balancing outcome prevents the FAILURE-majority collapse.
        self.register_buffer("outcome_w", outcome_class_weights, persistent=False)
        # Gold recovery is dominated by NONE. Sqrt weighting is deliberately
        # gentler than raw inverse frequency and is shared by both recovery heads.
        self.register_buffer("recovery_w", recovery_class_weights, persistent=False)
        # pos_weight for the recovery_success BCE head (else it pins at the prior).
        self.register_buffer("recovery_pos_w", recovery_success_pos_weight, persistent=False)
        # 10 soft-bin centers for the calibration surrogate.
        self.register_buffer("bin_centers", torch.linspace(0.05, 0.95, 10), persistent=False)
        task_names = [
            "outcome", "failtype", "action", "bbox", "memory", "recovery",
            "confidence", "recovery_outcome",
        ]
        if self.hierarchical_recovery:
            task_names.append("needs_recovery")
        self.log_vars = nn.ParameterDict()
        if self.dynamic_weighting == "uncertainty":
            self.log_vars.update({
                name: nn.Parameter(torch.zeros(())) for name in task_names
            })
        elif self.dynamic_weighting != "fixed":
            raise ValueError(f"unsupported dynamic weighting: {self.dynamic_weighting!r}")

    @staticmethod
    def _bbox_giou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Generalized IoU for normalized xywh boxes."""
        pred_xy2 = pred[:, :2] + pred[:, 2:]
        target_xy2 = target[:, :2] + target[:, 2:]
        inter_xy1 = torch.maximum(pred[:, :2], target[:, :2])
        inter_xy2 = torch.minimum(pred_xy2, target_xy2)
        inter_wh = (inter_xy2 - inter_xy1).clamp(min=0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        pred_area = pred[:, 2] * pred[:, 3]
        target_area = target[:, 2] * target[:, 3]
        union = (pred_area + target_area - inter).clamp(min=1e-7)
        iou = inter / union
        enclosing_xy1 = torch.minimum(pred[:, :2], target[:, :2])
        enclosing_xy2 = torch.maximum(pred_xy2, target_xy2)
        enclosing_wh = (enclosing_xy2 - enclosing_xy1).clamp(min=0)
        enclosing = (enclosing_wh[:, 0] * enclosing_wh[:, 1]).clamp(min=1e-7)
        return iou - (enclosing - union) / enclosing

    # ---- calibration: differentiable soft-binned |confidence - accuracy| ----
    def _calibration(self, conf: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
        conf = conf.view(-1)                       # [B]
        correct = correct.view(-1).float()          # [B], detached target
        centers = self.bin_centers.to(conf.device)  # [10]
        width = 0.1
        # triangular soft membership of each sample in each bin
        w = torch.clamp(1.0 - (conf[:, None] - centers[None, :]).abs() / width, min=0.0)  # [B,10]
        denom = w.sum(dim=0).clamp(min=1e-6)        # [10]
        mean_conf = (w * conf[:, None]).sum(dim=0) / denom
        mean_acc = (w * correct[:, None]).sum(dim=0) / denom
        pop = denom / denom.sum()                   # bin weighting by population
        return (pop * (mean_conf - mean_acc).abs()).sum()

    # ---- supervised contrastive (SupCon) on the outcome label ----
    def _supcon(self, fused, outcome_label) -> torch.Tensor:
        """Pull same-outcome embeddings together, push different-outcome apart.
        Needs >=2 samples and both classes present in the batch; else 0 (graph kept).
        """
        y = outcome_label.view(-1)
        if y.numel() < 2 or y.unique().numel() < 2:
            return fused.sum() * 0.0
        z = F.normalize(fused.float(), dim=-1)               # [B, d]
        sim = (z @ z.t()) / self.contrastive_temp            # [B, B]
        B = z.size(0)
        self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(self_mask, float("-inf"))      # drop self-similarity
        pos = (y[:, None] == y[None, :]) & ~self_mask        # same-outcome pairs
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # diagonal -> -inf
        # select positives with where (NOT *pos): -inf * 0 = nan at the self position.
        logp_pos = torch.where(pos, logp, torch.zeros_like(logp))
        pos_count = pos.sum(dim=1)
        valid = pos_count > 0                                 # anchors with >=1 positive
        if not valid.any():
            return fused.sum() * 0.0
        per = logp_pos.sum(dim=1)[valid] / pos_count[valid].clamp(min=1)
        return -per.mean()

    # ---- contrastive over same-task fused embeddings ----
    def _contrastive(self, fused, outcome_label, task_ids) -> torch.Tensor:
        z = F.normalize(fused, dim=-1)              # [B, 768]
        by_task: dict = defaultdict(lambda: {"fail": [], "succ": []})
        for i, t in enumerate(task_ids):
            key = "fail" if int(outcome_label[i]) == 1 else "succ"   # FAILURE=1
            by_task[t][key].append(i)

        losses = []
        for groups in by_task.values():
            f, s = groups["fail"], groups["succ"]
            if not f or not s:
                continue
            anchor = z[f[0]]
            d_neg = torch.norm(anchor - z[s[0]])    # push failure vs success apart
            if len(f) > 1:
                d_pos = torch.norm(anchor - z[f[1]])  # pull failures of same task together
                losses.append(F.relu(self.margin + d_pos - d_neg))
            else:
                losses.append(F.relu(self.margin - d_neg))
        if not losses:
            return fused.sum() * 0.0                 # keep graph; 0 contribution
        return torch.stack(losses).mean()

    def forward(self, preds: dict, batch: dict) -> dict:
        t: dict[str, torch.Tensor] = {}

        # No label smoothing on the binary outcome head: smoothing caps the minority
        # gradient and, with class weights, helps flip the collapse instead of fixing
        # it (plan P1-B). Class weighting carries the imbalance correction here.
        t["outcome"] = F.cross_entropy(
            preds["outcome"], batch["label_outcome"],
            weight=self.outcome_w, label_smoothing=0.0)
        t["failtype"] = F.cross_entropy(
            preds["failure_type"], batch["label_failtype"],
            weight=self.failtype_w, label_smoothing=self.smoothing)
        t["action"] = F.cross_entropy(
            preds["action_type"], batch["label_action"],
            weight=self.action_w, label_smoothing=self.smoothing)

        rec_t = batch["label_recovery"]
        strategy_mask = (
            batch["label_needs_recovery"].view(-1) > 0
            if self.hierarchical_recovery
            else torch.ones_like(rec_t, dtype=torch.bool)
        )
        rec_losses = []
        strategy_active = bool(strategy_mask.any())
        if strategy_active:
            rec_losses.append(F.cross_entropy(
                preds["recovery"][strategy_mask], rec_t[strategy_mask],
                weight=self.recovery_w,
                label_smoothing=self.smoothing,
            ))
            if "memory_recovery" in preds:
                rec_losses.append(F.cross_entropy(
                    preds["memory_recovery"][strategy_mask], rec_t[strategy_mask],
                    weight=self.recovery_w,
                    label_smoothing=self.smoothing,
                ))
        else:
            rec_losses.append(preds["recovery"].sum() * 0.0)
        t["recovery"] = torch.stack(rec_losses).mean()
        if self.hierarchical_recovery:
            t["needs_recovery"] = F.binary_cross_entropy_with_logits(
                preds["needs_recovery"], batch["label_needs_recovery"],
            )

        # Bbox is supervised only where a real target exists.
        mask = batch["bbox_mask"]
        if self.bbox_loss == "smooth_l1_giou":
            row_mask = mask.view(-1) > 0
            bbox_active = bool(row_mask.any())
            if bbox_active:
                smooth = F.smooth_l1_loss(
                    preds["bbox"][row_mask], batch["bbox"][row_mask],
                )
                giou = self._bbox_giou(
                    preds["bbox"][row_mask], batch["bbox"][row_mask],
                )
                t["bbox"] = (
                    float(self.w.get("bbox_smooth_l1_ratio", 1.0)) * smooth
                    + float(self.w.get("bbox_giou_ratio", 1.0)) * (1.0 - giou).mean()
                )
            else:
                t["bbox"] = preds["bbox"].sum() * 0.0
        else:
            bbox_active = bool(mask.sum() > 0)
            sq = (preds["bbox"] - batch["bbox"]) ** 2
            t["bbox"] = (sq * mask).sum() / (mask.sum().clamp(min=1.0) * sq.shape[1])

        # with_logits is autocast(fp16)-safe; heads emit logits for these two.
        t["memory"] = F.binary_cross_entropy_with_logits(
            preds["memory_flag"], batch["label_memory"])
        # recovery_success = -1 marks "no recovery attempted" -> masked out (only the
        # rows where a recovery actually happened supervise this head). Synthetic labels
        # are 0/1, so the mask is all-ones and this equals the plain mean BCE.
        rs = batch["label_recovery_success"]
        if "recovery_row_indices" in preds:
            indices = preds["recovery_row_indices"].to(rs.device)
            recovery_outcome_active = bool(indices.numel())
            if recovery_outcome_active:
                target = rs.index_select(0, indices)
                t["recovery_outcome"] = F.binary_cross_entropy_with_logits(
                    preds["recovery_outcome"], target,
                    pos_weight=self.recovery_pos_w,
                )
            else:
                t["recovery_outcome"] = preds["outcome"].sum() * 0.0
        else:
            ro_mask = (rs >= 0).float()
            recovery_outcome_active = bool(ro_mask.any())
            ro_per = F.binary_cross_entropy_with_logits(
                preds["recovery_outcome"], rs.clamp(min=0.0),
                pos_weight=self.recovery_pos_w, reduction="none")
            t["recovery_outcome"] = (
                (ro_per * ro_mask).sum() / ro_mask.sum().clamp(min=1.0)
            )

        conf = preds["confidence"].clamp(self.conf_lo, self.conf_hi)
        t["confidence"] = F.mse_loss(conf, batch["label_confidence"])

        # agent_confidence_before means confidence that the action will succeed.
        # Calibrate it against the observed SUCCESS target, not against whether the
        # separate outcome head happened to classify the row correctly.
        succeeded = (batch["label_outcome"] == 0).detach()
        t["calibration"] = self._calibration(conf, succeeded)

        if self.contrastive_mode == "supervised":
            t["contrastive"] = self._supcon(preds["fused"], batch["label_outcome"])
        else:
            t["contrastive"] = self._contrastive(
                preds["fused"], batch["label_outcome"], batch["original_task_id"])

        coefficient = {
            "outcome": self.w["outcome"],
            "failtype": self.w["failure_type"],
            "action": self.w["action_type"],
            "bbox": self.w["bbox"],
            "memory": self.w["memory_flag"],
            "recovery": self.w["recovery"],
            "confidence": self.w["confidence"],
            "calibration": self.w["calibration"],
            "contrastive": self.w["contrastive"],
            "recovery_outcome": self.w["recovery_outcome"],
        }
        if self.hierarchical_recovery:
            coefficient["needs_recovery"] = self.w["needs_recovery"]
        weighted = {name: float(coefficient[name]) * t[name] for name in coefficient}
        if self.dynamic_weighting == "uncertainty":
            active_names = set(weighted)
            if not strategy_active:
                active_names.discard("recovery")
            if not bbox_active:
                active_names.discard("bbox")
            if not recovery_outcome_active:
                active_names.discard("recovery_outcome")
            terms = []
            for name, value in weighted.items():
                if name in self.log_vars and name in active_names:
                    terms.append(
                        torch.exp(-self.log_vars[name]) * value + self.log_vars[name]
                    )
                else:
                    terms.append(value)
            t["total"] = sum(terms)
        else:
            t["total"] = sum(weighted.values())
        return t
