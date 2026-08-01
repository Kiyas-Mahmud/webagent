"""QLoRA Trainer (full-training spec). Reused by every model; the notebook drives it.

- Two AdamW param groups: LoRA+adapter (lr_lora) and the 5 heads (lr_heads).
- Cosine schedule + 10% warmup, fp16/bf16 autocast, grad accumulation, grad clip 1.0.
- Validate every epoch; early-stop on the configured validation metric.
- Rolling resume checkpoints preserve the exact next physical batch in an epoch.
- Checkpoint = model, loss, optimizer/scheduler/scaler, RNG, history and resume state.
- Append all metrics to a CSV every epoch.
"""

from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from web_agent.eval import metrics as M
from web_agent.labels import (
    ACTION_TYPE_INV,
    EXECUTION_OUTCOME_INV,
    FAILURE_TYPE_INV,
    RECOVERY_STRATEGY_INV,
)


BEST_METRIC_DIRECTIONS = {
    "outcome_mcc": "max",
    "failure_macro_f1": "max",
    "failtype_macro_f1": "max",
    "action_macro_f1": "max",
    "recovery_macro_f1": "max",
    "needs_recovery_macro_f1": "max",
    "strategy_attempted_macro_f1": "max",
    "recovery_outcome_mcc": "max",
    "memory_mcc": "max",
    "bbox_mean_iou": "max",
    "outcome_ece": "min",
    "confidence_mae": "min",
    "bbox_mae": "min",
}


def _best_epochs(history: list[dict]) -> dict[str, dict[str, float | int]]:
    """Record the independently best epoch for each paper-facing metric."""
    best: dict[str, dict[str, float | int]] = {}
    for metric, direction in BEST_METRIC_DIRECTIONS.items():
        candidates = [row for row in history if metric in row]
        if not candidates:
            continue
        selected = (max if direction == "max" else min)(
            candidates, key=lambda row: float(row[metric]),
        )
        best[metric] = {
            "epoch": int(selected["epoch"]),
            "value": float(selected[metric]),
            "direction": direction,
        }
    return best


def _to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def collect_predictions(model, loader, device) -> dict:
    """Run the model over a loader, return numpy arrays for every metric."""
    model.eval()
    acc: dict[str, list] = {k: [] for k in (
        "outcome_pred", "outcome_true", "failtype_pred", "failtype_true",
        "action_pred", "action_true", "recovery_pred", "recovery_true",
        "memory_pred", "memory_true", "recovery_outcome_pred", "recovery_outcome_true",
        "outcome_confidence", "outcome_failure_probability",
        "recovery_outcome_probability", "confidence", "confidence_true",
        "bbox_pred", "bbox_true", "bbox_mask",
        "needs_recovery_pred", "needs_recovery_true", "strategy_pred_raw")}
    for batch in loader:
        preds = model(batch)
        b = _to_device(batch, device)
        acc["outcome_pred"] += preds["outcome"].argmax(-1).cpu().tolist()
        acc["outcome_confidence"] += torch.softmax(
            preds["outcome"].float(), dim=-1,
        ).max(dim=-1).values.cpu().tolist()
        acc["outcome_failure_probability"] += torch.softmax(
            preds["outcome"].float(), dim=-1,
        )[:, 1].cpu().tolist()
        acc["outcome_true"] += b["label_outcome"].cpu().tolist()
        acc["failtype_pred"] += preds["failure_type"].argmax(-1).cpu().tolist()
        acc["failtype_true"] += b["label_failtype"].cpu().tolist()
        acc["action_pred"] += preds["action_type"].argmax(-1).cpu().tolist()
        acc["action_true"] += b["label_action"].cpu().tolist()
        if "needs_recovery" in preds:
            needs_pred = (preds["needs_recovery"] > 0).long().view(-1)
            strategy_pred = preds["recovery"][:, 1:].argmax(-1) + 1
            composed_recovery = torch.where(
                needs_pred.bool(), strategy_pred, torch.zeros_like(strategy_pred),
            )
            acc["needs_recovery_pred"] += needs_pred.cpu().tolist()
            acc["needs_recovery_true"] += b["label_needs_recovery"].view(-1).cpu().tolist()
        else:
            strategy_pred = preds["recovery"].argmax(-1)
            composed_recovery = strategy_pred
        acc["strategy_pred_raw"] += strategy_pred.cpu().tolist()
        acc["recovery_pred"] += composed_recovery.cpu().tolist()
        acc["recovery_true"] += b["label_recovery"].cpu().tolist()
        acc["memory_pred"] += (preds["memory_flag"] > 0).long().view(-1).cpu().tolist()
        acc["memory_true"] += b["label_memory"].view(-1).cpu().tolist()
        if "recovery_row_indices" in preds:
            indices = preds["recovery_row_indices"].to(device)
            if indices.numel():
                acc["recovery_outcome_pred"] += (
                    preds["recovery_outcome"] > 0
                ).long().view(-1).cpu().tolist()
                acc["recovery_outcome_probability"] += torch.sigmoid(
                    preds["recovery_outcome"].float(),
                ).view(-1).cpu().tolist()
                acc["recovery_outcome_true"] += b[
                    "label_recovery_success"
                ].index_select(0, indices).view(-1).cpu().tolist()
        else:
            acc["recovery_outcome_pred"] += (
                preds["recovery_outcome"] > 0
            ).long().view(-1).cpu().tolist()
            acc["recovery_outcome_probability"] += torch.sigmoid(
                preds["recovery_outcome"].float(),
            ).view(-1).cpu().tolist()
            acc["recovery_outcome_true"] += b[
                "label_recovery_success"
            ].view(-1).cpu().tolist()
        acc["confidence"] += preds["confidence"].view(-1).cpu().tolist()
        acc["confidence_true"] += b["label_confidence"].view(-1).cpu().tolist()
        acc["bbox_pred"] += preds["bbox"].cpu().tolist()
        acc["bbox_true"] += b["bbox"].cpu().tolist()
        acc["bbox_mask"] += b["bbox_mask"].view(-1).cpu().tolist()
    return {k: np.array(v) for k, v in acc.items()}


def compute_metrics(p: dict) -> dict:
    correct = (p["outcome_pred"] == p["outcome_true"]).astype(float)
    succeeded = (p["outcome_true"] == 0).astype(float)
    recovery_mask = p["recovery_outcome_true"] >= 0
    recovery_true = p["recovery_outcome_true"][recovery_mask]
    recovery_pred = p["recovery_outcome_pred"][recovery_mask]
    recovery_probability = p["recovery_outcome_probability"][recovery_mask]
    bbox = M.bbox_iou_summary(p["bbox_pred"], p["bbox_true"], p["bbox_mask"])
    failtype_majority = M.majority_baseline(p["failtype_true"])
    action_majority = M.majority_baseline(p["action_true"])
    recovery_majority = M.majority_baseline(p["recovery_true"])
    memory_majority = M.majority_baseline(p["memory_true"])
    outcome_majority = M.majority_baseline(p["outcome_true"])
    recovery_outcome_majority = M.majority_baseline(recovery_true)
    outcome_bal_acc = M.outcome_balanced_accuracy(p["outcome_true"], p["outcome_pred"])
    failtype_macro_f1 = M.macro_f1(p["failtype_true"], p["failtype_pred"])
    action_macro_f1 = M.macro_f1(p["action_true"], p["action_pred"])
    recovery_macro_f1 = M.macro_f1(p["recovery_true"], p["recovery_pred"])
    attempted_strategy_mask = p["recovery_true"] != 0
    attempted_strategy_true = p["recovery_true"][attempted_strategy_mask]
    raw_strategy_pred = p.get("strategy_pred_raw", p["recovery_pred"])
    attempted_strategy_pred = raw_strategy_pred[attempted_strategy_mask]
    needs_true = p.get("needs_recovery_true")
    needs_pred = p.get("needs_recovery_pred")
    if needs_true is None or len(needs_true) == 0:
        needs_true = (p["recovery_true"] != 0).astype(int)
        needs_pred = (p["recovery_pred"] != 0).astype(int)
    needs_majority = M.majority_baseline(needs_true)
    attempted_strategy_majority = M.majority_baseline(attempted_strategy_true)
    recovery_outcome_bal_acc = M.balanced_accuracy(recovery_true, recovery_pred)
    memory_bal_acc = M.balanced_accuracy(p["memory_true"], p["memory_pred"])
    return {
        "failure_f1": M.failure_detection_f1(p["outcome_true"], p["outcome_pred"]),
        # honest outcome metrics — these expose majority-class collapse that F1 hides
        "failure_macro_f1": M.failure_macro_f1(p["outcome_true"], p["outcome_pred"]),
        "outcome_bal_acc": outcome_bal_acc,
        "outcome_mcc": M.outcome_mcc(p["outcome_true"], p["outcome_pred"]),
        "success_recall": M.success_recall(p["outcome_true"], p["outcome_pred"]),
        "outcome_brier": M.brier_score(
            p["outcome_true"], p["outcome_failure_probability"],
        ),
        "outcome_majority_acc": outcome_majority["accuracy"],
        "outcome_majority_macro_f1": outcome_majority["macro_f1"],
        "failtype_acc": M.accuracy(p["failtype_true"], p["failtype_pred"]),
        "failtype_macro_f1": failtype_macro_f1,
        "failtype_bal_acc": M.balanced_accuracy(p["failtype_true"], p["failtype_pred"]),
        "failtype_mcc": M.classification_mcc(p["failtype_true"], p["failtype_pred"]),
        "failtype_majority_acc": failtype_majority["accuracy"],
        "failtype_majority_macro_f1": failtype_majority["macro_f1"],
        "action_acc": M.accuracy(p["action_true"], p["action_pred"]),
        "action_macro_f1": action_macro_f1,
        "action_bal_acc": M.balanced_accuracy(p["action_true"], p["action_pred"]),
        "action_mcc": M.classification_mcc(p["action_true"], p["action_pred"]),
        "action_majority_acc": action_majority["accuracy"],
        "action_majority_macro_f1": action_majority["macro_f1"],
        "recovery_acc": M.accuracy(p["recovery_true"], p["recovery_pred"]),
        "recovery_macro_f1": recovery_macro_f1,
        "recovery_bal_acc": M.balanced_accuracy(p["recovery_true"], p["recovery_pred"]),
        "recovery_mcc": M.classification_mcc(p["recovery_true"], p["recovery_pred"]),
        "recovery_majority_acc": recovery_majority["accuracy"],
        "recovery_majority_macro_f1": recovery_majority["macro_f1"],
        "recovery_pred_classes": int(len(set(p["recovery_pred"].tolist()))),
        "needs_recovery_acc": M.accuracy(needs_true, needs_pred),
        "needs_recovery_macro_f1": M.macro_f1(needs_true, needs_pred),
        "needs_recovery_bal_acc": M.balanced_accuracy(needs_true, needs_pred),
        "needs_recovery_mcc": M.classification_mcc(needs_true, needs_pred),
        "needs_recovery_pred_classes": int(len(set(needs_pred.tolist()))),
        "needs_recovery_majority_acc": needs_majority["accuracy"],
        "needs_recovery_majority_macro_f1": needs_majority["macro_f1"],
        "strategy_attempted_acc": M.accuracy(
            attempted_strategy_true, attempted_strategy_pred,
        ),
        "strategy_attempted_macro_f1": M.macro_f1(
            attempted_strategy_true, attempted_strategy_pred,
        ),
        "strategy_attempted_rows": int(len(attempted_strategy_true)),
        "strategy_attempted_pred_classes": int(
            len(set(attempted_strategy_pred.tolist()))
        ),
        "strategy_attempted_majority_acc": attempted_strategy_majority["accuracy"],
        "strategy_attempted_majority_macro_f1": attempted_strategy_majority["macro_f1"],
        "recovery_outcome_acc": M.accuracy(recovery_true, recovery_pred),
        "recovery_outcome_macro_f1": M.macro_f1(recovery_true, recovery_pred),
        "recovery_outcome_bal_acc": recovery_outcome_bal_acc,
        "recovery_outcome_mcc": M.outcome_mcc(recovery_true, recovery_pred),
        "recovery_outcome_brier": M.brier_score(recovery_true, recovery_probability),
        "recovery_outcome_rows": int(len(recovery_true)),
        "recovery_outcome_majority_acc": recovery_outcome_majority["accuracy"],
        "recovery_outcome_majority_macro_f1": recovery_outcome_majority["macro_f1"],
        "memory_acc": M.accuracy(p["memory_true"], p["memory_pred"]),
        "memory_macro_f1": M.macro_f1(p["memory_true"], p["memory_pred"]),
        "memory_bal_acc": memory_bal_acc,
        "memory_mcc": M.classification_mcc(p["memory_true"], p["memory_pred"]),
        "memory_majority_acc": memory_majority["accuracy"],
        "memory_majority_macro_f1": memory_majority["macro_f1"],
        "outcome_ece": M.expected_calibration_error(p["outcome_confidence"], correct),
        "confidence_mae": M.mean_absolute_error(p["confidence_true"], p["confidence"]),
        "confidence_success_ece": M.expected_calibration_error(p["confidence"], succeeded),
        "bbox_mae": M.bbox_mae(p["bbox_pred"], p["bbox_true"], p["bbox_mask"]),
        "bbox_mean_iou": bbox["mean"],
        "bbox_median_iou": bbox["median"],
        "bbox_recall_iou50": bbox["recall_50"],
        "bbox_rows": bbox["rows"],
        "pillar1_diag_score": (outcome_bal_acc + failtype_macro_f1) / 2.0,
        "pillar3_diag_score": (action_macro_f1 + bbox["mean"]) / 2.0,
        "pillar4_diag_score": (
            recovery_macro_f1 + recovery_outcome_bal_acc + memory_bal_acc
        ) / 3.0,
    }


def _bbox_prediction_distribution(p: dict) -> dict:
    """Coordinate summaries on valid bbox rows for detecting frozen/collapsed heads."""
    mask = np.asarray(p["bbox_mask"]).reshape(-1) > 0.5
    names = ("x", "y", "width", "height")
    if not mask.any():
        return {"rows": 0, "prediction": {}, "target": {}}
    pred = np.asarray(p["bbox_pred"], dtype=float)[mask]
    target = np.asarray(p["bbox_true"], dtype=float)[mask]

    def summarize(values):
        return {
            name: {
                "mean": float(values[:, index].mean()),
                "std": float(values[:, index].std()),
                "min": float(values[:, index].min()),
                "max": float(values[:, index].max()),
            }
            for index, name in enumerate(names)
        }

    return {
        "rows": int(mask.sum()),
        "prediction": summarize(pred),
        "target": summarize(target),
    }


def build_diagnostics(p: dict) -> dict:
    """Detailed JSON diagnostics kept outside the compact per-epoch CSV."""
    recovery_mask = p["recovery_outcome_true"] >= 0
    recovery_true = p["recovery_outcome_true"][recovery_mask]
    recovery_pred = p["recovery_outcome_pred"][recovery_mask]
    needs_true = p.get("needs_recovery_true")
    needs_pred = p.get("needs_recovery_pred")
    if needs_true is None or len(needs_true) == 0:
        needs_true = (p["recovery_true"] != 0).astype(int)
        needs_pred = (p["recovery_pred"] != 0).astype(int)
    attempted = p["recovery_true"] != 0
    raw_strategy_pred = p.get("strategy_pred_raw", p["recovery_pred"])
    def label_names(inverse):
        return [inverse[index] for index in sorted(inverse)]
    return {
        "outcome": M.classification_diagnostics(
            p["outcome_true"],
            p["outcome_pred"],
            label_names=label_names(EXECUTION_OUTCOME_INV),
        ),
        "failure_type": M.classification_diagnostics(
            p["failtype_true"],
            p["failtype_pred"],
            label_names=label_names(FAILURE_TYPE_INV),
        ),
        "action_type": M.classification_diagnostics(
            p["action_true"],
            p["action_pred"],
            label_names=label_names(ACTION_TYPE_INV),
        ),
        "recovery_strategy": M.classification_diagnostics(
            p["recovery_true"],
            p["recovery_pred"],
            label_names=label_names(RECOVERY_STRATEGY_INV),
        ),
        "needs_recovery": M.classification_diagnostics(
            needs_true,
            needs_pred,
            label_names=["NO", "YES"],
        ),
        "recovery_strategy_attempted_only": M.classification_diagnostics(
            p["recovery_true"][attempted],
            raw_strategy_pred[attempted],
            label_names=label_names(RECOVERY_STRATEGY_INV),
        ),
        "recovery_outcome_prediction": M.classification_diagnostics(
            recovery_true,
            recovery_pred,
            label_names=["FAILURE", "SUCCESS"],
        ),
        "memory_update": M.classification_diagnostics(
            p["memory_true"],
            p["memory_pred"],
            label_names=["FALSE", "TRUE"],
        ),
        "bbox": M.bbox_iou_summary(
            p["bbox_pred"], p["bbox_true"], p["bbox_mask"],
        ),
        "bbox_prediction_distribution": _bbox_prediction_distribution(p),
        "terminology": {
            "recovery_outcome_prediction": (
                "Offline prediction only on causal transitions containing failure "
                "state, executed recovery action, and post-recovery state."
            ),
            "pillar_scores": (
                "Diagnostic averages only; Pillar 2 requires multimodal ablation and "
                "has no standalone score."
            ),
        },
    }


class Trainer:
    def __init__(self, model, loss_fn, cfg, train_loader, val_loader, train_sampler=None):
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_sampler
        self.device = "cuda"

        o, tr = cfg["optim"], cfg["train"]
        self.accum = o.get("grad_accum", 1)
        self.clip = o.get("grad_clip", 1.0)
        self.epochs = tr.get("epochs", 10)
        self.patience = tr.get("early_stop_patience", 3)
        self.early_stop_metric = tr.get("early_stop_metric", "failure_f1")
        self.ckpt_dir = Path(cfg["output"]["checkpoint_dir"]) / cfg["name"]
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = Path(tr.get("metrics_csv", f"results/{cfg['name']}_metrics.csv"))
        self.keep_top_k = tr.get("keep_top_k", 3)
        self.log_every = tr.get("log_every", 50)
        self.checkpoint_every_steps = int(
            tr.get(
                "checkpoint_every_steps",
                cfg.get("checkpoint_every_steps", 500),
            )
        )
        if self.checkpoint_every_steps <= 0:
            raise ValueError("checkpoint_every_steps must be positive")
        self.steps_per_epoch = math.ceil(len(train_loader) / self.accum)

        precision = str(o.get("mixed_precision", "fp16")).lower()
        if precision not in {"fp16", "bf16"}:
            raise ValueError("optim.mixed_precision must be 'fp16' or 'bf16'")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but this GPU does not support it")
        self.mixed_precision = precision
        self.autocast_dtype = (
            torch.bfloat16 if precision == "bf16" else torch.float16
        )

        self.opt = torch.optim.AdamW([
            {"params": model.lora_parameters(), "lr": o["lr_lora"]},
            {
                "params": model.head_parameters() + list(loss_fn.parameters()),
                "lr": o["lr_heads"],
            },
        ], weight_decay=o.get("weight_decay", 0.01))

        total = self.steps_per_epoch * self.epochs
        warmup = int(total * o.get("warmup_ratio", 0.1))
        from transformers import get_cosine_schedule_with_warmup
        self.sched = get_cosine_schedule_with_warmup(self.opt, warmup, total)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=precision == "fp16"
        )

        self.best: list[tuple[float, Path]] = []   # (metric, path), kept top-k
        self.epoch_checkpoints: dict[int, Path] = {}
        self.global_step = 0
        self.start_epoch = 0
        self.resume_batch_in_epoch = 0
        self.resume_epoch_state: dict | None = None
        self.history: list[dict] = []
        self.diagnostics_history: list[dict] = []
        self.best_metric = float("-inf")
        self.bad_epochs = 0
        self.resumed_from: str | None = None

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.opt.zero_grad(set_to_none=True)
        t0 = time.time()
        done = 0  # optimizer steps completed in this process for this epoch
        probe_parameters = {
            "bbox": self.model.action_head.bbox.weight,
            "strategy": self.model.failure_head.recovery.weight,
            "recovery_outcome": self.model.recovery_outcome_head.recovery_outcome.weight,
        }
        needs_head = getattr(self.model.failure_head, "needs_recovery", None)
        if needs_head is not None:
            probe_parameters["needs_recovery"] = needs_head.weight
        grounding_adapter = (
            self.model.task_adapters["grounding"]
            if "grounding" in self.model.task_adapters else None
        )
        if grounding_adapter is not None:
            probe_parameters["grounding_adapter"] = next(
                grounding_adapter.parameters()
            )
        restored = (
            self.resume_epoch_state
            if self.resume_epoch_state
            and int(self.resume_epoch_state.get("epoch", -1)) == epoch
            else None
        )
        if restored is not None:
            loss_sum = float(restored["loss_sum"])
            term_sums = {
                str(name): float(value)
                for name, value in restored["term_sums"].items()
            }
            batch_count = int(restored["batch_count"])
            probe_before = {
                name: restored["probe_before"][name].to(
                    parameter.device, dtype=parameter.dtype
                )
                for name, parameter in probe_parameters.items()
            }
            first_gradient_norms = {
                str(name): float(value)
                for name, value in restored.get(
                    "first_gradient_norms", {}
                ).items()
            }
            print(
                f"resuming epoch {epoch} at physical batch "
                f"{self.resume_batch_in_epoch}/{len(self.train_loader)}",
                flush=True,
            )
        else:
            loss_sum = 0.0
            term_sums: dict[str, float] = {}
            batch_count = 0
            probe_before = {
                name: parameter.detach().clone()
                for name, parameter in probe_parameters.items()
            }
            first_gradient_norms: dict[str, float] = {}
        remainder = len(self.train_loader) % self.accum
        for i, batch in enumerate(self.train_loader):
            if i < self.resume_batch_in_epoch:
                continue
            denominator = (
                remainder
                if remainder and i >= len(self.train_loader) - remainder
                else self.accum
            )
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                preds = self.model(batch)
                terms = self.loss_fn(preds, _to_device(batch, self.device))
                loss = terms["total"] / denominator
            if not torch.isfinite(terms["total"]):
                raise FloatingPointError(f"non-finite loss at epoch={epoch}, batch={i}")
            loss_sum += float(terms["total"].detach())
            for name, value in terms.items():
                if name != "total":
                    term_sums[name] = term_sums.get(name, 0.0) + float(value.detach())
            batch_count += 1
            self.scaler.scale(loss).backward()
            should_step = (i + 1) % self.accum == 0 or i + 1 == len(self.train_loader)
            if should_step:
                self.scaler.unscale_(self.opt)
                if not first_gradient_norms:
                    first_gradient_norms = {
                        name: (
                            float(parameter.grad.detach().float().norm())
                            if parameter.grad is not None else 0.0
                        )
                        for name, parameter in probe_parameters.items()
                    }
                torch.nn.utils.clip_grad_norm_(
                    self.model.trainable_parameters() + list(self.loss_fn.parameters()),
                    self.clip,
                )
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
                self.sched.step()
                self.global_step += 1
                done += 1
                epoch_step = math.ceil((i + 1) / self.accum)
                if epoch_step % self.log_every == 0:
                    el = time.time() - t0
                    remaining = max(self.steps_per_epoch - epoch_step, 0)
                    eta = el / max(done, 1) * remaining
                    print(f"epoch {epoch} | step {epoch_step}/{self.steps_per_epoch} | "
                          f"loss {terms['total'].item():.3f} | "
                          f"elapsed {el/60:.1f}min | ETA {eta/60:.1f}min", flush=True)
                if self.global_step % self.checkpoint_every_steps == 0:
                    epoch_state = {
                        "epoch": epoch,
                        "next_batch_index": i + 1,
                        "loss_sum": loss_sum,
                        "term_sums": term_sums,
                        "batch_count": batch_count,
                        "probe_before": {
                            name: value.detach().cpu()
                            for name, value in probe_before.items()
                        },
                        "first_gradient_norms": first_gradient_norms,
                    }
                    self._save(
                        self.ckpt_dir / "last.ckpt",
                        epoch,
                        f1=None,
                        batch_in_epoch=i + 1,
                        epoch_state=epoch_state,
                    )
        self.resume_batch_in_epoch = 0
        self.resume_epoch_state = None
        denominator = max(batch_count, 1)
        stats = {"train_loss": loss_sum / denominator}
        loss_weight_keys = {
            "outcome": "outcome",
            "failtype": "failure_type",
            "action": "action_type",
            "bbox": "bbox",
            "memory": "memory_flag",
            "recovery": "recovery",
            "confidence": "confidence",
            "calibration": "calibration",
            "contrastive": "contrastive",
            "recovery_outcome": "recovery_outcome",
            "needs_recovery": "needs_recovery",
        }
        for name, total in term_sums.items():
            raw = total / denominator
            stats[f"train_raw_{name}_loss"] = raw
            if name not in loss_weight_keys:
                # Diagnostic subterms such as bbox L1/GIoU are already included
                # inside a registered task loss and must not be counted twice.
                continue
            weight_key = loss_weight_keys[name]
            stats[f"train_weighted_{name}_loss"] = raw * float(
                self.loss_fn.w[weight_key]
            )
        stats["train_nominal_weighted_loss"] = sum(
            value for key, value in stats.items()
            if key.startswith("train_weighted_") and key.endswith("_loss")
        )
        for name, parameter in probe_parameters.items():
            stats[f"train_grad_{name}_first"] = first_gradient_norms.get(name, 0.0)
            stats[f"train_update_{name}_norm"] = float(
                (parameter.detach() - probe_before[name]).float().norm()
            )
        for name, log_var in self.loss_fn.log_vars.items():
            value = float(log_var.detach())
            stats[f"train_uncertainty_log_var_{name}"] = value
            stats[f"train_uncertainty_multiplier_{name}"] = math.exp(-value)
        return stats

    def fit(self) -> dict:
        for epoch in range(self.start_epoch, self.epochs):
            if self.bad_epochs >= self.patience:
                print(
                    "resume checkpoint already satisfied the early-stop "
                    "condition; no additional epoch will run",
                    flush=True,
                )
                break
            train_stats = self._train_epoch(epoch)
            preds = collect_predictions(self.model, self.val_loader, self.device)
            m = compute_metrics(preds)
            m.update(train_stats)
            self.history.append({"epoch": epoch, **m})
            self.diagnostics_history.append(
                {"epoch": epoch, **build_diagnostics(preds)}
            )
            self._log_csv(epoch, m)
            print(
                f"epoch {epoch}: train_loss={train_stats['train_loss']:.4f}  "
                + "  ".join(
                    f"{k}={v:.4f}"
                    for k, v in m.items()
                    if k != "train_loss" and not k.startswith("train_")
                )
            )

            metric = m[self.early_stop_metric]
            if metric > self.best_metric + 1e-4:
                self.best_metric = metric
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            self._maybe_keep(epoch, metric)
            self._save(
                self.ckpt_dir / "last.ckpt",
                epoch,
                metric,
                epoch_complete=True,
            )
            if self.bad_epochs >= self.patience:
                print(
                    f"early stop at epoch {epoch} "
                    f"(no val {self.early_stop_metric} gain for "
                    f"{self.bad_epochs})"
                )
                break
        return {
            "early_stop_metric": self.early_stop_metric,
            "best_metric": self.best_metric,
            "checkpoints": [str(path) for _, path in self.best],
            "epoch_checkpoints": {
                str(epoch): str(path) for epoch, path in sorted(self.epoch_checkpoints.items())
            },
            "best_epochs": _best_epochs(self.history),
            "history": self.history,
            "diagnostics": self.diagnostics_history,
            "resume": {
                "resumed": self.resumed_from is not None,
                "checkpoint": self.resumed_from,
                "exact_next_batch_restored": self.resumed_from is not None,
            },
        }

    # ---- resumable checkpointing: model + optimizer/scheduler/scaler ----
    def _state(self) -> dict:
        from peft import get_peft_model_state_dict
        numpy_state = np.random.get_state()
        return {
            "lora": get_peft_model_state_dict(self.model.encoder.model),
            "adapter": self.model.adapter.state_dict(),
            "task_adapters": self.model.task_adapters.state_dict(),
            "failure": self.model.failure_head.state_dict(),
            "action": self.model.action_head.state_dict(),
            "memory": self.model.memory_head.state_dict(),
            "recovery_outcome": self.model.recovery_outcome_head.state_dict(),
            "loss": self.loss_fn.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.cfg,
            "history": self.history,
            "diagnostics_history": self.diagnostics_history,
            "best": [
                {"metric": float(metric), "path": str(path)}
                for metric, path in self.best
            ],
            "epoch_checkpoints": {
                str(epoch): str(path)
                for epoch, path in self.epoch_checkpoints.items()
            },
            "early_stop_state": {
                "best_metric": self.best_metric,
                "bad_epochs": self.bad_epochs,
            },
            "mixed_precision": self.mixed_precision,
            "rng_state": {
                "python": random.getstate(),
                "numpy": {
                    "bit_generator": numpy_state[0],
                    "keys": numpy_state[1].tolist(),
                    "position": int(numpy_state[2]),
                    "has_gauss": int(numpy_state[3]),
                    "cached_gaussian": float(numpy_state[4]),
                },
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            },
        }

    def _save(
        self,
        path: Path,
        epoch: int,
        f1,
        epoch_complete: bool = False,
        batch_in_epoch: int = 0,
        epoch_state: dict | None = None,
    ) -> None:
        s = self._state()
        s["epoch"] = epoch
        s["step"] = self.global_step
        s["selection_metric"] = self.early_stop_metric
        s["selection_value"] = f1
        s["epoch_complete"] = epoch_complete
        s["batch_in_epoch"] = batch_in_epoch
        s["epoch_state"] = epoch_state
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(s, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path: str | Path, resume_training: bool = True) -> dict:
        """Restore every trained module; optionally restore optimizer progress too."""
        from peft import set_peft_model_state_dict

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        set_peft_model_state_dict(self.model.encoder.model, checkpoint["lora"])
        self.model.adapter.load_state_dict(checkpoint["adapter"])
        if "task_adapters" in checkpoint:
            self.model.task_adapters.load_state_dict(checkpoint["task_adapters"])
        self.model.failure_head.load_state_dict(checkpoint["failure"])
        self.model.action_head.load_state_dict(checkpoint["action"])
        self.model.memory_head.load_state_dict(checkpoint["memory"])
        self.model.recovery_outcome_head.load_state_dict(checkpoint["recovery_outcome"])
        if "loss" in checkpoint:
            self.loss_fn.load_state_dict(checkpoint["loss"])
        if resume_training:
            self.opt.load_state_dict(checkpoint["optimizer"])
            self.sched.load_state_dict(checkpoint["scheduler"])
            self.scaler.load_state_dict(checkpoint["scaler"])
            self.global_step = int(checkpoint.get("step", 0))
            epoch = int(checkpoint.get("epoch", 0))
            epoch_complete = bool(checkpoint.get("epoch_complete", False))
            self.start_epoch = epoch + int(epoch_complete)
            self.resume_batch_in_epoch = (
                0 if epoch_complete else int(checkpoint.get("batch_in_epoch", 0))
            )
            self.resume_epoch_state = (
                None if epoch_complete else checkpoint.get("epoch_state")
            )
            if self.resume_batch_in_epoch and self.resume_epoch_state is None:
                raise ValueError(
                    "mid-epoch checkpoint lacks the partial epoch accumulator"
                )
            self.history = [dict(row) for row in checkpoint.get("history", [])]
            self.diagnostics_history = [
                dict(row)
                for row in checkpoint.get("diagnostics_history", [])
            ]
            self.best = [
                (
                    float(item["metric"]),
                    self.ckpt_dir / Path(item["path"]).name,
                )
                for item in checkpoint.get("best", [])
            ]
            self.epoch_checkpoints = {
                int(saved_epoch): self.ckpt_dir / Path(saved_path).name
                for saved_epoch, saved_path in checkpoint.get(
                    "epoch_checkpoints", {}
                ).items()
            }
            early_state = checkpoint.get("early_stop_state", {})
            self.best_metric = float(
                early_state.get("best_metric", float("-inf"))
            )
            self.bad_epochs = int(early_state.get("bad_epochs", 0))
            self.resumed_from = str(Path(path).resolve())
            rng_state = checkpoint.get("rng_state")
            if rng_state is not None:
                random.setstate(rng_state["python"])
                numpy_state = rng_state["numpy"]
                np.random.set_state((
                    numpy_state["bit_generator"],
                    np.asarray(numpy_state["keys"], dtype=np.uint32),
                    int(numpy_state["position"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                ))
                torch.set_rng_state(rng_state["torch_cpu"].cpu())
                if torch.cuda.is_available() and rng_state["torch_cuda"]:
                    torch.cuda.set_rng_state_all([
                        state.cpu() for state in rng_state["torch_cuda"]
                    ])
        return checkpoint

    def _maybe_keep(self, epoch: int, metric: float) -> None:
        if len(self.best) < self.keep_top_k or metric > min(x[0] for x in self.best):
            metric_name = self.early_stop_metric.replace("_", "-")
            path = self.ckpt_dir / f"best_e{epoch}_{metric_name}{metric:.3f}.ckpt"
            self.epoch_checkpoints[epoch] = path
            self.best.append((metric, path))
            self.best.sort(key=lambda x: x[0], reverse=True)
            for _, p in self.best[self.keep_top_k:]:
                p.unlink(missing_ok=True)
                for kept_epoch, kept_path in list(self.epoch_checkpoints.items()):
                    if kept_path == p:
                        del self.epoch_checkpoints[kept_epoch]
            self.best = self.best[:self.keep_top_k]
            self._save(path, epoch, metric, epoch_complete=True)

    def _log_csv(self, epoch: int, m: dict) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["epoch"] + list(m.keys()))
            w.writerow([epoch] + [f"{v:.5f}" for v in m.values()])
