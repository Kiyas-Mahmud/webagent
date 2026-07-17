"""QLoRA Trainer (full-training spec). Reused by every model; the notebook drives it.

- Two AdamW param groups: LoRA+adapter (lr_lora) and the 5 heads (lr_heads).
- Cosine schedule + 10% warmup, fp16 GradScaler, grad accumulation, grad clip 1.0.
- Validate every epoch; early-stop on the configured validation metric.
- Rolling resume checkpoint every 500 steps; keep top-k epoch checkpoints.
- Checkpoint = LoRA + adapter + 5 heads + optimizer/scheduler/scaler state.
- Append all metrics to a CSV every epoch.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from web_agent.eval import metrics as M


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
        "outcome_confidence", "confidence", "confidence_true",
        "bbox_pred", "bbox_true", "bbox_mask")}
    for batch in loader:
        preds = model(batch)
        b = _to_device(batch, device)
        acc["outcome_pred"] += preds["outcome"].argmax(-1).cpu().tolist()
        acc["outcome_confidence"] += torch.softmax(
            preds["outcome"].float(), dim=-1,
        ).max(dim=-1).values.cpu().tolist()
        acc["outcome_true"] += b["label_outcome"].cpu().tolist()
        acc["failtype_pred"] += preds["failure_type"].argmax(-1).cpu().tolist()
        acc["failtype_true"] += b["label_failtype"].cpu().tolist()
        acc["action_pred"] += preds["action_type"].argmax(-1).cpu().tolist()
        acc["action_true"] += b["label_action"].cpu().tolist()
        acc["recovery_pred"] += preds["recovery"].argmax(-1).cpu().tolist()
        acc["recovery_true"] += b["label_recovery"].cpu().tolist()
        acc["memory_pred"] += (preds["memory_flag"] > 0).long().view(-1).cpu().tolist()
        acc["memory_true"] += b["label_memory"].view(-1).cpu().tolist()
        acc["recovery_outcome_pred"] += (preds["recovery_outcome"] > 0).long().view(-1).cpu().tolist()
        acc["recovery_outcome_true"] += b["label_recovery_success"].view(-1).cpu().tolist()
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
    return {
        "failure_f1": M.failure_detection_f1(p["outcome_true"], p["outcome_pred"]),
        # honest outcome metrics — these expose majority-class collapse that F1 hides
        "failure_macro_f1": M.failure_macro_f1(p["outcome_true"], p["outcome_pred"]),
        "outcome_bal_acc": M.outcome_balanced_accuracy(p["outcome_true"], p["outcome_pred"]),
        "outcome_mcc": M.outcome_mcc(p["outcome_true"], p["outcome_pred"]),
        "success_recall": M.success_recall(p["outcome_true"], p["outcome_pred"]),
        "failtype_acc": M.accuracy(p["failtype_true"], p["failtype_pred"]),
        "failtype_macro_f1": M.macro_f1(p["failtype_true"], p["failtype_pred"]),
        "action_acc": M.accuracy(p["action_true"], p["action_pred"]),
        "action_macro_f1": M.macro_f1(p["action_true"], p["action_pred"]),
        "recovery_acc": M.accuracy(p["recovery_true"], p["recovery_pred"]),
        "recovery_outcome_acc": M.accuracy(recovery_true, recovery_pred),
        "recovery_outcome_mcc": M.outcome_mcc(recovery_true, recovery_pred),
        "memory_acc": M.accuracy(p["memory_true"], p["memory_pred"]),
        "outcome_ece": M.expected_calibration_error(p["outcome_confidence"], correct),
        "confidence_mae": M.mean_absolute_error(p["confidence_true"], p["confidence"]),
        "confidence_success_ece": M.expected_calibration_error(p["confidence"], succeeded),
        "bbox_mae": M.bbox_mae(p["bbox_pred"], p["bbox_true"], p["bbox_mask"]),
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
        self.steps_per_epoch = math.ceil(len(train_loader) / self.accum)

        self.opt = torch.optim.AdamW([
            {"params": model.lora_parameters(), "lr": o["lr_lora"]},
            {"params": model.head_parameters(), "lr": o["lr_heads"]},
        ], weight_decay=o.get("weight_decay", 0.01))

        total = self.steps_per_epoch * self.epochs
        warmup = int(total * o.get("warmup_ratio", 0.1))
        from transformers import get_cosine_schedule_with_warmup
        self.sched = get_cosine_schedule_with_warmup(self.opt, warmup, total)
        self.scaler = torch.amp.GradScaler("cuda")

        self.best: list[tuple[float, Path]] = []   # (metric, path), kept top-k
        self.global_step = 0
        self.start_epoch = 0

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.opt.zero_grad()
        t0 = time.time()
        done = 0  # optimizer steps this epoch
        loss_sum = 0.0
        batch_count = 0
        remainder = len(self.train_loader) % self.accum
        for i, batch in enumerate(self.train_loader):
            denominator = (
                remainder
                if remainder and i >= len(self.train_loader) - remainder
                else self.accum
            )
            with torch.autocast("cuda", dtype=torch.float16):
                preds = self.model(batch)
                terms = self.loss_fn(preds, _to_device(batch, self.device))
                loss = terms["total"] / denominator
            if not torch.isfinite(terms["total"]):
                raise FloatingPointError(f"non-finite loss at epoch={epoch}, batch={i}")
            loss_sum += float(terms["total"].detach())
            batch_count += 1
            self.scaler.scale(loss).backward()
            should_step = (i + 1) % self.accum == 0 or i + 1 == len(self.train_loader)
            if should_step:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), self.clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()
                self.sched.step()
                self.global_step += 1
                done += 1
                if done % self.log_every == 0:
                    el = time.time() - t0
                    eta = el / done * (self.steps_per_epoch - done)
                    print(f"epoch {epoch} | step {done}/{self.steps_per_epoch} | "
                          f"loss {terms['total'].item():.3f} | "
                          f"elapsed {el/60:.1f}min | ETA {eta/60:.1f}min", flush=True)
                if self.global_step % 500 == 0:
                    self._save(self.ckpt_dir / "last.ckpt", epoch, f1=None)
        return loss_sum / max(batch_count, 1)

    def fit(self) -> dict:
        bad = 0
        best_metric = -1.0
        history = []
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self._train_epoch(epoch)
            preds = collect_predictions(self.model, self.val_loader, self.device)
            m = compute_metrics(preds)
            m["train_loss"] = train_loss
            history.append({"epoch": epoch, **m})
            self._log_csv(epoch, m)
            print(
                f"epoch {epoch}: train_loss={train_loss:.4f}  "
                + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if k != "train_loss")
            )

            metric = m[self.early_stop_metric]
            self._maybe_keep(epoch, metric)
            if metric > best_metric + 1e-4:
                best_metric = metric; bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    print(
                        f"early stop at epoch {epoch} "
                        f"(no val {self.early_stop_metric} gain for {bad})"
                    )
                    break
        return {
            "early_stop_metric": self.early_stop_metric,
            "best_metric": best_metric,
            "checkpoints": [str(path) for _, path in self.best],
            "history": history,
        }

    # ---- resumable checkpointing: model + optimizer/scheduler/scaler ----
    def _state(self) -> dict:
        from peft import get_peft_model_state_dict
        return {
            "lora": get_peft_model_state_dict(self.model.encoder.model),
            "adapter": self.model.adapter.state_dict(),
            "failure": self.model.failure_head.state_dict(),
            "action": self.model.action_head.state_dict(),
            "memory": self.model.memory_head.state_dict(),
            "recovery_outcome": self.model.recovery_outcome_head.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.cfg,
        }

    def _save(self, path: Path, epoch: int, f1, epoch_complete: bool = False) -> None:
        s = self._state()
        s["epoch"] = epoch
        s["step"] = self.global_step
        s["selection_metric"] = self.early_stop_metric
        s["selection_value"] = f1
        s["epoch_complete"] = epoch_complete
        torch.save(s, path)

    def load_checkpoint(self, path: str | Path, resume_training: bool = True) -> dict:
        """Restore every trained module; optionally restore optimizer progress too."""
        from peft import set_peft_model_state_dict

        checkpoint = torch.load(path, map_location=self.device)
        set_peft_model_state_dict(self.model.encoder.model, checkpoint["lora"])
        self.model.adapter.load_state_dict(checkpoint["adapter"])
        self.model.failure_head.load_state_dict(checkpoint["failure"])
        self.model.action_head.load_state_dict(checkpoint["action"])
        self.model.memory_head.load_state_dict(checkpoint["memory"])
        self.model.recovery_outcome_head.load_state_dict(checkpoint["recovery_outcome"])
        if resume_training:
            self.opt.load_state_dict(checkpoint["optimizer"])
            self.sched.load_state_dict(checkpoint["scheduler"])
            self.scaler.load_state_dict(checkpoint["scaler"])
            self.global_step = int(checkpoint.get("step", 0))
            epoch = int(checkpoint.get("epoch", 0))
            self.start_epoch = epoch + int(bool(checkpoint.get("epoch_complete", False)))
        return checkpoint

    def _maybe_keep(self, epoch: int, metric: float) -> None:
        if len(self.best) < self.keep_top_k or metric > min(x[0] for x in self.best):
            metric_name = self.early_stop_metric.replace("_", "-")
            path = self.ckpt_dir / f"best_e{epoch}_{metric_name}{metric:.3f}.ckpt"
            self._save(path, epoch, metric, epoch_complete=True)
            self.best.append((metric, path))
            self.best.sort(key=lambda x: x[0], reverse=True)
            for _, p in self.best[self.keep_top_k:]:
                p.unlink(missing_ok=True)
            self.best = self.best[:self.keep_top_k]

    def _log_csv(self, epoch: int, m: dict) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["epoch"] + list(m.keys()))
            w.writerow([epoch] + [f"{v:.5f}" for v in m.values()])
