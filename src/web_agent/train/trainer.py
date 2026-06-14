"""QLoRA Trainer (full-training spec). Reused by every model; the notebook drives it.

- Two AdamW param groups: LoRA+adapter (lr_lora) and the 5 heads (lr_heads).
- Cosine schedule + 10% warmup, fp16 GradScaler, grad accumulation, grad clip 1.0.
- Validate every epoch; early stop (patience) on val Failure-F1, NOT loss.
- Rolling resume checkpoint every 500 steps; keep top-3 best-by-Failure-F1 at epoch end.
- One state dict = LoRA + adapter + 5 heads (the round-trip rule).
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
        "confidence", "bbox_pred", "bbox_true", "bbox_mask")}
    for batch in loader:
        preds = model(batch)
        b = _to_device(batch, device)
        acc["outcome_pred"] += preds["outcome"].argmax(-1).cpu().tolist()
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
        acc["bbox_pred"] += preds["bbox"].cpu().tolist()
        acc["bbox_true"] += b["bbox"].cpu().tolist()
        acc["bbox_mask"] += b["bbox_mask"].view(-1).cpu().tolist()
    return {k: np.array(v) for k, v in acc.items()}


def compute_metrics(p: dict) -> dict:
    correct = (p["outcome_pred"] == p["outcome_true"]).astype(float)
    return {
        "failure_f1": M.failure_detection_f1(p["outcome_true"], p["outcome_pred"]),
        "failtype_acc": M.accuracy(p["failtype_true"], p["failtype_pred"]),
        "action_acc": M.accuracy(p["action_true"], p["action_pred"]),
        "recovery_acc": M.accuracy(p["recovery_true"], p["recovery_pred"]),
        "recovery_outcome_acc": M.accuracy(p["recovery_outcome_true"], p["recovery_outcome_pred"]),
        "memory_acc": M.accuracy(p["memory_true"], p["memory_pred"]),
        "ece": M.expected_calibration_error(p["confidence"], correct),
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

        self.best: list[tuple[float, Path]] = []   # (f1, path), kept top-k
        self.global_step = 0

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.opt.zero_grad()
        t0 = time.time()
        done = 0  # optimizer steps this epoch
        for i, batch in enumerate(self.train_loader):
            with torch.autocast("cuda", dtype=torch.float16):
                preds = self.model(batch)
                terms = self.loss_fn(preds, _to_device(batch, self.device))
                loss = terms["total"] / self.accum
            self.scaler.scale(loss).backward()
            if (i + 1) % self.accum == 0:
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

    def fit(self) -> dict:
        bad = 0
        best_f1 = -1.0
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            preds = collect_predictions(self.model, self.val_loader, self.device)
            m = compute_metrics(preds)
            self._log_csv(epoch, m)
            print(f"epoch {epoch}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

            f1 = m["failure_f1"]
            self._maybe_keep(epoch, f1)
            if f1 > best_f1 + 1e-4:
                best_f1 = f1; bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    print(f"early stop at epoch {epoch} (no val Failure-F1 gain for {bad})")
                    break
        return {"best_failure_f1": best_f1, "checkpoints": [str(p) for _, p in self.best]}

    # ---- checkpointing (one dict: LoRA + adapter + 5 heads) ----
    def _state(self) -> dict:
        from peft import get_peft_model_state_dict
        return {
            "lora": get_peft_model_state_dict(self.model.encoder.model),
            "adapter": self.model.adapter.state_dict(),
            "failure": self.model.failure_head.state_dict(),
            "action": self.model.action_head.state_dict(),
            "memory": self.model.memory_head.state_dict(),
            "recovery_outcome": self.model.recovery_outcome_head.state_dict(),
            "config": self.cfg,
        }

    def _save(self, path: Path, epoch: int, f1) -> None:
        s = self._state(); s["epoch"] = epoch; s["step"] = self.global_step; s["f1"] = f1
        torch.save(s, path)

    def _maybe_keep(self, epoch: int, f1: float) -> None:
        if len(self.best) < self.keep_top_k or f1 > min(x[0] for x in self.best):
            path = self.ckpt_dir / f"best_e{epoch}_f1{f1:.3f}.ckpt"
            self._save(path, epoch, f1)
            self.best.append((f1, path))
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
