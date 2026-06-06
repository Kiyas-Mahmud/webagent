"""Two-phase trainer (SPEC 6.1 / IMPLEMENTATION_PLAN 5). Reused by every model.

Phase 1: encoders frozen, fusion+heads lr 1e-3, 3 epochs.
Phase 2: all unfrozen, encoder lr 1e-5, fusion/heads lr 1e-4, 7 epochs.
AdamW + cosine warmup, grad clip 1.0, fp16, early stop patience 3 on val_loss,
checkpoint every 500 steps (resume-safe). Logs each loss term + val metrics.
"""

from __future__ import annotations


class Trainer:
    def __init__(self, model, loss_fn, cfg: dict, train_loader, val_loader, logger=None):
        # TODO(Phase 5): optimizer from model.param_groups(), scheduler, scaler, resume.
        raise NotImplementedError("Build in Phase 5 (see docs/IMPLEMENTATION_PLAN.md §5).")

    def fit(self) -> dict:
        """Run both phases; return best metrics. Save best ckpt by val Failure-F1."""
        raise NotImplementedError

    def _run_epoch(self, loader, train: bool):
        raise NotImplementedError
