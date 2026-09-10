"""Recovery v2.9 trainer: warmup, early-stop and loss-sign corrections.

Three defects in the v2.8 training loop, all measured rather than assumed.

1. Warmup spanned an entire epoch.
   ``trainer.py`` builds the schedule as ``warmup = int(steps_per_epoch *
   epochs * warmup_ratio)``.  With 754 steps/epoch, 10 epochs and ratio 0.1
   that is 754 steps -- exactly 1.00 epochs.  Epoch 0 therefore never reached
   peak LR, yet its validation score became ``best_metric`` and the bar every
   later epoch had to clear.  On the 7B v2.8 run epoch 0 (0.6783) is still the
   selected checkpoint.  v2.9 sets warmup in absolute steps.

2. Early stopping could fire off that warmup baseline.
   With ``patience=3`` a run could stop at epoch 3 and ship a warmup-only
   checkpoint.  v2.9 refuses to arm early stopping before ``min_epochs``.

3. A dead config key.
   ``configs/base.yaml`` sets ``optim.early_stopping_patience`` but
   ``trainer.py`` reads ``train.early_stop_patience`` -- different section,
   different name.  The value is never read, yet ``optim`` is inside the resume
   signature, so editing it breaks resume and changes no behaviour.  v2.9 fails
   closed if the dead key is present.

Non-negative loss: v2.9 configs must set ``loss.dynamic_weighting: fixed``, which
makes the total a sum of non-negative weighted terms.  This is checked per epoch
in ``_log_csv`` so a regression is caught at the first epoch boundary rather than
after days of training.

This module adds a NEW class.  It never modifies ``trainer.py``.
"""

from __future__ import annotations

import math

from transformers import get_cosine_schedule_with_warmup

from web_agent.train.trainer import Trainer

DEAD_PATIENCE_KEY = "early_stopping_patience"


class TrainerV29(Trainer):
    """``Trainer`` with step-based warmup and a minimum-epoch early-stop floor."""

    def __init__(self, *args, **kwargs):
        # Set before super().__init__ because Trainer assigns self.bad_epochs,
        # which routes through the property below.
        self.min_epochs = 0
        super().__init__(*args, **kwargs)

        optim_cfg = self.cfg["optim"]
        # Deep-merge cannot delete an inherited key, so the v2.9 config nulls it.
        # A null is the expected, correct state; a real value is the defect.
        if optim_cfg.get(DEAD_PATIENCE_KEY) is not None:
            raise ValueError(
                f"optim.{DEAD_PATIENCE_KEY} is never read by the trainer "
                "(it reads train.early_stop_patience) but it IS part of the resume "
                "signature. Remove it from the v2.9 config and set "
                "train.early_stop_patience instead."
            )

        train_cfg = self.cfg["train"]
        self.min_epochs = int(train_cfg.get("min_epochs", 0))
        if not 0 <= self.min_epochs < self.epochs:
            raise ValueError(
                f"train.min_epochs must be in [0, {self.epochs}); got {self.min_epochs}"
            )

        if self.loss_fn.dynamic_weighting != "fixed":
            raise ValueError(
                "v2.9 requires loss.dynamic_weighting='fixed'. The learned "
                "uncertainty path drives the reported loss unbounded below "
                "(measured: +0.566 -> -17.458 over ten epochs) without doing "
                "per-task weighting."
            )

        warmup_steps = train_cfg.get("warmup_steps")
        if warmup_steps is None:
            raise ValueError(
                "v2.9 requires an absolute train.warmup_steps; the inherited "
                "optim.warmup_ratio spans a whole epoch at this step count"
            )
        self.warmup_steps = int(warmup_steps)
        total = self.steps_per_epoch * self.epochs
        if not 0 <= self.warmup_steps < total:
            raise ValueError(
                f"train.warmup_steps must be in [0, {total}); got {self.warmup_steps}"
            )
        if self.warmup_steps > self.steps_per_epoch // 2:
            raise ValueError(
                "train.warmup_steps exceeds half an epoch "
                f"({self.steps_per_epoch} steps/epoch) -- that is the v2.8 defect"
            )
        # Rebuild the schedule the parent already created with the ratio.
        self.sched = get_cosine_schedule_with_warmup(self.opt, self.warmup_steps, total)
        print(
            f"v2.9 schedule: warmup {self.warmup_steps} steps "
            f"({self.warmup_steps / self.steps_per_epoch:.2f} epochs), total {total}; "
            f"early stopping arms after epoch {self.min_epochs}",
            flush=True,
        )

    @property
    def bad_epochs(self) -> int:
        return self._bad_epochs

    @bad_epochs.setter
    def bad_epochs(self, value) -> None:
        # fit() appends to history before updating this, so len(history) is the
        # number of completed epochs at the moment of assignment.
        completed = len(getattr(self, "history", []))
        floor = getattr(self, "min_epochs", 0)
        self._bad_epochs = 0 if completed < floor else int(value)

    def _log_csv(self, epoch: int, m: dict) -> None:
        loss = float(m.get("train_loss", 0.0))
        if not math.isfinite(loss):
            raise FloatingPointError(f"epoch {epoch} train_loss is not finite: {loss}")
        if loss < 0.0:
            raise FloatingPointError(
                f"epoch {epoch} train_loss is negative ({loss:.4f}). With "
                "dynamic_weighting='fixed' the total is a sum of non-negative "
                "weighted terms, so this means a loss term changed sign."
            )
        super()._log_csv(epoch, m)
