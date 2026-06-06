"""STEP B — mini train on 5,000 rows (MODEL_TRAINING_PLAN §2 STEP B).

Prove it LEARNS: 3 epochs frozen encoders, loss drops in first 50 batches,
val Failure-F1 > random (~0.4) on 500 rows, checkpoint saves AND reloads to
identical predictions. If this fails: STOP, fix the bug, do not full-train.
"""

from __future__ import annotations


def run(cfg: dict) -> None:
    # TODO: stratified 5k subsample, short train, assert loss drop + F1 + ckpt round-trip.
    raise NotImplementedError("Build alongside Phase 4 (see docs/MODEL_TRAINING_PLAN.md §2).")
