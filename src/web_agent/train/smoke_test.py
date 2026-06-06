"""STEP A — smoke test on 16 rows (MODEL_TRAINING_PLAN §2 STEP A).

Prove: forward runs, all shapes correct (vision/text/fused = [16,768], 7 head
outputs right), combined loss is a finite scalar, loss.backward() works,
optimizer.step() changes one parameter. NO learning expected — just "it runs".
"""

from __future__ import annotations


def run(cfg: dict) -> None:
    # TODO: load 16 rows, build model+loss, one fwd/bwd/step, assert shapes + finite loss.
    raise NotImplementedError("Build alongside Phase 3/4 (see docs/MODEL_TRAINING_PLAN.md §2).")
