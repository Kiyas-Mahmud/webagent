"""B1 Random baseline — no training (MODEL_TRAINING_PLAN §5).

Predict majority class per head (FAILURE for outcome, CLICK for action, etc.),
compute metrics directly on the test splits. The comparison floor.
"""

from __future__ import annotations


def run(cfg: dict) -> dict:
    # TODO(Phase 9): majority-class predictions -> metrics -> results row.
    raise NotImplementedError("Build in Phase 9 (see docs/IMPLEMENTATION_PLAN.md §9).")
