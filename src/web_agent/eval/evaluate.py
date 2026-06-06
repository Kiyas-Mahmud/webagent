"""STEP D — evaluate on the 3 test splits separately (MODEL_TRAINING_PLAN §2 STEP D).

Runs test_task, test_website, test_domain independently (generalization curve),
records all primary + per-failure-type metrics, appends to results/results_table.csv.
Builds/uses the memory index for recovery retrieval accuracy.
"""

from __future__ import annotations


def run(cfg: dict, checkpoint_path: str) -> dict:
    # TODO(Phase 5/10): load ckpt, eval per split, compute metrics, log results row.
    raise NotImplementedError("Build in Phase 5/10 (see docs/IMPLEMENTATION_PLAN.md §10).")
