"""STEP C — full training entry (MODEL_TRAINING_PLAN §2 STEP C).

Builds dataloaders (full_labels), model, loss, Trainer; runs both phases for
each seed in cfg["seeds"]; saves best checkpoint + appends a row to results CSV.
Called by scripts/run_model.py --stage full and by the package console-script.
"""

from __future__ import annotations


def run(cfg: dict) -> None:
    # TODO(Phase 5): loop seeds -> Trainer.fit -> save best -> log results row.
    raise NotImplementedError("Build in Phase 5 (see docs/IMPLEMENTATION_PLAN.md §5).")


def main() -> None:
    """Console-script entry (web-agent). Parses --config and runs full stage."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
