from __future__ import annotations

import csv

from web_agent.utils.results import save_mini_result_csv


def test_save_mini_result_csv_writes_one_row_per_epoch(tmp_path):
    report = {
        "status": "PASS",
        "train_rows": 5_000,
        "val_rows": 500,
        "early_stop_metric": "outcome_mcc",
        "best_metric": 0.25,
        "best_checkpoint": "checkpoints/best.ckpt",
        "checkpoint_roundtrip": True,
        "loss_decreased": True,
        "history": [
            {"epoch": 0, "outcome_mcc": 0.10, "train_loss": 2.0},
            {"epoch": 1, "outcome_mcc": 0.25, "train_loss": 1.5},
        ],
    }

    result_path = save_mini_result_csv(report, tmp_path / "mini.csv")

    with result_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["epoch"] == "0"
    assert rows[0]["is_best"] == "False"
    assert rows[1]["is_best"] == "True"
    assert rows[1]["selection_metric"] == "outcome_mcc"
    assert rows[1]["best_checkpoint"] == "checkpoints/best.ckpt"
