from __future__ import annotations

import csv
import json

from web_agent.utils.results import save_mini_diagnostics_json, save_mini_result_csv


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


def test_save_mini_diagnostics_json_keeps_non_tabular_evidence(tmp_path):
    report = {
        "status": "PASS",
        "train_rows": 5_000,
        "val_rows": 500,
        "test_rows_read": 0,
        "early_stop_metric": "outcome_mcc",
        "best_metric": 0.4,
        "best_checkpoint": "checkpoints/e1.ckpt",
        "best_epochs": {"recovery_macro_f1": {"epoch": 1, "value": 0.3}},
        "epoch_checkpoints": {"0": "checkpoints/e0.ckpt", "1": "checkpoints/e1.ckpt"},
        "class_weights": {"recovery_strategy": [0.5, 1.5, 0, 0, 0, 0]},
        "sampling": {"duplicate_train_rows_scheduled": 0},
        "train_distribution": {"recovery_attempted": {"True": 100}},
        "validation_distribution": {"recovery_attempted": {"True": 10}},
        "experiment_control": {"baseline": "kaggle-gold-v14"},
        "diagnostics": [
            {
                "epoch": 0,
                "recovery_strategy": {
                    "confusion_matrix": [[1, 0], [0, 1]],
                },
            }
        ],
    }

    path = save_mini_diagnostics_json(report, tmp_path / "diagnostics.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["test_rows_read"] == 0
    assert payload["sampling"]["duplicate_train_rows_scheduled"] == 0
    assert payload["epochs"][0]["recovery_strategy"]["confusion_matrix"] == [
        [1, 0],
        [0, 1],
    ]


def test_save_mini_result_csv_marks_constrained_checkpoint(tmp_path):
    report = {
        "status": "PASS",
        "train_rows": 5_000,
        "val_rows": 500,
        "early_stop_metric": "outcome_mcc",
        "selection_rule": "all_gates_then_outcome_mcc",
        "selected_epoch": 4,
        "best_metric": 0.548,
        "best_checkpoint": "checkpoints/e4.ckpt",
        "unconstrained_best_metric": 0.554,
        "unconstrained_best_checkpoint": "checkpoints/e1.ckpt",
        "checkpoint_roundtrip": True,
        "loss_decreased": True,
        "history": [
            {"epoch": 1, "outcome_mcc": 0.554, "train_loss": 1.5},
            {"epoch": 4, "outcome_mcc": 0.548, "train_loss": 1.1},
        ],
    }

    result_path = save_mini_result_csv(report, tmp_path / "constrained.csv")
    with result_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["is_unconstrained_best"] == "True"
    assert rows[0]["is_selected"] == "False"
    assert rows[1]["is_best"] == "True"
    assert rows[1]["is_selected"] == "True"
    assert rows[1]["best_checkpoint"] == "checkpoints/e4.ckpt"
