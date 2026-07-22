from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import pytest

from web_agent.train.resume import (
    discover_epoch_checkpoints,
    load_epoch_metrics_csv,
    merge_epoch_histories,
    training_signature,
    validate_prior_history,
    validate_resume_checkpoint_metadata,
)


def _config() -> dict:
    return {
        "name": "Y_QWEN2VL_2B_GOLD_V2_7_MINI_RECOVERY_V2_7",
        "backbone": {"vlm_model": "qwen", "min_pixels": 50_176},
        "model": {"spatial_grounding": True},
        "loss": {"bbox_loss": "detr_log_size_giou"},
        "optim": {"batch_size": 4, "grad_accum": 8, "lr_lora": 1e-4},
        "data": {
            "root": "/different/mounts/are/allowed",
            "num_workers": 4,
            "train_json": "split_train.json",
            "causal_routing": True,
        },
        "train": {
            "epochs": 5,
            "early_stop_metric": "outcome_mcc",
            "controlled_experiment_tag": "recovery_v2_7",
            "quality_selection_rule": "all_gates_then_outcome_mcc",
            "metrics_csv": "/kaggle/working/resumed.csv",
        },
        "output": {"checkpoint_dir": "/kaggle/working/checkpoints"},
        "seeds": [42],
    }


def _history(end: int = 3) -> list[dict]:
    return [
        {"epoch": epoch, "outcome_mcc": 0.40 + epoch * 0.05}
        for epoch in range(end + 1)
    ]


def _checkpoint(config: dict | None = None) -> dict:
    return {
        "config": config or _config(),
        "optimizer": {"state": {}},
        "scheduler": {"last_epoch": 628},
        "scaler": {},
        "epoch": 3,
        "step": 628,
        "selection_metric": "outcome_mcc",
        "selection_value": 0.55000001,
        "epoch_complete": True,
    }


def test_load_prior_csv_and_require_consecutive_epochs(tmp_path):
    path = tmp_path / "metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "outcome_mcc"])
        writer.writeheader()
        for row in _history():
            writer.writerow(row)

    rows = load_epoch_metrics_csv(path)

    assert [row["epoch"] for row in rows] == [0, 1, 2, 3]
    assert validate_prior_history(rows, resume_epoch=3, total_epochs=5) == rows


def test_prior_history_rejects_a_gap_or_existing_epoch_four():
    with pytest.raises(ValueError, match="consecutive completed epochs"):
        validate_prior_history(
            [_history()[0], _history()[2]], resume_epoch=3, total_epochs=5
        )
    with pytest.raises(ValueError, match="consecutive completed epochs"):
        validate_prior_history(_history(4), resume_epoch=3, total_epochs=5)


def test_discover_epoch_checkpoints_requires_one_per_epoch(tmp_path):
    for epoch in range(4):
        (tmp_path / f"best_e{epoch}_outcome-mcc0.5.ckpt").touch()

    mapping = discover_epoch_checkpoints(tmp_path, range(4))

    assert sorted(mapping) == ["0", "1", "2", "3"]
    (tmp_path / "best_e3_outcome-mcc0.6.ckpt").touch()
    with pytest.raises(ValueError, match="exactly one epoch-3 checkpoint"):
        discover_epoch_checkpoints(tmp_path, range(4))


def test_checkpoint_metadata_proves_completed_same_config_lineage():
    check = validate_resume_checkpoint_metadata(
        _checkpoint(),
        expected_config=_config(),
        prior_history=_history(),
        resume_epoch=3,
        total_epochs=5,
        steps_per_epoch=157,
    )

    assert check["status"] == "PASS"
    assert check["next_epoch"] == 4
    assert check["optimizer_restored"] is True
    assert check["rng_state_available"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("epoch_complete", False, "not marked as a completed epoch"),
        ("epoch", 2, "expected 3"),
        ("step", 627, "expected 628"),
        ("selection_value", 0.3, "disagrees with the prior CSV"),
    ],
)
def test_checkpoint_metadata_rejects_wrong_lineage(field, value, message):
    checkpoint = _checkpoint()
    checkpoint[field] = value

    with pytest.raises(ValueError, match=message):
        validate_resume_checkpoint_metadata(
            checkpoint,
            expected_config=_config(),
            prior_history=_history(),
            resume_epoch=3,
            total_epochs=5,
            steps_per_epoch=157,
        )


def test_signature_allows_mount_and_output_changes_but_not_training_changes():
    saved = _config()
    current = _config()
    current["data"]["root"] = "/new/kaggle/input"
    current["data"]["num_workers"] = 0
    current["train"]["metrics_csv"] = "/new/output.csv"
    current["output"]["checkpoint_dir"] = "/new/checkpoints"
    assert training_signature(saved) == training_signature(current)

    current["optim"]["lr_lora"] = 2e-4
    checkpoint = _checkpoint(saved)
    with pytest.raises(ValueError, match=r"differing sections=\['optim'\]"):
        validate_resume_checkpoint_metadata(
            checkpoint,
            expected_config=current,
            prior_history=_history(),
            resume_epoch=3,
            total_epochs=5,
            steps_per_epoch=157,
        )


def test_merge_requires_exactly_five_unique_epochs():
    merged = merge_epoch_histories(
        _history(), [{"epoch": 4, "outcome_mcc": 0.60}], total_epochs=5
    )
    assert [row["epoch"] for row in merged] == [0, 1, 2, 3, 4]

    with pytest.raises(ValueError, match="duplicate epoch"):
        merge_epoch_histories(
            _history(), [{"epoch": 3, "outcome_mcc": 0.60}], total_epochs=5
        )


def test_resume_notebook_is_output_free_and_enforces_provenance():
    path = Path("notebooks/kaggle_gold_recovery_v2_7_resume.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "resume_v2_7_mini.py" in source
    assert "gold_mini_recovery_v2_7_metrics.csv" in source
    assert "best_e3_outcome-mcc*.ckpt" in source
    assert "2f4a7e415c6382f0983d2c705d2cd6526d5e766c42319ff40f2adcc71240f9ba" in source
    assert "independent_run_rows_combined" in source
    assert "rng_state_restored" in source
    assert "test_rows_read" in source
    assert "[0, 1, 2, 3, 4]" in source
    assert "kaggle_gold.ipynb" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
            ast.parse("".join(cell["source"]))


def test_resume_script_and_rng_checkpoint_code_compile():
    script = Path("scripts/resume_v2_7_mini.py").read_text(encoding="utf-8")
    trainer = Path("src/web_agent/train/trainer.py").read_text(encoding="utf-8")
    gold_resume = Path("src/web_agent/train/gold_resume.py").read_text(encoding="utf-8")

    ast.parse(script)
    ast.parse(gold_resume)
    assert '"rng_state"' in trainer
    assert "random.setstate" in trainer
    assert "torch.cuda.set_rng_state_all" in trainer
    assert "trainer.start_epoch != resume_epoch + 1" in gold_resume
    assert "independent_run_rows_combined" in gold_resume
