from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from web_agent.config import load_config
from web_agent.train.selection import (
    ALL_GATES_THEN_OUTCOME_RULE,
    LEGACY_OUTCOME_RULE,
    controlled_quality_gates,
    replay_selection_report,
    require_selected_checkpoint,
)


def test_v2_7_config_changes_only_experiment_control():
    v2_6 = load_config("configs/backbones/qwen2vl_2b_gold_v2_6.yaml")
    v2_7 = load_config("configs/backbones/qwen2vl_2b_gold_v2_7.yaml")

    assert v2_7["train"]["quality_selection_rule"] == ALL_GATES_THEN_OUTCOME_RULE
    assert v2_7["train"]["controlled_experiment_tag"] == "recovery_v2_7"
    assert v2_7["name"] == "Y_QWEN2VL_2B_GOLD_V2_7"
    assert v2_7["model"] == v2_6["model"]
    assert v2_7["loss"] == v2_6["loss"]
    assert v2_7["optim"] == v2_6["optim"]
    assert v2_7["data"] == v2_6["data"]
    assert v2_7["backbone"] == v2_6["backbone"]


def test_v2_7_notebook_is_output_free_and_compiles():
    path = Path("notebooks/kaggle_gold_recovery_v2_7.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "replay_v2_7_selection.py" in source
    assert "all_gates_then_outcome_mcc" in source
    assert "test_rows_read" in source
    assert "kaggle_gold.ipynb" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
            ast.parse("".join(cell["source"]))


def test_roundtrip_reloads_the_explicit_selected_checkpoint():
    source = Path("src/web_agent/train/gold_stages.py").read_text(encoding="utf-8")

    assert "trainer, val_loader, checkpoint" in source
    assert "trainer.load_checkpoint(checkpoint, resume_training=False)" in source
    assert "trainer.load_checkpoint(trainer.best[0][1]" not in source


def _epoch(
    number: int,
    outcome_mcc: float,
    action_acc: float,
    bbox_mean_iou: float,
    bbox_recall_iou50: float,
    *,
    needs_recovery_macro_f1: float = 0.56,
    strategy_attempted_macro_f1: float = 0.47,
    recovery_outcome_mcc: float = 0.70,
    outcome_ece: float = 0.10,
) -> dict:
    return {
        "epoch": number,
        "outcome_mcc": outcome_mcc,
        "action_acc": action_acc,
        "needs_recovery_macro_f1": needs_recovery_macro_f1,
        "needs_recovery_majority_macro_f1": 0.4286,
        "strategy_attempted_macro_f1": strategy_attempted_macro_f1,
        "strategy_attempted_majority_macro_f1": 0.2393,
        "recovery_outcome_mcc": recovery_outcome_mcc,
        "bbox_mean_iou": bbox_mean_iou,
        "bbox_recall_iou50": bbox_recall_iou50,
        "outcome_ece": outcome_ece,
    }


@pytest.fixture
def v2_6_result() -> dict:
    history = [
        _epoch(0, 0.4720684, 0.362, 0.0646997, 0.0320513),
        _epoch(1, 0.5542802, 0.366, 0.00897545, 0.0),
        _epoch(2, 0.5442960, 0.364, 0.0904231, 0.0769231),
        _epoch(3, 0.5423429, 0.414, 0.0949140, 0.0512821),
        _epoch(4, 0.5480665, 0.386, 0.0988959, 0.0576923),
    ]
    return {
        "history": history,
        "best_epochs": {
            "outcome_mcc": {"epoch": 1, "value": 0.5542802, "direction": "max"}
        },
        "epoch_checkpoints": {
            str(epoch): f"checkpoints/epoch-{epoch}.ckpt" for epoch in range(5)
        },
    }


def test_v2_7_selects_epoch_4_from_recorded_v2_6_history(v2_6_result):
    report = controlled_quality_gates(v2_6_result, ALL_GATES_THEN_OUTCOME_RULE)

    assert report["status"] == "PASS"
    assert report["eligible_epochs"] == [3, 4]
    assert report["unconstrained_outcome_epoch"] == 1
    assert report["selected_epoch"] == 4
    assert report["selected_checkpoint"] == "checkpoints/epoch-4.ckpt"
    assert report["selected_epoch_is_eligible"] is True
    assert all(report["checks"].values())
    assert require_selected_checkpoint(v2_6_result, report).endswith("epoch-4.ckpt")


def test_legacy_rule_preserves_v2_6_failure(v2_6_result):
    report = controlled_quality_gates(v2_6_result, LEGACY_OUTCOME_RULE)

    assert report["selected_epoch"] == 1
    assert report["status"] == "FAIL"
    assert report["eligible_epochs"] == [3, 4]
    assert report["checks"]["bbox_mean_iou_at_least_0_05"] is False
    assert report["checks"]["bbox_recall_iou50_at_least_0_01"] is False


def test_constrained_rule_cannot_force_pass_when_no_epoch_is_eligible(v2_6_result):
    result = deepcopy(v2_6_result)
    for row in result["history"]:
        row["bbox_mean_iou"] = 0.0
        row["bbox_recall_iou50"] = 0.0

    report = controlled_quality_gates(result, ALL_GATES_THEN_OUTCOME_RULE)

    assert report["status"] == "FAIL"
    assert report["eligible_epochs"] == []
    assert report["selected_epoch"] == 1
    assert report["selected_epoch_is_eligible"] is False


def test_constrained_rule_uses_earlier_epoch_as_deterministic_tie_break(v2_6_result):
    result = deepcopy(v2_6_result)
    result["history"][3]["outcome_mcc"] = 0.5480665

    report = controlled_quality_gates(result, ALL_GATES_THEN_OUTCOME_RULE)

    assert report["eligible_epochs"] == [3, 4]
    assert report["selected_epoch"] == 3


def test_selected_checkpoint_must_be_retained(v2_6_result):
    result = deepcopy(v2_6_result)
    del result["epoch_checkpoints"]["4"]
    report = controlled_quality_gates(result, ALL_GATES_THEN_OUTCOME_RULE)

    with pytest.raises(ValueError, match="selected epoch 4 has no retained checkpoint"):
        require_selected_checkpoint(result, report)


def test_recorded_primary_epoch_must_match_history(v2_6_result):
    result = deepcopy(v2_6_result)
    result["best_epochs"]["outcome_mcc"]["epoch"] = 0

    with pytest.raises(ValueError, match="disagrees with history"):
        controlled_quality_gates(result, ALL_GATES_THEN_OUTCOME_RULE)


def test_selection_replay_synchronizes_report_without_changing_weights(v2_6_result):
    source = {
        **v2_6_result,
        "status": "PASS",
        "test_rows_read": 0,
        "early_stop_metric": "outcome_mcc",
        "best_metric": 0.5542802,
        "best_checkpoint": "checkpoints/epoch-1.ckpt",
        "checkpoint_roundtrip": True,
        "experiment_control": {"experiment_tag": "recovery_v2_6"},
    }

    replay = replay_selection_report(source)

    assert replay["quality_gates"]["status"] == "PASS"
    assert replay["selected_epoch"] == 4
    assert replay["best_metric"] == pytest.approx(0.5480665)
    assert replay["best_checkpoint"] == "checkpoints/epoch-4.ckpt"
    assert replay["unconstrained_best_metric"] == pytest.approx(0.5542802)
    assert replay["unconstrained_best_checkpoint"] == "checkpoints/epoch-1.ckpt"
    assert replay["experiment_control"]["experiment_tag"] == "recovery_v2_7"
    assert replay["selection_replay"]["model_weights_changed"] is False
    assert replay["selection_replay"]["selected_checkpoint_roundtrip_rerun"] is False
    assert source["best_checkpoint"] == "checkpoints/epoch-1.ckpt"
