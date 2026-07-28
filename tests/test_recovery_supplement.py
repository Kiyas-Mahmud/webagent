from __future__ import annotations

import json

import torch

from web_agent.config import load_config
from web_agent.data.gold_dataloader import (
    load_gold_split,
    load_gold_split_sources,
)
from web_agent.data.recovery_supplement import (
    RECOVERY_ONLY_LOSS_MASKS,
    load_supplement_split,
)
from web_agent.data.recovery_transitions import build_recovery_transition_index
from web_agent.labels import (
    NUM_ACTION_TYPE,
    NUM_FAILURE_TYPE,
    NUM_OUTCOME,
    NUM_RECOVERY,
)
from web_agent.models.loss import CombinedLoss


def _row(sample_id: str, strategy: str, succeeded: bool) -> dict:
    return {
        "inputs": {
            "state_before": f"images/{sample_id}_before.png",
            "state_after": f"images/{sample_id}_after.png",
            "task_description": "Recover the browser task.",
            "website_domain": "example.test",
        },
        "labels": {
            "outcome_label": "FAILURE",
            "failure_type_4": "ACTION_MISMATCH",
            "action_type": "PRESS_KEY",
            "action_value": "Escape",
            "action_target_bbox": None,
            "recovery_strategy": strategy,
            "recovery_attempted": True,
            "recovery_success": succeeded,
            "agent_confidence_before": 0.4,
            "memory_update_flag": False,
        },
        "meta": {
            "sample_id": sample_id,
            "task_id": f"task-{sample_id}",
            "trajectory_id": f"trajectory-{sample_id}",
            "session_id": f"session-{sample_id}",
            "step_index": 0,
            "review_status": "approved",
        },
    }


def _write_split(root, split: str, rows: list[dict]) -> None:
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / f"supplement_{split}.json").write_text(
        json.dumps(rows),
        encoding="utf-8",
    )


def test_supplement_loader_annotates_without_mutating_source(tmp_path):
    raw = _row("r1", "RETRY", True)
    _write_split(tmp_path, "train", [raw])

    loaded = load_supplement_split(tmp_path, "train")

    assert "_loss_masks" not in raw["meta"]
    meta = loaded[0]["meta"]
    assert meta["_source_dataset"] == "retry_abort_supplement_v2"
    assert meta["_direct_recovery_transition"] is True
    assert meta["_strategy_source_pre"] is True
    assert meta["_loss_masks"] == RECOVERY_ONLY_LOSS_MASKS
    assert meta["_data_root"] == str(tmp_path.resolve())


def test_direct_transition_uses_same_row_before_action_after(tmp_path):
    _write_split(tmp_path, "train", [_row("a1", "ABORT", True)])
    records = load_supplement_split(tmp_path, "train")

    transitions, report = build_recovery_transition_index(records)

    assert report["proper_transitions"] == 1
    assert report["direct_transitions"] == 1
    transition = transitions["a1"]
    assert transition["failure_state"] == "images/a1_before.png"
    assert transition["executed_recovery_action"] == "PRESS_KEY"
    assert transition["post_recovery_state"] == "images/a1_after.png"
    assert transition["recovery_success"] is True
    assert transition["_data_root"] == str(tmp_path.resolve())


def test_primary_validation_stays_original_only(tmp_path):
    original = tmp_path / "original"
    supplement = tmp_path / "supplement"
    original.mkdir()
    original_row = _row("original", "NONE", False)
    for split in ("train", "val", "test"):
        (original / f"split_{split}.json").write_text(
            json.dumps([original_row]),
            encoding="utf-8",
        )
    _write_split(supplement, "train", [_row("retry", "RETRY", True)])
    _write_split(supplement, "val", [_row("abort", "ABORT", False)])
    cfg = {
        "data": {
            "root": str(original),
            "train_json": "split_train.json",
            "val_json": "split_val.json",
            "test_json": "split_test.json",
            "recovery_supplement": {
                "enabled": True,
                "root": str(supplement),
                "include_in_primary_validation": False,
            },
        },
    }

    assert len(load_gold_split(cfg, "train")) == 2
    assert len(load_gold_split(cfg, "val")) == 1
    assert len(load_gold_split(cfg, "test")) == 1
    assert len(
        load_gold_split_sources(cfg, "val")["retry_abort_supplement_v2"]
    ) == 1


def test_recovery_only_masks_zero_unrelated_losses():
    cfg = {
        "loss": {
            "outcome": 0.22,
            "failure_type": 0.18,
            "action_type": 0.15,
            "bbox": 0.10,
            "memory_flag": 0.10,
            "recovery": 0.05,
            "needs_recovery": 0.04,
            "confidence": 0.05,
            "calibration": 0.05,
            "contrastive": 0.04,
            "recovery_outcome": 0.09,
            "label_smoothing": 0.0,
            "confidence_clip": [0.05, 0.95],
            "contrastive_mode": "supervised",
            "contrastive_temp": 0.1,
            "hierarchical_recovery": True,
            "bbox_loss": "mse",
            "dynamic_weighting": "fixed",
        },
    }
    loss_fn = CombinedLoss(
        cfg,
        action_class_weights=torch.ones(NUM_ACTION_TYPE),
        failtype_class_weights=torch.ones(NUM_FAILURE_TYPE),
        outcome_class_weights=torch.ones(NUM_OUTCOME),
        recovery_class_weights=torch.ones(NUM_RECOVERY),
        recovery_success_pos_weight=torch.ones(1),
        needs_recovery_pos_weight=torch.ones(1),
    )
    batch_size = 2
    preds = {
        "outcome": torch.randn(batch_size, NUM_OUTCOME, requires_grad=True),
        "failure_type": torch.randn(
            batch_size, NUM_FAILURE_TYPE, requires_grad=True
        ),
        "action_type": torch.randn(
            batch_size, NUM_ACTION_TYPE, requires_grad=True
        ),
        "recovery": torch.randn(
            batch_size, NUM_RECOVERY, requires_grad=True
        ),
        "memory_recovery": torch.randn(
            batch_size, NUM_RECOVERY, requires_grad=True
        ),
        "needs_recovery": torch.randn(batch_size, 1, requires_grad=True),
        "bbox": torch.randn(batch_size, 4, requires_grad=True),
        "memory_flag": torch.randn(batch_size, 1, requires_grad=True),
        "recovery_outcome": torch.randn(batch_size, 1, requires_grad=True),
        "confidence": torch.rand(batch_size, 1, requires_grad=True),
        "fused": torch.randn(batch_size, 8, requires_grad=True),
    }
    batch = {
        "label_outcome": torch.tensor([0, 1]),
        "label_failtype": torch.tensor([1, 2]),
        "label_action": torch.tensor([4, 5]),
        "label_recovery": torch.tensor([1, 5]),
        "label_needs_recovery": torch.ones(batch_size, 1),
        "bbox": torch.zeros(batch_size, 4),
        "bbox_mask": torch.zeros(batch_size, 1),
        "label_memory": torch.zeros(batch_size, 1),
        "label_recovery_success": torch.tensor([[1.0], [0.0]]),
        "label_confidence": torch.full((batch_size, 1), 0.5),
        "original_task_id": ["retry", "abort"],
    }
    for name, enabled in RECOVERY_ONLY_LOSS_MASKS.items():
        key = {
            "failure_type": "loss_mask_failtype",
        }.get(name, f"loss_mask_{name}")
        batch[key] = torch.full((batch_size, 1), enabled)

    terms = loss_fn(preds, batch)

    assert terms["recovery"] > 0
    assert terms["recovery_outcome"] > 0
    for name in (
        "outcome",
        "failtype",
        "action",
        "bbox",
        "memory",
        "needs_recovery",
        "confidence",
        "calibration",
        "contrastive",
    ):
        assert terms[name].item() == 0.0
    terms["total"].backward()
    assert preds["recovery"].grad is not None
    assert preds["recovery"].grad.norm() > 0
    assert preds["recovery_outcome"].grad is not None
    assert preds["recovery_outcome"].grad.norm() > 0
    assert preds["outcome"].grad is not None
    assert preds["outcome"].grad.norm() == 0


def test_v2_8_registers_source_aware_supplement():
    cfg = load_config("configs/backbones/qwen2vl_2b_gold_v2_8.yaml")

    assert cfg["train"]["controlled_experiment_tag"] == "recovery_v2_8"
    assert cfg["train"]["quality_selection_rule"] == (
        "all_gates_then_outcome_mcc"
    )
    assert cfg["data"]["recovery_supplement"]["enabled"] is True
    assert (
        cfg["data"]["recovery_supplement"][
            "include_in_primary_validation"
        ]
        is False
    )
