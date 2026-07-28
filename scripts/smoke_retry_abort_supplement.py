#!/usr/bin/env python3
"""Run one T4 forward/backward pass on recovery-only supplement rows."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from web_agent.config import load_config
from web_agent.data.gold_dataloader import (
    build_gold_dataloader,
    gold_recovery_class_weights,
)
from web_agent.data.gold_dataset import view
from web_agent.data.recovery_supplement import (
    load_supplement_split,
    validate_supplement,
)
from web_agent.labels import (
    NUM_ACTION_TYPE,
    NUM_FAILURE_TYPE,
    NUM_OUTCOME,
)
from web_agent.models.loss import CombinedLoss
from web_agent.models.model import WebAgentModel
from web_agent.train.gold_stages import build_processor
from web_agent.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        default="configs/backbones/qwen2vl_2b_gold_v2_8.yaml",
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _select_rows(records: list[dict]) -> list[dict]:
    """Select one row for every strategy/outcome pair."""
    selected: dict[tuple[str, bool], dict] = {}
    for record in records:
        _, labels, _ = view(record)
        key = (
            str(labels["recovery_strategy"]),
            bool(labels["recovery_success"]),
        )
        if key not in selected:
            selected[key] = record
    expected = {
        ("RETRY", False),
        ("RETRY", True),
        ("ABORT", False),
        ("ABORT", True),
    }
    if set(selected) != expected:
        raise AssertionError(
            f"supplement sanity lacks strategy/outcome support: {set(selected)}"
        )
    # The first two form the physical batch and cover both heads/classes.
    return [
        selected[("RETRY", True)],
        selected[("ABORT", False)],
        selected[("RETRY", False)],
        selected[("ABORT", True)],
    ]


def _gradient_norm(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm().cpu())


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise AssertionError("Enable a Kaggle Tesla T4 GPU before sanity testing")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise AssertionError(
            f"Tesla T4 required for this controlled sanity run; found {gpu_name}"
        )

    package_report = validate_supplement(args.supplement_root)
    if package_report["status"] != "PASS":
        raise AssertionError(
            f"supplement validation failed: {package_report['errors']}"
        )

    set_seed(42)
    cfg = load_config(args.config)
    cfg["data"]["root"] = str(args.supplement_root.resolve())
    cfg["data"]["recovery_supplement"]["root"] = str(
        args.supplement_root.resolve()
    )
    cfg["data"]["num_workers"] = 0
    cfg["optim"]["batch_size"] = 2
    # The sanity pass proves routing and gradients; it does not benchmark bbox.
    cfg["backbone"]["min_pixels"] = 50_176
    cfg["backbone"]["max_pixels"] = 200_704

    all_rows = load_supplement_split(args.supplement_root, "train")
    selected_rows = _select_rows(all_rows)
    transition_rows = selected_rows[:2]
    processor = build_processor(cfg)
    loader = build_gold_dataloader(
        cfg,
        "train",
        processor,
        records=transition_rows,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        trajectory_records=transition_rows,
    )
    batch = next(iter(loader))
    if batch["source_dataset"] != [
        "retry_abort_supplement_v2",
        "retry_abort_supplement_v2",
    ]:
        raise AssertionError("source identity was not preserved by collation")
    if not bool(batch["strategy_source_pre"].all()):
        raise AssertionError("strategy must be routed from the failure-state stream")
    if "recovery_row_indices" not in batch:
        raise AssertionError("direct recovery transitions were not constructed")
    if batch["recovery_row_indices"].tolist() != [0, 1]:
        raise AssertionError("both supplement rows must supervise recovery success")

    model = WebAgentModel(cfg)
    for module in (
        model.adapter,
        model.task_adapters,
        model.failure_head,
        model.action_head,
        model.memory_head,
        model.recovery_outcome_head,
    ):
        module.to("cuda")

    recovery_weights = gold_recovery_class_weights(
        all_rows,
        cap=float(cfg["loss"].get("recovery_weight_cap", 3.0)),
        attempted_only=True,
    )
    labels = [view(record)[1] for record in all_rows]
    positives = sum(label["recovery_success"] is True for label in labels)
    negatives = sum(label["recovery_success"] is False for label in labels)
    recovery_pos_weight = torch.tensor([
        negatives / max(positives, 1)
    ], dtype=torch.float32)
    loss_fn = CombinedLoss(
        cfg,
        action_class_weights=torch.ones(NUM_ACTION_TYPE),
        failtype_class_weights=torch.ones(NUM_FAILURE_TYPE),
        outcome_class_weights=torch.ones(NUM_OUTCOME),
        recovery_class_weights=recovery_weights,
        recovery_success_pos_weight=recovery_pos_weight,
        needs_recovery_pos_weight=torch.ones(1),
    ).to("cuda")

    device_batch = {
        key: value.to("cuda") if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    model.train()
    model.zero_grad(set_to_none=True)
    loss_fn.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        predictions = model(batch)
        terms = loss_fn(predictions, device_batch)
    if not torch.isfinite(terms["total"]):
        raise FloatingPointError("supplement sanity loss is not finite")
    terms["total"].backward()

    active_losses = {
        "recovery": float(terms["recovery"].detach().cpu()),
        "recovery_outcome": float(
            terms["recovery_outcome"].detach().cpu()
        ),
    }
    masked_names = (
        "outcome",
        "failtype",
        "action",
        "bbox",
        "memory",
        "needs_recovery",
        "confidence",
        "calibration",
        "contrastive",
    )
    masked_losses = {
        name: float(terms[name].detach().cpu())
        for name in masked_names
    }
    if not all(value > 0.0 for value in active_losses.values()):
        raise AssertionError(f"recovery losses are not active: {active_losses}")
    if any(abs(value) > 1e-8 for value in masked_losses.values()):
        raise AssertionError(
            f"unrelated losses received supplement supervision: {masked_losses}"
        )

    active_gradients = {
        "failure_recovery": _gradient_norm(
            model.failure_head.recovery.weight
        ),
        "memory_recovery": _gradient_norm(
            model.memory_head.recovery.weight
        ),
        "recovery_outcome": _gradient_norm(
            model.recovery_outcome_head.recovery_outcome.weight
        ),
    }
    unrelated_gradients = {
        "outcome": _gradient_norm(model.failure_head.outcome.weight),
        "failure_type": _gradient_norm(
            model.failure_head.failure_type.weight
        ),
        "action_type": _gradient_norm(model.action_head.action_type.weight),
        "bbox": _gradient_norm(model.action_head.bbox.weight),
        "memory_flag": _gradient_norm(model.memory_head.memory_flag.weight),
    }
    if not all(value > 0.0 for value in active_gradients.values()):
        raise AssertionError(
            f"recovery heads have missing gradients: {active_gradients}"
        )
    if any(value > 1e-8 for value in unrelated_gradients.values()):
        raise AssertionError(
            f"unrelated output heads received gradients: {unrelated_gradients}"
        )

    report = {
        "status": "PASS",
        "schema": "retry-abort-supplement-v2-forward-backward-sanity",
        "gpu": gpu_name,
        "config": args.config,
        "test_rows_read": 0,
        "train_rows_available": len(all_rows),
        "sanity_rows_selected": len(selected_rows),
        "physical_batch_rows": len(transition_rows),
        "selected_support": {
            "strategy": dict(Counter(
                view(record)[1]["recovery_strategy"]
                for record in selected_rows
            )),
            "recovery_success": dict(Counter(
                str(bool(view(record)[1]["recovery_success"]))
                for record in selected_rows
            )),
        },
        "direct_recovery_rows": len(batch["recovery_row_indices"]),
        "strategy_input_stream": "state_before_failure_stream",
        "recovery_outcome_input_stream": (
            "failure_state_plus_executed_action_plus_post_recovery_state"
        ),
        "active_losses": active_losses,
        "masked_losses": masked_losses,
        "active_gradient_norms": active_gradients,
        "unrelated_output_gradient_norms": unrelated_gradients,
        "permitted_next_action": (
            "SHORT_CONTROLLED_MINI_WITH_SEPARATE_SOURCE_VALIDATION"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
