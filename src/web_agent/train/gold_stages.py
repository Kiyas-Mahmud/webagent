"""Reusable smoke and mini stages for the structured Gold 40K dataset.

The notebook should orchestrate experiments, not reimplement training logic. This
module owns component construction, causal smoke checks, validation-only mini
training, and checkpoint round-trip verification.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import random

import torch

from web_agent.data.gold_dataloader import (
    build_gold_dataloader,
    gold_class_weights,
    gold_recovery_class_weights,
    load_gold_split,
    select_smoke_records,
    stratified_gold_subsample,
)
from web_agent.data.bbox_audit import (
    audit_bbox_geometry,
    require_valid_bbox_geometry,
)
from web_agent.data.gold_dataset import view
from web_agent.data.recovery_transitions import (
    build_recovery_transition_index,
    recovery_class_audit,
)
from web_agent.data.gold_sampling import (
    label_distribution,
    recovery_aware_batch_indices,
    select_recovery_aware_gold_subset,
)
from web_agent.models.loss import CombinedLoss
from web_agent.models.bbox import detr_bbox_loss_terms
from web_agent.models.model import WebAgentModel
from web_agent.eval.metrics import bbox_iou_summary, bbox_mae
from web_agent.train.trainer import (
    Trainer,
    build_diagnostics,
    collect_predictions,
    compute_metrics,
)


@dataclass
class GoldComponents:
    processor: object
    model: WebAgentModel
    loss_fn: CombinedLoss
    train_records: list[dict]
    class_weight_report: dict
    device: str = "cuda"


V14_SELECTED_BASELINE = {
    "epoch": 3,
    "outcome_mcc": 0.5216751841203641,
    "action_acc": 0.396,
    "recovery_outcome_mcc": 0.18641092980036,
    "bbox_mae": 0.1862636590092959,
    "outcome_ece": 0.18528690659999844,
}


def _controlled_quality_gates(result: dict) -> dict:
    """Predeclared v14 comparison at the outcome-selected checkpoint."""
    selected_epoch = int(result["best_epochs"]["outcome_mcc"]["epoch"])
    selected = next(row for row in result["history"] if row["epoch"] == selected_epoch)
    if "needs_recovery_macro_f1" in selected:
        checks = {
            "outcome_mcc_within_0_03_of_v14": (
                selected["outcome_mcc"] >= V14_SELECTED_BASELINE["outcome_mcc"] - 0.03
            ),
            "action_accuracy_not_down_more_than_0_03": (
                selected["action_acc"] >= V14_SELECTED_BASELINE["action_acc"] - 0.03
            ),
            "needs_recovery_macro_f1_beats_majority_by_0_03": (
                selected["needs_recovery_macro_f1"]
                >= selected["needs_recovery_majority_macro_f1"] + 0.03
            ),
            "attempted_strategy_macro_f1_beats_majority_by_0_03": (
                selected["strategy_attempted_macro_f1"]
                >= selected["strategy_attempted_majority_macro_f1"] + 0.03
            ),
            "transition_recovery_outcome_mcc_improves_v14_by_0_01": (
                selected["recovery_outcome_mcc"]
                >= V14_SELECTED_BASELINE["recovery_outcome_mcc"] + 0.01
            ),
            "bbox_mean_iou_at_least_0_05": selected["bbox_mean_iou"] >= 0.05,
            "bbox_recall_iou50_at_least_0_01": (
                selected["bbox_recall_iou50"] >= 0.01
            ),
            "outcome_ece_not_up_more_than_0_03": (
                selected["outcome_ece"]
                <= V14_SELECTED_BASELINE["outcome_ece"] + 0.03
            ),
        }
        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "selected_epoch": selected_epoch,
            "comparison_rule": "all gates use the outcome_mcc-selected checkpoint",
            "v14_selected_checkpoint_reference": V14_SELECTED_BASELINE,
            "checks": checks,
            "limitations": [
                "recovery-outcome semantics changed to proper causal transitions",
                "bbox IoU thresholds are minimum functionality gates, not 90-percent claims",
                "full training remains blocked until review and class-audit gates pass",
            ],
        }
    checks = {
        "outcome_mcc_within_0_02_of_v14": (
            selected["outcome_mcc"] >= V14_SELECTED_BASELINE["outcome_mcc"] - 0.02
        ),
        "recovery_macro_f1_beats_majority_by_0_03": (
            selected["recovery_macro_f1"]
            >= selected["recovery_majority_macro_f1"] + 0.03
        ),
        "recovery_predicts_at_least_3_classes": selected["recovery_pred_classes"] >= 3,
        "recovery_outcome_mcc_improves_v14_by_0_01": (
            selected["recovery_outcome_mcc"]
            >= V14_SELECTED_BASELINE["recovery_outcome_mcc"] + 0.01
        ),
        "action_accuracy_not_down_more_than_0_02": (
            selected["action_acc"] >= V14_SELECTED_BASELINE["action_acc"] - 0.02
        ),
        "bbox_mae_not_up_more_than_0_02": (
            selected["bbox_mae"] <= V14_SELECTED_BASELINE["bbox_mae"] + 0.02
        ),
        "outcome_ece_not_up_more_than_0_02": (
            selected["outcome_ece"] <= V14_SELECTED_BASELINE["outcome_ece"] + 0.02
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "selected_epoch": selected_epoch,
        "comparison_rule": "all gates use the outcome_mcc-selected checkpoint",
        "v14_selected_checkpoint_reference": V14_SELECTED_BASELINE,
        "checks": checks,
        "limitations": [
            "v14 did not save recovery macro-F1 or confusion matrices",
            "exact recovery-macro delta requires re-evaluating the v14 checkpoint",
            "quality failure does not invalidate the engineering smoke/mini execution",
        ],
    }


def build_processor(cfg: dict):
    from transformers import AutoProcessor

    backbone = cfg["backbone"]
    return AutoProcessor.from_pretrained(
        backbone["vlm_model"],
        min_pixels=backbone["min_pixels"],
        max_pixels=backbone["max_pixels"],
    )


def build_gold_components(
    cfg: dict,
    processor=None,
    train_records=None,
    trajectory_records=None,
) -> GoldComponents:
    """Build the model and derive every weight from the rows actually trained."""
    processor = processor or build_processor(cfg)
    train_records = (
        list(train_records)
        if train_records is not None
        else load_gold_split(cfg, "train")
    )
    action_w, failure_w, outcome_w = gold_class_weights(train_records)
    recovery_scheme = cfg["loss"].get(
        "recovery_weight_scheme", "sqrt_inverse_frequency",
    )
    if recovery_scheme != "sqrt_inverse_frequency":
        raise ValueError(f"unsupported recovery weight scheme: {recovery_scheme!r}")
    recovery_cap = float(cfg["loss"].get("recovery_weight_cap", 3.0))
    recovery_w = gold_recovery_class_weights(
        train_records,
        cap=recovery_cap,
        attempted_only=bool(cfg["loss"].get("hierarchical_recovery", False)),
    )

    if cfg["data"].get("recovery_transitions"):
        transition_index, _ = build_recovery_transition_index(
            trajectory_records if trajectory_records is not None else train_records,
        )
        selected_ids = {
            str(meta.get("sample_id") or f"{meta.get('task_id', '')}:{meta.get('step_index', index)}")
            for index, record in enumerate(train_records)
            for _, _, meta in [view(record)]
        }
        recovery_values = [
            transition["recovery_success"]
            for sample_id, transition in transition_index.items()
            if sample_id in selected_ids
        ]
    else:
        recovery_values = [
            view(record)[1].get("recovery_success") for record in train_records
        ]
    positives = sum(value is True for value in recovery_values)
    negatives = sum(value is False for value in recovery_values)
    recovery_ratio = negatives / max(positives, 1)
    recovery_pos_weight = torch.tensor([
        min(recovery_ratio, 5.0) if positives and negatives else 1.0
    ])
    needs_positive = sum(
        view(record)[1].get("recovery_success") is not None
        for record in train_records
    )
    needs_negative = len(train_records) - needs_positive
    needs_scheme = cfg["loss"].get(
        "needs_recovery_weight_scheme", "exact_inverse_frequency",
    )
    if needs_scheme != "exact_inverse_frequency":
        raise ValueError(
            f"unsupported needs-recovery weight scheme: {needs_scheme!r}"
        )
    needs_cap = float(cfg["loss"].get("needs_recovery_weight_cap", 5.0))
    needs_ratio = needs_negative / max(needs_positive, 1)
    needs_recovery_pos_weight = torch.tensor([
        min(needs_ratio, needs_cap)
        if needs_positive and needs_negative else 1.0
    ])

    model = WebAgentModel(cfg)
    device = "cuda"
    for module in (
        model.adapter,
        model.task_adapters,
        model.failure_head,
        model.action_head,
        model.memory_head,
        model.recovery_outcome_head,
    ):
        module.to(device)

    loss_fn = CombinedLoss(
        cfg,
        action_class_weights=action_w.to(device),
        failtype_class_weights=failure_w.to(device),
        outcome_class_weights=outcome_w.to(device),
        recovery_class_weights=recovery_w.to(device),
        recovery_success_pos_weight=recovery_pos_weight.to(device),
        needs_recovery_pos_weight=needs_recovery_pos_weight.to(device),
    ).to(device)

    print("train rows:", len(train_records))
    print("action weights:", [round(value, 2) for value in action_w.tolist()])
    print("failure weights:", [round(value, 2) for value in failure_w.tolist()])
    print("outcome weights:", [round(value, 2) for value in outcome_w.tolist()])
    print("recovery weights:", [round(value, 2) for value in recovery_w.tolist()])
    print(
        "recovery pos_weight:", round(float(recovery_pos_weight), 2),
        f"(attempted success={positives}, failure={negatives})",
    )
    print(
        "needs-recovery pos_weight:", round(float(needs_recovery_pos_weight), 2),
        f"(needed={needs_positive}, not-needed={needs_negative})",
    )
    weight_report = {
        "source_rows": len(train_records),
        "action": [float(value) for value in action_w.tolist()],
        "failure_type": [float(value) for value in failure_w.tolist()],
        "outcome": [float(value) for value in outcome_w.tolist()],
        "recovery_strategy": [float(value) for value in recovery_w.tolist()],
        "recovery_strategy_scheme": recovery_scheme,
        "recovery_strategy_cap": recovery_cap,
        "recovery_outcome_pos_weight": float(recovery_pos_weight.item()),
        "recovery_outcome_attempted_success": positives,
        "recovery_outcome_attempted_failure": negatives,
        "needs_recovery_pos_weight": float(needs_recovery_pos_weight.item()),
        "needs_recovery_weight_scheme": needs_scheme,
        "needs_recovery_positive": needs_positive,
        "needs_recovery_negative": needs_negative,
    }
    return GoldComponents(
        processor, model, loss_fn, train_records, weight_report, device,
    )


def run_gold_bbox_audit(cfg: dict) -> dict:
    """Audit bbox geometry and decide whether training can safely continue.

    ``mask`` keeps the complete records but removes invalid bbox targets from
    localization loss and metrics. Missing or unreadable source images remain
    fatal because masking a bbox cannot make the multimodal row usable.
    """
    train_records = load_gold_split(cfg, "train")
    validation_records = load_gold_split(cfg, "val")
    train_report = audit_bbox_geometry(train_records, cfg["data"]["root"])
    validation_report = audit_bbox_geometry(
        validation_records,
        cfg["data"]["root"],
    )
    split_reports = (train_report, validation_report)
    geometry_passed = all(report["status"] == "PASS" for report in split_reports)
    invalid_policy = str(cfg["data"].get("invalid_bbox_policy", "error"))
    masking_enabled = (
        bool(cfg["data"].get("strict_bbox_geometry", False))
        and invalid_policy == "mask"
    )
    fatal_rows = sum(
        report["fatal_invalid_bbox_rows"] for report in split_reports
    )
    valid_rows = sum(report["valid_bbox_rows"] for report in split_reports)
    accepted_with_masking = (
        not geometry_passed
        and masking_enabled
        and fatal_rows == 0
        and all(report["valid_bbox_rows"] > 0 for report in split_reports)
    )
    training_passed = geometry_passed or accepted_with_masking
    masked_rows = (
        sum(report["maskable_invalid_bbox_rows"] for report in split_reports)
        if accepted_with_masking else 0
    )
    return {
        "status": "PASS" if training_passed else "FAIL",
        "raw_geometry_status": "PASS" if geometry_passed else "FAIL",
        "training_disposition": (
            "PASS_CLEAN_GEOMETRY"
            if geometry_passed
            else (
                "PASS_WITH_INVALID_BBOX_MASKED"
                if accepted_with_masking
                else "FAIL"
            )
        ),
        "invalid_bbox_policy": invalid_policy,
        "retained_records": sum(report["records"] for report in split_reports),
        "bbox_supervision_rows": valid_rows,
        "masked_bbox_rows": masked_rows,
        "fatal_invalid_bbox_rows": fatal_rows,
        "test_rows_read": 0,
        "train": train_report,
        "validation": validation_report,
    }


def _select_valid_bbox_records(
    records: list[dict],
    data_root: str | Path,
    rows: int,
    seed: int,
) -> list[dict]:
    """Select deterministic, verified in-image bbox rows for a small probe."""
    candidates = [
        record for record in records
        if view(record)[1].get("action_target_bbox") is not None
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = []
    for record in candidates:
        report = audit_bbox_geometry([record], data_root, max_examples=1)
        if report["status"] == "PASS":
            selected.append(record)
            if len(selected) == rows:
                return selected
    raise ValueError(
        f"bbox probe requested {rows} valid rows but only {len(selected)} were found"
    )


@torch.no_grad()
def _evaluate_bbox_overfit(model, loader, device: str) -> dict:
    """Evaluate localization only on the fixed micro-overfit records."""
    model.eval()
    predictions = []
    targets = []
    masks = []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            output = model.forward_bbox(batch)
        predictions.append(output["bbox"].float().cpu())
        targets.append(batch["bbox"].float().cpu())
        masks.append(batch["bbox_mask"].view(-1).float().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    mask = torch.cat(masks)
    valid = mask > 0.5
    valid_prediction = prediction[valid]
    summary = bbox_iou_summary(prediction.numpy(), target.numpy(), mask.numpy())
    coordinate_std = valid_prediction.std(dim=0, unbiased=False)
    full_screen = (
        (valid_prediction[:, 0] <= 0.01)
        & (valid_prediction[:, 1] <= 0.01)
        & (valid_prediction[:, 2] >= 0.95)
        & (valid_prediction[:, 3] >= 0.95)
    )
    return {
        "bbox_mae": bbox_mae(prediction.numpy(), target.numpy(), mask.numpy()),
        "bbox_mean_iou": summary["mean"],
        "bbox_median_iou": summary["median"],
        "bbox_recall_iou50": summary["recall_50"],
        "bbox_rows": summary["rows"],
        "prediction_coordinate_mean": valid_prediction.mean(dim=0).tolist(),
        "prediction_coordinate_std": coordinate_std.tolist(),
        "full_screen_fraction": float(full_screen.float().mean()),
    }


def run_gold_bbox_overfit(
    cfg: dict,
    processor=None,
    rows: int | None = None,
    steps: int | None = None,
    seed: int = 42,
) -> dict:
    """Prove bbox trainability cheaply before another 5k diagnostic epoch."""
    overfit_cfg = deepcopy(cfg)
    overfit_cfg["data"]["recovery_transitions"] = False
    rows = int(rows or overfit_cfg["train"].get("bbox_overfit_rows", 32))
    steps = int(steps or overfit_cfg["train"].get("bbox_overfit_steps", 100))
    if rows <= 0 or steps <= 0:
        raise ValueError("bbox overfit rows and steps must both be positive")
    all_train_records = load_gold_split(overfit_cfg, "train")
    selected_records = _select_valid_bbox_records(
        all_train_records,
        overfit_cfg["data"]["root"],
        rows,
        seed,
    )
    geometry = audit_bbox_geometry(
        selected_records,
        overfit_cfg["data"]["root"],
    )
    require_valid_bbox_geometry(geometry, "bbox-overfit")

    processor = processor or build_processor(overfit_cfg)
    loader = build_gold_dataloader(
        overfit_cfg,
        "train",
        processor,
        records=selected_records,
        batch_size=overfit_cfg["optim"]["batch_size"],
        shuffle=False,
        num_workers=0,
        seed=seed,
    )
    model = WebAgentModel(overfit_cfg)
    if not model.spatial_grounding or model.action_head.bbox_parameterization != "cxcywh":
        raise AssertionError(
            "bbox overfit requires spatial grounding with cxcywh parameterization"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if "grounding" not in model.task_adapters:
        raise AssertionError("bbox overfit requires the separate grounding adapter")
    for module in (model.adapter, model.task_adapters, model.action_head):
        module.to("cuda")
    trainable_modules = [
        model.adapter,
        model.task_adapters["grounding"],
        model.action_head.grounding_attention,
        model.action_head.bbox_trunk,
        model.action_head.bbox,
    ]
    parameters = []
    seen = set()
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))

    initial = _evaluate_bbox_overfit(model, loader, "cuda")
    bbox_probe = model.action_head.bbox.weight
    grounding_probe = next(model.task_adapters["grounding"].parameters())
    probe_before = {
        "bbox": bbox_probe.detach().clone(),
        "grounding_adapter": grounding_probe.detach().clone(),
    }
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(overfit_cfg["optim"]["lr_heads"]),
        weight_decay=float(overfit_cfg["optim"].get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda")
    l1_ratio = float(overfit_cfg["loss"].get("bbox_l1_ratio", 5.0))
    giou_ratio = float(overfit_cfg["loss"].get("bbox_giou_ratio", 2.0))
    loss_history = []
    first_gradient_norms = {}
    completed_steps = 0
    model.train()
    model.encoder.eval()
    while completed_steps < steps:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model.forward_bbox(batch)
                row_mask = (batch["bbox_mask"].view(-1) > 0).to("cuda")
                l1, giou = detr_bbox_loss_terms(
                    prediction["bbox_cxcywh"][row_mask],
                    batch["bbox"].to("cuda")[row_mask],
                )
                loss = l1_ratio * l1 + giou_ratio * giou
            if not torch.isfinite(loss):
                raise FloatingPointError("bbox micro-overfit loss is non-finite")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not first_gradient_norms:
                first_gradient_norms = {
                    "bbox": (
                        float(bbox_probe.grad.detach().float().norm())
                        if bbox_probe.grad is not None else 0.0
                    ),
                    "grounding_adapter": (
                        float(grounding_probe.grad.detach().float().norm())
                        if grounding_probe.grad is not None else 0.0
                    ),
                }
            torch.nn.utils.clip_grad_norm_(
                parameters,
                float(overfit_cfg["optim"].get("grad_clip", 1.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            loss_history.append({
                "step": completed_steps,
                "total": float(loss.detach()),
                "l1": float(l1.detach()),
                "giou": float(giou.detach()),
            })
            completed_steps += 1
            if completed_steps >= steps:
                break

    final = _evaluate_bbox_overfit(model, loader, "cuda")
    window = min(10, len(loss_history))
    initial_loss = sum(row["total"] for row in loss_history[:window]) / window
    final_loss = sum(row["total"] for row in loss_history[-window:]) / window
    iou_gain = final["bbox_mean_iou"] - initial["bbox_mean_iou"]
    checks = {
        "loss_decreased": final_loss < initial_loss,
        "mean_iou_gain_at_least_registered_minimum": (
            iou_gain
            >= float(overfit_cfg["train"].get("bbox_overfit_min_iou_gain", 0.10))
        ),
        "final_mean_iou_at_least_registered_minimum": (
            final["bbox_mean_iou"]
            >= float(overfit_cfg["train"].get("bbox_overfit_min_mean_iou", 0.20))
        ),
        "full_screen_fraction_below_registered_maximum": (
            final["full_screen_fraction"]
            <= float(overfit_cfg["train"].get(
                "bbox_overfit_max_full_screen_fraction", 0.10,
            ))
        ),
        "prediction_coordinates_are_not_constant": (
            max(final["prediction_coordinate_std"]) >= 0.01
        ),
        "bbox_gradient_nonzero": first_gradient_norms.get("bbox", 0.0) > 0.0,
        "grounding_gradient_nonzero": (
            first_gradient_norms.get("grounding_adapter", 0.0) > 0.0
        ),
    }
    updates = {
        "bbox": float((bbox_probe.detach() - probe_before["bbox"]).float().norm()),
        "grounding_adapter": float(
            (grounding_probe.detach() - probe_before["grounding_adapter"])
            .float().norm()
        ),
    }
    checks["bbox_update_nonzero"] = updates["bbox"] > 0.0
    checks["grounding_update_nonzero"] = updates["grounding_adapter"] > 0.0
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "rows": rows,
        "steps": steps,
        "test_rows_read": 0,
        "bbox_geometry": geometry,
        "loss": {
            "first_window_mean": initial_loss,
            "last_window_mean": final_loss,
            "first_step": loss_history[0],
            "last_step": loss_history[-1],
        },
        "initial": initial,
        "final": final,
        "mean_iou_gain": iou_gain,
        "first_gradient_norms": first_gradient_norms,
        "update_norms": updates,
        "checks": checks,
        "registered_thresholds": {
            "min_iou_gain": float(overfit_cfg["train"].get(
                "bbox_overfit_min_iou_gain", 0.10,
            )),
            "min_final_mean_iou": float(overfit_cfg["train"].get(
                "bbox_overfit_min_mean_iou", 0.20,
            )),
            "max_full_screen_fraction": float(overfit_cfg["train"].get(
                "bbox_overfit_max_full_screen_fraction", 0.10,
            )),
        },
    }


def run_gold_smoke(
    cfg: dict,
    processor=None,
    rows: int = 16,
    seed: int = 42,
) -> dict:
    """Verify causal streams, output shapes, finite loss, backward, and weight update."""
    all_train_records = load_gold_split(cfg, "train")
    smoke_records = select_smoke_records(all_train_records, rows, seed)
    smoke_bbox_geometry = audit_bbox_geometry(
        smoke_records,
        cfg["data"]["root"],
    )
    if smoke_bbox_geometry["valid_bbox_rows"] == 0:
        # Category coverage is chosen before select_smoke_records appends filler;
        # replacing the final filler row preserves that coverage.
        smoke_records[-1] = _select_valid_bbox_records(
            all_train_records,
            cfg["data"]["root"],
            1,
            seed,
        )[0]
        smoke_bbox_geometry = audit_bbox_geometry(
            smoke_records,
            cfg["data"]["root"],
        )
    components = build_gold_components(
        cfg,
        processor,
        train_records=smoke_records,
        trajectory_records=all_train_records,
    )
    loader = build_gold_dataloader(
        cfg,
        "train",
        components.processor,
        records=components.train_records,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=False,
        num_workers=0,
        seed=seed,
        trajectory_records=all_train_records,
    )
    if len(loader.dataset) != rows:
        raise AssertionError(f"smoke loader has {len(loader.dataset)} rows, expected {rows}")

    model = components.model
    model.train()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.lora_parameters(), "lr": cfg["optim"]["lr_lora"]},
            {
                "params": model.head_parameters() + list(components.loss_fn.parameters()),
                "lr": cfg["optim"]["lr_heads"],
            },
        ],
        weight_decay=cfg["optim"].get("weight_decay", 0.01),
    )
    scaler = torch.amp.GradScaler("cuda")
    probes = {
        "bbox": model.action_head.bbox.weight,
        "strategy": model.failure_head.recovery.weight,
        "recovery_outcome": model.recovery_outcome_head.recovery_outcome.weight,
    }
    if model.failure_head.needs_recovery is not None:
        probes["needs_recovery"] = model.failure_head.needs_recovery.weight
    if "grounding" in model.task_adapters:
        probes["grounding_adapter"] = next(
            model.task_adapters["grounding"].parameters()
        )
    before = {
        name: parameter.detach().clone()
        for name, parameter in probes.items()
    }

    optimizer.zero_grad(set_to_none=True)
    term_sums = {}
    first_batch = None
    first_predictions = None
    processed_rows = 0
    pre_images = 0
    post_images = 0
    bbox_rows = 0
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            predictions = model(batch)
            device_batch = {
                key: value.to(components.device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            terms = components.loss_fn(predictions, device_batch)
        if not torch.isfinite(terms["total"]):
            raise FloatingPointError("smoke loss is NaN or infinite")
        scaler.scale(terms["total"] / len(loader)).backward()
        for key, value in terms.items():
            term_sums[key] = term_sums.get(key, 0.0) + float(value.detach())
        if first_batch is None:
            first_batch = batch
            first_predictions = predictions
        processed_rows += int(batch["label_outcome"].shape[0])
        pre_images += int(batch["pre_image_grid_thw"].shape[0])
        post_images += int(batch["post_image_grid_thw"].shape[0])
        bbox_rows += int(batch["bbox_mask"].sum())

    if processed_rows != rows or first_batch is None or first_predictions is None:
        raise AssertionError(f"smoke processed {processed_rows} rows, expected {rows}")
    scaler.unscale_(optimizer)
    probe_gradient_norms = {
        name: (
            float(parameter.grad.detach().float().norm())
            if parameter.grad is not None else 0.0
        )
        for name, parameter in probes.items()
    }
    gradients = [
        parameter.grad
        for parameter in model.trainable_parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("smoke backward produced missing or non-finite gradients")
    scaler.step(optimizer)
    scaler.update()
    probe_update_norms = {
        name: float((parameter.detach() - before[name]).float().norm())
        for name, parameter in probes.items()
    }
    inactive_probes = [
        name for name in probes
        if probe_gradient_norms[name] <= 0.0 or probe_update_norms[name] <= 0.0
    ]
    if bbox_rows <= 0:
        raise AssertionError("smoke rows contain no bbox supervision")
    if inactive_probes:
        raise AssertionError(
            f"smoke probes did not receive gradient/update: {inactive_probes}"
        )

    batch_size = first_batch["label_outcome"].shape[0]
    expected_shapes = {
        "fused_pre": (batch_size, cfg["fused_dim"]),
        "fused_post": (batch_size, cfg["fused_dim"]),
        "outcome": (batch_size, 2),
        "failure_type": (batch_size, 4),
        "action_type": (batch_size, 6),
        "bbox": (batch_size, 4),
        "recovery": (batch_size, 6),
        "memory_flag": (batch_size, 1),
    }
    if cfg.get("model", {}).get("bbox_parameterization") == "cxcywh":
        expected_shapes["bbox_cxcywh"] = (batch_size, 4)
    if cfg["loss"].get("hierarchical_recovery"):
        expected_shapes["needs_recovery"] = (batch_size, 1)
    recovery_batch_size = int(first_batch.get(
        "recovery_row_indices", torch.empty(0),
    ).numel())
    expected_shapes["recovery_outcome"] = (
        recovery_batch_size if cfg["data"].get("recovery_transitions") else batch_size,
        1,
    )
    for key, expected in expected_shapes.items():
        actual = tuple(first_predictions[key].shape)
        if actual != expected:
            raise AssertionError(f"{key} shape={actual}, expected={expected}")
    spatial_counts = first_predictions.get("spatial_token_counts")
    if cfg.get("model", {}).get("spatial_grounding"):
        if spatial_counts is None or not bool((spatial_counts > 0).all()):
            raise AssertionError("spatial grounding received an empty image-token mask")

    return {
        "status": "PASS",
        "test_rows_read": 0,
        "dataset_rows": rows,
        "processed_rows": processed_rows,
        "forward_batch_size": batch_size,
        "loss": term_sums["total"] / len(loader),
        "loss_terms": {key: value / len(loader) for key, value in term_sums.items()},
        "class_weights": components.class_weight_report,
        "train_distribution": label_distribution(smoke_records),
        "pre_images_processed": pre_images,
        "post_images_processed": post_images,
        "parameter_changed": True,
        "probe_gradient_norms": probe_gradient_norms,
        "probe_update_norms": probe_update_norms,
        "bbox_supervised_rows": bbox_rows,
        "bbox_geometry": smoke_bbox_geometry,
        "spatial_tokens_per_row": (
            spatial_counts.detach().cpu().tolist()
            if spatial_counts is not None else []
        ),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def run_gold_mini(
    cfg: dict,
    processor=None,
    train_rows: int = 5_000,
    val_rows: int = 500,
    epochs: int = 5,
    seed: int = 42,
) -> dict:
    """Run the controlled recovery experiment without reading the test split.

    Training uses joint proportional coverage plus distribution-preserving
    recovery-aware batches. Validation intentionally retains the legacy v14
    failure-stratified selector so the 500-row comparison set is identical.
    """
    mini_cfg = deepcopy(cfg)
    mini_cfg["train"]["epochs"] = epochs
    # A requested mini run should finish every epoch; early stopping is for the
    # longer headline run, not this fixed-size development gate.
    mini_cfg["train"]["early_stop_patience"] = max(
        epochs, mini_cfg["train"].get("early_stop_patience", 0)
    )
    experiment_tag = str(
        mini_cfg["train"].get("controlled_experiment_tag", "recovery_v1")
    ).upper()
    mini_cfg["name"] = f"{cfg['name']}_MINI_{experiment_tag}"
    mini_cfg["train"]["metrics_csv"] = (
        f"results/gold_mini_{experiment_tag.lower()}_metrics.csv"
    )
    # Five mini epochs are cheap enough to preserve independently best heads.
    mini_cfg["train"]["keep_top_k"] = max(
        epochs, int(mini_cfg["train"].get("keep_top_k", 3)),
    )

    all_train_records = load_gold_split(mini_cfg, "train")
    selected_train_records = select_recovery_aware_gold_subset(
        all_train_records, train_rows, seed,
    )
    all_val_records = load_gold_split(mini_cfg, "val")
    selected_val_records = stratified_gold_subsample(all_val_records, val_rows, seed)
    selected_bbox_geometry = {
        "train": audit_bbox_geometry(
            selected_train_records,
            mini_cfg["data"]["root"],
        ),
        "validation": audit_bbox_geometry(
            selected_val_records,
            mini_cfg["data"]["root"],
        ),
    }
    components = build_gold_components(
        mini_cfg,
        processor,
        train_records=selected_train_records,
        trajectory_records=all_train_records,
    )
    train_loader = build_gold_dataloader(
        mini_cfg,
        "train",
        components.processor,
        records=selected_train_records,
        shuffle=False,
        num_workers=mini_cfg["data"].get("num_workers", 2),
        seed=seed,
        recovery_aware=True,
        trajectory_records=all_train_records,
    )
    val_loader = build_gold_dataloader(
        mini_cfg,
        "val",
        components.processor,
        records=selected_val_records,
        shuffle=False,
        num_workers=mini_cfg["data"].get("num_workers", 2),
        seed=seed,
        trajectory_records=all_val_records,
    )
    trainer = Trainer(
        components.model,
        components.loss_fn,
        mini_cfg,
        train_loader,
        val_loader,
        train_sampler=train_loader.batch_sampler,
    )
    result = trainer.fit()
    if not result["checkpoints"]:
        raise AssertionError("mini training did not create a checkpoint")

    checkpoint = Path(result["checkpoints"][0])
    trainer.load_checkpoint(checkpoint, resume_training=False)
    checkpoint_roundtrip = _verify_checkpoint_roundtrip(trainer, val_loader)
    history = result["history"]
    loss_decreased = (
        history[-1]["train_nominal_weighted_loss"]
        < history[0]["train_nominal_weighted_loss"]
        if len(history) > 1 else None
    )
    if len(history) > 1 and not loss_decreased:
        raise AssertionError("nominal weighted mini loss did not decrease across epochs")
    if result["best_metric"] <= 0.0:
        raise AssertionError(
            f"validation {result['early_stop_metric']} did not beat the degenerate floor"
        )

    scheduled_batches = recovery_aware_batch_indices(
        selected_train_records,
        mini_cfg["optim"]["batch_size"],
        seed=seed,
        epoch=0,
    )
    attempted_batches = sum(
        any(view(selected_train_records[index])[1].get("recovery_success") is not None
            for index in batch)
        for batch in scheduled_batches
    )
    sampling_report = {
        "train_selector": "joint_proportional_v1",
        "validation_selector": "legacy_failure_stratified_v14_fixed",
        "validation_comparable_to_v14": True,
        "batch_sampler": "recovery_aware_no_oversampling_v1",
        "physical_batch_size": int(mini_cfg["optim"]["batch_size"]),
        "batches": len(scheduled_batches),
        "batches_with_attempted_recovery": attempted_batches,
        "attempted_batch_coverage": attempted_batches / max(len(scheduled_batches), 1),
        "unique_train_rows_scheduled": len({
            index for batch in scheduled_batches for index in batch
        }),
        "duplicate_train_rows_scheduled": (
            sum(len(batch) for batch in scheduled_batches)
            - len({index for batch in scheduled_batches for index in batch})
        ),
    }
    quality_gates = _controlled_quality_gates(result)
    is_v2 = bool(mini_cfg["data"].get("recovery_transitions"))
    is_v2_1_or_later = experiment_tag.lower() in {
        "recovery_v2_1", "recovery_v2_2",
    }
    is_v2_2 = experiment_tag.lower() == "recovery_v2_2"
    training_changes = [
        "joint proportional training subset coverage",
        "recovery-aware distribution-preserving physical batches",
        "sqrt inverse-frequency recovery-strategy class weights",
        "weights derived from selected training rows only",
    ]
    if is_v2:
        training_changes.extend([
            "executed action added only to the post-action stream",
            "binary needs-recovery plus attempted-row-only strategy supervision",
            "proper sparse recovery-transition stream for recovery success",
            (
                "spatial grounding bbox head with centre L1(5) plus GIoU(2)"
                if is_v2_2
                else "spatial grounding bbox head with SmoothL1 plus GIoU"
            ),
            "task-specific residual adapters and uncertainty loss weighting",
        ])
    if is_v2_1_or_later:
        training_changes.extend([
            "exact selected-row needs-recovery positive weighting",
            "bbox-specific residual adapter separate from action classification",
        ])
    if is_v2_2:
        training_changes.extend([
            "strict bbox validation with invalid targets masked only for localization",
            "internal centre-format bbox regression with stable public xywh output",
            "separately logged bbox coordinate-L1 and GIoU components",
        ])
    fixed_factors = [
        "backbone",
        "validation_rows",
        "seed",
        "batch_size",
        "gradient_accumulation",
        "learning_rates",
        "primary_selection_metric",
    ]
    if not is_v2:
        fixed_factors.extend(["model_architecture", "loss_coefficients"])

    return {
        **result,
        "status": "PASS",
        "train_rows": len(train_loader.dataset),
        "val_rows": len(val_loader.dataset),
        "test_rows_read": 0,
        "loss_decreased": loss_decreased,
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "best_checkpoint": str(checkpoint),
        "class_weights": components.class_weight_report,
        "sampling": sampling_report,
        "quality_gates": quality_gates,
        "train_distribution": label_distribution(selected_train_records),
        "validation_distribution": label_distribution(selected_val_records),
        "bbox_geometry": selected_bbox_geometry,
        "recovery_transition_reports": {
            "train": train_loader.dataset.transition_report,
            "validation": val_loader.dataset.transition_report,
        },
        "recovery_class_audit": {
            "train": recovery_class_audit(all_train_records),
            "validation": recovery_class_audit(all_val_records),
        },
        "experiment_control": {
            "baseline": "kaggle-gold-v14",
            "experiment_tag": experiment_tag.lower(),
            "fixed": fixed_factors,
            "registered_structural_factors": (
                [
                    "causal recovery transition stream",
                    "hierarchical recovery prediction",
                    "post-action executed-action conditioning",
                    "spatial bbox grounding and IoU-aware loss",
                    "task adapters and uncertainty weighting",
                    *(
                        [
                            "exact needs-recovery balancing",
                            "bbox-specific grounding adapter",
                        ]
                        if is_v2_1_or_later else []
                    ),
                    *(
                        [
                            "strict source-image bbox audit and target-level invalid masking",
                            "DETR-style centre-format localization objective",
                            "bbox-only micro-overfit trainability gate",
                        ]
                        if is_v2_2 else []
                    ),
                ]
                if is_v2 else []
            ),
            "training_changes": training_changes,
            "observational_changes": [
                "honest scalar metrics",
                "per-class diagnostics",
                "raw and weighted loss terms",
                "raw conditional-strategy metrics separate from composed recovery",
                "per-head gradient and parameter-update norms",
                "uncertainty log-variances and effective multipliers",
                "bbox coordinate prediction and target distributions",
                *(
                    [
                        "separate bbox L1 and GIoU loss components",
                        "full train/validation bbox boundary audit",
                    ]
                    if is_v2_2 else []
                ),
                "all mini epoch checkpoints retained",
            ],
        },
    }


@torch.no_grad()
def reevaluate_gold_validation_checkpoint(
    cfg: dict,
    checkpoint: str | Path,
    processor=None,
    val_rows: int = 500,
    seed: int = 42,
) -> dict:
    """Re-score an old checkpoint on the fixed v14 validation subset only."""
    train_records = load_gold_split(cfg, "train")
    components = build_gold_components(cfg, processor, train_records=train_records)
    val_records = stratified_gold_subsample(load_gold_split(cfg, "val"), val_rows, seed)
    val_loader = build_gold_dataloader(
        cfg,
        "val",
        components.processor,
        records=val_records,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 2),
        seed=seed,
    )

    from peft import set_peft_model_state_dict

    saved = torch.load(checkpoint, map_location=components.device)
    set_peft_model_state_dict(components.model.encoder.model, saved["lora"])
    components.model.adapter.load_state_dict(saved["adapter"])
    if "task_adapters" in saved:
        components.model.task_adapters.load_state_dict(saved["task_adapters"])
    components.model.failure_head.load_state_dict(saved["failure"])
    components.model.action_head.load_state_dict(saved["action"])
    components.model.memory_head.load_state_dict(saved["memory"])
    components.model.recovery_outcome_head.load_state_dict(saved["recovery_outcome"])
    predictions = collect_predictions(components.model, val_loader, components.device)
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "validation_selector": "legacy_failure_stratified_v14_fixed",
        "val_rows": len(val_loader.dataset),
        "test_rows_read": 0,
        "metrics": compute_metrics(predictions),
        "diagnostics": build_diagnostics(predictions),
    }


@torch.no_grad()
def _verify_checkpoint_roundtrip(trainer: Trainer, val_loader) -> bool:
    """Perturb critical heads/adapters, reload, and prove their logits return."""
    trainer.model.eval()
    batch = next(iter(val_loader))
    with torch.autocast("cuda", dtype=torch.float16):
        original = trainer.model(batch)
        keys = ["action_type", "bbox", "recovery", "recovery_outcome"]
        if "bbox_cxcywh" in original:
            keys.append("bbox_cxcywh")
        if "needs_recovery" in original:
            keys.append("needs_recovery")
        expected = {key: original[key].float().cpu() for key in keys}

    parameters = [
        trainer.model.action_head.action_type.weight,
        trainer.model.action_head.bbox.weight,
        trainer.model.failure_head.recovery.weight,
        trainer.model.recovery_outcome_head.recovery_outcome.weight,
    ]
    if trainer.model.failure_head.needs_recovery is not None:
        parameters.append(trainer.model.failure_head.needs_recovery.weight)
    if "grounding" in trainer.model.task_adapters:
        parameters.append(next(trainer.model.task_adapters["grounding"].parameters()))
    for parameter in parameters:
        parameter.add_(0.25)
    trainer.load_checkpoint(trainer.best[0][1], resume_training=False)
    trainer.model.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        restored = trainer.model(batch)
    mismatched = [
        key for key in keys
        if not torch.allclose(
            expected[key], restored[key].float().cpu(), atol=1e-5, rtol=1e-5,
        )
    ]
    if mismatched:
        raise AssertionError(
            f"checkpoint reload did not restore identical outputs: {mismatched}"
        )
    return True
