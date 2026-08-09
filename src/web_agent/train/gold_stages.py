"""Reusable smoke and mini stages for the structured Gold 40K dataset.

The notebook should orchestrate experiments, not reimplement training logic. This
module owns component construction, causal smoke checks, validation-only mini
training, and checkpoint round-trip verification.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import math
import random

import torch
from PIL import Image, ImageDraw

from web_agent.data.gold_dataloader import (
    build_gold_dataloader,
    gold_class_weights,
    gold_recovery_class_weights,
    load_gold_split,
    load_gold_split_sources,
    select_smoke_records,
    source_loss_active,
    stratified_gold_subsample,
)
from web_agent.data.bbox_audit import (
    audit_bbox_geometry,
    audit_bbox_probe_feasibility,
    bbox_log_size_prior,
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
from web_agent.models.bbox import (
    bbox_attention_kl_loss,
    detr_bbox_log_size_loss_terms,
    detr_bbox_loss_terms,
)
from web_agent.data.review_overlay import ReviewOverlay
from web_agent.models.model import WebAgentModel
from web_agent.eval.metrics import bbox_iou_summary, bbox_iou_values, bbox_mae
from web_agent.train.trainer import (
    Trainer,
    build_diagnostics,
    collect_predictions,
    compute_metrics,
)
from web_agent.train.selection import (
    LEGACY_OUTCOME_RULE,
    controlled_quality_gates,
    require_selected_checkpoint,
)
from web_agent.train.resume import sha256_file, training_signature


def _validate_resume_checkpoint(path: str | Path, expected_cfg: dict) -> dict:
    """Reject a ``last.ckpt`` that cannot safely continue this exact run."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    required = {
        "optimizer",
        "scheduler",
        "scaler",
        "rng_state",
        "config",
        "epoch",
        "step",
        "epoch_complete",
        "history",
        "early_stop_state",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"resume checkpoint is missing state: {missing}")
    saved_signature = training_signature(checkpoint["config"])
    expected_signature = training_signature(expected_cfg)
    if saved_signature != expected_signature:
        differing = [
            key
            for key in expected_signature
            if saved_signature.get(key) != expected_signature.get(key)
        ]
        raise ValueError(
            "resume checkpoint does not match this run; "
            f"differing sections={differing}"
        )
    if not checkpoint["epoch_complete"]:
        if int(checkpoint.get("batch_in_epoch", 0)) <= 0:
            raise ValueError("mid-epoch checkpoint has no next-batch position")
        if checkpoint.get("epoch_state") is None:
            raise ValueError("mid-epoch checkpoint has no partial epoch state")
    report = {
        "status": "PASS",
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "epoch_complete": bool(checkpoint["epoch_complete"]),
        "next_batch_index": int(checkpoint.get("batch_in_epoch", 0)),
        "global_step": int(checkpoint["step"]),
        "completed_history_epochs": [
            int(row["epoch"]) for row in checkpoint["history"]
        ],
        "optimizer_restored": True,
        "scheduler_restored": True,
        "scaler_restored": True,
        "rng_state_restored": True,
    }
    del checkpoint
    return report


@dataclass
class GoldComponents:
    processor: object
    model: WebAgentModel
    loss_fn: CombinedLoss
    train_records: list[dict]
    class_weight_report: dict
    device: str = "cuda"


def _recovery_version_at_least(experiment_tag: str, minimum_minor: int) -> bool:
    """Compare recovery-v2 experiment suffixes without an omission-prone set."""
    normalized = experiment_tag.lower()
    if normalized == "recovery_v2":
        minor = 0
    elif normalized.startswith("recovery_v2_"):
        suffix = normalized.removeprefix("recovery_v2_")
        try:
            minor = int(suffix)
        except ValueError:
            return False
    else:
        return False
    return minor >= minimum_minor


def build_processor(cfg: dict):
    from transformers import AutoProcessor
    from web_agent.models.encoders.vlm_contract import get_vlm_contract

    backbone = cfg["backbone"]
    contract = get_vlm_contract(cfg)
    kwargs = {
        "revision": backbone.get("revision", "main"),
        "trust_remote_code": bool(backbone.get("trust_remote_code", False)),
    }
    if contract.processor_uses_pixel_bounds:
        kwargs.update({
            "min_pixels": backbone["min_pixels"],
            "max_pixels": backbone["max_pixels"],
        })
    return AutoProcessor.from_pretrained(backbone["vlm_model"], **kwargs)


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
            if source_loss_active(record, "recovery_outcome")
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
        if source_loss_active(record, "needs_recovery")
    )
    needs_rows = sum(
        source_loss_active(record, "needs_recovery")
        for record in train_records
    )
    needs_negative = needs_rows - needs_positive
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
    bbox_prior_report = None
    if (
        cfg.get("model", {}).get("bbox_size_parameterization", "sigmoid")
        == "log_space"
    ):
        bbox_prior_report = bbox_log_size_prior(
            [
                record for record in train_records
                if source_loss_active(record, "bbox")
            ],
            cfg["data"]["root"],
        )
        model.action_head.initialize_bbox_size_prior(
            bbox_prior_report["log_wh"]
        )
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
    if bbox_prior_report is not None:
        weight_report["bbox_log_size_prior"] = bbox_prior_report
        print("bbox log-size prior:", bbox_prior_report)
    return GoldComponents(
        processor, model, loss_fn, train_records, weight_report, device,
    )


def _training_autocast_dtype(cfg: dict):
    precision = str(cfg["optim"].get("mixed_precision", "fp16")).lower()
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but this GPU does not support it")
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise ValueError("optim.mixed_precision must be 'fp16' or 'bf16'")


def run_gold_bbox_audit(cfg: dict) -> dict:
    """Audit bbox geometry and decide whether training can safely continue.

    ``mask`` keeps the complete records but removes invalid bbox targets from
    localization loss and metrics. Missing or unreadable source images remain
    fatal because masking a bbox cannot make the multimodal row usable.
    """
    train_records = load_gold_split(cfg, "train")
    validation_records = load_gold_split(cfg, "val")
    train_report = audit_bbox_geometry(
        [
            record for record in train_records
            if source_loss_active(record, "bbox")
        ],
        cfg["data"]["root"],
    )
    validation_report = audit_bbox_geometry(
        [
            record for record in validation_records
            if source_loss_active(record, "bbox")
        ],
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


def _probe_record_id(record: dict, index: int) -> str:
    """Return the stable identifier shared by the montage and overfit report."""
    _, _, meta = view(record)
    return str(
        meta.get("sample_id") or meta.get("task_id") or f"selected-row:{index}"
    )


def save_gold_bbox_probe_montage(
    cfg: dict,
    output_path: str | Path,
    *,
    rows: int = 32,
    seed: int = 42,
    columns: int = 4,
) -> dict:
    """Draw the exact deterministic micro-overfit targets for human review."""
    records = _select_valid_bbox_records(
        load_gold_split(cfg, "train"),
        cfg["data"]["root"],
        rows,
        seed,
    )
    if columns <= 0:
        raise ValueError("montage columns must be positive")
    tile_width, image_height, label_height = 384, 240, 28
    tile_height = image_height + label_height
    montage_rows = (len(records) + columns - 1) // columns
    montage = Image.new(
        "RGB",
        (columns * tile_width, montage_rows * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(montage)
    record_ids = []
    for index, record in enumerate(records):
        inputs, labels, _ = view(record)
        record_id = _probe_record_id(record, index)
        record_ids.append(record_id)
        image_path = Path(cfg["data"]["root"]) / inputs["state_before"]
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            original_width, original_height = source.size
            scale = min(
                tile_width / original_width,
                image_height / original_height,
            )
            rendered_size = (
                max(1, round(original_width * scale)),
                max(1, round(original_height * scale)),
            )
            rendered = source.resize(rendered_size, Image.Resampling.LANCZOS)
        column = index % columns
        row = index // columns
        tile_x = column * tile_width
        tile_y = row * tile_height
        offset_x = tile_x + (tile_width - rendered_size[0]) // 2
        offset_y = tile_y + label_height + (image_height - rendered_size[1]) // 2
        montage.paste(rendered, (offset_x, offset_y))
        box = labels["action_target_bbox"]
        x0 = offset_x + float(box["x"]) * scale
        y0 = offset_y + float(box["y"]) * scale
        x1 = x0 + float(box["width"]) * scale
        y1 = y0 + float(box["height"]) * scale
        draw.rectangle((x0, y0, x1, y1), outline="#00ff00", width=3)
        draw.text((tile_x + 6, tile_y + 6), record_id[:54], fill="black")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output, quality=92)
    return {
        "status": "PASS",
        "path": str(output),
        "rows": len(records),
        "seed": seed,
        "columns": columns,
        "record_ids": record_ids,
        "test_rows_read": 0,
    }


@torch.no_grad()
def _evaluate_bbox_overfit(
    model,
    loader,
    device: str,
    *,
    force_fp32_grounding: bool = False,
    record_ids: list[str] | None = None,
) -> dict:
    """Evaluate localization only on the fixed micro-overfit records."""
    model.eval()
    predictions = []
    targets = []
    masks = []
    attention_entropies = []
    internal_boxes = []
    log_sizes = []
    attention_peaks = []
    attention_kl_measurements = []
    for batch in loader:
        if force_fp32_grounding:
            output = model.forward_bbox(batch, force_fp32_grounding=True)
        else:
            with torch.autocast("cuda", dtype=torch.float16):
                output = model.forward_bbox(batch)
        predictions.append(output["bbox"].float().cpu())
        targets.append(batch["bbox"].float().cpu())
        masks.append(batch["bbox_mask"].view(-1).float().cpu())
        if "bbox_cxcywh" in output:
            internal_boxes.append(output["bbox_cxcywh"].float().cpu())
        if "bbox_log_wh" in output:
            log_sizes.append(output["bbox_log_wh"].float().cpu())
        if "bbox_attention_entropy" in output:
            attention_entropies.append(
                output["bbox_attention_entropy"].float().cpu()
            )
        if "bbox_attention_weights" in output:
            attention = output["bbox_attention_weights"].float()
            attention_mask = output["bbox_spatial_mask"].bool()
            peak_index = attention.masked_fill(
                ~attention_mask, float("-inf")
            ).argmax(dim=-1)
            peak = output["bbox_spatial_coords"].gather(
                1,
                peak_index[:, None, None].expand(-1, 1, 2),
            ).squeeze(1)
            attention_peaks.append(peak.float().cpu())
            attention_row_mask = batch["bbox_mask"].view(-1).bool().to(
                attention.device
            )
            if attention_row_mask.any():
                attention_kl = bbox_attention_kl_loss(
                    output["bbox_attention_weights"][attention_row_mask],
                    output["bbox_spatial_coords"][attention_row_mask],
                    output["bbox_spatial_mask"][attention_row_mask],
                    batch["bbox"].to(attention.device)[attention_row_mask],
                )
                attention_kl_measurements.append((
                    float(attention_kl),
                    int(attention_row_mask.sum()),
                ))
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    mask = torch.cat(masks)
    valid = mask > 0.5
    valid_prediction = prediction[valid]
    valid_target = target[valid]
    summary = bbox_iou_summary(prediction.numpy(), target.numpy(), mask.numpy())
    coordinate_std = valid_prediction.std(dim=0, unbiased=False)
    full_screen = (
        (valid_prediction[:, 0] <= 0.01)
        & (valid_prediction[:, 1] <= 0.01)
        & (valid_prediction[:, 2] >= 0.95)
        & (valid_prediction[:, 3] >= 0.95)
    )
    report = {
        "bbox_mae": bbox_mae(prediction.numpy(), target.numpy(), mask.numpy()),
        "bbox_mean_iou": summary["mean"],
        "bbox_median_iou": summary["median"],
        "bbox_recall_iou50": summary["recall_50"],
        "bbox_rows": summary["rows"],
        "prediction_coordinate_mean": valid_prediction.mean(dim=0).tolist(),
        "prediction_coordinate_std": coordinate_std.tolist(),
        "prediction_coordinate_min": valid_prediction.min(dim=0).values.tolist(),
        "prediction_coordinate_max": valid_prediction.max(dim=0).values.tolist(),
        "target_coordinate_mean": valid_target.mean(dim=0).tolist(),
        "target_coordinate_std": valid_target.std(dim=0, unbiased=False).tolist(),
        "full_screen_fraction": float(full_screen.float().mean()),
    }
    entropy = None
    if attention_entropies:
        entropy = torch.cat(attention_entropies)
        report["attention_entropy_mean"] = float(entropy.mean())
        report["attention_entropy_std"] = float(entropy.std(unbiased=False))
    if attention_kl_measurements:
        total_rows = sum(rows for _, rows in attention_kl_measurements)
        report["bbox_attention_kl"] = sum(
            value * rows for value, rows in attention_kl_measurements
        ) / total_rows
    internal = None
    if internal_boxes:
        internal = torch.cat(internal_boxes)[valid]
        report["internal_cxcywh_mean"] = internal.mean(dim=0).tolist()
        report["internal_cxcywh_std"] = internal.std(
            dim=0, unbiased=False,
        ).tolist()
        report["internal_size_min"] = internal[:, 2:].min(dim=0).values.tolist()
        report["internal_size_max"] = internal[:, 2:].max(dim=0).values.tolist()
    log_wh = None
    if log_sizes:
        log_wh = torch.cat(log_sizes)[valid]
        report["predicted_log_wh_mean"] = log_wh.mean(dim=0).tolist()
        report["predicted_log_wh_std"] = log_wh.std(
            dim=0, unbiased=False,
        ).tolist()
        report["predicted_log_wh_min"] = log_wh.min(dim=0).values.tolist()
        report["predicted_log_wh_max"] = log_wh.max(dim=0).values.tolist()
    valid_indices = torch.nonzero(valid, as_tuple=False).view(-1).tolist()
    identifiers = record_ids or [f"row:{index}" for index in range(len(mask))]
    if len(identifiers) != len(mask):
        raise ValueError("record_ids must align with the evaluated loader rows")
    iou = bbox_iou_values(
        prediction.numpy(), target.numpy(), mask.numpy(),
    )
    peaks = torch.cat(attention_peaks)[valid] if attention_peaks else None
    per_row = []
    for local_index, source_index in enumerate(valid_indices):
        predicted_box = valid_prediction[local_index]
        target_box = valid_target[local_index]
        predicted_center = (
            internal[local_index, :2]
            if internal is not None
            else predicted_box[:2] + 0.5 * predicted_box[2:]
        )
        target_center = target_box[:2] + 0.5 * target_box[2:]
        row = {
            "record_id": identifiers[source_index],
            "iou": float(iou[local_index]),
            "prediction_xywh": predicted_box.tolist(),
            "target_xywh": target_box.tolist(),
            "prediction_center_xy": predicted_center.tolist(),
            "target_center_xy": target_center.tolist(),
            "center_abs_error_xy": (
                predicted_center - target_center
            ).abs().tolist(),
        }
        if peaks is not None:
            row["attention_peak_xy"] = peaks[local_index].tolist()
        if entropy is not None:
            row["attention_entropy"] = float(entropy[source_index])
        if log_wh is not None:
            row["predicted_log_wh"] = log_wh[local_index].tolist()
        per_row.append(row)
    report["per_row"] = per_row
    report["worst_rows"] = sorted(
        per_row,
        key=lambda row: (row["iou"], -sum(row["center_abs_error_xy"])),
    )[:10]
    return report


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
    all_train_records = [
        record for record in all_train_records
        if source_loss_active(record, "bbox")
    ]
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
    selected_record_ids = [
        _probe_record_id(record, index)
        for index, record in enumerate(selected_records)
    ]
    feasibility = audit_bbox_probe_feasibility(
        selected_records,
        overfit_cfg["data"]["root"],
    )
    if feasibility["status"] != "PASS":
        return {
            "status": "FAIL",
            "reason": "bbox probe is not a deterministic pre-action mapping",
            "rows": rows,
            "selected_record_ids": selected_record_ids,
            "test_rows_read": 0,
            "bbox_geometry": geometry,
            "pre_action_feasibility": feasibility,
        }

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
    bbox_prior_report = None
    if model.action_head.bbox_size_parameterization == "log_space":
        bbox_prior_report = bbox_log_size_prior(
            selected_records,
            overfit_cfg["data"]["root"],
        )
        model.action_head.initialize_bbox_size_prior(
            bbox_prior_report["log_wh"]
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
    if hasattr(model.action_head, "coordinate_projection"):
        trainable_modules.append(model.action_head.coordinate_projection)
    parameters = []
    seen = set()
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))

    force_fp32_grounding = bool(
        overfit_cfg["train"].get("bbox_overfit_fp32_grounding", False)
    )
    disable_dropout = bool(
        overfit_cfg["train"].get("bbox_overfit_disable_dropout", False)
    )
    require_finite_gradients = bool(
        overfit_cfg["train"].get(
            "bbox_overfit_require_finite_gradients", False,
        )
    )
    initial = _evaluate_bbox_overfit(
        model,
        loader,
        "cuda",
        force_fp32_grounding=force_fp32_grounding,
        record_ids=selected_record_ids,
    )
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
    scaler = None if force_fp32_grounding else torch.amp.GradScaler("cuda")
    l1_ratio = float(overfit_cfg["loss"].get("bbox_l1_ratio", 5.0))
    giou_ratio = float(overfit_cfg["loss"].get("bbox_giou_ratio", 2.0))
    bbox_loss_mode = str(overfit_cfg["loss"].get("bbox_loss", "mse"))
    attention_ratio = float(
        overfit_cfg["loss"].get("bbox_attention_ratio", 0.0)
    )
    if attention_ratio < 0.0:
        raise ValueError("bbox_attention_ratio must be non-negative")

    def registered_bbox_terms(prediction, batch, row_mask):
        target = batch["bbox"].to("cuda")[row_mask]
        if bbox_loss_mode == "detr_log_size_giou":
            if "bbox_log_wh" not in prediction:
                raise KeyError(
                    "detr_log_size_giou requires unclamped bbox_log_wh"
                )
            centre_l1, log_size, giou = detr_bbox_log_size_loss_terms(
                prediction["bbox_cxcywh"][row_mask],
                prediction["bbox_log_wh"][row_mask],
                target,
            )
            attention_kl = centre_l1.detach() * 0.0
            if attention_ratio > 0.0:
                attention_kl = bbox_attention_kl_loss(
                    prediction["bbox_attention_weights"][row_mask],
                    prediction["bbox_spatial_coords"][row_mask],
                    prediction["bbox_spatial_mask"][row_mask],
                    target,
                )
            return (
                centre_l1 + log_size,
                giou,
                centre_l1,
                log_size,
                attention_kl,
            )
        l1, giou = detr_bbox_loss_terms(
            prediction["bbox_cxcywh"][row_mask],
            target,
        )
        zero = l1.detach() * 0.0
        return l1, giou, l1, zero, zero

    loss_history = []
    first_gradient_norms = {}
    completed_steps = 0
    attempted_steps = 0
    nonfinite_gradient_steps = 0
    max_attempts = steps + max(10, steps // 5)
    intermediate_eval_steps = {
        int(value)
        for value in overfit_cfg["train"].get(
            "bbox_overfit_full_eval_steps", [],
        )
    }
    if any(value <= 0 or value >= steps for value in intermediate_eval_steps):
        raise ValueError(
            "bbox_overfit_full_eval_steps must be strictly between 0 and steps"
        )
    if intermediate_eval_steps and not disable_dropout:
        raise ValueError(
            "full-set bbox trajectories require bbox_overfit_disable_dropout"
        )
    full_set_trajectory = [{"optimizer_step": 0, **initial}]
    model.eval() if disable_dropout else model.train()
    model.encoder.eval()
    while completed_steps < steps and attempted_steps < max_attempts:
        for batch in loader:
            attempted_steps += 1
            optimizer.zero_grad(set_to_none=True)
            if force_fp32_grounding:
                prediction = model.forward_bbox(
                    batch,
                    force_fp32_grounding=True,
                )
                row_mask = (batch["bbox_mask"].view(-1) > 0).to("cuda")
                l1, giou, centre_l1, log_size, attention_kl = registered_bbox_terms(
                    prediction, batch, row_mask,
                )
                loss = (
                    l1_ratio * l1
                    + giou_ratio * giou
                    + attention_ratio * attention_kl
                )
            else:
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction = model.forward_bbox(batch)
                    row_mask = (batch["bbox_mask"].view(-1) > 0).to("cuda")
                    l1, giou, centre_l1, log_size, attention_kl = registered_bbox_terms(
                        prediction, batch, row_mask,
                    )
                    loss = (
                        l1_ratio * l1
                        + giou_ratio * giou
                        + attention_ratio * attention_kl
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError("bbox micro-overfit loss is non-finite")
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            finite_gradients = all(
                torch.isfinite(parameter.grad).all()
                for parameter in parameters
                if parameter.grad is not None
            )
            if not finite_gradients:
                nonfinite_gradient_steps += 1
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if attempted_steps >= max_attempts:
                    break
                continue
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
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            history_row = {
                "step": completed_steps,
                "optimizer_step": completed_steps + 1,
                "measurement": "pre_update_batch_forward",
                "total": float(loss.detach()),
                "l1": float(l1.detach()),
                "center_l1": float(centre_l1.detach()),
                "log_size_smooth_l1": float(log_size.detach()),
                "giou": float(giou.detach()),
                "attention_kl": float(attention_kl.detach()),
            }
            if "bbox_log_wh" in prediction:
                row_log_wh = prediction["bbox_log_wh"][row_mask].detach().float()
                row_size = prediction["bbox_cxcywh"][row_mask, 2:].detach().float()
                history_row.update({
                    "predicted_log_wh_mean": row_log_wh.mean(dim=0).tolist(),
                    "predicted_log_wh_std": row_log_wh.std(
                        dim=0, unbiased=False,
                    ).tolist(),
                    "predicted_internal_wh_mean": row_size.mean(dim=0).tolist(),
                    "predicted_internal_wh_min": row_size.min(dim=0).values.tolist(),
                })
            loss_history.append(history_row)
            completed_steps += 1
            if completed_steps in intermediate_eval_steps:
                full_set_trajectory.append({
                    "optimizer_step": completed_steps,
                    **_evaluate_bbox_overfit(
                        model,
                        loader,
                        "cuda",
                        force_fp32_grounding=force_fp32_grounding,
                        record_ids=selected_record_ids,
                    ),
                })
            if completed_steps >= steps:
                break

    if completed_steps != steps:
        raise FloatingPointError(
            "bbox micro-overfit could not complete the registered finite-gradient "
            f"steps ({completed_steps}/{steps}, attempts={attempted_steps})"
        )

    final = _evaluate_bbox_overfit(
        model,
        loader,
        "cuda",
        force_fp32_grounding=force_fp32_grounding,
        record_ids=selected_record_ids,
    )
    full_set_trajectory.append({"optimizer_step": steps, **final})
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
    minimum_predicted_size = overfit_cfg["train"].get(
        "bbox_overfit_min_predicted_size"
    )
    if minimum_predicted_size is not None:
        minimum_predicted_size = float(minimum_predicted_size)
        public_size_min = final["prediction_coordinate_min"][2:]
        internal_size_min = final.get("internal_size_min", public_size_min)
        checks["decoded_sizes_above_registered_minimum"] = (
            min(public_size_min) >= minimum_predicted_size
            and min(internal_size_min) >= minimum_predicted_size
        )
        log_values = (
            final.get("predicted_log_wh_min", [])
            + final.get("predicted_log_wh_max", [])
        )
        checks["log_size_outputs_finite"] = (
            len(log_values) == 4
            and all(math.isfinite(float(value)) for value in log_values)
        )
    if attention_ratio > 0.0:
        checks["attention_kl_decreased"] = (
            final["bbox_attention_kl"] < initial["bbox_attention_kl"]
        )
    updates = {
        "bbox": float((bbox_probe.detach() - probe_before["bbox"]).float().norm()),
        "grounding_adapter": float(
            (grounding_probe.detach() - probe_before["grounding_adapter"])
            .float().norm()
        ),
    }
    checks["bbox_update_nonzero"] = updates["bbox"] > 0.0
    checks["grounding_update_nonzero"] = updates["grounding_adapter"] > 0.0
    if require_finite_gradients:
        checks["all_gradient_steps_finite"] = nonfinite_gradient_steps == 0
    optimizer_trace = [
        row for row in loss_history
        if row["optimizer_step"] % 10 == 0
    ]
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "rows": rows,
        "selected_record_ids": selected_record_ids,
        "steps": steps,
        "attempted_steps": attempted_steps,
        "nonfinite_gradient_steps": nonfinite_gradient_steps,
        "fp32_grounding": force_fp32_grounding,
        "dropout_disabled": disable_dropout,
        "test_rows_read": 0,
        "bbox_geometry": geometry,
        "pre_action_feasibility": feasibility,
        "bbox_log_size_prior": bbox_prior_report,
        "bbox_loss_mode": bbox_loss_mode,
        "loss": {
            "first_window_mean": initial_loss,
            "last_window_mean": final_loss,
            "first_step": loss_history[0],
            "last_step": loss_history[-1],
        },
        "optimizer_trace": optimizer_trace,
        "full_set_trajectory": full_set_trajectory,
        "trajectory_contract": (
            "initial/final are full fixed-set evaluations; optimizer_trace rows "
            f"are the named pre-update batch forward for steps 10,20,...,{steps}"
        ),
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
            "min_predicted_size": minimum_predicted_size,
            "attention_ratio": attention_ratio,
            "full_eval_steps": sorted(intermediate_eval_steps),
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
    original_smoke_candidates = [
        record for record in all_train_records
        if source_loss_active(record, "action")
    ]
    smoke_records = select_smoke_records(
        original_smoke_candidates, rows, seed,
    )
    smoke_bbox_records = [
        record for record in smoke_records
        if source_loss_active(record, "bbox")
    ]
    smoke_bbox_geometry = audit_bbox_geometry(
        smoke_bbox_records,
        cfg["data"]["root"],
    )
    if smoke_bbox_geometry["valid_bbox_rows"] == 0:
        # Category coverage is chosen before select_smoke_records appends filler;
        # replacing the final filler row preserves that coverage.
        smoke_records[-1] = _select_valid_bbox_records(
            [
                record for record in all_train_records
                if source_loss_active(record, "bbox")
            ],
            cfg["data"]["root"],
            1,
            seed,
        )[0]
        smoke_bbox_geometry = audit_bbox_geometry(
            [
                record for record in smoke_records
                if source_loss_active(record, "bbox")
            ],
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
    autocast_dtype = _training_autocast_dtype(cfg)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=autocast_dtype == torch.float16
    )
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
    if hasattr(model.action_head, "coordinate_projection"):
        probes["coordinate_projection"] = next(
            model.action_head.coordinate_projection.parameters()
        )
    if float(cfg["loss"].get("bbox_attention_ratio", 0.0)) > 0.0:
        probes["grounding_attention"] = next(
            model.action_head.grounding_attention.parameters()
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
        with torch.autocast("cuda", dtype=autocast_dtype):
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
        pre_images += int(batch["pre_image_counts"].sum())
        post_images += int(batch["post_image_counts"].sum())
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
    if cfg.get("model", {}).get("bbox_size_parameterization") == "log_space":
        expected_shapes["bbox_log_wh"] = (batch_size, 2)
    if cfg.get("model", {}).get("bbox_grounding_mode") == "coordinate_softargmax":
        expected_shapes["bbox_attention_entropy"] = (batch_size,)
        expected_shapes["bbox_attention_weights"] = (
            batch_size,
            first_predictions["bbox_attention_weights"].shape[1],
        )
        expected_shapes["bbox_spatial_coords"] = (
            batch_size,
            first_predictions["bbox_spatial_coords"].shape[1],
            2,
        )
        expected_shapes["bbox_spatial_mask"] = (
            batch_size,
            first_predictions["bbox_spatial_mask"].shape[1],
        )
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
    resume_checkpoint: str | Path | None = None,
) -> dict:
    """Run the controlled recovery experiment without reading the test split.

    Training uses source-aware supplement inclusion, joint original-data
    coverage, and distribution-preserving recovery-aware batches. Primary
    validation intentionally retains the legacy v14 failure-stratified selector
    so the 500-row comparison set is identical; supplement validation is
    reported separately and cannot select the checkpoint.

    ``resume_checkpoint``, when given, must be an interrupted ``last.ckpt``
    from this exact same mini configuration; it is validated and restored
    (optimizer/scheduler/scaler/RNG/epoch position) before training resumes.
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
    review_overlay_root = mini_cfg["data"].get("review_overlay_dir")
    review_overlay = (
        ReviewOverlay.load(review_overlay_root)
        if review_overlay_root
        else None
    )

    all_train_records = load_gold_split(mini_cfg, "train")
    selected_train_records = select_recovery_aware_gold_subset(
        all_train_records, train_rows, seed,
    )
    all_val_records = load_gold_split(mini_cfg, "val")
    selected_val_records = stratified_gold_subsample(all_val_records, val_rows, seed)
    selected_bbox_geometry = {
        "train": audit_bbox_geometry(
            [
                record for record in selected_train_records
                if source_loss_active(record, "bbox")
            ],
            mini_cfg["data"]["root"],
        ),
        "validation": audit_bbox_geometry(
            [
                record for record in selected_val_records
                if source_loss_active(record, "bbox")
            ],
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
    resume_audit = None
    if resume_checkpoint is not None:
        resume_audit = _validate_resume_checkpoint(resume_checkpoint, mini_cfg)
        trainer.load_checkpoint(resume_checkpoint, resume_training=True)
    result = trainer.fit()
    if resume_audit is not None:
        result["resume_provenance"] = resume_audit
    if not result["checkpoints"]:
        raise AssertionError("mini training did not create a checkpoint")

    selection_rule = str(
        mini_cfg["train"].get("quality_selection_rule", LEGACY_OUTCOME_RULE)
    )
    quality_gates = controlled_quality_gates(result, selection_rule)
    checkpoint = Path(require_selected_checkpoint(result, quality_gates))
    trainer.load_checkpoint(checkpoint, resume_training=False)
    checkpoint_roundtrip = _verify_checkpoint_roundtrip(
        trainer, val_loader, checkpoint
    )
    supplement_validation = _supplement_validation_report(
        mini_cfg,
        components.processor,
        components.model,
        seed=seed,
    )
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
        "train_selector": (
            "source_aware_all_supplement_plus_joint_original_v1"
            if supplement_validation["enabled"]
            else "joint_proportional_v1"
        ),
        "validation_selector": "legacy_failure_stratified_v14_fixed",
        "validation_comparable_to_v14": review_overlay is None,
        "comparison_requires_baseline_reevaluation_on_overlay": (
            review_overlay is not None
        ),
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
        "supplement_train_rows": sum(
            view(record)[2].get("_source_dataset")
            == "retry_abort_supplement_v2"
            for record in selected_train_records
        ),
    }
    is_v2 = bool(mini_cfg["data"].get("recovery_transitions"))
    is_v2_1_or_later = _recovery_version_at_least(experiment_tag, 1)
    is_v2_2_or_later = _recovery_version_at_least(experiment_tag, 2)
    is_v2_3_or_later = _recovery_version_at_least(experiment_tag, 3)
    is_v2_4_or_later = _recovery_version_at_least(experiment_tag, 4)
    is_v2_5_or_later = _recovery_version_at_least(experiment_tag, 5)
    is_v2_6_or_later = _recovery_version_at_least(experiment_tag, 6)
    is_v2_7_or_later = _recovery_version_at_least(experiment_tag, 7)
    training_changes = [
        "joint proportional training subset coverage",
        "recovery-aware distribution-preserving physical batches",
        "sqrt inverse-frequency recovery-strategy class weights",
        "weights derived from selected training rows only",
    ]
    if review_overlay is not None:
        training_changes.append(
            "passed two-person review overlay on train and validation"
        )
    if supplement_validation["enabled"]:
        training_changes.extend([
            "all accepted RETRY/ABORT supplement rows included once in the mini",
            "supplement rows supervise only recovery strategy and recovery success",
            "supplement strategy routed from the pre-recovery failure state",
            "supplement validation reported separately from checkpoint selection",
        ])
    if is_v2:
        training_changes.extend([
            "executed action added only to the post-action stream",
            "binary needs-recovery plus attempted-row-only strategy supervision",
            "proper sparse recovery-transition stream for recovery success",
            (
                "spatial grounding bbox head with registered coordinate(5) plus GIoU(2)"
                if is_v2_2_or_later
                else "spatial grounding bbox head with SmoothL1 plus GIoU"
            ),
            "task-specific residual adapters and uncertainty loss weighting",
        ])
    if is_v2_1_or_later:
        training_changes.extend([
            "exact selected-row needs-recovery positive weighting",
            "bbox-specific residual adapter separate from action classification",
        ])
    if is_v2_2_or_later:
        training_changes.extend([
            "strict bbox validation with invalid targets masked only for localization",
            "internal centre-format bbox regression with stable public xywh output",
            "separately logged bbox coordinate-L1 and GIoU components",
        ])
    if is_v2_3_or_later:
        training_changes.extend([
            "explicit normalized 2D patch coordinates for localization",
            "attention soft-argmax bbox centre with learned width and height",
            "FP32 trainable grounding path with the frozen VLM kept in FP16",
        ])
    if is_v2_4_or_later:
        training_changes.extend([
            "train-only geometric width/height prior",
            "unclamped log-size SmoothL1 supervision with bounded exponential decode",
        ])
    if is_v2_5_or_later:
        training_changes.extend([
            "direct target-distribution supervision of spatial attention",
            "undropped grounding probabilities for the supervised attention map",
        ])
    if is_v2_6_or_later:
        training_changes.append(
            "higher-resolution bbox micro-overfit grid with extended optimizer trace"
        )
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

    selected_epoch = int(quality_gates["selected_epoch"])
    selected_history = next(
        row for row in history if int(row["epoch"]) == selected_epoch
    )
    unconstrained_checkpoint = str(result["checkpoints"][0])

    return {
        **result,
        "status": quality_gates["status"],
        "training_disposition": (
            "FULL_TRAINING_PERMITTED"
            if quality_gates["status"] == "PASS"
            else "STOP_BEFORE_FULL_TRAINING"
        ),
        "train_rows": len(train_loader.dataset),
        "val_rows": len(val_loader.dataset),
        "test_rows_read": 0,
        "loss_decreased": loss_decreased,
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "best_metric": float(selected_history[result["early_stop_metric"]]),
        "best_checkpoint": str(checkpoint),
        "selected_epoch": selected_epoch,
        "selection_rule": selection_rule,
        "unconstrained_best_metric": float(result["best_metric"]),
        "unconstrained_best_checkpoint": unconstrained_checkpoint,
        "class_weights": components.class_weight_report,
        "sampling": sampling_report,
        "quality_gates": quality_gates,
        "train_distribution": label_distribution(selected_train_records),
        "validation_distribution": label_distribution(selected_val_records),
        "source_validation": {
            "primary_original_gold": {
                "rows": len(selected_val_records),
                "checkpoint_selection_source": True,
                "selected_epoch_metrics": selected_history,
            },
            "supplement_retry_abort": supplement_validation,
            "combined": {
                "reported": False,
                "reason": (
                    "secondary combined metrics are intentionally deferred; "
                    "source-separated validation prevents frequency mixing"
                ),
            },
        },
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
            "baseline": (
                "v2.7 selected checkpoint re-evaluated on the same review overlay"
                if review_overlay is not None
                else "kaggle-gold-v14"
            ),
            "experiment_tag": experiment_tag.lower(),
            "quality_selection_rule": selection_rule,
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
                        if is_v2_2_or_later else []
                    ),
                    *(
                        [
                            "coordinate-aware attention and soft-argmax centre",
                            "FP32 trainable localization branch",
                            "manual montage review of the fixed overfit rows",
                        ]
                        if is_v2_3_or_later else []
                    ),
                    *(
                        [
                            "train-only bbox size prior",
                            "log-space width/height objective with non-saturating direct gradient",
                        ]
                        if is_v2_4_or_later else []
                    ),
                    *(
                        [
                            "duplicate/conflicting pre-action bbox feasibility audit",
                            "target-distribution patch-attention supervision",
                        ]
                        if is_v2_5_or_later else []
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
                    if is_v2_2_or_later else []
                ),
                *(
                    [
                        "bbox attention entropy",
                        "finite-gradient attempt and rejection counts",
                    ]
                    if is_v2_3_or_later else []
                ),
                *(
                    [
                        "internal log-size distributions and fixed-step optimizer trace",
                        "positive decoded-size collapse gate",
                    ]
                    if is_v2_4_or_later else []
                ),
                *(
                    [
                        "per-row bbox predictions, centres, attention peaks, and IoU",
                        "full fixed-set evaluations at registered optimizer steps",
                    ]
                    if is_v2_5_or_later else []
                ),
                *(
                    [
                        "all-gates eligibility before outcome-MCC checkpoint ranking",
                        "selected-checkpoint identity synchronized across report and exports",
                    ]
                    if is_v2_7_or_later else []
                ),
                "all mini epoch checkpoints retained",
            ],
        },
        "review_overlay": (
            review_overlay.provenance()
            if review_overlay is not None
            else {"enabled": False}
        ),
    }


def _supplement_validation_report(
    cfg: dict,
    processor,
    model,
    *,
    seed: int,
) -> dict:
    """Evaluate the complete supplement validation source separately."""
    sources = load_gold_split_sources(cfg, "val")
    records = sources.get("retry_abort_supplement_v2", [])
    if not records:
        return {"enabled": False, "rows": 0}
    loader = build_gold_dataloader(
        cfg,
        "val",
        processor,
        records=records,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 2),
        seed=seed,
        trajectory_records=records,
    )
    predictions = collect_predictions(model, loader, "cuda")
    metrics = compute_metrics(predictions)
    diagnostics = build_diagnostics(predictions)
    permitted_metrics = {
        name: value
        for name, value in metrics.items()
        if name.startswith("strategy_attempted_")
        or name.startswith("recovery_outcome_")
    }
    return {
        "enabled": True,
        "source": "retry_abort_supplement_v2",
        "rows": len(records),
        "test_rows_read": 0,
        "metrics": permitted_metrics,
        "diagnostics": {
            "recovery_strategy_attempted_only": diagnostics[
                "recovery_strategy_attempted_only"
            ],
            "recovery_outcome_prediction": diagnostics[
                "recovery_outcome_prediction"
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
    """Re-score an old checkpoint on validation, with any configured overlay."""
    review_overlay_root = cfg["data"].get("review_overlay_dir")
    review_overlay = (
        ReviewOverlay.load(review_overlay_root)
        if review_overlay_root
        else None
    )
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
        "validation_comparable_to_original_v14": review_overlay is None,
        "val_rows": len(val_loader.dataset),
        "test_rows_read": 0,
        "review_overlay": (
            review_overlay.provenance()
            if review_overlay is not None
            else {"enabled": False}
        ),
        "metrics": compute_metrics(predictions),
        "diagnostics": build_diagnostics(predictions),
    }


@torch.no_grad()
def _verify_checkpoint_roundtrip(
    trainer: Trainer, val_loader, checkpoint: str | Path,
) -> bool:
    """Perturb critical heads/adapters, reload, and prove their logits return."""
    trainer.model.eval()
    batch = next(iter(val_loader))
    with torch.autocast("cuda", dtype=trainer.autocast_dtype):
        original = trainer.model(batch)
        keys = ["action_type", "bbox", "recovery", "recovery_outcome"]
        if "bbox_cxcywh" in original:
            keys.append("bbox_cxcywh")
        if "bbox_log_wh" in original:
            keys.append("bbox_log_wh")
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
    trainer.load_checkpoint(checkpoint, resume_training=False)
    trainer.model.eval()
    with torch.autocast("cuda", dtype=trainer.autocast_dtype):
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
