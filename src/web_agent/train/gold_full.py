"""Resume-safe full training for the reviewed Gold dataset.

The locked test split is never loaded here.  Checkpoint selection uses the full
original-Gold validation split; the RETRY/ABORT supplement validation split is
reported separately and cannot influence selection.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from web_agent.data.bbox_audit import audit_bbox_geometry
from web_agent.data.gold_dataloader import (
    build_gold_dataloader,
    load_gold_split,
    source_loss_active,
)
from web_agent.data.gold_sampling import label_distribution
from web_agent.data.recovery_transitions import recovery_class_audit
from web_agent.data.review_overlay import ReviewOverlay
from web_agent.train.gold_stages import (
    _supplement_validation_report,
    _validate_resume_checkpoint,
    _verify_checkpoint_roundtrip,
    build_gold_components,
)
from web_agent.train.resume import sha256_file
from web_agent.train.selection import (
    controlled_quality_gates,
    require_selected_checkpoint,
)
from web_agent.train.trainer import Trainer
from web_agent.utils.seed import set_seed


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _split_hashes(cfg: dict) -> dict[str, str]:
    root = Path(cfg["data"]["root"])
    return {
        split: sha256_file(root / cfg["data"][f"{split}_json"])
        for split in ("train", "val")
    }


def run_gold_full(
    cfg: dict,
    *,
    processor=None,
    epochs: int = 15,
    seed: int = 42,
    checkpoint_root: str | Path,
    metrics_csv: str | Path,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train every reviewed train row and select only on original validation."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    set_seed(seed)
    full_cfg = deepcopy(cfg)
    full_cfg["name"] = f"{cfg['name']}_FULL_SEED{seed}"
    full_cfg["seeds"] = [seed]
    full_cfg["train"]["epochs"] = epochs
    full_cfg["train"]["metrics_csv"] = str(Path(metrics_csv))
    full_cfg["train"]["keep_top_k"] = epochs
    full_cfg["output"]["checkpoint_dir"] = str(Path(checkpoint_root))

    metrics_path = Path(metrics_csv)
    if resume_checkpoint is None and metrics_path.exists():
        raise FileExistsError(
            "fresh full run refuses an existing metrics CSV; choose a new run "
            f"directory or resume its checkpoint: {metrics_path}"
        )
    resume_audit = (
        _validate_resume_checkpoint(resume_checkpoint, full_cfg)
        if resume_checkpoint is not None
        else {"status": "NOT_REQUESTED"}
    )

    train_records = load_gold_split(full_cfg, "train")
    val_records = load_gold_split(full_cfg, "val")
    components = build_gold_components(
        full_cfg,
        processor,
        train_records=train_records,
        trajectory_records=train_records,
    )
    workers = int(full_cfg["data"].get("num_workers", 2))
    train_loader = build_gold_dataloader(
        full_cfg,
        "train",
        components.processor,
        records=train_records,
        shuffle=False,
        num_workers=workers,
        seed=seed,
        recovery_aware=True,
        trajectory_records=train_records,
    )
    val_loader = build_gold_dataloader(
        full_cfg,
        "val",
        components.processor,
        records=val_records,
        shuffle=False,
        num_workers=workers,
        seed=seed,
        trajectory_records=val_records,
    )
    trainer = Trainer(
        components.model,
        components.loss_fn,
        full_cfg,
        train_loader,
        val_loader,
        train_sampler=train_loader.batch_sampler,
    )
    if resume_checkpoint is not None:
        trainer.load_checkpoint(resume_checkpoint, resume_training=True)

    result = trainer.fit()
    if not result["history"]:
        raise AssertionError("full training produced no completed epoch")
    expected_epochs = list(range(len(result["history"])))
    completed_epochs = [int(row["epoch"]) for row in result["history"]]
    if completed_epochs != expected_epochs:
        raise AssertionError(
            "full history must contain consecutive epochs from zero; "
            f"found {completed_epochs}"
        )

    selection_rule = str(full_cfg["train"]["quality_selection_rule"])
    quality = controlled_quality_gates(result, selection_rule)
    checkpoint = Path(require_selected_checkpoint(result, quality))
    trainer.load_checkpoint(checkpoint, resume_training=False)
    checkpoint_roundtrip = _verify_checkpoint_roundtrip(
        trainer,
        val_loader,
        checkpoint,
    )
    selected_epoch = int(quality["selected_epoch"])
    selected_metrics = next(
        row for row in result["history"]
        if int(row["epoch"]) == selected_epoch
    )
    unconstrained_epoch = int(result["best_epochs"]["outcome_mcc"]["epoch"])
    unconstrained_metrics = next(
        row for row in result["history"]
        if int(row["epoch"]) == unconstrained_epoch
    )
    supplement_validation = _supplement_validation_report(
        full_cfg,
        components.processor,
        components.model,
        seed=seed,
    )
    loss_decreased = (
        len(result["history"]) == 1
        or float(result["history"][-1]["train_nominal_weighted_loss"])
        < float(result["history"][0]["train_nominal_weighted_loss"])
    )
    review_overlay_root = full_cfg["data"].get("review_overlay_dir")
    review_overlay = (
        ReviewOverlay.load(review_overlay_root)
        if review_overlay_root
        else None
    )
    bbox_geometry = {
        "train": audit_bbox_geometry(
            [
                record for record in train_records
                if source_loss_active(record, "bbox")
            ],
            full_cfg["data"]["root"],
        ),
        "validation": audit_bbox_geometry(
            [
                record for record in val_records
                if source_loss_active(record, "bbox")
            ],
            full_cfg["data"]["root"],
        ),
    }
    early_stopped = completed_epochs[-1] < epochs - 1
    report = {
        **result,
        "stage": "full",
        "status": quality["status"],
        "training_disposition": (
            "FULL_TRAINING_COMPLETE_SELECT_VALIDATION_CHECKPOINT"
            if quality["status"] == "PASS"
            else "FULL_TRAINING_COMPLETE_BUT_QUALITY_GATES_FAILED"
        ),
        "full_training_finished": True,
        "requested_epochs": epochs,
        "completed_epochs": completed_epochs,
        "early_stopped": early_stopped,
        "seed": seed,
        "train_rows": len(train_loader.dataset),
        "val_rows": len(val_loader.dataset),
        "test_rows_read": 0,
        "selection_rule": selection_rule,
        "selected_epoch": selected_epoch,
        "best_metric": float(selected_metrics[trainer.early_stop_metric]),
        "best_checkpoint": str(checkpoint),
        "unconstrained_best_metric": float(
            unconstrained_metrics[trainer.early_stop_metric]
        ),
        "unconstrained_best_checkpoint": result["epoch_checkpoints"][
            str(unconstrained_epoch)
        ],
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "quality_gates": quality,
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "loss_decreased": loss_decreased,
        "class_weights": components.class_weight_report,
        "train_distribution": label_distribution(train_records),
        "validation_distribution": label_distribution(val_records),
        "bbox_geometry": bbox_geometry,
        "recovery_transition_reports": {
            "train": train_loader.dataset.transition_report,
            "validation": val_loader.dataset.transition_report,
        },
        "recovery_class_audit": {
            "train": recovery_class_audit(train_records),
            "validation": recovery_class_audit(val_records),
        },
        "source_validation": {
            "primary_original_gold": {
                "rows": len(val_loader.dataset),
                "checkpoint_selection_source": True,
                "selected_epoch_metrics": selected_metrics,
            },
            "supplement_retry_abort": supplement_validation,
        },
        "review_overlay": (
            review_overlay.provenance()
            if review_overlay is not None
            else {"enabled": False}
        ),
        "resume_audit": resume_audit,
        "experiment_control": {
            "config_sha256": _canonical_sha256(full_cfg),
            "development_split_sha256": _split_hashes(full_cfg),
            "checkpoint_selection_source": "original_gold_validation_only",
            "supplement_validation_selects_checkpoint": False,
            "locked_test_read": False,
            "all_training_rows_used": True,
            "checkpoint_every_steps": trainer.checkpoint_every_steps,
            "mixed_precision": trainer.mixed_precision,
        },
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    return report
