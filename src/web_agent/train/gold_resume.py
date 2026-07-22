"""Time-safe continuation of an interrupted v2.7 controlled mini experiment."""

from __future__ import annotations

from copy import deepcopy
import gc
import math
from pathlib import Path
import shutil
from typing import Any

import torch

from web_agent.data.bbox_audit import audit_bbox_geometry
from web_agent.data.gold_dataloader import build_gold_dataloader, load_gold_split
from web_agent.data.gold_dataset import view
from web_agent.data.gold_sampling import (
    label_distribution,
    recovery_aware_batch_indices,
    select_recovery_aware_gold_subset,
)
from web_agent.data.gold_dataloader import stratified_gold_subsample
from web_agent.data.recovery_transitions import recovery_class_audit
from web_agent.train.gold_stages import (
    _verify_checkpoint_roundtrip,
    build_gold_components,
)
from web_agent.train.resume import (
    discover_epoch_checkpoints,
    load_epoch_metrics_csv,
    merge_epoch_histories,
    sha256_file,
    validate_prior_history,
    validate_resume_checkpoint_metadata,
)
from web_agent.train.selection import (
    ALL_GATES_THEN_OUTCOME_RULE,
    controlled_quality_gates,
    require_selected_checkpoint,
)
from web_agent.train.trainer import (
    Trainer,
    _best_epochs,
    build_diagnostics,
    collect_predictions,
    compute_metrics,
)
from web_agent.utils.seed import set_seed


_REVALIDATED_GATE_METRICS = (
    "outcome_mcc",
    "action_acc",
    "needs_recovery_macro_f1",
    "needs_recovery_majority_macro_f1",
    "strategy_attempted_macro_f1",
    "strategy_attempted_majority_macro_f1",
    "recovery_outcome_mcc",
    "bbox_mean_iou",
    "bbox_recall_iou50",
    "outcome_ece",
)


def _mini_config(
    cfg: dict,
    *,
    total_epochs: int,
    metrics_csv: str | Path | None,
    checkpoint_root: str | Path | None,
) -> tuple[dict, str]:
    """Reproduce the exact config transformation used by ``run_gold_mini``."""
    mini_cfg = deepcopy(cfg)
    mini_cfg["train"]["epochs"] = total_epochs
    mini_cfg["train"]["early_stop_patience"] = max(
        total_epochs, mini_cfg["train"].get("early_stop_patience", 0)
    )
    experiment_tag = str(
        mini_cfg["train"].get("controlled_experiment_tag", "recovery_v1")
    ).upper()
    mini_cfg["name"] = f"{cfg['name']}_MINI_{experiment_tag}"
    mini_cfg["train"]["metrics_csv"] = str(
        metrics_csv
        or f"results/gold_mini_{experiment_tag.lower()}_metrics.csv"
    )
    mini_cfg["train"]["keep_top_k"] = max(
        total_epochs, int(mini_cfg["train"].get("keep_top_k", 3))
    )
    if checkpoint_root is not None:
        mini_cfg["output"]["checkpoint_dir"] = str(checkpoint_root)
    return mini_cfg, experiment_tag.lower()


def _expected_steps_per_epoch(cfg: dict, train_rows: int) -> int:
    physical_batches = math.ceil(train_rows / int(cfg["optim"]["batch_size"]))
    return math.ceil(physical_batches / int(cfg["optim"].get("grad_accum", 1)))


def _sampling_report(records: list[dict], cfg: dict, seed: int) -> dict[str, Any]:
    scheduled = recovery_aware_batch_indices(
        records, int(cfg["optim"]["batch_size"]), seed=seed, epoch=0
    )
    scheduled_rows = [index for batch in scheduled for index in batch]
    attempted_batches = sum(
        any(view(records[index])[1].get("recovery_success") is not None for index in batch)
        for batch in scheduled
    )
    return {
        "train_selector": "joint_proportional_v1",
        "validation_selector": "legacy_failure_stratified_v14_fixed",
        "validation_comparable_to_v14": True,
        "batch_sampler": "recovery_aware_no_oversampling_v1",
        "physical_batch_size": int(cfg["optim"]["batch_size"]),
        "batches": len(scheduled),
        "batches_with_attempted_recovery": attempted_batches,
        "attempted_batch_coverage": attempted_batches / max(len(scheduled), 1),
        "unique_train_rows_scheduled": len(set(scheduled_rows)),
        "duplicate_train_rows_scheduled": len(scheduled_rows) - len(set(scheduled_rows)),
    }


def _revalidation_differences(
    historical: dict[str, Any], current: dict[str, Any]
) -> dict[str, float]:
    return {
        metric: abs(float(historical[metric]) - float(current[metric]))
        for metric in _REVALIDATED_GATE_METRICS
    }


def _audit_prior_checkpoints(
    checkpoint_paths: dict[str, str],
    *,
    expected_config: dict,
    prior_history: list[dict[str, Any]],
    total_epochs: int,
    steps_per_epoch: int,
) -> dict[str, dict[str, Any]]:
    """Validate every retained epoch artifact, releasing tensors between files."""
    audits: dict[str, dict[str, Any]] = {}
    for epoch_text, checkpoint_path in sorted(
        checkpoint_paths.items(), key=lambda item: int(item[0])
    ):
        epoch = int(epoch_text)
        checkpoint_state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        audits[epoch_text] = {
            **validate_resume_checkpoint_metadata(
                checkpoint_state,
                expected_config=expected_config,
                prior_history=prior_history[: epoch + 1],
                resume_epoch=epoch,
                total_epochs=total_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "checkpoint": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        }
        del checkpoint_state
        gc.collect()
    return audits


def resume_gold_mini(
    cfg: dict,
    *,
    prior_metrics_csv: str | Path,
    prior_checkpoint_dir: str | Path,
    resume_checkpoint: str | Path,
    expected_resume_sha256: str,
    selected_checkpoint_output: str | Path,
    processor=None,
    train_rows: int = 5_000,
    val_rows: int = 500,
    total_epochs: int = 5,
    resume_epoch: int = 3,
    seed: int = 42,
    resumed_metrics_csv: str | Path | None = None,
    checkpoint_output_root: str | Path | None = None,
    revalidation_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Continue one interrupted run from a completed epoch checkpoint.

    The old CSV and checkpoints must describe exactly epochs 0..``resume_epoch``.
    No row from an independent run is accepted.  The function restores the
    optimization state, trains only the missing epoch(s), applies the registered
    selector to the combined same-lineage history, reloads that exact selected
    artifact, and validates it again on the locked validation subset.
    """
    if str(cfg["train"].get("quality_selection_rule")) != ALL_GATES_THEN_OUTCOME_RULE:
        raise ValueError("resume requires the registered v2.7 constrained selector")
    if int(total_epochs) != 5 or int(resume_epoch) != 3:
        raise ValueError("the audited v2.7 recovery plan requires epoch 3 -> 5 total")

    mini_cfg, experiment_tag = _mini_config(
        cfg,
        total_epochs=total_epochs,
        metrics_csv=resumed_metrics_csv,
        checkpoint_root=checkpoint_output_root,
    )
    prior_history = validate_prior_history(
        load_epoch_metrics_csv(prior_metrics_csv),
        resume_epoch=resume_epoch,
        total_epochs=total_epochs,
    )
    prior_checkpoints = discover_epoch_checkpoints(
        prior_checkpoint_dir, range(resume_epoch + 1)
    )
    resume_path = Path(resume_checkpoint).resolve()
    expected_resume_path = Path(prior_checkpoints[str(resume_epoch)]).resolve()
    if resume_path != expected_resume_path:
        raise ValueError(
            "resume checkpoint is not the retained checkpoint mapped to the final "
            f"completed epoch: {resume_path} != {expected_resume_path}"
        )

    resume_sha256 = sha256_file(resume_path)
    if resume_sha256.lower() != expected_resume_sha256.lower():
        raise ValueError(
            "resume checkpoint SHA-256 does not match the audited artifact: "
            f"{resume_sha256} != {expected_resume_sha256}"
        )

    expected_steps = _expected_steps_per_epoch(mini_cfg, train_rows)
    prior_checkpoint_audit = _audit_prior_checkpoints(
        prior_checkpoints,
        expected_config=mini_cfg,
        prior_history=prior_history,
        total_epochs=total_epochs,
        steps_per_epoch=expected_steps,
    )
    provenance_check = prior_checkpoint_audit[str(resume_epoch)]
    rng_state_available = bool(provenance_check["rng_state_available"])

    # Match the original deterministic row selectors and physical batch schedule.
    set_seed(seed)
    all_train_records = load_gold_split(mini_cfg, "train")
    selected_train_records = select_recovery_aware_gold_subset(
        all_train_records, train_rows, seed
    )
    all_val_records = load_gold_split(mini_cfg, "val")
    selected_val_records = stratified_gold_subsample(all_val_records, val_rows, seed)
    selected_bbox_geometry = {
        "train": audit_bbox_geometry(selected_train_records, mini_cfg["data"]["root"]),
        "validation": audit_bbox_geometry(selected_val_records, mini_cfg["data"]["root"]),
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
    if trainer.steps_per_epoch != expected_steps:
        raise AssertionError(
            f"loader has {trainer.steps_per_epoch} optimizer steps per epoch, "
            f"expected {expected_steps}"
        )
    trainer.load_checkpoint(resume_path, resume_training=True)
    if trainer.start_epoch != resume_epoch + 1:
        raise AssertionError(
            f"checkpoint resumes at epoch {trainer.start_epoch}, expected {resume_epoch + 1}"
        )
    if trainer.global_step != provenance_check["expected_step"]:
        raise AssertionError("restored optimizer step disagrees with audited metadata")
    if not rng_state_available:
        # The historical v2.7 checkpoint predates RNG-state checkpointing.  Use
        # the registered experiment seed and disclose this non-bitwise fallback.
        set_seed(seed)

    resumed_result = trainer.fit()
    expected_resumed_epochs = list(range(resume_epoch + 1, total_epochs))
    actual_resumed_epochs = [int(row["epoch"]) for row in resumed_result["history"]]
    if actual_resumed_epochs != expected_resumed_epochs:
        raise AssertionError(
            f"resume trained epochs {actual_resumed_epochs}, expected {expected_resumed_epochs}"
        )
    history = merge_epoch_histories(
        prior_history, resumed_result["history"], total_epochs=total_epochs
    )
    epoch_checkpoints = {
        **prior_checkpoints,
        **{str(key): str(value) for key, value in resumed_result["epoch_checkpoints"].items()},
    }
    if sorted(map(int, epoch_checkpoints)) != list(range(total_epochs)):
        raise AssertionError("one or more completed epoch checkpoints are missing")

    combined = {
        "early_stop_metric": trainer.early_stop_metric,
        "history": history,
        "best_epochs": _best_epochs(history),
        "epoch_checkpoints": epoch_checkpoints,
    }
    quality = controlled_quality_gates(
        combined, mini_cfg["train"]["quality_selection_rule"]
    )
    if quality["status"] != "PASS":
        raise AssertionError(
            "the five-epoch history has no epoch passing every registered quality gate"
        )
    selected_source = Path(require_selected_checkpoint(combined, quality))
    selected_output = Path(selected_checkpoint_output)
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    if selected_source.resolve() != selected_output.resolve():
        shutil.copy2(selected_source, selected_output)
    epoch_checkpoints[str(quality["selected_epoch"])] = str(selected_output)
    combined["epoch_checkpoints"] = epoch_checkpoints
    quality = controlled_quality_gates(
        combined, mini_cfg["train"]["quality_selection_rule"]
    )
    checkpoint = Path(require_selected_checkpoint(combined, quality))
    combined["checkpoints"] = [
        epoch_checkpoints[str(int(row["epoch"]))]
        for row in sorted(
            history, key=lambda row: float(row[trainer.early_stop_metric]), reverse=True
        )
    ]

    trainer.load_checkpoint(checkpoint, resume_training=False)
    checkpoint_roundtrip = _verify_checkpoint_roundtrip(
        trainer, val_loader, checkpoint
    )
    predictions = collect_predictions(trainer.model, val_loader, trainer.device)
    selected_metrics = compute_metrics(predictions)
    selected_epoch = int(quality["selected_epoch"])
    selected_history = next(
        row for row in history if int(row["epoch"]) == selected_epoch
    )
    metric_differences = _revalidation_differences(
        selected_history, selected_metrics
    )
    mismatched = {
        key: value
        for key, value in metric_differences.items()
        if value > revalidation_tolerance
    }
    if mismatched:
        raise AssertionError(
            "selected-checkpoint validation disagrees with its historical row: "
            f"{mismatched}"
        )

    unconstrained_epoch = int(combined["best_epochs"]["outcome_mcc"]["epoch"])
    unconstrained_history = next(
        row for row in history if int(row["epoch"]) == unconstrained_epoch
    )
    diagnostics = [
        {
            "epoch": int(row["epoch"]),
            "source": "prior_metrics_csv_only",
            "details_available": False,
        }
        for row in prior_history
    ] + [dict(row, source="resumed_epoch_validation") for row in resumed_result["diagnostics"]]
    diagnostics = [
        (
            {
                "epoch": selected_epoch,
                "source": "selected_checkpoint_revalidation",
                "details_available": True,
                **build_diagnostics(predictions),
            }
            if int(row["epoch"]) == selected_epoch
            else row
        )
        for row in diagnostics
    ]
    diagnostics.sort(key=lambda row: int(row["epoch"]))
    loss_decreased = (
        float(history[-1]["train_nominal_weighted_loss"])
        < float(history[0]["train_nominal_weighted_loss"])
    )
    if not loss_decreased:
        raise AssertionError("nominal weighted loss did not decrease across five epochs")

    report = {
        **combined,
        "status": quality["status"],
        "train_rows": len(train_loader.dataset),
        "val_rows": len(val_loader.dataset),
        "test_rows_read": 0,
        "diagnostics": diagnostics,
        "selection_rule": mini_cfg["train"]["quality_selection_rule"],
        "selected_epoch": selected_epoch,
        "best_metric": float(selected_history[trainer.early_stop_metric]),
        "best_checkpoint": str(checkpoint),
        "unconstrained_best_metric": float(
            unconstrained_history[trainer.early_stop_metric]
        ),
        "unconstrained_best_checkpoint": epoch_checkpoints[str(unconstrained_epoch)],
        "quality_gates": quality,
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "loss_decreased": loss_decreased,
        "class_weights": components.class_weight_report,
        "sampling": _sampling_report(selected_train_records, mini_cfg, seed),
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
            "experiment_tag": experiment_tag,
            "quality_selection_rule": mini_cfg["train"]["quality_selection_rule"],
            "continuation_only": True,
            "architecture_loss_optimizer_changed": False,
            "test_split_locked": True,
        },
        "selected_checkpoint_revalidation": {
            "status": "PASS",
            "metrics": selected_metrics,
            "absolute_differences_from_history": metric_differences,
            "tolerance": revalidation_tolerance,
        },
        "resume_provenance": {
            "status": "PASS",
            "same_checkpoint_lineage": True,
            "independent_run_rows_combined": False,
            "prior_metrics_csv": str(Path(prior_metrics_csv)),
            "prior_epochs": list(range(resume_epoch + 1)),
            "resumed_epochs": expected_resumed_epochs,
            "resume_checkpoint": str(resume_path),
            "resume_checkpoint_sha256": resume_sha256,
            "prior_checkpoint_audit": prior_checkpoint_audit,
            "resume_metadata_check": provenance_check,
            "optimizer_restored": True,
            "scheduler_restored": True,
            "scaler_restored": True,
            "rng_state_restored": rng_state_available,
            "rng_fallback_seed": None if rng_state_available else seed,
            "bitwise_stochastic_continuation": rng_state_available,
            "limitation": (
                None
                if rng_state_available
                else "The historical epoch-3 checkpoint predates RNG-state saving; "
                "epoch 4 uses the registered seed but is not a bitwise replay of "
                "the interrupted in-memory epoch-4 stream."
            ),
            "selected_checkpoint": str(checkpoint),
            "selected_checkpoint_sha256": sha256_file(checkpoint),
            "completed_epochs": [int(row["epoch"]) for row in history],
        },
    }
    return report
