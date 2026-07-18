"""Reusable smoke and mini stages for the structured Gold 40K dataset.

The notebook should orchestrate experiments, not reimplement training logic. This
module owns component construction, causal smoke checks, validation-only mini
training, and checkpoint round-trip verification.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

from web_agent.data.gold_dataloader import (
    build_gold_dataloader,
    gold_class_weights,
    gold_recovery_class_weights,
    load_gold_split,
    select_smoke_records,
    stratified_gold_subsample,
)
from web_agent.data.gold_dataset import view
from web_agent.data.gold_sampling import (
    label_distribution,
    recovery_aware_batch_indices,
    select_recovery_aware_gold_subset,
)
from web_agent.models.loss import CombinedLoss
from web_agent.models.model import WebAgentModel
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


def build_gold_components(cfg: dict, processor=None, train_records=None) -> GoldComponents:
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
    recovery_w = gold_recovery_class_weights(train_records, cap=recovery_cap)

    recovery_values = [view(record)[1].get("recovery_success") for record in train_records]
    positives = sum(value is True for value in recovery_values)
    negatives = sum(value is False for value in recovery_values)
    recovery_pos_weight = torch.tensor([min(negatives / max(positives, 1), 5.0)])

    model = WebAgentModel(cfg)
    device = "cuda"
    for module in (
        model.adapter,
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
    }
    return GoldComponents(
        processor, model, loss_fn, train_records, weight_report, device,
    )


def run_gold_smoke(
    cfg: dict,
    processor=None,
    rows: int = 16,
    seed: int = 42,
) -> dict:
    """Verify causal streams, output shapes, finite loss, backward, and weight update."""
    all_train_records = load_gold_split(cfg, "train")
    smoke_records = select_smoke_records(all_train_records, rows, seed)
    components = build_gold_components(cfg, processor, train_records=smoke_records)
    loader = build_gold_dataloader(
        cfg,
        "train",
        components.processor,
        records=components.train_records,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=False,
        num_workers=0,
        seed=seed,
    )
    if len(loader.dataset) != rows:
        raise AssertionError(f"smoke loader has {len(loader.dataset)} rows, expected {rows}")

    model = components.model
    model.train()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.lora_parameters(), "lr": cfg["optim"]["lr_lora"]},
            {"params": model.head_parameters(), "lr": cfg["optim"]["lr_heads"]},
        ],
        weight_decay=cfg["optim"].get("weight_decay", 0.01),
    )
    scaler = torch.amp.GradScaler("cuda")
    probe = next(parameter for parameter in model.action_head.parameters() if parameter.requires_grad)
    before = probe.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    term_sums = {}
    first_batch = None
    first_predictions = None
    processed_rows = 0
    pre_images = 0
    post_images = 0
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

    if processed_rows != rows or first_batch is None or first_predictions is None:
        raise AssertionError(f"smoke processed {processed_rows} rows, expected {rows}")
    scaler.unscale_(optimizer)
    gradients = [
        parameter.grad
        for parameter in model.trainable_parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("smoke backward produced missing or non-finite gradients")
    scaler.step(optimizer)
    scaler.update()
    parameter_changed = not torch.equal(before, probe.detach())
    if not parameter_changed:
        raise AssertionError("optimizer step did not change the action head")

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
        "recovery_outcome": (batch_size, 1),
    }
    for key, expected in expected_shapes.items():
        actual = tuple(first_predictions[key].shape)
        if actual != expected:
            raise AssertionError(f"{key} shape={actual}, expected={expected}")

    return {
        "status": "PASS",
        "dataset_rows": rows,
        "processed_rows": processed_rows,
        "forward_batch_size": batch_size,
        "loss": term_sums["total"] / len(loader),
        "loss_terms": {key: value / len(loader) for key, value in term_sums.items()},
        "class_weights": components.class_weight_report,
        "train_distribution": label_distribution(smoke_records),
        "pre_images_processed": pre_images,
        "post_images_processed": post_images,
        "parameter_changed": parameter_changed,
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
    components = build_gold_components(
        mini_cfg,
        processor,
        train_records=selected_train_records,
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
    )
    val_loader = build_gold_dataloader(
        mini_cfg,
        "val",
        components.processor,
        records=selected_val_records,
        shuffle=False,
        num_workers=mini_cfg["data"].get("num_workers", 2),
        seed=seed,
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
    loss_decreased = len(history) > 1 and history[-1]["train_loss"] < history[0]["train_loss"]
    if not loss_decreased:
        raise AssertionError("mini training loss did not decrease across epochs")
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
        "experiment_control": {
            "baseline": "kaggle-gold-v14",
            "experiment_tag": experiment_tag.lower(),
            "fixed": [
                "model_architecture",
                "validation_rows",
                "seed",
                "batch_size",
                "gradient_accumulation",
                "learning_rates",
                "loss_coefficients",
                "primary_selection_metric",
            ],
            "training_changes": [
                "joint proportional training subset coverage",
                "recovery-aware distribution-preserving physical batches",
                "sqrt inverse-frequency recovery-strategy class weights",
                "weights derived from selected training rows only",
            ],
            "observational_changes": [
                "honest scalar metrics",
                "per-class diagnostics",
                "raw and weighted loss terms",
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
    """Perturb one head, reload, and prove deterministic validation logits return."""
    trainer.model.eval()
    batch = next(iter(val_loader))
    with torch.autocast("cuda", dtype=torch.float16):
        expected = trainer.model(batch)["action_type"].float().cpu()

    parameter = next(trainer.model.action_head.parameters())
    parameter.add_(0.25)
    trainer.load_checkpoint(trainer.best[0][1], resume_training=False)
    trainer.model.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        restored = trainer.model(batch)["action_type"].float().cpu()
    if not torch.allclose(expected, restored, atol=1e-5, rtol=1e-5):
        raise AssertionError("checkpoint reload did not restore identical predictions")
    return True
