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
    load_gold_split,
)
from web_agent.data.gold_dataset import view
from web_agent.models.loss import CombinedLoss
from web_agent.models.model import WebAgentModel
from web_agent.train.trainer import Trainer


@dataclass
class GoldComponents:
    processor: object
    model: WebAgentModel
    loss_fn: CombinedLoss
    train_records: list[dict]
    device: str = "cuda"


def build_processor(cfg: dict):
    from transformers import AutoProcessor

    backbone = cfg["backbone"]
    return AutoProcessor.from_pretrained(
        backbone["vlm_model"],
        min_pixels=backbone["min_pixels"],
        max_pixels=backbone["max_pixels"],
    )


def build_gold_components(cfg: dict, processor=None) -> GoldComponents:
    """Build one causal model and all loss weights from the training split."""
    processor = processor or build_processor(cfg)
    train_records = load_gold_split(cfg, "train")
    action_w, failure_w, outcome_w = gold_class_weights(train_records)

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
        recovery_success_pos_weight=recovery_pos_weight.to(device),
    ).to(device)

    print("train rows:", len(train_records))
    print("action weights:", [round(value, 2) for value in action_w.tolist()])
    print("failure weights:", [round(value, 2) for value in failure_w.tolist()])
    print("outcome weights:", [round(value, 2) for value in outcome_w.tolist()])
    print(
        "recovery pos_weight:", round(float(recovery_pos_weight), 2),
        f"(attempted success={positives}, failure={negatives})",
    )
    return GoldComponents(processor, model, loss_fn, train_records, device)


def run_gold_smoke(
    cfg: dict,
    processor=None,
    rows: int = 16,
    seed: int = 42,
) -> dict:
    """Verify causal streams, output shapes, finite loss, backward, and weight update."""
    components = build_gold_components(cfg, processor)
    loader = build_gold_dataloader(
        cfg,
        "train",
        components.processor,
        records=components.train_records,
        limit=rows,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=False,
        num_workers=0,
        seed=seed,
        smoke=True,
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
    epochs: int = 3,
    seed: int = 42,
) -> dict:
    """Train on a stratified subset and validate without reading the test split."""
    mini_cfg = deepcopy(cfg)
    mini_cfg["train"]["epochs"] = epochs
    mini_cfg["name"] = f"{cfg['name']}_MINI"
    mini_cfg["train"]["metrics_csv"] = "results/gold_mini_metrics.csv"

    components = build_gold_components(mini_cfg, processor)
    train_loader = build_gold_dataloader(
        mini_cfg,
        "train",
        components.processor,
        records=components.train_records,
        limit=train_rows,
        shuffle=True,
        num_workers=mini_cfg["data"].get("num_workers", 2),
        seed=seed,
    )
    val_loader = build_gold_dataloader(
        mini_cfg,
        "val",
        components.processor,
        limit=val_rows,
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
        train_sampler=None,
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

    return {
        **result,
        "status": "PASS",
        "train_rows": len(train_loader.dataset),
        "val_rows": len(val_loader.dataset),
        "test_rows_read": 0,
        "loss_decreased": loss_decreased,
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "best_checkpoint": str(checkpoint),
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
