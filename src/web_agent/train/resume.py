"""Validation helpers for continuing an interrupted controlled mini run.

These helpers deliberately avoid importing PyTorch.  They validate the small,
serializable portion of a checkpoint before the expensive model is constructed
and keep the provenance rules independently unit-testable.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


_NON_TRAINING_DATA_KEYS = {"num_workers", "root"}
_TRAIN_KEYS = {
    "controlled_experiment_tag",
    "early_stop_metric",
    "epochs",
    "quality_selection_rule",
}


def load_epoch_metrics_csv(path: str | Path) -> list[dict[str, float | int]]:
    """Load the trainer CSV and reject missing, duplicate, or non-finite values."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"prior metrics CSV does not exist: {source}")

    with source.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"prior metrics CSV is empty: {source}")
    if "epoch" not in raw_rows[0]:
        raise ValueError("prior metrics CSV has no epoch column")

    rows: list[dict[str, float | int]] = []
    for row_index, raw in enumerate(raw_rows, start=2):
        parsed: dict[str, float | int] = {}
        for key, value in raw.items():
            if key is None or value is None or value.strip() == "":
                raise ValueError(
                    f"prior metrics CSV has an empty value at row {row_index}"
                )
            try:
                number = float(value)
            except ValueError as error:
                raise ValueError(
                    f"prior metrics CSV has a non-numeric {key!r} value "
                    f"at row {row_index}: {value!r}"
                ) from error
            if not math.isfinite(number):
                raise ValueError(
                    f"prior metrics CSV has a non-finite {key!r} value "
                    f"at row {row_index}: {value!r}"
                )
            parsed[key] = int(number) if key == "epoch" else number
        rows.append(parsed)
    return rows


def validate_prior_history(
    history: Sequence[Mapping[str, Any]],
    *,
    resume_epoch: int,
    total_epochs: int,
) -> list[dict[str, Any]]:
    """Require exactly epochs 0..resume_epoch and leave one or more to resume."""
    if not 0 <= resume_epoch < total_epochs - 1:
        raise ValueError(
            f"resume_epoch must be between 0 and {total_epochs - 2}, got {resume_epoch}"
        )
    normalized = [dict(row) for row in history]
    actual = [int(row["epoch"]) for row in normalized]
    expected = list(range(resume_epoch + 1))
    if actual != expected:
        raise ValueError(
            "prior history must contain consecutive completed epochs "
            f"{expected}, got {actual}"
        )
    return normalized


def discover_epoch_checkpoints(
    checkpoint_dir: str | Path,
    epochs: Sequence[int],
) -> dict[str, str]:
    """Find exactly one completed best checkpoint for each requested epoch."""
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"prior checkpoint directory does not exist: {root}")
    found: dict[str, str] = {}
    for epoch in epochs:
        matches = sorted(root.glob(f"best_e{int(epoch)}_*.ckpt"))
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one epoch-{epoch} checkpoint under {root}, "
                f"found {len(matches)}: {matches}"
            )
        found[str(int(epoch))] = str(matches[0])
    return found


def training_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that can alter the registered optimization experiment."""
    data = {
        key: value
        for key, value in dict(config.get("data", {})).items()
        if key not in _NON_TRAINING_DATA_KEYS
    }
    train = {
        key: config.get("train", {}).get(key)
        for key in sorted(_TRAIN_KEYS)
    }
    return {
        "name": config.get("name"),
        "backbone": config.get("backbone"),
        "model": config.get("model"),
        "loss": config.get("loss"),
        "optim": config.get("optim"),
        "data": data,
        "train": train,
        "seeds": config.get("seeds"),
    }


def validate_resume_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any],
    prior_history: Sequence[Mapping[str, Any]],
    resume_epoch: int,
    total_epochs: int,
    steps_per_epoch: int,
    metric_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Reject a checkpoint that cannot be the completed end of prior history."""
    validate_prior_history(
        prior_history, resume_epoch=resume_epoch, total_epochs=total_epochs
    )
    required_state = (
        "config",
        "optimizer",
        "scheduler",
        "scaler",
        "epoch",
        "step",
        "selection_metric",
        "selection_value",
        "epoch_complete",
    )
    missing = [key for key in required_state if key not in checkpoint]
    if missing:
        raise ValueError(f"resume checkpoint is missing state: {missing}")
    if checkpoint["epoch_complete"] is not True:
        raise ValueError("resume checkpoint is not marked as a completed epoch")
    if int(checkpoint["epoch"]) != resume_epoch:
        raise ValueError(
            f"resume checkpoint epoch is {checkpoint['epoch']}, expected {resume_epoch}"
        )

    expected_step = (resume_epoch + 1) * int(steps_per_epoch)
    if int(checkpoint["step"]) != expected_step:
        raise ValueError(
            f"resume checkpoint step is {checkpoint['step']}, expected {expected_step}"
        )
    selection_metric = str(expected_config["train"]["early_stop_metric"])
    if checkpoint["selection_metric"] != selection_metric:
        raise ValueError(
            "resume checkpoint selection metric disagrees with current config: "
            f"{checkpoint['selection_metric']!r} != {selection_metric!r}"
        )
    prior_row = next(
        row for row in prior_history if int(row["epoch"]) == resume_epoch
    )
    if selection_metric not in prior_row:
        raise ValueError(
            f"prior history is missing selection metric {selection_metric!r}"
        )
    difference = abs(
        float(checkpoint["selection_value"]) - float(prior_row[selection_metric])
    )
    if difference > metric_tolerance:
        raise ValueError(
            "checkpoint selection value disagrees with the prior CSV row: "
            f"difference={difference:.8g}, tolerance={metric_tolerance}"
        )

    saved_signature = training_signature(checkpoint["config"])
    expected_signature = training_signature(expected_config)
    if saved_signature != expected_signature:
        differing = [
            key
            for key in expected_signature
            if saved_signature.get(key) != expected_signature.get(key)
        ]
        raise ValueError(
            "checkpoint training configuration does not match this resume: "
            f"differing sections={differing}"
        )
    return {
        "status": "PASS",
        "resume_epoch": resume_epoch,
        "next_epoch": resume_epoch + 1,
        "expected_step": expected_step,
        "selection_metric": selection_metric,
        "selection_value_difference": difference,
        "training_signature": expected_signature,
        "optimizer_restored": True,
        "scheduler_restored": True,
        "scaler_restored": True,
        "rng_state_available": "rng_state" in checkpoint,
    }


def merge_epoch_histories(
    prior_history: Sequence[Mapping[str, Any]],
    resumed_history: Sequence[Mapping[str, Any]],
    *,
    total_epochs: int,
) -> list[dict[str, Any]]:
    """Merge only non-overlapping epochs and require a complete 0..N-1 history."""
    rows = [dict(row) for row in prior_history] + [dict(row) for row in resumed_history]
    epochs = [int(row["epoch"]) for row in rows]
    if len(epochs) != len(set(epochs)):
        raise ValueError(f"cannot merge duplicate epoch rows: {epochs}")
    rows.sort(key=lambda row: int(row["epoch"]))
    actual = [int(row["epoch"]) for row in rows]
    expected = list(range(total_epochs))
    if actual != expected:
        raise ValueError(
            f"combined history must contain exactly epochs {expected}, got {actual}"
        )
    return rows


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash an artifact without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
