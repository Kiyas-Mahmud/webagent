"""Read-only bounding-box geometry audit for structured Gold records."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev

from PIL import Image

REQUIRED_BBOX_KEYS = ("x", "y", "width", "height")
FATAL_IMAGE_REASONS = {
    "missing_state_before_path",
    "unreadable_state_before_image",
    "invalid_image_size",
}


def _view(record: dict) -> tuple[dict, dict, dict]:
    """Read nested v12/v14 or legacy flat records without importing PyTorch."""
    if "inputs" in record and "labels" in record:
        return record["inputs"], record["labels"], record.get("meta", {})
    return record, record, record


def _record_id(record: dict, index: int) -> str:
    _, _, meta = _view(record)
    return str(
        meta.get("sample_id")
        or meta.get("task_id")
        or f"row:{index}"
    )


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": fmean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def audit_bbox_geometry(
    records: list[dict],
    data_root: str | Path,
    *,
    tolerance: float = 1e-6,
    max_examples: int = 20,
) -> dict:
    """Check boxes against the actual state-before image dimensions.

    No coordinate is clipped or repaired. The report identifies the exact rows
    that must either be excluded from bbox supervision or corrected from source
    evidence instead of silently changing thesis labels.
    """
    root = Path(data_root)
    image_sizes: dict[str, tuple[int, int]] = {}
    reasons: Counter[str] = Counter()
    examples: list[dict] = []
    bbox_rows = 0
    valid_rows = 0
    fatal_invalid_rows = 0
    maskable_invalid_rows = 0
    coordinates = {name: [] for name in REQUIRED_BBOX_KEYS}
    log_sizes = {"width": [], "height": []}

    for index, record in enumerate(records):
        inputs, labels, _ = _view(record)
        box = labels.get("action_target_bbox")
        if box is None:
            continue
        bbox_rows += 1
        row_reasons: list[str] = []
        if not isinstance(box, dict):
            missing_keys = list(REQUIRED_BBOX_KEYS)
            row_reasons.append("bbox_not_object")
        else:
            missing_keys = [key for key in REQUIRED_BBOX_KEYS if key not in box]
        if missing_keys:
            row_reasons.append("missing_keys")
            values = None
        else:
            try:
                values = tuple(float(box[key]) for key in REQUIRED_BBOX_KEYS)
            except (TypeError, ValueError):
                values = None
                row_reasons.append("non_numeric")
        image_path = str(inputs.get("state_before") or "")
        image_size = None
        if not image_path:
            row_reasons.append("missing_state_before_path")
        else:
            try:
                if image_path not in image_sizes:
                    with Image.open(root / image_path) as image:
                        image_sizes[image_path] = image.size
                image_size = image_sizes[image_path]
            except (FileNotFoundError, OSError):
                row_reasons.append("unreadable_state_before_image")

        normalized = None
        if values is not None:
            x, y, width, height = values
            if not all(math.isfinite(value) for value in values):
                row_reasons.append("non_finite")
            if x < 0 or y < 0:
                row_reasons.append("negative_origin")
            if width <= 0 or height <= 0:
                row_reasons.append("non_positive_size")
            if image_size is not None:
                image_width, image_height = image_size
                if image_width <= 0 or image_height <= 0:
                    row_reasons.append("invalid_image_size")
                else:
                    normalized = (
                        x / image_width,
                        y / image_height,
                        width / image_width,
                        height / image_height,
                    )
                    if x > image_width + tolerance:
                        row_reasons.append("x_origin_outside")
                    if y > image_height + tolerance:
                        row_reasons.append("y_origin_outside")
                    if x + width > image_width + tolerance:
                        row_reasons.append("right_boundary_overflow")
                    if y + height > image_height + tolerance:
                        row_reasons.append("bottom_boundary_overflow")

        unique_reasons = sorted(set(row_reasons))
        reasons.update(unique_reasons)
        if unique_reasons:
            if FATAL_IMAGE_REASONS.intersection(unique_reasons):
                fatal_invalid_rows += 1
            else:
                maskable_invalid_rows += 1
            if len(examples) < max_examples:
                examples.append({
                    "record_id": _record_id(record, index),
                    "state_before": image_path,
                    "image_size": list(image_size) if image_size else None,
                    "bbox": dict(box) if isinstance(box, dict) else box,
                    "reasons": unique_reasons,
                })
            continue

        valid_rows += 1
        for name, value in zip(REQUIRED_BBOX_KEYS, normalized, strict=True):
            coordinates[name].append(value)
        log_sizes["width"].append(math.log(normalized[2]))
        log_sizes["height"].append(math.log(normalized[3]))

    invalid_rows = bbox_rows - valid_rows
    return {
        "status": "PASS" if bbox_rows > 0 and invalid_rows == 0 else "FAIL",
        "records": len(records),
        "bbox_rows": bbox_rows,
        "valid_bbox_rows": valid_rows,
        "invalid_bbox_rows": invalid_rows,
        "maskable_invalid_bbox_rows": maskable_invalid_rows,
        "fatal_invalid_bbox_rows": fatal_invalid_rows,
        "invalid_reason_counts": dict(sorted(reasons.items())),
        "invalid_examples": examples,
        "unique_images_opened": len(image_sizes),
        "normalized_valid_distribution": {
            name: _summary(values) for name, values in coordinates.items()
        },
        "normalized_valid_log_size_distribution": {
            name: _summary(values) for name, values in log_sizes.items()
        },
        "contract": (
            "finite x/y, positive width/height, and the complete box must stay "
            "inside the actual state_before image"
        ),
    }


def bbox_log_size_prior(
    records: list[dict],
    data_root: str | Path,
) -> dict:
    """Compute a geometric width/height prior from valid training rows.

    Invalid localization labels are excluded exactly as they are from bbox
    supervision.  Fatal image errors still stop construction because those rows
    are unusable for the multimodal model, not merely for localization.
    """
    report = audit_bbox_geometry(records, data_root, max_examples=5)
    if report["fatal_invalid_bbox_rows"]:
        raise ValueError(
            "cannot derive bbox prior with fatal state-before image errors"
        )
    if report["valid_bbox_rows"] <= 0:
        raise ValueError("cannot derive bbox prior without a valid bbox row")
    distribution = report["normalized_valid_log_size_distribution"]
    log_width = float(distribution["width"]["mean"])
    log_height = float(distribution["height"]["mean"])
    return {
        "source": "valid training bbox rows only",
        "valid_rows": int(report["valid_bbox_rows"]),
        "excluded_invalid_rows": int(report["invalid_bbox_rows"]),
        "log_wh": [log_width, log_height],
        "geometric_mean_wh": [math.exp(log_width), math.exp(log_height)],
    }


def audit_bbox_probe_feasibility(
    records: list[dict],
    data_root: str | Path,
    *,
    tolerance: float = 1e-6,
    max_examples: int = 10,
) -> dict:
    """Detect impossible bbox supervision for identical pre-action inputs.

    The policy/bbox stream sees state-before image content, task description,
    and website domain. If those exact inputs repeat with different normalized
    boxes, no deterministic model can memorize every row. Image content is
    hashed so duplicate files under different relative paths are still found.
    """
    geometry = audit_bbox_geometry(records, data_root, max_examples=max_examples)
    if geometry["status"] != "PASS":
        return {
            "status": "FAIL",
            "reason": "selected probe contains invalid bbox geometry",
            "records": len(records),
            "bbox_geometry": geometry,
            "test_rows_read": 0,
        }

    root = Path(data_root)
    image_hashes: dict[str, str] = {}
    image_sizes: dict[str, tuple[int, int]] = {}
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for index, record in enumerate(records):
        inputs, labels, _ = _view(record)
        relative_path = str(inputs["state_before"])
        if relative_path not in image_hashes:
            digest = hashlib.sha256()
            with Image.open(root / relative_path) as image:
                rgb = image.convert("RGB")
                image_sizes[relative_path] = rgb.size
                digest.update(f"{rgb.width}x{rgb.height}:RGB".encode("ascii"))
                digest.update(rgb.tobytes())
            image_hashes[relative_path] = digest.hexdigest()
        width, height = image_sizes[relative_path]
        box = labels["action_target_bbox"]
        normalized_box = (
            float(box["x"]) / width,
            float(box["y"]) / height,
            float(box["width"]) / width,
            float(box["height"]) / height,
        )
        key = (
            image_hashes[relative_path],
            str(inputs.get("task_description") or ""),
            str(inputs.get("website_domain") or ""),
        )
        groups.setdefault(key, []).append({
            "record_id": _record_id(record, index),
            "state_before": relative_path,
            "normalized_bbox": list(normalized_box),
        })

    duplicate_groups = [rows for rows in groups.values() if len(rows) > 1]
    conflicting_groups = []
    for rows in duplicate_groups:
        reference = rows[0]["normalized_bbox"]
        if any(
            any(abs(left - right) > tolerance for left, right in zip(
                reference,
                row["normalized_bbox"],
                strict=True,
            ))
            for row in rows[1:]
        ):
            conflicting_groups.append(rows)
    conflicting_rows = sum(len(rows) for rows in conflicting_groups)
    return {
        "status": "PASS" if not conflicting_groups else "FAIL",
        "reason": (
            "no identical pre-action inputs have conflicting bbox targets"
            if not conflicting_groups
            else "identical pre-action inputs have conflicting bbox targets"
        ),
        "records": len(records),
        "unique_pre_action_inputs": len(groups),
        "duplicate_input_groups": len(duplicate_groups),
        "conflicting_input_groups": len(conflicting_groups),
        "conflicting_rows": conflicting_rows,
        "conflict_examples": conflicting_groups[:max_examples],
        "image_identity": "sha256 decoded RGB pixels plus dimensions",
        "text_identity": "exact task_description plus website_domain",
        "test_rows_read": 0,
    }


def require_valid_bbox_geometry(report: dict, split_name: str) -> None:
    """Stop training when geometry errors would otherwise be clipped silently."""
    if report["status"] != "PASS":
        raise AssertionError(
            f"{split_name} bbox geometry failed for "
            f"{report['invalid_bbox_rows']}/{report['bbox_rows']} rows; "
            f"review invalid_examples in the saved audit report"
        )
