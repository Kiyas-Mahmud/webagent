"""Validate the pre-training model/data compatibility smoke evidence.

The historical ``model_compatibility_report.json`` is useful evidence that the
registered training pipeline exercised every Table 2 head before a candidate
was trained.  It is not checkpoint-runtime evidence and it is not a ranking
metric.  This module therefore validates its measured contents and byte
identity without treating its top-level ``PASS`` string as sufficient proof.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

from .common import SchemaError, sha256_file
from .pc01_artifacts import PC01_MODEL_ID


PC01_EXPECTED_MODEL_COMPATIBILITY_REPORT_SHA256 = (
    "1a8e9bb008daab5dfe893db7738daf12ee6e57e1ce2469ae29810d793030a441"
)
PC01_EXPECTED_FULL_REPORT_SHA256 = (
    "4d8990ecf4ed642a17c5be54af3a897a7785e76458343170d4edc098fbdcdcc8"
)
PC01_EXPECTED_RUN_CONTRACT_SHA256 = (
    "7ed17a7c68ff02a4cd693908aeb85710cee7ef0190339a52eb277cabd083a930"
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "status",
        "test_rows_read",
        "dataset_rows",
        "processed_rows",
        "forward_batch_size",
        "loss",
        "loss_terms",
        "class_weights",
        "train_distribution",
        "pre_images_processed",
        "post_images_processed",
        "parameter_changed",
        "probe_gradient_norms",
        "probe_update_norms",
        "bbox_supervised_rows",
        "bbox_geometry",
        "spatial_tokens_per_row",
        "peak_gpu_gb",
    }
)
_LOSS_TERM_FIELDS = frozenset(
    {
        "outcome",
        "failtype",
        "action",
        "recovery",
        "needs_recovery",
        "bbox",
        "memory",
        "recovery_outcome",
        "confidence",
        "calibration",
        "contrastive",
        "total",
        "bbox_l1",
        "bbox_center_l1",
        "bbox_log_size_smooth_l1",
        "bbox_giou",
        "bbox_attention_kl",
    }
)
_PROBE_FIELDS = frozenset(
    {
        "bbox",
        "strategy",
        "recovery_outcome",
        "needs_recovery",
        "grounding_adapter",
        "coordinate_projection",
        "grounding_attention",
    }
)
_DISTRIBUTION_FIELDS = frozenset(
    {
        "failure_type",
        "action_type",
        "recovery_attempted",
        "recovery_strategy",
        "recovery_success",
        "memory_update_flag",
        "bbox_available",
        "source_dataset",
    }
)
_ACTION_CLASSES = frozenset(
    {"CLICK", "TYPE", "SELECT", "SCROLL", "NAVIGATE", "PRESS_KEY"}
)
_FAILURE_CLASSES = frozenset(
    {"NONE", "PERCEPTION_ERROR", "ACTION_MISMATCH", "LOOP_DETECTED"}
)
_DISTRIBUTION_CLASS_KEYS = {
    "failure_type": _FAILURE_CLASSES,
    "action_type": _ACTION_CLASSES,
    "recovery_attempted": frozenset({"False", "True"}),
    "recovery_strategy": frozenset({"ALTERNATIVE_TARGET", "NONE"}),
    "recovery_success": frozenset({"False", "NOT_ATTEMPTED", "True"}),
    "memory_update_flag": frozenset({"False", "True"}),
    "bbox_available": frozenset({"False", "True"}),
    "source_dataset": frozenset({"original_gold"}),
}
_CLASS_VECTOR_LENGTHS = {
    "action": 6,
    "failure_type": 4,
    "outcome": 2,
    "recovery_strategy": 6,
}
_CLASS_WEIGHT_FIELDS = frozenset(
    {
        "source_rows",
        "action",
        "failure_type",
        "outcome",
        "recovery_strategy",
        "recovery_strategy_scheme",
        "recovery_strategy_cap",
        "recovery_outcome_pos_weight",
        "recovery_outcome_attempted_success",
        "recovery_outcome_attempted_failure",
        "needs_recovery_pos_weight",
        "needs_recovery_weight_scheme",
        "needs_recovery_positive",
        "needs_recovery_negative",
        "bbox_log_size_prior",
    }
)
_BBOX_LOG_SIZE_PRIOR_FIELDS = frozenset(
    {"source", "valid_rows", "excluded_invalid_rows", "log_wh", "geometric_mean_wh"}
)
_BBOX_GEOMETRY_FIELDS = frozenset(
    {
        "status",
        "records",
        "bbox_rows",
        "valid_bbox_rows",
        "invalid_bbox_rows",
        "maskable_invalid_bbox_rows",
        "fatal_invalid_bbox_rows",
        "invalid_reason_counts",
        "invalid_examples",
        "unique_images_opened",
        "normalized_valid_distribution",
        "normalized_valid_log_size_distribution",
        "contract",
    }
)
_BBOX_EXAMPLE_FIELDS = frozenset(
    {"record_id", "state_before", "image_size", "bbox", "reasons"}
)
_BBOX_FIELDS = frozenset({"x", "y", "width", "height"})
_SUMMARY_FIELDS = frozenset({"count", "mean", "std", "min", "max"})


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SchemaError(f"model compatibility report repeats JSON key: {key}")
        value[key] = item
    return value


def _read_unique_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SchemaError("model compatibility report must be one regular file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SchemaError(
                    f"model compatibility report contains non-finite JSON: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError("model compatibility report is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SchemaError("model compatibility report must be a JSON object")
    # ``scripts/run_gold.py`` is the registered writer.  Requiring its exact
    # serialization removes alternate-encoding ambiguity before a digest is
    # admitted to a frozen selection package.
    registered_serialization = json.dumps(
        value,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if path.read_bytes() != registered_serialization:
        raise SchemaError(
            "model compatibility report does not use the registered JSON serialization"
        )
    return value


def _require_exact_fields(
    value: object, expected: frozenset[str], *, context: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != expected:
        raise SchemaError(f"{context} fields differ from the registered schema")
    return value


def _finite_number(value: object, *, context: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise SchemaError(f"{context} must be {qualifier}")
    return result


def _require_exact_int(value: object, expected: int, *, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise SchemaError(f"{context} must equal {expected}")


def _validate_count_map(
    value: object,
    *,
    context: str,
    expected_keys: frozenset[str] | None = None,
    require_positive: bool = False,
) -> None:
    if not isinstance(value, Mapping) or not value:
        raise SchemaError(f"{context} must be a non-empty count mapping")
    if expected_keys is not None and frozenset(value) != expected_keys:
        raise SchemaError(f"{context} classes differ from the registered schema")
    total = 0
    for label, count in value.items():
        if not isinstance(label, str) or not label:
            raise SchemaError(f"{context} contains an invalid class label")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SchemaError(f"{context}.{label} must be a non-negative integer")
        if require_positive and count == 0:
            raise SchemaError(f"{context}.{label} must be represented in the smoke set")
        total += count
    if total != 16:
        raise SchemaError(f"{context} counts must sum to the 16 smoke rows")


def _validate_class_weights(value: object) -> None:
    value = _require_exact_fields(
        value, _CLASS_WEIGHT_FIELDS, context="class_weights"
    )
    _require_exact_int(value.get("source_rows"), 16, context="class_weights.source_rows")
    for name, expected_length in _CLASS_VECTOR_LENGTHS.items():
        vector = value.get(name)
        if not isinstance(vector, list) or len(vector) != expected_length:
            raise SchemaError(
                f"class_weights.{name} must contain {expected_length} values"
            )
        for index, item in enumerate(vector):
            number = _finite_number(
                item, context=f"class_weights.{name}[{index}]"
            )
            if number < 0.0:
                raise SchemaError(f"class_weights.{name} cannot be negative")
    if value.get("recovery_strategy_scheme") != "sqrt_inverse_frequency":
        raise SchemaError("class_weights recovery strategy scheme changed")
    if value.get("needs_recovery_weight_scheme") != "exact_inverse_frequency":
        raise SchemaError("class_weights needs-recovery scheme changed")
    for field in (
        "recovery_strategy_cap",
        "recovery_outcome_pos_weight",
        "needs_recovery_pos_weight",
    ):
        number = _finite_number(value.get(field), context=f"class_weights.{field}")
        if number < 0.0:
            raise SchemaError(f"class_weights.{field} cannot be negative")
    for field, expected in {
        "recovery_outcome_attempted_success": 1,
        "recovery_outcome_attempted_failure": 1,
        "needs_recovery_positive": 2,
        "needs_recovery_negative": 14,
    }.items():
        _require_exact_int(value.get(field), expected, context=f"class_weights.{field}")
    prior = _require_exact_fields(
        value.get("bbox_log_size_prior"),
        _BBOX_LOG_SIZE_PRIOR_FIELDS,
        context="class_weights.bbox_log_size_prior",
    )
    _require_exact_int(
        prior.get("valid_rows"), 4, context="bbox_log_size_prior.valid_rows"
    )
    _require_exact_int(
        prior.get("excluded_invalid_rows"),
        2,
        context="bbox_log_size_prior.excluded_invalid_rows",
    )
    if prior.get("source") != "valid training bbox rows only":
        raise SchemaError("bbox_log_size_prior source changed")
    for field in ("log_wh", "geometric_mean_wh"):
        vector = prior.get(field)
        if not isinstance(vector, list) or len(vector) != 2:
            raise SchemaError(f"bbox_log_size_prior.{field} must contain two values")
        for index, item in enumerate(vector):
            number = _finite_number(
                item, context=f"bbox_log_size_prior.{field}[{index}]"
            )
            if field == "geometric_mean_wh" and number <= 0.0:
                raise SchemaError("bbox_log_size_prior dimensions must be positive")


def _validate_summary(
    value: object,
    *,
    context: str,
    expected_count: int,
    normalized: bool,
) -> None:
    summary = _require_exact_fields(value, _SUMMARY_FIELDS, context=context)
    _require_exact_int(summary.get("count"), expected_count, context=f"{context}.count")
    mean = _finite_number(summary.get("mean"), context=f"{context}.mean")
    std = _finite_number(summary.get("std"), context=f"{context}.std")
    minimum = _finite_number(summary.get("min"), context=f"{context}.min")
    maximum = _finite_number(summary.get("max"), context=f"{context}.max")
    if std < 0.0 or minimum > mean or mean > maximum:
        raise SchemaError(f"{context} summary statistics are inconsistent")
    if normalized and not (0.0 <= minimum <= maximum <= 1.0):
        raise SchemaError(f"{context} must remain inside normalized image bounds")


def _validate_bbox_geometry(value: object) -> None:
    value = _require_exact_fields(
        value, _BBOX_GEOMETRY_FIELDS, context="bbox_geometry"
    )
    expected_counts = {
        "records": 16,
        "bbox_rows": 6,
        "valid_bbox_rows": 4,
        "invalid_bbox_rows": 2,
        "maskable_invalid_bbox_rows": 2,
        "fatal_invalid_bbox_rows": 0,
    }
    for field, expected in expected_counts.items():
        _require_exact_int(value.get(field), expected, context=f"bbox_geometry.{field}")
    # The authentic PC-01 report intentionally records two maskable bad source
    # boxes, so nested FAIL is expected and is not a failed smoke run.
    if value.get("status") != "FAIL":
        raise SchemaError("bbox_geometry.status must preserve the registered maskable FAIL")
    examples = value.get("invalid_examples")
    if not isinstance(examples, list) or len(examples) != 2:
        raise SchemaError("bbox_geometry must preserve two invalid examples")
    reason_counts = value.get("invalid_reason_counts")
    if not isinstance(reason_counts, Mapping) or not reason_counts:
        raise SchemaError("bbox_geometry.invalid_reason_counts must be non-empty")
    for reason, count in reason_counts.items():
        if (
            not isinstance(reason, str)
            or not reason
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise SchemaError("bbox_geometry.invalid_reason_counts is malformed")
    record_ids: set[str] = set()
    observed_reason_counts = {str(reason): 0 for reason in reason_counts}
    for index, raw_example in enumerate(examples):
        example = _require_exact_fields(
            raw_example,
            _BBOX_EXAMPLE_FIELDS,
            context=f"bbox_geometry.invalid_examples[{index}]",
        )
        record_id = example.get("record_id")
        state_before = example.get("state_before")
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id in record_ids
            or not isinstance(state_before, str)
            or not state_before
        ):
            raise SchemaError("bbox_geometry invalid-example identity is malformed")
        record_ids.add(record_id)
        image_size = example.get("image_size")
        if (
            not isinstance(image_size, list)
            or len(image_size) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in image_size
            )
        ):
            raise SchemaError("bbox_geometry invalid-example image_size is malformed")
        bbox = _require_exact_fields(
            example.get("bbox"),
            _BBOX_FIELDS,
            context=f"bbox_geometry.invalid_examples[{index}].bbox",
        )
        for coordinate, coordinate_value in bbox.items():
            number = _finite_number(
                coordinate_value,
                context=(
                    f"bbox_geometry.invalid_examples[{index}].bbox.{coordinate}"
                ),
            )
            if coordinate in {"width", "height"} and number <= 0.0:
                raise SchemaError("bbox_geometry invalid-example sizes must be positive")
        reasons = example.get("reasons")
        if (
            not isinstance(reasons, list)
            or not reasons
            or len(reasons) != len(set(reasons))
            or any(reason not in reason_counts for reason in reasons)
        ):
            raise SchemaError("bbox_geometry invalid-example reasons are malformed")
        for reason in reasons:
            observed_reason_counts[str(reason)] += 1
    if dict(reason_counts) != observed_reason_counts:
        raise SchemaError("bbox_geometry reason counts disagree with invalid examples")
    _require_exact_int(
        value.get("unique_images_opened"),
        6,
        context="bbox_geometry.unique_images_opened",
    )
    normalized = _require_exact_fields(
        value.get("normalized_valid_distribution"),
        frozenset({"x", "y", "width", "height"}),
        context="bbox_geometry.normalized_valid_distribution",
    )
    for coordinate in ("x", "y", "width", "height"):
        _validate_summary(
            normalized.get(coordinate),
            context=f"bbox_geometry.normalized_valid_distribution.{coordinate}",
            expected_count=4,
            normalized=True,
        )
        if coordinate in {"width", "height"} and float(
            normalized[coordinate]["min"]
        ) <= 0.0:
            raise SchemaError("bbox_geometry valid sizes must be positive")
    normalized_log_size = _require_exact_fields(
        value.get("normalized_valid_log_size_distribution"),
        frozenset({"width", "height"}),
        context="bbox_geometry.normalized_valid_log_size_distribution",
    )
    for dimension in ("width", "height"):
        _validate_summary(
            normalized_log_size.get(dimension),
            context=(
                "bbox_geometry.normalized_valid_log_size_distribution."
                f"{dimension}"
            ),
            expected_count=4,
            normalized=False,
        )
    if value.get("contract") != (
        "finite x/y, positive width/height, and the complete box must stay "
        "inside the actual state_before image"
    ):
        raise SchemaError("bbox_geometry contract changed")


def validate_model_compatibility_report(
    report_path: str | Path,
    *,
    model_id: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate and return one immutable candidate smoke report.

    ``expected_sha256`` is frozen by the selection package.  PC-01 has an
    additional source-registered digest and cannot be rebound by a caller.
    PC-02/PC-03 digests must be captured during their later preregistration.
    """

    path = Path(report_path).resolve()
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise SchemaError("registered model compatibility SHA-256 is malformed")
    if (
        model_id == PC01_MODEL_ID
        and expected_sha256 != PC01_EXPECTED_MODEL_COMPATIBILITY_REPORT_SHA256
    ):
        raise SchemaError("PC-01 model compatibility identity is not registered")
    report = _read_unique_json(path)
    if sha256_file(path) != expected_sha256:
        raise SchemaError("model compatibility report bytes differ from registration")
    _require_exact_fields(report, _TOP_LEVEL_FIELDS, context="model compatibility")
    if report.get("status") != "PASS":
        raise SchemaError("model compatibility smoke did not pass")
    for field, expected in {
        "test_rows_read": 0,
        "dataset_rows": 16,
        "processed_rows": 16,
        "forward_batch_size": 16,
        "pre_images_processed": 16,
        "post_images_processed": 32,
        "bbox_supervised_rows": 4,
    }.items():
        _require_exact_int(report.get(field), expected, context=field)
    if report.get("parameter_changed") is not True:
        raise SchemaError("model compatibility smoke did not update parameters")

    terms = _require_exact_fields(
        report.get("loss_terms"), _LOSS_TERM_FIELDS, context="loss_terms"
    )
    for name, value in terms.items():
        number = _finite_number(value, context=f"loss_terms.{name}")
        if number < 0.0:
            raise SchemaError(f"loss_terms.{name} cannot be negative")
    loss = _finite_number(report.get("loss"), context="loss")
    if not math.isclose(loss, float(terms["total"]), rel_tol=0.0, abs_tol=1e-9):
        raise SchemaError("model compatibility loss differs from loss_terms.total")

    for field in ("probe_gradient_norms", "probe_update_norms"):
        probes = _require_exact_fields(
            report.get(field), _PROBE_FIELDS, context=field
        )
        for name, value in probes.items():
            _finite_number(value, context=f"{field}.{name}", positive=True)

    distributions = _require_exact_fields(
        report.get("train_distribution"),
        _DISTRIBUTION_FIELDS,
        context="train_distribution",
    )
    for name, value in distributions.items():
        _validate_count_map(
            value,
            context=f"train_distribution.{name}",
            expected_keys=_DISTRIBUTION_CLASS_KEYS[name],
            require_positive=name in {"action_type", "failure_type"},
        )
    recovery_attempted = distributions["recovery_attempted"]
    recovery_success = distributions["recovery_success"]
    if (
        recovery_attempted["True"]
        != recovery_success["True"] + recovery_success["False"]
        or recovery_attempted["False"] != recovery_success["NOT_ATTEMPTED"]
        or recovery_success["True"] <= 0
        or recovery_success["False"] <= 0
    ):
        raise SchemaError(
            "train_distribution recovery attempts and outcomes are inconsistent"
        )
    if distributions["source_dataset"] != {"original_gold": 16}:
        raise SchemaError("compatibility smoke must use only original Gold training rows")
    if distributions["bbox_available"]["True"] != 6:
        raise SchemaError("train_distribution bbox coverage changed")

    _validate_class_weights(report.get("class_weights"))
    _validate_bbox_geometry(report.get("bbox_geometry"))

    spatial = report.get("spatial_tokens_per_row")
    if not isinstance(spatial, list) or len(spatial) != 16:
        raise SchemaError("spatial_tokens_per_row must contain all 16 smoke rows")
    for index, count in enumerate(spatial):
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise SchemaError(f"spatial_tokens_per_row[{index}] must be positive")
    peak_gpu_gb = _finite_number(report.get("peak_gpu_gb"), context="peak_gpu_gb")
    if peak_gpu_gb < 0.0:
        raise SchemaError("peak_gpu_gb cannot be negative")
    return report
