from __future__ import annotations

import json
from pathlib import Path

import pytest

from web_agent.eval.table2.common import SchemaError, sha256_file
from web_agent.eval.table2.model_compatibility import (
    PC01_EXPECTED_MODEL_COMPATIBILITY_REPORT_SHA256,
    validate_model_compatibility_report,
)
from web_agent.eval.table2.pc01_artifacts import PC01_MODEL_ID


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PC01_REPORT = (
    REPOSITORY_ROOT
    / "webagent_comparison/outputs/model_comparison"
    / PC01_MODEL_ID
    / "seed_42/model_compatibility_report.json"
)


def _mutated_report(tmp_path: Path, mutator) -> Path:
    payload = json.loads(PC01_REPORT.read_text(encoding="utf-8"))
    mutator(payload)
    destination = tmp_path / "model_compatibility_report.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def test_registered_pc01_model_compatibility_report_passes() -> None:
    assert sha256_file(PC01_REPORT) == PC01_EXPECTED_MODEL_COMPATIBILITY_REPORT_SHA256
    report = validate_model_compatibility_report(
        PC01_REPORT,
        model_id=PC01_MODEL_ID,
        expected_sha256=PC01_EXPECTED_MODEL_COMPATIBILITY_REPORT_SHA256,
    )
    assert report["test_rows_read"] == 0
    assert report["bbox_geometry"]["status"] == "FAIL"


def test_pc01_compatibility_digest_cannot_be_rebound(tmp_path: Path) -> None:
    changed = _mutated_report(tmp_path, lambda value: value.update(peak_gpu_gb=8.2))
    with pytest.raises(SchemaError, match="identity is not registered"):
        validate_model_compatibility_report(
            changed,
            model_id=PC01_MODEL_ID,
            expected_sha256=sha256_file(changed),
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda value: value.update(test_rows_read=1), "test_rows_read"),
        (lambda value: value.update(processed_rows=15), "processed_rows"),
        (
            lambda value: value["probe_update_norms"].update(bbox=0.0),
            "probe_update_norms.bbox",
        ),
        (
            lambda value: value["train_distribution"]["action_type"].pop("TYPE"),
            "action_type classes",
        ),
        (
            lambda value: value["bbox_geometry"].update(fatal_invalid_bbox_rows=1),
            "fatal_invalid_bbox_rows",
        ),
        (
            lambda value: value["train_distribution"]["recovery_success"].update(
                {"False": 0, "True": 2}
            ),
            "recovery attempts and outcomes",
        ),
        (
            lambda value: value["loss_terms"].update(action=-0.1),
            "loss_terms.action cannot be negative",
        ),
        (
            lambda value: value["bbox_geometry"]["invalid_examples"][0].update(
                reasons=["bottom_boundary_overflow"]
            ),
            "reason counts disagree",
        ),
        (lambda value: value.update(spatial_tokens_per_row=[252] * 15), "16 smoke"),
    ),
)
def test_semantic_validator_rejects_false_passes(
    tmp_path: Path, mutator, message: str
) -> None:
    changed = _mutated_report(tmp_path, mutator)
    with pytest.raises(SchemaError, match=message):
        validate_model_compatibility_report(
            changed,
            model_id="future-fixture-candidate",
            expected_sha256=sha256_file(changed),
        )


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    original = PC01_REPORT.read_text(encoding="utf-8")
    duplicate = original.replace(
        '"status": "PASS",', '"status": "PASS",\n  "status": "PASS",', 1
    )
    destination = tmp_path / "duplicate.json"
    destination.write_text(duplicate, encoding="utf-8")
    with pytest.raises(SchemaError, match="repeats JSON key"):
        validate_model_compatibility_report(
            destination,
            model_id="future-fixture-candidate",
            expected_sha256=sha256_file(destination),
        )


def test_nonregistered_json_serialization_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(PC01_REPORT.read_text(encoding="utf-8"))
    destination = tmp_path / "compact.json"
    destination.write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="registered JSON serialization"):
        validate_model_compatibility_report(
            destination,
            model_id="future-fixture-candidate",
            expected_sha256=sha256_file(destination),
        )
