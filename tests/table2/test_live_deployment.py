from __future__ import annotations

import json
from pathlib import Path

import pytest

from web_agent.eval.table2.common import SchemaError, sha256_file
from web_agent.eval.table2.live_deployment import (
    LIVE_CAPABILITY_EVIDENCE_RECORD_TYPE,
    LIVE_CAPABILITY_EVIDENCE_SCHEMA_VERSION,
    LIVE_DEPLOYMENT_RECORD_TYPE,
    LIVE_DEPLOYMENT_SCHEMA_VERSION,
    PINNED_LIBWEBARENA_VERSION,
    PINNED_LIBWEBARENA_WHEEL_SHA256,
    PINNED_UPSTREAM_EVALUATORS_SHA256,
    PINNED_UPSTREAM_JUDGE_SOURCE_SHA256,
    PINNED_UPSTREAM_WEBARENA_REPOSITORY,
    PINNED_UPSTREAM_WEBARENA_REVISION,
    PINNED_WEBARENA_JUDGE_DECODING,
    PINNED_WEBARENA_JUDGE_MODEL_ID,
    PINNED_WEBARENA_JUDGE_TASK_COUNT,
    PINNED_WEBARENA_JUDGE_TASK_SET_SHA256,
    PINNED_WEBARENA_STRING_MATCH_TASK_COUNT,
    PINNED_WEBARENA_STRING_MATCH_TASK_SET_SHA256,
    REGISTERED_BROWSERGYM_VERSION,
    REGISTERED_SEALED_BROKER_CONTRACT,
    REGISTERED_START_STATE_CONTRACT,
    REGISTERED_VALIDATION_DISABLED_EXECUTION_PATH,
    REQUIRED_CAPABILITY_CLAIMS,
    REQUIRED_LIVE_CAPABILITIES,
    UPSTREAM_WEBARENA_START_STATE_FIELDS,
    LIVE_DEPLOYMENT_BINDING_FIELD,
    stage_pc01_live_deployment_package,
    validate_bound_pc01_live_deployment,
    validate_pc01_live_deployment_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RELATIVE = Path(__file__).resolve().relative_to(ROOT).as_posix()


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_bytes(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _evaluator_provenance(tmp_path: Path) -> dict:
    evidence = tmp_path / "evaluator-provenance"
    runtime_evaluator = _write_bytes(
        evidence / "runtime_evaluators.py",
        b"# measured libwebarena evaluator module\n",
    )
    runtime_judge = _write_bytes(
        evidence / "runtime_helper_functions.py",
        b"# measured libwebarena judge module\n",
    )
    delta = _write_json(
        evidence / "compatibility_delta.json",
        {"classification": "compatibility_port", "differences": ["relative imports"]},
    )
    review = _write_json(
        evidence / "compatibility_review.json",
        {"status": "PASS", "scope": "registered public tasks 0-49"},
    )
    evaluator_configuration = _write_json(
        evidence / "evaluator_configuration.json",
        {"task_set": "public-development-0-49", "sealed": True},
    )
    response_schema = _write_json(
        evidence / "judge_response_schema.json",
        {"type": "string", "allowed_verdicts": ["correct", "incorrect"]},
    )
    availability = _write_json(
        evidence / "judge_api_availability.json",
        {
            "schema_version": "table2-webarena-judge-api-preflight-v1",
            "record_type": "WebArenaJudgeAPIPreflightReceipt",
            "model_id": PINNED_WEBARENA_JUDGE_MODEL_ID,
            "availability_status": "PASS",
            "decoding_parameters": dict(PINNED_WEBARENA_JUDGE_DECODING),
            "prompt_source_sha256": PINNED_UPSTREAM_JUDGE_SOURCE_SHA256,
            "response_schema_sha256": sha256_file(response_schema),
            "request_succeeded": True,
            "response_conformed_to_schema": True,
            "credential_material_embedded": False,
            "evaluator_output_returned_to_runtime": False,
            "response_content_sha256": "8" * 64,
            "measured_at_utc": "2026-09-04T12:00:00+06:00",
        },
    )

    def relative(path: Path) -> str:
        return str(path.relative_to(tmp_path))

    return {
        "upstream_repository": PINNED_UPSTREAM_WEBARENA_REPOSITORY,
        "upstream_revision": PINNED_UPSTREAM_WEBARENA_REVISION,
        "upstream_evaluator_source_sha256": PINNED_UPSTREAM_EVALUATORS_SHA256,
        "upstream_judge_source_sha256": PINNED_UPSTREAM_JUDGE_SOURCE_SHA256,
        "runtime_distribution": "libwebarena",
        "runtime_distribution_version": PINNED_LIBWEBARENA_VERSION,
        "runtime_distribution_sha256": PINNED_LIBWEBARENA_WHEEL_SHA256,
        "runtime_evaluator_module": "webarena.evaluation_harness.evaluators",
        "runtime_evaluator_module_evidence_path": relative(runtime_evaluator),
        "runtime_evaluator_module_sha256": sha256_file(runtime_evaluator),
        "runtime_judge_module_evidence_path": relative(runtime_judge),
        "runtime_judge_module_sha256": sha256_file(runtime_judge),
        "implementation_classification": "reviewed_compatibility_port",
        "byte_identical_to_upstream": False,
        "compatibility_delta_evidence_path": relative(delta),
        "compatibility_delta_sha256": sha256_file(delta),
        "compatibility_review_evidence_path": relative(review),
        "compatibility_review_sha256": sha256_file(review),
        "evaluator_configuration_evidence_path": relative(evaluator_configuration),
        "evaluator_configuration_sha256": sha256_file(evaluator_configuration),
        "judge_model_id": PINNED_WEBARENA_JUDGE_MODEL_ID,
        "judge_prompt_source_sha256": PINNED_UPSTREAM_JUDGE_SOURCE_SHA256,
        "judge_decoding_parameters": dict(PINNED_WEBARENA_JUDGE_DECODING),
        "judge_required_task_count": PINNED_WEBARENA_JUDGE_TASK_COUNT,
        "judge_required_task_set_sha256": PINNED_WEBARENA_JUDGE_TASK_SET_SHA256,
        "string_match_task_count": PINNED_WEBARENA_STRING_MATCH_TASK_COUNT,
        "string_match_task_set_sha256": (
            PINNED_WEBARENA_STRING_MATCH_TASK_SET_SHA256
        ),
        "judge_api_response_schema_evidence_path": relative(response_schema),
        "judge_api_response_schema_sha256": sha256_file(response_schema),
        "judge_api_availability_receipt_evidence_path": relative(availability),
        "judge_api_availability_receipt_sha256": sha256_file(availability),
        "judge_api_preflight_status": "PASS",
        "judge_model_substitution_allowed": False,
        "runtime_fallback_allowed": False,
        "unavailable_task_policy": (
            "preregistered_exclusion_or_disclosed_evaluator_change_before_outcomes"
        ),
        "judge_outputs_returned_to_runtime": 0,
    }


def _manifest(tmp_path: Path) -> dict:
    source_sha256 = sha256_file(ROOT / SOURCE_RELATIVE)
    evaluator_provenance = _evaluator_provenance(tmp_path)
    capabilities = {}
    for index, (capability_id, (plane, return_contract)) in enumerate(
        sorted(REQUIRED_LIVE_CAPABILITIES.items()), start=1
    ):
        evidence = _write_json(
            tmp_path / "evidence" / f"{capability_id}.json",
            {
                "schema_version": LIVE_CAPABILITY_EVIDENCE_SCHEMA_VERSION,
                "record_type": LIVE_CAPABILITY_EVIDENCE_RECORD_TYPE,
                "capability_id": capability_id,
                "status": "PASS",
                "implementation_source_sha256": source_sha256,
                "deployment_state_sha256": f"{index:x}" * 64,
                "measured_at_utc": "2026-09-04T12:00:00+06:00",
                "locked_test_rows_read": 0,
                "oracle_values_returned_to_runtime": 0,
                "claims": REQUIRED_CAPABILITY_CLAIMS[capability_id],
                "provenance": (
                    evaluator_provenance
                    if capability_id == "sealed_webarena_evaluator"
                    else {}
                ),
            },
        )
        capabilities[capability_id] = {
            "capability_id": capability_id,
            "plane": plane,
            "implementation_id": f"measured-{capability_id}",
            "implementation_version": "v1",
            "source_relative_path": SOURCE_RELATIVE,
            "source_sha256": source_sha256,
            "readiness_evidence_path": str(evidence.relative_to(tmp_path)),
            "readiness_evidence_sha256": sha256_file(evidence),
            "runtime_return_contract": return_contract,
            "frozen": True,
        }
    return {
        "schema_version": LIVE_DEPLOYMENT_SCHEMA_VERSION,
        "record_type": LIVE_DEPLOYMENT_RECORD_TYPE,
        "protocol_id": "table2-pc01-pilot-v1",
        "model_id": "qwen2vl_2b_gold_v2_8_dgx:epoch6:seed42",
        "model_seed": 42,
        "campaign_scope": "PROVISIONAL_ENGINEERING_PILOT",
        "evidence_label": "PILOT_ONLY",
        "paper_table_status": "N/R",
        "benchmark": "webarena",
        "benchmark_version": REGISTERED_BROWSERGYM_VERSION,
        "credential_material_embedded": False,
        "locked_test_rows_read": 0,
        "browsergym_validation_boundary": {
            "browsergym_version": REGISTERED_BROWSERGYM_VERSION,
            "execution_path": REGISTERED_VALIDATION_DISABLED_EXECUTION_PATH,
            "default_step_called": False,
            "task_validation_called": False,
            "reward_slot_read": False,
            "termination_slots_read": False,
            "info_slot_read": False,
            "oracle_output_exposed_to_runtime": False,
            "upstream_start_state_fields": list(
                UPSTREAM_WEBARENA_START_STATE_FIELDS
            ),
            "generic_webarena_task_used": False,
            "adapter_close_deferred_until_sealed_finalization": True,
            "cleanup_on_final_success": True,
            "cleanup_on_final_error": True,
            "cleanup_on_runtime_abort": True,
        },
        "sealed_page_broker": {
            "contract": REGISTERED_SEALED_BROKER_CONTRACT,
            "source_relative_path": SOURCE_RELATIVE,
            "source_sha256": source_sha256,
            "runtime_can_evaluate": False,
            "evaluator_can_publish": False,
            "live_page_returned_to_runtime": False,
            "evaluation_result_return_contract": "opaque_terminal_signal_only",
            "start_state_contract": REGISTERED_START_STATE_CONTRACT,
            "generic_webarena_task_allowed": False,
            "adapter_close_deferred_until_sealed_finalization": True,
            "cleanup_on_final_success": True,
            "cleanup_on_final_error": True,
            "cleanup_on_runtime_abort": True,
        },
        "capabilities": capabilities,
    }


def test_complete_measured_deployment_manifest_validates(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    assert validate_pc01_live_deployment_manifest(
        value,
        repository_root=ROOT,
        evidence_root=tmp_path,
    ) == value


def test_staging_copies_only_transitively_referenced_live_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    value = _manifest(source)
    manifest = _write_json(source / "deployment.json", value)
    _write_bytes(source / "credentials.json", b"must-not-be-copied")
    destination = tmp_path / "handoff"

    staged = stage_pc01_live_deployment_package(
        manifest_path=manifest,
        evidence_root=source,
        repository_root=ROOT,
        destination_artifact_root=destination,
    )
    assert not (staged.package_root / "credentials.json").exists()
    assert staged.binding["manifest_sha256"] == sha256_file(
        staged.package_root / "manifest.json"
    )
    environment = {LIVE_DEPLOYMENT_BINDING_FIELD: staged.binding}
    reopened = validate_bound_pc01_live_deployment(
        environment,
        artifact_root=destination,
        repository_root=ROOT,
    )
    assert reopened.binding == staged.binding
    assert len(reopened.capability_source_files) == 1


def test_staged_live_evidence_rejects_tampering_and_unreferenced_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    manifest = _write_json(source / "deployment.json", _manifest(source))
    destination = tmp_path / "handoff"
    staged = stage_pc01_live_deployment_package(
        manifest_path=manifest,
        evidence_root=source,
        repository_root=ROOT,
        destination_artifact_root=destination,
    )
    environment = {LIVE_DEPLOYMENT_BINDING_FIELD: staged.binding}
    extra = _write_json(staged.package_root / "unreferenced.json", {"secret": True})
    with pytest.raises(SchemaError, match="unreferenced evidence"):
        validate_bound_pc01_live_deployment(
            environment,
            artifact_root=destination,
            repository_root=ROOT,
        )
    extra.unlink()
    readiness = next(
        path
        for path in staged.package_files
        if path.name == "deterministic_reset.json"
    )
    readiness.write_bytes(readiness.read_bytes() + b" ")
    with pytest.raises(SchemaError, match="readiness evidence hash differs"):
        validate_bound_pc01_live_deployment(
            environment,
            artifact_root=destination,
            repository_root=ROOT,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value["capabilities"].pop("deterministic_reset"),
            "seven registered capabilities",
        ),
        (
            lambda value: value["browsergym_validation_boundary"].update(
                {"task_validation_called": True}
            ),
            "validation-disabled execution",
        ),
        (
            lambda value: value["sealed_page_broker"].update(
                {"runtime_can_evaluate": True}
            ),
            "runtime_can_evaluate",
        ),
        (
            lambda value: value["capabilities"]["recovery_action_planner"].update(
                {"implementation_id": "TBD placeholder"}
            ),
            "non-production marker",
        ),
    ),
)
def test_manifest_fails_closed_on_missing_or_weakened_capabilities(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    value = _manifest(tmp_path)
    mutation(value)
    with pytest.raises(SchemaError, match=message):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_readiness_claims_and_evidence_bytes_are_not_self_attested(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    row = value["capabilities"]["exclusive_input_audit"]
    evidence = tmp_path / row["readiness_evidence_path"]
    evidence_value = json.loads(evidence.read_text(encoding="utf-8"))
    evidence_value["claims"]["manual_input_event_count"] = 1
    _write_json(evidence, evidence_value)

    with pytest.raises(SchemaError, match="readiness evidence hash differs"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )

    row["readiness_evidence_sha256"] = sha256_file(evidence)
    with pytest.raises(SchemaError, match="readiness claims differ"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_evaluator_capability_cannot_return_oracle_values_to_runtime(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    row = value["capabilities"]["sealed_webarena_evaluator"]
    evidence = tmp_path / row["readiness_evidence_path"]
    evidence_value = json.loads(evidence.read_text(encoding="utf-8"))
    evidence_value["oracle_values_returned_to_runtime"] = 1
    _write_json(evidence, evidence_value)
    row["readiness_evidence_sha256"] = sha256_file(evidence)
    with pytest.raises(SchemaError, match="oracle_values_returned_to_runtime mismatch"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        (
            "implementation_classification",
            "official_webarena_evaluator",
            "implementation_classification",
        ),
        ("judge_model_id", "newer-judge", "judge_model_id"),
        ("judge_api_preflight_status", "UNAVAILABLE", "judge_api_preflight_status"),
        ("runtime_fallback_allowed", True, "runtime_fallback_allowed"),
    ),
)
def test_evaluator_provenance_forbids_bare_official_claim_or_judge_fallback(
    tmp_path: Path,
    field: str,
    replacement,
    message: str,
) -> None:
    value = _manifest(tmp_path)
    row = value["capabilities"]["sealed_webarena_evaluator"]
    evidence = tmp_path / row["readiness_evidence_path"]
    evidence_value = json.loads(evidence.read_text(encoding="utf-8"))
    evidence_value["provenance"][field] = replacement
    _write_json(evidence, evidence_value)
    row["readiness_evidence_sha256"] = sha256_file(evidence)
    with pytest.raises(SchemaError, match=message):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_evaluator_compatibility_review_and_api_receipt_bytes_are_required(
    tmp_path: Path,
) -> None:
    value = _manifest(tmp_path)
    row = value["capabilities"]["sealed_webarena_evaluator"]
    evidence = tmp_path / row["readiness_evidence_path"]
    evidence_value = json.loads(evidence.read_text(encoding="utf-8"))
    provenance = evidence_value["provenance"]
    review = tmp_path / provenance["compatibility_review_evidence_path"]
    review.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="compatibility_review_sha256 differs"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_measured_api_receipt_cannot_silently_use_a_newer_judge(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    row = value["capabilities"]["sealed_webarena_evaluator"]
    readiness = tmp_path / row["readiness_evidence_path"]
    readiness_value = json.loads(readiness.read_text(encoding="utf-8"))
    provenance = readiness_value["provenance"]
    receipt = tmp_path / provenance["judge_api_availability_receipt_evidence_path"]
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_value["model_id"] = "newer-judge"
    _write_json(receipt, receipt_value)
    provenance["judge_api_availability_receipt_sha256"] = sha256_file(receipt)
    _write_json(readiness, readiness_value)
    row["readiness_evidence_sha256"] = sha256_file(readiness)
    with pytest.raises(SchemaError, match="availability receipt model_id mismatch"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_manifest_rejects_boolean_zero_for_strict_leakage_counts(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    value["locked_test_rows_read"] = False
    with pytest.raises(SchemaError, match="locked_test_rows_read mismatch"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )
