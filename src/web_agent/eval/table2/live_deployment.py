"""Strict readiness contract for the deployment-owned PC-01 live services.

The checkpoint/runtime bridge is repository code, but a real Table 2 episode
also depends on measured browser, reset, controller, recovery, measurement and
evaluator services.  This module validates their source and readiness evidence
without inventing those services or accepting a fixture as production proof.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
import shutil
from typing import Any

from .common import (
    SchemaError,
    canonical_json_bytes,
    read_json,
    safe_relative_path,
    sha256_file,
    sha256_json,
)


LIVE_DEPLOYMENT_SCHEMA_VERSION = "table2-pc01-live-deployment-v1"
LIVE_CAPABILITY_EVIDENCE_SCHEMA_VERSION = "table2-live-capability-evidence-v1"
LIVE_DEPLOYMENT_RECORD_TYPE = "PC01LiveDeploymentCapabilityManifest"
LIVE_CAPABILITY_EVIDENCE_RECORD_TYPE = "PC01LiveCapabilityReadinessEvidence"
LIVE_DEPLOYMENT_BINDING_SCHEMA_VERSION = "table2-live-deployment-binding-v1"
LIVE_DEPLOYMENT_BINDING_FIELD = "pc01_live_deployment"
LIVE_DEPLOYMENT_PACKAGE_RELATIVE_ROOT = "live_deployment"
LIVE_DEPLOYMENT_MANIFEST_RELATIVE_PATH = "live_deployment/manifest.json"
REGISTERED_BROWSERGYM_VERSION = "0.14.3"
REGISTERED_VALIDATION_DISABLED_EXECUTION_PATH = (
    "BrowserEnv.pre_step+execute_python_code+post_step(validate=False)"
)
REGISTERED_SEALED_BROKER_CONTRACT = "one-way-sealed-page-broker-v1"
REGISTERED_START_STATE_CONTRACT = "upstream-six-field-webarena-start-state-v1"
PINNED_UPSTREAM_WEBARENA_REPOSITORY = (
    "https://github.com/web-arena-x/webarena"
)
PINNED_UPSTREAM_WEBARENA_REVISION = (
    "c6475f0e9affe5252a2966e26b8cb4c834a4ae40"
)
PINNED_UPSTREAM_EVALUATORS_SHA256 = (
    "6cc7b68d8200a349bb959e80855cc42539e6d8e03b646f5a0f27262ed7c970fa"
)
PINNED_UPSTREAM_JUDGE_SOURCE_SHA256 = (
    "ec758da4f82e5659e95805ab23222cfd7c4a0f2a67f64a8f85bac9626997e275"
)
PINNED_LIBWEBARENA_VERSION = "0.0.4"
PINNED_LIBWEBARENA_WHEEL_SHA256 = (
    "9ebee3b4371502c4f0f7e727a72e5846235d6750d420db9a3b8a168107654feb"
)
PINNED_WEBARENA_JUDGE_MODEL_ID = "gpt-4-1106-preview"
PINNED_WEBARENA_JUDGE_DECODING: Mapping[str, Any] = {
    "temperature": 0,
    "max_tokens": 768,
    "top_p": 1.0,
    "context_length": 0,
}
PINNED_WEBARENA_JUDGE_TASK_COUNT = 13
PINNED_WEBARENA_JUDGE_TASK_SET_SHA256 = (
    "e69ac4c48dfddb0ccf4473261be0772e33302edf2065e502b0e65e9b3b7dcb14"
)
PINNED_WEBARENA_STRING_MATCH_TASK_COUNT = 47
PINNED_WEBARENA_STRING_MATCH_TASK_SET_SHA256 = (
    "c7f57c3c48d1c5b1d2f1c269c1a39b218efdca8ccb3767167ae669adf6e333de"
)
UPSTREAM_WEBARENA_START_STATE_FIELDS = (
    "sites",
    "start_url",
    "require_login",
    "storage_state",
    "geolocation",
    "require_reset",
)


REQUIRED_LIVE_CAPABILITIES: Mapping[str, tuple[str, str]] = {
    "deterministic_reset": ("runtime_control", "hashes_only_receipt"),
    "exclusive_input_audit": ("runtime_control", "hashes_only_receipt"),
    "oracle_blind_browser_mapping": ("runtime", "causal_observation_only"),
    "action_safety_fault_classification": (
        "runtime",
        "registered_decision_only",
    ),
    "recovery_action_planner": ("runtime", "concrete_action_only"),
    "efficiency_measurement": ("runtime", "registered_metrics_only"),
    "sealed_webarena_evaluator": (
        "sealed_evaluator",
        "opaque_terminal_signal_only",
    ),
}


REQUIRED_CAPABILITY_CLAIMS: Mapping[str, Mapping[str, Any]] = {
    "deterministic_reset": {
        "reset_applied": True,
        "matched_block_state_committed": True,
        "commitment_roles": ["account", "database", "service", "start_state"],
        "oracle_labels_observed": False,
    },
    "exclusive_input_audit": {
        "exclusive_automation_control": True,
        "manual_input_event_count": 0,
        "manual_rescue_detected": False,
    },
    "oracle_blind_browser_mapping": {
        "live_reset_passed": True,
        "browsergym_default_step_called": False,
        "browsergym_task_validation_called": False,
        "reward_slot_read": False,
        "termination_slots_read": False,
        "info_slot_read": False,
        "upstream_start_state_fields_applied": list(
            UPSTREAM_WEBARENA_START_STATE_FIELDS
        ),
        "generic_webarena_task_used": False,
        "adapter_close_deferred_until_sealed_finalization": True,
    },
    "action_safety_fault_classification": {
        "unknown_action_fails_closed": True,
        "unknown_fault_fails_closed": True,
        "destructive_actions_denied": True,
    },
    "recovery_action_planner": {
        "strategies_supported": ["ALTERNATIVE_TARGET", "REPLAN"],
        "observation_bound_output": True,
        "oracle_inputs_observed": False,
    },
    "efficiency_measurement": {
        "registered_fields_complete": True,
        "model_call_ledger_reconciled": True,
        "oracle_inputs_observed": False,
    },
    "sealed_webarena_evaluator": {
        "evaluator_label": "source-reviewed-libwebarena-compatibility-port",
        "exact_official_source": False,
        "compatibility_review_required": True,
        "write_only_sealed_capability": True,
        "live_page_returned_to_runtime": False,
        "runtime_return_contract": "opaque_terminal_signal_only",
        "cleanup_on_final_success": True,
        "cleanup_on_final_error": True,
        "cleanup_on_runtime_abort": True,
    },
}


_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "protocol_id",
        "model_id",
        "model_seed",
        "campaign_scope",
        "evidence_label",
        "paper_table_status",
        "benchmark",
        "benchmark_version",
        "credential_material_embedded",
        "locked_test_rows_read",
        "browsergym_validation_boundary",
        "sealed_page_broker",
        "capabilities",
    }
)
_CAPABILITY_FIELDS = frozenset(
    {
        "capability_id",
        "plane",
        "implementation_id",
        "implementation_version",
        "source_relative_path",
        "source_sha256",
        "readiness_evidence_path",
        "readiness_evidence_sha256",
        "runtime_return_contract",
        "frozen",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "capability_id",
        "status",
        "implementation_source_sha256",
        "deployment_state_sha256",
        "measured_at_utc",
        "locked_test_rows_read",
        "oracle_values_returned_to_runtime",
        "claims",
        "provenance",
    }
)
_SEALED_EVALUATOR_PROVENANCE_FIELDS = frozenset(
    {
        "upstream_repository",
        "upstream_revision",
        "upstream_evaluator_source_sha256",
        "upstream_judge_source_sha256",
        "runtime_distribution",
        "runtime_distribution_version",
        "runtime_distribution_sha256",
        "runtime_evaluator_module",
        "runtime_evaluator_module_evidence_path",
        "runtime_evaluator_module_sha256",
        "runtime_judge_module_evidence_path",
        "runtime_judge_module_sha256",
        "implementation_classification",
        "byte_identical_to_upstream",
        "compatibility_delta_evidence_path",
        "compatibility_delta_sha256",
        "compatibility_review_evidence_path",
        "compatibility_review_sha256",
        "evaluator_configuration_evidence_path",
        "evaluator_configuration_sha256",
        "judge_model_id",
        "judge_prompt_source_sha256",
        "judge_decoding_parameters",
        "judge_required_task_count",
        "judge_required_task_set_sha256",
        "string_match_task_count",
        "string_match_task_set_sha256",
        "judge_api_response_schema_evidence_path",
        "judge_api_response_schema_sha256",
        "judge_api_availability_receipt_evidence_path",
        "judge_api_availability_receipt_sha256",
        "judge_api_preflight_status",
        "judge_model_substitution_allowed",
        "runtime_fallback_allowed",
        "unavailable_task_policy",
        "judge_outputs_returned_to_runtime",
    }
)
_BOUNDARY_FIELDS = frozenset(
    {
        "browsergym_version",
        "execution_path",
        "default_step_called",
        "task_validation_called",
        "reward_slot_read",
        "termination_slots_read",
        "info_slot_read",
        "oracle_output_exposed_to_runtime",
        "upstream_start_state_fields",
        "generic_webarena_task_used",
        "adapter_close_deferred_until_sealed_finalization",
        "cleanup_on_final_success",
        "cleanup_on_final_error",
        "cleanup_on_runtime_abort",
    }
)
_BROKER_FIELDS = frozenset(
    {
        "contract",
        "source_relative_path",
        "source_sha256",
        "runtime_can_evaluate",
        "evaluator_can_publish",
        "live_page_returned_to_runtime",
        "evaluation_result_return_contract",
        "start_state_contract",
        "generic_webarena_task_allowed",
        "adapter_close_deferred_until_sealed_finalization",
        "cleanup_on_final_success",
        "cleanup_on_final_error",
        "cleanup_on_runtime_abort",
    }
)
_JUDGE_AVAILABILITY_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "model_id",
        "availability_status",
        "decoding_parameters",
        "prompt_source_sha256",
        "response_schema_sha256",
        "request_succeeded",
        "response_conformed_to_schema",
        "credential_material_embedded",
        "evaluator_output_returned_to_runtime",
        "response_content_sha256",
        "measured_at_utc",
    }
)
_LIVE_DEPLOYMENT_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "package_root_relative_path",
        "manifest_relative_path",
        "manifest_sha256",
        "manifest_content_sha256",
        "package_files",
        "package_file_set_sha256",
        "capability_source_files",
        "capability_source_set_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class ValidatedPC01LiveDeployment:
    """A byte-exact live-deployment package reopened from an artifact root."""

    binding: dict[str, Any]
    manifest: dict[str, Any]
    package_root: Path
    package_files: tuple[Path, ...]
    capability_source_files: tuple[Path, ...]


def _exact_json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's ``False == 0`` coercion."""

    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _sha256(value: object, *, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _nonzero_sha256(value: object, *, field: str) -> str:
    digest = _sha256(value, field=field)
    if digest == "0" * 64:
        raise SchemaError(f"{field} cannot be an empty SHA-256 sentinel")
    return digest


def _measured_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise SchemaError(f"{field} must be nonempty measured text")
    normalized = value.strip().casefold()
    if any(marker in normalized for marker in ("placeholder", "fixture", "tbd", "unknown")):
        raise SchemaError(f"{field} contains a non-production marker")
    return value.strip()


def _source_path(
    repository_root: Path,
    relative_value: object,
    expected_sha256: object,
    *,
    field: str,
) -> Path:
    relative = safe_relative_path(str(relative_value or ""))
    unresolved = repository_root / relative
    source = unresolved.resolve()
    if (
        unresolved.is_symlink()
        or repository_root not in source.parents
        or not source.is_file()
    ):
        raise SchemaError(f"{field} source is unsafe or missing: {relative}")
    if source.suffix != ".py":
        raise SchemaError(f"{field} source must be an attested Python module")
    if sha256_file(source) != _sha256(expected_sha256, field=f"{field}.source_sha256"):
        raise SchemaError(f"{field} source hash differs from repository bytes")
    return source


def _evidence_path(evidence_root: Path, value: object, *, field: str) -> Path:
    relative = safe_relative_path(str(value or ""))
    if PurePosixPath(str(relative)).suffix != ".json":
        raise SchemaError(f"{field} must name a JSON evidence record")
    unresolved = evidence_root / relative
    path = unresolved.resolve()
    if (
        unresolved.is_symlink()
        or evidence_root not in path.parents
        or not path.is_file()
    ):
        raise SchemaError(f"{field} is unsafe or missing: {relative}")
    return path


def _evidence_artifact(
    evidence_root: Path,
    value: Mapping[str, Any],
    *,
    path_field: str,
    sha256_field: str,
) -> Path:
    relative = safe_relative_path(str(value.get(path_field) or ""))
    unresolved = evidence_root / relative
    path = unresolved.resolve()
    if (
        unresolved.is_symlink()
        or evidence_root not in path.parents
        or not path.is_file()
        or path.stat().st_size <= 0
    ):
        raise SchemaError(f"{path_field} is unsafe, missing, or empty: {relative}")
    expected = _nonzero_sha256(value.get(sha256_field), field=sha256_field)
    if sha256_file(path) != expected:
        raise SchemaError(f"{sha256_field} differs from evidence bytes")
    return path


def _parse_measured_at(value: object, *, field: str) -> datetime:
    captured = str(value or "")
    try:
        parsed = datetime.fromisoformat(captured.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaError(f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise SchemaError(f"{field} lacks a timezone")
    return parsed


def _validate_judge_availability_receipt(
    path: Path,
    *,
    response_schema_sha256: str,
) -> None:
    value = read_json(path)
    if set(value) != _JUDGE_AVAILABILITY_RECEIPT_FIELDS:
        raise SchemaError("judge API availability receipt schema has extra/missing fields")
    expected = {
        "schema_version": "table2-webarena-judge-api-preflight-v1",
        "record_type": "WebArenaJudgeAPIPreflightReceipt",
        "model_id": PINNED_WEBARENA_JUDGE_MODEL_ID,
        "availability_status": "PASS",
        "decoding_parameters": dict(PINNED_WEBARENA_JUDGE_DECODING),
        "prompt_source_sha256": PINNED_UPSTREAM_JUDGE_SOURCE_SHA256,
        "response_schema_sha256": response_schema_sha256,
        "request_succeeded": True,
        "response_conformed_to_schema": True,
        "credential_material_embedded": False,
        "evaluator_output_returned_to_runtime": False,
    }
    for field, expected_value in expected.items():
        if not _exact_json_equal(value.get(field), expected_value):
            raise SchemaError(f"judge API availability receipt {field} mismatch")
    _nonzero_sha256(
        value.get("response_content_sha256"),
        field="judge_api.response_content_sha256",
    )
    _parse_measured_at(value.get("measured_at_utc"), field="judge_api.measured_at_utc")


def _validate_evaluator_provenance(
    value: object,
    *,
    evidence_root: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SEALED_EVALUATOR_PROVENANCE_FIELDS:
        raise SchemaError(
            "sealed WebArena evaluator provenance schema has extra/missing fields"
        )
    provenance = dict(value)
    exact = {
        "upstream_repository": PINNED_UPSTREAM_WEBARENA_REPOSITORY,
        "upstream_revision": PINNED_UPSTREAM_WEBARENA_REVISION,
        "upstream_evaluator_source_sha256": PINNED_UPSTREAM_EVALUATORS_SHA256,
        "upstream_judge_source_sha256": PINNED_UPSTREAM_JUDGE_SOURCE_SHA256,
        "runtime_distribution": "libwebarena",
        "runtime_distribution_version": PINNED_LIBWEBARENA_VERSION,
        "runtime_distribution_sha256": PINNED_LIBWEBARENA_WHEEL_SHA256,
        "runtime_evaluator_module": "webarena.evaluation_harness.evaluators",
        "implementation_classification": "reviewed_compatibility_port",
        "byte_identical_to_upstream": False,
        "judge_model_id": PINNED_WEBARENA_JUDGE_MODEL_ID,
        "judge_prompt_source_sha256": PINNED_UPSTREAM_JUDGE_SOURCE_SHA256,
        "judge_decoding_parameters": dict(PINNED_WEBARENA_JUDGE_DECODING),
        "judge_required_task_count": PINNED_WEBARENA_JUDGE_TASK_COUNT,
        "judge_required_task_set_sha256": (
            PINNED_WEBARENA_JUDGE_TASK_SET_SHA256
        ),
        "string_match_task_count": PINNED_WEBARENA_STRING_MATCH_TASK_COUNT,
        "string_match_task_set_sha256": (
            PINNED_WEBARENA_STRING_MATCH_TASK_SET_SHA256
        ),
        "judge_api_preflight_status": "PASS",
        "judge_model_substitution_allowed": False,
        "runtime_fallback_allowed": False,
        "unavailable_task_policy": (
            "preregistered_exclusion_or_disclosed_evaluator_change_before_outcomes"
        ),
        "judge_outputs_returned_to_runtime": 0,
    }
    for field, expected in exact.items():
        if not _exact_json_equal(provenance.get(field), expected):
            raise SchemaError(f"sealed WebArena evaluator provenance {field} mismatch")

    runtime_module = _evidence_artifact(
        evidence_root,
        provenance,
        path_field="runtime_evaluator_module_evidence_path",
        sha256_field="runtime_evaluator_module_sha256",
    )
    _evidence_artifact(
        evidence_root,
        provenance,
        path_field="runtime_judge_module_evidence_path",
        sha256_field="runtime_judge_module_sha256",
    )
    if sha256_file(runtime_module) == PINNED_UPSTREAM_EVALUATORS_SHA256:
        raise SchemaError(
            "libwebarena evaluator was labelled a compatibility port but is byte-identical"
        )
    for stem in (
        "compatibility_delta",
        "compatibility_review",
        "evaluator_configuration",
        "judge_api_response_schema",
    ):
        _evidence_artifact(
            evidence_root,
            provenance,
            path_field=f"{stem}_evidence_path",
            sha256_field=f"{stem}_sha256",
        )
    receipt_path = _evidence_artifact(
        evidence_root,
        provenance,
        path_field="judge_api_availability_receipt_evidence_path",
        sha256_field="judge_api_availability_receipt_sha256",
    )
    _validate_judge_availability_receipt(
        receipt_path,
        response_schema_sha256=_nonzero_sha256(
            provenance.get("judge_api_response_schema_sha256"),
            field="judge_api_response_schema_sha256",
        ),
    )
    return provenance


def _validate_browsergym_boundary(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _BOUNDARY_FIELDS:
        raise SchemaError("live deployment has an invalid BrowserGym boundary schema")
    result = dict(value)
    expected = {
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
    }
    if not _exact_json_equal(result, expected):
        raise SchemaError(
            "BrowserGym runtime must use validation-disabled execution without "
            "reading reward, termination, info, or evaluator output"
        )
    return result


def _validate_broker(
    value: object,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _BROKER_FIELDS:
        raise SchemaError("live deployment has an invalid sealed-page broker schema")
    result = dict(value)
    expected = {
        "contract": REGISTERED_SEALED_BROKER_CONTRACT,
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
    }
    for field, expected_value in expected.items():
        if not _exact_json_equal(result.get(field), expected_value):
            raise SchemaError(f"sealed-page broker violates {field}")
    _source_path(
        repository_root,
        result.get("source_relative_path"),
        result.get("source_sha256"),
        field="sealed_page_broker",
    )
    return result


def _validate_readiness_evidence(
    path: Path,
    *,
    capability_id: str,
    source_sha256: str,
    evidence_root: Path,
) -> dict[str, Any]:
    value = read_json(path)
    if set(value) != _EVIDENCE_FIELDS:
        raise SchemaError(f"{capability_id} readiness evidence schema has extra/missing fields")
    expected_identity = {
        "schema_version": LIVE_CAPABILITY_EVIDENCE_SCHEMA_VERSION,
        "record_type": LIVE_CAPABILITY_EVIDENCE_RECORD_TYPE,
        "capability_id": capability_id,
        "status": "PASS",
        "implementation_source_sha256": source_sha256,
        "locked_test_rows_read": 0,
        "oracle_values_returned_to_runtime": 0,
    }
    for field, expected in expected_identity.items():
        if not _exact_json_equal(value.get(field), expected):
            raise SchemaError(f"{capability_id} readiness evidence {field} mismatch")
    _sha256(
        value.get("deployment_state_sha256"),
        field=f"{capability_id}.deployment_state_sha256",
    )
    _parse_measured_at(
        value.get("measured_at_utc"),
        field=f"{capability_id} measured_at_utc",
    )
    claims = value.get("claims")
    if not _exact_json_equal(claims, REQUIRED_CAPABILITY_CLAIMS[capability_id]):
        raise SchemaError(f"{capability_id} readiness claims differ from registration")
    provenance = value.get("provenance")
    if capability_id == "sealed_webarena_evaluator":
        _validate_evaluator_provenance(provenance, evidence_root=evidence_root)
    elif not _exact_json_equal(provenance, {}):
        raise SchemaError(f"{capability_id} cannot claim evaluator provenance")
    return value


def validate_pc01_live_deployment_manifest(
    value: Mapping[str, Any],
    *,
    repository_root: str | Path,
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Validate a complete measured live deployment or fail closed.

    The function deliberately has no partial/readiness-warning mode.  A caller
    that lacks any real capability or evidence byte cannot turn this validator
    into permission to run the pilot.
    """

    if not isinstance(value, Mapping) or set(value) != _MANIFEST_FIELDS:
        raise SchemaError("PC-01 live deployment manifest schema has extra/missing fields")
    manifest = dict(value)
    fixed = {
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
    }
    for field, expected in fixed.items():
        if not _exact_json_equal(manifest.get(field), expected):
            raise SchemaError(f"PC-01 live deployment {field} mismatch")

    repo = Path(repository_root).resolve()
    evidence = Path(evidence_root).resolve()
    if not repo.is_dir() or not evidence.is_dir():
        raise SchemaError("live deployment repository/evidence roots must exist")
    _validate_browsergym_boundary(manifest.get("browsergym_validation_boundary"))
    _validate_broker(manifest.get("sealed_page_broker"), repository_root=repo)

    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, Mapping) or set(capabilities) != set(
        REQUIRED_LIVE_CAPABILITIES
    ):
        raise SchemaError("live deployment must cover the seven registered capabilities exactly")
    validated: dict[str, Any] = {}
    for capability_id, (expected_plane, expected_return) in sorted(
        REQUIRED_LIVE_CAPABILITIES.items()
    ):
        row = capabilities[capability_id]
        if not isinstance(row, Mapping) or set(row) != _CAPABILITY_FIELDS:
            raise SchemaError(f"{capability_id} capability schema has extra/missing fields")
        item = dict(row)
        exact = {
            "capability_id": capability_id,
            "plane": expected_plane,
            "runtime_return_contract": expected_return,
            "frozen": True,
        }
        for field, expected in exact.items():
            if not _exact_json_equal(item.get(field), expected):
                raise SchemaError(f"{capability_id} {field} mismatch")
        _measured_text(item.get("implementation_id"), field=f"{capability_id}.implementation_id")
        _measured_text(
            item.get("implementation_version"),
            field=f"{capability_id}.implementation_version",
        )
        source_sha256 = _sha256(
            item.get("source_sha256"), field=f"{capability_id}.source_sha256"
        )
        _source_path(
            repo,
            item.get("source_relative_path"),
            source_sha256,
            field=capability_id,
        )
        readiness = _evidence_path(
            evidence,
            item.get("readiness_evidence_path"),
            field=f"{capability_id}.readiness_evidence_path",
        )
        if sha256_file(readiness) != _sha256(
            item.get("readiness_evidence_sha256"),
            field=f"{capability_id}.readiness_evidence_sha256",
        ):
            raise SchemaError(f"{capability_id} readiness evidence hash differs")
        _validate_readiness_evidence(
            readiness,
            capability_id=capability_id,
            source_sha256=source_sha256,
            evidence_root=evidence,
        )
        validated[capability_id] = item
    manifest["capabilities"] = validated
    return manifest


def load_pc01_live_deployment_manifest(
    path: str | Path,
    *,
    repository_root: str | Path,
    evidence_root: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise SchemaError("PC-01 live deployment manifest is unsafe or missing")
    value = read_json(source)
    return validate_pc01_live_deployment_manifest(
        value,
        repository_root=repository_root,
        evidence_root=(source.parent if evidence_root is None else evidence_root),
    )


def _referenced_evidence_relative_paths(
    manifest: Mapping[str, Any],
    *,
    evidence_root: Path,
) -> tuple[Path, ...]:
    """Return only evidence bytes transitively referenced by the manifest."""

    referenced: dict[str, Path] = {}
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise SchemaError("live deployment capabilities are absent")
    for capability_id in sorted(REQUIRED_LIVE_CAPABILITIES):
        row = capabilities.get(capability_id)
        if not isinstance(row, Mapping):
            raise SchemaError(f"{capability_id} capability is absent")
        readiness = _evidence_path(
            evidence_root,
            row.get("readiness_evidence_path"),
            field=f"{capability_id}.readiness_evidence_path",
        )
        relative = readiness.relative_to(evidence_root)
        referenced[relative.as_posix()] = relative
        if capability_id != "sealed_webarena_evaluator":
            continue
        evidence = read_json(readiness)
        provenance = evidence.get("provenance")
        if not isinstance(provenance, Mapping):
            raise SchemaError("sealed evaluator provenance is absent")
        for stem in (
            "runtime_evaluator_module",
            "runtime_judge_module",
            "compatibility_delta",
            "compatibility_review",
            "evaluator_configuration",
            "judge_api_response_schema",
            "judge_api_availability_receipt",
        ):
            artifact = _evidence_artifact(
                evidence_root,
                provenance,
                path_field=f"{stem}_evidence_path",
                sha256_field=f"{stem}_sha256",
            )
            artifact_relative = artifact.relative_to(evidence_root)
            referenced[artifact_relative.as_posix()] = artifact_relative
    if "manifest.json" in referenced:
        raise SchemaError(
            "live readiness evidence cannot collide with the staged manifest path"
        )
    return tuple(referenced[key] for key in sorted(referenced))


def _capability_source_rows(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: dict[str, str] = {}

    def add(relative_value: object, sha_value: object, *, field: str) -> None:
        relative = safe_relative_path(str(relative_value or "")).as_posix()
        digest = _sha256(sha_value, field=f"{field}.source_sha256")
        previous = rows.get(relative)
        if previous is not None and previous != digest:
            raise SchemaError(
                f"live deployment repeats source path with different hashes: {relative}"
            )
        rows[relative] = digest

    broker = manifest.get("sealed_page_broker")
    if not isinstance(broker, Mapping):
        raise SchemaError("live deployment sealed-page broker is absent")
    add(
        broker.get("source_relative_path"),
        broker.get("source_sha256"),
        field="sealed_page_broker",
    )
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise SchemaError("live deployment capabilities are absent")
    for capability_id in sorted(REQUIRED_LIVE_CAPABILITIES):
        row = capabilities.get(capability_id)
        if not isinstance(row, Mapping):
            raise SchemaError(f"{capability_id} capability is absent")
        add(
            row.get("source_relative_path"),
            row.get("source_sha256"),
            field=capability_id,
        )
    return [
        {"relative_path": relative, "sha256": rows[relative]}
        for relative in sorted(rows)
    ]


def _build_live_deployment_binding(
    *,
    manifest_path: Path,
    package_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_relatives = _referenced_evidence_relative_paths(
        manifest,
        evidence_root=package_root,
    )
    package_paths = (Path("manifest.json"), *evidence_relatives)
    package_rows = [
        {
            "relative_path": relative.as_posix(),
            "sha256": sha256_file(package_root / relative),
        }
        for relative in package_paths
    ]
    source_rows = _capability_source_rows(manifest)
    return {
        "schema_version": LIVE_DEPLOYMENT_BINDING_SCHEMA_VERSION,
        "package_root_relative_path": LIVE_DEPLOYMENT_PACKAGE_RELATIVE_ROOT,
        "manifest_relative_path": LIVE_DEPLOYMENT_MANIFEST_RELATIVE_PATH,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_content_sha256": sha256_json(dict(manifest)),
        "package_files": package_rows,
        "package_file_set_sha256": sha256_json(package_rows),
        "capability_source_files": source_rows,
        "capability_source_set_sha256": sha256_json(source_rows),
    }


def stage_pc01_live_deployment_package(
    *,
    manifest_path: str | Path,
    evidence_root: str | Path,
    repository_root: str | Path,
    destination_artifact_root: str | Path,
) -> ValidatedPC01LiveDeployment:
    """Stage exactly one validated manifest and its referenced evidence bytes.

    Unreferenced files in the operator evidence directory are deliberately not
    copied.  This both closes the identity boundary and prevents credentials,
    browser profiles, or unrelated diagnostics from entering a handoff.
    """

    repository = Path(repository_root).resolve()
    unresolved_manifest = Path(manifest_path)
    unresolved_evidence = Path(evidence_root)
    source_manifest = unresolved_manifest.resolve()
    source_evidence = unresolved_evidence.resolve()
    destination_root = Path(destination_artifact_root).resolve()
    if unresolved_manifest.is_symlink() or not source_manifest.is_file():
        raise SchemaError("PC-01 live deployment manifest is unsafe or missing")
    if not source_evidence.is_dir() or unresolved_evidence.is_symlink():
        raise SchemaError("PC-01 live deployment evidence root is unsafe or missing")
    manifest = load_pc01_live_deployment_manifest(
        source_manifest,
        repository_root=repository,
        evidence_root=source_evidence,
    )
    referenced = _referenced_evidence_relative_paths(
        manifest,
        evidence_root=source_evidence,
    )
    package_root = destination_root / LIVE_DEPLOYMENT_PACKAGE_RELATIVE_ROOT
    if package_root.exists():
        raise SchemaError("refusing to overwrite staged live-deployment package")
    package_root.mkdir(parents=True)
    staged_manifest = package_root / "manifest.json"
    shutil.copy2(source_manifest, staged_manifest)
    for relative in referenced:
        source = source_evidence / relative
        destination = package_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(destination) != sha256_file(source):
            raise SchemaError(
                f"staged live-deployment evidence changed: {relative.as_posix()}"
            )
    staged_manifest_value = load_pc01_live_deployment_manifest(
        staged_manifest,
        repository_root=repository,
        evidence_root=package_root,
    )
    if staged_manifest_value != manifest:
        raise SchemaError("staged live-deployment manifest content changed")
    binding = _build_live_deployment_binding(
        manifest_path=staged_manifest,
        package_root=package_root,
        manifest=staged_manifest_value,
    )
    return validate_pc01_live_deployment_binding(
        binding,
        artifact_root=destination_root,
        repository_root=repository,
    )


def validate_pc01_live_deployment_binding(
    value: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    repository_root: str | Path,
) -> ValidatedPC01LiveDeployment:
    """Reopen a staged/frozen package and reproduce every bound identity."""

    if not isinstance(value, Mapping) or set(value) != _LIVE_DEPLOYMENT_BINDING_FIELDS:
        raise SchemaError("live-deployment binding schema has extra/missing fields")
    binding = dict(value)
    if binding.get("schema_version") != LIVE_DEPLOYMENT_BINDING_SCHEMA_VERSION:
        raise SchemaError("live-deployment binding version is not registered")
    if binding.get("package_root_relative_path") != LIVE_DEPLOYMENT_PACKAGE_RELATIVE_ROOT:
        raise SchemaError("live-deployment package root is not registered")
    if binding.get("manifest_relative_path") != LIVE_DEPLOYMENT_MANIFEST_RELATIVE_PATH:
        raise SchemaError("live-deployment manifest path is not registered")
    root = Path(artifact_root).resolve()
    unresolved_package_root = root / LIVE_DEPLOYMENT_PACKAGE_RELATIVE_ROOT
    package_root = unresolved_package_root.resolve()
    if (
        root not in package_root.parents
        or unresolved_package_root.is_symlink()
        or not package_root.is_dir()
    ):
        raise SchemaError("frozen live-deployment package root is unsafe or missing")
    if any(candidate.is_symlink() for candidate in package_root.rglob("*")):
        raise SchemaError("frozen live-deployment package contains a symlink")
    manifest_path = package_root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SchemaError("frozen live-deployment manifest is unsafe or missing")
    manifest = load_pc01_live_deployment_manifest(
        manifest_path,
        repository_root=repository_root,
        evidence_root=package_root,
    )
    expected = _build_live_deployment_binding(
        manifest_path=manifest_path,
        package_root=package_root,
        manifest=manifest,
    )
    if not _exact_json_equal(binding, expected):
        raise SchemaError("live-deployment binding differs from frozen package bytes")
    expected_relatives = {
        str(row["relative_path"]) for row in expected["package_files"]
    }
    actual_relatives = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
    }
    if actual_relatives != expected_relatives:
        raise SchemaError(
            "live-deployment package contains missing or unreferenced evidence files"
        )
    source_files = tuple(
        (Path(repository_root).resolve() / row["relative_path"]).resolve()
        for row in expected["capability_source_files"]
    )
    return ValidatedPC01LiveDeployment(
        binding=expected,
        manifest=manifest,
        package_root=package_root,
        package_files=tuple(
            package_root / str(row["relative_path"])
            for row in expected["package_files"]
        ),
        capability_source_files=source_files,
    )


def validate_bound_pc01_live_deployment(
    environment: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    repository_root: str | Path,
) -> ValidatedPC01LiveDeployment:
    """Validate the live-deployment binding carried by an environment record."""

    value = environment.get(LIVE_DEPLOYMENT_BINDING_FIELD)
    if not isinstance(value, Mapping):
        raise SchemaError("environment lacks a bound PC-01 live-deployment package")
    return validate_pc01_live_deployment_binding(
        value,
        artifact_root=artifact_root,
        repository_root=repository_root,
    )
