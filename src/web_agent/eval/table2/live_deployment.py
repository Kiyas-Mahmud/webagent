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
import stat
from typing import Any

from .common import (
    SchemaError,
    canonical_json_bytes,
    read_json,
    safe_relative_path,
    sha256_file,
    sha256_json,
)
from .public_task_registry import PUBLIC_DEVELOPMENT_TASK_COUNT


LIVE_DEPLOYMENT_SCHEMA_VERSION = "table2-pc01-live-deployment-v2"
LIVE_CAPABILITY_EVIDENCE_SCHEMA_VERSION = "table2-live-capability-evidence-v2"
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
EVALUATOR_REQUIREMENTS_SCHEMA_VERSION = (
    "table2-webarena-evaluator-requirements-v2"
)
EVALUATOR_REQUIREMENTS_RECORD_TYPE = "WebArenaEvaluatorRequirements"
EVALUATOR_REQUIREMENTS_DERIVATION_CONTRACT = (
    "resolved-task-export-plus-interface-audit-and-page-state-compile-v2"
)
COMPATIBILITY_REVIEW_SCHEMA_VERSION = (
    "table2-webarena-compatibility-port-review-v2"
)
# This source-attested allowlist is the trust anchor for external evaluator
# review.  A review/parity package is campaign-ready only after an independent
# reviewer has supplied the final bytes and a later reviewed source commit has
# pinned the resulting binding digest here.  Self-hashes inside caller-supplied
# JSON are integrity fields, not authentication.  No external review has yet
# been registered, so the current compatibility port deliberately remains
# unpromotable.
EXTERNAL_EVALUATOR_REVIEW_TRUST_AUTHORITY_VERSION = (
    "table2-external-evaluator-review-binding-allowlist-v1"
)
PINNED_EXTERNAL_EVALUATOR_REVIEW_BINDING_SHA256S: frozenset[str] = frozenset()
JUDGE_API_NOT_APPLICABLE = "NOT_APPLICABLE_NO_JUDGE_REQUIRED_TASKS"
PAGE_STATE_EVALUATOR_TYPES = frozenset({"url_match", "program_html"})
STRING_MATCH_EVALUATOR_TYPE = "string_match"
JUDGE_ANSWER_MODE = "fuzzy_match"
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
        "evaluator_label": (
            "independently-reviewed-libwebarena-compatibility-port"
        ),
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
        "upstream_parity_reference_evidence_path",
        "upstream_parity_reference_sha256",
        "compatibility_parity_evidence_path",
        "compatibility_parity_sha256",
        "compatibility_review_evidence_path",
        "compatibility_review_sha256",
        "evaluator_configuration_evidence_path",
        "evaluator_configuration_sha256",
        "evaluator_requirements_evidence_path",
        "evaluator_requirements_sha256",
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
_EVALUATOR_REQUIREMENTS_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "derivation_contract",
        "registry_manifest_id",
        "registry_manifest_sha256",
        "resolved_task_set_sha256",
        "task_export_content_sha256",
        "task_interface_audit_content_sha256",
        "task_count",
        "tasks",
        "ordered_task_ids_sha256",
        "evaluator_types",
        "judge_required",
        "string_match",
        "interface_compatible",
        "interface_audit_status",
        "page_state_compile_report",
        "page_state_compile_report_content_sha256",
    }
)
_EVALUATOR_REQUIREMENT_TASK_FIELDS = frozenset(
    {"task_id", "upstream_index", "evaluator_types", "answer_matching_modes"}
)
_EVALUATOR_TYPE_REQUIREMENT_FIELDS = frozenset(
    {"evaluator_type", "task_count", "task_ids", "task_id_set_sha256"}
)
_TASK_SET_REQUIREMENT_FIELDS = frozenset(
    {"task_count", "task_ids", "task_id_set_sha256"}
)
_PAGE_STATE_COMPILE_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "compile_contract",
        "status",
        "all_exact_task_configs_compiled",
        "external_independent_review_required",
        "campaign_authority",
        "task_count",
        "criterion_count",
        "evaluator_id",
        "evaluator_version",
        "compiler_source_relative_path",
        "compiler_source_sha256",
        "exact_task_projection_sha256",
        "implementation_specific_report_status",
        "implementation_specific_compile_report",
        "implementation_specific_compile_report_content_sha256",
        "tasks",
        "report_sha256",
    }
)
_PAGE_STATE_COMPILE_TASK_FIELDS = frozenset(
    {
        "position",
        "task_id",
        "upstream_index",
        "benchmark_task_id",
        "task_projection_sha256",
        "evaluator_config_sha256",
        "eval_types",
        "criterion_count",
    }
)
_COMPATIBILITY_REVIEW_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "status",
        "review_scope",
        "evaluator_access_mode",
        "state_mutation_permitted",
        "evaluator_remapping_permitted",
        "task_outcomes_observed_before_freeze",
        "registry_manifest_sha256",
        "resolved_task_set_sha256",
        "evaluator_requirements_sha256",
        "page_state_compile_report_content_sha256",
        "page_state_compiler_source_sha256",
        "implementation_id",
        "implementation_version",
        "implementation_source_sha256",
        "implementation_classification",
        "compatibility_delta_sha256",
        "compatibility_parity_sha256",
        "review_authority_id",
        "review_authority_type",
        "reviewer_id",
        "reviewer_organization",
        "reviewer_role",
        "reviewer_independent_of_implementation",
        "reviewer_independent_of_campaign_execution",
        "reviewed_at_utc",
        "review_statement_sha256",
    }
)
_FINAL_BYTE_PARITY_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "status",
        "parity_scope",
        "implementation_id",
        "implementation_version",
        "implementation_source_sha256",
        "page_state_compiler_source_sha256",
        "upstream_reference_sha256",
        "case_count",
        "all_cases_matched",
        "state_mutation_observed",
        "oracle_output_returned_to_runtime",
        "task_outcomes_observed",
        "measured_at_utc",
        "parity_statement_sha256",
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
    evaluator_requirements: dict[str, Any]
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


def _task_set_requirement(task_ids: list[str]) -> dict[str, Any]:
    canonical_ids = sorted(task_ids)
    if len(canonical_ids) != len(set(canonical_ids)):
        raise SchemaError("evaluator requirement task IDs must be unique")
    return {
        "task_count": len(canonical_ids),
        "task_ids": canonical_ids,
        "task_id_set_sha256": sha256_json(canonical_ids),
    }


def _derived_evaluator_aggregates(
    task_rows: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], bool]:
    by_type: dict[str, list[str]] = {}
    judge_ids: list[str] = []
    for position, row in enumerate(task_rows):
        if set(row) != _EVALUATOR_REQUIREMENT_TASK_FIELDS:
            raise SchemaError(
                f"evaluator requirement task {position} fields differ from schema"
            )
        task_id = row.get("task_id")
        upstream_index = row.get("upstream_index")
        evaluator_types = row.get("evaluator_types")
        answer_modes = row.get("answer_matching_modes")
        if not isinstance(task_id, str) or not task_id.strip():
            raise SchemaError(f"evaluator requirement task {position} lacks a task ID")
        if type(upstream_index) is not int or upstream_index < 0:
            raise SchemaError(
                f"evaluator requirement task {position} has an invalid upstream index"
            )
        if (
            not isinstance(evaluator_types, list)
            or not evaluator_types
            or any(not isinstance(item, str) or not item for item in evaluator_types)
            or evaluator_types != sorted(set(evaluator_types))
        ):
            raise SchemaError(
                f"evaluator requirement task {position} has invalid evaluator types"
            )
        if (
            not isinstance(answer_modes, list)
            or any(not isinstance(item, str) or not item for item in answer_modes)
            or answer_modes != sorted(set(answer_modes))
        ):
            raise SchemaError(
                f"evaluator requirement task {position} has invalid answer modes"
            )
        if answer_modes and STRING_MATCH_EVALUATOR_TYPE not in evaluator_types:
            raise SchemaError(
                f"evaluator requirement task {position} has answer modes without "
                "string_match"
            )
        for evaluator_type in evaluator_types:
            by_type.setdefault(evaluator_type, []).append(task_id)
        if JUDGE_ANSWER_MODE in answer_modes:
            judge_ids.append(task_id)

    evaluator_requirements = [
        {
            "evaluator_type": evaluator_type,
            **_task_set_requirement(by_type[evaluator_type]),
        }
        for evaluator_type in sorted(by_type)
    ]
    string_match = _task_set_requirement(
        by_type.get(STRING_MATCH_EVALUATOR_TYPE, [])
    )
    judge_required = _task_set_requirement(judge_ids)
    interface_compatible = (
        set(by_type).issubset(PAGE_STATE_EVALUATOR_TYPES)
        and not string_match["task_count"]
        and not judge_required["task_count"]
    )
    return (
        evaluator_requirements,
        judge_required,
        string_match,
        interface_compatible,
    )


def validate_evaluator_requirements_artifact(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all evaluator counts and task-set hashes from per-task rows."""

    if not isinstance(value, Mapping) or set(value) != _EVALUATOR_REQUIREMENTS_FIELDS:
        raise SchemaError("evaluator requirements schema has extra/missing fields")
    result = dict(value)
    fixed = {
        "schema_version": EVALUATOR_REQUIREMENTS_SCHEMA_VERSION,
        "record_type": EVALUATOR_REQUIREMENTS_RECORD_TYPE,
        "derivation_contract": EVALUATOR_REQUIREMENTS_DERIVATION_CONTRACT,
        "task_count": PUBLIC_DEVELOPMENT_TASK_COUNT,
    }
    for field, expected in fixed.items():
        if not _exact_json_equal(result.get(field), expected):
            raise SchemaError(f"evaluator requirements {field} mismatch")
    registry_id = result.get("registry_manifest_id")
    if not isinstance(registry_id, str) or not registry_id.strip():
        raise SchemaError("evaluator requirements lack active registry identity")
    for field in (
        "registry_manifest_sha256",
        "resolved_task_set_sha256",
        "task_export_content_sha256",
        "task_interface_audit_content_sha256",
    ):
        _nonzero_sha256(result.get(field), field=f"evaluator_requirements.{field}")

    tasks = result.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != PUBLIC_DEVELOPMENT_TASK_COUNT
        or not all(isinstance(row, Mapping) for row in tasks)
    ):
        raise SchemaError("evaluator requirements must contain exactly 50 task rows")
    task_ids = [str(row.get("task_id") or "") for row in tasks]
    upstream_indices = [row.get("upstream_index") for row in tasks]
    if (
        any(not task_id for task_id in task_ids)
        or len(task_ids) != len(set(task_ids))
        or len(upstream_indices) != len(set(upstream_indices))
    ):
        raise SchemaError("evaluator requirement ordered task identities are invalid")
    if result.get("ordered_task_ids_sha256") != sha256_json(task_ids):
        raise SchemaError("evaluator requirements ordered task-ID hash mismatch")

    (
        expected_types,
        expected_judge,
        expected_string,
        expected_compatible,
    ) = _derived_evaluator_aggregates(tasks)
    supplied_types = result.get("evaluator_types")
    if not isinstance(supplied_types, list) or not all(
        isinstance(row, Mapping) and set(row) == _EVALUATOR_TYPE_REQUIREMENT_FIELDS
        for row in supplied_types
    ):
        raise SchemaError("evaluator type requirements schema is invalid")
    for label, supplied, expected in (
        ("evaluator types", supplied_types, expected_types),
        ("judge-required", result.get("judge_required"), expected_judge),
        ("string-match", result.get("string_match"), expected_string),
    ):
        if label != "evaluator types" and (
            not isinstance(supplied, Mapping)
            or set(supplied) != _TASK_SET_REQUIREMENT_FIELDS
        ):
            raise SchemaError(f"{label} evaluator requirement schema is invalid")
        if not _exact_json_equal(supplied, expected):
            raise SchemaError(
                f"{label} counts or task-ID set hashes differ from task rows"
            )
    if result.get("interface_compatible") is not expected_compatible:
        raise SchemaError("evaluator requirements interface compatibility mismatch")
    expected_status = "PASS" if expected_compatible else "FAIL"
    if result.get("interface_audit_status") != expected_status:
        raise SchemaError("evaluator requirements interface audit status mismatch")
    compile_report = result.get("page_state_compile_report")
    compile_sha256 = result.get("page_state_compile_report_content_sha256")
    if expected_compatible:
        if (
            not isinstance(compile_report, Mapping)
            or set(compile_report) != _PAGE_STATE_COMPILE_REPORT_FIELDS
        ):
            raise SchemaError(
                "compatible evaluator requirements lack the embedded 50-task "
                "page-state compile report"
            )
        unsigned = dict(compile_report)
        report_sha256 = unsigned.pop("report_sha256", None)
        if report_sha256 != sha256_json(unsigned):
            raise SchemaError("embedded page-state compile report hash mismatch")
        if compile_sha256 != sha256_json(compile_report):
            raise SchemaError(
                "embedded page-state compile report content hash mismatch"
            )
        if (
            compile_report.get("schema_version")
            != "table2-webarena-page-state-compile-authority-v1"
            or compile_report.get("record_type")
            != "WebArenaPageStateEvaluatorCompileAuthority"
            or compile_report.get("compile_contract")
            != "strict-read-only-page-state-config-compiler-exact-50-v1"
            or compile_report.get("task_count") != PUBLIC_DEVELOPMENT_TASK_COUNT
            or compile_report.get("status") != "COMPILED"
            or compile_report.get("all_exact_task_configs_compiled") is not True
            or compile_report.get("external_independent_review_required") is not True
            or compile_report.get("campaign_authority") is not False
        ):
            raise SchemaError(
                "embedded page-state compile report makes an invalid readiness claim"
            )
        _nonzero_sha256(
            compile_report.get("compiler_source_sha256"),
            field="page_state_compile_report.compiler_source_sha256",
        )
        _nonzero_sha256(
            compile_report.get("exact_task_projection_sha256"),
            field="page_state_compile_report.exact_task_projection_sha256",
        )
        if compile_report.get("compiler_source_relative_path") != (
            "src/web_agent/eval/table2/webarena_page_state_evaluator.py"
        ):
            raise SchemaError("page-state compile report compiler path mismatch")
        report_tasks = compile_report.get("tasks")
        if (
            not isinstance(report_tasks, list)
            or len(report_tasks) != PUBLIC_DEVELOPMENT_TASK_COUNT
            or not all(
                isinstance(row, Mapping)
                and set(row) == _PAGE_STATE_COMPILE_TASK_FIELDS
                for row in report_tasks
            )
            or [row.get("task_id") for row in report_tasks] != task_ids
            or [row.get("position") for row in report_tasks]
            != list(range(PUBLIC_DEVELOPMENT_TASK_COUNT))
        ):
            raise SchemaError(
                "embedded page-state compile report task identities differ"
            )
        for position, row in enumerate(report_tasks):
            for field in ("task_projection_sha256", "evaluator_config_sha256"):
                _nonzero_sha256(
                    row.get(field),
                    field=f"page_state_compile_report.tasks[{position}].{field}",
                )
            if type(row.get("criterion_count")) is not int or row[
                "criterion_count"
            ] <= 0:
                raise SchemaError(
                    "page-state compile report criterion counts must be positive"
                )
            requirement_row = tasks[position]
            if (
                row.get("upstream_index")
                != requirement_row.get("upstream_index")
                or row.get("eval_types")
                != requirement_row.get("evaluator_types")
            ):
                raise SchemaError(
                    "page-state compile report differs from evaluator "
                    f"requirements at task position {position}"
                )
        if compile_report.get("criterion_count") != sum(
            int(row["criterion_count"]) for row in report_tasks
        ):
            raise SchemaError("page-state compile report criterion total mismatch")
        for field in ("evaluator_id", "evaluator_version"):
            value = compile_report.get(field)
            if not isinstance(value, str) or not value.strip():
                raise SchemaError(
                    f"page_state_compile_report.{field} must be nonempty text"
                )
        implementation_status = compile_report.get(
            "implementation_specific_report_status"
        )
        implementation_report = compile_report.get(
            "implementation_specific_compile_report"
        )
        implementation_report_sha256 = compile_report.get(
            "implementation_specific_compile_report_content_sha256"
        )
        if implementation_status == "BOUND_PENDING_EXTERNAL_REVIEW_INPUT":
            if (
                not isinstance(implementation_report, Mapping)
                or implementation_report_sha256
                != sha256_json(implementation_report)
                or implementation_report.get("campaign_ready") is not False
                or implementation_report.get("external_review_status")
                != "PENDING"
            ):
                raise SchemaError(
                    "implementation-specific compile report lost its pending "
                    "review semantics"
                )
        elif implementation_status == (
            "NOT_APPLICABLE_DIFFERENT_EVALUATOR_IDENTITY"
        ):
            if implementation_report is not None or (
                implementation_report_sha256 is not None
            ):
                raise SchemaError(
                    "non-applicable implementation-specific compile report "
                    "must be null"
                )
        else:
            raise SchemaError(
                "page-state compile report has an unregistered implementation "
                "binding status"
            )
    elif compile_report is not None or compile_sha256 is not None:
        raise SchemaError(
            "interface-incompatible evaluator requirements cannot claim a "
            "page-state compile report"
        )
    return result


def build_evaluator_requirements_artifact(
    *,
    task_export: Mapping[str, Any],
    task_interface_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive evaluator requirements from an exact export/audit pair."""

    from .task_interface_audit import validate_webarena_task_interface_audit

    audit = validate_webarena_task_interface_audit(
        task_interface_audit,
        task_export=task_export,
    )
    task_rows = task_export.get("tasks")
    audit_rows = audit.get("tasks")
    if (
        not isinstance(task_rows, list)
        or not isinstance(audit_rows, list)
        or len(task_rows) != PUBLIC_DEVELOPMENT_TASK_COUNT
        or len(audit_rows) != PUBLIC_DEVELOPMENT_TASK_COUNT
    ):
        raise SchemaError("evaluator requirements require the exact 50-task export")
    if task_export.get("resolved_task_set_sha256") != sha256_json(task_rows):
        raise SchemaError("evaluator requirements task-export hash mismatch")
    registry_id = task_export.get("registry_manifest_id")
    if not isinstance(registry_id, str) or not registry_id.strip():
        raise SchemaError("task export lacks active registry identity")
    registry_sha256 = _nonzero_sha256(
        task_export.get("registry_manifest_sha256"),
        field="task_export.registry_manifest_sha256",
    )

    tasks: list[dict[str, Any]] = []
    for position, (task, audit_row) in enumerate(
        zip(task_rows, audit_rows, strict=True)
    ):
        if (
            not isinstance(task, Mapping)
            or not isinstance(audit_row, Mapping)
            or task.get("task_id") != audit_row.get("task_id")
            or task.get("upstream_index") != audit_row.get("upstream_index")
        ):
            raise SchemaError(
                f"evaluator requirements export/audit identity mismatch at {position}"
            )
        raw_types = audit_row.get("eval_types")
        raw_modes = audit_row.get("answer_matching_modes")
        if not isinstance(raw_types, list) or not isinstance(raw_modes, list):
            raise SchemaError(
                f"evaluator requirements audit row {position} is incomplete"
            )
        tasks.append(
            {
                "task_id": str(task["task_id"]),
                "upstream_index": int(task["upstream_index"]),
                "evaluator_types": sorted(str(item) for item in raw_types),
                "answer_matching_modes": sorted(str(item) for item in raw_modes),
            }
        )
    evaluator_types, judge_required, string_match, compatible = (
        _derived_evaluator_aggregates(tasks)
    )
    expected_status = "PASS" if compatible else "FAIL"
    if audit.get("status") != expected_status or audit.get("handoff_eligible") is not compatible:
        raise SchemaError(
            "task-interface audit status differs from derived evaluator requirements"
        )
    compile_report: dict[str, Any] | None = None
    compile_report_sha256: str | None = None
    if compatible:
        from .task_interface_audit import build_page_state_compile_authority

        compile_report = build_page_state_compile_authority(
            [dict(row) for row in task_rows]
        )
        compile_report_sha256 = sha256_json(compile_report)

    artifact = {
        "schema_version": EVALUATOR_REQUIREMENTS_SCHEMA_VERSION,
        "record_type": EVALUATOR_REQUIREMENTS_RECORD_TYPE,
        "derivation_contract": EVALUATOR_REQUIREMENTS_DERIVATION_CONTRACT,
        "registry_manifest_id": registry_id.strip(),
        "registry_manifest_sha256": registry_sha256,
        "resolved_task_set_sha256": str(task_export["resolved_task_set_sha256"]),
        "task_export_content_sha256": sha256_json(task_export),
        "task_interface_audit_content_sha256": sha256_json(audit),
        "task_count": len(tasks),
        "tasks": tasks,
        "ordered_task_ids_sha256": sha256_json(
            [row["task_id"] for row in tasks]
        ),
        "evaluator_types": evaluator_types,
        "judge_required": judge_required,
        "string_match": string_match,
        "interface_compatible": compatible,
        "interface_audit_status": expected_status,
        "page_state_compile_report": compile_report,
        "page_state_compile_report_content_sha256": compile_report_sha256,
    }
    return validate_evaluator_requirements_artifact(artifact)


def validate_evaluator_requirements_derivation(
    value: Mapping[str, Any],
    *,
    task_export: Mapping[str, Any],
    task_interface_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute an artifact from its export/audit authorities exactly."""

    validated = validate_evaluator_requirements_artifact(value)
    expected = build_evaluator_requirements_artifact(
        task_export=task_export,
        task_interface_audit=task_interface_audit,
    )
    if not _exact_json_equal(validated, expected):
        raise SchemaError(
            "evaluator requirements differ from exact task-export/interface-audit "
            "derivation"
        )
    return validated


def validate_evaluator_requirements_resolved_snapshot_binding(
    value: Mapping[str, Any],
    *,
    task_export: Mapping[str, Any],
    resolved_task_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind evaluator requirements to the exact frozen task snapshot.

    A self-consistent requirements artifact is not sufficient campaign
    authority: its claimed export hashes could describe a different 50-task
    set.  Reopen the independently authenticated export, reproduce the
    snapshot's embedded interface audit from its task rows, and require the
    evaluator-relevant task content to be byte-for-byte equivalent before
    accepting the derivation.
    """

    from .task_interface_audit import validate_resolved_task_interface_binding

    if not isinstance(task_export, Mapping):
        raise SchemaError("evaluator requirements task export must be a mapping")
    if not isinstance(resolved_task_snapshot, Mapping):
        raise SchemaError(
            "evaluator requirements resolved task snapshot must be a mapping"
        )
    if sha256_json(task_export) != resolved_task_snapshot.get(
        "upstream_export_content_sha256"
    ):
        raise SchemaError(
            "evaluator requirements task export differs from the frozen "
            "resolved-task authority"
        )
    identity_fields = (
        ("snapshot_id", "snapshot_id"),
        ("resolved_task_set_sha256", "resolved_task_set_sha256"),
        ("registry_manifest_id", "registry_manifest_id"),
        ("registry_manifest_sha256", "upstream_registry_manifest_sha256"),
    )
    for export_field, snapshot_field in identity_fields:
        if task_export.get(export_field) != resolved_task_snapshot.get(
            snapshot_field
        ):
            raise SchemaError(
                "evaluator requirements task export identity differs from "
                f"resolved snapshot: {export_field}"
            )

    export_rows = task_export.get("tasks")
    snapshot_rows = resolved_task_snapshot.get("tasks")
    if (
        not isinstance(export_rows, list)
        or not isinstance(snapshot_rows, list)
        or len(export_rows) != PUBLIC_DEVELOPMENT_TASK_COUNT
        or len(snapshot_rows) != PUBLIC_DEVELOPMENT_TASK_COUNT
    ):
        raise SchemaError(
            "evaluator requirements snapshot binding requires exactly 50 task rows"
        )
    relevant_fields = (
        "task_id",
        "upstream_index",
        "benchmark_task_id",
        "task_config",
        "evaluator",
    )
    for position, (export_row, snapshot_row) in enumerate(
        zip(export_rows, snapshot_rows, strict=True)
    ):
        if not isinstance(export_row, Mapping) or not isinstance(
            snapshot_row, Mapping
        ):
            raise SchemaError(
                f"evaluator requirements task row {position} is malformed"
            )
        for field in relevant_fields:
            if not _exact_json_equal(
                export_row.get(field), snapshot_row.get(field)
            ):
                raise SchemaError(
                    "evaluator requirements task export differs from resolved "
                    f"snapshot at row {position}: {field}"
                )

    audit = validate_resolved_task_interface_binding(resolved_task_snapshot)
    validated = validate_evaluator_requirements_derivation(
        value,
        task_export=task_export,
        task_interface_audit=audit,
    )
    if not _exact_json_equal(
        validated.get("page_state_compile_report"),
        resolved_task_snapshot.get("page_state_evaluator_compile_report"),
    ):
        raise SchemaError(
            "evaluator requirements and resolved snapshot carry different "
            "page-state compile authorities"
        )
    if validated.get(
        "page_state_compile_report_content_sha256"
    ) != resolved_task_snapshot.get(
        "page_state_evaluator_compile_report_content_sha256"
    ):
        raise SchemaError(
            "evaluator requirements and resolved snapshot compile-report "
            "hashes differ"
        )
    return validated


def _measured_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise SchemaError(f"{field} must be nonempty measured text")
    normalized = value.strip().casefold()
    if any(marker in normalized for marker in ("placeholder", "fixture", "tbd", "unknown")):
        raise SchemaError(f"{field} contains a non-production marker")
    return value.strip()


def _reject_symlink_ancestry(path: Path, *, field: str) -> None:
    """Reject a symlink in any existing component of an authority path."""

    absolute = path if path.is_absolute() else path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise SchemaError(f"{field} must not use symlink ancestry")


def _source_path(
    repository_root: Path,
    relative_value: object,
    expected_sha256: object,
    *,
    field: str,
) -> Path:
    relative = safe_relative_path(str(relative_value or ""))
    unresolved = repository_root / relative
    _reject_symlink_ancestry(unresolved, field=field)
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
    _reject_symlink_ancestry(unresolved, field=field)
    path = unresolved.resolve()
    if (
        unresolved.is_symlink()
        or evidence_root not in path.parents
        or not path.is_file()
    ):
        raise SchemaError(f"{field} is unsafe or missing: {relative}")
    if path.stat().st_nlink != 1:
        raise SchemaError(f"{field} must not be a hard-linked evidence file")
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
    _reject_symlink_ancestry(unresolved, field=path_field)
    path = unresolved.resolve()
    if (
        unresolved.is_symlink()
        or evidence_root not in path.parents
        or not path.is_file()
        or path.stat().st_size <= 0
    ):
        raise SchemaError(f"{path_field} is unsafe, missing, or empty: {relative}")
    if path.stat().st_nlink != 1:
        raise SchemaError(
            f"{path_field} must not be a hard-linked evidence artifact"
        )
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


def _validate_local_compatibility_parity(path: Path) -> dict[str, Any]:
    """Validate the pinned local parity input without treating it as review."""

    value = read_json(path)
    if not isinstance(value, Mapping):
        raise SchemaError("compatibility parity evidence must be a JSON object")
    exact = {
        "schema_version": "table2-webarena-page-state-reference-v1",
        "classification": "generated_pinned_upstream_executable_reference",
        "implementation_classification": (
            "reviewed_compatibility_port_pending_external_review"
        ),
        "reference_role": "local_compatibility_parity_only_not_external_review",
    }
    for field, expected in exact.items():
        if not _exact_json_equal(value.get(field), expected):
            raise SchemaError(f"compatibility parity evidence {field} mismatch")
    unsigned = dict(value)
    payload_sha256 = unsigned.pop("payload_sha256", None)
    if payload_sha256 != sha256_json(unsigned):
        raise SchemaError("compatibility parity evidence payload hash mismatch")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise SchemaError("compatibility parity evidence lacks provenance")
    if (
        provenance.get("wheel_sha256") != PINNED_LIBWEBARENA_WHEEL_SHA256
        or provenance.get("libwebarena_version") != PINNED_LIBWEBARENA_VERSION
        or provenance.get("installed_evaluator_source_matches_wheel") is not True
    ):
        raise SchemaError(
            "compatibility parity evidence differs from the pinned libwebarena "
            "reference"
        )
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SchemaError("compatibility parity evidence contains no cases")
    return dict(value)


def _validate_final_byte_compatibility_parity(
    path: Path,
    *,
    capability_identity: Mapping[str, Any],
    page_state_compiler_source_sha256: str | None,
    upstream_reference_sha256: str,
) -> dict[str, Any]:
    """Require measured parity evidence for the exact reviewed source bytes."""

    value = read_json(path)
    if not isinstance(value, Mapping) or set(value) != _FINAL_BYTE_PARITY_FIELDS:
        raise SchemaError(
            "final-byte compatibility parity schema has extra/missing fields"
        )
    exact = {
        "schema_version": "table2-webarena-final-byte-parity-v1",
        "record_type": "WebArenaEvaluatorFinalByteParityReceipt",
        "status": "PASS",
        "parity_scope": "FINAL_EVALUATOR_BYTES_AGAINST_PINNED_REFERENCE",
        "implementation_id": capability_identity["implementation_id"],
        "implementation_version": capability_identity[
            "implementation_version"
        ],
        "implementation_source_sha256": capability_identity["source_sha256"],
        "page_state_compiler_source_sha256": (
            page_state_compiler_source_sha256
        ),
        "upstream_reference_sha256": upstream_reference_sha256,
        "case_count": 12,
        "all_cases_matched": True,
        "state_mutation_observed": False,
        "oracle_output_returned_to_runtime": False,
        "task_outcomes_observed": False,
    }
    for field, expected in exact.items():
        if not _exact_json_equal(value.get(field), expected):
            raise SchemaError(f"final-byte compatibility parity {field} mismatch")
    _parse_measured_at(
        value.get("measured_at_utc"),
        field="final_byte_parity.measured_at_utc",
    )
    unsigned = dict(value)
    statement_sha256 = unsigned.pop("parity_statement_sha256", None)
    if statement_sha256 != sha256_json(unsigned):
        raise SchemaError("final-byte compatibility parity statement hash mismatch")
    return dict(value)


def _validate_compatibility_review(
    path: Path,
    *,
    evaluator_requirements: Mapping[str, Any],
    evaluator_requirements_sha256: str,
    capability_identity: Mapping[str, Any],
    implementation_classification: str,
    compatibility_delta_sha256: str,
    compatibility_parity_sha256: str,
) -> dict[str, Any]:
    value = read_json(path)
    if set(value) != _COMPATIBILITY_REVIEW_FIELDS:
        raise SchemaError(
            "evaluator compatibility-port review schema has extra/missing fields"
        )
    compile_report = evaluator_requirements.get("page_state_compile_report")
    compile_report_sha256 = evaluator_requirements.get(
        "page_state_compile_report_content_sha256"
    )
    compiler_source_sha256 = (
        compile_report.get("compiler_source_sha256")
        if isinstance(compile_report, Mapping)
        else None
    )
    if isinstance(compile_report, Mapping) and (
        compile_report.get("evaluator_id")
        != capability_identity.get("implementation_id")
        or compile_report.get("evaluator_version")
        != capability_identity.get("implementation_version")
    ):
        raise SchemaError(
            "compatibility review compile report and evaluator capability "
            "identity differ"
        )
    expected = {
        "schema_version": COMPATIBILITY_REVIEW_SCHEMA_VERSION,
        "record_type": "WebArenaEvaluatorCompatibilityPortReview",
        "status": "PASS",
        "review_scope": "READ_ONLY_SEALED_EVALUATOR_COMPATIBILITY_PORT",
        "evaluator_access_mode": "READ_ONLY_SEALED",
        "state_mutation_permitted": False,
        "evaluator_remapping_permitted": False,
        "task_outcomes_observed_before_freeze": False,
        "registry_manifest_sha256": evaluator_requirements[
            "registry_manifest_sha256"
        ],
        "resolved_task_set_sha256": evaluator_requirements[
            "resolved_task_set_sha256"
        ],
        "evaluator_requirements_sha256": evaluator_requirements_sha256,
        "page_state_compile_report_content_sha256": compile_report_sha256,
        "page_state_compiler_source_sha256": compiler_source_sha256,
        "implementation_id": capability_identity["implementation_id"],
        "implementation_version": capability_identity[
            "implementation_version"
        ],
        "implementation_source_sha256": capability_identity["source_sha256"],
        "implementation_classification": implementation_classification,
        "compatibility_delta_sha256": compatibility_delta_sha256,
        "compatibility_parity_sha256": compatibility_parity_sha256,
        "review_authority_type": "INDEPENDENT_EXTERNAL_TECHNICAL_REVIEW",
        "reviewer_independent_of_implementation": True,
        "reviewer_independent_of_campaign_execution": True,
    }
    for field, expected_value in expected.items():
        if not _exact_json_equal(value.get(field), expected_value):
            raise SchemaError(
                "evaluator compatibility-port review is not the registered "
                f"independent, read-only, outcome-blind review: {field}"
            )
    for field in (
        "review_authority_id",
        "reviewer_id",
        "reviewer_organization",
        "reviewer_role",
    ):
        _measured_text(value.get(field), field=f"compatibility_review.{field}")
    if value["reviewer_id"] == value["implementation_id"]:
        raise SchemaError(
            "compatibility reviewer cannot be the evaluator implementation"
        )
    if value["review_authority_id"] == value["implementation_id"]:
        raise SchemaError(
            "compatibility review authority cannot be the evaluator implementation"
        )
    _parse_measured_at(
        value.get("reviewed_at_utc"), field="compatibility_review.reviewed_at_utc"
    )
    unsigned = dict(value)
    statement_sha256 = unsigned.pop("review_statement_sha256", None)
    if statement_sha256 != sha256_json(unsigned):
        raise SchemaError("compatibility review statement hash mismatch")
    return dict(value)


def _external_review_trust_binding_sha256(
    *,
    review_receipt: Mapping[str, Any],
    review_file_sha256: str,
    parity_receipt: Mapping[str, Any],
    parity_file_sha256: str,
) -> str:
    """Commit the exact externally reviewed bytes and asserted authority.

    The two receipt-local statement hashes remain useful tamper checks, but
    they are trivially recomputable by the artifact author.  This additional
    digest has authority only when it is present in the source-tracked
    allowlist above, which must be updated in a reviewed commit after the
    independent evidence exists.
    """

    return sha256_json(
        {
            "trust_authority_version": (
                EXTERNAL_EVALUATOR_REVIEW_TRUST_AUTHORITY_VERSION
            ),
            "review_file_sha256": _sha256(
                review_file_sha256,
                field="external-review file",
            ),
            "review_statement_sha256": _sha256(
                review_receipt.get("review_statement_sha256"),
                field="external-review statement",
            ),
            "parity_file_sha256": _sha256(
                parity_file_sha256,
                field="final-byte parity file",
            ),
            "parity_statement_sha256": _sha256(
                parity_receipt.get("parity_statement_sha256"),
                field="final-byte parity statement",
            ),
            "implementation_id": review_receipt.get("implementation_id"),
            "implementation_version": review_receipt.get(
                "implementation_version"
            ),
            "implementation_source_sha256": review_receipt.get(
                "implementation_source_sha256"
            ),
            "review_authority_id": review_receipt.get("review_authority_id"),
            "reviewer_id": review_receipt.get("reviewer_id"),
            "reviewed_at_utc": review_receipt.get("reviewed_at_utc"),
            "parity_measured_at_utc": parity_receipt.get("measured_at_utc"),
        }
    )


def _require_pinned_external_review_authority(
    *,
    review_receipt: Mapping[str, Any],
    review_file_sha256: str,
    parity_receipt: Mapping[str, Any],
    parity_file_sha256: str,
) -> str:
    """Reject caller-authored review claims absent a tracked trust anchor."""

    binding_sha256 = _external_review_trust_binding_sha256(
        review_receipt=review_receipt,
        review_file_sha256=review_file_sha256,
        parity_receipt=parity_receipt,
        parity_file_sha256=parity_file_sha256,
    )
    if binding_sha256 not in PINNED_EXTERNAL_EVALUATOR_REVIEW_BINDING_SHA256S:
        raise SchemaError(
            "external evaluator review/parity evidence is not authenticated by "
            "the source-tracked trust authority; self-hashed caller JSON cannot "
            "promote a compatibility port"
        )
    return binding_sha256


def _validate_evaluator_provenance(
    value: object,
    *,
    evidence_root: Path,
    capability_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SEALED_EVALUATOR_PROVENANCE_FIELDS:
        raise SchemaError(
            "sealed WebArena evaluator provenance schema has extra/missing fields"
        )
    provenance = dict(value)
    requirements_path = _evidence_artifact(
        evidence_root,
        provenance,
        path_field="evaluator_requirements_evidence_path",
        sha256_field="evaluator_requirements_sha256",
    )
    requirements = validate_evaluator_requirements_artifact(
        read_json(requirements_path)
    )
    requirements_sha256 = sha256_file(requirements_path)
    judge_required = requirements["judge_required"]
    string_match = requirements["string_match"]
    judge_count = int(judge_required["task_count"])

    exact = {
        "upstream_repository": PINNED_UPSTREAM_WEBARENA_REPOSITORY,
        "upstream_revision": PINNED_UPSTREAM_WEBARENA_REVISION,
        "upstream_evaluator_source_sha256": PINNED_UPSTREAM_EVALUATORS_SHA256,
        "upstream_judge_source_sha256": PINNED_UPSTREAM_JUDGE_SOURCE_SHA256,
        "runtime_distribution": "libwebarena",
        "runtime_distribution_version": PINNED_LIBWEBARENA_VERSION,
        "runtime_distribution_sha256": PINNED_LIBWEBARENA_WHEEL_SHA256,
        "runtime_evaluator_module": "webarena.evaluation_harness.evaluators",
        "implementation_classification": "externally_reviewed_compatibility_port",
        "byte_identical_to_upstream": False,
        "evaluator_requirements_sha256": requirements_sha256,
        "judge_required_task_count": judge_count,
        "judge_required_task_set_sha256": judge_required[
            "task_id_set_sha256"
        ],
        "string_match_task_count": string_match["task_count"],
        "string_match_task_set_sha256": string_match[
            "task_id_set_sha256"
        ],
        "judge_model_substitution_allowed": False,
        "runtime_fallback_allowed": False,
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
    if sha256_file(runtime_module) == PINNED_UPSTREAM_EVALUATORS_SHA256:
        raise SchemaError(
            "libwebarena evaluator was labelled a compatibility port but is byte-identical"
        )
    delta_path = _evidence_artifact(
        evidence_root,
        provenance,
        path_field="compatibility_delta_evidence_path",
        sha256_field="compatibility_delta_sha256",
    )
    upstream_reference_path = _evidence_artifact(
        evidence_root,
        provenance,
        path_field="upstream_parity_reference_evidence_path",
        sha256_field="upstream_parity_reference_sha256",
    )
    _validate_local_compatibility_parity(upstream_reference_path)
    parity_path = _evidence_artifact(
        evidence_root,
        provenance,
        path_field="compatibility_parity_evidence_path",
        sha256_field="compatibility_parity_sha256",
    )
    compile_report = requirements.get("page_state_compile_report")
    compiler_source_sha256 = (
        str(compile_report.get("compiler_source_sha256"))
        if isinstance(compile_report, Mapping)
        else None
    )
    parity_receipt = _validate_final_byte_compatibility_parity(
        parity_path,
        capability_identity=capability_identity,
        page_state_compiler_source_sha256=compiler_source_sha256,
        upstream_reference_sha256=sha256_file(upstream_reference_path),
    )
    review_path = _evidence_artifact(
        evidence_root,
        provenance,
        path_field="compatibility_review_evidence_path",
        sha256_field="compatibility_review_sha256",
    )
    review_receipt = _validate_compatibility_review(
        review_path,
        evaluator_requirements=requirements,
        evaluator_requirements_sha256=requirements_sha256,
        capability_identity=capability_identity,
        implementation_classification=str(
            provenance["implementation_classification"]
        ),
        compatibility_delta_sha256=sha256_file(delta_path),
        compatibility_parity_sha256=sha256_file(parity_path),
    )
    if _parse_measured_at(
        review_receipt["reviewed_at_utc"],
        field="compatibility_review.reviewed_at_utc",
    ) < _parse_measured_at(
        parity_receipt["measured_at_utc"],
        field="final_byte_parity.measured_at_utc",
    ):
        raise SchemaError(
            "independent compatibility review predates final-byte parity evidence"
        )
    _require_pinned_external_review_authority(
        review_receipt=review_receipt,
        review_file_sha256=sha256_file(review_path),
        parity_receipt=parity_receipt,
        parity_file_sha256=sha256_file(parity_path),
    )
    _evidence_artifact(
        evidence_root,
        provenance,
        path_field="evaluator_configuration_evidence_path",
        sha256_field="evaluator_configuration_sha256",
    )

    if judge_count > 0:
        conditional = {
            "judge_model_id": PINNED_WEBARENA_JUDGE_MODEL_ID,
            "judge_prompt_source_sha256": PINNED_UPSTREAM_JUDGE_SOURCE_SHA256,
            "judge_decoding_parameters": dict(PINNED_WEBARENA_JUDGE_DECODING),
            "judge_api_preflight_status": "PASS",
            "unavailable_task_policy": (
                "preregistered_exclusion_or_disclosed_evaluator_change_before_outcomes"
            ),
        }
        for field, expected in conditional.items():
            if not _exact_json_equal(provenance.get(field), expected):
                raise SchemaError(f"sealed WebArena evaluator provenance {field} mismatch")
        _evidence_artifact(
            evidence_root,
            provenance,
            path_field="runtime_judge_module_evidence_path",
            sha256_field="runtime_judge_module_sha256",
        )
        _evidence_artifact(
            evidence_root,
            provenance,
            path_field="judge_api_response_schema_evidence_path",
            sha256_field="judge_api_response_schema_sha256",
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
    else:
        not_applicable = {
            "runtime_judge_module_evidence_path": None,
            "runtime_judge_module_sha256": None,
            "judge_model_id": None,
            "judge_prompt_source_sha256": None,
            "judge_decoding_parameters": None,
            "judge_api_response_schema_evidence_path": None,
            "judge_api_response_schema_sha256": None,
            "judge_api_availability_receipt_evidence_path": None,
            "judge_api_availability_receipt_sha256": None,
            "judge_api_preflight_status": "NOT_APPLICABLE",
            "unavailable_task_policy": JUDGE_API_NOT_APPLICABLE,
        }
        for field, expected in not_applicable.items():
            if not _exact_json_equal(provenance.get(field), expected):
                raise SchemaError(
                    f"zero-judge evaluator provenance requires {field}={expected!r}"
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


def _capability_source_identity(
    value: object,
    *,
    field: str,
) -> tuple[str, str]:
    """Return one canonical, hash-authenticated source-plane identity."""

    if not isinstance(value, Mapping):
        raise SchemaError(f"{field} source identity is malformed")
    supplied_relative = value.get("source_relative_path")
    relative = safe_relative_path(str(supplied_relative or "")).as_posix()
    if supplied_relative != relative:
        raise SchemaError(f"{field}.source_relative_path is not canonical")
    digest = _sha256(
        value.get("source_sha256"),
        field=f"{field}.source_sha256",
    )
    return relative, digest


def validate_live_capability_source_plane_disjointness(
    value: Mapping[str, Any],
) -> None:
    """Keep every runtime capability outside sealed and broker source planes.

    Source authority is represented by its canonical repository path together
    with its authenticated bytes.  Reusing either the path or the bytes would
    make the purported runtime/sealed split ambiguous, so both aliases fail
    closed.  Runtime capabilities may share a runtime-only implementation with
    one another; this check protects the two forbidden planes only.
    """

    capabilities = value.get("capabilities")
    broker = value.get("sealed_page_broker")
    if not isinstance(capabilities, Mapping) or not isinstance(broker, Mapping):
        raise SchemaError("live deployment capability source planes are malformed")
    sealed = capabilities.get("sealed_webarena_evaluator")
    sealed_identity = _capability_source_identity(
        sealed,
        field="sealed_webarena_evaluator",
    )
    broker_identity = _capability_source_identity(
        broker,
        field="sealed_page_broker",
    )
    forbidden_paths = {sealed_identity[0], broker_identity[0]}
    forbidden_hashes = {sealed_identity[1], broker_identity[1]}
    for capability_id in sorted(
        set(REQUIRED_LIVE_CAPABILITIES) - {"sealed_webarena_evaluator"}
    ):
        runtime_identity = _capability_source_identity(
            capabilities.get(capability_id),
            field=capability_id,
        )
        if (
            runtime_identity[0] in forbidden_paths
            or runtime_identity[1] in forbidden_hashes
        ):
            raise SchemaError(
                "runtime capability source plane overlaps the sealed evaluator "
                f"or broker: {capability_id}"
            )


def _validate_readiness_evidence(
    path: Path,
    *,
    capability_id: str,
    source_sha256: str,
    evidence_root: Path,
    capability_identity: Mapping[str, Any],
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
        _validate_evaluator_provenance(
            provenance,
            evidence_root=evidence_root,
            capability_identity=capability_identity,
        )
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
        if capability_id == "sealed_webarena_evaluator" and (
            "pending" in str(item.get("implementation_version")).casefold()
            or "pending" in str(item.get("implementation_id")).casefold()
        ):
            raise SchemaError(
                "sealed evaluator pending external-review identity is not "
                "campaign-ready"
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
        if (
            capability_id == "sealed_webarena_evaluator"
            and item.get("source_relative_path")
            == "src/web_agent/eval/table2/webarena_page_state_evaluator.py"
        ):
            from .webarena_page_state_evaluator import EVALUATOR_VERSION

            if "pending_external_review" in EVALUATOR_VERSION.casefold():
                raise SchemaError(
                    "the repository page-state evaluator implementation remains "
                    "pending independent external review and cannot be promoted "
                    "by a manifest PASS"
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
            capability_identity=item,
        )
        validated[capability_id] = item
    manifest["capabilities"] = validated
    validate_live_capability_source_plane_disjointness(manifest)
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
    if source.stat().st_nlink != 1:
        raise SchemaError(
            "PC-01 live deployment manifest must not be a hard-linked evidence file"
        )
    value = read_json(source)
    return validate_pc01_live_deployment_manifest(
        value,
        repository_root=repository_root,
        evidence_root=(source.parent if evidence_root is None else evidence_root),
    )


def load_pc01_live_deployment_evaluator_requirements(
    manifest: Mapping[str, Any],
    *,
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Reopen and validate the evaluator requirements bound by a manifest.

    The returned artifact is suitable for
    :func:`validate_evaluator_requirements_derivation` against the authenticated
    task export and interface audit used by a downstream handoff. Claimed
    counts and hashes are never returned without revalidating the complete
    sealed-evaluator provenance and the referenced artifact bytes.
    """

    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise SchemaError("live deployment capabilities are absent")
    evaluator = capabilities.get("sealed_webarena_evaluator")
    if not isinstance(evaluator, Mapping):
        raise SchemaError("sealed WebArena evaluator capability is absent")
    root = Path(evidence_root).resolve()
    readiness_path = _evidence_path(
        root,
        evaluator.get("readiness_evidence_path"),
        field="sealed_webarena_evaluator.readiness_evidence_path",
    )
    if sha256_file(readiness_path) != _sha256(
        evaluator.get("readiness_evidence_sha256"),
        field="sealed_webarena_evaluator.readiness_evidence_sha256",
    ):
        raise SchemaError("sealed_webarena_evaluator readiness evidence hash differs")
    readiness = read_json(readiness_path)
    provenance = readiness.get("provenance")
    _validate_evaluator_provenance(
        provenance,
        evidence_root=root,
        capability_identity=evaluator,
    )
    if not isinstance(provenance, Mapping):  # pragma: no cover - guarded above
        raise SchemaError("sealed evaluator provenance is absent")
    requirements_path = _evidence_artifact(
        root,
        provenance,
        path_field="evaluator_requirements_evidence_path",
        sha256_field="evaluator_requirements_sha256",
    )
    return validate_evaluator_requirements_artifact(read_json(requirements_path))


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
            "compatibility_delta",
            "upstream_parity_reference",
            "compatibility_parity",
            "compatibility_review",
            "evaluator_configuration",
            "evaluator_requirements",
        ):
            artifact = _evidence_artifact(
                evidence_root,
                provenance,
                path_field=f"{stem}_evidence_path",
                sha256_field=f"{stem}_sha256",
            )
            artifact_relative = artifact.relative_to(evidence_root)
            referenced[artifact_relative.as_posix()] = artifact_relative
        requirements_path = _evidence_artifact(
            evidence_root,
            provenance,
            path_field="evaluator_requirements_evidence_path",
            sha256_field="evaluator_requirements_sha256",
        )
        requirements = validate_evaluator_requirements_artifact(
            read_json(requirements_path)
        )
        if int(requirements["judge_required"]["task_count"]) > 0:
            for stem in (
                "runtime_judge_module",
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
    _reject_symlink_ancestry(
        unresolved_manifest.absolute(), field="PC-01 live deployment manifest"
    )
    _reject_symlink_ancestry(
        unresolved_evidence.absolute(), field="PC-01 live deployment evidence root"
    )
    source_manifest = unresolved_manifest.resolve()
    source_evidence = unresolved_evidence.resolve()
    destination_root = Path(destination_artifact_root).resolve()
    for source_tree, label in (
        (source_evidence, "live-deployment evidence root"),
        (source_manifest.parent, "live-deployment manifest directory"),
        (repository, "repository root"),
    ):
        if (
            destination_root == source_tree
            or destination_root in source_tree.parents
            or source_tree in destination_root.parents
        ):
            raise SchemaError(
                "live-deployment staging destination must be tree-disjoint "
                f"from the {label}"
            )
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
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for candidate in package_root.rglob("*"):
        relative = candidate.relative_to(package_root).as_posix()
        metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise SchemaError("frozen live-deployment package contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            actual_directories.add(relative)
        elif stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise SchemaError(
                    "frozen live-deployment package contains a hard-linked file"
                )
            actual_files.add(relative)
        else:
            raise SchemaError(
                "frozen live-deployment package contains a non-regular entry"
            )
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
    if actual_files != expected_relatives:
        raise SchemaError(
            "live-deployment package contains missing or unreferenced evidence files"
        )
    expected_directories: set[str] = set()
    for relative in expected_relatives:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if actual_directories != expected_directories:
        raise SchemaError(
            "live-deployment package directory closure differs from bound files"
        )
    source_files = tuple(
        (Path(repository_root).resolve() / row["relative_path"]).resolve()
        for row in expected["capability_source_files"]
    )
    evaluator_requirements = load_pc01_live_deployment_evaluator_requirements(
        manifest,
        evidence_root=package_root,
    )
    return ValidatedPC01LiveDeployment(
        binding=expected,
        manifest=manifest,
        evaluator_requirements=evaluator_requirements,
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
    validated = validate_pc01_live_deployment_binding(
        value,
        artifact_root=artifact_root,
        repository_root=repository_root,
    )
    evaluator = environment.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise SchemaError(
            "environment lacks the evaluator identity bound to live deployment"
        )
    capabilities = validated.manifest.get("capabilities")
    sealed = (
        capabilities.get("sealed_webarena_evaluator")
        if isinstance(capabilities, Mapping)
        else None
    )
    if not isinstance(sealed, Mapping):
        raise SchemaError("live deployment lacks its sealed evaluator capability")
    identity_fields = (
        ("evaluator_id", "implementation_id"),
        ("evaluator_version", "implementation_version"),
        ("source_relative_path", "source_relative_path"),
        ("source_sha256", "source_sha256"),
    )
    for environment_field, capability_field in identity_fields:
        if not _exact_json_equal(
            evaluator.get(environment_field), sealed.get(capability_field)
        ):
            raise SchemaError(
                "environment evaluator differs from live sealed-evaluator "
                f"capability: {environment_field}"
            )
    return validated
