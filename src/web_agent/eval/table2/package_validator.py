"""Campaign freezing, strict package validation, and sealed analysis loading."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from fractions import Fraction
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from web_agent.benchmarks.webarena import (
    WebArenaInfrastructureRule,
    WebArenaManualRescueCheck,
    WebArenaManualRescueReceipt,
)
from web_agent.runtime.contracts import (
    EPISODE_FOREIGN_KEY_RECEIPT_VERSION,
    ExecutionResult,
    ExecutionStatus,
    MEMORY_QUERY_VECTOR_DIMENSION,
    PreActionParseRejection,
    ParameterResolutionTrace,
    SCHEMA_VERSION as RUNTIME_SCHEMA_VERSION,
    float32_vector_sha256,
)
from web_agent.runtime.model_calls import REGISTERED_MODEL_CALL_STAGES
from web_agent.memory.joint_duplicate_audit import (
    AUDIT_TOOL_DEPENDENCY_RELATIVE_PATHS,
    AUDIT_TOOL_SOURCE_RELATIVE_PATH,
    JointDuplicateAuditError,
    validate_compact_joint_duplicate_evidence,
)
from web_agent.memory.eligibility import (
    EXCLUSION_REASONS,
    ProvenanceManifest,
    canonical_memory_id,
    is_explicitly_non_admitted,
)
from web_agent.memory.manifest import ELIGIBILITY_POLICY_VERSION, canonical_sha256
from web_agent.memory.kaggle_prepare_only import (
    EXECUTED_SOURCE_RELATIVE_PATHS,
    PREPARE_ONLY_CONFIG_RELATIVE,
    PREPARATION_EXECUTION_RECEIPT_VALIDATED,
    locate_prepare_only_execution_receipt,
)
from web_agent.memory.preparation import (
    P4_REGISTERED_SOURCE_AUTHORITY_SHA256,
    P4PreparationError,
    reconstruct_p4_selection_from_preparation,
)
from web_agent.memory.verification import (
    VerificationEvidenceError,
    validate_provenance_verification_evidence,
)

from .common import (
    CAMPAIGN_PROFILE_FINAL,
    CAMPAIGN_PROFILE_PILOT,
    FINAL_EVIDENCE_LABEL,
    PILOT_EVIDENCE_LABEL,
    SCHEMA_VERSION,
    SchemaError,
    Table2Error,
    atomic_write_json,
    canonical_json_bytes,
    classify_campaign_profile,
    read_json,
    read_jsonl,
    read_records,
    require_keys,
    sha256_file,
    sha256_json,
    strict_bool,
)
from .execution_guard import (
    ANALYSIS_REQUIRED_SOURCE_RELATIVE_PATHS,
    ENGINEERING_SMOKE_SCOPE,
    EVALUATION_RUNNER_SCOPE,
    FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH,
    PC01_PAGE_BROKER_SECURITY_FIELD,
    PC01_PROVIDER_BOOTSTRAP_BINDING_FIELD,
    PC01ProviderInstallationReceipt,
    PRODUCTION_RUNNER_ENTRYPOINT,
    validate_dependency_lock_for_environment,
    validate_frozen_dependency_lock,
    validate_infrastructure_evidence_record,
    validate_pc01_page_broker_security_binding,
    validate_analysis_source_identity,
    validate_runner_attestation_payload,
)
from .handoff_authority import validate_handoff_freeze_authority
from .evidence_validation import (
    validate_included_block_causal_trace,
    validate_ordered_verifier_receipts,
    validate_runtime_screenshot_artifacts,
)
from .locked_mount_preflight import (
    LockedMountPreflightError,
    assert_locked_mount_inaccessible,
    validate_locked_mount_attestation,
)
from .metrics import compute_table2_metrics
from .live_deployment import (
    LIVE_DEPLOYMENT_BINDING_FIELD,
    ValidatedPC01LiveDeployment,
    validate_bound_pc01_live_deployment,
    validate_evaluator_requirements_resolved_snapshot_binding,
)
from .live_compatibility import (
    LIVE_COMPATIBILITY_RELATIVE_PATH,
    LIVE_COMPATIBILITY_REQUIRED_SOURCE_RELATIVE_PATHS,
    LIVE_COMPATIBILITY_SIDECAR_RELATIVE_PATH,
    validate_live_compatibility_receipt,
)
from .outcome_semantics import (
    normalize_runtime_terminal_reason,
    validate_primary_runtime_outcome,
)
from .paper_claims import (
    FROZEN_REGISTRY_RELATIVE_PATH as FROZEN_PAPER_CLAIM_REGISTRY_RELATIVE_PATH,
    REGISTRY_ID as PAPER_CLAIM_REGISTRY_ID,
    SOURCE_REGISTRY_RELATIVE_PATH as PAPER_CLAIM_REGISTRY_SOURCE_RELATIVE_PATH,
    validate_claim_registry,
    validate_claim_registry_binding,
)
from .retrieval_metrics import compute_retrieval_diagnostics
from .pilot_task_exclusion import (
    FROZEN_PILOT_TASK_EXCLUSION_RELATIVE_PATH,
    PILOT_TASK_EXCLUSION_IDENTITY_VERSION,
    PILOT_TASK_EXCLUSION_REGISTRY_RELATIVE_PATH,
    load_pilot_task_exclusion_authority,
    pilot_task_exclusion_provenance,
    validate_locked_final_pilot_exclusion,
)
from .resolved_config import (
    ResolvedConfigIdentityError,
    load_resolved_config_identity,
)
from .pc01_artifacts import (
    PC01_EXPECTED_BASE_SNAPSHOT_SHA256,
    PC01_EXPECTED_EXPORT_MANIFEST_SHA256,
    PC01_MODEL_ID,
    PC01_MODEL_REVISION,
    PC01_MODEL_SEED,
    PC01_TRAINING_ENVIRONMENT_SHA256,
    build_pc01_base_snapshot_manifest,
    load_pc01_base_snapshot_manifest,
    validate_pc01_training_action_value_evidence,
)
from .pc01_processor_parity import validate_pc01_processor_parity_receipt
from .pc01_checkpoint_compatibility import (
    PC01CheckpointCompatibilityError,
    validate_pc01_checkpoint_compatibility_receipt,
)
from .schedule import (
    DEFAULT_STAGE_KEYS,
    SYSTEM_IDS,
    _runtime_stage_seed,
    build_paired_schedule,
    discover_block_attempt_directories,
    resolve_block_attempts,
    validate_schedule,
)
from .sealed_verifier import assert_no_verifier_evidence, verify_sealed_stream
from .state_isolation_validation import (
    frozen_runner_source_hashes,
    validate_episode_state_reset,
    validate_included_block_state_isolation,
    validate_webarena_reset_state_receipt,
)
from .statistics import compute_clustered_ratio_contrasts, compute_paired_contrasts
from .task_interface_audit import (
    require_webarena_task_interface_compatible,
    validate_resolved_task_interface_binding,
)
from .webarena_preflight_binding import (
    PREFLIGHT_ARTIFACT_RELATIVE_PATH,
    PREFLIGHT_BINDING_FIELD,
    PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH,
    ValidatedDeploymentPreflight,
    validate_bound_deployment_preflight,
)


RUNTIME_FILES = (
    "episode_manifest.json",
    "rng_provenance.json",
    "episode_summary.json",
    "actions.jsonl",
    "transitions.jsonl",
    "recoveries.jsonl",
    "environment_events.jsonl",
    "terminal_signals.jsonl",
    "artifact_hashes.json",
)
FINAL_READY_STATUS = "READY_FOR_TABLE2"
DRAFT_PILOT_STATUS = "DRAFT_PILOT_ONLY"
CAMPAIGN_COMPLETION_KEYS = (
    "schema_version",
    "campaign_id",
    "status",
    "scheduled_block_count",
    "processed_block_count",
    "included_block_count",
    "infrastructure_excluded_block_count",
    "publication_status",
)
MANUAL_AUDIT_MANIFEST_ID = "table2-outcome-label-hidden-audit-20-v2"
MANUAL_AUDIT_BLINDING_MODE = "OUTCOME_LABELS_HIDDEN_SYSTEM_CONDITION_VISIBLE"
MANUAL_AUDIT_CODEBOOK_SCHEMA_VERSION = "table2-manual-audit-reviewer-codebook-v2"
MANUAL_AUDIT_CODEBOOK_ID = "table2-outcome-label-hidden-reviewer-codebook-v2"
MANUAL_AUDIT_AGREEMENT_METHOD = (
    "unweighted_cohen_kappa_over_complete_ordered_label_vectors_v1"
)
MANUAL_AUDIT_COMPOSITE_AGREEMENT_METHOD = (
    "complete_ordered_label_vector_exact_agreement_v1"
)
MANUAL_AUDIT_PER_FIELD_AGREEMENT_METHOD = (
    "unweighted_cohen_kappa_per_registered_field_v1"
)
MANUAL_AUDIT_FINAL_VS_SEALED_METHOD = (
    "adjudicated_labels_vs_selected_sealed_evidence_counts_v1"
)
MANUAL_AUDIT_KAPPA_DEFINED_STATUS = "DEFINED"
MANUAL_AUDIT_KAPPA_UNDEFINED_STATUS = "UNDEFINED"
MANUAL_AUDIT_KAPPA_UNDEFINED_REASON = "EXPECTED_AGREEMENT_EQUALS_ONE"
MANUAL_AUDIT_CASE_CATEGORIES = (
    "successful_recovery",
    "failed_recovery",
    "memory_help",
    "memory_harm",
    "environment_failure",
    "bbox_or_parameter_failure",
    "loop",
    "unnecessary_intervention",
)
MANUAL_AUDIT_LABEL_FIELDS = (
    "task_outcome",
    "recovery_outcome",
    "memory_effect",
    "failure_attribution",
    "intervention_assessment",
)
MODEL_PAYLOAD_ROLES = (
    "selected_checkpoint",
    "resolved_config",
    "processor_contract",
    "e0_backbone",
    "e0_resolved_config",
    "e0_processor_contract",
    "e0_parser",
)
MODEL_PAYLOAD_HASH_FIELDS = {
    "selected_checkpoint": "selected_checkpoint_sha256",
    "resolved_config": "resolved_config_sha256",
    "processor_contract": "processor_contract_sha256",
    "e0_backbone": "e0_backbone_sha256",
    "e0_resolved_config": "e0_resolved_config_sha256",
    "e0_processor_contract": "e0_processor_contract_sha256",
    "e0_parser": "e0_parser_sha256",
}
MODEL_EVIDENCE_BUNDLE_SCHEMA_VERSION = "table2-model-evidence-bundle-v1"
MODEL_EVIDENCE_PRODUCER_SCHEMA_VERSION = "table2.pc01-export.v3"
MODEL_EVIDENCE_STATUS = "SUPPORTING_EVIDENCE_ONLY_NOT_RUNTIME_READINESS"
MODEL_EVIDENCE_ROLES = (
    "export_manifest",
    "base_snapshot_manifest",
    "processor_artifact_manifest",
    "processor_parity_receipt",
    "training_environment",
    "training_source_manifest",
    "training_action_value_evidence",
)
MODEL_EVIDENCE_FILENAMES = {
    "export_manifest": "pc01_export_manifest.json",
    "base_snapshot_manifest": "base_snapshot_manifest.json",
    "processor_artifact_manifest": "processor_artifact_manifest.json",
    "processor_parity_receipt": "processor_parity_receipt.json",
    "training_environment": "training_environment.json",
    "training_source_manifest": "training_source_manifest.json",
    "training_action_value_evidence": "training_action_value_evidence.json",
}
PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD = (
    "pc01_checkpoint_compatibility"
)
PC01_CHECKPOINT_COMPATIBILITY_RECEIPT_RELATIVE_PATH = Path(
    "runtime_readiness/pc01_checkpoint_compatibility_receipt.json"
)
HANDOFF_CONSUMPTION_SCHEMA_VERSION = "table2-handoff-consumption-v1"
FROZEN_HANDOFF_CONSUMPTION_RELATIVE_PATH = Path(
    "frozen/handoff_consumption.json"
)
FROZEN_MEMORY_STORE_FILES = (
    "calibration_evidence.json",
    "verification_evidence.json",
    "manifest.json",
    "manifest.sha256",
    "embeddings.npy",
    "items.jsonl",
    "threshold_calibration.json",
)
JOINT_DUPLICATE_EVIDENCE_BINDING_SCHEMA_VERSION = (
    "table2-memory-joint-duplicate-evidence-binding-v3"
)
P4_SOURCE_AUTHORITY_RELATIVE_PATH = Path(
    "configs/eval/table2/p4_source_authority_v1.json"
)
_CANONICAL_P4_REGISTERED_SOURCE_AUTHORITY_SHA256 = (
    P4_REGISTERED_SOURCE_AUTHORITY_SHA256
)
JOINT_DUPLICATE_AUDIT_CONFIG_RELATIVE_PATH = Path(
    "configs/eval/table2/joint_duplicate_audit_v1.json"
)
JOINT_DUPLICATE_AUDIT_REGISTRATION_RELATIVE_PATH = Path(
    "benchmarks/table2/pilot/duplicate_audit_manifest.json"
)
JOINT_DUPLICATE_EVIDENCE_RELATIVE_PATH = Path(
    "frozen/joint_duplicate_evidence"
)
JOINT_DUPLICATE_ASSIGNMENT_FILES = (
    "audit_config.json",
    "entities.jsonl",
    "clusters.json",
    "read_ledger.json",
    "source_manifest.json",
    "assignment_manifest.json",
    "assignment_manifest.sha256",
)
P4_PREPARATION_EVIDENCE_FILES = (
    "candidate_audit.json",
    "read_ledger.json",
    "review_queue.jsonl",
    "source_authority.json",
    "preparation_manifest.json",
    "preparation_manifest.sha256",
)
P4_PREPARATION_EXECUTION_EVIDENCE_FILES = (
    "execution_receipt.json",
    "execution_receipt.sha256",
)
REGISTERED_RECOVERY_LOW_LEVEL_ACTION_LIMITS = {
    "RETRY": 1,
    "REPLAN": 1,
    "BACKTRACK": 1,
    "ALTERNATIVE_TARGET": 1,
    "ABORT": 0,
}
EVALUATION_CONTROL_SOURCE_RELATIVE_PATHS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *LIVE_COMPATIBILITY_REQUIRED_SOURCE_RELATIVE_PATHS,
            *ANALYSIS_REQUIRED_SOURCE_RELATIVE_PATHS,
            "scripts/build_table2_dependency_lock.py",
            "scripts/freeze_table2_campaign.py",
            "scripts/validate_table2_paper_claims.py",
            str(PAPER_CLAIM_REGISTRY_SOURCE_RELATIVE_PATH),
            "src/web_agent/__init__.py",
            "src/web_agent/eval/__init__.py",
            "src/web_agent/eval/table2/__init__.py",
            "src/web_agent/eval/table2/handoff.py",
            "src/web_agent/eval/table2/handoff_authority.py",
            "src/web_agent/eval/table2/dependency_lock.py",
            "src/web_agent/eval/table2/paper_claims.py",
            "src/web_agent/eval/table2/process_broker.py",
            "src/web_agent/eval/table2/process_broker_protocol.py",
            "src/web_agent/eval/table2/process_broker_runtime.py",
            "src/web_agent/eval/table2/process_broker_timeout.py",
            "src/web_agent/eval/table2/process_broker_worker.py",
            "src/web_agent/eval/table2/task_interface_audit.py",
            "src/web_agent/eval/table2/webarena_page_state_evaluator.py",
        )
    )
)


def _expected_manual_audit_reviewer_codebook() -> dict[str, Any]:
    """Return the immutable categorical codebook registered for Table 2 review."""

    return {
        "schema_version": MANUAL_AUDIT_CODEBOOK_SCHEMA_VERSION,
        "codebook_id": MANUAL_AUDIT_CODEBOOK_ID,
        "label_vector_order": [
            "task_outcome",
            "recovery_outcome",
            "memory_effect",
            "failure_attribution",
            "intervention_assessment",
        ],
        "fields": [
            {
                "name": "task_outcome",
                "allowed_labels": ["SUCCESS", "FAILURE", "UNDETERMINABLE"],
                "semantics": {
                    "SUCCESS": (
                        "Observable episode evidence establishes completion of the "
                        "assigned browser task."
                    ),
                    "FAILURE": (
                        "Observable episode evidence establishes that the assigned "
                        "browser task was not completed."
                    ),
                    "UNDETERMINABLE": (
                        "Available artifacts are insufficient to establish either "
                        "success or failure; success must not be inferred."
                    ),
                },
            },
            {
                "name": "recovery_outcome",
                "allowed_labels": [
                    "SUCCESSFUL_RECOVERY",
                    "FAILED_RECOVERY",
                    "NO_RECOVERY_ATTEMPT",
                    "UNDETERMINABLE",
                ],
                "semantics": {
                    "SUCCESSFUL_RECOVERY": (
                        "An executed recovery resolved the observed failure and the "
                        "post-recovery evidence shows restored progress."
                    ),
                    "FAILED_RECOVERY": (
                        "A recovery was executed but did not resolve the observed "
                        "failure within the registered recovery window."
                    ),
                    "NO_RECOVERY_ATTEMPT": (
                        "The reviewed episode contains no executed recovery attempt."
                    ),
                    "UNDETERMINABLE": (
                        "The artifacts do not support a reliable recovery-outcome "
                        "decision."
                    ),
                },
            },
            {
                "name": "memory_effect",
                "allowed_labels": [
                    "MEMORY_HELP",
                    "MEMORY_HARM",
                    "NO_DEMONSTRABLE_MEMORY_EFFECT",
                    "NOT_APPLICABLE",
                    "UNDETERMINABLE",
                ],
                "semantics": {
                    "MEMORY_HELP": (
                        "Paired causal evidence attributes a better E3 outcome to an "
                        "admitted frozen-memory intervention."
                    ),
                    "MEMORY_HARM": (
                        "Paired causal evidence attributes a worse E3 outcome to an "
                        "admitted frozen-memory intervention."
                    ),
                    "NO_DEMONSTRABLE_MEMORY_EFFECT": (
                        "The paired evidence does not demonstrate either help or harm "
                        "from frozen memory."
                    ),
                    "NOT_APPLICABLE": (
                        "The reviewed evidence has no admitted frozen-memory "
                        "intervention suitable for an E2/E3 effect judgment."
                    ),
                    "UNDETERMINABLE": (
                        "A memory-effect judgment is applicable but the available "
                        "paired artifacts are insufficient."
                    ),
                },
            },
            {
                "name": "failure_attribution",
                "allowed_labels": [
                    "NO_FAILURE_OBSERVED",
                    "BBOX_OR_PARAMETER_FAILURE",
                    "LOOP_FAILURE",
                    "OTHER_AGENT_FAILURE",
                    "ENVIRONMENT_OR_INFRASTRUCTURE_FAILURE",
                    "MIXED_FAILURE",
                    "UNDETERMINABLE",
                ],
                "semantics": {
                    "NO_FAILURE_OBSERVED": (
                        "No failure is supported by the reviewed observable evidence."
                    ),
                    "BBOX_OR_PARAMETER_FAILURE": (
                        "The primary supported failure is target grounding or concrete "
                        "action-parameter resolution."
                    ),
                    "LOOP_FAILURE": (
                        "The primary supported failure is repeated non-progressing "
                        "agent behavior."
                    ),
                    "OTHER_AGENT_FAILURE": (
                        "The primary supported failure is agent-caused but is neither "
                        "bbox/parameter failure nor a loop."
                    ),
                    "ENVIRONMENT_OR_INFRASTRUCTURE_FAILURE": (
                        "The primary supported failure is external to the agent policy "
                        "or recovery mechanism."
                    ),
                    "MIXED_FAILURE": (
                        "Agent and environment causes are both materially supported and "
                        "cannot be separated as the primary cause."
                    ),
                    "UNDETERMINABLE": (
                        "The available artifacts do not support a reliable primary "
                        "failure attribution."
                    ),
                },
            },
            {
                "name": "intervention_assessment",
                "allowed_labels": [
                    "JUSTIFIED_INTERVENTION",
                    "UNNECESSARY_INTERVENTION",
                    "NO_INTERVENTION",
                    "UNDETERMINABLE",
                ],
                "semantics": {
                    "JUSTIFIED_INTERVENTION": (
                        "An intervention was executed in response to observable failure "
                        "or non-progress evidence."
                    ),
                    "UNNECESSARY_INTERVENTION": (
                        "An intervention was executed without observable evidence that "
                        "it was needed."
                    ),
                    "NO_INTERVENTION": (
                        "The reviewed episode contains no recovery or memory "
                        "intervention."
                    ),
                    "UNDETERMINABLE": (
                        "The artifacts are insufficient to judge whether an executed "
                        "intervention was necessary."
                    ),
                },
            },
        ],
        "label_object_rule": (
            "exactly_one_registered_string_label_for_every_ordered_field_no_extra_fields"
        ),
        "disagreement_rule": (
            "true_if_any_ordered_label_value_differs_between_exactly_two_reviewers"
        ),
        "agreement_unit": "complete_ordered_label_vector_per_selected_episode",
        "agreement_method": MANUAL_AUDIT_AGREEMENT_METHOD,
        "composite_exact_agreement_method": (
            MANUAL_AUDIT_COMPOSITE_AGREEMENT_METHOD
        ),
        "per_field_agreement_method": MANUAL_AUDIT_PER_FIELD_AGREEMENT_METHOD,
        "final_vs_sealed_comparison_method": (
            MANUAL_AUDIT_FINAL_VS_SEALED_METHOD
        ),
        "kappa_weighting": "unweighted",
        "undefined_kappa": {
            "status": MANUAL_AUDIT_KAPPA_UNDEFINED_STATUS,
            "reason": MANUAL_AUDIT_KAPPA_UNDEFINED_REASON,
            "condition": "expected_chance_agreement_equals_one",
        },
        "applicability_rules": [
            (
                "NO_RECOVERY_ATTEMPT_if_and_only_if_selected_evidence_has_no_"
                "executed_recovery"
            ),
            (
                "NOT_APPLICABLE_memory_effect_if_and_only_if_no_E3_admitted_memory_"
                "intervention_with_paired_E2_evidence"
            ),
            (
                "NO_INTERVENTION_if_and_only_if_selected_evidence_has_neither_"
                "executed_recovery_nor_admitted_memory_intervention"
            ),
        ],
    }


def validate_manual_audit_reviewer_codebook(
    audit_definition: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact registered codebook and return its immutable binding."""

    codebook = audit_definition.get("reviewer_codebook")
    expected = _expected_manual_audit_reviewer_codebook()
    if not isinstance(codebook, Mapping) or dict(codebook) != expected:
        raise SchemaError("manual-audit reviewer codebook differs from registration")
    return {
        "reviewer_codebook_id": MANUAL_AUDIT_CODEBOOK_ID,
        "reviewer_codebook_sha256": sha256_json(codebook),
    }


def _require_exact_json_object(
    value: Any,
    expected_keys: Sequence[str],
    *,
    context: str,
) -> Mapping[str, Any]:
    """Require an exact JSON-object key closure for audit evidence."""

    if not isinstance(value, Mapping):
        raise SchemaError(f"{context} must be a JSON object")
    expected = set(expected_keys)
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise SchemaError(
            f"{context} has the wrong exact key schema; "
            f"missing={missing!r}, extra={extra!r}"
        )
    return value


def _load_audit_json_without_duplicate_keys(
    text: str,
    *,
    context: str,
) -> Any:
    """Parse reviewer evidence while rejecting duplicate JSON object keys."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise SchemaError(f"{context} contains duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        return json.loads(text, object_pairs_hook=object_pairs)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{context} is invalid JSON: {exc}") from exc


def _read_manual_audit_json(path: Path, *, context: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise SchemaError(f"{context} is absent or symlinked")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaError(f"{context} is not UTF-8 JSON") from exc
    return _load_audit_json_without_duplicate_keys(text, context=context)


def _audit_exact_bool(value: Any, *, context: str) -> bool:
    """Reject integer/string lookalikes in human-audit JSON artifacts."""

    if type(value) is not bool:
        raise SchemaError(f"{context} must be an exact JSON boolean")
    return value


def _audit_exact_int(
    value: Any,
    *,
    context: str,
    minimum: int | None = None,
) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise SchemaError(f"{context} must be an exact JSON integer{suffix}")
    return value


def _audit_nonempty_string(value: Any, *, context: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise SchemaError(f"{context} must be a nonempty canonical JSON string")
    return value


def _audit_sha256(value: Any, *, context: str) -> str:
    text = _audit_nonempty_string(value, context=context)
    if not _is_sha256(text):
        raise SchemaError(f"{context} must be a lowercase SHA-256 digest")
    return text


def _manual_audit_label_vector(
    value: Any,
    *,
    audit_definition: Mapping[str, Any],
    context: str,
) -> tuple[str, ...]:
    """Validate one reviewer/final label and return its registered ordered vector."""

    validate_manual_audit_reviewer_codebook(audit_definition)
    codebook = audit_definition["reviewer_codebook"]
    field_order = tuple(codebook["label_vector_order"])
    fields = codebook["fields"]
    allowed_by_field = {
        str(field["name"]): tuple(field["allowed_labels"]) for field in fields
    }
    if not isinstance(value, Mapping) or set(map(str, value)) != set(field_order):
        raise SchemaError(
            f"{context} must contain exactly the registered reviewer-codebook fields"
        )
    vector: list[str] = []
    for field in field_order:
        label = value.get(field)
        if type(label) is not str or label not in allowed_by_field[field]:
            raise SchemaError(
                f"{context}.{field} is not a registered categorical label"
            )
        vector.append(label)
    return tuple(vector)


def _validate_manual_audit_label_applicability(
    label: Mapping[str, Any],
    *,
    selected_evidence: Mapping[str, Any],
    context: str,
) -> None:
    """Enforce registered applicability without replacing human judgments."""

    has_recovery = _audit_exact_bool(
        selected_evidence.get("has_recovery_attempt"),
        context=f"{context}.selected_evidence.has_recovery_attempt",
    )
    has_memory = _audit_exact_bool(
        selected_evidence.get("has_admitted_memory_intervention"),
        context=f"{context}.selected_evidence.has_admitted_memory_intervention",
    )
    memory_applicable = _audit_exact_bool(
        selected_evidence.get("memory_effect_applicable"),
        context=f"{context}.selected_evidence.memory_effect_applicable",
    )
    recovery_outcome = label["recovery_outcome"]
    if (recovery_outcome == "NO_RECOVERY_ATTEMPT") != (not has_recovery):
        raise SchemaError(f"{context}.recovery_outcome violates evidence applicability")

    memory_effect = label["memory_effect"]
    if (memory_effect == "NOT_APPLICABLE") != (not memory_applicable):
        raise SchemaError(f"{context}.memory_effect violates evidence applicability")

    intervention = label["intervention_assessment"]
    has_intervention = has_recovery or has_memory
    if (intervention == "NO_INTERVENTION") != (not has_intervention):
        raise SchemaError(
            f"{context}.intervention_assessment violates evidence applicability"
        )


def _cohen_kappa_for_categories(
    reviewer_a: Sequence[Any],
    reviewer_b: Sequence[Any],
) -> dict[str, Any]:
    if not reviewer_a or len(reviewer_a) != len(reviewer_b):
        raise SchemaError("manual-audit reviewer categories are empty or unequal")
    sample_size = len(reviewer_a)
    agreement_count = sum(
        left == right for left, right in zip(reviewer_a, reviewer_b, strict=True)
    )
    observed = Fraction(agreement_count, sample_size)
    left_counts = Counter(reviewer_a)
    right_counts = Counter(reviewer_b)
    expected = sum(
        (
            Fraction(left_counts.get(category, 0), sample_size)
            * Fraction(right_counts.get(category, 0), sample_size)
        )
        for category in set(left_counts) | set(right_counts)
    )
    if expected == 1:
        kappa: float | None = None
        kappa_status = MANUAL_AUDIT_KAPPA_UNDEFINED_STATUS
        undefined_reason: str | None = MANUAL_AUDIT_KAPPA_UNDEFINED_REASON
    else:
        kappa = float((observed - expected) / (1 - expected))
        kappa_status = MANUAL_AUDIT_KAPPA_DEFINED_STATUS
        undefined_reason = None
    return {
        "sample_size": sample_size,
        "agreement_count": agreement_count,
        "raw_agreement": float(observed),
        "cohen_kappa": kappa,
        "cohen_kappa_status": kappa_status,
        "cohen_kappa_undefined_reason": undefined_reason,
    }


def _recompute_manual_audit_agreement(
    reviewer_a: Sequence[tuple[str, ...]],
    reviewer_b: Sequence[tuple[str, ...]],
) -> dict[str, Any]:
    """Recompute raw agreement and exact unweighted Cohen's kappa."""

    if not reviewer_a or len(reviewer_a) != len(reviewer_b):
        raise SchemaError("manual-audit reviewer vectors are empty or unequal")
    composite = _cohen_kappa_for_categories(reviewer_a, reviewer_b)
    per_field: list[dict[str, Any]] = []
    for index, field_name in enumerate(MANUAL_AUDIT_LABEL_FIELDS):
        field_result = _cohen_kappa_for_categories(
            [vector[index] for vector in reviewer_a],
            [vector[index] for vector in reviewer_b],
        )
        per_field.append(
            {
                "field": field_name,
                **field_result,
                "calculation_method": MANUAL_AUDIT_PER_FIELD_AGREEMENT_METHOD,
            }
        )
    return {
        "raw_agreement": composite["raw_agreement"],
        "cohen_kappa": composite["cohen_kappa"],
        "cohen_kappa_status": composite["cohen_kappa_status"],
        "cohen_kappa_undefined_reason": composite[
            "cohen_kappa_undefined_reason"
        ],
        "calculation_method": MANUAL_AUDIT_AGREEMENT_METHOD,
        "composite_exact_agreement_count": composite["agreement_count"],
        "composite_exact_agreement": composite["raw_agreement"],
        "composite_exact_agreement_method": (
            MANUAL_AUDIT_COMPOSITE_AGREEMENT_METHOD
        ),
        "per_field_agreement_method": MANUAL_AUDIT_PER_FIELD_AGREEMENT_METHOD,
        "per_field_agreement": per_field,
    }


def _recompute_manual_audit_final_vs_sealed(
    final_labels_by_id: Mapping[str, Mapping[str, Any]],
    sealed_by_audit_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Disclose, without suppressing, adjudicated-vs-evaluator discrepancies.

    The human labels remain independent judgments.  This comparison is a
    post-adjudication diagnostic, not an applicability rule and not a
    requirement that either source agree with the other.
    """

    if not final_labels_by_id or set(final_labels_by_id) != set(sealed_by_audit_id):
        raise SchemaError(
            "manual-audit final/sealed comparison has a missing or extra audit ID"
        )

    comparisons: dict[str, list[tuple[str | None, str]]] = {
        "task_outcome": [],
        "recovery_outcome": [],
        "memory_effect": [],
    }
    for audit_id in sorted(final_labels_by_id):
        final_label = final_labels_by_id[audit_id]
        sealed = sealed_by_audit_id[audit_id]
        categories_value = sealed.get("case_categories")
        if (
            not isinstance(categories_value, list)
            or not all(type(value) is str for value in categories_value)
        ):
            raise SchemaError(
                "manual-audit selected sealed case categories are malformed"
            )
        categories = set(categories_value)

        task_success = _audit_exact_bool(
            sealed.get("task_success"),
            context=f"manual-audit sealed[{audit_id}].task_success",
        )
        comparisons["task_outcome"].append(
            ("SUCCESS" if task_success else "FAILURE", final_label["task_outcome"])
        )

        has_recovery = _audit_exact_bool(
            sealed.get("has_recovery_attempt"),
            context=f"manual-audit sealed[{audit_id}].has_recovery_attempt",
        )
        if not has_recovery:
            sealed_recovery: str | None = "NO_RECOVERY_ATTEMPT"
        else:
            recovery_categories = categories & {
                "successful_recovery",
                "failed_recovery",
            }
            sealed_recovery = (
                "SUCCESSFUL_RECOVERY"
                if recovery_categories == {"successful_recovery"}
                else "FAILED_RECOVERY"
                if recovery_categories == {"failed_recovery"}
                else None
            )
        comparisons["recovery_outcome"].append(
            (sealed_recovery, final_label["recovery_outcome"])
        )

        memory_applicable = _audit_exact_bool(
            sealed.get("memory_effect_applicable"),
            context=f"manual-audit sealed[{audit_id}].memory_effect_applicable",
        )
        if not memory_applicable:
            sealed_memory: str | None = "NOT_APPLICABLE"
        else:
            memory_categories = categories & {"memory_help", "memory_harm"}
            sealed_memory = (
                "MEMORY_HELP"
                if memory_categories == {"memory_help"}
                else "MEMORY_HARM"
                if memory_categories == {"memory_harm"}
                else "NO_DEMONSTRABLE_MEMORY_EFFECT"
                if not memory_categories
                else None
            )
        comparisons["memory_effect"].append(
            (sealed_memory, final_label["memory_effect"])
        )

    rows: list[dict[str, Any]] = []
    for field_name in ("task_outcome", "recovery_outcome", "memory_effect"):
        pairs = comparisons[field_name]
        uniquely_mappable = [pair for pair in pairs if pair[0] is not None]
        comparable = [pair for pair in uniquely_mappable if pair[1] != "UNDETERMINABLE"]
        agreement_count = sum(sealed == human for sealed, human in comparable)
        disagreement_count = len(comparable) - agreement_count
        rows.append(
            {
                "field": field_name,
                "eligible_count": len(pairs),
                "uniquely_mappable_count": len(uniquely_mappable),
                "comparable_count": len(comparable),
                "agreement_count": agreement_count,
                "disagreement_count": disagreement_count,
                "human_undeterminable_count": sum(
                    human == "UNDETERMINABLE" for _, human in uniquely_mappable
                ),
                "not_uniquely_mappable_count": len(pairs) - len(uniquely_mappable),
            }
        )
    return {
        "final_vs_sealed_comparison_method": MANUAL_AUDIT_FINAL_VS_SEALED_METHOD,
        "final_vs_sealed_comparison": rows,
    }


@dataclass
class ValidationReport:
    campaign_dir: str
    status: str = "PASS"
    publication_status: str = "N/R"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "PASS" and not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)
        self.status = "FAIL"

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_canonical_pc01_checkpoint_compatibility_receipt(
    path: str | Path,
) -> dict[str, Any]:
    """Reopen one immutable, duplicate-key-free canonical DGX receipt."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise SchemaError(
            "PC-01 checkpoint compatibility receipt is missing or unsafe"
        )
    if source.stat().st_mode & 0o222:
        raise SchemaError(
            "PC-01 checkpoint compatibility receipt must be read-only"
        )
    payload = source.read_bytes()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise SchemaError(
                    "PC-01 checkpoint compatibility receipt repeats a JSON key"
                )
            value[key] = item
        return value

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError(
            "PC-01 checkpoint compatibility receipt is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise SchemaError(
            "PC-01 checkpoint compatibility receipt must be a JSON object"
        )
    if payload != canonical_json_bytes(value):
        raise SchemaError(
            "PC-01 checkpoint compatibility receipt is not exact canonical JSON"
        )
    return dict(value)


def _pc01_checkpoint_compatibility_source_files(
    receipt: Mapping[str, Any],
    *,
    repository_root: Path,
) -> tuple[Path, ...]:
    """Recompute every receipt source row from the supplied source closure."""

    root = repository_root.resolve()
    attestation = receipt.get("source_attestation")
    rows = attestation.get("source_files") if isinstance(attestation, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise SchemaError(
            "PC-01 checkpoint compatibility receipt has no source attestation"
        )
    resolved: list[Path] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise SchemaError(
                "PC-01 checkpoint compatibility source row is malformed"
            )
        relative = str(row.get("path") or "")
        relative_path = Path(relative)
        if (
            not relative
            or relative in seen
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise SchemaError(
                "PC-01 checkpoint compatibility source path is unsafe or duplicated"
            )
        seen.add(relative)
        candidate = root / relative_path
        if candidate.is_symlink() or not candidate.is_file():
            raise SchemaError(
                "PC-01 checkpoint compatibility source is absent or unsafe: "
                + relative
            )
        source = candidate.resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise SchemaError(
                "PC-01 checkpoint compatibility source escaped the source root"
            ) from exc
        if source != candidate.absolute():
            raise SchemaError(
                "PC-01 checkpoint compatibility source traverses a symlink: "
                + relative
            )
        if (
            row.get("sha256") != sha256_file(source)
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] != source.stat().st_size
        ):
            raise SchemaError(
                "PC-01 checkpoint compatibility source bytes differ: " + relative
            )
        resolved.append(source)
    return tuple(resolved)


def _selection_candidate_artifact_hashes(
    selection_manifest_path: Path,
    *,
    model_id: str,
) -> tuple[str, str]:
    """Return actual full-report/run-contract hashes for the selected model."""

    if selection_manifest_path.is_symlink() or not selection_manifest_path.is_file():
        raise SchemaError(
            "validation-selection manifest is absent or unsafe"
        )
    selection = read_json(selection_manifest_path)
    if selection.get("selected_model_id") != model_id:
        raise SchemaError(
            "checkpoint compatibility receipt and validation-selected model differ"
        )
    candidates = selection.get("candidates")
    matches = (
        [row for row in candidates if isinstance(row, Mapping) and row.get("model_id") == model_id]
        if isinstance(candidates, list)
        else []
    )
    if len(matches) != 1:
        raise SchemaError(
            "validation-selection evidence lacks one selected PC-01 candidate"
        )
    row = matches[0]
    root = selection_manifest_path.parent.resolve()
    hashes: list[str] = []
    for path_field, hash_field in (
        ("full_report_path", "full_report_sha256"),
        ("run_contract_path", "run_contract_sha256"),
    ):
        relative = Path(str(row.get(path_field) or ""))
        if not str(relative) or relative.is_absolute() or ".." in relative.parts:
            raise SchemaError(
                "validation-selection checkpoint evidence path is unsafe"
            )
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise SchemaError(
                "validation-selection checkpoint evidence is absent"
            )
        source = candidate.resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise SchemaError(
                "validation-selection checkpoint evidence escaped its package"
            ) from exc
        if source != candidate.absolute():
            raise SchemaError(
                "validation-selection checkpoint evidence traverses a symlink"
            )
        digest = sha256_file(source)
        if row.get(hash_field) != digest:
            raise SchemaError(
                "validation-selection checkpoint evidence hash differs"
            )
        hashes.append(digest)
    return hashes[0], hashes[1]


def _validate_pc01_checkpoint_compatibility_readiness(
    receipt_path: str | Path,
    *,
    repository_root: Path,
    expected_source_commit: str,
    model_manifest_path: Path,
    selection_evidence_path: Path,
    path_base: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[Path, ...]]:
    """Authenticate and cross-bind the separate PC-01 runtime-readiness gate.

    The seven producer-evidence roles remain unchanged.  This receipt is
    independently reopened and bound to the executable model bytes, every v3
    evidence digest, the selected report/run contract, and the exact source
    tree which produced the measured DGX result.
    """

    supplied_source = Path(receipt_path)
    receipt = _read_canonical_pc01_checkpoint_compatibility_receipt(
        supplied_source
    )
    source = supplied_source.resolve()
    try:
        receipt = validate_pc01_checkpoint_compatibility_receipt(
            receipt,
            expected_source_commit=expected_source_commit,
        )
    except PC01CheckpointCompatibilityError as exc:
        raise SchemaError(
            f"PC-01 checkpoint compatibility receipt failed validation: {exc}"
        ) from exc
    source_files = _pc01_checkpoint_compatibility_source_files(
        receipt,
        repository_root=repository_root,
    )

    model = read_json(model_manifest_path)
    if (
        model.get("selected_model_id") != receipt.get("model_id")
        or model.get("model_seed") != receipt.get("model_seed")
        or model.get("selected_epoch") != receipt.get("checkpoint_epoch")
    ):
        raise SchemaError(
            "PC-01 checkpoint compatibility receipt identifies another model"
        )
    executable = _validate_model_artifact_payloads(
        model_manifest_path,
        model,
        path_base=path_base,
    )
    evidence = _validate_model_evidence_bundle(
        model_manifest_path,
        model,
        path_base=path_base,
        executable_payloads=executable,
    )
    export = _canonical_model_evidence_json(
        evidence["export_manifest"][0],
        role="export_manifest",
    )
    report_sha256, run_contract_sha256 = _selection_candidate_artifact_hashes(
        selection_evidence_path,
        model_id=str(receipt["model_id"]),
    )
    if (
        export.get("report_sha256") != report_sha256
        or export.get("run_contract_sha256") != run_contract_sha256
    ):
        raise SchemaError(
            "PC-01 v3 export and validation-selection report/run-contract "
            "bytes differ"
        )
    expected_bindings = {
        "export_manifest_sha256": sha256_file(evidence["export_manifest"][0]),
        "checkpoint_sha256": sha256_file(executable["selected_checkpoint"][0]),
        "checkpoint_size_bytes": executable["selected_checkpoint"][0].stat().st_size,
        "resolved_config_sha256": str(model["resolved_config_record_sha256"]),
        "processor_contract_sha256": sha256_file(
            executable["processor_contract"][0]
        ),
        "processor_artifact_manifest_sha256": sha256_file(
            evidence["processor_artifact_manifest"][0]
        ),
        "processor_parity_receipt_sha256": sha256_file(
            evidence["processor_parity_receipt"][0]
        ),
        "base_snapshot_manifest_sha256": sha256_file(
            evidence["base_snapshot_manifest"][0]
        ),
        "base_snapshot_directory_payload_sha256": str(
            executable["e0_backbone"][1]["sha256"]
        ),
        "training_environment_record_sha256": sha256_file(
            evidence["training_environment"][0]
        ),
        "training_environment_source_sha256": str(
            export["training_environment_source_sha256"]
        ),
        "training_source_manifest_sha256": sha256_file(
            evidence["training_source_manifest"][0]
        ),
        "training_action_value_evidence_sha256": sha256_file(
            evidence["training_action_value_evidence"][0]
        ),
        "report_sha256": report_sha256,
        "run_contract_sha256": run_contract_sha256,
    }
    if dict(receipt["artifact_bindings"]) != expected_bindings:
        raise SchemaError(
            "PC-01 checkpoint compatibility receipt is rebound from its "
            "executable/v3/selection artifacts"
        )

    binding = _pc01_checkpoint_compatibility_binding(source, receipt)
    return receipt, binding, source_files


def _pc01_checkpoint_compatibility_binding(
    receipt_path: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a canonical receipt into the frozen runtime identity."""

    return {
        "receipt_sha256": sha256_file(receipt_path),
        "schema_version": receipt["schema_version"],
        "artifact_role": receipt["artifact_role"],
        "gate_id": receipt["gate_id"],
        "status": receipt["status"],
        "model_id": receipt["model_id"],
        "model_seed": receipt["model_seed"],
        "checkpoint_epoch": receipt["checkpoint_epoch"],
        "checkpoint_sha256": receipt["artifact_bindings"]["checkpoint_sha256"],
        "export_manifest_sha256": receipt["artifact_bindings"][
            "export_manifest_sha256"
        ],
        "artifact_bindings": dict(receipt["artifact_bindings"]),
        "source_git_commit": receipt["source_attestation"]["git_commit"],
        "source_manifest_sha256": receipt["source_attestation"][
            "source_manifest_sha256"
        ],
    }


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaError(f"expected YAML mapping: {path}")
    return value


def freeze_campaign(
    *,
    repository_root: str | Path,
    campaign_config_path: str | Path,
    campaign_dir: str | Path,
    campaign_id: str | None = None,
    model_manifest_paths: Sequence[str | Path] = (),
    memory_manifest_paths: Sequence[str | Path] = (),
    environment_manifest_path: str | Path | None = None,
    runner_attestation_path: str | Path | None = None,
    resolved_task_snapshot_path: str | Path | None = None,
    pc01_checkpoint_compatibility_receipt_path: str | Path | None = None,
    handoff_manifest_path: str | Path | None = None,
    allow_dirty_pilot: bool = False,
) -> dict[str, Any]:
    """Copy and hash every campaign-defining input before any scored run."""

    repo = Path(repository_root).resolve()
    config_source = _resolve_input(repo, campaign_config_path)
    campaign_config = load_yaml(config_source)
    protocol_source = _resolve_input(repo, campaign_config["protocol"])
    protocol = load_yaml(protocol_source)
    _validate_registered_protocol(protocol)
    paper_claims = protocol.get("paper_claims")
    if not isinstance(paper_claims, Mapping):
        raise SchemaError("frozen protocol lacks its paper-claim registry binding")
    configured_claim_registry = campaign_config.get("paper_claim_registry")
    if configured_claim_registry is None:
        raise SchemaError("campaign configuration lacks paper_claim_registry")
    paper_claim_registry_source = _resolve_input(repo, configured_claim_registry)
    protocol_claim_registry_source = _resolve_input(
        repo, str(paper_claims.get("registry", ""))
    )
    paper_claim_registry_binding = validate_claim_registry_binding(
        campaign_registry_path=paper_claim_registry_source,
        protocol_registry_path=protocol_claim_registry_source,
        protocol_registry_id=paper_claims.get("registry_id"),
    )
    identifier = campaign_id or str(protocol.get("protocol_id", ""))
    if not identifier:
        raise SchemaError("campaign ID or protocol.protocol_id is required")
    campaign_mode = campaign_config.get("campaign_mode", "evaluation")
    campaign_profile = classify_campaign_profile(
        campaign_config,
        context="campaign configuration",
        default_campaign_mode="evaluation",
    )
    kind = campaign_config["campaign_kind"]
    pilot_only = campaign_profile == CAMPAIGN_PROFILE_PILOT
    configured_handoff = handoff_manifest_path or campaign_config.get(
        "handoff_manifest"
    )
    handoff_manifest_source = (
        _resolve_input(repo, configured_handoff)
        if configured_handoff is not None
        else None
    )
    destination = Path(campaign_dir).resolve()
    if handoff_manifest_source is not None:
        handoff_root = handoff_manifest_source.parent.resolve()
        if (
            destination == handoff_root
            or destination in handoff_root.parents
            or handoff_root in destination.parents
        ):
            raise SchemaError(
                "campaign destination and handoff authority package must be "
                "disjoint (neither may contain the other)"
            )
    if destination.exists() and any(destination.iterdir()):
        raise Table2Error(
            f"campaign directory is not empty and will not be overwritten: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    configured_handoff_in_campaign = campaign_config.get("handoff_manifest")
    if campaign_mode == "smoke" and configured_handoff is not None:
        raise SchemaError(
            "ENGINEERING_SMOKE_ONLY cannot claim an evaluation handoff authority"
        )
    if (
        campaign_mode != "smoke"
        and handoff_manifest_path is not None
        and configured_handoff_in_campaign is not None
        and _resolve_input(repo, configured_handoff_in_campaign)
        != handoff_manifest_source
    ):
        raise SchemaError(
            "explicit handoff manifest differs from frozen campaign configuration"
        )
    registered_exclusion_path = repo / PILOT_TASK_EXCLUSION_REGISTRY_RELATIVE_PATH
    if registered_exclusion_path.is_symlink() or not registered_exclusion_path.is_file():
        raise SchemaError(
            "permanent pilot-task exclusion registry is missing or symlinked"
        )
    pilot_exclusion_source = registered_exclusion_path.resolve()
    try:
        pilot_exclusion_source.relative_to(repo)
    except ValueError as exc:
        raise SchemaError(
            "permanent pilot-task exclusion registry escaped the repository"
        ) from exc
    pilot_exclusion_authority = load_pilot_task_exclusion_authority(
        pilot_exclusion_source
    )
    _validate_protocol_access_boundary(
        protocol,
        campaign_config,
        pilot_only=pilot_only,
    )
    commit, dirty = _git_identity(repo)
    if dirty and not (pilot_only and allow_dirty_pilot):
        raise Table2Error("campaign freeze requires a clean repository; dirty runs are pilot-only")

    matched_seeds = [int(seed) for seed in campaign_config.get("matched_seeds", [42])]
    if matched_seeds != [42]:
        raise SchemaError("Table 2 matched_seeds must be exactly [42]")
    repeat_ids = [int(value) for value in campaign_config.get("repeat_ids", [0])]
    if repeat_ids != list(range(len(repeat_ids))):
        raise SchemaError("repeat_ids must be consecutive from zero for immutable paths")

    # Resolve and validate every defining source before writing the campaign.
    # This ordering is what makes a pilot's zero-locked-read claim auditable.
    configured_environment = environment_manifest_path or campaign_config.get(
        "environment_manifest"
    )
    environment_source = (
        _resolve_input(repo, configured_environment) if configured_environment else None
    )
    if campaign_mode != "smoke" and environment_source is None:
        raise SchemaError("evaluation campaign requires a supplied frozen environment manifest")
    environment_value: Mapping[str, Any] | None = None
    dependency_lock_source: Path | None = None
    deployment_preflight: ValidatedDeploymentPreflight | None = None
    live_deployment: ValidatedPC01LiveDeployment | None = None
    if environment_source is not None:
        environment_value = read_json(environment_source)
        _validate_environment_manifest(environment_value, protocol=protocol)
        if campaign_mode != "smoke":
            deployment_preflight = validate_bound_deployment_preflight(
                environment_value,
                artifact_root=environment_source.parent,
            )
            dependency_lock_source = validate_dependency_lock_for_environment(
                environment_source,
                environment_value,
            )
            live_deployment = validate_bound_pc01_live_deployment(
                environment_value,
                artifact_root=environment_source.parent,
                repository_root=repo,
            )
        elif LIVE_DEPLOYMENT_BINDING_FIELD in environment_value:
            raise SchemaError(
                "engineering smoke cannot claim measured PC-01 live-deployment evidence"
            )
    if campaign_mode != "smoke":
        try:
            assert_locked_mount_inaccessible(repo, protocol)
        except LockedMountPreflightError as exc:
            raise SchemaError(f"evaluation freeze locked-mount preflight failed: {exc}") from exc
    evaluator_source: Path | None = None

    source_reads: list[tuple[str, Path]] = [
        ("campaign_configuration", config_source),
        ("protocol", protocol_source),
        ("paper_claim_registry", paper_claim_registry_source),
        ("permanent_pilot_task_exclusion_registry", pilot_exclusion_source),
    ]
    if protocol_claim_registry_source != paper_claim_registry_source:
        source_reads.append(
            ("registered_paper_claim_registry_source", protocol_claim_registry_source)
        )
    if environment_source is not None:
        source_reads.append(("environment_manifest", environment_source))
    if dependency_lock_source is not None:
        source_reads.append(("dependency_lock", dependency_lock_source))
    if deployment_preflight is not None:
        source_reads.extend(
            (
                ("webarena_deployment_preflight", deployment_preflight.evidence_path),
                (
                    "webarena_service_url_map",
                    deployment_preflight.service_url_map_path,
                ),
            )
        )
    if live_deployment is not None:
        source_reads.extend(
            (
                "pc01_live_deployment:"
                + path.relative_to(live_deployment.package_root).as_posix(),
                path,
            )
            for path in live_deployment.package_files
        )
    task_registry_source = _resolve_input(repo, campaign_config["task_manifest"])
    _reject_locked_mount_path(task_registry_source, repo, protocol, pilot_only=pilot_only)
    registry_metadata, registry_records = _load_task_manifest(task_registry_source)
    _validate_task_boundary(
        registry_records,
        campaign_config,
        protocol,
        metadata=registry_metadata,
        pilot_only=pilot_only,
    )
    source_reads.append(("normal_task_registry", task_registry_source))

    resolved_task_source: Path | None = None
    task_source = task_registry_source
    task_metadata = registry_metadata
    task_records = registry_records
    if campaign_mode != "smoke":
        configured_snapshot = resolved_task_snapshot_path or campaign_config.get(
            "resolved_task_snapshot"
        )
        if configured_snapshot is None:
            raise SchemaError(
                "evaluation campaign requires an explicit resolved WebArena task snapshot"
            )
        if environment_source is None:
            raise SchemaError("resolved task snapshot requires a frozen environment")
        resolved_task_source = _resolve_input(repo, configured_snapshot)
        _reject_locked_mount_path(
            resolved_task_source, repo, protocol, pilot_only=pilot_only
        )
        task_metadata, task_records = _validate_resolved_task_snapshot(
            resolved_task_source,
            registry_path=task_registry_source,
            registry_metadata=registry_metadata,
            registry_tasks=registry_records,
            environment_path=environment_source,
        )
        task_source = resolved_task_source
        source_reads.append(("resolved_normal_task_snapshot", resolved_task_source))
    if deployment_preflight is not None and pilot_only:
        first_registered_index = registry_records[0].get("upstream_index")
        if (
            type(first_registered_index) is not int
            or deployment_preflight.binding.get(
                "expected_live_reset_task_index"
            )
            != first_registered_index
        ):
            raise SchemaError(
                "WebArena deployment preflight did not reset the first task in "
                "exact tracked-registry order"
            )
    if not pilot_only:
        validate_locked_final_pilot_exclusion(
            task_records,
            task_metadata=task_metadata,
            authority=pilot_exclusion_authority,
        )

    benchmark_sources: dict[str, Path] = {}
    for key in (
        "recovery_scenarios",
        "recovery_oracle_rules",
        "audit_manifest",
        "duplicate_audit_manifest",
    ):
        if key not in campaign_config:
            raise SchemaError(f"campaign configuration is missing {key}")
        source = _resolve_input(repo, campaign_config[key])
        _reject_locked_mount_path(source, repo, protocol, pilot_only=pilot_only)
        benchmark_sources[key] = source
        source_reads.append((key, source))
    validate_manual_audit_reviewer_codebook(
        read_json(benchmark_sources["audit_manifest"])
    )
    if environment_source is not None:
        environment_value = read_json(environment_source)
        evaluator_value = environment_value.get("evaluator")
        if not isinstance(evaluator_value, Mapping):
            raise SchemaError("environment evaluator identity is absent")
        evaluator_source = _resolve_input(
            repo, str(evaluator_value.get("source_relative_path", ""))
        )
        try:
            evaluator_source.relative_to(repo)
        except ValueError as exc:
            raise SchemaError("environment evaluator source escaped repository") from exc
        if (
            not evaluator_source.is_file()
            or sha256_file(evaluator_source) != evaluator_value.get("source_sha256")
        ):
            raise SchemaError("environment evaluator source differs from frozen checkout")
        if sha256_file(benchmark_sources["recovery_oracle_rules"]) != evaluator_value.get(
            "oracle_rules_sha256"
        ):
            raise SchemaError("environment evaluator oracle-rules hash differs from campaign")
        source_reads.append(("evaluation_evaluator_source", evaluator_source))
    recovery_records, recovery_metadata = _load_recovery_scenarios(
        benchmark_sources["recovery_scenarios"]
    )
    _validate_recovery_campaign_counts(
        task_records,
        recovery_records,
        recovery_metadata,
        campaign_config,
        matched_seeds=matched_seeds,
        repeat_count=len(repeat_ids),
        pilot_only=pilot_only,
    )
    _validate_duplicate_audit_manifest(
        benchmark_sources["duplicate_audit_manifest"],
        normal_tasks=task_records,
        recovery_scenarios=recovery_records,
        require_verified_normal=campaign_mode != "smoke",
    )
    configured_duplicate = str(campaign_config["duplicate_audit_manifest"])
    if str(task_metadata.get("duplicate_audit_manifest")) != configured_duplicate:
        raise SchemaError("task manifest references a different duplicate-audit manifest")
    if str(recovery_metadata.get("duplicate_audit_manifest")) != configured_duplicate:
        raise SchemaError("recovery manifest references a different duplicate-audit manifest")

    joint_assignment_source: Path | None = None
    joint_preparation_source: Path | None = None
    joint_task_export_source: Path | None = None
    joint_registered_recovery_source: Path | None = None
    joint_registration_source: Path | None = None
    joint_provenance_source: Path | None = None
    if campaign_mode != "smoke":
        required_joint_inputs = {
            "joint_duplicate_assignment_package",
            "p4_preparation_package",
            "joint_duplicate_resolved_task_export",
            "joint_duplicate_registered_recovery_scenarios",
            "joint_duplicate_audit_registration",
            "joint_duplicate_provenance_manifest",
        }
        missing_joint_inputs = sorted(
            required_joint_inputs - set(campaign_config)
        )
        if missing_joint_inputs:
            raise SchemaError(
                "evaluation campaign lacks compact joint duplicate evidence: "
                f"{missing_joint_inputs}"
            )
        joint_assignment_source = _resolve_input(
            repo, campaign_config["joint_duplicate_assignment_package"]
        )
        joint_preparation_source = _resolve_input(
            repo, campaign_config["p4_preparation_package"]
        )
        joint_task_export_source = _resolve_input(
            repo, campaign_config["joint_duplicate_resolved_task_export"]
        )
        joint_registered_recovery_source = _resolve_input(
            repo,
            campaign_config["joint_duplicate_registered_recovery_scenarios"],
        )
        joint_registration_source = _resolve_input(
            repo, campaign_config["joint_duplicate_audit_registration"]
        )
        joint_provenance_source = _resolve_input(
            repo, campaign_config["joint_duplicate_provenance_manifest"]
        )
        for label, package_root, names in (
            (
                "joint_duplicate_assignment",
                joint_assignment_source,
                JOINT_DUPLICATE_ASSIGNMENT_FILES,
            ),
            (
                "p4_preparation",
                joint_preparation_source,
                P4_PREPARATION_EVIDENCE_FILES,
            ),
        ):
            if package_root.is_symlink() or not package_root.is_dir():
                raise SchemaError(f"{label} package is missing or symlinked")
            for name in names:
                path = package_root / name
                if path.is_symlink() or not path.is_file():
                    raise SchemaError(f"{label} package lacks {name}")
                source_reads.append((f"{label}:{name}", path))
        try:
            preparation_receipt_source, preparation_receipt_sidecar_source = (
                locate_prepare_only_execution_receipt(joint_preparation_source)
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SchemaError(
                f"P4 preparation lacks its source-attested execution receipt: {exc}"
            ) from exc
        source_reads.extend(
            (
                (
                    "p4_preparation:execution_receipt.json",
                    preparation_receipt_source,
                ),
                (
                    "p4_preparation:execution_receipt.sha256",
                    preparation_receipt_sidecar_source,
                ),
            )
        )
        for label, path in (
            ("joint_duplicate_resolved_task_export", joint_task_export_source),
            (
                "joint_duplicate_registered_recovery_scenarios",
                joint_registered_recovery_source,
            ),
            ("joint_duplicate_audit_registration", joint_registration_source),
            ("joint_duplicate_provenance_manifest", joint_provenance_source),
            (
                "joint_duplicate_audit_config",
                repo / JOINT_DUPLICATE_AUDIT_CONFIG_RELATIVE_PATH,
            ),
            (
                "p4_source_authority",
                repo / P4_SOURCE_AUTHORITY_RELATIVE_PATH,
            ),
            (
                "joint_duplicate_audit_source",
                repo / AUDIT_TOOL_SOURCE_RELATIVE_PATH,
            ),
        ):
            if path.is_symlink() or not path.is_file():
                raise SchemaError(f"{label} is missing or symlinked")
            source_reads.append((label, path))
    prompt_sources = {
        "parameter_provider_v1.txt": _resolve_input(
            repo, protocol["parameter_provider"]["prompt"]
        ),
        "e0_action_v1.txt": protocol_source.parent / "prompts" / "e0_action_v1.txt",
    }
    source_reads.extend((f"prompt:{name}", path) for name, path in prompt_sources.items())
    overlay = _resolve_input(repo, protocol["systems"]["overlay_directory"])
    overlay_sources: dict[str, Path] = {}
    for system_id in SYSTEM_IDS:
        source = overlay / f"{system_id.lower()}.yaml"
        config = load_yaml(source)
        _validate_system_overlay(config, system_id, source)
        overlay_sources[system_id] = source
        source_reads.append((f"system_overlay:{system_id}", source))

    if campaign_mode != "smoke" and str(
        campaign_config.get("checkpoint_selection", "")
    ) != "validation_only":
        raise SchemaError("evaluation campaigns require validation-only checkpoint selection")
    model_sources = [_resolve_input(repo, value) for value in model_manifest_paths]
    memory_sources = [_resolve_input(repo, value) for value in memory_manifest_paths]
    model_by_seed = _validate_seed_manifest_coverage(
        model_sources,
        matched_seeds,
        kind="model",
        required=campaign_mode != "smoke",
    )
    selection_evidence_source: Path | None = None
    selection_evidence_source_files: tuple[Path, ...] = ()
    configured_selection_evidence = campaign_config.get(
        "checkpoint_selection_evidence"
    )
    if configured_selection_evidence is not None:
        if campaign_mode == "smoke":
            raise SchemaError(
                "engineering smoke must not claim validation-selection evidence"
            )
        selection_evidence_source = _resolve_input(
            repo, configured_selection_evidence
        )
        from .selection_evidence import validate_selection_evidence

        validate_selection_evidence(
            selection_evidence_source,
            repository_root=repo,
            selected_model_manifests={
                seed: read_json(path) for seed, path in sorted(model_by_seed.items())
            },
            expected_model_seeds=matched_seeds,
            expected_selection_mode=_protocol_selection_mode(protocol),
        )
        selection_value = read_json(selection_evidence_source)
        source_identity_fields = [
            "selection_implementation",
            "compatibility_implementation",
            "compatibility_report_writer",
            "compatibility_validator",
            "selection_evidence_implementation",
        ]
        if _protocol_selection_mode(protocol) == "three_candidate_final":
            source_identity_fields.append("comparison_script")
        selection_evidence_source_files = tuple(
            (
                repo
                / str(selection_value[field]["relative_path"])
            ).resolve()
            for field in source_identity_fields
        )
        selection_root = selection_evidence_source.parent
        selection_files = tuple(
            sorted(candidate for candidate in selection_root.rglob("*") if candidate.is_file())
        )
        if not selection_files or any(
            candidate.is_symlink() for candidate in selection_root.rglob("*")
        ):
            raise SchemaError("validation-selection evidence package is empty or unsafe")
        source_reads.extend(
            (
                f"validation_selection_evidence:{candidate.relative_to(selection_root)}",
                candidate,
            )
            for candidate in selection_files
        )
    model_payload_sources: dict[int, dict[str, tuple[Path, dict[str, Any]]]] = {}
    model_evidence_sources: dict[int, dict[str, tuple[Path, dict[str, Any]]]] = {}
    if campaign_mode != "smoke":
        for seed, manifest_path in sorted(model_by_seed.items()):
            model_payload_sources[seed] = _validate_model_artifact_payloads(
                manifest_path, read_json(manifest_path)
            )
            model_evidence_sources[seed] = _validate_model_evidence_bundle(
                manifest_path,
                read_json(manifest_path),
                executable_payloads=model_payload_sources[seed],
            )
            source_reads.extend(
                (f"model_payload:seed_{seed}:{role}", source)
                for role, (source, _) in model_payload_sources[seed].items()
            )
            source_reads.extend(
                (f"model_evidence:seed_{seed}:{role}", source)
                for role, (source, _) in model_evidence_sources[seed].items()
            )

    memory_by_seed = _validate_seed_manifest_coverage(
        memory_sources,
        matched_seeds,
        kind="memory",
        required=campaign_mode != "smoke",
    )
    _validate_model_memory_source_bindings(
        model_by_seed,
        memory_by_seed,
        protocol_source=protocol_source,
    )

    checkpoint_compatibility_source: Path | None = None
    checkpoint_compatibility_binding: dict[str, Any] | None = None
    checkpoint_compatibility_source_files: tuple[Path, ...] = ()
    configured_checkpoint_compatibility = (
        pc01_checkpoint_compatibility_receipt_path
        or campaign_config.get("pc01_checkpoint_compatibility_receipt")
    )
    requires_pc01_checkpoint_compatibility = (
        campaign_mode != "smoke" and pilot_only
    )
    if campaign_mode == "smoke":
        if configured_checkpoint_compatibility is not None:
            raise SchemaError(
                "ENGINEERING_SMOKE_ONLY cannot claim a PC-01 checkpoint "
                "compatibility receipt"
            )
    elif requires_pc01_checkpoint_compatibility:
        if configured_checkpoint_compatibility is None:
            raise SchemaError(
                "PC-01 PILOT_ONLY evaluation requires "
                "pc01_checkpoint_compatibility_receipt"
            )
        if selection_evidence_source is None:
            raise SchemaError(
                "PC-01 checkpoint compatibility requires validation-selection "
                "report/run-contract evidence"
            )
        if set(model_by_seed) != {42}:
            raise SchemaError(
                "PC-01 checkpoint compatibility requires exactly model seed 42"
            )
        checkpoint_compatibility_source = _resolve_input(
            repo, configured_checkpoint_compatibility
        )
        (
            _,
            checkpoint_compatibility_binding,
            checkpoint_compatibility_source_files,
        ) = _validate_pc01_checkpoint_compatibility_readiness(
            checkpoint_compatibility_source,
            repository_root=repo,
            expected_source_commit=commit,
            model_manifest_path=model_by_seed[42],
            selection_evidence_path=selection_evidence_source,
        )
        source_reads.append(
            (
                "pc01_checkpoint_compatibility_receipt",
                checkpoint_compatibility_source,
            )
        )
    elif configured_checkpoint_compatibility is not None:
        raise SchemaError(
            "the PC-01 provisional checkpoint compatibility receipt cannot "
            "authorize a locked-final campaign"
        )
    task_content_manifest = _build_task_content_binding_manifest(
        task_records,
        task_manifest_sha256=sha256_file(task_source),
        memory_by_seed=memory_by_seed,
        evidence_scope=(
            ENGINEERING_SMOKE_SCOPE
            if campaign_mode == "smoke"
            else EVALUATION_RUNNER_SCOPE
        ),
    )
    _validate_duplicate_audit_bindings(
        benchmark_sources["duplicate_audit_manifest"],
        task_content_manifest=task_content_manifest,
        require_verified_normal=campaign_mode != "smoke",
    )
    joint_duplicate_evidence_binding = (
        _validate_registered_joint_duplicate_memory_bindings(
            memory_by_seed,
            duplicate_audit_path=benchmark_sources["duplicate_audit_manifest"],
            required=campaign_mode != "smoke",
            repository_root=repo,
            assignment_package_root=joint_assignment_source,
            preparation_package_root=joint_preparation_source,
            resolved_task_export_path=joint_task_export_source,
            approved_task_registry_path=task_registry_source,
            registered_recovery_scenarios_path=(
                joint_registered_recovery_source
            ),
            duplicate_audit_registration_path=joint_registration_source,
            provenance_manifest_path=joint_provenance_source,
        )
    )
    runner_attestation_source: Path | None = None
    runner_source_files: tuple[Path, ...] = ()
    runtime_integration: dict[str, str] = {}
    if campaign_mode == "smoke":
        if runner_attestation_path is not None or campaign_config.get("runner_attestation"):
            raise SchemaError(
                "engineering smoke must not claim an evaluation runner attestation"
            )
    else:
        integration_entrypoint = str(
            campaign_config.get("runtime_integration_entrypoint", "")
        ).strip()
        integration_source_value = campaign_config.get("runtime_integration_source")
        if ":" not in integration_entrypoint or integration_source_value is None:
            raise SchemaError(
                "evaluation campaign requires a frozen runtime integration entrypoint/source"
            )
        integration_source = _resolve_input(repo, integration_source_value)
        try:
            integration_relative = str(integration_source.relative_to(repo))
        except ValueError as exc:
            raise SchemaError("runtime integration source must be inside the repository") from exc
        if not integration_source.is_file():
            raise SchemaError("runtime integration source is missing from the repository")
        runtime_integration = {
            "entrypoint": integration_entrypoint,
            "source_relative_path": integration_relative,
            "source_sha256": sha256_file(integration_source),
        }
        configured_attestation = (
            runner_attestation_path or campaign_config.get("runner_attestation")
        )
        if configured_attestation is None:
            raise SchemaError("evaluation campaign requires a frozen runner attestation")
        runner_attestation_source = _resolve_input(repo, configured_attestation)
        attestation_value = read_json(runner_attestation_source)
        if (
            attestation_value.get("runner_entrypoint")
            == "web_agent.eval.table2.production_runner:create_runner"
            and selection_evidence_source is None
        ):
            raise SchemaError(
                "production Table 2 runner requires recomputed validation-selection evidence"
            )
        expected_runtime_identity = _expected_runner_runtime_identity(
            model_by_seed=model_by_seed,
            memory_by_seed=memory_by_seed,
            protocol=protocol,
            prompt_sources=prompt_sources,
            environment_path=environment_source,
            resolved_task_snapshot_path=task_source,
            runtime_integration=runtime_integration,
            selection_evidence_path=selection_evidence_source,
            checkpoint_compatibility_receipt_path=(
                checkpoint_compatibility_source
            ),
        )
        runner_source_files = validate_runner_attestation_payload(
            attestation_value,
            repository_root=repo,
            repository_commit=commit,
            expected_runtime_identity=expected_runtime_identity,
        )
        if live_deployment is None:
            raise SchemaError(
                "evaluation campaign lacks measured PC-01 live-deployment evidence"
            )
        required_live_sources = set(live_deployment.capability_source_files)
        required_live_sources.add(
            (repo / "src/web_agent/eval/table2/live_deployment.py").resolve()
        )
        missing_live_sources = required_live_sources - set(runner_source_files)
        if missing_live_sources:
            missing = ", ".join(
                str(path.relative_to(repo)) for path in sorted(missing_live_sources)
            )
            raise SchemaError(
                "runner source attestation omits live capability implementation "
                f"source: {missing}"
            )
        missing_checkpoint_sources = set(
            checkpoint_compatibility_source_files
        ) - set(runner_source_files)
        if missing_checkpoint_sources:
            missing = ", ".join(
                str(path.relative_to(repo))
                for path in sorted(missing_checkpoint_sources)
            )
            raise SchemaError(
                "runner source attestation omits PC-01 checkpoint compatibility "
                f"source: {missing}"
            )
        missing_selection_sources = set(selection_evidence_source_files) - set(
            runner_source_files
        )
        if missing_selection_sources:
            missing = ", ".join(
                str(path.relative_to(repo))
                for path in sorted(missing_selection_sources)
            )
            raise SchemaError(
                "runner source attestation omits validation-selection generator/"
                f"validator source: {missing}"
            )
        required_joint_sources = {
            (repo / AUDIT_TOOL_SOURCE_RELATIVE_PATH).resolve(),
            (repo / JOINT_DUPLICATE_AUDIT_CONFIG_RELATIVE_PATH).resolve(),
            (repo / P4_SOURCE_AUTHORITY_RELATIVE_PATH).resolve(),
            *(
                (repo / relative).resolve()
                for relative in AUDIT_TOOL_DEPENDENCY_RELATIVE_PATHS
            ),
        }
        missing_joint_sources = required_joint_sources - set(runner_source_files)
        if missing_joint_sources:
            missing = ", ".join(
                str(path.relative_to(repo)) for path in sorted(missing_joint_sources)
            )
            raise SchemaError(
                "runner source attestation omits joint duplicate producer/config: "
                f"{missing}"
            )
        required_control_sources = {
            (repo / relative).resolve()
            for relative in EVALUATION_CONTROL_SOURCE_RELATIVE_PATHS
        }
        missing_control_sources = required_control_sources - set(
            runner_source_files
        )
        if missing_control_sources:
            missing = ", ".join(
                str(path.relative_to(repo))
                for path in sorted(missing_control_sources)
            )
            raise SchemaError(
                "runner source attestation omits Table 2 freeze/live/analysis "
                f"control source: {missing}"
            )
        source_reads.append(("evaluation_runner_attestation", runner_attestation_source))
        source_reads.extend(
            (f"evaluation_runner_source:{path.relative_to(repo)}", path)
            for path in runner_source_files
        )
        if integration_source not in runner_source_files:
            raise SchemaError("runtime integration source is absent from runner source set")
        if evaluator_source is None or evaluator_source not in runner_source_files:
            raise SchemaError("evaluator source is absent from runner source set")
    handoff_authority: dict[str, Any] | None = None
    handoff_manifest_initial_sha256: str | None = None
    if campaign_mode != "smoke":
        if handoff_manifest_source is None:
            raise SchemaError(
                "evaluation campaign requires the source-attested handoff manifest"
            )
        if configured_handoff_in_campaign is None:
            raise SchemaError(
                "evaluation campaign configuration must bind handoff_manifest"
            )
        handoff_authority = validate_handoff_freeze_authority(
            handoff_manifest_source,
            repository_root=repo,
            repository_commit=commit,
            campaign_config_path=config_source,
            selection_evidence_path=selection_evidence_source,
            model_manifest_paths=model_sources,
            memory_manifest_paths=memory_sources,
            environment_manifest_path=environment_source,
            runner_attestation_path=runner_attestation_source,
            resolved_task_snapshot_path=resolved_task_source,
            paper_claim_registry_path=paper_claim_registry_source,
            pc01_checkpoint_compatibility_receipt_path=(
                checkpoint_compatibility_source
            ),
        )
        handoff_package = handoff_manifest_source.parent.resolve()
        staged_campaign_inputs: dict[str, Path | None] = {
            "resolved_task_snapshot": resolved_task_source,
            "environment_manifest": environment_source,
            "runner_attestation": runner_attestation_source,
            "checkpoint_selection_evidence": selection_evidence_source,
            "paper_claim_registry": paper_claim_registry_source,
            "pc01_checkpoint_compatibility_receipt": (
                checkpoint_compatibility_source
            ),
            "recovery_scenarios": benchmark_sources["recovery_scenarios"],
            "duplicate_audit_manifest": benchmark_sources[
                "duplicate_audit_manifest"
            ],
            "joint_duplicate_assignment_package": joint_assignment_source,
            "p4_preparation_package": joint_preparation_source,
            "joint_duplicate_resolved_task_export": joint_task_export_source,
            "joint_duplicate_registered_recovery_scenarios": (
                joint_registered_recovery_source
            ),
            "joint_duplicate_audit_registration": joint_registration_source,
            "joint_duplicate_provenance_manifest": joint_provenance_source,
        }
        for field, staged_path in staged_campaign_inputs.items():
            if staged_path is None:
                raise SchemaError(f"evaluation handoff lacks staged {field}")
            resolved_staged_path = staged_path.resolve()
            if (
                resolved_staged_path != handoff_package
                and handoff_package not in resolved_staged_path.parents
            ):
                raise SchemaError(
                    f"evaluation campaign {field} escaped the authenticated "
                    "handoff package"
                )
        handoff_manifest_initial_sha256 = sha256_file(handoff_manifest_source)
        source_reads.append(("evaluation_handoff_manifest", handoff_manifest_source))
        for relative in sorted(handoff_authority["files"]):
            source = handoff_manifest_source.parent / relative
            _reject_locked_mount_path(source, repo, protocol, pilot_only=pilot_only)
            source_reads.append((f"evaluation_handoff_inventory:{relative}", source))
        if (
            live_deployment is None
            or resolved_task_source is None
            or joint_task_export_source is None
        ):
            raise SchemaError(
                "evaluation handoff lacks live deployment or resolved tasks"
            )
        validate_evaluator_requirements_resolved_snapshot_binding(
            live_deployment.evaluator_requirements,
            task_export=read_json(joint_task_export_source),
            resolved_task_snapshot=read_json(resolved_task_source),
        )
    source_reads.extend((f"selected_model_manifest:seed_{seed}", path) for seed, path in model_by_seed.items())
    for seed, manifest_path in memory_by_seed.items():
        for name in FROZEN_MEMORY_STORE_FILES:
            source_reads.append(
                (f"frozen_memory_store:seed_{seed}:{name}", manifest_path.parent / name)
            )

    handoff_package_root = (
        handoff_manifest_source.parent.resolve()
        if handoff_manifest_source is not None
        else None
    )
    handoff_inventory = (
        {str(relative): str(digest) for relative, digest in handoff_authority["files"].items()}
        if handoff_authority is not None
        else {}
    )
    handoff_copy_bindings: list[dict[str, str]] = []
    handoff_copy_destinations: set[str] = set()

    def copy_freeze_input(source: Path, target: Path) -> Path:
        """Copy one input and authenticate handoff bytes on both sides."""

        raw_source = Path(source)
        if raw_source.is_symlink() or not raw_source.is_file():
            raise SchemaError(f"freeze input is missing, not a file, or symlinked: {source}")
        resolved_source = raw_source.resolve()
        expected: str | None = None
        source_relative: str | None = None
        if handoff_package_root is not None:
            try:
                source_relative = resolved_source.relative_to(
                    handoff_package_root
                ).as_posix()
            except ValueError:
                source_relative = None
            if source_relative is not None:
                expected = handoff_inventory.get(source_relative)
                if expected is None:
                    raise SchemaError(
                        "freeze attempted to copy an un-inventoried handoff file: "
                        f"{source_relative}"
                    )
                if sha256_file(resolved_source) != expected:
                    raise SchemaError(
                        "handoff source changed before authenticated copy: "
                        f"{source_relative}"
                    )
        copied_path = _copy_exact(resolved_source, target)
        if expected is not None and source_relative is not None:
            destination_digest = sha256_file(copied_path)
            if (
                destination_digest != expected
                or sha256_file(resolved_source) != expected
            ):
                raise SchemaError(
                    "handoff source changed during authenticated copy: "
                    f"{source_relative}"
                )
            destination_relative = copied_path.relative_to(destination).as_posix()
            if destination_relative in handoff_copy_destinations:
                raise SchemaError(
                    "handoff consumption maps more than one source onto a "
                    f"frozen destination: {destination_relative}"
                )
            handoff_copy_destinations.add(destination_relative)
            handoff_copy_bindings.append(
                {
                    "source_relative_path": source_relative,
                    "campaign_relative_path": destination_relative,
                    "sha256": expected,
                }
            )
        return copied_path

    frozen = destination / "frozen"
    systems_dir = frozen / "systems"
    schedule_dir = destination / "schedule"
    memory_dir = destination / "memory"
    for directory in (
        systems_dir,
        schedule_dir,
        memory_dir,
        destination / "component_test",
        destination / "paired_blocks",
        destination / "aggregate",
        destination / "manual_audit",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    copied.append(copy_freeze_input(config_source, frozen / "campaign.yaml"))
    handoff_manifest_sha256: str | None = None
    handoff_inventory_sha256: str | None = None
    if handoff_manifest_source is not None:
        if handoff_manifest_initial_sha256 is None:
            raise SchemaError("evaluation handoff manifest was not authenticated")
        handoff_copy = _copy_exact(
            handoff_manifest_source, frozen / "handoff_manifest.json"
        )
        if (
            sha256_file(handoff_manifest_source)
            != handoff_manifest_initial_sha256
            or sha256_file(handoff_copy) != handoff_manifest_initial_sha256
        ):
            raise SchemaError("handoff manifest changed during campaign freeze")
        copied.append(handoff_copy)
        handoff_manifest_sha256 = sha256_file(handoff_copy)
        if handoff_authority is None:
            raise SchemaError("evaluation handoff authority was not validated")
        handoff_inventory_sha256 = sha256_json(handoff_authority["files"])
    copied.append(copy_freeze_input(protocol_source, frozen / "protocol.yaml"))
    frozen_claim_registry = copy_freeze_input(
        paper_claim_registry_source,
        destination / FROZEN_PAPER_CLAIM_REGISTRY_RELATIVE_PATH,
    )
    validate_claim_registry(frozen_claim_registry)
    if sha256_file(frozen_claim_registry) != paper_claim_registry_binding[
        "registry_sha256"
    ]:
        raise SchemaError("frozen paper-claim registry bytes changed during copy")
    copied.append(frozen_claim_registry)
    copied.append(
        copy_freeze_input(task_registry_source, frozen / "task_registry.json")
    )

    task_suffix = task_source.suffix.lower() if task_source.suffix.lower() in {".json", ".jsonl", ".csv"} else ".json"
    task_copy = frozen / f"task_manifest{task_suffix}"
    copied.append(copy_freeze_input(task_source, task_copy))
    task_content_path = frozen / "task_content_hashes.json"
    atomic_write_json(task_content_path, task_content_manifest)
    copied.append(task_content_path)

    benchmark_inputs = frozen / "benchmark"
    benchmark_destination_names = {
        "recovery_scenarios": "recovery_scenarios.json",
        "recovery_oracle_rules": "recovery_oracle_rules.json",
        "audit_manifest": "audit_manifest.json",
        "duplicate_audit_manifest": "duplicate_audit_manifest.json",
    }
    for key, source in benchmark_sources.items():
        copied.append(
            copy_freeze_input(
                source, benchmark_inputs / benchmark_destination_names[key]
            )
        )
    if campaign_mode != "smoke":
        assert joint_assignment_source is not None
        assert joint_preparation_source is not None
        assert joint_task_export_source is not None
        assert joint_registered_recovery_source is not None
        assert joint_registration_source is not None
        assert joint_provenance_source is not None
        joint_destination = destination / JOINT_DUPLICATE_EVIDENCE_RELATIVE_PATH
        for name in JOINT_DUPLICATE_ASSIGNMENT_FILES:
            copied.append(
                copy_freeze_input(
                    joint_assignment_source / name,
                    joint_destination / "assignment" / name,
                )
            )
        for name in P4_PREPARATION_EVIDENCE_FILES:
            copied.append(
                copy_freeze_input(
                    joint_preparation_source / name,
                    joint_destination / "preparation" / name,
                )
            )
        for source, name in zip(
            (
                preparation_receipt_source,
                preparation_receipt_sidecar_source,
            ),
            P4_PREPARATION_EXECUTION_EVIDENCE_FILES,
            strict=True,
        ):
            copied.append(copy_freeze_input(source, joint_destination / name))
        copied.extend(
            (
                copy_freeze_input(
                    joint_task_export_source,
                    joint_destination / "resolved_task_export.json",
                ),
                copy_freeze_input(
                    joint_registered_recovery_source,
                    joint_destination / "registered_recovery_scenarios.json",
                ),
                copy_freeze_input(
                    joint_provenance_source,
                    joint_destination / "provenance_manifest.json",
                ),
                copy_freeze_input(
                    repo / JOINT_DUPLICATE_AUDIT_CONFIG_RELATIVE_PATH,
                    joint_destination
                    / "tracked_repository"
                    / JOINT_DUPLICATE_AUDIT_CONFIG_RELATIVE_PATH,
                ),
                copy_freeze_input(
                    repo / P4_SOURCE_AUTHORITY_RELATIVE_PATH,
                    joint_destination
                    / "tracked_repository"
                    / P4_SOURCE_AUTHORITY_RELATIVE_PATH,
                ),
                copy_freeze_input(
                    repo / AUDIT_TOOL_SOURCE_RELATIVE_PATH,
                    joint_destination
                    / "tracked_repository"
                    / AUDIT_TOOL_SOURCE_RELATIVE_PATH,
                ),
                *(
                    copy_freeze_input(
                        repo / relative,
                        joint_destination / "tracked_repository" / relative,
                    )
                    for relative in AUDIT_TOOL_DEPENDENCY_RELATIVE_PATHS
                ),
                *(
                    copy_freeze_input(
                        repo / relative,
                        joint_destination / "tracked_repository" / relative,
                    )
                    for relative in EXECUTED_SOURCE_RELATIVE_PATHS
                    if relative
                    not in {
                        AUDIT_TOOL_SOURCE_RELATIVE_PATH,
                        *AUDIT_TOOL_DEPENDENCY_RELATIVE_PATHS,
                    }
                ),
                copy_freeze_input(
                    repo / PREPARE_ONLY_CONFIG_RELATIVE,
                    joint_destination
                    / "tracked_repository"
                    / PREPARE_ONLY_CONFIG_RELATIVE,
                ),
                copy_freeze_input(
                    joint_registration_source,
                    joint_destination
                    / "tracked_repository"
                    / JOINT_DUPLICATE_AUDIT_REGISTRATION_RELATIVE_PATH,
                ),
            )
        )
    copied.append(
        copy_freeze_input(
            pilot_exclusion_source,
            destination / FROZEN_PILOT_TASK_EXCLUSION_RELATIVE_PATH,
        )
    )

    prompts_dir = frozen / "prompts"
    for destination_name, source in prompt_sources.items():
        copied.append(copy_freeze_input(source, prompts_dir / destination_name))

    for source in overlay_sources.values():
        copied.append(copy_freeze_input(source, systems_dir / source.name))

    selection_evidence_sha256: str | None = None
    if selection_evidence_source is not None:
        selection_root = selection_evidence_source.parent
        selection_destination = frozen / "selection_evidence"
        for source in sorted(
            candidate for candidate in selection_root.rglob("*") if candidate.is_file()
        ):
            copied.append(
                copy_freeze_input(
                    source,
                    selection_destination / source.relative_to(selection_root),
                )
            )
        frozen_selection_manifest = selection_destination / "manifest.json"
        selection_evidence_sha256 = sha256_file(frozen_selection_manifest)
        if handoff_package_root is not None:
            selection_relative = selection_evidence_source.relative_to(
                handoff_package_root
            ).as_posix()
            expected_binding = {
                "source_relative_path": selection_relative,
                "campaign_relative_path": frozen_selection_manifest.relative_to(
                    destination
                ).as_posix(),
                "sha256": handoff_inventory[selection_relative],
            }
            if expected_binding not in handoff_copy_bindings:
                raise SchemaError(
                    "authenticated checkpoint-selection manifest was not copied "
                    "to its canonical frozen destination"
                )

    checkpoint_compatibility_sha256: str | None = None
    if checkpoint_compatibility_source is not None:
        checkpoint_compatibility_copy = (
            frozen / PC01_CHECKPOINT_COMPATIBILITY_RECEIPT_RELATIVE_PATH
        )
        copied.append(
            copy_freeze_input(
                checkpoint_compatibility_source,
                checkpoint_compatibility_copy,
            )
        )
        checkpoint_compatibility_copy.chmod(0o444)
        checkpoint_compatibility_sha256 = sha256_file(
            checkpoint_compatibility_copy
        )
        if checkpoint_compatibility_binding is None or (
            checkpoint_compatibility_sha256
            != checkpoint_compatibility_binding["receipt_sha256"]
        ):
            raise SchemaError(
                "copied PC-01 checkpoint compatibility receipt changed bytes"
            )

    runner_attestation_sha256: str | None = None
    runner_entrypoint: str | None = None
    if runner_attestation_source is not None:
        runner_attestation_copy = frozen / "runner_attestation.json"
        copied.append(
            copy_freeze_input(runner_attestation_source, runner_attestation_copy)
        )
        runner_attestation_sha256 = sha256_file(runner_attestation_copy)
        runner_entrypoint = str(read_json(runner_attestation_copy)["runner_entrypoint"])
        for source in runner_source_files:
            relative = source.relative_to(repo)
            copied.append(
                copy_freeze_input(source, frozen / "runner_source" / relative)
            )

    model_payloads_by_seed: dict[str, dict[str, dict[str, Any]]] = {}
    model_evidence_by_seed: dict[str, dict[str, Any]] = {}
    for seed, source in sorted(model_by_seed.items()):
        if campaign_mode == "smoke":
            copied.append(
                copy_freeze_input(
                    source, frozen / "models" / f"seed_{seed}.json"
                )
            )
            continue
        _, descriptors, evidence_bundle = _copy_model_payloads(
            campaign_root=destination,
            seed=seed,
            manifest_path=source,
            manifest=read_json(source),
            copied=copied,
            copy_file=copy_freeze_input,
        )
        model_payloads_by_seed[str(seed)] = descriptors
        model_evidence_by_seed[str(seed)] = evidence_bundle
    for seed, source in sorted(memory_by_seed.items()):
        store_destination = memory_dir / f"seed_{seed}"
        for name in FROZEN_MEMORY_STORE_FILES:
            copied.append(
                copy_freeze_input(source.parent / name, store_destination / name)
            )
        _load_and_verify_frozen_memory_store(store_destination)

    campaign_seed = int(protocol["statistics"].get("seed", 20250831))
    reruns = int(protocol["budgets"].get("whole_block_infrastructure_reruns", 0))
    # The research-locked pilot runs the controlled recovery gate first.  Only
    # after all 15 matched diagnostic blocks are attempted does the schedule
    # advance to the 50 ordinary WebArena development tasks.
    scheduled_tasks = [
        {
            **dict(row),
            "task_id": str(row["scenario_id"]),
            "task_partition": "recovery_diagnostic",
        }
        for row in recovery_records
    ] + [
        {**dict(row), "task_partition": "normal"} for row in task_records
    ]
    schedule = build_paired_schedule(
        scheduled_tasks,
        model_seeds=matched_seeds,
        repetitions=len(repeat_ids),
        campaign_seed=campaign_seed,
        protocol_id=str(protocol["protocol_id"]),
        campaign_id=identifier,
        max_block_attempts=1 + reruns,
    )
    schedule_path = schedule_dir / "schedule.jsonl"
    schedule_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in schedule))
    copied.append(schedule_path)

    frozen_at = datetime.now(timezone.utc).isoformat()
    environment_path = frozen / "environment.json"
    if environment_source is not None:
        copy_freeze_input(environment_source, environment_path)
    else:
        atomic_write_json(
            environment_path,
            {
                "schema_version": SCHEMA_VERSION,
                "smoke_only": True,
                "captured_at_utc": frozen_at,
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "executable": sys.executable,
                "hardware_and_driver_details": "SMOKE_ONLY_NOT_MEASURED",
            },
        )
    copied.append(environment_path)
    if deployment_preflight is not None:
        frozen_preflight = copy_freeze_input(
            deployment_preflight.evidence_path,
            frozen / PREFLIGHT_ARTIFACT_RELATIVE_PATH,
        )
        frozen_url_map = copy_freeze_input(
            deployment_preflight.service_url_map_path,
            frozen / PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH,
        )
        copied.extend((frozen_preflight, frozen_url_map))
        frozen_preflight_receipt = validate_bound_deployment_preflight(
            read_json(environment_path),
            artifact_root=frozen,
        )
        if frozen_preflight_receipt.binding != deployment_preflight.binding:
            raise SchemaError("frozen WebArena deployment preflight binding changed")
    if live_deployment is not None:
        for source in live_deployment.package_files:
            relative = source.relative_to(live_deployment.package_root)
            copied.append(
                copy_freeze_input(
                    source,
                    frozen / "live_deployment" / relative,
                )
            )
        frozen_live_deployment = validate_bound_pc01_live_deployment(
            read_json(environment_path),
            artifact_root=frozen,
            repository_root=frozen / "runner_source",
        )
        if frozen_live_deployment.binding != live_deployment.binding:
            raise SchemaError("frozen PC-01 live-deployment binding changed")
    if dependency_lock_source is not None:
        frozen_dependency_lock = copy_freeze_input(
            dependency_lock_source,
            frozen / FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH,
        )
        if sha256_file(frozen_dependency_lock) != read_json(environment_path).get(
            "dependency_lock_sha256"
        ):
            raise SchemaError("copied dependency lock differs from environment")
        copied.append(frozen_dependency_lock)

    handoff_consumption_sha256: str | None = None
    if handoff_authority is not None:
        if handoff_manifest_source is None or handoff_package_root is None:
            raise SchemaError("validated handoff authority lost its package root")
        arguments = handoff_authority.get("freeze_arguments")
        if not isinstance(arguments, Mapping):
            raise SchemaError("validated handoff authority lost freeze arguments")
        registered_argument_paths: list[Path] = []
        for field in (
            "campaign_config",
            "resolved_task_snapshot",
            "environment_manifest",
            "runner_attestation",
            "checkpoint_selection_evidence",
            "paper_claim_registry",
            "pc01_checkpoint_compatibility_receipt",
        ):
            raw_value = arguments.get(field)
            if raw_value is not None:
                registered_argument_paths.append(Path(str(raw_value)).resolve())
        for field in ("model_manifests", "memory_manifests"):
            raw_values = arguments.get(field)
            if not isinstance(raw_values, list):
                raise SchemaError(
                    f"validated handoff authority {field} is not a path array"
                )
            registered_argument_paths.extend(
                Path(str(raw_value)).resolve() for raw_value in raw_values
            )

        consumed_sources = {
            row["source_relative_path"] for row in handoff_copy_bindings
        }
        required_argument_relatives: set[str] = set()
        for source in registered_argument_paths:
            try:
                source_relative = source.relative_to(
                    handoff_package_root
                ).as_posix()
            except ValueError as exc:
                raise SchemaError(
                    "handoff freeze argument escaped its authenticated package"
                ) from exc
            required_argument_relatives.add(source_relative)
            if source_relative not in consumed_sources:
                copied.append(
                    copy_freeze_input(
                        source,
                        frozen / "handoff_inputs" / source_relative,
                    )
                )
                consumed_sources.add(source_relative)
        if not required_argument_relatives.issubset(consumed_sources):
            raise SchemaError(
                "not every authenticated handoff freeze argument was consumed"
            )

        replayed_handoff = validate_handoff_freeze_authority(
            handoff_manifest_source,
            repository_root=repo,
            repository_commit=commit,
            campaign_config_path=config_source,
            selection_evidence_path=selection_evidence_source,
            model_manifest_paths=model_sources,
            memory_manifest_paths=memory_sources,
            environment_manifest_path=environment_source,
            runner_attestation_path=runner_attestation_source,
            resolved_task_snapshot_path=resolved_task_source,
            paper_claim_registry_path=paper_claim_registry_source,
            pc01_checkpoint_compatibility_receipt_path=(
                checkpoint_compatibility_source
            ),
        )
        if replayed_handoff != handoff_authority:
            raise SchemaError("handoff authority changed during campaign freeze")
        if (
            handoff_manifest_initial_sha256 is None
            or sha256_file(handoff_manifest_source)
            != handoff_manifest_initial_sha256
            or sha256_file(frozen / "handoff_manifest.json")
            != handoff_manifest_initial_sha256
        ):
            raise SchemaError("handoff manifest changed during campaign freeze")

        handoff_consumption_path = (
            destination / FROZEN_HANDOFF_CONSUMPTION_RELATIVE_PATH
        )
        atomic_write_json(
            handoff_consumption_path,
            {
                "schema_version": HANDOFF_CONSUMPTION_SCHEMA_VERSION,
                "handoff_manifest_sha256": handoff_manifest_initial_sha256,
                "handoff_inventory_sha256": sha256_json(handoff_inventory),
                "required_freeze_argument_sources": sorted(
                    required_argument_relatives
                ),
                "bindings": sorted(
                    handoff_copy_bindings,
                    key=lambda row: (
                        row["source_relative_path"],
                        row["campaign_relative_path"],
                    ),
                ),
            },
        )
        copied.append(handoff_consumption_path)
        handoff_consumption_sha256 = sha256_file(handoff_consumption_path)
    provenance_path = frozen / "provenance.json"
    atomic_write_json(
        provenance_path,
        {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": identifier,
            "repository_commit": commit,
            "repository_dirty": dirty,
            "checkpoint_selection": campaign_config.get("checkpoint_selection"),
            "selection_mode": _protocol_selection_mode(protocol),
            "matched_seeds": matched_seeds,
            "model_manifest_seeds": sorted(model_by_seed),
            "memory_manifest_seeds": sorted(memory_by_seed),
            "pilot_task_exclusion_authority": pilot_task_exclusion_provenance(
                pilot_exclusion_authority
            ),
            "locked_test_reads": sum(
                _is_locked_source(path, repo, protocol) for _, path in source_reads
            ),
            "source_files": {
                purpose: {
                    "path": _display_source(path, repo),
                    "sha256": _payload_sha256(path),
                }
                for purpose, path in source_reads
            },
        },
    )
    copied.append(provenance_path)

    access_records = [
        {
            "event_type": "campaign_defining_file_read",
            "timestamp_utc": frozen_at,
            "payload": {
                "purpose": purpose,
                "source_path": _display_source(path, repo),
                "source_sha256": _payload_sha256(path),
                "locked_test_content": _is_locked_source(path, repo, protocol),
            },
        }
        for purpose, path in source_reads
    ]
    if handoff_authority is not None:
        if handoff_manifest_source is None:
            raise SchemaError("validated handoff authority lost its manifest")
        final_handoff_replay = validate_handoff_freeze_authority(
            handoff_manifest_source,
            repository_root=repo,
            repository_commit=commit,
            campaign_config_path=config_source,
            selection_evidence_path=selection_evidence_source,
            model_manifest_paths=model_sources,
            memory_manifest_paths=memory_sources,
            environment_manifest_path=environment_source,
            runner_attestation_path=runner_attestation_source,
            resolved_task_snapshot_path=resolved_task_source,
            paper_claim_registry_path=paper_claim_registry_source,
            pc01_checkpoint_compatibility_receipt_path=(
                checkpoint_compatibility_source
            ),
        )
        if final_handoff_replay != handoff_authority:
            raise SchemaError(
                "handoff authority changed while provenance was recorded"
            )
        if (
            handoff_manifest_initial_sha256 is None
            or sha256_file(handoff_manifest_source)
            != handoff_manifest_initial_sha256
            or sha256_file(frozen / "handoff_manifest.json")
            != handoff_manifest_initial_sha256
        ):
            raise SchemaError(
                "handoff manifest changed while provenance was recorded"
            )
    locked_reads = sum(
        strict_bool(
            record["payload"]["locked_test_content"],
            context="freeze.locked_test_content",
        )
        for record in access_records
    )
    if pilot_only and locked_reads:
        raise SchemaError("pilot freeze recorded a forbidden locked-test read")
    access_tail = _write_ledger(destination / "access_ledger.jsonl", "access", access_records)
    deviation_tail = _write_ledger(destination / "deviation_ledger.jsonl", "deviation", [])

    hashes = {
        str(path.relative_to(destination)): sha256_file(path)
        for path in sorted(copied)
    }
    hash_path = destination / "artifact_hashes.json"
    atomic_write_json(
        hash_path,
        {
            "schema_version": SCHEMA_VERSION,
            "hash_algorithm": "sha256",
            "files": hashes,
        },
    )
    # The frozen campaign exists before the outcome-label-hidden human audit.
    # ``evidence_label`` records the registered engineering scope, whereas
    # ``publication_status`` must remain a draft until the separate,
    # post-execution adjudication authority validates.
    publication_status = DRAFT_PILOT_STATUS if pilot_only else "N/R"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": identifier,
        "campaign_kind": kind,
        "campaign_mode": campaign_mode,
        "protocol_id": protocol.get("protocol_id"),
        "evidence_label": (
            PILOT_EVIDENCE_LABEL if pilot_only else FINAL_EVIDENCE_LABEL
        ),
        "publication_status": publication_status,
        "paper_table_status": "N/R",
        "paper_claim_registry_id": paper_claim_registry_binding["registry_id"],
        "paper_claim_registry_sha256": paper_claim_registry_binding[
            "registry_sha256"
        ],
        "repository_commit": commit,
        "repository_dirty_at_freeze": dirty,
        "runner_identity_scope": (
            ENGINEERING_SMOKE_SCOPE
            if campaign_mode == "smoke"
            else EVALUATION_RUNNER_SCOPE
        ),
        "runner_entrypoint": runner_entrypoint,
        "runner_attestation_sha256": runner_attestation_sha256,
        "handoff_manifest_sha256": handoff_manifest_sha256,
        "handoff_inventory_sha256": handoff_inventory_sha256,
        "handoff_consumption_sha256": handoff_consumption_sha256,
        "runtime_integration_entrypoint": runtime_integration.get("entrypoint"),
        "runtime_integration": runtime_integration,
        "selection_mode": _protocol_selection_mode(protocol),
        "checkpoint_selection_evidence_sha256": selection_evidence_sha256,
        "pc01_checkpoint_compatibility_receipt_sha256": (
            checkpoint_compatibility_sha256
        ),
        PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD: (
            checkpoint_compatibility_binding
        ),
        "task_registry_sha256": sha256_file(frozen / "task_registry.json"),
        "resolved_task_snapshot_sha256": sha256_file(task_copy),
        "webarena_deployment_topology": (
            deployment_preflight.binding["deployment_topology"]
            if deployment_preflight is not None
            else None
        ),
        "webarena_preflight_artifact_sha256": (
            deployment_preflight.binding["preflight_artifact_sha256"]
            if deployment_preflight is not None
            else None
        ),
        "webarena_preflight_content_sha256": (
            deployment_preflight.binding["preflight_content_sha256"]
            if deployment_preflight is not None
            else None
        ),
        "webarena_service_url_map_sha256": (
            deployment_preflight.binding["service_url_map_content_sha256"]
            if deployment_preflight is not None
            else None
        ),
        "webarena_expected_dgx_model_runtime_identity_sha256": (
            deployment_preflight.binding[
                "expected_dgx_model_runtime_identity_sha256"
            ]
            if deployment_preflight is not None
            else None
        ),
        "webarena_expected_bridge_identity_sha256": (
            deployment_preflight.binding["expected_bridge_identity_sha256"]
            if deployment_preflight is not None
            else None
        ),
        "pilot_task_exclusion_registry_relative_path": str(
            FROZEN_PILOT_TASK_EXCLUSION_RELATIVE_PATH
        ),
        "pilot_task_exclusion_registry_sha256": (
            pilot_exclusion_authority.registry_sha256
        ),
        "pilot_task_exclusion_identity_version": (
            PILOT_TASK_EXCLUSION_IDENTITY_VERSION
        ),
        "model_payloads_by_seed": model_payloads_by_seed,
        "model_evidence_by_seed": model_evidence_by_seed,
        "joint_duplicate_evidence_binding": joint_duplicate_evidence_binding,
        "frozen_at_utc": frozen_at,
        "systems": list(SYSTEM_IDS),
        "matched_seeds": matched_seeds,
        "campaign_seed": campaign_seed,
        "repeat_ids": repeat_ids,
        "scheduled_block_count": len(schedule),
        "normal_task_count": len(task_records),
        "recovery_scenario_count": len(recovery_records),
        "normal_block_count": len(task_records) * len(matched_seeds) * len(repeat_ids),
        "recovery_block_count": len(recovery_records) * len(matched_seeds) * len(repeat_ids),
        "planned_episode_count": len(schedule) * len(SYSTEM_IDS),
        "max_block_attempts": 1 + reruns,
        "schedule_sha256": sha256_file(schedule_path),
        "artifact_hashes_sha256": sha256_file(hash_path),
        "locked_test_access": campaign_config.get("locked_test_access", "forbidden"),
        "manual_rescue": campaign_config.get("manual_rescue"),
        "locked_test_reads_at_freeze": locked_reads,
        "access_ledger_records_at_freeze": len(access_records),
        "access_ledger_tail": access_tail,
        "deviation_ledger_tail_at_freeze": deviation_tail,
        "runtime_oracle_access": "forbidden",
        "sealed_verifier_return": "opaque_terminal_signal",
    }
    if live_deployment is not None:
        manifest[LIVE_DEPLOYMENT_BINDING_FIELD] = live_deployment.binding
    atomic_write_json(destination / "campaign_manifest.json", manifest)
    return manifest


def validate_campaign(
    campaign_dir: str | Path,
    *,
    require_complete: bool = True,
    require_aggregates: bool = False,
) -> ValidationReport:
    """Validate a campaign with every public research boundary enforced."""

    return _validate_campaign_core(
        campaign_dir,
        require_complete=require_complete,
        require_aggregates=require_aggregates,
        enforce_live_readiness=True,
    )


def _validate_campaign_for_live_readiness_replay(
    campaign_dir: str | Path,
) -> ValidationReport:
    """Structural replay used only while issuing/reopening the readiness gate."""

    return _validate_campaign_core(
        campaign_dir,
        require_complete=False,
        require_aggregates=False,
        enforce_live_readiness=False,
    )


def _validate_campaign_core(
    campaign_dir: str | Path,
    *,
    require_complete: bool,
    require_aggregates: bool,
    enforce_live_readiness: bool,
) -> ValidationReport:
    root = Path(campaign_dir).resolve()
    report = ValidationReport(campaign_dir=str(root))
    try:
        manifest = read_json(root / "campaign_manifest.json")
        require_keys(
            manifest,
            (
                "schema_version",
                "campaign_id",
                "campaign_kind",
                "campaign_mode",
                "evidence_label",
                "publication_status",
                "paper_claim_registry_id",
                "paper_claim_registry_sha256",
                "systems",
                "matched_seeds",
                "campaign_seed",
                "runner_identity_scope",
                "runner_entrypoint",
                "runner_attestation_sha256",
                "handoff_manifest_sha256",
                "handoff_inventory_sha256",
                "handoff_consumption_sha256",
                "runtime_integration_entrypoint",
                "runtime_integration",
                "checkpoint_selection_evidence_sha256",
                "pc01_checkpoint_compatibility_receipt_sha256",
                PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD,
                "task_registry_sha256",
                "resolved_task_snapshot_sha256",
                "webarena_deployment_topology",
                "webarena_preflight_artifact_sha256",
                "webarena_preflight_content_sha256",
                "webarena_service_url_map_sha256",
                "webarena_expected_dgx_model_runtime_identity_sha256",
                "webarena_expected_bridge_identity_sha256",
                "pilot_task_exclusion_registry_relative_path",
                "pilot_task_exclusion_registry_sha256",
                "pilot_task_exclusion_identity_version",
                "model_payloads_by_seed",
                "model_evidence_by_seed",
                "scheduled_block_count",
                "normal_block_count",
                "recovery_block_count",
                "planned_episode_count",
                "schedule_sha256",
                "artifact_hashes_sha256",
                "access_ledger_tail",
                "manual_rescue",
            ),
            context="campaign_manifest",
        )
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise SchemaError("campaign_manifest schema version does not match evaluator")
        campaign_profile = classify_campaign_profile(
            manifest,
            context="campaign_manifest",
            require_campaign_mode=True,
        )
        pilot = campaign_profile == CAMPAIGN_PROFILE_PILOT
        expected_initial_publication_status = (
            DRAFT_PILOT_STATUS if pilot else "N/R"
        )
        if manifest["publication_status"] != expected_initial_publication_status:
            raise SchemaError(
                "campaign_manifest publication_status must remain at its frozen "
                f"initial value {expected_initial_publication_status!r}"
            )
        if tuple(manifest["systems"]) != SYSTEM_IDS:
            raise SchemaError("campaign_manifest must register exactly E0--E3")
        for directory in (
            "frozen",
            "component_test",
            "memory",
            "schedule",
            "paired_blocks",
            "aggregate",
            "manual_audit",
        ):
            if not (root / directory).is_dir():
                raise SchemaError(f"campaign is missing canonical directory: {directory}/")
        access_count, access_tail = _verify_ledger(root / "access_ledger.jsonl", "access")
        _verify_ledger(root / "deviation_ledger.jsonl", "deviation")
        frozen_access_count = int(manifest.get("access_ledger_records_at_freeze", -1))
        access_records = read_jsonl(root / "access_ledger.jsonl")
        if access_count < frozen_access_count or frozen_access_count < 0:
            raise SchemaError("access ledger lost its frozen prefix")
        prefix_tail = (
            str(access_records[frozen_access_count - 1]["record_hash"])
            if frozen_access_count
            else "0" * 64
        )
        if prefix_tail != manifest["access_ledger_tail"]:
            raise SchemaError("access ledger frozen-prefix tail differs from campaign_manifest")
        for record in access_records:
            payload = record.get("payload", {})
            if not isinstance(payload, Mapping):
                raise SchemaError("campaign access ledger payload must be an object")
            if "locked_test_content" not in payload:
                raise SchemaError("campaign access record lacks locked_test_content flag")
            locked = strict_bool(
                payload["locked_test_content"], context="access.locked_test_content"
            )
            if pilot and locked:
                raise SchemaError("campaign access ledger contains a locked-test read")
        _validate_frozen_hashes(root, manifest)
        _validate_frozen_handoff_binding(root, manifest)
        pilot_exclusion_authority = _validate_frozen_pilot_task_exclusion_binding(
            root, manifest
        )
        _validate_frozen_model_and_memory_manifests(root, manifest)
        campaign_config = load_yaml(root / "frozen" / "campaign.yaml")
        protocol = load_yaml(root / "frozen" / "protocol.yaml")
        _validate_registered_protocol(protocol)
        frozen_claim_registry_path = (
            root / FROZEN_PAPER_CLAIM_REGISTRY_RELATIVE_PATH
        )
        frozen_claim_registry = validate_claim_registry(
            frozen_claim_registry_path
        )
        protocol_claims = protocol.get("paper_claims")
        if not isinstance(protocol_claims, Mapping):
            raise SchemaError("frozen protocol lost its paper-claim binding")
        if (
            frozen_claim_registry.get("registry_id")
            != PAPER_CLAIM_REGISTRY_ID
            or protocol_claims.get("registry_id") != PAPER_CLAIM_REGISTRY_ID
            or manifest.get("paper_claim_registry_id")
            != PAPER_CLAIM_REGISTRY_ID
            or frozen_claim_registry.get("paper_table_status") != "N/R"
            or manifest.get("paper_table_status") != "N/R"
        ):
            raise SchemaError("frozen paper-claim registry identity differs")
        if manifest.get("paper_claim_registry_sha256") != sha256_file(
            frozen_claim_registry_path
        ):
            raise SchemaError("frozen paper-claim registry hash differs")
        if not isinstance(campaign_config.get("paper_claim_registry"), str) or not str(
            campaign_config["paper_claim_registry"]
        ).strip():
            raise SchemaError("frozen campaign lost paper_claim_registry")
        _validate_protocol_access_boundary(
            protocol,
            campaign_config,
            pilot_only=pilot,
        )
        environment = read_json(root / "frozen" / "environment.json")
        if str(manifest.get("campaign_mode")) == "smoke":
            if environment.get("smoke_only") is not True:
                _validate_environment_manifest(environment, protocol=protocol)
            if (
                LIVE_DEPLOYMENT_BINDING_FIELD in environment
                or LIVE_DEPLOYMENT_BINDING_FIELD in manifest
                or (root / "frozen" / "live_deployment").exists()
            ):
                raise SchemaError(
                    "engineering smoke cannot claim measured PC-01 live-deployment evidence"
                )
            if (
                manifest.get("pc01_checkpoint_compatibility_receipt_sha256")
                is not None
                or manifest.get(PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD)
                is not None
                or (
                    root
                    / "frozen"
                    / PC01_CHECKPOINT_COMPATIBILITY_RECEIPT_RELATIVE_PATH
                ).exists()
            ):
                raise SchemaError(
                    "ENGINEERING_SMOKE_ONLY cannot claim PC-01 checkpoint "
                    "compatibility evidence"
                )
        else:
            _validate_environment_manifest(environment, protocol=protocol)
            deployment_preflight = validate_bound_deployment_preflight(
                environment,
                artifact_root=root / "frozen",
            )
            expected_preflight_manifest = {
                "webarena_deployment_topology": deployment_preflight.binding[
                    "deployment_topology"
                ],
                "webarena_preflight_artifact_sha256": (
                    deployment_preflight.binding["preflight_artifact_sha256"]
                ),
                "webarena_preflight_content_sha256": (
                    deployment_preflight.binding["preflight_content_sha256"]
                ),
                "webarena_service_url_map_sha256": (
                    deployment_preflight.binding["service_url_map_content_sha256"]
                ),
                "webarena_expected_dgx_model_runtime_identity_sha256": (
                    deployment_preflight.binding[
                        "expected_dgx_model_runtime_identity_sha256"
                    ]
                ),
                "webarena_expected_bridge_identity_sha256": (
                    deployment_preflight.binding[
                        "expected_bridge_identity_sha256"
                    ]
                ),
            }
            for field, expected in expected_preflight_manifest.items():
                if manifest.get(field) != expected:
                    raise SchemaError(
                        f"campaign {field} differs from frozen WebArena preflight"
                    )
            live_deployment = validate_bound_pc01_live_deployment(
                environment,
                artifact_root=root / "frozen",
                repository_root=root / "frozen" / "runner_source",
            )
            if manifest.get(LIVE_DEPLOYMENT_BINDING_FIELD) != (
                live_deployment.binding
            ):
                raise SchemaError(
                    "campaign PC-01 live-deployment binding differs from frozen evidence"
                )
            validate_frozen_dependency_lock(root)
        frozen_campaign_profile = classify_campaign_profile(
            campaign_config,
            context="frozen campaign configuration",
            default_campaign_mode="evaluation",
        )
        if frozen_campaign_profile != campaign_profile:
            raise SchemaError(
                "frozen campaign profile differs from campaign_manifest"
            )
        if campaign_config.get("campaign_mode", "evaluation") != manifest.get(
            "campaign_mode"
        ):
            raise SchemaError("frozen campaign mode differs from campaign_manifest")
        if str(protocol.get("protocol_id")) != str(manifest.get("protocol_id")):
            raise SchemaError("frozen protocol ID differs from campaign_manifest")
        if manifest.get("selection_mode") != _protocol_selection_mode(protocol):
            raise SchemaError(
                "frozen protocol selection mode differs from campaign_manifest"
            )
        if campaign_config.get("manual_rescue") != manifest.get("manual_rescue"):
            raise SchemaError(
                "frozen manual-rescue policy differs from campaign_manifest"
            )
        for system_id in SYSTEM_IDS:
            overlay_path = root / "frozen" / "systems" / f"{system_id.lower()}.yaml"
            _validate_system_overlay(load_yaml(overlay_path), system_id, overlay_path)
        if str(manifest.get("campaign_mode")) != "smoke" and str(
            campaign_config.get("checkpoint_selection")
        ) != "validation_only":
            raise SchemaError("evaluation campaign is not validation-only selected")
        production_entrypoint = (
            "web_agent.eval.table2.production_runner:create_runner"
        )
        selection_path = root / "frozen" / "selection_evidence" / "manifest.json"
        if manifest.get("runner_entrypoint") == production_entrypoint:
            expected_selection_sha = manifest.get(
                "checkpoint_selection_evidence_sha256"
            )
            if (
                not selection_path.is_file()
                or sha256_file(selection_path) != expected_selection_sha
            ):
                raise SchemaError(
                    "production campaign lacks frozen validation-selection evidence"
                )
            from .selection_evidence import validate_selection_evidence

            validate_selection_evidence(
                selection_path,
                repository_root=root / "frozen" / "runner_source",
                selected_model_manifests={
                    int(seed): read_json(
                        root / "frozen" / "models" / f"seed_{int(seed)}.json"
                    )
                    for seed in manifest["matched_seeds"]
                },
                expected_model_seeds=[int(seed) for seed in manifest["matched_seeds"]],
                expected_selection_mode=_protocol_selection_mode(protocol),
            )
        elif manifest.get("checkpoint_selection_evidence_sha256") is not None:
            # A source-attested evaluation harness may carry the same replayed
            # validation package because the separate DGX compatibility
            # receipt is cryptographically bound to its report/run contract.
            from .selection_evidence import validate_selection_evidence

            validate_selection_evidence(
                selection_path,
                repository_root=root / "frozen" / "runner_source",
                selected_model_manifests={
                    int(seed): read_json(
                        root / "frozen" / "models" / f"seed_{int(seed)}.json"
                    )
                    for seed in manifest["matched_seeds"]
                },
                expected_model_seeds=[int(seed) for seed in manifest["matched_seeds"]],
                expected_selection_mode=_protocol_selection_mode(protocol),
            )
        if (
            str(manifest.get("campaign_mode")) != "smoke"
            and pilot
        ):
            compatibility_path = (
                root
                / "frozen"
                / PC01_CHECKPOINT_COMPATIBILITY_RECEIPT_RELATIVE_PATH
            )
            if selection_path.is_file():
                compatibility_selection_path = selection_path
            else:
                configured_selection = root / "frozen" / "selection_evidence" / "manifest.json"
                if not configured_selection.is_file():
                    raise SchemaError(
                        "PC-01 checkpoint compatibility requires frozen "
                        "validation-selection evidence"
                    )
                compatibility_selection_path = configured_selection
            _, compatibility_binding, _ = (
                _validate_pc01_checkpoint_compatibility_readiness(
                    compatibility_path,
                    repository_root=root / "frozen" / "runner_source",
                    expected_source_commit=str(manifest["repository_commit"]),
                    model_manifest_path=(
                        root / "frozen" / "models" / "seed_42.json"
                    ),
                    selection_evidence_path=compatibility_selection_path,
                    path_base=root,
                )
            )
            if manifest.get(
                "pc01_checkpoint_compatibility_receipt_sha256"
            ) != compatibility_binding["receipt_sha256"]:
                raise SchemaError(
                    "campaign checkpoint compatibility receipt hash differs"
                )
            if manifest.get(
                PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD
            ) != compatibility_binding:
                raise SchemaError(
                    "campaign PC-01 checkpoint compatibility binding differs"
                )
        elif not pilot and (
            manifest.get("pc01_checkpoint_compatibility_receipt_sha256")
            is not None
            or manifest.get(PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD)
            is not None
            or (
                root
                / "frozen"
                / PC01_CHECKPOINT_COMPATIBILITY_RECEIPT_RELATIVE_PATH
            ).exists()
        ):
            raise SchemaError(
                "locked-final campaign falsely claims the provisional PC-01 "
                "checkpoint compatibility receipt"
            )
        if pilot:
            if protocol.get("evidence_label") != PILOT_EVIDENCE_LABEL:
                raise SchemaError("pilot evidence labels disagree across frozen artifacts")
        else:
            if campaign_profile != CAMPAIGN_PROFILE_FINAL:
                raise SchemaError("non-pilot campaign must be registered locked_final")
        task_candidates = list((root / "frozen").glob("task_manifest.*"))
        if len(task_candidates) != 1:
            raise SchemaError("frozen campaign must contain exactly one task_manifest.*")
        task_registry_path = root / "frozen" / "task_registry.json"
        if (
            not task_registry_path.is_file()
            or sha256_file(task_registry_path) != manifest.get("task_registry_sha256")
            or sha256_file(task_candidates[0])
            != manifest.get("resolved_task_snapshot_sha256")
        ):
            raise SchemaError("frozen task registry/snapshot hash binding differs")
        registry_metadata, registry_records = _load_task_manifest(task_registry_path)
        if str(manifest.get("campaign_mode")) == "smoke":
            task_metadata, task_records = _load_task_manifest(task_candidates[0])
            if task_metadata != registry_metadata or task_records != registry_records:
                raise SchemaError("engineering smoke task manifest differs from registry")
        else:
            task_metadata, task_records = _validate_resolved_task_snapshot(
                task_candidates[0],
                registry_path=task_registry_path,
                registry_metadata=registry_metadata,
                registry_tasks=registry_records,
                environment_path=root / "frozen" / "environment.json",
            )
            validate_evaluator_requirements_resolved_snapshot_binding(
                live_deployment.evaluator_requirements,
                task_export=read_json(
                    root
                    / JOINT_DUPLICATE_EVIDENCE_RELATIVE_PATH
                    / "resolved_task_export.json"
                ),
                resolved_task_snapshot=read_json(task_candidates[0]),
            )
        _validate_task_boundary(
            task_records,
            campaign_config,
            protocol,
            metadata=task_metadata,
            pilot_only=pilot,
        )
        if str(manifest.get("campaign_mode")) != "smoke" and pilot:
            first_registered_index = registry_records[0].get("upstream_index")
            if (
                type(first_registered_index) is not int
                or deployment_preflight.binding.get(
                    "expected_live_reset_task_index"
                )
                != first_registered_index
            ):
                raise SchemaError(
                    "frozen WebArena deployment preflight did not reset the "
                    "first task in exact tracked-registry order"
                )
        if not pilot:
            validate_locked_final_pilot_exclusion(
                task_records,
                task_metadata=task_metadata,
                authority=pilot_exclusion_authority,
            )
        recovery_records, recovery_metadata = _load_recovery_scenarios(
            root / "frozen" / "benchmark" / "recovery_scenarios.json"
        )
        _validate_recovery_campaign_counts(
            task_records,
            recovery_records,
            recovery_metadata,
            campaign_config,
            matched_seeds=[int(seed) for seed in manifest["matched_seeds"]],
            repeat_count=len(manifest.get("repeat_ids", [])),
            pilot_only=pilot,
        )
        duplicate_audit_path = (
            root / "frozen" / "benchmark" / "duplicate_audit_manifest.json"
        )
        _validate_duplicate_audit_manifest(
            duplicate_audit_path,
            normal_tasks=task_records,
            recovery_scenarios=recovery_records,
            require_verified_normal=str(manifest.get("campaign_mode")) != "smoke",
        )
        frozen_task_content = _build_task_content_binding_manifest(
            task_records,
            task_manifest_sha256=sha256_file(task_candidates[0]),
            memory_by_seed={
                int(path.parent.name.removeprefix("seed_")): path
                for path in sorted((root / "memory").glob("seed_*/manifest.json"))
            },
            evidence_scope=(
                ENGINEERING_SMOKE_SCOPE
                if str(manifest.get("campaign_mode")) == "smoke"
                else EVALUATION_RUNNER_SCOPE
            ),
        )
        if read_json(root / "frozen" / "task_content_hashes.json") != frozen_task_content:
            raise SchemaError("frozen task/corpus binding manifest differs from campaign inputs")
        _validate_duplicate_audit_bindings(
            duplicate_audit_path,
            task_content_manifest=frozen_task_content,
            require_verified_normal=str(manifest.get("campaign_mode")) != "smoke",
        )
        joint_duplicate_evidence_binding = (
            _validate_registered_joint_duplicate_memory_bindings(
                {
                    int(path.parent.name.removeprefix("seed_")): path
                    for path in sorted(
                        (root / "memory").glob("seed_*/manifest.json")
                    )
                },
                duplicate_audit_path=duplicate_audit_path,
                required=str(manifest.get("campaign_mode")) != "smoke",
                repository_root=(
                    root
                    / JOINT_DUPLICATE_EVIDENCE_RELATIVE_PATH
                    / "tracked_repository"
                    if str(manifest.get("campaign_mode")) != "smoke"
                    else None
                ),
                assignment_package_root=(
                    root / JOINT_DUPLICATE_EVIDENCE_RELATIVE_PATH / "assignment"
                    if str(manifest.get("campaign_mode")) != "smoke"
                    else None
                ),
                preparation_package_root=(
                    root / JOINT_DUPLICATE_EVIDENCE_RELATIVE_PATH / "preparation"
                    if str(manifest.get("campaign_mode")) != "smoke"
                    else None
                ),
                resolved_task_export_path=(
                    root
                    / JOINT_DUPLICATE_EVIDENCE_RELATIVE_PATH
                    / "resolved_task_export.json"
                    if str(manifest.get("campaign_mode")) != "smoke"
                    else None
                ),
                approved_task_registry_path=(
                    root / "frozen" / "task_registry.json"
                    if str(manifest.get("campaign_mode")) != "smoke"
                    else None
                ),
                registered_recovery_scenarios_path=(
                    root
                    / JOINT_DUPLICATE_EVIDENCE_RELATIVE_PATH
                    / "registered_recovery_scenarios.json"
                    if str(manifest.get("campaign_mode")) != "smoke"
                    else None
                ),
                duplicate_audit_registration_path=(
                    root
                    / JOINT_DUPLICATE_EVIDENCE_RELATIVE_PATH
                    / "tracked_repository"
                    / JOINT_DUPLICATE_AUDIT_REGISTRATION_RELATIVE_PATH
                    if str(manifest.get("campaign_mode")) != "smoke"
                    else None
                ),
                provenance_manifest_path=(
                    root
                    / JOINT_DUPLICATE_EVIDENCE_RELATIVE_PATH
                    / "provenance_manifest.json"
                    if str(manifest.get("campaign_mode")) != "smoke"
                    else None
                ),
            )
        )
        if manifest.get("joint_duplicate_evidence_binding") != (
            joint_duplicate_evidence_binding
        ):
            raise SchemaError(
                "campaign joint duplicate-evidence binding differs from frozen inputs"
            )
        _validate_frozen_runner_attestation(
            root,
            manifest,
            protocol=protocol,
        )
        _validate_pc01_provider_installation_ledger(
            root,
            manifest,
            access_records=access_records,
        )
        configured_duplicate = str(campaign_config.get("duplicate_audit_manifest", ""))
        if str(task_metadata.get("duplicate_audit_manifest")) != configured_duplicate:
            raise SchemaError("frozen task/duplicate-audit references disagree")
        if str(recovery_metadata.get("duplicate_audit_manifest")) != configured_duplicate:
            raise SchemaError("frozen recovery/duplicate-audit references disagree")
        schedule_path = root / "schedule" / "schedule.jsonl"
        if sha256_file(schedule_path) != manifest["schedule_sha256"]:
            raise SchemaError("frozen schedule hash differs from campaign_manifest")
        schedule = read_jsonl(schedule_path)
        validate_schedule(schedule)
        if len(schedule) != int(manifest["scheduled_block_count"]):
            raise SchemaError("scheduled block count differs from campaign_manifest")
        partition_counts = {
            partition: sum(str(row.get("task_partition")) == partition for row in schedule)
            for partition in ("normal", "recovery_diagnostic")
        }
        if partition_counts["normal"] != int(manifest["normal_block_count"]):
            raise SchemaError("normal schedule block count differs from campaign_manifest")
        if partition_counts["recovery_diagnostic"] != int(manifest["recovery_block_count"]):
            raise SchemaError("recovery schedule block count differs from campaign_manifest")
        if set(str(row.get("task_partition")) for row in schedule) != {
            "normal",
            "recovery_diagnostic",
        }:
            raise SchemaError("schedule contains an unregistered task partition")
        if len(schedule) * len(SYSTEM_IDS) != int(manifest["planned_episode_count"]):
            raise SchemaError("planned episode count differs from schedule")
        _validate_schedule_rng_anchors(schedule, manifest)
        counts = defaultdict(int)
        included = 0
        excluded = 0
        for row in schedule:
            outcome = _validate_block(root, row, report, require_complete=require_complete)
            counts[outcome] += 1
            included += int(outcome == "INCLUDED")
            excluded += int(outcome == "EXCLUDED_INFRASTRUCTURE")
        report.counts.update(
            {
                "scheduled_blocks": len(schedule),
                "included_blocks": included,
                "infrastructure_excluded_blocks": excluded,
                **{f"block_status_{key.lower()}": value for key, value in counts.items()},
            }
        )
        if require_complete and included + excluded != len(schedule):
            report.error("not every scheduled block is included or preregistered infrastructure-excluded")
        processed = len(schedule) - counts["NOT_STARTED"]
        terminal_completion = _validate_campaign_completion_publication_status(
            root,
            manifest,
            scheduled_block_count=len(schedule),
            processed_block_count=processed,
            included_block_count=included,
            infrastructure_excluded_block_count=excluded,
        )
        _validate_episode_access_ledger(root, schedule, access_records)
        if enforce_live_readiness:
            _validate_live_readiness_boundary(
                root,
                manifest,
                access_records=access_records,
                require_complete=require_complete,
            )
        if str(manifest.get("campaign_mode")) != "smoke":
            _validate_campaign_provider_fairness(root, schedule)
        if require_aggregates:
            _validate_aggregate_package(root, report)
        report.publication_status = _publication_status(
            root,
            manifest,
            report,
            require_complete,
            campaign_complete=terminal_completion,
        )
    except (FileNotFoundError, OSError, ValueError, KeyError, SchemaError, Table2Error) as exc:
        report.error(str(exc))
        try:
            manifest = read_json(root / "campaign_manifest.json")
            failed_profile = classify_campaign_profile(
                manifest,
                context="campaign_manifest",
                require_campaign_mode=True,
            )
            frozen_campaign = load_yaml(root / "frozen" / "campaign.yaml")
            frozen_profile = classify_campaign_profile(
                frozen_campaign,
                context="frozen campaign configuration",
                default_campaign_mode="evaluation",
            )
            if (
                frozen_profile != failed_profile
                or frozen_campaign.get("campaign_mode", "evaluation")
                != manifest["campaign_mode"]
            ):
                raise SchemaError(
                    "frozen campaign profile differs from campaign_manifest"
                )
            report.publication_status = (
                DRAFT_PILOT_STATUS
                if failed_profile == CAMPAIGN_PROFILE_PILOT
                else "N/R"
            )
        except Exception:
            report.publication_status = "N/R"
    return report


def _validate_live_readiness_boundary(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    access_records: Sequence[Mapping[str, Any]],
    require_complete: bool,
) -> None:
    """Enforce the non-scored matched E0--E3 gate on real pilot evidence."""

    readiness_root = root / LIVE_COMPATIBILITY_RELATIVE_PATH.parent
    readiness_exists = readiness_root.exists() or readiness_root.is_symlink()
    profile = classify_campaign_profile(
        manifest,
        context="live-readiness campaign manifest",
        require_campaign_mode=True,
    )
    mode = manifest["campaign_mode"]
    pilot = profile == CAMPAIGN_PROFILE_PILOT
    if mode == "smoke" or not pilot:
        if readiness_exists:
            raise SchemaError(
                "smoke/final campaign cannot contain provisional runtime_readiness"
            )
        return

    if mode != "evaluation":
        raise SchemaError("PILOT_ONLY runtime readiness requires evaluation mode")
    readiness_artifacts = readiness_exists and any(
        True for _ in readiness_root.iterdir()
    )
    paired_started = any(
        path.is_file() or path.is_symlink()
        for path in (root / "paired_blocks").rglob("*")
    )
    access_started = any(
        str(record.get("event_type") or "") == "episode_task_load"
        for record in access_records
    )
    started = paired_started or access_started or (root / "completion.json").exists()
    if readiness_artifacts:
        validate_live_compatibility_receipt(root)
        return
    if require_complete or started:
        raise SchemaError(
            "PILOT_ONLY evaluation started without the required live-readiness receipt"
        )


def _validate_episode_access_ledger(
    root: Path,
    schedule: Sequence[Mapping[str, Any]],
    access_records: Sequence[Mapping[str, Any]],
) -> None:
    """Reconcile append-only task reads with every physical system launch."""

    expected: set[tuple[str, str, str, str, int, str]] = set()
    for row in schedule:
        base = _block_base(root, row)
        for attempt_id in range(int(row["max_block_attempts"])):
            manifest_path = base / f"rerun_{attempt_id}" / "block_manifest.json"
            if not manifest_path.is_file():
                continue
            block = read_json(manifest_path)
            for system_id in SYSTEM_IDS:
                status = block.get("systems", {}).get(system_id, {})
                if not isinstance(status, Mapping) or not strict_bool(
                    status.get("launched", False), context="access.launched"
                ):
                    continue
                expected.add(
                    (
                        str(row["block_id"]),
                        str(row["task_id"]),
                        str(row["task_partition"]),
                        system_id,
                        attempt_id,
                        str(status.get("episode_id", "")),
                    )
                )
    recorded: list[tuple[str, str, str, str, int, str]] = []
    for record in access_records:
        if str(record.get("event_type")) != "episode_task_load":
            continue
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise SchemaError("episode_task_load access record has no payload")
        require_keys(
            payload,
            (
                "block_id",
                "task_id",
                "task_partition",
                "system_id",
                "attempt_id",
                "episode_id",
                "locked_test_content",
            ),
            context="episode_task_load",
        )
        recorded.append(
            (
                str(payload["block_id"]),
                str(payload["task_id"]),
                str(payload["task_partition"]),
                str(payload["system_id"]),
                int(payload["attempt_id"]),
                str(payload["episode_id"]),
            )
        )
    if len(recorded) != len(set(recorded)):
        raise SchemaError("access ledger contains duplicate episode_task_load records")
    if set(recorded) != expected:
        missing = sorted(expected - set(recorded))
        extra = sorted(set(recorded) - expected)
        raise SchemaError(
            "episode task-load ledger does not match physical launches: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )


def _provider_campaign_state_sha256(root: Path) -> str:
    """Recompute the static campaign authority bound before/after the factory."""

    paths = (
        root / "campaign_manifest.json",
        root / "frozen/runner_attestation.json",
        root / "frozen/environment.json",
        root / "frozen/protocol.yaml",
    )
    return hashlib.sha256(
        "".join(f"{path.name}:{sha256_file(path)}\n" for path in paths).encode(
            "utf-8"
        )
    ).hexdigest()


def _validate_pc01_provider_installation_ledger(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    access_records: Sequence[Mapping[str, Any]],
) -> None:
    """Authenticate the durable provider receipt before any episode task read."""

    installation_rows = [
        (index, record)
        for index, record in enumerate(access_records)
        if str(record.get("event_type") or "") == "pc01_provider_installation"
    ]
    task_load_indices = [
        index
        for index, record in enumerate(access_records)
        if str(record.get("event_type") or "") == "episode_task_load"
    ]
    canonical_pc01 = (
        str(manifest.get("campaign_mode")) == "evaluation"
        and str(manifest.get("runner_entrypoint")) == PRODUCTION_RUNNER_ENTRYPOINT
    )
    if not canonical_pc01:
        if installation_rows:
            raise SchemaError(
                "non-PC-01 campaign contains a provider installation receipt"
            )
        return
    if task_load_indices and not installation_rows:
        raise SchemaError(
            "PC-01 episode task load precedes required provider installation receipt"
        )
    if installation_rows and task_load_indices:
        if installation_rows[0][0] >= task_load_indices[0]:
            raise SchemaError(
                "PC-01 provider installation receipt must precede episode_task_load"
            )
    if not installation_rows:
        # A newly frozen, otherwise-unstarted campaign is intentionally valid so
        # the production CLI can perform its pre-factory validation.  The first
        # task load makes the receipt mandatory.
        return

    attestation = read_json(root / "frozen/runner_attestation.json")
    binding = attestation.get(PC01_PROVIDER_BOOTSTRAP_BINDING_FIELD)
    if not isinstance(binding, Mapping):
        raise SchemaError("PC-01 provider bootstrap binding is absent")
    expected_contract = str(
        binding.get("expected_provider_public_contract_sha256") or ""
    )
    campaign_state = _provider_campaign_state_sha256(root)
    expected_factory = {
        "factory_entrypoint": str(binding.get("factory_entrypoint") or ""),
        "factory_module": str(binding.get("factory_module") or ""),
        "factory_qualname": str(binding.get("factory_qualname") or ""),
        "factory_source_relative_path": str(
            binding.get("source_relative_path") or ""
        ),
        "factory_source_sha256": str(binding.get("source_sha256") or ""),
    }
    for _, record in installation_rows:
        payload = record.get("payload")
        if not isinstance(payload, Mapping) or set(payload) != {
            "installation_receipt",
            "installation_receipt_sha256",
            "locked_test_content",
        }:
            raise SchemaError(
                "PC-01 provider installation ledger payload differs from schema"
            )
        if strict_bool(
            payload["locked_test_content"],
            context="provider_installation.locked_test_content",
        ):
            raise SchemaError(
                "PC-01 provider installation receipt cannot read locked content"
            )
        receipt = PC01ProviderInstallationReceipt.from_mapping(
            payload["installation_receipt"]
        )
        if payload["installation_receipt_sha256"] != receipt.receipt_sha256:
            raise SchemaError(
                "PC-01 provider installation receipt hash mismatch"
            )
        for field_name, expected in expected_factory.items():
            if getattr(receipt, field_name) != expected:
                raise SchemaError(
                    f"PC-01 provider installation {field_name} differs from handoff"
                )
        if (
            receipt.expected_provider_public_contract_sha256 != expected_contract
            or receipt.actual_provider_public_contract_sha256 != expected_contract
        ):
            raise SchemaError(
                "PC-01 actual provider contract differs from frozen expectation"
            )
        if (
            receipt.pre_factory_campaign_state_sha256 != campaign_state
            or receipt.post_factory_campaign_state_sha256 != campaign_state
        ):
            raise SchemaError(
                "PC-01 provider installation campaign authority hash mismatch"
            )


def require_pc01_provider_installation_ledger(
    root: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> PC01ProviderInstallationReceipt:
    """Require a durable valid installation event before runner construction."""

    campaign_root = Path(root).resolve()
    campaign_manifest = (
        read_json(campaign_root / "campaign_manifest.json")
        if manifest is None
        else dict(manifest)
    )
    records = read_jsonl(campaign_root / "access_ledger.jsonl")
    _validate_pc01_provider_installation_ledger(
        campaign_root,
        campaign_manifest,
        access_records=records,
    )
    matching = [
        record
        for record in records
        if str(record.get("event_type") or "") == "pc01_provider_installation"
    ]
    if not matching:
        raise SchemaError(
            "production runner requires a provider installation ledger receipt"
        )
    payload = matching[-1].get("payload")
    if not isinstance(payload, Mapping):  # pragma: no cover - checked above
        raise SchemaError("provider installation ledger payload is absent")
    return PC01ProviderInstallationReceipt.from_mapping(
        payload["installation_receipt"]
    )


def _validate_campaign_provider_fairness(
    root: Path, schedule: Sequence[Mapping[str, Any]]
) -> None:
    signatures: set[str] = set()
    for row in schedule:
        resolution_path = _block_base(root, row) / "resolution.json"
        if not resolution_path.is_file():
            continue
        resolution = read_json(resolution_path)
        if resolution.get("status") != "INCLUDED":
            continue
        attempt_id = int(resolution["selected_attempt_id"])
        for system_id in SYSTEM_IDS:
            actions_path = (
                _system_package(_block_base(root, row), attempt_id, system_id)
                / "runtime"
                / "actions.jsonl"
            )
            for record in read_jsonl(actions_path):
                if str(record.get("event_type", "")) != "normal_action":
                    continue
                parameters = _event_payload(record).get("parameters")
                if not isinstance(parameters, Mapping):
                    continue
                signatures.add(
                    sha256_json(
                        {
                            "provider_id": parameters.get("provider_id"),
                            "provider_version": parameters.get("provider_version"),
                            "prompt_sha256": parameters.get("prompt_sha256"),
                            "decoding_parameters": parameters.get("decoding_parameters"),
                        }
                    )
                )
    if len(signatures) > 1:
        raise SchemaError(
            "E0--E3 action parameters used more than one frozen provider signature"
        )


def load_selected_analysis_records(campaign_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Join unsealed runtime counts with sealed labels for offline analysis only."""

    root = Path(campaign_dir).resolve()
    validation = validate_campaign(root, require_complete=True, require_aggregates=False)
    if not validation.passed:
        raise Table2Error("cannot analyze an invalid campaign: " + "; ".join(validation.errors))
    return _load_selected_analysis_records_unchecked(root)


def _load_selected_analysis_records_unchecked(
    root: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load selected records after the caller has validated the campaign.

    Aggregate validation already runs inside :func:`validate_campaign`.  Keeping
    this narrow loader separate prevents a recursive second validation while
    still rebuilding every result from the selected runtime and sealed evidence.
    """

    schedule = read_jsonl(root / "schedule" / "schedule.jsonl")
    episodes: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    incidents: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    launches: list[dict[str, Any]] = []
    for row in schedule:
        base = _block_base(root, row)
        resolution = read_json(base / "resolution.json")
        for attempt in resolution.get("attempts", []):
            attempt_id = int(attempt["attempt_id"])
            for system_id, system_status in attempt["systems"].items():
                if strict_bool(system_status.get("launched", False), context="launched"):
                    launches.append(
                        {
                            "block_id": row["block_id"],
                            "task_id": row["task_id"],
                            "task_partition": row["task_partition"],
                            "system_id": system_id,
                            "attempt_id": attempt_id,
                            "launched": True,
                            "infrastructure_invalid": strict_bool(
                                system_status.get("infrastructure_invalid", False),
                                context="infrastructure_invalid",
                            ),
                        }
                    )
        if resolution["status"] != "INCLUDED":
            continue
        attempt_id = int(resolution["selected_attempt_id"])
        for system_id in SYSTEM_IDS:
            package = _system_package(base, attempt_id, system_id)
            runtime = package / "runtime"
            summary = read_json(runtime / "episode_summary.json")
            final = _read_final_verification(package / "sealed", str(summary["episode_id"]))
            evidence = final["evidence"]
            episode = {
                **summary,
                "block_id": str(row["block_id"]),
                "task_id": str(row["task_id"]),
                "task_partition": str(row["task_partition"]),
                "matched_model_seed": int(row["matched_model_seed"]),
                "repeat_id": int(row["repeat_id"]),
                "system_id": system_id,
                "runtime_terminal_reason": normalize_runtime_terminal_reason(
                    summary.get("runtime_terminal_reason"),
                    context=f"runtime summary {summary['episode_id']}",
                ),
                "evaluator_terminal_reason": str(evidence["terminal_reason"]),
                "task_success": evidence["task_success"],
                "loop_detected": evidence["loop_detected"],
                "environment_failure": evidence["environment_failure"],
                "verified_failure_event_count": int(
                    evidence["verified_failure_event_count"]
                ),
                "repeated_error_events": int(evidence["repeated_error_event_count"]),
            }
            episodes.append(episode)
            final_incidents = evidence.get("failure_incidents", [])
            final_recoveries = {
                str(item.get("recovery_attempt_id", item.get("attempt_id"))): item
                for item in evidence.get("recovery_verifications", [])
                if item.get("recovery_attempt_id", item.get("attempt_id")) is not None
            }
            for incident in final_incidents:
                incident_row = dict(incident)
                incidents.append(
                    {
                        **incident_row,
                        "system_id": system_id,
                        "episode_id": summary["episode_id"],
                    }
                )
            for event in read_jsonl(runtime / "recoveries.jsonl"):
                if str(event.get("event_type", "")) != "recovery_attempt":
                    continue
                attempt = _recovery_attempt_payload(event)
                attempt_id_value = str(attempt["attempt_id"])
                verification = final_recoveries.get(attempt_id_value, {})
                recoveries.append(
                    {
                        **dict(attempt),
                        "system_id": system_id,
                        "episode_id": summary["episode_id"],
                        "recovery_attempt_id": attempt_id_value,
                        "attempt_index": attempt.get(
                            "attempt_index", attempt.get("incident_attempt_index", 0)
                        ),
                        "verified_failure_present": verification.get(
                            "verified_failure_present",
                            verification.get("verified_failure", False),
                        ),
                        "successful": verification.get(
                            "successful", verification.get("recovery_success", False)
                        ),
                        "failure_incident_id": verification.get(
                            "failure_incident_id",
                            verification.get(
                                "incident_id",
                                attempt.get(
                                    "failure_incident_id", attempt.get("incident_id", "")
                                ),
                            ),
                        ),
                    }
                )
            if system_id == "E3":
                relevance = evidence.get("memory_relevance", {})
                memory_manifest_path = (
                    root
                    / "memory"
                    / f"seed_{int(row['matched_model_seed'])}"
                    / "manifest.json"
                )
                memory_index_size = (
                    int(read_json(memory_manifest_path)["item_count"])
                    if memory_manifest_path.is_file()
                    else None
                )
                if episode.get("memory_index_size") is None:
                    # Index cardinality is frozen store provenance, not a
                    # runner-estimated value.  Retain a measured runtime value
                    # when present, otherwise hydrate the episode companion
                    # evidence from the exact seed-bound manifest.
                    episode["memory_index_size"] = memory_index_size
                for event in read_jsonl(runtime / "memory_queries.jsonl"):
                    if str(event.get("event_type", "")) != "post_failure_query":
                        continue
                    query = _memory_query_payload(event)
                    query_id = str(query["query_id"])
                    sealed_relevance = _sealed_relevance_labels(relevance, query_id)
                    candidates = query.get("retrieved_ids", query.get("candidate_ids", []))
                    changed_strategy = strict_bool(
                        query.get("changed_strategy", False), context="changed_strategy"
                    )
                    changed_target = strict_bool(
                        query.get("changed_target_or_parameters", False),
                        context="changed_target_or_parameters",
                    )
                    admitted = strict_bool(query.get("admitted", False), context="admitted")
                    queries.append(
                        {
                            **dict(query),
                            "system_id": "E3",
                            "episode_id": summary["episode_id"],
                            "task_id": str(row["task_id"]),
                            "task_partition": str(row["task_partition"]),
                            "retrieved_ids": list(candidates),
                            "shadow_decision_hash": query.get(
                                "shadow_decision_hash", query.get("shadow_decision_sha256")
                            ),
                            "changed_decision": changed_strategy or changed_target,
                            "changed_strategy": changed_strategy,
                            "changed_target_or_parameters": changed_target,
                            "admitted": admitted,
                            "abstained": not admitted,
                            "retrieval_depth": int(
                                query.get("retrieval_depth", 3)
                            ),
                            "query_latency_ms": query.get(
                                "query_latency_ms", query.get("latency_ms", 0.0)
                            ),
                            "index_size": memory_index_size,
                            **sealed_relevance,
                        }
                    )
    return {
        "episodes": episodes,
        "recovery_attempts": recoveries,
        "failure_incidents": incidents,
        "memory_queries": queries,
        "schedule_attempts": launches,
    }


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _validate_block(
    root: Path,
    schedule_row: Mapping[str, Any],
    report: ValidationReport,
    *,
    require_complete: bool,
) -> str:
    base = _block_base(root, schedule_row)
    if not base.exists():
        if require_complete:
            raise SchemaError(f"missing scheduled paired block directory: {base}")
        report.warn(f"scheduled block has not started: {schedule_row['block_id']}")
        return "NOT_STARTED"
    attempts: list[dict[str, Any]] = []
    for attempt_id, rerun in discover_block_attempt_directories(
        base,
        maximum=int(schedule_row["max_block_attempts"]),
    ):
        manifest_path = rerun / "block_manifest.json"
        attempt = read_json(manifest_path)
        require_keys(attempt, ("attempt_id", "block_id", "systems"), context=str(manifest_path))
        if int(attempt["attempt_id"]) != attempt_id or str(attempt["block_id"]) != str(schedule_row["block_id"]):
            raise SchemaError(f"attempt manifest identity mismatch: {manifest_path}")
        if set(attempt["systems"]) != set(SYSTEM_IDS):
            raise SchemaError(f"attempt manifest lacks exactly E0--E3: {manifest_path}")
        for system_id in SYSTEM_IDS:
            status = attempt["systems"][system_id]
            if strict_bool(
                status.get("fatal_noninfrastructure_error", False),
                context="fatal_noninfrastructure_error",
            ):
                raise SchemaError(
                    f"fatal non-infrastructure runtime error is retained at "
                    f"{schedule_row['block_id']}/rerun_{attempt_id}/{system_id}"
                )
            if strict_bool(status.get("launched", False), context="launched"):
                if strict_bool(
                    status.get("infrastructure_invalid", False),
                    context="infrastructure_invalid",
                ):
                    package = _system_package(base, attempt_id, system_id)
                    _validate_infrastructure_invalid_package(
                        root,
                        package,
                        schedule_row=schedule_row,
                        attempt_id=attempt_id,
                        system_id=system_id,
                        status=status,
                    )
                    _validate_infrastructure_partial_runtime_package(
                        root,
                        package,
                        schedule_row=schedule_row,
                        attempt_id=attempt_id,
                        system_id=system_id,
                        status=status,
                    )
                if not strict_bool(status.get("completed", False), context="completed"):
                    report.warn(
                        f"retained interrupted infrastructure launch: "
                        f"{schedule_row['block_id']}/rerun_{attempt_id}/{system_id}"
                    )
                    continue
                _validate_system_package(
                    root,
                    _system_package(base, attempt_id, system_id),
                    schedule_row,
                    attempt_id,
                    system_id,
                    status,
                )
        attempts.append(attempt)
    calculated = resolve_block_attempts(schedule_row, attempts)
    resolution_path = base / "resolution.json"
    if resolution_path.exists():
        recorded = read_json(resolution_path)
        if recorded != calculated:
            raise SchemaError(f"block resolution differs from attempt evidence: {resolution_path}")
    elif require_complete:
        raise SchemaError(f"missing block resolution: {resolution_path}")
    else:
        report.warn(f"block resolution not yet written: {schedule_row['block_id']}")
    if require_complete and calculated["status"] == "RERUN_REQUIRED":
        raise SchemaError(f"whole block rerun remains required: {schedule_row['block_id']}")
    if calculated["status"] == "INTERRUPTED_UNAUTHORIZED":
        raise SchemaError(
            f"partial block lacks typed infrastructure evidence: {schedule_row['block_id']}"
        )
    if calculated["status"] == "INCLUDED":
        selected_attempt = calculated.get("selected_attempt_id")
        if type(selected_attempt) is not int or selected_attempt < 0:
            raise SchemaError(
                f"included block lacks an exact selected attempt: {schedule_row['block_id']}"
            )
        loop_rule = load_yaml(root / "frozen" / "protocol.yaml").get("loop_rule")
        if not isinstance(loop_rule, Mapping):
            raise SchemaError("frozen protocol lacks registered loop_rule")
        state_fingerprint_fields = loop_rule.get("state_fingerprint_fields")
        if not isinstance(state_fingerprint_fields, list):
            raise SchemaError(
                "frozen protocol loop_rule.state_fingerprint_fields must be a list"
            )
        campaign_manifest = read_json(root / "campaign_manifest.json")
        validate_included_block_causal_trace(
            base / f"rerun_{selected_attempt}",
            state_fingerprint_fields=state_fingerprint_fields,
            require_trained_first_pre_action=(
                campaign_manifest.get("campaign_mode") != "smoke"
            ),
        )
        if (
            campaign_manifest.get("runner_entrypoint")
            == "web_agent.eval.table2.production_runner:create_runner"
        ):
            validate_included_block_state_isolation(
                base / f"rerun_{selected_attempt}",
                schedule_row=schedule_row,
                attested_source_sha256=frozen_runner_source_hashes(root),
            )
    return calculated["status"]


def _validate_infrastructure_invalid_package(
    campaign_root: Path,
    package: Path,
    *,
    schedule_row: Mapping[str, Any],
    attempt_id: int,
    system_id: str,
    status: Mapping[str, Any],
) -> None:
    evidence_path = package / "sealed" / "infrastructure_invalid.json"
    if not evidence_path.is_file():
        raise SchemaError("infrastructure-invalid launch lacks sealed adapter evidence")
    expected_hash = status.get("infrastructure_evidence_sha256")
    if not _is_sha256(expected_hash) or sha256_file(evidence_path) != expected_hash:
        raise SchemaError("infrastructure-invalid adapter evidence hash mismatch")
    value = read_json(evidence_path)
    campaign = read_json(campaign_root / "campaign_manifest.json")
    episode_id = str(status.get("episode_id") or "")
    if not episode_id:
        raise SchemaError("infrastructure-invalid launch lacks episode identity")
    validate_infrastructure_evidence_record(
        value,
        campaign_mode=str(campaign.get("campaign_mode", "evaluation")),
        block_id=str(schedule_row["block_id"]),
        attempt_id=attempt_id,
        system_id=system_id,
        episode_id=episode_id,
    )
    if status.get("infrastructure_reason") != value.get("reason_code"):
        raise SchemaError("block status infrastructure reason differs from sealed evidence")
    if str(campaign.get("campaign_mode", "evaluation")) != "smoke":
        environment = read_json(campaign_root / "frozen" / "environment.json")
        if (
            value.get("adapter_id") != environment.get("environment_adapter_id")
            or value.get("adapter_version")
            != environment.get("environment_adapter_version")
        ):
            raise SchemaError(
                "infrastructure evidence adapter identity differs from environment"
            )
        adapter_evidence = value.get("adapter_evidence")
        if not isinstance(adapter_evidence, Mapping):
            raise SchemaError("infrastructure adapter evidence is malformed")
        operation = str(value.get("operation") or "")
        if operation == "post_reset_terminal_check":
            expected_diagnostic = {
                "episode_id": episode_id,
                "task_id": str(schedule_row["task_id"]),
                "terminal_reason": "RESET_ALREADY_SUCCESS",
            }
            if (
                value.get("reason_code") != "ENVIRONMENT_RESET_FAILED"
                or adapter_evidence.get("failure_class")
                != "TASK_ALREADY_COMPLETE_AFTER_RESET"
                or any(
                    adapter_evidence.get(field) != expected
                    for field, expected in expected_diagnostic.items()
                )
                or adapter_evidence.get("diagnostic_sha256")
                != sha256_json(expected_diagnostic)
            ):
                raise SchemaError(
                    "already-complete reset infrastructure evidence is inconsistent"
                )
        else:
            classifier = environment.get("infrastructure_classifier")
            if not isinstance(classifier, Mapping):
                raise SchemaError(
                    "frozen environment lacks its infrastructure classifier"
                )
            expected_classifier = {
                "classifier_id": classifier.get("classifier_id"),
                "classifier_version": classifier.get("classifier_version"),
                "classifier_rules_sha256": classifier.get("rules_sha256"),
                "benchmark_version": environment.get("benchmark_version"),
                "operation": operation,
                "episode_id": episode_id,
                "task_id": str(schedule_row["task_id"]),
            }
            failure_class = str(adapter_evidence.get("failure_class") or "")
            rules = classifier.get("rules")
            if not isinstance(rules, list):
                raise SchemaError("frozen infrastructure classifier rules are malformed")
            compact_fields = {
                "adapter_event_id",
                "failure_class",
                "diagnostic_sha256",
                "retryable",
            }
            if set(adapter_evidence) == compact_fields:
                event_prefix = f"{episode_id}:infrastructure:"
                event_id = adapter_evidence.get("adapter_event_id")
                event_index = (
                    event_id[len(event_prefix) :]
                    if isinstance(event_id, str) and event_id.startswith(event_prefix)
                    else ""
                )
                if (
                    adapter_evidence.get("retryable") is not True
                    or not event_index.isdigit()
                    or int(event_index) <= 0
                    or str(int(event_index)) != event_index
                ):
                    raise SchemaError(
                        "compact infrastructure evidence is not episode-bound"
                    )
                matches = [
                    rule
                    for rule in rules
                    if isinstance(rule, Mapping)
                    and rule.get("operation") == operation
                    and rule.get("reason_code") == value.get("reason_code")
                    and rule.get("failure_class") == failure_class
                ]
                if len(matches) != 1:
                    raise SchemaError(
                        "compact infrastructure evidence lacks one unique frozen rule"
                    )
                exception_type = str(matches[0].get("exception_type") or "")
                if not exception_type:
                    raise SchemaError(
                        "compact infrastructure evidence rule lacks exception type"
                    )
            else:
                for field, expected in expected_classifier.items():
                    if adapter_evidence.get(field) != expected:
                        raise SchemaError(
                            f"infrastructure classifier evidence changed {field}"
                        )
                exception_type = str(adapter_evidence.get("exception_type") or "")
                matches = [
                    rule
                    for rule in rules
                    if isinstance(rule, Mapping)
                    and rule.get("operation") == operation
                    and rule.get("exception_type") == exception_type
                ]
                if len(matches) != 1:
                    raise SchemaError(
                        "infrastructure exception/operation is absent from frozen rules"
                    )
                rule = matches[0]
                if (
                    rule.get("reason_code") != value.get("reason_code")
                    or rule.get("failure_class") != failure_class
                ):
                    raise SchemaError(
                        "infrastructure evidence differs from its frozen exact-match rule"
                    )
            diagnostic = {
                **expected_classifier,
                "failure_class": failure_class,
                "exception_type": exception_type,
            }
            if adapter_evidence.get("diagnostic_sha256") != sha256_json(diagnostic):
                raise SchemaError(
                    "infrastructure classifier diagnostic commitment differs"
                )


def _validate_infrastructure_partial_runtime_package(
    campaign_root: Path,
    package: Path,
    *,
    schedule_row: Mapping[str, Any],
    attempt_id: int,
    system_id: str,
    status: Mapping[str, Any],
) -> None:
    """Hash-close and reconcile all work consumed before an infra fault."""

    runtime = package / "runtime"
    if not runtime.is_dir():
        raise SchemaError("infrastructure-invalid launch lacks its runtime package")
    for filename in RUNTIME_FILES:
        if not (runtime / filename).is_file():
            raise SchemaError(
                f"infrastructure partial runtime is missing {filename}: {runtime}"
            )
    memory_path = runtime / "memory_queries.jsonl"
    if system_id == "E3" and not memory_path.is_file():
        raise SchemaError("infrastructure-interrupted E3 lacks memory query stream")
    if system_id != "E3" and memory_path.exists() and read_jsonl(memory_path):
        raise SchemaError(
            "infrastructure-interrupted non-E3 system emitted memory queries"
        )
    _validate_runtime_artifact_hashes(runtime, system_id=system_id)

    episode_manifest = read_json(runtime / "episode_manifest.json")
    expected_manifest = {
        "block_id": str(schedule_row["block_id"]),
        "attempt_id": int(attempt_id),
        "system_id": system_id,
        "stage_seeds": schedule_row["stage_seeds"],
    }
    for field, expected in expected_manifest.items():
        if episode_manifest.get(field) != expected:
            raise SchemaError(
                f"infrastructure partial episode manifest changed {field}"
            )
    episode_id = str(episode_manifest.get("episode_id") or "")
    if not episode_id or episode_id != str(status.get("episode_id") or ""):
        raise SchemaError("infrastructure partial episode identity mismatch")

    summary = read_json(runtime / "episode_summary.json")
    assert_no_verifier_evidence(
        summary,
        context=str(runtime / "episode_summary.json"),
    )
    expected_summary = {
        "episode_id": episode_id,
        "system_id": system_id,
        "task_id": str(schedule_row["task_id"]),
        "repeat_id": int(schedule_row["repeat_id"]),
        "model_seed": int(schedule_row["matched_model_seed"]),
        "completed": False,
        "infrastructure_invalid": True,
        "valid_for_primary": False,
        "infrastructure_reason": status.get("infrastructure_reason"),
        "infrastructure_evidence_sha256": status.get(
            "infrastructure_evidence_sha256"
        ),
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise SchemaError(
                f"infrastructure partial summary changed {field}"
            )
    for field in (
        "step_count",
        "executed_action_count",
        "rejected_action_count",
        "recovery_action_count",
        "recovery_attempt_count",
        "memory_query_count",
    ):
        if type(summary.get(field)) is not int or int(summary[field]) < 0:
            raise SchemaError(
                f"infrastructure partial summary {field} must be nonnegative integer"
            )
    if summary["step_count"] != (
        summary["executed_action_count"] + summary["rejected_action_count"]
    ):
        raise SchemaError("infrastructure partial step accounting does not reconcile")
    wall_clock = summary.get("task_wall_clock_seconds")
    if (
        isinstance(wall_clock, bool)
        or not isinstance(wall_clock, (int, float))
        or not math.isfinite(float(wall_clock))
        or float(wall_clock) < 0
    ):
        raise SchemaError("infrastructure partial wall clock is invalid")
    partial_sha = summary.get("partial_episode_summary_sha256")
    if partial_sha is not None and not _is_sha256(partial_sha):
        raise SchemaError("infrastructure partial-summary receipt hash is invalid")

    streams = (
        "actions.jsonl",
        "transitions.jsonl",
        "recoveries.jsonl",
        "environment_events.jsonl",
        "terminal_signals.jsonl",
    )
    records_by_name: dict[str, list[dict[str, Any]]] = {}
    event_identity = {
        "system_id": system_id,
        "task_id": str(schedule_row["task_id"]),
        "repeat_id": int(schedule_row["repeat_id"]),
        "matched_seed": int(schedule_row["matched_model_seed"]),
    }
    for filename in streams:
        records = read_jsonl(runtime / filename)
        records_by_name[filename] = records
        for index, record in enumerate(records, start=1):
            assert_no_verifier_evidence(
                record,
                context=f"{runtime / filename}:{index}",
            )
            for field, expected in event_identity.items():
                if record.get(field) != expected:
                    raise SchemaError(
                        f"infrastructure partial event changed {field}: "
                        f"{runtime / filename}:{index}"
                    )
        _verify_runtime_event_stream(
            runtime / filename,
            filename.removesuffix(".jsonl"),
            episode_id,
        )
    if memory_path.exists():
        memory_records = read_jsonl(memory_path)
        for index, record in enumerate(memory_records, start=1):
            assert_no_verifier_evidence(
                record,
                context=f"{memory_path}:{index}",
            )
            for field, expected in event_identity.items():
                if record.get(field) != expected:
                    raise SchemaError(
                        f"infrastructure partial memory event changed {field}"
                    )
        _verify_runtime_event_stream(memory_path, "memory_queries", episode_id)
    else:
        memory_records = []

    actions = records_by_name["actions.jsonl"]
    statuses: list[str] = []
    for expected_step, record in enumerate(actions, start=1):
        event_type = str(record.get("event_type") or "")
        if event_type not in {
            "normal_action",
            "recovery_action",
            "pre_action_parse_rejection",
        }:
            raise SchemaError(
                "infrastructure partial action stream has an unknown event type"
            )
        payload = _event_payload(record)
        execution = payload.get("execution")
        if not isinstance(execution, Mapping):
            raise SchemaError(
                "infrastructure partial action lacks execution evidence"
            )
        if execution.get("executor_step") != expected_step:
            raise SchemaError(
                "infrastructure partial executor steps are not exact and sequential"
            )
        execution_status = str(execution.get("status") or "").upper()
        if execution_status not in {"EXECUTED", "REJECTED", "ERROR"}:
            raise SchemaError("infrastructure partial execution status is invalid")
        statuses.append(execution_status)
    recovery_actions = sum(
        str(record.get("event_type") or "") == "recovery_action"
        for record in actions
    )
    recovery_attempts = sum(
        str(record.get("event_type") or "") == "recovery_attempt"
        for record in records_by_name["recoveries.jsonl"]
    )
    memory_queries = sum(
        str(record.get("event_type") or "") == "post_failure_query"
        for record in memory_records
    )
    exact_counts = {
        "step_count": len(actions),
        "executed_action_count": sum(value == "EXECUTED" for value in statuses),
        "rejected_action_count": sum(value != "EXECUTED" for value in statuses),
        "recovery_action_count": recovery_actions,
        "recovery_attempt_count": recovery_attempts,
        "memory_query_count": memory_queries,
    }
    for field, expected in exact_counts.items():
        if summary[field] != expected:
            raise SchemaError(
                f"infrastructure partial summary {field} differs from frozen logs"
            )
    protocol = load_yaml(campaign_root / "frozen" / "protocol.yaml")
    _validate_frozen_runtime_budget_evidence(
        summary=summary,
        actions=actions,
        recovery_events=records_by_name["recoveries.jsonl"],
        environment_events=records_by_name["environment_events.jsonl"],
        protocol=protocol,
        evaluation_mode=False,
    )
    terminal_records = records_by_name["terminal_signals.jsonl"]
    partial_campaign = read_json(campaign_root / "campaign_manifest.json")
    _validate_evaluator_backend_guards(
        environment_events=records_by_name["environment_events.jsonl"],
        terminal_signals=terminal_records,
        required=(
            bool(terminal_records)
            and partial_campaign.get("runner_entrypoint")
            == "web_agent.eval.table2.production_runner:create_runner"
            and str(schedule_row.get("task_partition")) == "normal"
        ),
    )
    sealed_episode = package / "sealed" / _path_id(episode_id) / "verifier_events.jsonl"
    if terminal_records or sealed_episode.exists():
        _validate_terminal_signal_receipts(
            package / "sealed",
            episode_id,
            terminal_records,
        )


def _validate_system_package(
    campaign_root: Path,
    package: Path,
    schedule_row: Mapping[str, Any],
    attempt_id: int,
    system_id: str,
    status: Mapping[str, Any],
) -> None:
    runtime = package / "runtime"
    sealed = package / "sealed"
    if not runtime.is_dir() or not sealed.is_dir():
        raise SchemaError(f"launched package lacks runtime/sealed separation: {package}")
    for filename in RUNTIME_FILES:
        if not (runtime / filename).is_file():
            raise SchemaError(f"runtime package is missing {filename}: {runtime}")
    memory_path = runtime / "memory_queries.jsonl"
    if system_id == "E3" and not memory_path.is_file():
        raise SchemaError(f"E3 runtime package lacks memory_queries.jsonl: {runtime}")
    if system_id != "E3" and memory_path.exists() and read_jsonl(memory_path):
        raise SchemaError(f"{system_id} emitted memory queries: {memory_path}")
    _validate_runtime_artifact_hashes(runtime, system_id=system_id)

    episode_manifest = read_json(runtime / "episode_manifest.json")
    require_keys(
        episode_manifest,
        (
            "schema_version",
            "episode_id",
            "block_id",
            "attempt_id",
            "system_id",
            "stage_seeds",
            "rng_provenance_sha256",
            "campaign_manifest_sha256",
            "runner_identity_scope",
            "runner_entrypoint",
            "runner_attestation_sha256",
            "frozen_artifact_hashes_sha256",
            "protocol_sha256",
            "environment_manifest_relative_path",
            "environment_manifest_sha256",
            "system_overlay_sha256",
            "parameter_provider_prompt_sha256",
            "resolved_features",
            "policy_source",
            "model_manifest_relative_path",
            "model_manifest_sha256",
            "memory_store_relative_path",
            "memory_manifest_sha256",
            "duplicate_audit_relative_path",
            "duplicate_audit_manifest_sha256",
        ),
        context=str(runtime / "episode_manifest.json"),
    )
    expected = {
        "block_id": str(schedule_row["block_id"]),
        "attempt_id": int(attempt_id),
        "system_id": system_id,
    }
    for key, value in expected.items():
        if episode_manifest[key] != value:
            raise SchemaError(f"episode manifest {key} mismatch in {runtime}")
    if episode_manifest["stage_seeds"] != schedule_row["stage_seeds"]:
        raise SchemaError(f"episode changed registered stage seeds: {runtime}")
    frozen_system = load_yaml(
        campaign_root / "frozen" / "systems" / f"{system_id.lower()}.yaml"
    )
    provenance_expected = {
        "rng_provenance_sha256": sha256_file(runtime / "rng_provenance.json"),
        "campaign_manifest_sha256": sha256_file(campaign_root / "campaign_manifest.json"),
        "runner_identity_scope": read_json(campaign_root / "campaign_manifest.json")[
            "runner_identity_scope"
        ],
        "runner_entrypoint": read_json(campaign_root / "campaign_manifest.json").get(
            "runner_entrypoint"
        ),
        "runner_attestation_sha256": read_json(
            campaign_root / "campaign_manifest.json"
        ).get("runner_attestation_sha256"),
        "frozen_artifact_hashes_sha256": sha256_file(campaign_root / "artifact_hashes.json"),
        "protocol_sha256": sha256_file(campaign_root / "frozen" / "protocol.yaml"),
        "environment_manifest_relative_path": "frozen/environment.json",
        "environment_manifest_sha256": sha256_file(
            campaign_root / "frozen" / "environment.json"
        ),
        "system_overlay_sha256": sha256_file(
            campaign_root / "frozen" / "systems" / f"{system_id.lower()}.yaml"
        ),
        "parameter_provider_prompt_sha256": sha256_file(
            campaign_root / "frozen" / "prompts" / "parameter_provider_v1.txt"
        ),
        "resolved_features": frozen_system["features"],
        "policy_source": frozen_system["policy_source"],
    }
    for key, expected_value in provenance_expected.items():
        if episode_manifest[key] != expected_value:
            raise SchemaError(f"episode provenance {key} mismatch: {runtime}")
    seed = int(schedule_row["matched_model_seed"])
    model_path = campaign_root / "frozen" / "models" / f"seed_{seed}.json"
    expected_model_hash = sha256_file(model_path) if model_path.exists() else None
    expected_model_relative = (
        str(model_path.relative_to(campaign_root)) if model_path.exists() else None
    )
    if episode_manifest["model_manifest_relative_path"] != expected_model_relative:
        raise SchemaError(f"episode model manifest path mismatch: {runtime}")
    if episode_manifest["model_manifest_sha256"] != expected_model_hash:
        raise SchemaError(f"episode model manifest hash mismatch: {runtime}")
    memory_manifest = campaign_root / "memory" / f"seed_{seed}" / "manifest.json"
    expected_memory_hash = (
        sha256_file(memory_manifest)
        if system_id == "E3" and memory_manifest.exists()
        else None
    )
    expected_memory_relative = (
        str(memory_manifest.parent.relative_to(campaign_root))
        if system_id == "E3" and memory_manifest.exists()
        else None
    )
    if episode_manifest["memory_store_relative_path"] != expected_memory_relative:
        raise SchemaError(f"episode memory store path mismatch: {runtime}")
    if episode_manifest["memory_manifest_sha256"] != expected_memory_hash:
        raise SchemaError(f"episode memory manifest hash mismatch: {runtime}")
    duplicate_audit_path = (
        campaign_root / "frozen" / "benchmark" / "duplicate_audit_manifest.json"
    )
    expected_duplicate_path = (
        str(duplicate_audit_path.relative_to(campaign_root)) if system_id == "E3" else None
    )
    expected_duplicate_hash = sha256_file(duplicate_audit_path) if system_id == "E3" else None
    if (
        episode_manifest["duplicate_audit_relative_path"] != expected_duplicate_path
        or episode_manifest["duplicate_audit_manifest_sha256"] != expected_duplicate_hash
    ):
        raise SchemaError(f"episode duplicate-audit provenance mismatch: {runtime}")
    _validate_rng_provenance(runtime, schedule_row, campaign_root)

    summary = read_json(runtime / "episode_summary.json")
    require_keys(
        summary,
        (
            "episode_id",
            "completed",
            "infrastructure_invalid",
            "step_count",
            "executed_action_count",
            "rejected_action_count",
            "recovery_action_count",
            "recovery_attempt_count",
            "runtime_terminal_reason",
        ),
        context=str(runtime / "episode_summary.json"),
    )
    if str(summary["episode_id"]) != str(episode_manifest["episode_id"]):
        raise SchemaError(f"runtime summary episode ID mismatch: {runtime}")
    assert_no_verifier_evidence(summary, context=str(runtime / "episode_summary.json"))
    if strict_bool(summary["completed"], context="completed") != strict_bool(status.get("completed", False), context="status.completed"):
        raise SchemaError(f"attempt/runtime completion status mismatch: {runtime}")
    if strict_bool(summary["infrastructure_invalid"], context="infrastructure_invalid") != strict_bool(status.get("infrastructure_invalid", False), context="status.infrastructure_invalid"):
        raise SchemaError(f"attempt/runtime infrastructure status mismatch: {runtime}")

    step_count = int(summary["step_count"])
    executed_count = int(summary["executed_action_count"])
    rejected_count = int(summary["rejected_action_count"])
    recovery_action_count = int(summary["recovery_action_count"])
    for key in (
        "step_count",
        "executed_action_count",
        "rejected_action_count",
        "recovery_action_count",
        "recovery_attempt_count",
    ):
        if type(summary[key]) is not int or int(summary[key]) < 0:
            raise SchemaError(f"runtime summary {key} must be an exact nonnegative integer")
    evaluation_mode = read_json(campaign_root / "campaign_manifest.json").get(
        "campaign_mode"
    ) != "smoke"
    model_calls = summary.get("model_call_count")
    wall_clock = summary.get("task_wall_clock_seconds")
    if model_calls is not None and (type(model_calls) is not int or model_calls < 0):
        raise SchemaError("runtime model_call_count must be an exact nonnegative integer")
    if wall_clock is not None:
        if isinstance(wall_clock, bool):
            raise SchemaError("runtime wall-clock cannot be boolean")
        wall_clock = float(wall_clock)
        if not math.isfinite(wall_clock) or wall_clock < 0:
            raise SchemaError("runtime wall-clock must be finite and nonnegative")
    if evaluation_mode and (model_calls is None or wall_clock is None):
        raise SchemaError("evaluation episode lacks measured model-call/wall-clock evidence")
    for key in (
        "input_token_count",
        "output_token_count",
        "model_parameter_count",
        "trainable_parameter_count",
        "memory_index_size",
    ):
        value = summary.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise SchemaError(f"runtime summary {key} must be an exact nonnegative integer")
    for key in (
        "peak_gpu_memory_mb",
        "peak_system_memory_mb",
        "training_gpu_hours",
    ):
        value = summary.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            raise SchemaError(f"runtime summary {key} cannot be boolean")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise SchemaError(f"runtime summary {key} must be finite and nonnegative")
    for key in (
        "decision_latency_ms",
        "provider_latency_ms",
        "recovery_latency_ms",
        "retrieval_latency_ms",
    ):
        if key not in summary or isinstance(summary[key], bool):
            raise SchemaError(f"runtime summary lacks measured {key}")
        value = float(summary[key])
        if not math.isfinite(value) or value < 0:
            raise SchemaError(f"runtime summary {key} must be finite and nonnegative")
    if step_count != executed_count + rejected_count:
        raise SchemaError(
            f"step_count must equal executed plus rejected requests: {runtime}"
        )
    if recovery_action_count > step_count:
        raise SchemaError(f"recovery actions must be a subset of steps: {runtime}")

    actions: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    recovery_events: list[dict[str, Any]] = []
    environment_events: list[dict[str, Any]] = []
    terminal_signals: list[dict[str, Any]] = []
    for filename in (
        "actions.jsonl",
        "transitions.jsonl",
        "recoveries.jsonl",
        "environment_events.jsonl",
        "terminal_signals.jsonl",
    ):
        records = read_jsonl(runtime / filename)
        if filename == "actions.jsonl":
            actions = records
        elif filename == "transitions.jsonl":
            transitions = records
        elif filename == "recoveries.jsonl":
            recovery_events = records
        elif filename == "environment_events.jsonl":
            environment_events = records
        elif filename == "terminal_signals.jsonl":
            terminal_signals = records
        for index, record in enumerate(records):
            assert_no_verifier_evidence(record, context=f"{runtime / filename}:{index + 1}")
            event_identity = {
                "system_id": system_id,
                "task_id": str(schedule_row["task_id"]),
                "repeat_id": int(schedule_row["repeat_id"]),
                "matched_seed": int(schedule_row["matched_model_seed"]),
            }
            for key, expected_value in event_identity.items():
                if record.get(key) != expected_value:
                    raise SchemaError(
                        f"runtime event {key} differs from package: "
                        f"{runtime / filename}:{index + 1}"
                    )
            payload = _event_payload(record)
            if "episode_id" in payload and str(payload["episode_id"]) != str(summary["episode_id"]):
                raise SchemaError(f"event has wrong episode_id: {runtime / filename}:{index + 1}")
        _verify_runtime_event_stream(
            runtime / filename,
            filename.removesuffix(".jsonl"),
            str(summary["episode_id"]),
        )
    if (
        read_json(campaign_root / "campaign_manifest.json").get("runner_entrypoint")
        == "web_agent.eval.table2.production_runner:create_runner"
    ):
        reset_stage_seed = schedule_row.get("stage_seeds", {}).get("reset")
        validate_episode_state_reset(
            runtime,
            expected_episode_id=str(summary["episode_id"]),
            expected_system_id=system_id,
            expected_reset_stage_seed=reset_stage_seed,
            attested_source_sha256=frozen_runner_source_hashes(campaign_root),
        )
    if (
        read_json(campaign_root / "campaign_manifest.json").get("runner_entrypoint")
        == "web_agent.eval.table2.production_runner:create_runner"
        and str(schedule_row.get("task_partition")) == "normal"
    ):
        benchmark_version, start_state_id = _normal_task_reset_identity(
            campaign_root,
            task_id=str(schedule_row["task_id"]),
        )
        validate_webarena_reset_state_receipt(
            runtime,
            expected_episode_id=str(summary["episode_id"]),
            expected_system_id=system_id,
            expected_task_id=str(schedule_row["task_id"]),
            expected_benchmark_version=benchmark_version,
            expected_start_state_id=start_state_id,
            expected_reset_stage_seed=reset_stage_seed,
            attested_source_sha256=frozen_runner_source_hashes(campaign_root),
        )
        validate_runtime_screenshot_artifacts(runtime)
    if evaluation_mode:
        _validate_duplicate_audit_runtime_binding(
            campaign_root,
            schedule_row=schedule_row,
            system_id=system_id,
            environment_records=environment_events,
        )
        _validate_memory_store_runtime_binding(
            campaign_root,
            schedule_row=schedule_row,
            system_id=system_id,
            environment_records=environment_events,
        )
    budget_actions = [
        row
        for row in actions
        if strict_bool(
            _event_payload(row).get("budget_consumed", True),
            context="action.budget_consumed",
        )
    ]
    if len(budget_actions) != len(actions):
        raise SchemaError(
            "action stream contains an uncharged executor request"
        )
    if len(budget_actions) != step_count:
        raise SchemaError(
            f"budget-consuming action log count differs from step_count: {runtime}"
        )
    statuses: list[str] = []
    parse_rejection_count = 0
    for record in budget_actions:
        action_payload = _event_payload(record)
        event_type = str(record.get("event_type", ""))
        if event_type not in {
            "normal_action",
            "recovery_action",
            "pre_action_parse_rejection",
        }:
            raise SchemaError(f"action stream has unknown event type: {event_type}")
        execution = action_payload.get("execution")
        if not isinstance(execution, Mapping):
            raise SchemaError(f"action event lacks execution result: {runtime}")
        try:
            execution_record = ExecutionResult.from_dict(execution)
        except (TypeError, ValueError) as exc:
            raise SchemaError(
                f"action event execution contract is invalid: {runtime}"
            ) from exc
        if not isinstance(execution_record, ExecutionResult):  # pragma: no cover
            raise SchemaError("action event reconstructed the wrong execution contract")
        if evaluation_mode and execution_record.evidence is None:
            raise SchemaError("evaluation action lacks versioned execution evidence")
        if execution_record.evidence is not None:
            evidence = execution_record.evidence
            if evidence.action_id != execution_record.action_id:
                raise SchemaError("execution evidence belongs to a different request")
            if evidence.controller_command is not None:
                action = action_payload.get("action")
                if not isinstance(action, Mapping) or (
                    evidence.controller_command.action_id != action.get("action_id")
                ):
                    raise SchemaError(
                        "controller command commitment is not bound to the logged action"
                    )
        status_value = str(execution.get("status", "")).upper()
        if status_value not in {"EXECUTED", "REJECTED", "ERROR"}:
            raise SchemaError(f"action event has invalid execution status: {runtime}")
        statuses.append(status_value)
        if event_type == "pre_action_parse_rejection":
            parse_rejection_count += 1
            _validate_pre_action_parse_rejection(
                action_payload,
                system_id=system_id,
            )
        if evaluation_mode and event_type == "normal_action":
            _validate_normal_action_parameters(
                action_payload,
                execution_status=status_value,
                prompt_path=(
                    campaign_root
                    / "frozen"
                    / "prompts"
                    / "parameter_provider_v1.txt"
                ),
            )
    if parse_rejection_count > 1:
        raise SchemaError("an episode cannot contain repeated pre-action parse rejections")
    logged_executed = sum(value == "EXECUTED" for value in statuses)
    if logged_executed != executed_count or len(statuses) - logged_executed != rejected_count:
        raise SchemaError(f"action executed/rejected breakdown differs from summary: {runtime}")
    logged_recovery_actions = sum(
        str(record.get("event_type", "")) == "recovery_action" for record in actions
    )
    if logged_recovery_actions != recovery_action_count:
        raise SchemaError(f"recovery-action subset differs from summary: {runtime}")
    recovery_attempt_events = [
        record
        for record in recovery_events
        if str(record.get("event_type", "")) == "recovery_attempt"
    ]
    if len(recovery_attempt_events) != int(summary["recovery_attempt_count"]):
        raise SchemaError(f"recovery-attempt log count differs from summary: {runtime}")
    frozen_protocol = load_yaml(campaign_root / "frozen" / "protocol.yaml")
    _validate_frozen_runtime_budget_evidence(
        summary=summary,
        actions=actions,
        recovery_events=recovery_events,
        environment_events=environment_events,
        protocol=frozen_protocol,
        evaluation_mode=evaluation_mode,
    )
    campaign_manifest = read_json(campaign_root / "campaign_manifest.json")
    production_runtime = campaign_manifest.get("runner_entrypoint") == (
        "web_agent.eval.table2.production_runner:create_runner"
    )
    _validate_evaluator_backend_guards(
        environment_events=environment_events,
        terminal_signals=terminal_signals,
        required=(
            production_runtime
            and str(schedule_row.get("task_partition")) == "normal"
        ),
    )
    if production_runtime and str(schedule_row.get("task_partition")) == "normal":
        _validate_manual_rescue_guard_events(
            environment_events=environment_events,
            actions=actions,
            terminal_signals=terminal_signals,
            environment=read_json(campaign_root / "frozen" / "environment.json"),
            campaign_manifest=campaign_manifest,
            episode_id=str(summary["episode_id"]),
            task_id=str(schedule_row["task_id"]),
        )
    _validate_episode_contract_validation_receipt(
        runtime=runtime,
        summary=summary,
        actions=actions,
        transitions=transitions,
        recovery_events=recovery_events,
        environment_events=environment_events,
        required=production_runtime and evaluation_mode,
    )
    if system_id in {"E0", "E1"} and (
        recovery_events
        or recovery_action_count
        or int(summary["recovery_attempt_count"])
        or transitions
    ):
        raise SchemaError(f"{system_id} violated its disabled diagnosis/recovery switches")

    memory_records = read_jsonl(memory_path) if memory_path.exists() else []
    if memory_path.exists():
        _verify_runtime_event_stream(
            memory_path, "memory_queries", str(summary["episode_id"])
        )
    post_queries: list[dict[str, Any]] = []
    post_query_ids: set[str] = set()
    pending_shadows: list[dict[str, Any]] = []
    replay_store: Any | None = None
    if evaluation_mode and system_id == "E3":
        from web_agent.memory.frozen_store import FrozenMemoryStore

        try:
            replay_store = FrozenMemoryStore.load(
                campaign_root
                / "memory"
                / f"seed_{int(schedule_row['matched_model_seed'])}"
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SchemaError(
                f"evaluation E3 frozen store cannot be replayed: {exc}"
            ) from exc
    for index, record in enumerate(memory_records):
        assert_no_verifier_evidence(record, context=f"{memory_path}:{index + 1}")
        event_type = str(record.get("event_type", ""))
        if event_type == "no_memory_shadow_before_retrieval":
            pending_shadows.append(_event_payload(record))
            continue
        if event_type != "post_failure_query":
            raise SchemaError(f"E3 memory stream has unknown event type: {event_type}")
        query = _memory_query_payload(record)
        if not pending_shadows:
            raise SchemaError("E3 query occurred without a prior no-memory shadow")
        shadow_decision = pending_shadows.pop(0)
        expected_shadow = sha256_json(shadow_decision)
        if query["shadow_decision_sha256"] != expected_shadow:
            raise SchemaError("E3 query shadow hash differs from the logged pre-query decision")
        if evaluation_mode:
            expected_store = (
                campaign_root
                / "memory"
                / f"seed_{int(schedule_row['matched_model_seed'])}"
                / "manifest.json"
            )
            query_hash = query.get("query_embedding_sha256")
            if not _is_sha256(query_hash):
                raise SchemaError(
                    "evaluation E3 query lacks a lowercase query-embedding SHA-256"
                )
            embedding_request_hash = query.get("embedding_request_sha256")
            processed_batch_hash = query.get("processed_batch_sha256")
            if not _is_sha256(embedding_request_hash) or not _is_sha256(
                processed_batch_hash
            ):
                raise SchemaError(
                    "evaluation E3 query lacks backend request/processed-batch hashes"
                )
            payload = _event_payload(record)
            query_contract = payload.get("query")
            if not isinstance(query_contract, Mapping):
                raise SchemaError("evaluation E3 query lacks its causal query contract")
            binding_fields = (
                "post_failure_observation_sha256",
                "post_action_input_sha256",
                "processor_contract_sha256",
                "checkpoint_sha256",
            )
            if any(
                not _is_sha256(query_contract.get(field))
                for field in binding_fields
            ):
                raise SchemaError(
                    "evaluation E3 query lacks observation/input/processor/checkpoint hashes"
                )
            if query_contract.get("query_id") != query.get("query_id"):
                raise SchemaError("evaluation E3 query/result IDs differ")
            embedding_request = payload.get("embedding_request")
            if not isinstance(embedding_request, Mapping):
                raise SchemaError(
                    "evaluation E3 query lacks its exact model embedding request"
                )
            if sha256_json(embedding_request) != embedding_request_hash:
                raise SchemaError(
                    "evaluation E3 embedding-request hash differs from logged input"
                )
            request_bindings = (
                "query_id",
                "post_failure_observation_id",
                "failed_action_id",
                *binding_fields,
            )
            if any(
                embedding_request.get(field) != query_contract.get(field)
                for field in request_bindings
            ):
                raise SchemaError(
                    "evaluation E3 embedding request differs from causal query binding"
                )
            observation_id = str(
                query_contract.get("post_failure_observation_id") or ""
            )
            matched_observations = [
                _event_payload(item)
                for item in environment_events
                if _event_payload(item).get("observation_id") == observation_id
            ]
            if len(matched_observations) != 1 or (
                matched_observations[0].get("observation_record_sha256")
                != query_contract.get("post_failure_observation_sha256")
            ):
                raise SchemaError(
                    "evaluation E3 embedding is not bound to its post-failure observation"
                )
            matched_transitions = [
                _event_payload(item).get("input")
                for item in transitions
                if isinstance(_event_payload(item).get("input"), Mapping)
                and _event_payload(item)["input"].get("post_observation", {}).get(
                    "observation_id"
                )
                == observation_id
            ]
            if len(matched_transitions) != 1 or sha256_json(
                matched_transitions[0]
            ) != query_contract.get("post_action_input_sha256"):
                raise SchemaError(
                    "evaluation E3 embedding is not bound to its post-action model input"
                )
            if embedding_request.get("post_action_input") != matched_transitions[0]:
                raise SchemaError(
                    "evaluation E3 backend request does not contain the exact transition"
                )
            frozen_model = read_json(
                campaign_root
                / "frozen"
                / "models"
                / f"seed_{int(schedule_row['matched_model_seed'])}.json"
            )
            if (
                query_contract.get("processor_contract_sha256")
                != frozen_model.get("processor_contract_sha256")
                or query_contract.get("checkpoint_sha256")
                != frozen_model.get("selected_checkpoint_sha256")
            ):
                raise SchemaError(
                    "evaluation E3 embedding processor/checkpoint binding differs"
                )
            expected_embedding_binding = sha256_json(
                {
                    "query_id": query_contract["query_id"],
                    "post_failure_observation_id": observation_id,
                    **{
                        field: query_contract[field]
                        for field in binding_fields
                    },
                    "embedding_request_sha256": embedding_request_hash,
                    "processed_batch_sha256": processed_batch_hash,
                    "embedding_stage": "post_action_memory_task_adapter",
                    "embedding_sha256": query_hash,
                    "normalized_embedding_sha256": query.get(
                        "normalized_query_embedding_sha256"
                    ),
                }
            )
            if query.get("embedding_binding_sha256") != expected_embedding_binding:
                raise SchemaError(
                    "evaluation E3 embedding-binding digest is not reproducible"
                )
            if (
                query.get("reader_id") != "frozen-memory-store-v1"
                or query.get("store_manifest_sha256") != sha256_file(expected_store)
            ):
                raise SchemaError(
                    "evaluation E3 query reader/store identity differs from its seed attestation"
                )
            considered = query.get("reader_considered_count")
            eligible = query.get("reader_eligible_count")
            if (
                type(considered) is not int
                or type(eligible) is not int
                or considered < 0
                or eligible < 0
                or eligible > considered
                or len(query.get("candidate_ids", ())) > eligible
            ):
                raise SchemaError("evaluation E3 query reader counts are invalid")
            _validate_and_replay_evaluation_memory_query(
                schedule_row=schedule_row,
                payload=payload,
                query_contract=query_contract,
                query_result=query,
                shadow_decision=shadow_decision,
                store=replay_store,
            )
        candidates = query["candidate_ids"]
        scores = query["scores"]
        if not isinstance(candidates, list) or not isinstance(scores, list):
            raise SchemaError("E3 query candidates/scores must be arrays")
        if len(candidates) != len(scores) or len(candidates) > 3:
            raise SchemaError("E3 query violates frozen top-3 candidate contract")
        candidate_ids = [str(value) for value in candidates]
        if any(not value for value in candidate_ids) or len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise SchemaError("E3 query candidate IDs are empty/duplicated")
        numeric_scores: list[float] = []
        for score in scores:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise SchemaError("E3 cosine score must be an exact JSON number")
            numeric = float(score)
            if not math.isfinite(numeric) or not -1.0 <= numeric <= 1.0:
                raise SchemaError("E3 cosine score must be finite in [-1, 1]")
            numeric_scores.append(numeric)
        if list(zip(numeric_scores, candidate_ids)) != sorted(
            zip(numeric_scores, candidate_ids), key=lambda item: (-item[0], item[1])
        ):
            raise SchemaError("E3 query violates cosine-score/memory-ID ordering")
        query_id = str(query["query_id"])
        if not query_id or query_id in post_query_ids:
            raise SchemaError("E3 query IDs are empty/duplicated")
        post_query_ids.add(query_id)
        admitted = strict_bool(query["admitted"], context="query.admitted")
        strict_bool(query["changed_strategy"], context="query.changed_strategy")
        strict_bool(
            query["changed_target_or_parameters"],
            context="query.changed_target_or_parameters",
        )
        admitted_id = query.get("admitted_candidate_id")
        if evaluation_mode and admitted != (admitted_id is not None):
            raise SchemaError("E3 query admission flag/candidate identity disagree")
        if admitted_id is not None and str(admitted_id) not in set(candidate_ids):
            raise SchemaError("E3 admitted memory was not among returned candidates")
        post_queries.append(query)
    if pending_shadows:
        raise SchemaError("E3 memory stream contains an unused no-memory shadow")
    if system_id != "E3" and memory_records:
        raise SchemaError(f"{system_id} emitted forbidden memory events")

    completed = strict_bool(summary["completed"], context="completed")
    if completed:
        final = _read_final_verification(sealed, str(summary["episode_id"]))
        _validate_terminal_signal_receipts(
            sealed,
            str(summary["episode_id"]),
            terminal_signals,
        )
        validate_ordered_verifier_receipts(
            runtime,
            sealed,
            episode_id=str(summary["episode_id"]),
        )
        evidence = final["evidence"]
        required_final = {
            "task_success",
            "terminal_reason",
            "loop_detected",
            "environment_failure",
            "failure_incidents",
            "recovery_verifications",
            "verified_failure_event_count",
            "repeated_error_event_count",
            "memory_relevance",
        }
        missing = sorted(required_final - set(evidence))
        if missing:
            raise SchemaError(f"sealed episode_final lacks fields {missing}: {sealed}")
        _validate_final_evidence(
            evidence,
            system_id=system_id,
            summary=summary,
            recovery_attempt_events=recovery_attempt_events,
            post_queries=post_queries,
            context=str(sealed),
        )


def _validate_frozen_runtime_budget_evidence(
    *,
    summary: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    recovery_events: Sequence[Mapping[str, Any]],
    environment_events: Sequence[Mapping[str, Any]] = (),
    protocol: Mapping[str, Any],
    evaluation_mode: bool,
) -> None:
    """Replay frozen request, recovery, timeout, and strategy limits from logs."""

    budgets = protocol.get("budgets")
    if not isinstance(budgets, Mapping):
        raise SchemaError("frozen protocol lacks runtime budgets")

    def registered_limit(name: str) -> int:
        value = budgets.get(name)
        if type(value) is not int or value < 0:
            raise SchemaError(f"frozen protocol budget {name} must be an exact integer")
        return value

    request_limit = registered_limit("executor_requests_per_episode")
    incident_limit = registered_limit("recovery_attempts_per_incident")
    episode_limit = registered_limit("recovery_attempts_per_episode")
    model_call_limit = registered_limit("model_calls_per_episode")
    timeout_limit = registered_limit("task_timeout_seconds")

    step_count = summary.get("step_count")
    attempt_count = summary.get("recovery_attempt_count")
    if type(step_count) is not int or step_count < 0:
        raise SchemaError("runtime step_count must be an exact nonnegative integer")
    if type(attempt_count) is not int or attempt_count < 0:
        raise SchemaError(
            "runtime recovery_attempt_count must be an exact nonnegative integer"
        )
    if step_count > request_limit:
        raise SchemaError(
            f"runtime exceeded frozen executor-request budget {request_limit}"
        )
    if attempt_count > episode_limit:
        raise SchemaError(
            f"runtime exceeded frozen per-episode recovery budget {episode_limit}"
        )

    model_call_count = summary.get("model_call_count")
    if model_call_count is None:
        if evaluation_mode:
            raise SchemaError("evaluation episode lacks runner-owned model-call evidence")
    elif type(model_call_count) is not int or model_call_count < 0:
        raise SchemaError(
            "runtime model_call_count must be an exact nonnegative integer"
        )
    elif model_call_count > model_call_limit:
        raise SchemaError(
            f"runtime exceeded frozen model-call budget {model_call_limit}"
        )

    dispatches = [
        record
        for record in environment_events
        if str(record.get("event_type", "")) == "model_call_dispatch"
    ]
    # Evaluation packages must carry independently replayable dispatch events.
    # Lightweight synthetic smoke runners may report an aggregate count without
    # exercising EpisodeRunner; when they do emit dispatches, those events are
    # still validated exactly.
    if (
        model_call_count is not None
        and (evaluation_mode or dispatches)
        and len(dispatches) != model_call_count
    ):
        raise SchemaError(
            "runner-owned model-call dispatch count differs from runtime summary"
        )
    if model_call_count is None and dispatches:
        raise SchemaError(
            "runtime emitted model-call dispatches without a summary count"
        )
    for expected_index, record in enumerate(dispatches, start=1):
        payload = _event_payload(record)
        if payload.get("model_call_index") != expected_index:
            raise SchemaError(
                "model-call dispatch indices must be exact, sequential, and one-based"
            )
        if payload.get("maximum_model_calls") != model_call_limit:
            raise SchemaError(
                "model-call dispatch cites a different frozen episode cap"
            )
        if payload.get("stage") not in REGISTERED_MODEL_CALL_STAGES:
            raise SchemaError("model-call dispatch uses an unregistered stage")
        component_id = payload.get("component_id")
        if not isinstance(component_id, str) or not component_id.strip():
            raise SchemaError("model-call dispatch lacks a component identity")
        episode_id = summary.get("episode_id")
        if episode_id is not None and payload.get("episode_id") != episode_id:
            raise SchemaError("model-call dispatch cites a different episode")

    wall_clock = summary.get("task_wall_clock_seconds")
    if wall_clock is None:
        if evaluation_mode:
            raise SchemaError("evaluation episode lacks frozen-timeout evidence")
    for duration_field in ("task_wall_clock_seconds", "elapsed_seconds"):
        duration = summary.get(duration_field)
        if duration is None:
            continue
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise SchemaError(f"runtime {duration_field} must be numeric")
        measured = float(duration)
        if not math.isfinite(measured) or measured < 0:
            raise SchemaError(
                f"runtime {duration_field} must be finite and nonnegative"
            )
        if measured > timeout_limit:
            raise SchemaError(
                f"runtime {duration_field} exceeded frozen task timeout "
                f"{timeout_limit} seconds"
            )

    budget_actions = [
        record
        for record in actions
        if strict_bool(
            _event_payload(record).get("budget_consumed", True),
            context="action.budget_consumed",
        )
    ]
    if len(budget_actions) != len(actions):
        raise SchemaError("action stream contains an uncharged executor request")
    if len(budget_actions) != step_count:
        raise SchemaError("budget-consuming requests differ from runtime step_count")
    recovery_actions_by_attempt: dict[str, list[str]] = defaultdict(list)
    for expected_step, record in enumerate(budget_actions, start=1):
        payload = _event_payload(record)
        execution = payload.get("execution")
        if not isinstance(execution, Mapping):
            raise SchemaError("budget-consuming action lacks execution evidence")
        executor_step = execution.get("executor_step")
        if type(executor_step) is not int or executor_step != expected_step:
            raise SchemaError(
                "executor_step values must be exact, sequential, and one-based"
            )
        if str(record.get("event_type", "")) != "recovery_action":
            continue
        action = payload.get("action")
        if not isinstance(action, Mapping):
            raise SchemaError("recovery action event lacks its concrete action")
        action_id_value = action.get("action_id")
        attempt_id_value = payload.get("attempt_id") or action.get(
            "recovery_attempt_id"
        )
        if (
            not isinstance(action_id_value, str)
            or not action_id_value
            or not isinstance(attempt_id_value, str)
            or not attempt_id_value
        ):
            raise SchemaError("recovery action lacks action/attempt identity")
        action_id = action_id_value
        attempt_id = attempt_id_value
        if action.get("recovery_attempt_id") != attempt_id:
            raise SchemaError("recovery action cites a different attempt")
        if execution.get("action_id") != action_id:
            raise SchemaError("recovery execution cites a different action")
        recovery_actions_by_attempt[attempt_id].append(action_id)

    plan_events: list[Mapping[str, Any]] = []
    attempt_events: list[Mapping[str, Any]] = []
    for record in recovery_events:
        event_type = str(record.get("event_type", ""))
        if event_type == "recovery_plan":
            plan_events.append(record)
        elif event_type == "recovery_attempt":
            attempt_events.append(record)
        else:
            raise SchemaError(f"recovery stream has unknown event type: {event_type}")
    if len(attempt_events) != attempt_count:
        raise SchemaError("recovery-attempt events differ from runtime summary")
    if len(plan_events) != len(attempt_events):
        raise SchemaError("every recovery attempt requires exactly one recovery plan")

    plans_by_id: dict[str, dict[str, Any]] = {}
    for record in plan_events:
        payload = _event_payload(record)
        nested = payload.get("plan")
        if not isinstance(nested, Mapping):
            raise SchemaError("recovery_plan event lacks its plan object")
        plan = dict(nested)
        require_keys(
            plan,
            (
                "attempt_id",
                "incident_id",
                "strategy",
                "incident_attempt_index",
                "episode_attempt_index",
                "actions",
                "resolution_status",
            ),
            context="runtime recovery_plan",
        )
        plan_id_value = plan["attempt_id"]
        incident_id_value = plan["incident_id"]
        strategy_value = plan["strategy"]
        if (
            not isinstance(plan_id_value, str)
            or not plan_id_value
            or not isinstance(incident_id_value, str)
            or not incident_id_value
            or not isinstance(strategy_value, str)
        ):
            raise SchemaError("recovery plan identity/strategy must be canonical strings")
        plan_id = plan_id_value
        if plan_id in plans_by_id:
            raise SchemaError("recovery plan IDs are empty or duplicated")
        for index_field in ("incident_attempt_index", "episode_attempt_index"):
            if type(plan[index_field]) is not int or plan[index_field] <= 0:
                raise SchemaError(
                    f"recovery plan {index_field} must be an exact positive integer"
                )
        plans_by_id[plan_id] = plan

    incident_counts: dict[str, int] = defaultdict(int)
    seen_attempt_ids: set[str] = set()
    abort_positions: list[int] = []
    for expected_episode_index, event in enumerate(attempt_events, start=1):
        attempt = _recovery_attempt_payload(event)
        require_keys(
            attempt,
            (
                "strategy",
                "episode_attempt_index",
                "action_ids",
                "completed",
            ),
            context="runtime recovery_attempt",
        )
        attempt_id_value = attempt["attempt_id"]
        incident_id_value = attempt["incident_id"]
        if (
            not isinstance(attempt_id_value, str)
            or not attempt_id_value
            or not isinstance(incident_id_value, str)
            or not incident_id_value
        ):
            raise SchemaError("recovery attempt identity must use canonical strings")
        attempt_id = attempt_id_value
        incident_id = incident_id_value
        if attempt_id in seen_attempt_ids:
            raise SchemaError("recovery attempt IDs are empty or duplicated")
        seen_attempt_ids.add(attempt_id)
        incident_counts[incident_id] += 1
        incident_index = attempt["incident_attempt_index"]
        episode_index = attempt["episode_attempt_index"]
        if (
            type(incident_index) is not int
            or incident_index != incident_counts[incident_id]
            or type(episode_index) is not int
            or episode_index != expected_episode_index
        ):
            raise SchemaError(
                "recovery attempt indices must be exact, sequential, and one-based"
            )
        if incident_counts[incident_id] > incident_limit:
            raise SchemaError(
                f"incident {incident_id} exceeded frozen recovery budget {incident_limit}"
            )
        if not isinstance(attempt["strategy"], str):
            raise SchemaError("recovery attempt strategy must be a canonical string")
        strategy = attempt["strategy"]
        if strategy not in REGISTERED_RECOVERY_LOW_LEVEL_ACTION_LIMITS:
            raise SchemaError(f"unregistered recovery strategy: {strategy}")
        action_ids = attempt["action_ids"]
        if not isinstance(action_ids, list) or any(
            not isinstance(action_id, str) or not action_id
            for action_id in action_ids
        ):
            raise SchemaError("recovery attempt action_ids must be a string list")
        if len(action_ids) != len(set(action_ids)):
            raise SchemaError("recovery attempt contains duplicate action IDs")
        if action_ids != recovery_actions_by_attempt.get(attempt_id, []):
            raise SchemaError(
                "recovery attempt action IDs differ from browser-step evidence"
            )
        completed = strict_bool(
            attempt["completed"], context="recovery_attempt.completed"
        )
        limit = REGISTERED_RECOVERY_LOW_LEVEL_ACTION_LIMITS[strategy]
        if len(action_ids) > limit:
            raise SchemaError(
                f"{strategy} exceeded registered low-level action limit {limit}"
            )

        plan = plans_by_id.get(attempt_id)
        if plan is None:
            raise SchemaError(f"recovery attempt lacks matching plan: {attempt_id}")
        for field in (
            "incident_id",
            "strategy",
            "incident_attempt_index",
            "episode_attempt_index",
        ):
            if plan[field] != attempt[field]:
                raise SchemaError(
                    f"recovery plan/attempt {field} differs: {attempt_id}"
                )
        planned_actions = plan["actions"]
        if not isinstance(planned_actions, list) or any(
            not isinstance(action, Mapping) for action in planned_actions
        ):
            raise SchemaError("recovery plan actions must be an object list")
        planned_id_values = [action.get("action_id") for action in planned_actions]
        if any(
            not isinstance(action_id, str) or not action_id
            for action_id in planned_id_values
        ):
            raise SchemaError("recovery plan contains an action without identity")
        planned_ids = list(planned_id_values)
        if len(planned_ids) > limit:
            raise SchemaError(
                f"{strategy} plan exceeded registered low-level action limit {limit}"
            )
        if action_ids != planned_ids[: len(action_ids)]:
            raise SchemaError(
                "executed recovery action IDs are not an ordered plan prefix"
            )
        resolution_status = str(plan["resolution_status"])
        if resolution_status not in {"READY", "REJECTED", "ABORT"}:
            raise SchemaError("recovery plan has an unregistered resolution status")
        if strategy == "ABORT":
            abort_positions.append(expected_episode_index)
            if (
                action_ids
                or planned_ids
                or resolution_status != "ABORT"
                or not completed
            ):
                raise SchemaError(
                    "ABORT must complete safely with zero action IDs/browser steps"
                )
        elif resolution_status == "ABORT":
            raise SchemaError("non-ABORT strategy used ABORT plan status")
        elif resolution_status == "READY" and len(planned_ids) != 1:
            raise SchemaError(
                f"{strategy} READY plan must contain exactly one low-level action"
            )
        elif resolution_status == "REJECTED" and planned_ids:
            raise SchemaError("rejected recovery plan cannot contain browser actions")

    unknown_action_attempts = set(recovery_actions_by_attempt) - seen_attempt_ids
    if unknown_action_attempts:
        raise SchemaError("recovery browser steps cite unknown attempts")
    if abort_positions and (
        len(abort_positions) != 1 or abort_positions[0] != len(attempt_events)
    ):
        raise SchemaError("ABORT must be the final recovery attempt in an episode")


def _validate_evaluator_backend_guards(
    *,
    environment_events: Sequence[Mapping[str, Any]],
    terminal_signals: Sequence[Mapping[str, Any]],
    required: bool,
) -> None:
    """Prove a monotonic one-to-one guard for every sealed callback."""

    guards = [
        record
        for record in environment_events
        if str(record.get("event_type", "")) == "sealed_evaluator_backend_guard"
    ]
    if not guards and not required:
        return

    expected_callbacks: list[str] = []
    for record in terminal_signals:
        event_type = str(record.get("event_type", ""))
        if event_type in {
            "after_reset",
            "after_normal_action",
            "after_recovery_action",
        }:
            expected_callbacks.append("transition_verifier")
        elif event_type == "episode_final_receipt":
            expected_callbacks.append("episode_final_verifier")
        else:
            raise SchemaError(
                f"terminal signal has no registered evaluator callback: {event_type}"
            )
    if required and not expected_callbacks:
        raise SchemaError("production episode lacks sealed evaluator callbacks")
    if len(guards) != len(expected_callbacks):
        raise SchemaError(
            "sealed evaluator backend guards do not cover callbacks 1:1"
        )
    for expected_index, (record, callback_kind) in enumerate(
        zip(guards, expected_callbacks, strict=True),
        start=1,
    ):
        payload = _event_payload(record)
        require_keys(
            payload,
            (
                "guard_index",
                "callback_kind",
                "before_sha256",
                "after_sha256",
                "unchanged",
            ),
            context="sealed evaluator backend guard",
        )
        if type(payload["guard_index"]) is not int or payload["guard_index"] != (
            expected_index
        ):
            raise SchemaError("sealed evaluator guard indices are not monotonic")
        if payload["callback_kind"] != callback_kind:
            raise SchemaError("sealed evaluator guard callback order differs")
        before = payload["before_sha256"]
        after = payload["after_sha256"]
        if not _is_sha256(before) or not _is_sha256(after) or before != after:
            raise SchemaError(
                "sealed evaluator backend digest changed across callback"
            )
        if not strict_bool(payload["unchanged"], context="backend_guard.unchanged"):
            raise SchemaError("sealed evaluator backend guard must be unchanged=true")


def _validate_manual_rescue_guard_events(
    *,
    environment_events: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    terminal_signals: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
    campaign_manifest: Mapping[str, Any],
    episode_id: str,
    task_id: str,
) -> None:
    """Reconstruct clean, chained guard coverage for one live WebArena episode."""

    identity = environment.get("manual_rescue_guard")
    if not isinstance(identity, Mapping):
        raise SchemaError("frozen environment lacks manual-rescue guard identity")
    integration = campaign_manifest.get("runtime_integration")
    source_sha256 = (
        integration.get("source_sha256") if isinstance(integration, Mapping) else None
    )
    if not _is_sha256(source_sha256):
        raise SchemaError("campaign lacks attested manual-rescue guard source")

    observation_by_id: dict[str, str] = {}
    for record in environment_events:
        payload = _event_payload(record)
        observation_id = payload.get("observation_id")
        observation_sha256 = payload.get("observation_record_sha256")
        if isinstance(observation_id, str) and _is_sha256(observation_sha256):
            if observation_id in observation_by_id:
                raise SchemaError("manual-rescue observation binding is ambiguous")
            observation_by_id[observation_id] = str(observation_sha256)

    action_by_id: dict[str, Mapping[str, Any]] = {}
    for record in actions:
        payload = _event_payload(record)
        action = payload.get("action")
        if not isinstance(action, Mapping):
            continue
        action_id = str(action.get("action_id") or "")
        action_sha256 = payload.get("action_sha256")
        if not action_id or not _is_sha256(action_sha256) or action_id in action_by_id:
            raise SchemaError("manual-rescue action binding is missing or ambiguous")
        action_by_id[action_id] = payload

    guard_records = [
        record
        for record in environment_events
        if str(record.get("event_type", "")) == "manual_rescue_guard"
    ]
    if not guard_records:
        raise SchemaError("production WebArena episode lacks manual-rescue evidence")
    checks: list[WebArenaManualRescueCheck] = []
    receipts: list[WebArenaManualRescueReceipt] = []
    previous_receipt_sha256 = "0" * 64
    for expected_index, record in enumerate(guard_records, start=1):
        payload = _event_payload(record)
        check_value = payload.get("check")
        receipt_value = payload.get("receipt")
        if not isinstance(check_value, Mapping) or not isinstance(
            receipt_value, Mapping
        ):
            raise SchemaError("manual-rescue event lacks typed check/receipt evidence")
        try:
            check = WebArenaManualRescueCheck(
                **{
                    item.name: check_value[item.name]
                    for item in dataclass_fields(WebArenaManualRescueCheck)
                }
            )
            receipt = WebArenaManualRescueReceipt(
                **{
                    item.name: receipt_value[item.name]
                    for item in dataclass_fields(WebArenaManualRescueReceipt)
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"manual-rescue evidence is invalid: {exc}") from exc
        if payload.get("check_sha256") != check.record_sha256 or payload.get(
            "receipt_sha256"
        ) != receipt.record_sha256:
            raise SchemaError("manual-rescue event digest differs from typed evidence")
        if (
            check.episode_id != episode_id
            or check.task_id != task_id
            or check.check_index != expected_index
            or check.previous_receipt_sha256 != previous_receipt_sha256
            or check.guard_id != identity.get("guard_id")
            or check.guard_version != identity.get("guard_version")
            or check.evidence_mode != identity.get("evidence_mode")
        ):
            raise SchemaError("manual-rescue check identity/chain differs")
        if (
            receipt.check_sha256 != check.record_sha256
            or receipt.check_index != check.check_index
            or receipt.stage != check.stage
            or receipt.guard_id != check.guard_id
            or receipt.guard_version != check.guard_version
            or receipt.evidence_mode != check.evidence_mode
            or receipt.source_sha256 != source_sha256
            or receipt.expected_registered_browser_steps
            != check.expected_registered_browser_steps
            or receipt.observed_registered_browser_steps
            != check.expected_registered_browser_steps
            or receipt.exclusive_automation_control is not True
            or receipt.non_agent_input_event_count != 0
            or receipt.manual_rescue_detected is not False
            or receipt.oracle_labels_observed is not False
        ):
            raise SchemaError("manual-rescue receipt is unclean or misbound")
        if observation_by_id.get(check.observation_id) != check.observation_sha256:
            raise SchemaError("manual-rescue check cites an unknown observation")
        if check.action_id is not None:
            action_payload = action_by_id.get(check.action_id)
            if (
                action_payload is None
                or action_payload.get("action_sha256") != check.action_sha256
            ):
                raise SchemaError("manual-rescue check cites an unknown action")
        previous_receipt_sha256 = receipt.record_sha256
        checks.append(check)
        receipts.append(receipt)

    terminals = [
        record
        for record in terminal_signals
        if str(record.get("event_type", "")) != "episode_final_receipt"
    ]
    if not terminals:
        raise SchemaError("manual-rescue guard has no terminal-check boundary")
    expected: list[tuple[str, str, str | None, int]] = []
    reset_observations = [
        _event_payload(record)
        for record in environment_events
        if str(record.get("event_type", "")) == "reset"
    ]
    if len(reset_observations) != 1:
        raise SchemaError("manual-rescue guard requires one reset observation")
    previous_observation_id = str(reset_observations[0].get("observation_id") or "")
    browser_steps = 0
    expected.append(("after_reset", previous_observation_id, None, browser_steps))
    for terminal in terminals:
        binding = _event_payload(terminal).get("receipt_binding")
        if not isinstance(binding, Mapping):
            raise SchemaError("manual-rescue terminal boundary lacks receipt binding")
        action_id = binding.get("action_id")
        observation_id = str(binding.get("observation_id") or "")
        if action_id is not None:
            action_payload = action_by_id.get(str(action_id))
            if action_payload is None:
                raise SchemaError("manual-rescue terminal cites an unknown action")
            execution = action_payload.get("execution")
            if not isinstance(execution, Mapping):
                raise SchemaError("manual-rescue action lacks execution evidence")
            error_kind = str(execution.get("error_kind") or "")
            reached_adapter = error_kind not in {
                "parameter_resolution_rejected",
                "pre_action_parse_rejected",
                "safety_rejection",
            }
            if reached_adapter:
                expected.append(
                    ("before_step", previous_observation_id, str(action_id), browser_steps)
                )
                if error_kind != "registered_safety_denial":
                    browser_steps += 1
                after_observation = (
                    previous_observation_id
                    if error_kind == "registered_safety_denial"
                    else observation_id
                )
                expected.append(
                    ("after_step", after_observation, str(action_id), browser_steps)
                )
        expected.append(
            ("before_terminal", observation_id, str(action_id) if action_id else None, browser_steps)
        )
        previous_observation_id = observation_id
    actual = [
        (
            check.stage,
            check.observation_id,
            check.action_id,
            check.expected_registered_browser_steps,
        )
        for check in checks
    ]
    if actual != expected:
        raise SchemaError("manual-rescue guard stage/action coverage differs")


def _validate_episode_contract_validation_receipt(
    *,
    runtime: Path,
    summary: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    recovery_events: Sequence[Mapping[str, Any]],
    environment_events: Sequence[Mapping[str, Any]],
    required: bool,
) -> None:
    """Verify the digest-only receipt from pre-redaction in-memory contracts."""

    receipt_positions = [
        index
        for index, record in enumerate(environment_events)
        if str(record.get("event_type", "")) == "episode_contract_validation"
    ]
    if not receipt_positions:
        if required:
            raise SchemaError(
                "production evaluation episode lacks in-memory contract validation receipt"
            )
        return
    if len(receipt_positions) != 1:
        raise SchemaError("episode has duplicate in-memory contract validation receipts")
    receipt_position = receipt_positions[0]
    trailing_types = {
        str(record.get("event_type", ""))
        for record in environment_events[receipt_position + 1 :]
    }
    if trailing_types - {"sealed_evaluator_backend_guard"}:
        raise SchemaError(
            "runtime decision evidence was appended after contract validation"
        )

    payload = _event_payload(environment_events[receipt_position])
    expected_payload_keys = {
        "receipt_version",
        "episode_id",
        "bundle_record_type",
        "bundle_sha256",
        "summary_sha256",
        "causal_log_snapshot_sha256",
        "validation",
        "validation_sha256",
    }
    if set(payload) != expected_payload_keys:
        raise SchemaError(
            "in-memory contract validation receipt has an invalid schema"
        )
    if payload["receipt_version"] != EPISODE_FOREIGN_KEY_RECEIPT_VERSION:
        raise SchemaError("in-memory contract validation receipt version differs")
    if payload["bundle_record_type"] != "EpisodeContractBundle":
        raise SchemaError("contract validation receipt cites another bundle type")
    if str(payload["episode_id"]) != str(summary.get("episode_id", "")):
        raise SchemaError("contract validation receipt belongs to another episode")
    for name in (
        "bundle_sha256",
        "summary_sha256",
        "causal_log_snapshot_sha256",
        "validation_sha256",
    ):
        if not _is_sha256(payload.get(name)):
            raise SchemaError(f"contract validation receipt has an invalid {name}")
    summary_snapshot = summary.get("event_log_sha256")
    if summary_snapshot != payload["causal_log_snapshot_sha256"]:
        raise SchemaError(
            "runtime summary differs from the receipt's causal-log snapshot"
        )
    reconstructed_snapshot = _pre_receipt_causal_log_snapshot_sha256(runtime)
    if reconstructed_snapshot != payload["causal_log_snapshot_sha256"]:
        raise SchemaError(
            "pre-receipt causal-log snapshot differs from hash-chained runtime files"
        )

    validation = payload["validation"]
    if not isinstance(validation, Mapping):
        raise SchemaError("contract validation receipt lacks its typed result")
    expected_validation_keys = {
        "schema_version",
        "record_type",
        "episode_id",
        "status",
        "observations",
        "actions",
        "executions",
        "transition_assessments",
        "recovery_attempts",
        "recovery_assessments",
        "memory_queries",
    }
    if set(validation) != expected_validation_keys:
        raise SchemaError("contract validation result has an invalid schema")
    if (
        validation["schema_version"] != RUNTIME_SCHEMA_VERSION
        or validation["record_type"] != "EpisodeForeignKeyValidation"
        or validation["status"] != "PASS"
        or str(validation["episode_id"]) != str(summary.get("episode_id", ""))
    ):
        raise SchemaError("contract validation result identity/status differs")
    if sha256_json(validation) != payload["validation_sha256"]:
        raise SchemaError("contract validation result hash differs")

    count_fields = expected_validation_keys - {
        "schema_version",
        "record_type",
        "episode_id",
        "status",
    }
    for name in count_fields:
        value = validation[name]
        if type(value) is not int or value < 0:
            raise SchemaError(f"contract validation {name} is not a nonnegative integer")

    parse_rejections = sum(
        str(record.get("event_type", "")) == "pre_action_parse_rejection"
        for record in actions
    )
    observation_count = sum(
        str(record.get("event_type", ""))
        in {"reset", "post_action_observation", "post_recovery_observation"}
        for record in environment_events
    )
    transition_count = sum(
        str(record.get("event_type", "")) == "post_action_assessment"
        for record in transitions
    )
    attempt_records = [
        record
        for record in recovery_events
        if str(record.get("event_type", "")) == "recovery_attempt"
    ]
    recovery_assessment_count = sum(
        isinstance(_event_payload(record).get("transition"), Mapping)
        and isinstance(_event_payload(record).get("predicted_assessment"), Mapping)
        for record in attempt_records
    )
    memory_query_count = summary.get("memory_queries")
    if type(memory_query_count) is not int or memory_query_count < 0:
        raise SchemaError(
            "production runtime summary lacks exact in-memory memory-query count"
        )
    expected_counts = {
        "observations": observation_count,
        "actions": len(actions) - parse_rejections,
        "executions": len(actions),
        "transition_assessments": transition_count,
        "recovery_attempts": len(attempt_records),
        "recovery_assessments": recovery_assessment_count,
        "memory_queries": memory_query_count,
    }
    for name, expected in expected_counts.items():
        if validation[name] != expected:
            raise SchemaError(
                f"contract validation {name} differs from hash-chained runtime logs"
            )


def _pre_receipt_causal_log_snapshot_sha256(runtime: Path) -> str:
    """Rebuild EpisodeRunner's pre-receipt aggregate file commitment.

    The contract-validation event and orchestration-owned episode-final receipt
    are intentionally outside this snapshot. Transition verifier receipts and
    their backend guards remain inside because EpisodeRunner observed them
    before validating its in-memory bundle.
    """

    stream_files = {
        "actions": "actions.jsonl",
        "transitions": "transitions.jsonl",
        "recoveries": "recoveries.jsonl",
        "environment_events": "environment_events.jsonl",
        "memory_queries": "memory_queries.jsonl",
        "terminal_signals": "terminal_signals.jsonl",
    }
    file_hashes: dict[str, str] = {}
    for stream, filename in stream_files.items():
        path = runtime / filename
        if not path.is_file():
            if stream == "memory_queries":
                continue
            raise SchemaError(f"causal-log snapshot lacks runtime stream: {path}")
        retained: list[bytes] = []
        found_contract_receipt = False
        for raw in path.read_bytes().splitlines(keepends=True):
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"invalid runtime stream while rebuilding {path}") from exc
            event_type = str(record.get("event_type", ""))
            if stream == "environment_events" and event_type == (
                "episode_contract_validation"
            ):
                found_contract_receipt = True
                break
            if stream == "terminal_signals" and event_type == "episode_final_receipt":
                continue
            retained.append(raw)
        if stream == "environment_events" and not found_contract_receipt:
            raise SchemaError(
                "cannot reconstruct causal-log snapshot without its receipt boundary"
            )
        file_hashes[stream] = hashlib.sha256(b"".join(retained)).hexdigest()
    return sha256_json(file_hashes)


def _validate_abort_terminal_semantics(
    *,
    evidence: Mapping[str, Any],
    summary: Mapping[str, Any],
    recovery_attempt_events: Sequence[Mapping[str, Any]],
) -> None:
    abort_attempt_ids = {
        str(attempt["attempt_id"])
        for attempt in (
            _recovery_attempt_payload(event) for event in recovery_attempt_events
        )
        if str(attempt.get("strategy")) == "ABORT"
    }
    if not abort_attempt_ids:
        return
    if strict_bool(evidence.get("task_success"), context="sealed.task_success"):
        raise SchemaError("ABORT cannot be counted as task success")
    if normalize_runtime_terminal_reason(
        summary.get("runtime_terminal_reason"),
        context="ABORT runtime summary",
    ) != "ABORT":
        raise SchemaError("ABORT requires runtime terminal_reason=ABORT")
    verifications = evidence.get("recovery_verifications")
    if not isinstance(verifications, list):
        raise SchemaError("ABORT evidence lacks recovery verifications")
    for verification in verifications:
        if not isinstance(verification, Mapping):
            raise SchemaError("ABORT recovery verification must be an object")
        if str(verification.get("recovery_attempt_id")) in abort_attempt_ids and (
            strict_bool(
                verification.get("successful"),
                context="abort_recovery.successful",
            )
        ):
            raise SchemaError("ABORT cannot be counted as recovery success")


def _validate_final_evidence(
    evidence: Mapping[str, Any],
    *,
    system_id: str,
    summary: Mapping[str, Any],
    recovery_attempt_events: Sequence[Mapping[str, Any]],
    post_queries: Sequence[Mapping[str, Any]],
    context: str,
) -> None:
    task_success = strict_bool(evidence["task_success"], context="sealed.task_success")
    strict_bool(evidence["loop_detected"], context="sealed.loop_detected")
    environment_failure = strict_bool(
        evidence["environment_failure"], context="sealed.environment_failure"
    )
    if not str(evidence["terminal_reason"]).strip():
        raise SchemaError(f"sealed terminal_reason is empty: {context}")
    if task_success and environment_failure:
        raise SchemaError(f"sealed episode cannot be success and environment failure: {context}")
    validate_primary_runtime_outcome(
        runtime_terminal_reason=summary.get("runtime_terminal_reason"),
        task_success=task_success,
        context=context,
    )
    _validate_abort_terminal_semantics(
        evidence=evidence,
        summary=summary,
        recovery_attempt_events=recovery_attempt_events,
    )
    if environment_failure and not strict_bool(
        summary["infrastructure_invalid"], context="summary.infrastructure_invalid"
    ):
        raise SchemaError(f"sealed environment failure entered a primary-valid package: {context}")
    for key in ("verified_failure_event_count", "repeated_error_event_count"):
        if type(evidence[key]) is not int or int(evidence[key]) < 0:
            raise SchemaError(f"sealed {key} must be a nonnegative integer: {context}")
    if int(evidence["repeated_error_event_count"]) > int(
        evidence["verified_failure_event_count"]
    ):
        raise SchemaError(f"sealed repeated errors exceed verified failure events: {context}")

    incidents = evidence["failure_incidents"]
    verifications = evidence["recovery_verifications"]
    if not isinstance(incidents, list) or not all(isinstance(item, Mapping) for item in incidents):
        raise SchemaError(f"sealed failure_incidents must be an object list: {context}")
    if not isinstance(verifications, list) or not all(
        isinstance(item, Mapping) for item in verifications
    ):
        raise SchemaError(f"sealed recovery_verifications must be an object list: {context}")
    incident_ids: set[str] = set()
    incident_attempt_counts: dict[str, int] = defaultdict(int)
    runtime_attempt_ids: set[str] = set()
    runtime_attempt_order: list[str] = []
    runtime_attempt_incidents: dict[str, str] = {}
    runtime_attempt_indices: dict[str, int] = {}
    for event in recovery_attempt_events:
        attempt = _recovery_attempt_payload(event)
        attempt_id = str(attempt["attempt_id"])
        if attempt_id in runtime_attempt_ids:
            raise SchemaError(f"duplicate runtime recovery attempt ID: {attempt_id}")
        runtime_attempt_ids.add(attempt_id)
        runtime_attempt_order.append(attempt_id)
        runtime_incident_id = str(attempt["incident_id"])
        runtime_attempt_incidents[attempt_id] = runtime_incident_id
        incident_attempt_index = attempt["incident_attempt_index"]
        if type(incident_attempt_index) is not int or incident_attempt_index <= 0:
            raise SchemaError(
                f"runtime recovery attempt has invalid incident index: {attempt_id}"
            )
        runtime_attempt_indices[attempt_id] = incident_attempt_index
        incident_attempt_counts[runtime_incident_id] += 1
    incident_by_id: dict[str, Mapping[str, Any]] = {}
    for item in incidents:
        incident_id = str(item.get("failure_incident_id", item.get("incident_id", "")))
        if not incident_id or incident_id in incident_ids:
            raise SchemaError(f"sealed failure incident IDs are empty/duplicate: {context}")
        incident_ids.add(incident_id)
        incident_by_id[incident_id] = item
        require_keys(
            item,
            ("verified_agent_failure", "resolved", "attempt_count"),
            context=f"sealed incident {incident_id}",
        )
        strict_bool(item["verified_agent_failure"], context="verified_agent_failure")
        strict_bool(item["resolved"], context="incident.resolved")
        resolved_attempt_index = item.get("resolved_attempt_index")
        if resolved_attempt_index not in (None, "") and (
            type(resolved_attempt_index) is not int or resolved_attempt_index <= 0
        ):
            raise SchemaError(
                f"sealed incident has invalid resolved_attempt_index: {incident_id}"
            )
        if type(item["attempt_count"]) is not int or int(item["attempt_count"]) < 0:
            raise SchemaError(f"sealed incident has invalid attempt_count: {incident_id}")
        if int(item["attempt_count"]) != incident_attempt_counts.get(incident_id, 0):
            raise SchemaError(f"sealed/runtime incident attempt counts disagree: {incident_id}")
    if int(evidence["verified_failure_event_count"]) < sum(
        strict_bool(item["verified_agent_failure"], context="verified_agent_failure")
        for item in incidents
    ):
        raise SchemaError(f"verified failure events are fewer than verified incidents: {context}")
    if not set(incident_attempt_counts).issubset(incident_ids):
        raise SchemaError(f"runtime recovery incidents lack sealed incident records: {context}")

    verified_attempt_ids: set[str] = set()
    verified_attempt_order: list[str] = []
    successful_attempt_indices: dict[str, list[int]] = defaultdict(list)
    for item in verifications:
        require_keys(
            item,
            (
                "recovery_attempt_id",
                "failure_incident_id",
                "verified_failure_present",
                "successful",
            ),
            context="sealed recovery verification",
        )
        attempt_id = str(item["recovery_attempt_id"])
        if not attempt_id or attempt_id in verified_attempt_ids:
            raise SchemaError(f"sealed recovery verification IDs are empty/duplicate: {context}")
        verified_attempt_ids.add(attempt_id)
        verified_attempt_order.append(attempt_id)
        verified_incident_id = str(item["failure_incident_id"])
        if not verified_incident_id:
            raise SchemaError(
                f"sealed recovery verification has an empty incident ID: {attempt_id}"
            )
        if runtime_attempt_incidents.get(attempt_id) != verified_incident_id:
            raise SchemaError(
                "sealed/runtime recovery attempt incident IDs disagree: "
                f"{attempt_id}"
            )
        failure_present = strict_bool(
            item["verified_failure_present"], context="verified_failure_present"
        )
        successful = strict_bool(item["successful"], context="recovery.successful")
        if successful and not failure_present:
            raise SchemaError("false-positive recovery trigger cannot be verified successful")
        if successful:
            successful_attempt_indices[verified_incident_id].append(
                runtime_attempt_indices[attempt_id]
            )
    if verified_attempt_ids != runtime_attempt_ids:
        raise SchemaError(f"sealed recovery labels do not cover runtime attempts 1:1: {context}")
    if verified_attempt_order != runtime_attempt_order:
        raise SchemaError(
            f"sealed recovery labels are not ordered like runtime attempts: {context}"
        )

    for incident_id, incident in incident_by_id.items():
        attempt_count = incident_attempt_counts.get(incident_id, 0)
        resolved = strict_bool(incident["resolved"], context="incident.resolved")
        declared_index = incident.get("resolved_attempt_index")
        declared_index = None if declared_index in (None, "") else declared_index
        successful_indices = successful_attempt_indices.get(incident_id, [])

        if attempt_count == 0:
            # E0/E1 or incidental later progress may resolve an incident without
            # a project-controller attempt.  Such a resolution cannot claim a
            # recovery-attempt ordinal.
            if declared_index is not None:
                raise SchemaError(
                    "sealed incident without recovery attempts cannot declare "
                    f"resolved_attempt_index: {incident_id}"
                )
            continue
        expected_resolved = bool(successful_indices)
        if resolved is not expected_resolved:
            raise SchemaError(
                "sealed incident resolved status differs from linked recovery "
                f"verifications: {incident_id}"
            )
        if not expected_resolved:
            if declared_index is not None:
                raise SchemaError(
                    "unresolved sealed incident cannot declare "
                    f"resolved_attempt_index: {incident_id}"
                )
            continue
        successful_index = min(successful_indices)
        if declared_index != successful_index:
            raise SchemaError(
                "sealed incident resolved_attempt_index differs from the linked "
                f"successful recovery verification: {incident_id}"
            )
    if system_id in {"E0", "E1"} and (incidents and runtime_attempt_ids):
        raise SchemaError(f"{system_id} has recovery attempts despite disabled controller")

    relevance = evidence["memory_relevance"]
    if not isinstance(relevance, Mapping):
        raise SchemaError(f"sealed memory_relevance must be an object: {context}")
    query_ids = {str(query["query_id"]) for query in post_queries}
    if system_id == "E3":
        if set(map(str, relevance)) != query_ids:
            raise SchemaError(f"sealed relevance does not cover E3 queries exactly: {context}")
        for query in post_queries:
            labels = _sealed_relevance_labels(relevance, str(query["query_id"]))
            if not isinstance(labels.get("relevant_ids"), list) or not str(
                labels.get("relevance_definition", "")
            ):
                raise SchemaError("E3 relevance label lacks relevant_ids/definition")
            if strict_bool(query["admitted"], context="query.admitted"):
                if labels.get("useful_intervention") is None or labels.get(
                    "harmful_intervention"
                ) is None:
                    raise SchemaError("admitted E3 intervention lacks useful/harmful labels")
    elif relevance:
        raise SchemaError(f"{system_id} contains forbidden memory relevance labels")


def _read_final_verification(sealed: Path, episode_id: str) -> dict[str, Any]:
    path = sealed / _path_id(episode_id) / "verifier_events.jsonl"
    records = verify_sealed_stream(path)
    finals = [record for record in records if record["event_kind"] == "episode_final"]
    if len(finals) != 1:
        raise SchemaError(f"expected exactly one sealed episode_final event: {path}")
    if not strict_bool(finals[0]["should_terminate"], context="episode_final.should_terminate"):
        raise SchemaError(f"sealed episode_final did not emit terminal signal: {path}")
    return finals[0]


def _validate_terminal_signal_receipts(
    sealed: Path,
    episode_id: str,
    runtime_records: list[dict[str, Any]],
) -> None:
    path = sealed / _path_id(episode_id) / "verifier_events.jsonl"
    verifier_records = verify_sealed_stream(path)
    runtime_by_id: dict[str, dict[str, Any]] = {}
    for record in runtime_records:
        payload = _event_payload(record)
        signal_id = str(payload.get("event_id", ""))
        if not signal_id:
            raise SchemaError(f"terminal signal runtime payload lacks event_id: {path}")
        if signal_id in runtime_by_id:
            raise SchemaError(f"duplicate runtime terminal signal receipt: {signal_id}")
        runtime_by_id[signal_id] = payload
    if set(runtime_by_id) != {str(record["event_id"]) for record in verifier_records}:
        raise SchemaError(f"runtime/sealed terminal signal event IDs differ: {path}")
    for record in verifier_records:
        payload = runtime_by_id[str(record["event_id"])]
        token = payload.get("token_sha256")
        terminate = payload.get("terminate")
        token_matches = token == record["opaque_token_sha256"]
        if isinstance(token, Mapping) and token.get("redacted") is True:
            token_matches = token.get("sha256") == hashlib.sha256(
                str(record["opaque_token_sha256"]).encode("utf-8")
            ).hexdigest()
        if not token_matches:
            raise SchemaError(f"runtime terminal token does not match sealed receipt: {path}")
        if strict_bool(terminate, context="terminal_signal.terminate") != strict_bool(
            record["should_terminate"], context="sealed.should_terminate"
        ):
            raise SchemaError(f"runtime terminal bit does not match sealed receipt: {path}")


def _verify_runtime_event_stream(path: Path, stream: str, episode_id: str) -> None:
    allowed_envelope = {
        "schema_version",
        "record_type",
        "stream",
        "episode_id",
        "system_id",
        "task_id",
        "repeat_id",
        "matched_seed",
        "event_id",
        "sequence",
        "previous_record_hash",
        "event_type",
        "timestamp_utc",
        "payload",
        "record_hash",
    }
    for index, record in enumerate(read_jsonl(path), start=1):
        if set(record) != allowed_envelope:
            raise SchemaError(
                "runtime event envelope has extra/missing fields: "
                f"{path}:{index}"
            )
    try:
        from web_agent.runtime.event_log import verify_event_log

        verify_event_log(
            path,
            expected_stream=stream,
            expected_episode_id=episode_id,
        )
    except ImportError:  # Defensive fallback for evaluating detached artifacts.
        records = read_jsonl(path)
        previous = "0" * 64
        for index, record in enumerate(records, start=1):
            required = {
                "sequence",
                "previous_record_hash",
                "record_hash",
                "stream",
                "episode_id",
            }
            if not required.issubset(record):
                raise SchemaError(f"runtime event stream lacks hash fields: {path}:{index}")
            if int(record["sequence"]) != index or record["previous_record_hash"] != previous:
                raise SchemaError(f"runtime event stream chain mismatch: {path}:{index}")
            body = {key: value for key, value in record.items() if key != "record_hash"}
            expected = sha256_json(body)
            if (
                record["record_hash"] != expected
                or record["stream"] != stream
                or record["episode_id"] != episode_id
            ):
                raise SchemaError(f"runtime event stream hash/type mismatch: {path}:{index}")
            previous = expected
    except ValueError as exc:
        raise SchemaError(str(exc)) from exc


def _event_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = record.get("payload", record)
    if not isinstance(payload, Mapping):
        raise SchemaError("runtime event payload must be an object")
    return dict(payload)


def _recovery_attempt_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = _event_payload(record)
    nested = payload.get("attempt", payload)
    if not isinstance(nested, Mapping):
        raise SchemaError("recovery_attempt event has no attempt object")
    attempt = dict(nested)
    require_keys(
        attempt,
        ("attempt_id", "incident_id", "incident_attempt_index"),
        context="runtime recovery_attempt",
    )
    return attempt


def _memory_query_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = _event_payload(record)
    nested = payload.get("query_result", payload)
    if not isinstance(nested, Mapping):
        raise SchemaError("post_failure_query event has no query_result object")
    query = dict(nested)
    require_keys(
        query,
        (
            "query_id",
            "candidate_ids",
            "scores",
            "shadow_decision_sha256",
            "admitted",
            "changed_strategy",
            "changed_target_or_parameters",
        ),
        context="runtime post_failure_query",
    )
    return query


def _validate_and_replay_evaluation_memory_query(
    *,
    schedule_row: Mapping[str, Any],
    payload: Mapping[str, Any],
    query_contract: Mapping[str, Any],
    query_result: Mapping[str, Any],
    shadow_decision: Mapping[str, Any],
    store: Any,
) -> None:
    """Independently replay E3 exclusion, cosine ranking, and intervention.

    The runtime-reported candidate IDs, scores, admission, and strategy are not
    accepted as authoritative.  Validation reconstructs the exact float32
    L2-normalized query, reopens the immutable per-seed store, and recomputes
    every downstream decision from frozen artifacts.
    """

    import numpy as np

    from web_agent.memory.frozen_store import FrozenMemoryStore, QueryExclusions

    if not isinstance(store, FrozenMemoryStore):
        raise SchemaError("evaluation E3 retrieval replay lacks a frozen store")
    if store.model_seed != int(schedule_row["matched_model_seed"]):
        raise SchemaError("evaluation E3 replay store belongs to another seed")

    vector_payload = query_result.get("normalized_query_embedding")
    vector_sha256 = query_result.get("normalized_query_embedding_sha256")
    if not isinstance(vector_payload, list) or len(vector_payload) != (
        MEMORY_QUERY_VECTOR_DIMENSION
    ):
        raise SchemaError(
            "evaluation E3 query lacks its normalized 768-dimensional vector"
        )
    if not _is_sha256(vector_sha256):
        raise SchemaError(
            "evaluation E3 query lacks a normalized-vector SHA-256"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in vector_payload
    ):
        raise SchemaError("evaluation E3 normalized vector must contain JSON numbers")
    vector = np.asarray(vector_payload, dtype=np.float32)
    if vector.shape != (MEMORY_QUERY_VECTOR_DIMENSION,) or not np.isfinite(
        vector
    ).all():
        raise SchemaError("evaluation E3 normalized vector is malformed/non-finite")
    if not math.isclose(
        float(np.linalg.norm(vector)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-5,
    ):
        raise SchemaError("evaluation E3 query vector is not L2-normalized")
    if float32_vector_sha256(vector.tolist()) != vector_sha256:
        raise SchemaError(
            "evaluation E3 normalized-vector hash differs from its float32 values"
        )

    task_id = query_contract.get("task_id")
    episode_id = query_contract.get("episode_id")
    duplicate_clusters = query_contract.get("duplicate_cluster_ids")
    if (
        not isinstance(task_id, str)
        or not task_id
        or not isinstance(episode_id, str)
        or not episode_id
        or not isinstance(duplicate_clusters, list)
        or not duplicate_clusters
        or any(not isinstance(value, str) or not value for value in duplicate_clusters)
    ):
        raise SchemaError(
            "evaluation E3 replay lacks task/episode/duplicate-cluster exclusions"
        )
    if len(duplicate_clusters) != len(set(duplicate_clusters)):
        raise SchemaError("evaluation E3 duplicate-cluster exclusions are duplicated")

    replayed = store.query(
        vector,
        exclusions=QueryExclusions(
            current_task_id=task_id,
            current_episode_id=episode_id,
            duplicate_cluster_ids=frozenset(duplicate_clusters),
        ),
    )
    expected_ids = [hit.memory_id for hit in replayed.hits]
    expected_scores = [float(hit.cosine_similarity) for hit in replayed.hits]
    actual_ids = [str(value) for value in query_result.get("candidate_ids", ())]
    actual_scores = query_result.get("scores")
    if actual_ids != expected_ids:
        raise SchemaError(
            "evaluation E3 candidate IDs differ from frozen-store replay"
        )
    if not isinstance(actual_scores, list) or len(actual_scores) != len(
        expected_scores
    ):
        raise SchemaError("evaluation E3 scores differ from frozen-store replay")
    if any(
        not math.isclose(
            float(actual),
            expected,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        for actual, expected in zip(actual_scores, expected_scores, strict=True)
    ):
        raise SchemaError("evaluation E3 scores differ from frozen-store replay")
    logged_exclusions = query_result.get("exclusion_reasons")
    if not isinstance(logged_exclusions, Mapping):
        raise SchemaError("evaluation E3 exclusion reasons must be an object")
    if dict(logged_exclusions) != dict(replayed.exclusion_reasons):
        raise SchemaError(
            "evaluation E3 exclusion reasons differ from frozen-store replay"
        )
    if (
        query_result.get("reader_considered_count") != replayed.considered_count
        or query_result.get("reader_eligible_count") != replayed.eligible_count
    ):
        raise SchemaError("evaluation E3 reader counts differ from frozen-store replay")

    admitted_hit = (
        replayed.hits[0]
        if replayed.hits
        and replayed.hits[0].cosine_similarity >= store.admission_threshold
        else None
    )
    expected_admitted_id = admitted_hit.memory_id if admitted_hit is not None else None
    if query_result.get("admitted_candidate_id") != expected_admitted_id or (
        query_result.get("admitted") is not (admitted_hit is not None)
    ):
        raise SchemaError(
            "evaluation E3 admission differs from frozen threshold replay"
        )

    shadow_strategy = str(shadow_decision.get("strategy") or "")
    if not shadow_strategy:
        raise SchemaError("evaluation E3 shadow decision lacks a strategy")
    expected_strategy = (
        str(admitted_hit.item.get("strategy"))
        if admitted_hit is not None
        else shadow_strategy
    )
    expected_changed_strategy = expected_strategy != shadow_strategy
    if query_result.get("final_strategy") != expected_strategy:
        raise SchemaError(
            "evaluation E3 final strategy differs from frozen retrieval replay"
        )
    if query_result.get("changed_strategy") is not expected_changed_strategy:
        raise SchemaError(
            "evaluation E3 changed-strategy flag differs from retrieval replay"
        )
    if query_result.get("changed_target_or_parameters") is not False:
        raise SchemaError(
            "evaluation E3 changed target/parameters outside the registered intervention"
        )

    logged_shadow = payload.get("shadow_decision")
    logged_final = payload.get("final_decision")
    if logged_shadow != shadow_decision or not isinstance(logged_final, Mapping):
        raise SchemaError("evaluation E3 decision payload differs from its shadow event")
    expected_final = dict(shadow_decision)
    if admitted_hit is not None:
        decision_id = str(shadow_decision.get("decision_id") or "")
        if not decision_id:
            raise SchemaError("evaluation E3 admitted shadow lacks a decision ID")
        expected_final["decision_id"] = (
            f"{decision_id}:memory:{admitted_hit.memory_id}"
        )
        expected_final["strategy"] = expected_strategy
    if dict(logged_final) != expected_final:
        raise SchemaError(
            "evaluation E3 final decision differs from frozen retrieval replay"
        )


def _validate_normal_action_parameters(
    payload: Mapping[str, Any],
    *,
    execution_status: str,
    prompt_path: Path,
) -> None:
    parameters = payload.get("parameters")
    error = payload.get("parameter_error")
    attempts = payload.get("resolution_attempts")
    trace_payload = payload.get("parameter_resolution")
    if not isinstance(attempts, list):
        raise SchemaError("normal action resolution_attempts must be an array")
    if not isinstance(trace_payload, Mapping):
        raise SchemaError("normal action lacks typed parameter-resolution evidence")
    try:
        trace = ParameterResolutionTrace.from_dict(trace_payload)
    except (TypeError, ValueError) as exc:
        raise SchemaError("normal action parameter-resolution trace is invalid") from exc
    if not isinstance(trace, ParameterResolutionTrace):  # pragma: no cover
        raise SchemaError("normal action reconstructed the wrong parameter trace")
    if [item.to_dict() for item in trace.attempts] != attempts:
        raise SchemaError("normal action attempt evidence differs from its trace")
    if parameters is None:
        expected_attempts = (
            ("deterministic", "REJECTED"),
            ("frozen_base_fallback", "REJECTED"),
        )
        actual_attempts: list[tuple[str, str]] = []
        for row in attempts:
            if not isinstance(row, Mapping):
                raise SchemaError("parameter resolution attempt must be an object")
            actual_attempts.append((str(row.get("source")), str(row.get("status"))))
        action = payload.get("action")
        if (
            tuple(actual_attempts) != expected_attempts
            or trace.resolved
            or trace.resolution_source is not None
            or tuple(item.source for item in trace.attempts)
            != tuple(source for source, _ in expected_attempts)
            or not isinstance(error, str)
            or not error.strip()
            or execution_status != "REJECTED"
            or not isinstance(action, Mapping)
            or action.get("parameters") != {}
        ):
            raise SchemaError(
                "invalid provider output must be deterministic+single-fallback then one rejection"
            )
        return
    if not isinstance(parameters, Mapping):
        raise SchemaError("normal action parameters must be an object or null")
    require_keys(
        parameters,
        (
            "provider_id",
            "provider_version",
            "prompt_sha256",
            "resolution_source",
            "attempted_sources",
            "decoding_parameters",
            "latency_ms",
            "resolution_trace",
        ),
        context="normal action parameters",
    )
    expected_prompt_hash = sha256_file(prompt_path)
    if (
        parameters["provider_id"] != "deterministic_then_frozen_base_fallback"
        or parameters["provider_version"] != "v1"
        or parameters["prompt_sha256"] != expected_prompt_hash
        or not isinstance(parameters["decoding_parameters"], Mapping)
        or error is not None
        or parameters["resolution_trace"] != trace_payload
        or not trace.resolved
    ):
        raise SchemaError("normal action provider signature differs from frozen provider")
    resolution_source = str(parameters["resolution_source"])
    expected_sources = {
        "deterministic": ["deterministic"],
        "frozen_base_fallback": ["deterministic", "frozen_base_fallback"],
    }
    if resolution_source not in expected_sources or parameters["attempted_sources"] != expected_sources[
        resolution_source
    ]:
        raise SchemaError("normal action parameter resolution provenance is invalid")
    if (
        trace.resolution_source != resolution_source
        or [item.source for item in trace.attempts]
        != expected_sources[resolution_source]
        or float(parameters["latency_ms"]) != float(trace.latency_ms)
    ):
        raise SchemaError("normal action provider timing provenance is inconsistent")


def _validate_pre_action_parse_rejection(
    payload: Mapping[str, Any],
    *,
    system_id: str,
) -> None:
    """Validate one charged parser request that never reached the browser."""

    if system_id != "E0":
        raise SchemaError("pre-action parser rejection is registered only for E0")
    expected_keys = {
        "rejection",
        "decision",
        "parameters",
        "action",
        "action_sha256",
        "execution",
    }
    if set(payload) != expected_keys:
        raise SchemaError("pre-action parse rejection payload fields differ from schema")
    if any(
        payload[name] is not None
        for name in ("decision", "parameters", "action", "action_sha256")
    ):
        raise SchemaError("pre-action parse rejection must not invent an action")
    rejection_payload = payload.get("rejection")
    execution_payload = payload.get("execution")
    if not isinstance(rejection_payload, Mapping) or not isinstance(
        execution_payload, Mapping
    ):
        raise SchemaError("pre-action parse rejection lacks typed evidence")
    try:
        rejection = PreActionParseRejection.from_dict(rejection_payload)
    except (TypeError, ValueError) as exc:
        raise SchemaError("pre-action parse rejection contract is invalid") from exc
    if execution_payload.get("action_id") != rejection.request_id:
        raise SchemaError(
            "pre-action parse rejection execution receipt is inconsistent"
        )
    try:
        execution = ExecutionResult.from_dict(execution_payload)
    except (TypeError, ValueError) as exc:
        raise SchemaError("pre-action parse rejection contract is invalid") from exc
    if not isinstance(rejection, PreActionParseRejection) or not isinstance(
        execution, ExecutionResult
    ):  # pragma: no cover - strict from_dict invariant
        raise SchemaError("pre-action parse rejection reconstructed wrong contracts")
    if (
        execution.action_id != rejection.request_id
        or execution.status is not ExecutionStatus.REJECTED
        or execution.state_changed
        or execution.environment_error
        or execution.error_kind != "pre_action_parse_rejected"
        or execution.internal_retry_count != 0
    ):
        raise SchemaError("pre-action parse rejection execution receipt is inconsistent")


def _sealed_relevance_labels(
    relevance: Any, query_id: str
) -> dict[str, Any]:
    if not isinstance(relevance, Mapping):
        raise SchemaError("sealed memory_relevance must be keyed by query ID")
    raw = relevance.get(query_id)
    if not isinstance(raw, Mapping):
        raise SchemaError(f"sealed memory relevance is missing query {query_id}")
    allowed = {
        "relevant_ids",
        "relevance_definition",
        "useful_intervention",
        "harmful_intervention",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise SchemaError(
            f"sealed relevance for {query_id} contains runtime/unknown fields: {sorted(unknown)}"
        )
    return {key: raw[key] for key in allowed if key in raw}


def _validate_runtime_artifact_hashes(runtime: Path, *, system_id: str) -> None:
    manifest_path = runtime / "artifact_hashes.json"
    manifest = read_json(manifest_path)
    require_keys(manifest, ("schema_version", "hash_algorithm", "files"), context=str(manifest_path))
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["hash_algorithm"] != "sha256":
        raise SchemaError(f"unsupported runtime artifact hash manifest: {manifest_path}")
    files = manifest["files"]
    if not isinstance(files, Mapping):
        raise SchemaError(f"runtime artifact hash files must be a mapping: {manifest_path}")
    required_names = set(RUNTIME_FILES) - {"artifact_hashes.json"}
    if system_id == "E3":
        required_names.add("memory_queries.jsonl")
    if not required_names.issubset(files):
        raise SchemaError(
            f"runtime artifact hashes omit required files {sorted(required_names - set(files))}: {manifest_path}"
        )
    listed = set(map(str, files))
    entries = list(runtime.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise SchemaError(f"runtime artifact package contains a symlink: {runtime}")
    actual = {
        str(path.relative_to(runtime))
        for path in entries
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    if listed != actual:
        raise SchemaError(f"runtime artifact hash manifest lacks exact file closure: {runtime}")
    allowed_regular = required_names | {"memory_queries.jsonl"}
    for relative in sorted(actual - allowed_regular):
        screenshot_match = re.fullmatch(
            r"screenshots/[0-9]{6}-([0-9a-f]{64})\.png",
            relative,
        )
        if screenshot_match is None or files.get(relative) != screenshot_match.group(1):
            raise SchemaError(
                "runtime artifact package contains an unregistered extra file: "
                f"{runtime / relative}"
            )
    for relative, expected in files.items():
        candidate = (runtime / str(relative)).resolve()
        if runtime.resolve() not in candidate.parents:
            raise SchemaError(f"runtime artifact hash path escapes package: {relative}")
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise SchemaError(f"runtime artifact hash mismatch: {candidate}")


def _validate_frozen_hashes(root: Path, manifest: Mapping[str, Any]) -> None:
    hash_path = root / "artifact_hashes.json"
    hashes = read_json(hash_path)
    require_keys(hashes, ("schema_version", "hash_algorithm", "files"), context="artifact_hashes")
    if (
        hashes["schema_version"] != SCHEMA_VERSION
        or hashes["hash_algorithm"] != "sha256"
        or not isinstance(hashes["files"], Mapping)
    ):
        raise SchemaError("artifact_hashes has unsupported format")
    listed = set(map(str, hashes["files"]))
    required = {
        "frozen/campaign.yaml",
        "frozen/protocol.yaml",
        str(FROZEN_PAPER_CLAIM_REGISTRY_RELATIVE_PATH),
        "frozen/environment.json",
        "frozen/provenance.json",
        "frozen/task_registry.json",
        "frozen/task_content_hashes.json",
        "frozen/benchmark/recovery_scenarios.json",
        "frozen/benchmark/recovery_oracle_rules.json",
        "frozen/benchmark/audit_manifest.json",
        "frozen/benchmark/duplicate_audit_manifest.json",
        str(FROZEN_PILOT_TASK_EXCLUSION_RELATIVE_PATH),
        "frozen/prompts/parameter_provider_v1.txt",
        "frozen/prompts/e0_action_v1.txt",
        "schedule/schedule.jsonl",
        *(f"frozen/systems/{system_id.lower()}.yaml" for system_id in SYSTEM_IDS),
    }
    if str(manifest.get("campaign_mode")) != "smoke":
        required.update(
            {
                f"frozen/{FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH}",
                f"frozen/{PREFLIGHT_ARTIFACT_RELATIVE_PATH}",
                f"frozen/{PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH}",
                str(FROZEN_HANDOFF_CONSUMPTION_RELATIVE_PATH),
            }
        )
    task_manifests = {
        relative for relative in listed if relative.startswith("frozen/task_manifest.")
    }
    if len(task_manifests) != 1:
        raise SchemaError("artifact hashes must bind exactly one frozen task manifest")
    required.update(task_manifests)
    if not required.issubset(listed):
        raise SchemaError(
            f"artifact hashes omit required frozen inputs: {sorted(required - listed)}"
        )
    actual_immutable = {
        str(path.relative_to(root))
        for directory in (root / "frozen", root / "schedule", root / "memory")
        if directory.exists()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if listed != actual_immutable:
        raise SchemaError(
            "artifact hash manifest does not exactly close over frozen/schedule/memory inputs"
        )
    for relative, expected in hashes["files"].items():
        candidate = (root / relative).resolve()
        if root not in candidate.parents:
            raise SchemaError(f"artifact hash path escapes campaign: {relative}")
        if not candidate.is_file():
            raise SchemaError(f"frozen artifact is missing: {relative}")
        if sha256_file(candidate) != expected:
            raise SchemaError(f"frozen artifact hash mismatch: {relative}")
    if "artifact_hashes_sha256" in manifest and sha256_file(hash_path) != manifest["artifact_hashes_sha256"]:
        raise SchemaError("artifact_hashes file differs from campaign_manifest")


def _validate_frozen_model_and_memory_manifests(
    root: Path, campaign_manifest: Mapping[str, Any]
) -> None:
    campaign_profile = classify_campaign_profile(
        campaign_manifest,
        context="frozen model/memory campaign manifest",
        require_campaign_mode=True,
    )
    final = campaign_profile == CAMPAIGN_PROFILE_FINAL
    model_paths = sorted((root / "frozen" / "models").glob("seed_*.json"))
    memory_paths = sorted((root / "memory").glob("seed_*/manifest.json"))
    smoke = campaign_manifest["campaign_mode"] == "smoke"
    expected_seeds = {int(seed) for seed in campaign_manifest.get("matched_seeds", [])}
    if not smoke and len(model_paths) != len(expected_seeds):
        raise SchemaError("evaluation campaign lacks one model manifest per matched seed")
    if not smoke and len(memory_paths) != len(expected_seeds):
        raise SchemaError("evaluation campaign lacks one memory manifest per matched seed")
    if final and smoke:
        raise SchemaError("a locked final campaign cannot use smoke mode")
    model_hashes: dict[int, str] = {}
    model_config_hashes: dict[int, str] = {}
    model_config_record_hashes: dict[int, str] = {}
    for path in model_paths:
        value = read_json(path)
        if type(value.get("model_seed")) is not int:
            raise SchemaError(f"model manifest lacks exact integer model_seed: {path}")
        seed = int(value["model_seed"])
        if path.stem != f"seed_{seed}" or seed not in expected_seeds:
            raise SchemaError(f"model manifest filename/seed mismatch: {path}")
        checkpoint_hash = value.get("selected_checkpoint_sha256")
        if not _is_sha256(checkpoint_hash):
            raise SchemaError(f"model manifest lacks selected checkpoint hash: {path}")
        if (
            value.get("selection_scope") != "validation_only"
            or value.get("checkpoint_selection") != "validation_only"
            or type(value.get("validation_rows_read")) is not int
            or int(value["validation_rows_read"]) <= 0
            or type(value.get("test_rows_read")) is not int
            or int(value["test_rows_read"]) != 0
            or type(value.get("locked_test_rows_read")) is not int
            or int(value["locked_test_rows_read"]) != 0
        ):
            raise SchemaError(f"model manifest is not validation-only/zero-test: {path}")
        for hash_key in (
            "resolved_config_sha256",
            "resolved_config_record_sha256",
            "processor_contract_sha256",
        ):
            if not _is_sha256(value.get(hash_key)):
                raise SchemaError(f"model manifest lacks {hash_key}: {path}")
        for text_key in (
            "e0_backbone_id",
            "e0_backbone_revision",
            "e0_parser_id",
            "e0_parser_version",
            "e0_parser_module",
            "e0_parser_attribute",
        ):
            if not str(value.get(text_key) or "").strip():
                raise SchemaError(f"model manifest lacks {text_key}: {path}")
        for hash_key in (
            "e0_backbone_sha256",
            "e0_resolved_config_sha256",
            "e0_processor_contract_sha256",
            "e0_base_prompt_sha256",
            "e0_parser_sha256",
        ):
            if not _is_sha256(value.get(hash_key)):
                raise SchemaError(f"model manifest lacks {hash_key}: {path}")
        if value.get("e0_base_prompt_sha256") != sha256_file(
            root / "frozen" / "prompts" / "e0_action_v1.txt"
        ):
            raise SchemaError(f"model manifest E0 prompt differs from frozen prompt: {path}")
        if not smoke:
            payloads = _validate_model_artifact_payloads(path, value, path_base=root)
            expected_payloads = campaign_manifest.get("model_payloads_by_seed")
            if not isinstance(expected_payloads, Mapping) or expected_payloads.get(
                str(seed)
            ) != value.get("artifact_payloads"):
                raise SchemaError(f"campaign/model payload descriptors disagree for seed {seed}")
            if set(payloads) != set(MODEL_PAYLOAD_ROLES):
                raise SchemaError(f"frozen model payload role coverage differs for seed {seed}")
            evidence = _validate_model_evidence_bundle(
                path,
                value,
                path_base=root,
                executable_payloads=payloads,
            )
            expected_evidence = campaign_manifest.get("model_evidence_by_seed")
            if not isinstance(expected_evidence, Mapping) or expected_evidence.get(
                str(seed)
            ) != value.get("model_evidence_bundle"):
                raise SchemaError(
                    f"campaign/model evidence bundles disagree for seed {seed}"
                )
            if set(evidence) != set(MODEL_EVIDENCE_ROLES):
                raise SchemaError(
                    f"frozen model evidence role coverage differs for seed {seed}"
                )
        model_hashes[seed] = str(checkpoint_hash)
        model_config_hashes[seed] = str(value["resolved_config_sha256"])
        model_config_record_hashes[seed] = str(
            value["resolved_config_record_sha256"]
        )
    required_memory = {
        "embedding_stage": "post_action_memory_task_adapter",
        "embedding_dimension": 768,
        "normalization": "l2",
        "similarity": "cosine",
        "top_k": 3,
        "tie_break": "memory_id_ascending",
        "admission_threshold_source": "train_only_calibration",
        "same_task_exclusion": True,
        "duplicate_exclusion": True,
        "runtime_writes_allowed": False,
    }
    for path in memory_paths:
        value = read_json(path)
        seed = int(value.get("model_seed", -1))
        if path.parent.name != f"seed_{seed}" or seed not in expected_seeds:
            raise SchemaError(f"memory manifest filename/seed mismatch: {path}")
        _load_and_verify_frozen_memory_store(path.parent)
        if seed in model_hashes and value.get("checkpoint_sha256") != model_hashes[seed]:
            raise SchemaError(f"memory/model checkpoint hashes disagree for seed {seed}")
        if seed in model_config_hashes and value.get("resolved_config_sha256") != model_config_hashes[seed]:
            raise SchemaError(f"memory/model resolved-config hashes disagree for seed {seed}")
        if (
            seed in model_config_record_hashes
            and value.get("resolved_config_record_sha256")
            != model_config_record_hashes[seed]
        ):
            raise SchemaError(
                "memory/model canonical resolved-config record hashes disagree "
                f"for seed {seed}"
            )
        if value.get("protocol_sha256") != sha256_file(root / "frozen" / "protocol.yaml"):
            raise SchemaError(f"memory store protocol hash differs from frozen protocol: {path}")
        for key, expected in required_memory.items():
            if value.get(key) != expected:
                raise SchemaError(
                    f"memory manifest {path.name} has {key}={value.get(key)!r}; expected {expected!r}"
                )
        threshold = value.get("admission_threshold")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise SchemaError(f"memory manifest {path.name} lacks numeric admission_threshold")
        if not -1.0 <= float(threshold) <= 1.0:
            raise SchemaError(f"memory manifest {path.name} admission_threshold is outside [-1,1]")
        for field in ("validation_rows_read", "test_rows_read", "locked_test_rows_read"):
            if type(value.get(field)) is not int or int(value[field]) != 0:
                raise SchemaError(
                    f"memory manifest {path} lacks explicit zero {field}"
                )


def _validate_schedule_rng_anchors(
    schedule: Sequence[Mapping[str, Any]],
    campaign_manifest: Mapping[str, Any],
) -> None:
    protocol_id = str(campaign_manifest.get("protocol_id", ""))
    campaign_id = str(campaign_manifest.get("campaign_id", ""))
    campaign_seed = int(campaign_manifest["campaign_seed"])
    for row in schedule:
        if int(row.get("stage_seed_decision_index", -1)) != 0:
            raise SchemaError("schedule stage-seed anchor must use decision_index=0")
        seeds = row.get("stage_seeds")
        if not isinstance(seeds, Mapping) or set(seeds) != set(DEFAULT_STAGE_KEYS):
            raise SchemaError("schedule stage seeds differ from registered runtime stages")
        for stage in DEFAULT_STAGE_KEYS:
            expected = _runtime_stage_seed(
                protocol_id=protocol_id,
                campaign_id=campaign_id,
                campaign_seed=campaign_seed,
                task_id=str(row["task_id"]),
                repeat_id=int(row["repeat_id"]),
                matched_seed=int(row["matched_model_seed"]),
                stage=stage,
            )
            if int(seeds[stage]) != expected:
                raise SchemaError(
                    f"schedule RNG anchor mismatch for {row['block_id']}/{stage}"
                )


def _validate_rng_provenance(
    runtime: Path,
    schedule_row: Mapping[str, Any],
    campaign_root: Path,
) -> None:
    manifest = read_json(campaign_root / "campaign_manifest.json")
    provenance = read_json(runtime / "rng_provenance.json")
    expected = {
        "algorithm": "sha256_stage_keyed_v1",
        "protocol_id": manifest["protocol_id"],
        "campaign_id": manifest["campaign_id"],
        "campaign_seed": manifest["campaign_seed"],
        "task_id": schedule_row["task_id"],
        "repeat_id": schedule_row["repeat_id"],
        "matched_seed": schedule_row["matched_model_seed"],
        "anchor_decision_index": 0,
        "stage_seed_anchors": schedule_row["stage_seeds"],
        "system_id_in_seed_key": False,
        "rerun_id_in_seed_key": False,
    }
    for key, expected_value in expected.items():
        if provenance.get(key) != expected_value:
            raise SchemaError(f"runtime RNG provenance mismatch for {key}: {runtime}")

    environment_records = read_jsonl(runtime / "environment_events.jsonl")
    plans = [
        _event_payload(record)
        for record in environment_records
        if str(record.get("event_type", "")) == "rng_seed_plan"
    ]
    uses = [
        _event_payload(record)
        for record in environment_records
        if str(record.get("event_type", "")) == "stage_rng_use"
    ]
    if len(plans) != 1 or not uses:
        raise SchemaError(f"runtime must log one RNG seed plan and consumed seeds: {runtime}")
    plan = plans[0]
    plan_expected = {
        "protocol_id": manifest["protocol_id"],
        "campaign_id": manifest["campaign_id"],
        "campaign_seed": int(manifest["campaign_seed"]),
        "task_id": str(schedule_row["task_id"]),
        "repeat_id": int(schedule_row["repeat_id"]),
        "matched_seed": int(schedule_row["matched_model_seed"]),
        "decision_index": 0,
        "stage_seeds": schedule_row["stage_seeds"],
    }
    for key, expected_value in plan_expected.items():
        if plan.get(key) != expected_value:
            raise SchemaError(f"runtime RNG seed plan mismatch for {key}: {runtime}")
    try:
        from web_agent.runtime.protocol import StageRNGFactory

        factory = StageRNGFactory(
            protocol_id=str(manifest["protocol_id"]),
            campaign_id=str(manifest["campaign_id"]),
            campaign_seed=int(manifest["campaign_seed"]),
        )
        for use in uses:
            key = use.get("key")
            if not isinstance(key, Mapping) or "system_id" in key:
                raise SchemaError("runtime RNG use has invalid/system-specific key")
            kwargs = {
                "task_id": str(key["task_id"]),
                "repeat_id": int(key["repeat_id"]),
                "matched_seed": int(key["matched_seed"]),
                "stage": str(key["stage"]),
                "decision_index": int(key["decision_index"]),
                "incident_index": int(key.get("incident_index", 0)),
                "attempt_index": int(key.get("attempt_index", 0)),
                "stream": str(key.get("stream", "default")),
            }
            if (
                kwargs["task_id"] != str(schedule_row["task_id"])
                or kwargs["repeat_id"] != int(schedule_row["repeat_id"])
                or kwargs["matched_seed"] != int(schedule_row["matched_model_seed"])
            ):
                raise SchemaError("runtime RNG use escaped its scheduled block")
            if int(use.get("seed", -1)) != factory.seed_for(**kwargs):
                raise SchemaError("runtime RNG use seed does not match its frozen key")
    except ImportError as exc:
        raise SchemaError("runtime RNG verifier is unavailable") from exc


def _validate_duplicate_audit_runtime_binding(
    campaign_root: Path,
    *,
    schedule_row: Mapping[str, Any],
    system_id: str,
    environment_records: Sequence[Mapping[str, Any]],
) -> None:
    bindings = [
        _event_payload(record)
        for record in environment_records
        if str(record.get("event_type", "")) == "duplicate_audit_binding"
    ]
    if system_id != "E3":
        if bindings:
            raise SchemaError(f"{system_id} consumed duplicate-audit evidence")
        return
    if len(bindings) != 1:
        raise SchemaError("evaluation E3 must log exactly one duplicate-audit binding")
    path = campaign_root / "frozen" / "benchmark" / "duplicate_audit_manifest.json"
    try:
        from web_agent.runtime.duplicate_audit import FrozenDuplicateAuditManifest

        registry = FrozenDuplicateAuditManifest.from_path(path)
        task_id = str(schedule_row["task_id"])
        partition = str(schedule_row["task_partition"])
        clusters = registry.clusters_for(
            task_id,
            task_partition=partition,
            allow_synthetic_diagnostic=partition == "recovery_diagnostic",
        )
        expected = {
            "manifest_id": registry.manifest_id,
            "manifest_sha256": registry.manifest_sha256,
            "entry_binding_sha256": registry.entry_binding_sha256(task_id),
            "task_partition": partition,
            "duplicate_cluster_ids": list(clusters),
        }
    except (ImportError, TypeError, ValueError, RuntimeError) as exc:
        raise SchemaError(f"cannot validate E3 duplicate-audit binding: {exc}") from exc
    if bindings[0] != expected:
        raise SchemaError("E3 duplicate-audit runtime binding differs from frozen evidence")


def _validate_memory_store_runtime_binding(
    campaign_root: Path,
    *,
    schedule_row: Mapping[str, Any],
    system_id: str,
    environment_records: Sequence[Mapping[str, Any]],
) -> None:
    """Bind E3, before reset, to the exact attested immutable seed store."""

    binding_records = [
        record
        for record in environment_records
        if str(record.get("event_type", "")) == "memory_store_binding"
    ]
    if system_id != "E3":
        if binding_records:
            raise SchemaError(f"{system_id} consumed a memory-store binding")
        return
    if len(binding_records) != 1:
        raise SchemaError("evaluation E3 must log exactly one memory-store binding")
    binding_record = binding_records[0]
    reset_sequences = [
        int(record["sequence"])
        for record in environment_records
        if str(record.get("event_type", "")) == "reset"
    ]
    if not reset_sequences or int(binding_record["sequence"]) >= min(reset_sequences):
        raise SchemaError("E3 memory-store binding must be logged before environment reset")

    seed = int(schedule_row["matched_model_seed"])
    manifest_path = campaign_root / "memory" / f"seed_{seed}" / "manifest.json"
    expected_hash = sha256_file(manifest_path)
    binding = _event_payload(binding_record)
    expected = {
        "reader_id": "frozen-memory-store-v1",
        "reader_evidence_scope": "EVALUATION_ATTESTED",
        "store_manifest_sha256": expected_hash,
        "expected_store_manifest_sha256": expected_hash,
        "model_seed": seed,
        "frozen": True,
        "write_enabled": False,
    }
    if binding != expected:
        raise SchemaError(
            "E3 memory-store runtime binding differs from its seed attestation"
        )


def _validate_aggregate_package(root: Path, report: ValidationReport) -> None:
    aggregate = root / "aggregate"
    required = (
        "episodes.csv",
        "metrics.json",
        "metrics.csv",
        "recovery_diagnostic_metrics.json",
        "retrieval_diagnostics.json",
        "recovery_diagnostic_retrieval.json",
        "statistics.json",
        "analysis_source_identity.json",
        "validation_report.json",
        "table2_main.csv",
        "table2_companion.csv",
        "results_manifest.json",
        "artifact_hashes.json",
    )
    for filename in required:
        if not (aggregate / filename).is_file():
            report.error(f"aggregate package is missing {filename}")
    if report.errors:
        return
    metrics = read_json(aggregate / "metrics.json")
    diagnostics = read_json(aggregate / "recovery_diagnostic_metrics.json")
    if metrics.get("headline_task_partition") != "normal":
        raise SchemaError("headline metrics are not restricted to normal tasks")
    if diagnostics.get("task_partition") != "recovery_diagnostic":
        raise SchemaError("recovery diagnostics lost their task-partition label")
    manifest = read_json(root / "campaign_manifest.json")
    campaign_profile = classify_campaign_profile(
        manifest,
        context="aggregate campaign manifest",
        require_campaign_mode=True,
    )
    analysis_identity = validate_analysis_source_identity(root)
    if read_json(aggregate / "analysis_source_identity.json") != analysis_identity:
        raise SchemaError(
            "aggregate analysis-source identity differs from exact attestation replay"
        )
    if campaign_profile == CAMPAIGN_PROFILE_PILOT:
        expected_status = str(metrics.get("publication_status", ""))
        if expected_status not in {DRAFT_PILOT_STATUS, "PILOT_ONLY"}:
            raise SchemaError(
                "pilot aggregate must be explicitly DRAFT_PILOT_ONLY or PILOT_ONLY"
            )
        if expected_status == "PILOT_ONLY":
            try:
                validate_manual_adjudication_completion(root)
            except (
                FileNotFoundError,
                OSError,
                TypeError,
                ValueError,
                KeyError,
                SchemaError,
            ) as exc:
                raise SchemaError(
                    "final PILOT_ONLY aggregate requires completed, human-attested, "
                    "outcome-label-hidden "
                    f"adjudication: {exc}"
                ) from exc
    else:
        expected_status = FINAL_READY_STATUS
    if metrics.get("publication_status") != expected_status:
        raise SchemaError("aggregate publication status differs from campaign boundary")
    _validate_recomputed_aggregate_outputs(
        root,
        publication_status=expected_status,
        campaign_manifest=manifest,
    )
    _validate_table_csv_guards(root, expected_status)
    _validate_manual_audit_package(root)
    _validate_results_package(root, expected_status)
    _validate_aggregate_hashes(root)
    _validate_campaign_evidence_manifest(root)


def _validate_recomputed_aggregate_outputs(
    root: Path,
    *,
    publication_status: str,
    campaign_manifest: Mapping[str, Any],
) -> None:
    """Rebuild all numeric JSON aggregates from immutable episode evidence.

    Artifact hashes prove that files have not changed relative to a manifest,
    but an attacker (or a faulty export step) can regenerate both together.  The
    independent source of truth is the selected runtime package joined with the
    post-episode sealed verifier evidence.  Recomputing here makes a self-
    consistent but numerically falsified aggregate fail closed.
    """

    campaign_profile = classify_campaign_profile(
        campaign_manifest,
        context="aggregate reconstruction campaign manifest",
        require_campaign_mode=True,
    )

    protocol = load_yaml(root / "frozen" / "protocol.yaml")
    records = _load_selected_analysis_records_unchecked(root)
    normal_episode_ids = {
        str(row["episode_id"])
        for row in records["episodes"]
        if row.get("task_partition") == "normal"
    }
    recovery_episode_ids = {
        str(row["episode_id"])
        for row in records["episodes"]
        if row.get("task_partition") == "recovery_diagnostic"
    }
    normal_episodes = [
        row
        for row in records["episodes"]
        if str(row["episode_id"]) in normal_episode_ids
    ]
    recovery_episodes = [
        row
        for row in records["episodes"]
        if str(row["episode_id"]) in recovery_episode_ids
    ]
    normal_attempts = [
        row
        for row in records["recovery_attempts"]
        if str(row["episode_id"]) in normal_episode_ids
    ]
    recovery_attempts = [
        row
        for row in records["recovery_attempts"]
        if str(row["episode_id"]) in recovery_episode_ids
    ]
    normal_incidents = [
        row
        for row in records["failure_incidents"]
        if str(row["episode_id"]) in normal_episode_ids
    ]
    recovery_incidents = [
        row
        for row in records["failure_incidents"]
        if str(row["episode_id"]) in recovery_episode_ids
    ]
    normal_queries = [
        row
        for row in records["memory_queries"]
        if str(row["episode_id"]) in normal_episode_ids
    ]
    recovery_queries = [
        row
        for row in records["memory_queries"]
        if str(row["episode_id"]) in recovery_episode_ids
    ]

    recovery_k = int(protocol["budgets"].get("recovery_at_k", 2))
    statistics_cfg = protocol.get("statistics", {})
    bootstrap_samples = int(statistics_cfg.get("bootstrap_samples", 10_000))
    confidence = float(statistics_cfg.get("confidence_level", 0.95))
    bootstrap_seed = int(statistics_cfg.get("seed", 20250831))
    rate_inference = {
        "bootstrap_samples": bootstrap_samples,
        "confidence": confidence,
        "bootstrap_seed": bootstrap_seed,
    }

    metrics = compute_table2_metrics(
        normal_episodes,
        recovery_attempts=normal_attempts,
        failure_incidents=normal_incidents,
        schedule_attempts=records["schedule_attempts"],
        recovery_k=recovery_k,
        **rate_inference,
    )
    diagnostic_metrics = compute_table2_metrics(
        recovery_episodes,
        recovery_attempts=recovery_attempts,
        failure_incidents=recovery_incidents,
        recovery_k=recovery_k,
        **rate_inference,
    )
    retrieval = compute_retrieval_diagnostics(
        normal_queries,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        bootstrap_seed=bootstrap_seed,
    )
    diagnostic_retrieval = compute_retrieval_diagnostics(
        recovery_queries,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        bootstrap_seed=bootstrap_seed,
    )

    statistic_keys = ["task_success", "step_count", "loop_detected"]
    if all(row.get("task_wall_clock_seconds") is not None for row in normal_episodes):
        statistic_keys.append("task_wall_clock_seconds")
    if all(row.get("model_call_count") is not None for row in normal_episodes):
        statistic_keys.append("model_call_count")
    statistics = compute_paired_contrasts(
        normal_episodes,
        metric_keys=tuple(statistic_keys),
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=bootstrap_seed,
    )
    statistics["clustered_ratio_contrasts"] = compute_clustered_ratio_contrasts(
        _aggregate_ratio_statistic_rows(
            normal_episodes,
            recovery_attempts=normal_attempts,
            failure_incidents=normal_incidents,
        ),
        ratio_fields={
            "success_after_initial_failure": (
                "saf_success_count",
                "verified_failure_episode_count",
            ),
            "registered_recovery_success": (
                "successful_recovery_attempt_count",
                "initiated_recovery_attempt_count",
            ),
        },
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=bootstrap_seed,
    )

    diagnostic_status = (
        publication_status
        if campaign_profile == CAMPAIGN_PROFILE_PILOT
        else "PILOT_ONLY"
    )
    metrics["publication_status"] = publication_status
    metrics["headline_task_partition"] = "normal"
    metrics["paper_table_status"] = (
        "READY" if publication_status == FINAL_READY_STATUS else "N/R"
    )
    diagnostic_metrics["publication_status"] = diagnostic_status
    diagnostic_metrics["task_partition"] = "recovery_diagnostic"
    retrieval["publication_status"] = publication_status
    diagnostic_retrieval["publication_status"] = diagnostic_status
    statistics["publication_status"] = publication_status

    expected_by_name = {
        "metrics.json": metrics,
        "recovery_diagnostic_metrics.json": diagnostic_metrics,
        "retrieval_diagnostics.json": retrieval,
        "recovery_diagnostic_retrieval.json": diagnostic_retrieval,
        "statistics.json": statistics,
    }
    for filename, expected in expected_by_name.items():
        expected_json = json.loads(canonical_json_bytes(expected))
        observed = read_json(root / "aggregate" / filename)
        mismatch = _first_json_mismatch(expected_json, observed)
        if mismatch is not None:
            location, expected_value, observed_value = mismatch
            raise SchemaError(
                "aggregate numeric recomputation mismatch: "
                f"{filename} at {location}; expected={expected_value!r}, "
                f"observed={observed_value!r}"
            )
        canonical_payload = (
            json.dumps(
                expected_json,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        if (root / "aggregate" / filename).read_bytes() != canonical_payload:
            raise SchemaError(
                f"aggregate JSON is not the canonical recomputation: {filename}"
            )
    _validate_canonical_summary_exports(
        root,
        records=records,
        metrics=metrics,
        retrieval=retrieval,
        statistics=statistics,
        recovery_k=recovery_k,
        publication_status=publication_status,
    )


def _validate_canonical_summary_exports(
    root: Path,
    *,
    records: Mapping[str, list[dict[str, Any]]],
    metrics: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    statistics: Mapping[str, Any],
    recovery_k: int,
    publication_status: str,
) -> None:
    """Recreate every shareable table/result byte from verified raw evidence."""

    # Lazy import avoids the intentional summary -> package-validator import
    # direction during module initialization. At validation time the package
    # module is fully loaded and these are the exact attested writers used by
    # the summarization command.
    from .summary import (
        _export_results,
        _write_companion_table,
        _write_episode_csv,
        _write_main_table,
        _write_metric_csv,
    )

    aggregate = root / "aggregate"
    with tempfile.TemporaryDirectory(prefix="table2-canonical-export-") as raw_temp:
        temporary = Path(raw_temp)
        expected_aggregate = temporary / "aggregate"
        expected_aggregate.mkdir()
        _write_episode_csv(expected_aggregate, records["episodes"])
        _write_metric_csv(expected_aggregate, metrics)
        _write_main_table(expected_aggregate, metrics, publication_status)
        _write_companion_table(
            expected_aggregate,
            metrics,
            recovery_k,
            publication_status,
        )
        for filename in (
            "episodes.csv",
            "metrics.csv",
            "table2_main.csv",
            "table2_companion.csv",
        ):
            if (aggregate / filename).read_bytes() != (
                expected_aggregate / filename
            ).read_bytes():
                raise SchemaError(
                    f"aggregate canonical export recomputation mismatch: {filename}"
                )

        audit_selection = read_json(
            root / "manual_audit" / "selection_manifest.json"
        )
        expected_results = temporary / "results"
        expected_manifest = _export_results(
            root,
            expected_results,
            metrics=metrics,
            retrieval=retrieval,
            statistics=statistics,
            validation={"status": "PASS"},
            audit_selection=audit_selection,
        )
        observed_manifest = read_json(aggregate / "results_manifest.json")
        observed_results = Path(str(observed_manifest.get("results_dir", "")))
        expected_names = set(expected_manifest["files"])
        if set(observed_manifest.get("files", {})) != expected_names:
            raise SchemaError("redacted result manifest has wrong canonical closure")
        for filename in sorted(expected_names):
            observed = observed_results / filename
            expected = expected_results / filename
            if (
                observed.is_symlink()
                or not observed.is_file()
                or observed.read_bytes() != expected.read_bytes()
            ):
                raise SchemaError(
                    f"redacted result canonical recomputation mismatch: {filename}"
                )


def _aggregate_ratio_statistic_rows(
    episodes: Sequence[Mapping[str, Any]],
    *,
    recovery_attempts: Sequence[Mapping[str, Any]],
    failure_incidents: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reconstruct the registered ratio-bootstrap inputs from sealed evidence."""

    attempts_by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in recovery_attempts:
        attempts_by_episode[str(row["episode_id"])].append(row)
    incidents_by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in failure_incidents:
        incidents_by_episode[str(row["episode_id"])].append(row)

    output: list[dict[str, Any]] = []
    for source in episodes:
        episode = dict(source)
        episode_id = str(episode["episode_id"])
        verified_failure = any(
            strict_bool(
                row["verified_agent_failure"],
                context="statistics.verified_agent_failure",
            )
            for row in incidents_by_episode.get(episode_id, [])
        )
        initiated = [
            row
            for row in attempts_by_episode.get(episode_id, [])
            if strict_bool(row.get("initiated", True), context="statistics.initiated")
        ]
        successful = sum(
            strict_bool(
                row["verified_failure_present"],
                context="statistics.verified_failure_present",
            )
            and strict_bool(row["successful"], context="statistics.recovery_success")
            for row in initiated
        )
        output.append(
            {
                **episode,
                "verified_failure_episode_count": int(verified_failure),
                "saf_success_count": int(
                    verified_failure
                    and strict_bool(
                        episode["task_success"], context="statistics.task_success"
                    )
                ),
                "initiated_recovery_attempt_count": len(initiated),
                "successful_recovery_attempt_count": successful,
            }
        )
    return output


def _first_json_mismatch(
    expected: Any,
    observed: Any,
    *,
    path: str = "$",
) -> tuple[str, Any, Any] | None:
    """Return the first deterministic, type-sensitive JSON-tree mismatch."""

    if type(expected) is not type(observed):
        return path, expected, observed
    if isinstance(expected, Mapping):
        expected_keys = set(expected)
        observed_keys = set(observed)
        if expected_keys != observed_keys:
            return (
                f"{path}.__keys__",
                sorted(map(str, expected_keys)),
                sorted(map(str, observed_keys)),
            )
        for key in sorted(expected_keys, key=str):
            mismatch = _first_json_mismatch(
                expected[key], observed[key], path=f"{path}.{key}"
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f"{path}.__length__", len(expected), len(observed)
        for index, (expected_item, observed_item) in enumerate(
            zip(expected, observed, strict=True)
        ):
            mismatch = _first_json_mismatch(
                expected_item, observed_item, path=f"{path}[{index}]"
            )
            if mismatch is not None:
                return mismatch
        return None
    if expected != observed:
        return path, expected, observed
    return None


def _validate_table_csv_guards(root: Path, publication_status: str) -> None:
    aggregate = root / "aggregate"
    with (aggregate / "table2_main.csv").open(encoding="utf-8", newline="") as handle:
        main_rows = list(csv.DictReader(handle))
    with (aggregate / "table2_companion.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        companion_rows = list(csv.DictReader(handle))
    if [row.get("System") for row in main_rows] != list(SYSTEM_IDS):
        raise SchemaError("Table 2 main CSV must contain E0--E3 exactly once")
    if [row.get("System") for row in companion_rows] != list(SYSTEM_IDS):
        raise SchemaError("Table 2 companion CSV must contain E0--E3 exactly once")
    if publication_status in {DRAFT_PILOT_STATUS, "PILOT_ONLY"}:
        for rows in (main_rows, companion_rows):
            for row in rows:
                for key, value in row.items():
                    if key == "System":
                        continue
                    if key == "Evidence Status":
                        if value != publication_status:
                            raise SchemaError("pilot table has a non-pilot evidence status")
                    elif value != "N/R":
                        raise SchemaError("pilot table exposes values instead of N/R")


def _derive_manual_audit_selection_evidence(
    records: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Derive sampling strata/categories only from selected raw episode evidence."""

    required_record_sets = {
        "episodes",
        "recovery_attempts",
        "failure_incidents",
        "memory_queries",
        "schedule_attempts",
    }
    if set(records) != required_record_sets or not all(
        isinstance(records[name], list) for name in required_record_sets
    ):
        raise SchemaError("manual-audit source records have the wrong exact schema")
    episodes = [
        dict(row)
        for row in records["episodes"]
        if row.get("task_partition") == "normal"
    ]
    by_episode: dict[str, dict[str, Any]] = {}
    for row in episodes:
        episode_id = _audit_nonempty_string(
            row.get("episode_id"), context="manual-audit raw episode_id"
        )
        if episode_id in by_episode:
            raise SchemaError("manual-audit raw evidence repeats an episode ID")
        if row.get("system_id") not in SYSTEM_IDS:
            raise SchemaError("manual-audit raw evidence has an invalid system ID")
        by_episode[episode_id] = row

    recovery_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records["recovery_attempts"]:
        episode_id = str(row.get("episode_id", ""))
        initiated = _audit_exact_bool(
            row.get("initiated", True),
            context="manual-audit raw recovery initiated",
        )
        if episode_id in by_episode and initiated:
            recovery_by_episode[episode_id].append(dict(row))
    incidents_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records["failure_incidents"]:
        episode_id = str(row.get("episode_id", ""))
        if episode_id in by_episode:
            incidents_by_episode[episode_id].append(dict(row))
    queries_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records["memory_queries"]:
        episode_id = str(row.get("episode_id", ""))
        if episode_id in by_episode:
            queries_by_episode[episode_id].append(dict(row))

    systems_by_block: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for episode in episodes:
        block_id = _audit_nonempty_string(
            episode.get("block_id"), context="manual-audit raw block_id"
        )
        system_id = str(episode["system_id"])
        if system_id in systems_by_block[block_id]:
            raise SchemaError("manual-audit raw evidence repeats a block/system cell")
        systems_by_block[block_id][system_id] = episode

    paired_e2_by_e3: dict[str, str] = {}
    disagreements: set[str] = set()
    for systems in systems_by_block.values():
        if "E2" not in systems or "E3" not in systems:
            continue
        e2 = systems["E2"]
        e3 = systems["E3"]
        e3_id = str(e3["episode_id"])
        paired_e2_by_e3[e3_id] = str(e2["episode_id"])
        e2_success = _audit_exact_bool(
            e2.get("task_success"), context="manual-audit raw E2 task_success"
        )
        e3_success = _audit_exact_bool(
            e3.get("task_success"), context="manual-audit raw E3 task_success"
        )
        if e2_success != e3_success:
            disagreements.add(e3_id)

    categories_by_episode: dict[str, set[str]] = {
        episode_id: set() for episode_id in by_episode
    }
    evidence_by_episode: dict[str, dict[str, Any]] = {}
    failure_markers = ("BBOX", "GROUND", "TARGET", "PARAMETER", "INVALID_ACTION")
    for episode_id, episode in by_episode.items():
        task_success = _audit_exact_bool(
            episode.get("task_success"), context="manual-audit raw task_success"
        )
        loop_detected = _audit_exact_bool(
            episode.get("loop_detected"), context="manual-audit raw loop_detected"
        )
        environment_failure = _audit_exact_bool(
            episode.get("environment_failure"),
            context="manual-audit raw environment_failure",
        )
        if environment_failure:
            categories_by_episode[episode_id].add("environment_failure")
        if loop_detected:
            categories_by_episode[episode_id].add("loop")

        attempts = recovery_by_episode.get(episode_id, [])
        for attempt in attempts:
            verified_failure = _audit_exact_bool(
                attempt.get("verified_failure_present", False),
                context="manual-audit raw verified_failure_present",
            )
            successful = _audit_exact_bool(
                attempt.get("successful", False),
                context="manual-audit raw recovery successful",
            )
            if not verified_failure:
                categories_by_episode[episode_id].add("unnecessary_intervention")
            elif successful:
                categories_by_episode[episode_id].add("successful_recovery")
            else:
                categories_by_episode[episode_id].add("failed_recovery")

        admitted_memory_intervention = False
        for query in queries_by_episode.get(episode_id, []):
            admitted = _audit_exact_bool(
                query.get("admitted", False),
                context="manual-audit raw memory admitted",
            )
            admitted_memory_intervention = admitted_memory_intervention or admitted
            if admitted and query.get("useful_intervention") is True:
                categories_by_episode[episode_id].add("memory_help")
            if admitted and query.get("harmful_intervention") is True:
                categories_by_episode[episode_id].add("memory_harm")

        verified_failure = False
        for incident in incidents_by_episode.get(episode_id, []):
            verified_failure = verified_failure or _audit_exact_bool(
                incident.get("verified_agent_failure", False),
                context="manual-audit raw verified_agent_failure",
            )
            descriptor = " ".join(
                str(incident.get(key, ""))
                for key in (
                    "failure_type",
                    "failure_kind",
                    "diagnosis",
                    "error_kind",
                )
            ).upper()
            if any(marker in descriptor for marker in failure_markers):
                categories_by_episode[episode_id].add(
                    "bbox_or_parameter_failure"
                )

        system_id = str(episode["system_id"])
        paired_e2_episode_id = paired_e2_by_e3.get(episode_id)
        memory_effect_applicable = bool(
            system_id == "E3"
            and paired_e2_episode_id is not None
            and admitted_memory_intervention
        )
        has_failure_evidence = bool(
            not task_success
            or verified_failure
            or loop_detected
            or environment_failure
        )
        evidence_by_episode[episode_id] = {
            "task_success": task_success,
            "loop_detected": loop_detected,
            "environment_failure": environment_failure,
            "has_verified_failure": verified_failure,
            "has_recovery_attempt": bool(attempts),
            "has_admitted_memory_intervention": admitted_memory_intervention,
            "memory_effect_applicable": memory_effect_applicable,
            "has_failure_evidence": has_failure_evidence,
            "paired_e2_episode_id": paired_e2_episode_id,
            "e2_e3_task_outcome_disagreement": episode_id in disagreements,
            "case_categories": sorted(categories_by_episode[episode_id]),
        }

    recovery_episode_ids = set(recovery_by_episode)
    verified_failure_episode_ids = {
        episode_id
        for episode_id, evidence in evidence_by_episode.items()
        if evidence["has_verified_failure"]
    }
    candidates = {
        "ordinary_success": {
            episode_id
            for episode_id, episode in by_episode.items()
            if evidence_by_episode[episode_id]["task_success"]
            and episode_id not in recovery_episode_ids
            and episode_id not in verified_failure_episode_ids
        },
        "terminal_failure": {
            episode_id
            for episode_id, evidence in evidence_by_episode.items()
            if not evidence["task_success"]
        },
        "recovery": set(recovery_episode_ids),
        "e2_e3_disagreement": set(disagreements),
    }
    return {
        "episodes_by_id": by_episode,
        "evidence_by_episode": evidence_by_episode,
        "categories_by_episode": categories_by_episode,
        "candidates": candidates,
    }


def _runtime_file_hashes_for_manual_audit(source: Path) -> dict[str, str]:
    if (
        source.is_symlink()
        or source.absolute() != source.resolve()
        or not source.is_dir()
        or source.name not in {"runtime", "primary_runtime", "paired_e2_runtime"}
    ):
        raise SchemaError("manual-audit reviewer source is not a runtime directory")
    output: dict[str, str] = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise SchemaError("manual-audit reviewer evidence contains a symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if "sealed" in {part.lower() for part in relative.parts}:
            raise SchemaError("manual-audit reviewer evidence includes a sealed path")
        if path.suffix == ".json":
            payload = _load_audit_json_without_duplicate_keys(
                path.read_text(encoding="utf-8"), context=str(path)
            )
            assert_no_verifier_evidence(
                {"packet_value": payload}, context=str(path)
            )
        elif path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    payload = _load_audit_json_without_duplicate_keys(
                        line, context=f"{path}:{line_number}"
                    )
                    assert_no_verifier_evidence(
                        {"packet_value": payload},
                        context=f"{path}:{line_number}",
                    )
        output[str(relative)] = sha256_file(path)
    if not output:
        raise SchemaError("manual-audit reviewer evidence is empty")
    return output


def _validate_manual_audit_reviewer_packet(
    *,
    root: Path,
    campaign_id: str,
    audit_id: str,
    primary_source: Path,
    primary_episode_id: str,
    primary_system_id: str,
    task_id: str,
    task_context_path: Any,
    artifact_path: Any,
    paired_e2_source: Path | None,
    paired_e2_episode_id: str | None,
    paired_e2_artifact_path: Any,
) -> None:
    packet_root = root / "manual_audit" / "reviewer_packets" / audit_id
    if packet_root.is_symlink() or not packet_root.is_dir():
        raise SchemaError("manual-audit reviewer packet is absent or symlinked")
    manifest_path = packet_root / "packet_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SchemaError("manual-audit reviewer packet manifest is absent")
    if manifest_path.stat().st_mode & 0o077:
        raise SchemaError("manual-audit reviewer packet manifest is not private")
    manifest = _read_manual_audit_json(
        manifest_path, context="manual-audit reviewer packet manifest"
    )
    _require_exact_json_object(
        manifest,
        (
            "schema_version",
            "campaign_id",
            "audit_id",
            "blinding_mode",
            "system_condition_visible",
            "official_outcome_labels_included",
            "sealed_evaluator_files_included",
            "task_context",
            "primary",
            "paired_e2",
        ),
        context="manual-audit reviewer packet manifest",
    )
    if (
        manifest["schema_version"] != "table2-manual-audit-reviewer-packet-v1"
        or manifest["campaign_id"] != campaign_id
        or manifest["audit_id"] != audit_id
        or manifest["blinding_mode"] != MANUAL_AUDIT_BLINDING_MODE
        or _audit_exact_bool(
            manifest["system_condition_visible"],
            context="manual-audit packet.system_condition_visible",
        )
        is not True
        or _audit_exact_bool(
            manifest["official_outcome_labels_included"],
            context="manual-audit packet.official_outcome_labels_included",
        )
        is not False
        or _audit_exact_bool(
            manifest["sealed_evaluator_files_included"],
            context="manual-audit packet.sealed_evaluator_files_included",
        )
        is not False
    ):
        raise SchemaError("manual-audit reviewer packet has a false blinding claim")

    task_context_binding = _require_exact_json_object(
        manifest["task_context"],
        (
            "task_id",
            "packet_relative_path",
            "sha256",
            "context_status",
            "source_task_snapshot_sha256",
            "source_task_record_sha256",
        ),
        context="manual-audit reviewer packet task_context",
    )
    expected_task_context_path = packet_root / "task_context.json"
    if (
        task_context_binding["task_id"] != task_id
        or task_context_binding["packet_relative_path"] != "task_context.json"
        or task_context_path != str(expected_task_context_path.relative_to(root))
        or expected_task_context_path.is_symlink()
        or not expected_task_context_path.is_file()
        or expected_task_context_path.stat().st_mode & 0o077
        or _audit_sha256(
            task_context_binding["sha256"],
            context="manual-audit task_context.sha256",
        )
        != sha256_file(expected_task_context_path)
    ):
        raise SchemaError("manual-audit reviewer task context binding differs")
    from .summary import _manual_audit_task_context

    expected_task_context = _manual_audit_task_context(root, task_id=task_id)
    observed_task_context = _read_manual_audit_json(
        expected_task_context_path, context="manual-audit reviewer task context"
    )
    if canonical_json_bytes(observed_task_context) != canonical_json_bytes(
        expected_task_context
    ):
        raise SchemaError("manual-audit task context differs from frozen task snapshot")
    for key in (
        "context_status",
        "source_task_snapshot_sha256",
        "source_task_record_sha256",
    ):
        if task_context_binding[key] != expected_task_context[key]:
            raise SchemaError(f"manual-audit task context binding differs: {key}")
    assert_no_verifier_evidence(
        {"task_context": observed_task_context}, context=str(expected_task_context_path)
    )

    def validate_side(
        value: Any,
        *,
        role: str,
        source: Path,
        episode_id: str,
        system_id: str,
        public_path: Any,
    ) -> None:
        side = _require_exact_json_object(
            value,
            (
                "episode_id",
                "system_id",
                "packet_relative_path",
                "source_runtime_relative_path",
                "files",
            ),
            context=f"manual-audit reviewer packet {role}",
        )
        packet_relative_name = (
            "primary_runtime" if role == "primary" else "paired_e2_runtime"
        )
        expected_packet_path = packet_root / packet_relative_name
        expected_public_path = str(expected_packet_path.relative_to(root))
        if (
            side["episode_id"] != episode_id
            or side["system_id"] != system_id
            or side["packet_relative_path"] != packet_relative_name
            or side["source_runtime_relative_path"] != str(source.relative_to(root))
            or public_path != expected_public_path
        ):
            raise SchemaError(f"manual-audit reviewer packet {role} binding differs")
        files = side["files"]
        if not isinstance(files, Mapping) or not files:
            raise SchemaError(f"manual-audit reviewer packet {role} file map is invalid")
        for relative, digest in files.items():
            if type(relative) is not str or not relative or not _is_sha256(digest):
                raise SchemaError(
                    f"manual-audit reviewer packet {role} file map is invalid"
                )
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise SchemaError("manual-audit reviewer packet file path escapes")
        source_hashes = _runtime_file_hashes_for_manual_audit(source)
        packet_hashes = _runtime_file_hashes_for_manual_audit(expected_packet_path)
        if dict(files) != source_hashes or packet_hashes != source_hashes:
            raise SchemaError(
                f"manual-audit reviewer packet {role} differs from runtime evidence"
            )
        summary = _read_manual_audit_json(
            expected_packet_path / "episode_summary.json",
            context=f"manual-audit reviewer packet {role} episode summary",
        )
        if summary.get("episode_id") != episode_id:
            raise SchemaError("manual-audit reviewer packet episode identity differs")
        assert_no_verifier_evidence(
            summary, context=str(expected_packet_path / "episode_summary.json")
        )

    validate_side(
        manifest["primary"],
        role="primary",
        source=primary_source,
        episode_id=primary_episode_id,
        system_id=primary_system_id,
        public_path=artifact_path,
    )
    if paired_e2_source is None:
        if (
            manifest["paired_e2"] is not None
            or paired_e2_episode_id is not None
            or paired_e2_artifact_path is not None
        ):
            raise SchemaError("manual-audit reviewer packet has an unexpected E2 pair")
    else:
        if paired_e2_episode_id is None:
            raise SchemaError("manual-audit reviewer packet lacks the paired E2 identity")
        validate_side(
            manifest["paired_e2"],
            role="paired_e2",
            source=paired_e2_source,
            episode_id=paired_e2_episode_id,
            system_id="E2",
            public_path=paired_e2_artifact_path,
        )
    allowed_packet_files = {
        "packet_manifest.json",
        "task_context.json",
        *(f"primary_runtime/{relative}" for relative in manifest["primary"]["files"]),
        *(
            f"paired_e2_runtime/{relative}"
            for relative in (
                manifest["paired_e2"]["files"]
                if isinstance(manifest["paired_e2"], Mapping)
                else {}
            )
        ),
    }
    actual_packet_files = {
        str(path.relative_to(packet_root))
        for path in packet_root.rglob("*")
        if path.is_file()
    }
    if actual_packet_files != allowed_packet_files:
        raise SchemaError("manual-audit reviewer packet lacks exact file closure")


def _validate_manual_audit_package(root: Path) -> None:
    campaign = read_json(root / "campaign_manifest.json")
    campaign_profile = classify_campaign_profile(
        campaign,
        context="manual-audit campaign manifest",
        require_campaign_mode=True,
    )
    campaign_id = _audit_nonempty_string(
        campaign.get("campaign_id"), context="campaign_manifest.campaign_id"
    )
    definition = _read_manual_audit_json(
        root / "frozen" / "benchmark" / "audit_manifest.json",
        context="frozen manual-audit definition",
    )
    codebook_binding = validate_manual_audit_reviewer_codebook(definition)
    if (
        definition.get("manifest_id") != MANUAL_AUDIT_MANIFEST_ID
        or definition.get("blinding_mode") != MANUAL_AUDIT_BLINDING_MODE
        or definition.get("selection_occurs_after_episode_completion") is not True
    ):
        raise SchemaError("manual audit frozen definition differs from registration")
    selection_path = root / "manual_audit" / "selection_manifest.json"
    labels_path = root / "manual_audit" / "sealed" / "selection_labels.json"
    selection = _read_manual_audit_json(
        selection_path, context="manual-audit selection manifest"
    )
    labels = _read_manual_audit_json(
        labels_path, context="manual-audit sealed selection labels"
    )
    _require_exact_json_object(
        selection,
        (
            "schema_version",
            "manifest_id",
            "campaign_id",
            "reviewer_codebook_id",
            "reviewer_codebook_sha256",
            "selection_algorithm",
            "selection_seed",
            "blinding_mode",
            "outcome_labels_location",
            "outcome_labels_sha256",
            "total_target",
            "selected_count",
            "total_shortfall",
            "strata",
            "category_coverage",
            "selected",
        ),
        context="manual audit selection",
    )
    _require_exact_json_object(
        labels,
        (
            "schema_version",
            "manifest_id",
            "campaign_id",
            "reviewer_codebook_id",
            "reviewer_codebook_sha256",
            "offline_sealed_labels",
            "labels",
        ),
        context="manual audit sealed labels",
    )
    selection_seed = _audit_exact_int(
        selection["selection_seed"],
        context="manual audit selection_seed",
        minimum=0,
    )
    _audit_sha256(
        selection["outcome_labels_sha256"],
        context="manual audit outcome_labels_sha256",
    )
    if selection["campaign_id"] != campaign_id:
        raise SchemaError("manual audit selection has the wrong campaign ID")
    if (
        selection["schema_version"] != SCHEMA_VERSION
        or selection["manifest_id"] != definition["manifest_id"]
        or selection_seed != definition["selection_seed"]
        or selection["total_target"] != definition["total_target"]
        or selection["selection_algorithm"] != "sha256-lowest-v1"
        or selection["blinding_mode"] != MANUAL_AUDIT_BLINDING_MODE
        or selection["outcome_labels_location"] != "sealed/selection_labels.json"
    ):
        raise SchemaError("manual audit selection differs from its frozen definition")
    for key, expected in codebook_binding.items():
        if selection.get(key) != expected or labels.get(key) != expected:
            raise SchemaError(f"manual audit {key} differs from its frozen definition")
    if (
        labels.get("schema_version") != SCHEMA_VERSION
        or labels.get("manifest_id") != definition["manifest_id"]
        or labels.get("campaign_id") != campaign_id
        or _audit_exact_bool(
            labels.get("offline_sealed_labels"),
            context="manual audit labels.offline_sealed_labels",
        )
        is not True
    ):
        raise SchemaError("manual audit sealed selection labels are malformed")
    if labels_path.stat().st_mode & 0o077:
        raise SchemaError("manual audit outcome labels are not private")
    if sha256_file(labels_path) != selection["outcome_labels_sha256"]:
        raise SchemaError("manual audit outcome-label hash mismatch")
    selected = selection["selected"]
    label_rows = labels["labels"]
    if not isinstance(selected, list) or not isinstance(label_rows, list):
        raise SchemaError("manual audit selected/label rows must be arrays")
    for row in selected:
        _require_exact_json_object(
            row,
            (
                "audit_id",
                "episode_id",
                "block_id",
                "task_id",
                "system_id",
                "task_context_path",
                "artifact_path",
                "paired_e2_artifact_path",
            ),
            context="manual audit selected row",
        )
    sealed_label_keys = (
        "episode_id",
        "block_id",
        "task_id",
        "system_id",
        "stratum",
        "task_success",
        "loop_detected",
        "environment_failure",
        "has_verified_failure",
        "has_recovery_attempt",
        "has_admitted_memory_intervention",
        "memory_effect_applicable",
        "has_failure_evidence",
        "paired_e2_episode_id",
        "e2_e3_task_outcome_disagreement",
        "case_categories",
        "selection_score",
        "audit_id",
    )
    for row in label_rows:
        _require_exact_json_object(
            row, sealed_label_keys, context="manual audit sealed label row"
        )
    selected_ids = [
        _audit_nonempty_string(row["audit_id"], context="manual audit audit_id")
        for row in selected
    ]
    label_ids = [
        _audit_nonempty_string(row["audit_id"], context="manual audit label audit_id")
        for row in label_rows
    ]
    total_target = _audit_exact_int(
        selection["total_target"], context="manual audit total_target", minimum=0
    )
    selected_count = _audit_exact_int(
        selection["selected_count"], context="manual audit selected_count", minimum=0
    )
    total_shortfall = _audit_exact_int(
        selection["total_shortfall"], context="manual audit total_shortfall", minimum=0
    )
    if (
        len(selected_ids) != len(set(selected_ids))
        or set(selected_ids) != set(label_ids)
        or selected_count != len(selected_ids)
        or total_shortfall != total_target - len(selected_ids)
    ):
        raise SchemaError("manual audit public/sealed selection identities disagree")
    public_episodes = {
        _audit_nonempty_string(
            row["episode_id"], context="manual audit selected episode_id"
        )
        for row in selected
    }
    if len(public_episodes) != len(selected):
        raise SchemaError("manual audit repeats an episode")
    public_by_id = {row["audit_id"]: row for row in selected}
    sealed_by_id = {row["audit_id"]: row for row in label_rows}

    raw_records = _load_selected_analysis_records_unchecked(root)
    derived = _derive_manual_audit_selection_evidence(raw_records)
    raw_by_episode = derived["episodes_by_id"]
    evidence_by_episode = derived["evidence_by_episode"]
    schedule_by_block = {
        str(row["block_id"]): row
        for row in read_jsonl(root / "schedule" / "schedule.jsonl")
    }
    for audit_id in selected_ids:
        public = public_by_id[audit_id]
        sealed_row = sealed_by_id[audit_id]
        for key in ("episode_id", "block_id", "task_id", "system_id"):
            if public[key] != sealed_row[key]:
                raise SchemaError(f"manual audit public/sealed {key} mismatch: {audit_id}")
        episode_id = sealed_row["episode_id"]
        raw_episode = raw_by_episode.get(episode_id)
        if raw_episode is None:
            raise SchemaError("manual audit selected episode is absent from raw evidence")
        for key in ("block_id", "task_id", "system_id"):
            if sealed_row[key] != raw_episode[key]:
                raise SchemaError(
                    f"manual audit sealed {key} differs from selected raw evidence"
                )
        for key, expected in evidence_by_episode[episode_id].items():
            if canonical_json_bytes(sealed_row[key]) != canonical_json_bytes(expected):
                raise SchemaError(
                    "manual audit sealed evidence differs from immutable selected "
                    f"raw evidence: {audit_id}.{key}"
                )
        schedule_row = schedule_by_block.get(sealed_row["block_id"])
        if schedule_row is None:
            raise SchemaError("manual audit block is absent from the frozen schedule")
        block_base = _block_base(root, schedule_row)
        resolution = read_json(block_base / "resolution.json")
        if resolution.get("status") != "INCLUDED":
            raise SchemaError("manual audit selected a non-included block")
        selected_attempt_id = _audit_exact_int(
            resolution.get("selected_attempt_id"),
            context="manual audit selected_attempt_id",
            minimum=0,
        )
        primary_source = (
            _system_package(
                block_base, selected_attempt_id, sealed_row["system_id"]
            )
            / "runtime"
        )
        paired_e2_episode_id = sealed_row["paired_e2_episode_id"]
        if paired_e2_episode_id is not None:
            paired_e2_episode_id = _audit_nonempty_string(
                paired_e2_episode_id,
                context="manual audit paired_e2_episode_id",
            )
        paired_e2_source = (
            _system_package(block_base, selected_attempt_id, "E2") / "runtime"
            if paired_e2_episode_id is not None
            else None
        )
        _validate_manual_audit_reviewer_packet(
            root=root,
            campaign_id=campaign_id,
            audit_id=audit_id,
            primary_source=primary_source,
            primary_episode_id=episode_id,
            primary_system_id=sealed_row["system_id"],
            task_id=sealed_row["task_id"],
            task_context_path=public["task_context_path"],
            artifact_path=public["artifact_path"],
            paired_e2_source=paired_e2_source,
            paired_e2_episode_id=paired_e2_episode_id,
            paired_e2_artifact_path=public["paired_e2_artifact_path"],
        )
    reviewer_packet_root = root / "manual_audit" / "reviewer_packets"
    packet_ids = {
        path.name for path in reviewer_packet_root.iterdir() if path.is_dir()
    }
    if packet_ids != set(selected_ids) or any(
        path.is_symlink() or not path.is_dir()
        for path in reviewer_packet_root.iterdir()
    ):
        raise SchemaError("manual-audit reviewer packet directory closure differs")

    # The aggregate CSV is deliberately not an audit authority. Sampling and
    # category coverage are replayed from the included runtime packages joined
    # with their sealed verifier streams above.
    candidates = derived["candidates"]
    priority = (
        "e2_e3_disagreement",
        "recovery",
        "terminal_failure",
        "ordinary_success",
    )
    frozen_targets = {
        str(row["name"]): _audit_exact_int(
            row["target"], context="manual audit frozen target", minimum=0
        )
        for row in definition["strata"]
    }
    expected_selected: dict[str, list[str]] = {}
    used: set[str] = set()
    for stratum in priority:
        ordered = sorted(
            candidates[stratum] - used,
            key=lambda episode_id: (
                sha256_json(
                    {
                        "algorithm": "sha256-lowest-v1",
                        "campaign_id": campaign_id,
                        "selection_seed": selection["selection_seed"],
                        "stratum": stratum,
                        "episode_id": episode_id,
                    }
                ),
                episode_id,
            ),
        )
        expected_selected[stratum] = ordered[: frozen_targets[stratum]]
        used.update(expected_selected[stratum])
    expected_presentation = sorted(
        (
            (episode_id, stratum)
            for stratum, episode_ids in expected_selected.items()
            for episode_id in episode_ids
        ),
        key=lambda value: (
            sha256_json(
                {
                    "algorithm": "sha256-lowest-v1",
                    "campaign_id": campaign_id,
                    "selection_seed": selection_seed,
                    "stratum": "audit-presentation-order",
                    "episode_id": value[0],
                }
            ),
            value[0],
        ),
    )
    if len(expected_presentation) != len(label_rows):
        raise SchemaError("manual audit selected row count differs from raw replay")
    for index, ((episode_id, stratum), sealed_row, public_row) in enumerate(
        zip(expected_presentation, label_rows, selected, strict=True), start=1
    ):
        expected_audit_id = f"audit-{index:02d}"
        if (
            sealed_row["audit_id"] != expected_audit_id
            or public_row["audit_id"] != expected_audit_id
            or sealed_row["episode_id"] != episode_id
            or public_row["episode_id"] != episode_id
            or sealed_row["stratum"] != stratum
        ):
            raise SchemaError(
                "manual audit presentation order/identity differs from raw replay"
            )
    actual_selected: dict[str, set[str]] = defaultdict(set)
    for row in label_rows:
        actual_selected[str(row.get("stratum"))].add(str(row.get("episode_id")))
    for stratum, expected in expected_selected.items():
        if actual_selected.get(stratum, set()) != set(expected):
            raise SchemaError(
                f"manual audit did not select the lowest deterministic {stratum} scores"
            )
    label_by_stratum: dict[str, int] = defaultdict(int)
    required_categories = definition.get("required_case_categories")
    if not isinstance(required_categories, list) or tuple(required_categories) != (
        MANUAL_AUDIT_CASE_CATEGORIES
    ):
        raise SchemaError("manual audit required categories differ from registration")
    selected_category_counts: dict[str, int] = defaultdict(int)
    for row in label_rows:
        for key in (
            "task_success",
            "loop_detected",
            "environment_failure",
            "has_verified_failure",
            "has_recovery_attempt",
            "has_admitted_memory_intervention",
            "memory_effect_applicable",
            "has_failure_evidence",
            "e2_e3_task_outcome_disagreement",
        ):
            _audit_exact_bool(row[key], context=f"manual audit label {key}")
        stratum = _audit_nonempty_string(
            row["stratum"], context="manual audit label stratum"
        )
        label_by_stratum[stratum] += 1
        case_categories = row["case_categories"]
        if (
            not isinstance(case_categories, list)
            or not all(type(value) is str for value in case_categories)
            or case_categories != sorted(set(case_categories))
            or not set(case_categories).issubset(MANUAL_AUDIT_CASE_CATEGORIES)
        ):
            raise SchemaError("manual audit case_categories are invalid")
        for category in case_categories:
            selected_category_counts[str(category)] += 1
        expected_score = sha256_json(
            {
                "algorithm": "sha256-lowest-v1",
                "campaign_id": campaign_id,
                "selection_seed": selection["selection_seed"],
                "stratum": stratum,
                "episode_id": str(row["episode_id"]),
            }
        )
        if row.get("selection_score") != expected_score:
            raise SchemaError("manual audit deterministic selection score mismatch")
    if not isinstance(selection["strata"], list):
        raise SchemaError("manual audit strata must be an array")
    expected_strata_order = [str(row["name"]) for row in definition["strata"]]
    if [row.get("name") for row in selection["strata"]] != expected_strata_order:
        raise SchemaError("manual audit strata order differs from registration")
    for row in selection["strata"]:
        _require_exact_json_object(
            row,
            (
                "name",
                "target",
                "eligible_count_before_cross_stratum_deduplication",
                "selected_count",
                "shortfall",
                "substitution_allowed",
            ),
            context="manual audit stratum row",
        )
        name = _audit_nonempty_string(row["name"], context="manual audit stratum")
        target = frozen_targets.get(name)
        if target is None or _audit_exact_int(
            row["target"], context="manual audit target", minimum=0
        ) != target:
            raise SchemaError("manual audit target differs from frozen stratum")
        stratum_selected_count = _audit_exact_int(
            row["selected_count"],
            context="manual audit stratum selected_count",
            minimum=0,
        )
        if (
            _audit_exact_int(
                row["eligible_count_before_cross_stratum_deduplication"],
                context="manual audit stratum eligible_count",
                minimum=0,
            )
            != len(candidates[name])
            or stratum_selected_count != label_by_stratum.get(name, 0)
            or _audit_exact_int(
                row["shortfall"], context="manual audit stratum shortfall", minimum=0
            )
            != target - stratum_selected_count
            or _audit_exact_bool(
                row["substitution_allowed"],
                context="manual audit stratum substitution_allowed",
            )
            is not False
        ):
            raise SchemaError("manual audit shortfall/substitution record is invalid")
        if name != "e2_e3_disagreement" and stratum_selected_count != target:
            raise SchemaError(
                f"manual audit requires five {name} cases; observed {stratum_selected_count}"
            )
    coverage = selection["category_coverage"]
    if not isinstance(coverage, list) or [row.get("name") for row in coverage] != list(
        MANUAL_AUDIT_CASE_CATEGORIES
    ):
        raise SchemaError("manual audit category coverage has the wrong registered closure")
    final_campaign = campaign_profile == CAMPAIGN_PROFILE_FINAL
    for row in coverage:
        _require_exact_json_object(
            row,
            (
                "name",
                "eligible_count",
                "selected_count",
                "status",
                "substitution_allowed",
            ),
            context="manual audit category coverage row",
        )
        category = _audit_nonempty_string(
            row["name"], context="manual audit coverage category"
        )
        eligible = _audit_exact_int(
            row["eligible_count"], context="manual audit eligible_count", minimum=0
        )
        category_selected_count = _audit_exact_int(
            row["selected_count"],
            context="manual audit category selected_count",
            minimum=0,
        )
        expected_eligible = sum(
            category in categories
            for categories in derived["categories_by_episode"].values()
        )
        if (
            eligible != expected_eligible
            or category_selected_count > eligible
            or category_selected_count
            != selected_category_counts.get(category, 0)
            or _audit_exact_bool(
                row["substitution_allowed"],
                context="manual audit coverage substitution_allowed",
            )
            is not False
        ):
            raise SchemaError(f"manual audit category coverage is invalid: {category}")
        expected_status = (
            "NOT_APPLICABLE"
            if eligible == 0
            else "COVERED"
            if category_selected_count > 0
            else "SHORTFALL_NO_SUBSTITUTION"
        )
        if row.get("status") != expected_status:
            raise SchemaError(f"manual audit category status is invalid: {category}")
        if final_campaign and expected_status == "SHORTFALL_NO_SUBSTITUTION":
            raise SchemaError(
                f"locked-final manual audit missed an available required case: {category}"
            )
    if final_campaign:
        validate_manual_adjudication_completion(root)


def validate_manual_adjudication_completion(
    campaign_dir: str | Path,
) -> dict[str, Any]:
    """Validate hash-bound, human-attested review required for final evidence."""

    root = Path(campaign_dir).resolve()
    campaign = read_json(root / "campaign_manifest.json")
    classify_campaign_profile(
        campaign,
        context="manual-adjudication campaign manifest",
        require_campaign_mode=True,
    )
    campaign_id = _audit_nonempty_string(
        campaign.get("campaign_id"), context="campaign_manifest.campaign_id"
    )
    audit_definition = _read_manual_audit_json(
        root / "frozen" / "benchmark" / "audit_manifest.json",
        context="frozen human-audit definition",
    )
    codebook_binding = validate_manual_audit_reviewer_codebook(audit_definition)
    if (
        audit_definition.get("manifest_id") != MANUAL_AUDIT_MANIFEST_ID
        or audit_definition.get("blinding_mode") != MANUAL_AUDIT_BLINDING_MODE
    ):
        raise SchemaError("human audit frozen definition differs from registration")
    selection_path = root / "manual_audit" / "selection_manifest.json"
    labels_path = root / "manual_audit" / "sealed" / "selection_labels.json"
    adjudication_path = root / "manual_audit" / "sealed" / "adjudication.json"
    agreement_path = root / "manual_audit" / "agreement.json"
    completion_path = root / "manual_audit" / "completion_manifest.json"
    for path in (
        selection_path,
        labels_path,
        adjudication_path,
        agreement_path,
        completion_path,
    ):
        if path.is_symlink() or not path.is_file():
            raise SchemaError(f"manual-audit artifact is absent or symlinked: {path}")
    selection = _read_manual_audit_json(
        selection_path, context="human adjudication selection"
    )
    labels = _read_manual_audit_json(
        labels_path, context="human adjudication sealed selection labels"
    )
    adjudication = _read_manual_audit_json(
        adjudication_path, context="manual-audit adjudication"
    )
    agreement = _read_manual_audit_json(
        agreement_path, context="manual-audit agreement"
    )
    completion = _read_manual_audit_json(
        completion_path, context="manual-audit completion"
    )

    _require_exact_json_object(
        selection,
        (
            "schema_version",
            "manifest_id",
            "campaign_id",
            "reviewer_codebook_id",
            "reviewer_codebook_sha256",
            "selection_algorithm",
            "selection_seed",
            "blinding_mode",
            "outcome_labels_location",
            "outcome_labels_sha256",
            "total_target",
            "selected_count",
            "total_shortfall",
            "strata",
            "category_coverage",
            "selected",
        ),
        context="human adjudication selection",
    )
    _require_exact_json_object(
        labels,
        (
            "schema_version",
            "manifest_id",
            "campaign_id",
            "reviewer_codebook_id",
            "reviewer_codebook_sha256",
            "offline_sealed_labels",
            "labels",
        ),
        context="human adjudication sealed selection labels",
    )
    selection_hash = sha256_file(selection_path)
    if selection["campaign_id"] != campaign_id:
        raise SchemaError("human adjudication selection has the wrong campaign ID")
    if (
        selection["schema_version"] != SCHEMA_VERSION
        or selection["manifest_id"] != audit_definition.get("manifest_id")
        or selection["blinding_mode"] != MANUAL_AUDIT_BLINDING_MODE
    ):
        raise SchemaError("human adjudication selection has the wrong audit manifest ID")
    if (
        labels["schema_version"] != SCHEMA_VERSION
        or labels["manifest_id"] != audit_definition.get("manifest_id")
        or labels["campaign_id"] != campaign_id
        or _audit_exact_bool(
            labels["offline_sealed_labels"],
            context="human adjudication labels.offline_sealed_labels",
        )
        is not True
        or selection["outcome_labels_location"] != "sealed/selection_labels.json"
        or sha256_file(labels_path)
        != _audit_sha256(
            selection["outcome_labels_sha256"],
            context="human adjudication outcome_labels_sha256",
        )
    ):
        raise SchemaError("human adjudication sealed selection labels are invalid")
    for key, expected in codebook_binding.items():
        if selection.get(key) != expected or labels.get(key) != expected:
            raise SchemaError(f"human adjudication selection has the wrong {key}")
    selected_rows = selection["selected"]
    label_rows = labels["labels"]
    if not isinstance(selected_rows, list) or not isinstance(label_rows, list):
        raise SchemaError("human adjudication selection rows must be arrays")
    for row in selected_rows:
        _require_exact_json_object(
            row,
            (
                "audit_id",
                "episode_id",
                "block_id",
                "task_id",
                "system_id",
                "task_context_path",
                "artifact_path",
                "paired_e2_artifact_path",
            ),
            context="human adjudication selected row",
        )
        for key in (
            "audit_id",
            "episode_id",
            "block_id",
            "task_id",
            "system_id",
            "task_context_path",
            "artifact_path",
        ):
            _audit_nonempty_string(
                row[key], context=f"human adjudication selected row.{key}"
            )
        paired_path = row["paired_e2_artifact_path"]
        if paired_path is not None:
            _audit_nonempty_string(
                paired_path,
                context="human adjudication selected row.paired_e2_artifact_path",
            )
    sealed_label_keys = (
        "episode_id",
        "block_id",
        "task_id",
        "system_id",
        "stratum",
        "task_success",
        "loop_detected",
        "environment_failure",
        "has_verified_failure",
        "has_recovery_attempt",
        "has_admitted_memory_intervention",
        "memory_effect_applicable",
        "has_failure_evidence",
        "paired_e2_episode_id",
        "e2_e3_task_outcome_disagreement",
        "case_categories",
        "selection_score",
        "audit_id",
    )
    for row in label_rows:
        _require_exact_json_object(
            row,
            sealed_label_keys,
            context="human adjudication sealed selection label row",
        )
        for key in (
            "episode_id",
            "block_id",
            "task_id",
            "system_id",
            "stratum",
            "audit_id",
        ):
            _audit_nonempty_string(
                row[key],
                context=f"human adjudication sealed selection label row.{key}",
            )
        for key in (
            "task_success",
            "loop_detected",
            "environment_failure",
            "has_verified_failure",
            "has_recovery_attempt",
            "has_admitted_memory_intervention",
            "memory_effect_applicable",
            "has_failure_evidence",
            "e2_e3_task_outcome_disagreement",
        ):
            _audit_exact_bool(
                row[key],
                context=f"human adjudication sealed selection label row.{key}",
            )
        paired_episode = row["paired_e2_episode_id"]
        if paired_episode is not None:
            _audit_nonempty_string(
                paired_episode,
                context=(
                    "human adjudication sealed selection label row."
                    "paired_e2_episode_id"
                ),
            )
        case_categories = row["case_categories"]
        if (
            not isinstance(case_categories, list)
            or not all(type(value) is str for value in case_categories)
            or case_categories != sorted(set(case_categories))
            or not set(case_categories).issubset(MANUAL_AUDIT_CASE_CATEGORIES)
        ):
            raise SchemaError(
                "human adjudication sealed selection case categories are invalid"
            )
        _audit_sha256(
            row["selection_score"],
            context="human adjudication sealed selection label row.selection_score",
        )
    selected_ids = {
        _audit_nonempty_string(row["audit_id"], context="human audit audit_id")
        for row in selected_rows
    }
    sealed_by_audit_id = {
        _audit_nonempty_string(row["audit_id"], context="human audit label audit_id"): row
        for row in label_rows
    }
    selected_count = _audit_exact_int(
        selection["selected_count"],
        context="human adjudication selected_count",
        minimum=1,
    )
    total_target = _audit_exact_int(
        selection["total_target"],
        context="human adjudication total_target",
        minimum=1,
    )
    total_shortfall = _audit_exact_int(
        selection["total_shortfall"],
        context="human adjudication total_shortfall",
        minimum=0,
    )
    _audit_exact_int(
        selection["selection_seed"],
        context="human adjudication selection_seed",
        minimum=0,
    )
    if (
        not selected_ids
        or len(selected_ids) != selected_count
        or len(sealed_by_audit_id) != selected_count
        or set(sealed_by_audit_id) != selected_ids
        or total_target != selected_count + total_shortfall
        or not isinstance(selection["strata"], list)
        or not isinstance(selection["category_coverage"], list)
    ):
        raise SchemaError("human adjudication requires a valid nonempty audit selection")

    _require_exact_json_object(
        adjudication,
        (
            "schema_version",
            "campaign_id",
            "selection_manifest_sha256",
            "reviewer_codebook_id",
            "reviewer_codebook_sha256",
            "status",
            "all_selected_items_reviewed",
            "all_disagreements_resolved",
            "reviewer_ids",
            "adjudicator_id",
            "attested_by",
            "attested_at_utc",
            "records",
        ),
        context="manual-audit adjudication",
    )
    _require_exact_json_object(
        agreement,
        (
            "schema_version",
            "campaign_id",
            "selection_manifest_sha256",
            "reviewer_codebook_id",
            "reviewer_codebook_sha256",
            "status",
            "reviewer_ids",
            "sample_size",
            "disagreement_count",
            "adjudicated_disagreement_count",
            "raw_agreement",
            "cohen_kappa",
            "cohen_kappa_status",
            "cohen_kappa_undefined_reason",
            "calculation_method",
            "composite_exact_agreement_count",
            "composite_exact_agreement",
            "composite_exact_agreement_method",
            "per_field_agreement_method",
            "per_field_agreement",
            "final_vs_sealed_comparison_method",
            "final_vs_sealed_comparison",
            "attested_by",
            "attested_at_utc",
        ),
        context="manual-audit agreement",
    )
    _require_exact_json_object(
        completion,
        (
            "schema_version",
            "campaign_id",
            "selection_manifest_sha256",
            "reviewer_codebook_id",
            "reviewer_codebook_sha256",
            "completion_status",
            "adjudication_relative_path",
            "adjudication_sha256",
            "agreement_relative_path",
            "agreement_sha256",
            "attested_by",
            "attested_at_utc",
        ),
        context="manual-audit completion",
    )

    for label, value in (
        ("adjudication", adjudication),
        ("agreement", agreement),
        ("completion", completion),
    ):
        if value.get("schema_version") != SCHEMA_VERSION:
            raise SchemaError(f"manual-audit {label} schema version is invalid")
        if value.get("campaign_id") != campaign_id:
            raise SchemaError(f"manual-audit {label} has the wrong campaign ID")
        if _audit_sha256(
            value.get("selection_manifest_sha256"),
            context=f"manual-audit {label}.selection_manifest_sha256",
        ) != selection_hash:
            raise SchemaError(f"manual-audit {label} is not bound to the selection")
        for key, expected in codebook_binding.items():
            if value.get(key) != expected:
                raise SchemaError(
                    f"manual-audit {label} is not bound to the reviewer codebook: {key}"
                )

    if labels_path.stat().st_mode & 0o077 or adjudication_path.stat().st_mode & 0o077:
        raise SchemaError("human adjudication labels are not private")
    if adjudication.get("status") != "COMPLETE":
        raise SchemaError("human adjudication is not complete")
    if _audit_exact_bool(
        adjudication.get("all_selected_items_reviewed"),
        context="adjudication.all_selected_items_reviewed",
    ) is not True:
        raise SchemaError("human adjudication has unreviewed selected items")
    if _audit_exact_bool(
        adjudication.get("all_disagreements_resolved"),
        context="adjudication.all_disagreements_resolved",
    ) is not True:
        raise SchemaError("human adjudication has unresolved disagreements")

    reviewer_vector = _two_reviewer_identity_vector(
        adjudication.get("reviewer_ids"), "adjudication.reviewer_ids"
    )
    reviewer_ids = set(reviewer_vector)
    adjudicator_id = _audit_nonempty_string(
        adjudication.get("adjudicator_id"),
        context="adjudication.adjudicator_id",
    )
    adjudication_attestors = _attested_identity_list(
        adjudication.get("attested_by"), "adjudication.attested_by", minimum=3
    )
    if (
        not set(reviewer_ids).issubset(adjudication_attestors)
        or adjudicator_id not in adjudication_attestors
    ):
        raise SchemaError(
            "human adjudication is not attested by all reviewers/adjudicator"
        )
    _validate_attested_at(
        adjudication.get("attested_at_utc"), "adjudication.attested_at_utc"
    )

    records = adjudication.get("records")
    if not isinstance(records, list) or len(records) != selected_count:
        raise SchemaError("human adjudication does not cover every selected item")
    record_ids: set[str] = set()
    disagreement_count = 0
    reviewer_labels_by_id: dict[str, list[tuple[str, ...]]] = {
        reviewer_id: [] for reviewer_id in reviewer_vector
    }
    final_labels_by_id: dict[str, Mapping[str, Any]] = {}
    for row in sorted(records, key=lambda value: str(value.get("audit_id", ""))):
        _require_exact_json_object(
            row,
            (
                "audit_id",
                "reviewer_labels",
                "final_label",
                "disagreement",
                "resolved",
            ),
            context="human adjudication record",
        )
        audit_id = _audit_nonempty_string(
            row["audit_id"], context="human adjudication record.audit_id"
        )
        if audit_id in record_ids:
            raise SchemaError("human adjudication audit IDs are empty/duplicate")
        record_ids.add(audit_id)
        if _audit_exact_bool(
            row["resolved"], context="adjudication.record.resolved"
        ) is not True:
            raise SchemaError(f"human adjudication item is unresolved: {audit_id}")
        reviewer_labels = row["reviewer_labels"]
        final_label = row["final_label"]
        if (
            not isinstance(reviewer_labels, Mapping)
            or set(reviewer_labels) != set(reviewer_ids)
        ):
            raise SchemaError(f"human adjudication labels are incomplete: {audit_id}")
        selected_evidence = sealed_by_audit_id.get(audit_id)
        if selected_evidence is None:
            raise SchemaError("human adjudication record is absent from selected evidence")
        vectors = {
            reviewer_id: _manual_audit_label_vector(
                reviewer_labels[reviewer_id],
                audit_definition=audit_definition,
                context=f"adjudication.record[{audit_id}].reviewer_labels[{reviewer_id}]",
            )
            for reviewer_id in reviewer_vector
        }
        for reviewer_id in reviewer_vector:
            _validate_manual_audit_label_applicability(
                reviewer_labels[reviewer_id],
                selected_evidence=selected_evidence,
                context=(
                    f"adjudication.record[{audit_id}].reviewer_labels[{reviewer_id}]"
                ),
            )
        final_vector = _manual_audit_label_vector(
            final_label,
            audit_definition=audit_definition,
            context=f"adjudication.record[{audit_id}].final_label",
        )
        _validate_manual_audit_label_applicability(
            final_label,
            selected_evidence=selected_evidence,
            context=f"adjudication.record[{audit_id}].final_label",
        )
        final_labels_by_id[audit_id] = final_label
        computed_disagreement = vectors[reviewer_vector[0]] != vectors[reviewer_vector[1]]
        supplied_disagreement = _audit_exact_bool(
            row["disagreement"], context="adjudication.record.disagreement"
        )
        if supplied_disagreement is not computed_disagreement:
            raise SchemaError(
                f"human adjudication disagreement differs from reviewer labels: {audit_id}"
            )
        if not computed_disagreement and final_vector != vectors[reviewer_vector[0]]:
            raise SchemaError(
                f"human adjudication changed an agreed reviewer label: {audit_id}"
            )
        disagreement_count += int(computed_disagreement)
        for reviewer_id in reviewer_vector:
            reviewer_labels_by_id[reviewer_id].append(vectors[reviewer_id])
    if record_ids != selected_ids:
        raise SchemaError(
            "human adjudication IDs differ from the outcome-label-hidden selection"
        )
    if agreement.get("status") != "COMPLETE":
        raise SchemaError("manual-audit agreement calculation is not complete")
    agreement_reviewers = _two_reviewer_identity_vector(
        agreement.get("reviewer_ids"), "agreement.reviewer_ids"
    )
    if agreement_reviewers != reviewer_vector:
        raise SchemaError(
            "agreement reviewer vector/order differs from adjudication reviewers"
        )
    if _audit_exact_int(
        agreement["sample_size"], context="agreement.sample_size", minimum=1
    ) != selected_count:
        raise SchemaError("agreement sample size differs from the audit selection")
    if (
        _audit_exact_int(
            agreement["disagreement_count"],
            context="agreement.disagreement_count",
            minimum=0,
        )
        != disagreement_count
        or _audit_exact_int(
            agreement["adjudicated_disagreement_count"],
            context="agreement.adjudicated_disagreement_count",
            minimum=0,
        )
        != disagreement_count
    ):
        raise SchemaError("agreement disagreement counts are not fully adjudicated")
    recomputed_agreement = _recompute_manual_audit_agreement(
        reviewer_labels_by_id[reviewer_vector[0]],
        reviewer_labels_by_id[reviewer_vector[1]],
    )
    recomputed_final_vs_sealed = _recompute_manual_audit_final_vs_sealed(
        final_labels_by_id,
        sealed_by_audit_id,
    )
    raw_agreement = _bounded_number(
        agreement.get("raw_agreement"),
        "agreement.raw_agreement",
        lower=0.0,
        upper=1.0,
    )
    expected_raw_agreement = recomputed_agreement["raw_agreement"]
    if not math.isclose(raw_agreement, expected_raw_agreement, rel_tol=0.0, abs_tol=1e-12):
        raise SchemaError("raw agreement differs from recomputed reviewer vectors")
    if agreement.get("calculation_method") != recomputed_agreement["calculation_method"]:
        raise SchemaError("manual-audit agreement calculation method is not registered")
    if agreement.get("cohen_kappa_status") != recomputed_agreement["cohen_kappa_status"]:
        raise SchemaError("Cohen's kappa status differs from recomputed reviewer vectors")
    if (
        agreement.get("cohen_kappa_undefined_reason")
        != recomputed_agreement["cohen_kappa_undefined_reason"]
    ):
        raise SchemaError("Cohen's kappa undefined reason is not the registered result")
    expected_kappa = recomputed_agreement["cohen_kappa"]
    supplied_kappa = agreement.get("cohen_kappa")
    if expected_kappa is None:
        if supplied_kappa is not None:
            raise SchemaError("undefined Cohen's kappa must be null")
    else:
        observed_kappa = _bounded_number(
            supplied_kappa, "agreement.cohen_kappa", lower=-1.0, upper=1.0
        )
        if not math.isclose(
            observed_kappa, expected_kappa, rel_tol=0.0, abs_tol=1e-12
        ):
            raise SchemaError("Cohen's kappa differs from recomputed reviewer vectors")
    for key in (
        "composite_exact_agreement_count",
        "composite_exact_agreement",
        "composite_exact_agreement_method",
        "per_field_agreement_method",
        "per_field_agreement",
    ):
        if canonical_json_bytes(agreement[key]) != canonical_json_bytes(
            recomputed_agreement[key]
        ):
            raise SchemaError(
                f"manual-audit agreement {key} differs from reviewer-vector replay"
            )
    for key in (
        "final_vs_sealed_comparison_method",
        "final_vs_sealed_comparison",
    ):
        if canonical_json_bytes(agreement[key]) != canonical_json_bytes(
            recomputed_final_vs_sealed[key]
        ):
            raise SchemaError(
                "manual-audit agreement "
                f"{key} differs from adjudicated-vs-sealed replay"
            )
    per_field_rows = agreement["per_field_agreement"]
    if not isinstance(per_field_rows, list):
        raise SchemaError("manual-audit per-field agreement must be an array")
    for row in per_field_rows:
        _require_exact_json_object(
            row,
            (
                "field",
                "sample_size",
                "agreement_count",
                "raw_agreement",
                "cohen_kappa",
                "cohen_kappa_status",
                "cohen_kappa_undefined_reason",
                "calculation_method",
            ),
            context="manual-audit per-field agreement row",
        )
    comparison_rows = agreement["final_vs_sealed_comparison"]
    if (
        not isinstance(comparison_rows, list)
        or [row.get("field") for row in comparison_rows]
        != ["task_outcome", "recovery_outcome", "memory_effect"]
    ):
        raise SchemaError(
            "manual-audit final-vs-sealed comparison has the wrong field closure"
        )
    for row in comparison_rows:
        _require_exact_json_object(
            row,
            (
                "field",
                "eligible_count",
                "uniquely_mappable_count",
                "comparable_count",
                "agreement_count",
                "disagreement_count",
                "human_undeterminable_count",
                "not_uniquely_mappable_count",
            ),
            context="manual-audit final-vs-sealed comparison row",
        )
        for key in (
            "eligible_count",
            "uniquely_mappable_count",
            "comparable_count",
            "agreement_count",
            "disagreement_count",
            "human_undeterminable_count",
            "not_uniquely_mappable_count",
        ):
            _audit_exact_int(
                row[key],
                context=f"manual-audit final-vs-sealed {row['field']}.{key}",
                minimum=0,
            )
    agreement_attestors = _attested_identity_list(
        agreement.get("attested_by"), "agreement.attested_by", minimum=2
    )
    if not set(reviewer_ids).issubset(agreement_attestors):
        raise SchemaError("agreement output is not attested by every reviewer")
    _validate_attested_at(
        agreement.get("attested_at_utc"), "agreement.attested_at_utc"
    )

    expected_completion = {
        "completion_status": "HUMAN_ADJUDICATION_COMPLETE",
        "adjudication_relative_path": "sealed/adjudication.json",
        "adjudication_sha256": sha256_file(adjudication_path),
        "agreement_relative_path": "agreement.json",
        "agreement_sha256": sha256_file(agreement_path),
    }
    _audit_sha256(
        completion["adjudication_sha256"],
        context="completion.adjudication_sha256",
    )
    _audit_sha256(
        completion["agreement_sha256"], context="completion.agreement_sha256"
    )
    for key, expected in expected_completion.items():
        if completion.get(key) != expected:
            raise SchemaError(f"manual-audit completion binding mismatch: {key}")
    completion_attestors = _attested_identity_list(
        completion.get("attested_by"), "completion.attested_by", minimum=3
    )
    required_attestors = set(reviewer_ids) | {adjudicator_id}
    if not required_attestors.issubset(completion_attestors):
        raise SchemaError("manual-audit completion lacks all human attestations")
    _validate_attested_at(
        completion.get("attested_at_utc"), "completion.attested_at_utc"
    )
    return {
        "campaign_id": campaign_id,
        "selection_manifest_sha256": selection_hash,
        **codebook_binding,
        "selected_count": selected_count,
        "reviewer_count": len(reviewer_ids),
        "disagreement_count": disagreement_count,
        **recomputed_agreement,
        **recomputed_final_vs_sealed,
        "completion_manifest_sha256": sha256_file(completion_path),
    }


def adjudication_gated_pilot_publication_status(
    campaign_dir: str | Path,
    *,
    campaign_complete: bool | None = None,
) -> str:
    """Return the pilot label only after terminal completion and audit validation.

    The campaign runner supplies its just-recomputed terminal state because its
    completion artifact does not exist until after this decision.  All other
    callers must present a self-consistent terminal ``completion.json``.
    """

    root = Path(campaign_dir).resolve()
    if campaign_complete is None:
        campaign_complete = _terminal_campaign_completion_artifact(root)
    elif type(campaign_complete) is not bool:
        raise TypeError("campaign_complete must be an exact boolean or None")
    if not campaign_complete:
        return DRAFT_PILOT_STATUS
    try:
        validate_manual_adjudication_completion(root)
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        KeyError,
        SchemaError,
        Table2Error,
    ):
        return DRAFT_PILOT_STATUS
    return PILOT_EVIDENCE_LABEL


def _terminal_campaign_completion_artifact(root: Path) -> bool:
    """Return whether the on-disk completion claim is internally terminal."""

    path = root / "completion.json"
    if path.is_symlink() or not path.is_file():
        return False
    try:
        campaign = _read_manual_audit_json(
            root / "campaign_manifest.json",
            context="campaign completion manifest",
        )
        if (
            classify_campaign_profile(
                campaign,
                context="campaign completion manifest",
                require_campaign_mode=True,
            )
            != CAMPAIGN_PROFILE_PILOT
        ):
            return False
        completion = _read_manual_audit_json(
            path,
            context="campaign completion artifact",
        )
        _require_exact_json_object(
            completion,
            CAMPAIGN_COMPLETION_KEYS,
            context="campaign completion artifact",
        )
        scheduled = _completion_nonnegative_int(
            completion["scheduled_block_count"], "scheduled_block_count"
        )
        processed = _completion_nonnegative_int(
            completion["processed_block_count"], "processed_block_count"
        )
        included = _completion_nonnegative_int(
            completion["included_block_count"], "included_block_count"
        )
        excluded = _completion_nonnegative_int(
            completion["infrastructure_excluded_block_count"],
            "infrastructure_excluded_block_count",
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError, SchemaError):
        return False
    return (
        completion["schema_version"] == SCHEMA_VERSION
        and completion["campaign_id"] == campaign["campaign_id"]
        and completion["status"] == "COMPLETE"
        and scheduled > 0
        and processed == scheduled
        and included + excluded == scheduled
    )


def _completion_nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise SchemaError(
            f"campaign completion artifact {field} must be a nonnegative integer"
        )
    return value


def _attested_identity_list(value: Any, context: str, *, minimum: int) -> set[str]:
    if not isinstance(value, list) or not all(type(item) is str for item in value):
        raise SchemaError(f"{context} must be an identity list")
    identities = set(value)
    if (
        "" in identities
        or any(item != item.strip() for item in identities)
        or len(identities) != len(value)
        or len(identities) < minimum
    ):
        raise SchemaError(f"{context} lacks distinct nonempty attestors")
    return identities


def _two_reviewer_identity_vector(value: Any, context: str) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2 or not all(
        type(item) is str and item and item == item.strip() for item in value
    ):
        raise SchemaError(f"{context} must contain exactly two reviewer identities")
    identities = tuple(value)
    if identities[0] == identities[1]:
        raise SchemaError(f"{context} must contain two distinct nonempty reviewers")
    return identities


def _validate_attested_at(value: Any, context: str) -> None:
    text = _audit_nonempty_string(value, context=context)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaError(f"{context} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise SchemaError(f"{context} must include a timezone")


def _bounded_number(value: Any, context: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{context} must be a JSON number")
    number = float(value)
    if not math.isfinite(number) or not lower <= number <= upper:
        raise SchemaError(f"{context} must be finite in [{lower}, {upper}]")
    return number


def _validate_results_package(root: Path, publication_status: str) -> None:
    manifest = read_json(root / "aggregate" / "results_manifest.json")
    required_names = {
        "pilot_summary.json",
        "table2_main.csv",
        "table2_companion.csv",
        "metrics.csv",
        "metrics.json",
        "paired_contrasts.csv",
        "retrieval_metrics.csv",
        "statistics.json",
        "provenance.json",
    }
    if manifest.get("raw_evidence_exported") is not False or set(
        manifest.get("files", {})
    ) != required_names:
        raise SchemaError("redacted result manifest has wrong closure")
    results_dir = Path(str(manifest.get("results_dir", "")))
    if not results_dir.is_absolute():
        raise SchemaError("results manifest must bind an exact absolute output directory")
    for name, expected in manifest["files"].items():
        path = results_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise SchemaError(f"redacted result artifact hash mismatch: {name}")
    summary = read_json(results_dir / "pilot_summary.json")
    provenance = read_json(results_dir / "provenance.json")
    statistics = read_json(results_dir / "statistics.json")
    if (
        summary.get("publication_status") != publication_status
        or provenance.get("publication_status") != publication_status
        or statistics.get("publication_status") != publication_status
        or provenance.get("contains_raw_oracle_evidence") is not False
    ):
        raise SchemaError("redacted result publication/provenance guard mismatch")
    contrasts = statistics.get("contrasts")
    if not isinstance(contrasts, Mapping):
        raise SchemaError("redacted statistics lacks registered contrasts")
    for contrast_name, metrics in contrasts.items():
        if not isinstance(metrics, Mapping):
            raise SchemaError(
                f"redacted statistics contrast is malformed: {contrast_name}"
            )
        task_success = metrics.get("task_success")
        significance = (
            task_success.get("task_clustered_significance")
            if isinstance(task_success, Mapping)
            else None
        )
        if not isinstance(significance, Mapping):
            raise SchemaError(
                f"redacted statistics lacks task significance: {contrast_name}"
            )
        if (
            "task_effects" in significance
            or significance.get("per_task_effects_exported") is not False
        ):
            raise SchemaError(
                "redacted statistics exposes per-task effect identities"
            )


def _validate_aggregate_hashes(root: Path) -> None:
    aggregate = root / "aggregate"
    manifest = read_json(aggregate / "artifact_hashes.json")
    if manifest.get("hash_algorithm") != "sha256" or not isinstance(
        manifest.get("files"), Mapping
    ):
        raise SchemaError("aggregate artifact hash manifest is invalid")
    actual = {
        str(path.relative_to(aggregate)): sha256_file(path)
        for path in sorted(aggregate.rglob("*"))
        if path.is_file()
        and path.name not in {"artifact_hashes.json", "validation_report.json"}
    }
    if dict(manifest["files"]) != actual:
        raise SchemaError("aggregate artifact hash manifest lacks exact closure")


def _validate_campaign_evidence_manifest(root: Path) -> None:
    manifest = read_json(root / "campaign_evidence_manifest.json")
    expected_bindings = {
        "campaign_manifest_sha256": sha256_file(root / "campaign_manifest.json"),
        "frozen_artifact_hashes_sha256": sha256_file(root / "artifact_hashes.json"),
        "access_ledger_sha256": sha256_file(root / "access_ledger.jsonl"),
        "deviation_ledger_sha256": sha256_file(root / "deviation_ledger.jsonl"),
    }
    for key, expected in expected_bindings.items():
        if manifest.get(key) != expected:
            raise SchemaError(f"campaign evidence binding mismatch: {key}")
    actual: dict[str, str] = {}
    for directory_name in (
        "paired_blocks",
        "manual_audit",
        "runtime_readiness",
        "aggregate",
    ):
        for path in sorted((root / directory_name).rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(root))
            if relative == "aggregate/validation_report.json":
                continue
            actual[relative] = sha256_file(path)
    if (root / "completion.json").is_file():
        actual["completion.json"] = sha256_file(root / "completion.json")
    if not isinstance(manifest.get("files"), Mapping) or dict(manifest["files"]) != actual:
        raise SchemaError("campaign evidence manifest lacks exact post-run closure")


def _publication_status(
    root: Path,
    manifest: Mapping[str, Any],
    report: ValidationReport,
    require_complete: bool,
    *,
    campaign_complete: bool | None = None,
) -> str:
    campaign_profile = classify_campaign_profile(
        manifest,
        context="publication campaign manifest",
        require_campaign_mode=True,
    )
    if campaign_profile == CAMPAIGN_PROFILE_PILOT:
        if campaign_complete is None:
            scheduled = report.counts.get("scheduled_blocks")
            included = report.counts.get("included_blocks")
            excluded = report.counts.get("infrastructure_excluded_blocks")
            campaign_complete = (
                report.passed
                and type(scheduled) is int
                and scheduled > 0
                and type(included) is int
                and type(excluded) is int
                and included + excluded == scheduled
                and _terminal_campaign_completion_artifact(root)
            )
        elif type(campaign_complete) is not bool:
            raise TypeError("campaign_complete must be an exact boolean or None")
        if not report.passed or not campaign_complete:
            return DRAFT_PILOT_STATUS
        metrics_path = root / "aggregate" / "metrics.json"
        if metrics_path.is_file():
            status = str(read_json(metrics_path).get("publication_status", ""))
            if status in {DRAFT_PILOT_STATUS, "PILOT_ONLY"}:
                if status == DRAFT_PILOT_STATUS:
                    return status
                return adjudication_gated_pilot_publication_status(
                    root, campaign_complete=True
                )
        return adjudication_gated_pilot_publication_status(
            root, campaign_complete=True
        )
    if not require_complete or not report.passed:
        return "N/R"
    if report.counts.get("included_blocks", 0) <= 0:
        return "N/R"
    try:
        validate_manual_adjudication_completion(root)
    except (FileNotFoundError, OSError, ValueError, KeyError, SchemaError):
        return "N/R"
    return FINAL_READY_STATUS


def _validate_campaign_completion_publication_status(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    scheduled_block_count: int,
    processed_block_count: int,
    included_block_count: int,
    infrastructure_excluded_block_count: int,
) -> bool:
    """Reject a completion artifact that promotes a pilot without audit authority."""

    path = root / "completion.json"
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or not path.is_file():
        raise SchemaError("campaign completion artifact must be a regular file")
    completion = _read_manual_audit_json(
        path,
        context="campaign completion artifact",
    )
    _require_exact_json_object(
        completion,
        CAMPAIGN_COMPLETION_KEYS,
        context="campaign completion artifact",
    )
    if completion["schema_version"] != SCHEMA_VERSION:
        raise SchemaError("campaign completion artifact has the wrong schema version")
    if completion["campaign_id"] != manifest["campaign_id"]:
        raise SchemaError("campaign completion artifact has the wrong campaign ID")
    expected_counts = {
        "scheduled_block_count": scheduled_block_count,
        "processed_block_count": processed_block_count,
        "included_block_count": included_block_count,
        "infrastructure_excluded_block_count": (
            infrastructure_excluded_block_count
        ),
    }
    for field, expected in expected_counts.items():
        actual = _completion_nonnegative_int(completion[field], field)
        if actual != expected:
            raise SchemaError(
                f"campaign completion artifact {field} differs from schedule evidence"
            )
    terminal = (
        scheduled_block_count > 0
        and included_block_count + infrastructure_excluded_block_count
        == scheduled_block_count
    )
    expected_completion_status = "COMPLETE" if terminal else "INCOMPLETE"
    if completion["status"] != expected_completion_status:
        raise SchemaError(
            "campaign completion artifact status differs from terminal schedule evidence"
        )
    profile = classify_campaign_profile(
        manifest,
        context="campaign completion publication boundary",
        require_campaign_mode=True,
    )
    status = completion["publication_status"]
    if profile == CAMPAIGN_PROFILE_PILOT:
        if status not in {DRAFT_PILOT_STATUS, PILOT_EVIDENCE_LABEL}:
            raise SchemaError("pilot campaign completion has an unregistered publication status")
        if (
            status == PILOT_EVIDENCE_LABEL
            and (
                not terminal
                or adjudication_gated_pilot_publication_status(
                    root, campaign_complete=terminal
                )
                != PILOT_EVIDENCE_LABEL
            )
        ):
            raise SchemaError(
                "pilot campaign completion cannot claim PILOT_ONLY unless the "
                "campaign is complete and registered manual adjudication validates"
            )
        return terminal
    if status != "N/R":
        raise SchemaError("locked-final campaign completion must remain N/R")
    return terminal


def _validate_frozen_handoff_binding(
    root: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Require the frozen campaign to retain its authenticated handoff origin."""

    classify_campaign_profile(
        manifest,
        context="handoff-bound campaign manifest",
        require_campaign_mode=True,
    )
    path = root / "frozen" / "handoff_manifest.json"
    consumption_path = root / FROZEN_HANDOFF_CONSUMPTION_RELATIVE_PATH
    mode = manifest["campaign_mode"]
    if mode == "smoke":
        if (
            path.exists()
            or consumption_path.exists()
            or manifest.get("handoff_manifest_sha256") is not None
            or manifest.get("handoff_inventory_sha256") is not None
            or manifest.get("handoff_consumption_sha256") is not None
        ):
            raise SchemaError(
                "ENGINEERING_SMOKE_ONLY cannot claim an evaluation handoff authority"
            )
        return
    if path.is_symlink() or not path.is_file():
        raise SchemaError("evaluation campaign lacks frozen handoff authority")
    if sha256_file(path) != manifest.get("handoff_manifest_sha256"):
        raise SchemaError("frozen handoff manifest hash differs from campaign")
    value = read_json(path)
    if value.get("schema_version") != "table2-handoff-bundle-v1":
        raise SchemaError("frozen handoff authority schema is invalid")
    inventory = value.get("files")
    if not isinstance(inventory, Mapping) or not inventory:
        raise SchemaError("frozen handoff authority inventory is invalid")
    if sha256_json(inventory) != manifest.get("handoff_inventory_sha256"):
        raise SchemaError("frozen handoff inventory hash differs from campaign")
    frozen_claim_registry_path = root / FROZEN_PAPER_CLAIM_REGISTRY_RELATIVE_PATH
    if (
        value.get("paper_claim_registry_id") != PAPER_CLAIM_REGISTRY_ID
        or value.get("paper_claim_registry_id")
        != manifest.get("paper_claim_registry_id")
        or value.get("paper_claim_registry_sha256")
        != manifest.get("paper_claim_registry_sha256")
        or frozen_claim_registry_path.is_symlink()
        or not frozen_claim_registry_path.is_file()
        or value.get("paper_claim_registry_sha256")
        != sha256_file(frozen_claim_registry_path)
    ):
        raise SchemaError(
            "frozen handoff paper-claim registry identity differs from campaign"
        )
    provenance = read_json(root / "frozen" / "provenance.json")
    provenance_sources = provenance.get("source_files")
    if not isinstance(provenance_sources, Mapping):
        raise SchemaError("frozen provenance lacks source-file evidence")
    for relative, digest in inventory.items():
        row = provenance_sources.get(
            f"evaluation_handoff_inventory:{relative}"
        )
        if not isinstance(row, Mapping) or row.get("sha256") != digest:
            raise SchemaError(
                "frozen provenance differs from the authenticated handoff "
                f"inventory: {relative}"
            )
    if consumption_path.is_symlink() or not consumption_path.is_file():
        raise SchemaError("evaluation campaign lacks frozen handoff consumption")
    if sha256_file(consumption_path) != manifest.get(
        "handoff_consumption_sha256"
    ):
        raise SchemaError("frozen handoff consumption hash differs from campaign")
    arguments = value.get("freeze_arguments")
    if not isinstance(arguments, Mapping):
        raise SchemaError("frozen handoff authority lacks freeze arguments")
    campaign_config = load_yaml(root / "frozen" / "campaign.yaml")
    if campaign_config.get("handoff_manifest") != arguments.get(
        "handoff_manifest"
    ):
        raise SchemaError(
            "frozen campaign/handoff self-authority paths differ"
        )

    registered_manifest = Path(str(arguments.get("handoff_manifest", "")))
    if not registered_manifest.is_absolute():
        raise SchemaError("frozen handoff authority path is not absolute")
    registered_root = registered_manifest.parent.resolve()
    required_argument_paths: list[Path] = []
    for field in (
        "campaign_config",
        "resolved_task_snapshot",
        "environment_manifest",
        "runner_attestation",
        "checkpoint_selection_evidence",
        "paper_claim_registry",
        "pc01_checkpoint_compatibility_receipt",
    ):
        raw_value = arguments.get(field)
        if raw_value is not None:
            required_argument_paths.append(Path(str(raw_value)).resolve())
    for field in ("model_manifests", "memory_manifests"):
        raw_values = arguments.get(field)
        if not isinstance(raw_values, list):
            raise SchemaError(
                f"frozen handoff authority {field} is not a path array"
            )
        required_argument_paths.extend(
            Path(str(raw_value)).resolve() for raw_value in raw_values
        )
    required_relatives: set[str] = set()
    for argument_path in required_argument_paths:
        try:
            relative = argument_path.relative_to(registered_root).as_posix()
        except ValueError as exc:
            raise SchemaError(
                "frozen handoff freeze argument escaped its registered package"
            ) from exc
        if inventory.get(relative) is None:
            raise SchemaError(
                "frozen handoff freeze argument is absent from its inventory"
            )
        required_relatives.add(relative)

    consumption = read_json(consumption_path)
    if set(consumption) != {
        "schema_version",
        "handoff_manifest_sha256",
        "handoff_inventory_sha256",
        "required_freeze_argument_sources",
        "bindings",
    }:
        raise SchemaError("frozen handoff consumption has the wrong fields")
    if (
        consumption.get("schema_version")
        != HANDOFF_CONSUMPTION_SCHEMA_VERSION
        or consumption.get("handoff_manifest_sha256")
        != manifest.get("handoff_manifest_sha256")
        or consumption.get("handoff_inventory_sha256")
        != manifest.get("handoff_inventory_sha256")
        or consumption.get("required_freeze_argument_sources")
        != sorted(required_relatives)
    ):
        raise SchemaError("frozen handoff consumption authority differs")
    bindings = consumption.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise SchemaError("frozen handoff consumption bindings are empty")
    consumed_sources: set[str] = set()
    consumed_destinations: set[str] = set()
    for row in bindings:
        if not isinstance(row, Mapping) or set(row) != {
            "source_relative_path",
            "campaign_relative_path",
            "sha256",
        }:
            raise SchemaError("frozen handoff consumption binding is malformed")
        source_relative = str(row["source_relative_path"])
        campaign_relative = str(row["campaign_relative_path"])
        digest = str(row["sha256"])
        source_path = Path(source_relative)
        destination_path = Path(campaign_relative)
        if (
            not source_relative
            or source_path.is_absolute()
            or ".." in source_path.parts
            or not campaign_relative
            or destination_path.is_absolute()
            or ".." in destination_path.parts
            or inventory.get(source_relative) != digest
            or campaign_relative in consumed_destinations
        ):
            raise SchemaError("frozen handoff consumption binding is invalid")
        frozen_destination = (root / destination_path).resolve()
        if (
            root not in frozen_destination.parents
            or frozen_destination.is_symlink()
            or not frozen_destination.is_file()
            or sha256_file(frozen_destination) != digest
        ):
            raise SchemaError(
                "frozen handoff consumption destination differs from inventory"
            )
        consumed_sources.add(source_relative)
        consumed_destinations.add(campaign_relative)
    if not required_relatives.issubset(consumed_sources):
        raise SchemaError(
            "frozen handoff consumption omits a freeze argument source"
        )
    try:
        selection_relative = Path(
            str(arguments["checkpoint_selection_evidence"])
        ).resolve().relative_to(registered_root).as_posix()
    except (KeyError, ValueError) as exc:
        raise SchemaError(
            "frozen checkpoint-selection authority escaped its handoff package"
        ) from exc
    canonical_selection_binding = {
        "source_relative_path": selection_relative,
        "campaign_relative_path": "frozen/selection_evidence/manifest.json",
        "sha256": str(inventory.get(selection_relative, "")),
    }
    if canonical_selection_binding not in bindings:
        raise SchemaError(
            "frozen checkpoint-selection authority is not bound to its "
            "canonical destination"
        )
    try:
        claim_registry_relative = Path(
            str(arguments["paper_claim_registry"])
        ).resolve().relative_to(registered_root).as_posix()
    except (KeyError, ValueError) as exc:
        raise SchemaError(
            "frozen paper-claim registry escaped its handoff package"
        ) from exc
    canonical_claim_registry_binding = {
        "source_relative_path": claim_registry_relative,
        "campaign_relative_path": str(FROZEN_PAPER_CLAIM_REGISTRY_RELATIVE_PATH),
        "sha256": str(inventory.get(claim_registry_relative, "")),
    }
    if canonical_claim_registry_binding not in bindings:
        raise SchemaError(
            "frozen paper-claim registry is not bound to its canonical destination"
        )


def _validate_frozen_pilot_task_exclusion_binding(
    root: Path,
    manifest: Mapping[str, Any],
):
    relative = manifest.get("pilot_task_exclusion_registry_relative_path")
    if relative != str(FROZEN_PILOT_TASK_EXCLUSION_RELATIVE_PATH):
        raise SchemaError(
            "campaign manifest changed the permanent pilot-exclusion registry path"
        )
    path = root / FROZEN_PILOT_TASK_EXCLUSION_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise SchemaError("frozen pilot-task exclusion registry is missing or symlinked")
    authority = load_pilot_task_exclusion_authority(path)
    if manifest.get("pilot_task_exclusion_registry_sha256") != authority.registry_sha256:
        raise SchemaError(
            "campaign manifest pilot-exclusion registry hash differs from frozen bytes"
        )
    if (
        manifest.get("pilot_task_exclusion_identity_version")
        != PILOT_TASK_EXCLUSION_IDENTITY_VERSION
    ):
        raise SchemaError("campaign manifest changed pilot-exclusion identity semantics")
    provenance = read_json(root / "frozen" / "provenance.json")
    if provenance.get("pilot_task_exclusion_authority") != (
        pilot_task_exclusion_provenance(authority)
    ):
        raise SchemaError("frozen provenance pilot-exclusion binding differs")
    source_files = provenance.get("source_files")
    expected_source = {
        "path": str(PILOT_TASK_EXCLUSION_REGISTRY_RELATIVE_PATH),
        "sha256": authority.registry_sha256,
    }
    if (
        not isinstance(source_files, Mapping)
        or source_files.get("permanent_pilot_task_exclusion_registry")
        != expected_source
    ):
        raise SchemaError(
            "frozen provenance lacks the tracked pilot-exclusion source hash"
        )
    return authority


def _validate_task_boundary(
    tasks: Iterable[Mapping[str, Any]],
    campaign: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    pilot_only: bool,
) -> None:
    rows = list(tasks)
    if not rows:
        raise SchemaError("task manifest is empty")
    for index, row in enumerate(rows):
        require_keys(row, ("task_id",), context=f"task[{index}]")
    # The runtime protocol remains oracle/locked-content blind in every mode.
    # A final benchmark opening is authorized only by the separate, frozen
    # campaign access boundary and explicit final protocol identity.
    locked_allowed = (
        not pilot_only
        and str(campaign.get("locked_test_access", "forbidden"))
        == "single_frozen_campaign"
        and str(protocol.get("protocol_status")) == "FINAL_FROZEN"
        and str(protocol.get("evidence_label")) == "FINAL_LOCKED"
        and protocol["benchmark"].get("allow_locked_reads") is False
    )
    final_partitions = {"locked", "locked_test", "test", "final", "scored"}
    has_final = any(str(row.get("partition", "")).lower() in final_partitions for row in rows)
    if has_final and not locked_allowed:
        raise SchemaError("development/pilot campaign attempted to freeze locked/final tasks")
    if pilot_only:
        required_metadata = {
            "partition": "development",
            "locked_test_content": False,
            "final_paper_evaluation_eligible": False,
        }
        for key, expected in required_metadata.items():
            if metadata.get(key) != expected:
                raise SchemaError(
                    f"pilot task manifest requires {key}={expected!r}, got {metadata.get(key)!r}"
                )
        if int(metadata.get("required_task_count", -1)) != 50:
            raise SchemaError("pilot task manifest must register exactly 50 normal tasks")
        if int(campaign.get("normal_task_count", -1)) != 50 or len(rows) != 50:
            raise SchemaError("pilot campaign and task rows must both contain exactly 50 normal tasks")
    else:
        if not locked_allowed:
            raise SchemaError("locked final campaign lacks its single frozen access authorization")
        registered_benchmark = str(
            protocol.get("benchmark", {}).get("name", "")
        ).strip().casefold()
        frozen_task_benchmark = str(metadata.get("benchmark", "")).strip().casefold()
        if (
            not registered_benchmark
            or frozen_task_benchmark != registered_benchmark
        ):
            raise SchemaError(
                "locked final task benchmark differs from the registered protocol"
            )
        if str(metadata.get("partition", "")).lower() not in final_partitions:
            raise SchemaError("locked final task manifest is not a locked/scored partition")
        if metadata.get("locked_test_content") is not True:
            raise SchemaError("locked final task manifest must declare locked_test_content=true")
        if metadata.get("final_paper_evaluation_eligible") is not True:
            raise SchemaError("locked final tasks are not explicitly paper-evaluation eligible")


def _load_task_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.suffix.lower() != ".json":
        return {}, read_records(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return {}, [dict(row) for row in value]
    if not isinstance(value, dict) or not isinstance(value.get("tasks"), list):
        raise SchemaError(f"task manifest must contain a top-level tasks list: {path}")
    rows = value["tasks"]
    if not all(isinstance(row, Mapping) for row in rows):
        raise SchemaError(f"task manifest contains non-object task rows: {path}")
    metadata = {key: item for key, item in value.items() if key != "tasks"}
    return metadata, [dict(row) for row in rows]


def _validate_resolved_task_snapshot(
    path: Path,
    *,
    registry_path: Path,
    registry_metadata: Mapping[str, Any],
    registry_tasks: Sequence[Mapping[str, Any]],
    environment_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate full WebArena task content against the tracked ordered registry."""

    metadata, rows = _load_task_manifest(path)
    expected_metadata = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "ResolvedWebArenaTaskSnapshot",
        "registry_manifest_sha256": sha256_file(registry_path),
        "environment_manifest_sha256": sha256_file(environment_path),
        "partition": registry_metadata.get("partition"),
        "locked_test_content": registry_metadata.get("locked_test_content"),
        "final_paper_evaluation_eligible": registry_metadata.get(
            "final_paper_evaluation_eligible"
        ),
        "required_task_count": len(registry_tasks),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise SchemaError(f"resolved task snapshot {key} differs from its frozen authority")
    environment = read_json(environment_path)
    if metadata.get("benchmark") != environment.get("benchmark") or metadata.get(
        "benchmark_version"
    ) != environment.get("benchmark_version"):
        raise SchemaError("resolved task snapshot benchmark identity differs from environment")
    evaluator = environment.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise SchemaError("resolved task snapshot requires environment evaluator identity")
    evaluator_identity = {
        "evaluator_id": evaluator.get("evaluator_id"),
        "evaluator_version": evaluator.get("evaluator_version"),
        "source_relative_path": evaluator.get("source_relative_path"),
        "source_sha256": evaluator.get("source_sha256"),
        "oracle_rules_sha256": evaluator.get("oracle_rules_sha256"),
    }
    if metadata.get("evaluator_identity_sha256") != sha256_json(evaluator_identity):
        raise SchemaError("resolved task snapshot evaluator identity is not frozen environment identity")

    registry_identity = [
        (
            str(row.get("task_id", "")),
            row.get("upstream_index"),
            str(row.get("benchmark_task_id", row.get("upstream_index", ""))),
        )
        for row in registry_tasks
    ]
    resolved_identity = [
        (
            str(row.get("task_id", "")),
            row.get("upstream_index"),
            str(row.get("benchmark_task_id", "")),
        )
        for row in rows
    ]
    if resolved_identity != registry_identity:
        raise SchemaError(
            "resolved task snapshot must preserve the tracked WebArena registry order"
        )
    if len(rows) != 50:
        raise SchemaError("resolved pilot task snapshot must contain exactly 50 tasks")

    required_fields = (
        "task_id",
        "upstream_index",
        "benchmark_task_id",
        "benchmark_task_version",
        "instruction",
        "start_state",
        "task_config",
        "evaluator",
        "source_content_sha256",
    )
    for index, row in enumerate(rows):
        require_keys(row, required_fields, context=f"resolved_task[{index}]")
        for key in (
            "task_id",
            "benchmark_task_id",
            "benchmark_task_version",
            "instruction",
        ):
            text = str(row.get(key, "")).strip()
            if not text or any(
                marker in text.lower() for marker in ("placeholder", "tbd", "unknown")
            ):
                raise SchemaError(f"resolved task {index} requires explicit {key}")
        if type(row.get("upstream_index")) is not int:
            raise SchemaError(f"resolved task {index} upstream_index must be an integer")
        start_state = row.get("start_state")
        if not isinstance(start_state, Mapping):
            raise SchemaError(f"resolved task {index} requires a start_state mapping")
        sites = start_state.get("sites")
        if (
            not isinstance(sites, list)
            or not sites
            or not all(isinstance(site, str) and site.strip() for site in sites)
            or not str(start_state.get("start_url", "")).strip()
        ):
            raise SchemaError(f"resolved task {index} start_state lacks sites/start_url")
        task_config = row.get("task_config")
        task_evaluator = row.get("evaluator")
        if not isinstance(task_config, Mapping) or not task_config:
            raise SchemaError(f"resolved task {index} requires full task_config content")
        if not isinstance(task_evaluator, Mapping) or not task_evaluator:
            raise SchemaError(f"resolved task {index} requires evaluator configuration")
        for key in ("evaluator_id", "evaluator_version"):
            if task_evaluator.get(key) != evaluator_identity[key]:
                raise SchemaError(
                    f"resolved task {index} {key} differs from frozen evaluator"
                )
        if not isinstance(task_evaluator.get("config"), Mapping):
            raise SchemaError(f"resolved task {index} evaluator config must be a mapping")
        content = {key: row[key] for key in required_fields if key != "source_content_sha256"}
        if row.get("source_content_sha256") != sha256_json(content):
            raise SchemaError(f"resolved task {index} source-content hash mismatch")
    task_interface_audit = validate_resolved_task_interface_binding(read_json(path))
    require_webarena_task_interface_compatible(task_interface_audit)
    return metadata, rows


def _normal_task_reset_identity(
    campaign_root: Path,
    *,
    task_id: str,
) -> tuple[str, str]:
    """Resolve the exact frozen benchmark/start-state identity for a receipt."""

    metadata, rows = _load_task_manifest(
        campaign_root / "frozen" / "task_manifest.json"
    )
    matches = [row for row in rows if str(row.get("task_id")) == task_id]
    if len(matches) != 1:
        raise SchemaError(
            "WebArena reset-state validation cannot resolve exactly one frozen task"
        )
    start_state = matches[0].get("start_state")
    if not isinstance(start_state, Mapping):
        raise SchemaError("frozen WebArena task lacks its start_state mapping")
    benchmark_version = str(metadata.get("benchmark_version") or "")
    if not benchmark_version:
        raise SchemaError("frozen WebArena task snapshot lacks benchmark_version")
    return benchmark_version, sha256_json(dict(start_state))


def _load_recovery_scenarios(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    value = read_json(path)
    rows = value.get("scenarios")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise SchemaError(f"recovery manifest must contain a scenarios list: {path}")
    records = [dict(row) for row in rows]
    for index, row in enumerate(records):
        require_keys(row, ("scenario_id", "failure_kind"), context=f"recovery_scenario[{index}]")
    return records, {key: item for key, item in value.items() if key != "scenarios"}


def _validate_recovery_campaign_counts(
    normal_tasks: Sequence[Mapping[str, Any]],
    recovery_scenarios: Sequence[Mapping[str, Any]],
    recovery_metadata: Mapping[str, Any],
    campaign: Mapping[str, Any],
    *,
    matched_seeds: Sequence[int],
    repeat_count: int,
    pilot_only: bool,
) -> None:
    normal_ids = [str(row["task_id"]) for row in normal_tasks]
    scenario_ids = [str(row["scenario_id"]) for row in recovery_scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise SchemaError("recovery scenario IDs must be unique")
    overlap = set(normal_ids).intersection(scenario_ids)
    if overlap:
        raise SchemaError(f"normal/recovery task ID collision: {sorted(overlap)}")
    required = int(recovery_metadata.get("required_scenario_count", -1))
    if required != len(recovery_scenarios):
        raise SchemaError("recovery scenario manifest count disagrees with its rows")
    if pilot_only:
        if (
            recovery_metadata.get("evidence_label") != "PILOT_ONLY"
            or recovery_metadata.get("final_paper_evaluation_eligible") is not False
        ):
            raise SchemaError("pilot recovery manifest lacks explicit PILOT_ONLY boundary")
    elif (
        recovery_metadata.get("final_paper_evaluation_eligible") is not True
        or recovery_metadata.get("evidence_label") == "PILOT_ONLY"
    ):
        raise SchemaError("locked final campaign uses non-final recovery scenarios")
    if "normal_task_count" in campaign and int(campaign["normal_task_count"]) != len(normal_tasks):
        raise SchemaError("campaign normal_task_count disagrees with task manifest")
    if "recovery_scenario_count" in campaign and int(campaign["recovery_scenario_count"]) != len(recovery_scenarios):
        raise SchemaError("campaign recovery_scenario_count disagrees with scenario manifest")
    multiplier = len(SYSTEM_IDS) * len(matched_seeds) * repeat_count
    expected_normal_episodes = len(normal_tasks) * multiplier
    expected_recovery_episodes = len(recovery_scenarios) * multiplier
    expected_total = expected_normal_episodes + expected_recovery_episodes
    registered_counts = {
        "normal_episode_count": expected_normal_episodes,
        "recovery_episode_count": expected_recovery_episodes,
        "planned_episode_count": expected_total,
    }
    for key, expected in registered_counts.items():
        if key in campaign and int(campaign[key]) != expected:
            raise SchemaError(f"campaign {key} must be {expected}, got {campaign[key]}")


def _validate_duplicate_audit_manifest(
    path: Path,
    *,
    normal_tasks: Sequence[Mapping[str, Any]],
    recovery_scenarios: Sequence[Mapping[str, Any]],
    require_verified_normal: bool,
) -> None:
    normal_ids = [str(row["task_id"]) for row in normal_tasks]
    recovery_ids = [str(row["scenario_id"]) for row in recovery_scenarios]
    try:
        from web_agent.runtime.duplicate_audit import (
            validate_pilot_duplicate_audit_manifest,
        )

        validate_pilot_duplicate_audit_manifest(
            path,
            normal_task_ids=normal_ids,
            recovery_scenario_ids=recovery_ids,
            require_normal_verified=require_verified_normal,
        )
    except ImportError as exc:
        raise SchemaError("runtime duplicate-audit validator is unavailable") from exc
    except (TypeError, ValueError, RuntimeError) as exc:
        raise SchemaError(f"duplicate-audit manifest is invalid: {exc}") from exc
    value = read_json(path)
    if int(value.get("required_normal_task_count", -1)) != 50:
        raise SchemaError("duplicate-audit manifest must register 50 normal tasks")
    if int(value.get("required_recovery_scenario_count", -1)) != 15:
        raise SchemaError("duplicate-audit manifest must register 15 recovery scenarios")
    if value.get("synthetic_recovery_evidence_scope") != "DIAGNOSTIC_ONLY_NOT_PRIMARY":
        raise SchemaError("synthetic recovery duplicate evidence must remain diagnostic-only")


def _train_corpus_binding(memory_manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = memory_manifest.get("records_sha256")
    provenance = memory_manifest.get("provenance_manifest_sha256")
    if not _is_sha256(records) or not _is_sha256(provenance):
        raise SchemaError(
            "memory manifest lacks records/provenance hashes for duplicate-audit binding"
        )
    try:
        from web_agent.runtime.duplicate_audit import (
            JointDuplicateClusterNamespace,
        )

        duplicate_namespace = JointDuplicateClusterNamespace.from_mapping(
            memory_manifest.get("duplicate_cluster_namespace"),
            require_hashes=True,
        ).to_dict()
    except (ImportError, TypeError, ValueError, RuntimeError) as exc:
        raise SchemaError(
            f"memory manifest has no evaluation-ready duplicate-cluster namespace: {exc}"
        ) from exc
    value: dict[str, Any] = {
        "records_sha256": str(records),
        "provenance_manifest_sha256": str(provenance),
        "duplicate_cluster_namespace": duplicate_namespace,
        "duplicate_cluster_namespace_sha256": sha256_json(duplicate_namespace),
    }
    return {**value, "corpus_binding_sha256": sha256_json(value)}


def _validate_registered_joint_duplicate_memory_bindings(
    memory_by_seed: Mapping[int, Path],
    *,
    duplicate_audit_path: Path,
    required: bool,
    repository_root: Path | None = None,
    assignment_package_root: Path | None = None,
    preparation_package_root: Path | None = None,
    resolved_task_export_path: Path | None = None,
    approved_task_registry_path: Path | None = None,
    registered_recovery_scenarios_path: Path | None = None,
    duplicate_audit_registration_path: Path | None = None,
    provenance_manifest_path: Path | None = None,
) -> dict[str, Any] | None:
    """Cross-bind every memory seed to the registered final duplicate audit.

    The store loader validates each binding's complete hash schema.  This
    campaign-level gate additionally proves that the final audit, completed
    provenance, deterministic assignment package, registered config/source,
    and namespace are the same evidence for every matched seed.
    """

    if not memory_by_seed:
        if required:
            raise SchemaError(
                "evaluation duplicate evidence has no frozen memory stores"
            )
        return None
    evidence_paths = {
        "repository_root": repository_root,
        "assignment_package_root": assignment_package_root,
        "preparation_package_root": preparation_package_root,
        "resolved_task_export_path": resolved_task_export_path,
        "approved_task_registry_path": approved_task_registry_path,
        "registered_recovery_scenarios_path": (
            registered_recovery_scenarios_path
        ),
        "duplicate_audit_registration_path": (
            duplicate_audit_registration_path
        ),
        "provenance_manifest_path": provenance_manifest_path,
    }
    if any(value is None for value in evidence_paths.values()):
        missing = sorted(
            field for field, value in evidence_paths.items() if value is None
        )
        raise SchemaError(
            "registered joint duplicate evidence closure is incomplete: "
            f"{missing}"
        )
    assert repository_root is not None
    assert assignment_package_root is not None
    assert preparation_package_root is not None
    assert resolved_task_export_path is not None
    assert approved_task_registry_path is not None
    assert registered_recovery_scenarios_path is not None
    assert duplicate_audit_registration_path is not None
    assert provenance_manifest_path is not None
    tracked_config = (
        repository_root / JOINT_DUPLICATE_AUDIT_CONFIG_RELATIVE_PATH
    ).resolve()
    tracked_source_authority = (
        repository_root / P4_SOURCE_AUTHORITY_RELATIVE_PATH
    ).resolve()
    tracked_source = (
        repository_root / AUDIT_TOOL_SOURCE_RELATIVE_PATH
    ).resolve()
    registered_audit = (
        repository_root / JOINT_DUPLICATE_AUDIT_REGISTRATION_RELATIVE_PATH
    ).resolve()
    if sha256_file(tracked_source_authority) != (
        P4_REGISTERED_SOURCE_AUTHORITY_SHA256
    ):
        raise SchemaError(
            "P4 source authority differs from the registered PC-01 corpus"
        )
    if duplicate_audit_registration_path.resolve() != registered_audit:
        if sha256_file(duplicate_audit_registration_path) != sha256_file(
            registered_audit
        ):
            raise SchemaError(
                "joint duplicate registration differs from tracked authority"
            )
    try:
        assignment_package = validate_compact_joint_duplicate_evidence(
            package_root=assignment_package_root,
            config_path=tracked_config,
            source_authority_path=tracked_source_authority,
            preparation_root=preparation_package_root,
            resolved_task_export_path=resolved_task_export_path,
            approved_task_registry_path=approved_task_registry_path,
            recovery_scenarios_path=registered_recovery_scenarios_path,
            duplicate_audit_registration_path=duplicate_audit_registration_path,
            provenance_manifest_path=provenance_manifest_path,
            final_audit_path=duplicate_audit_path,
        )
    except (JointDuplicateAuditError, OSError, TypeError, ValueError) as exc:
        raise SchemaError(
            f"compact joint duplicate evidence is invalid: {exc}"
        ) from exc
    assignment_manifest_path = (
        assignment_package.root / "assignment_manifest.json"
    )
    preparation_manifest_path = (
        preparation_package_root / "preparation_manifest.json"
    )
    provenance_payload = read_json(provenance_manifest_path)
    provenance_records = provenance_payload.get("records")
    if not isinstance(provenance_records, Mapping):
        raise SchemaError("completed provenance records are malformed")
    try:
        expected_selection = reconstruct_p4_selection_from_preparation(
            package_root=preparation_package_root,
            provenance_manifest_path=provenance_manifest_path,
        )
    except (P4PreparationError, OSError, TypeError, ValueError) as exc:
        raise SchemaError(
            f"compact P4 eligible-selection replay failed: {exc}"
        ) from exc
    expected_items = [
        dict(candidate.item) for candidate in expected_selection.candidates
    ]
    expected_evidence = {
        candidate.memory_id: dict(candidate.verification_evidence)
        for candidate in expected_selection.candidates
    }
    preparation_manifest = read_json(preparation_manifest_path)
    preparation_audit = read_json(
        preparation_package_root / "candidate_audit.json"
    )
    expected_eligibility = {
        "input_rows": expected_selection.input_rows,
        "pre_dedup_eligible_rows": expected_selection.pre_dedup_eligible_rows,
        "stored_rows": len(expected_items),
        "exclusion_counts": dict(expected_selection.exclusion_counts),
    }
    candidate_assignments = {
        str(row["source_id"]): row for row in assignment_package.candidates
    }
    audit = read_json(duplicate_audit_path)
    audit_sha256 = sha256_file(duplicate_audit_path)
    audit_assignment = audit.get("joint_duplicate_assignment_binding")
    expected_assignment_fields = {
        "schema_version",
        "assignment_manifest_sha256",
        "entities_sha256",
        "clusters_sha256",
        "duplicate_cluster_namespace_sha256",
    }
    if not isinstance(audit_assignment, Mapping) or set(audit_assignment) != (
        expected_assignment_fields
    ):
        raise SchemaError(
            "final duplicate audit lacks its exact joint-assignment binding"
        )
    for field in expected_assignment_fields - {"schema_version"}:
        if not _is_sha256(audit_assignment.get(field)):
            raise SchemaError(
                f"final duplicate audit has invalid assignment {field}"
            )
    if audit_assignment.get("schema_version") != (
        "table2-provenance-joint-duplicate-assignment-binding-v1"
    ):
        raise SchemaError("final duplicate audit assignment schema is not registered")
    provenance_sha256 = audit.get("provenance_manifest_sha256")
    if not _is_sha256(provenance_sha256):
        raise SchemaError("final duplicate audit lacks completed provenance identity")
    if provenance_sha256 != sha256_file(provenance_manifest_path):
        raise SchemaError(
            "final duplicate audit differs from frozen completed provenance"
        )
    try:
        from web_agent.runtime.duplicate_audit import (
            JointDuplicateClusterNamespace,
        )

        audit_namespace = JointDuplicateClusterNamespace.from_mapping(
            audit.get("duplicate_cluster_namespace"),
            require_hashes=True,
        ).to_dict()
    except (ImportError, TypeError, ValueError, RuntimeError) as exc:
        raise SchemaError(
            f"final duplicate audit has invalid cluster namespace: {exc}"
        ) from exc
    if audit_assignment.get("duplicate_cluster_namespace_sha256") != sha256_json(
        audit_namespace
    ):
        raise SchemaError(
            "final duplicate audit assignment/namespace hash is inconsistent"
        )
    corpus = audit.get("train_corpus_binding")
    if not isinstance(corpus, Mapping) or corpus.get(
        "provenance_manifest_sha256"
    ) != provenance_sha256:
        raise SchemaError(
            "final duplicate audit train corpus cites another provenance manifest"
        )

    common: dict[str, Any] | None = None
    assignment_input_binding = assignment_package.manifest.get("input_binding")
    if not isinstance(assignment_input_binding, Mapping):
        raise SchemaError("joint assignment lacks its preparation input binding")
    preparation_receipt_status = assignment_input_binding.get(
        "preparation_execution_receipt_status"
    )
    if (
        sha256_file(tracked_source_authority)
        == _CANONICAL_P4_REGISTERED_SOURCE_AUTHORITY_SHA256
        and preparation_receipt_status != PREPARATION_EXECUTION_RECEIPT_VALIDATED
    ):
        raise SchemaError(
            "joint assignment lacks a validated registered preparation receipt"
        )
    for seed, manifest_path in sorted(memory_by_seed.items()):
        memory = read_json(manifest_path)
        if memory.get("model_seed") != seed:
            raise SchemaError(
                f"joint duplicate binding memory seed mismatch for seed {seed}"
            )
        binding = memory.get("joint_duplicate_audit_binding")
        if not isinstance(binding, Mapping):
            raise SchemaError(
                f"memory seed {seed} lacks registered joint duplicate-audit binding"
            )
        if binding.get("schema_version") != (
            JOINT_DUPLICATE_EVIDENCE_BINDING_SCHEMA_VERSION
        ):
            raise SchemaError(
                f"memory seed {seed} has an unregistered duplicate-evidence binding"
            )
        expected = {
            "final_duplicate_audit_sha256": audit_sha256,
            "provenance_manifest_sha256": provenance_sha256,
            "assignment_manifest_sha256": audit_assignment[
                "assignment_manifest_sha256"
            ],
            "entities_sha256": audit_assignment["entities_sha256"],
            "clusters_sha256": audit_assignment["clusters_sha256"],
            "audit_config_sha256": audit_namespace[
                "audit_tool_config_sha256"
            ],
            "audit_source_sha256": audit_namespace[
                "audit_tool_source_sha256"
            ],
            "duplicate_cluster_namespace": audit_namespace,
            "preparation_manifest_sha256": sha256_file(
                preparation_manifest_path
            ),
            "preparation_execution_receipt_status": preparation_receipt_status,
            "preparation_execution_receipt_sha256": assignment_input_binding.get(
                "preparation_execution_receipt_sha256"
            ),
            "preparation_executed_source_set_sha256": assignment_input_binding.get(
                "preparation_executed_source_set_sha256"
            ),
            "preparation_source_commit": assignment_input_binding.get(
                "preparation_source_commit"
            ),
            "recovery_scenarios_sha256": sha256_file(
                registered_recovery_scenarios_path
            ),
            "duplicate_audit_registration_sha256": sha256_file(
                duplicate_audit_registration_path
            ),
            "source_authority_sha256": sha256_file(tracked_source_authority),
        }
        for field, expected_value in expected.items():
            if binding.get(field) != expected_value:
                raise SchemaError(
                    f"memory seed {seed} joint duplicate binding mismatch at {field}"
                )
        if binding.get("assignment_manifest_sha256") != sha256_file(
            assignment_manifest_path
        ):
            raise SchemaError(
                f"memory seed {seed} assignment-manifest file hash mismatch"
            )
        if binding.get("entities_sha256") != assignment_package.manifest.get(
            "entities_sha256"
        ) or binding.get("clusters_sha256") != assignment_package.manifest.get(
            "clusters_sha256"
        ):
            raise SchemaError(
                f"memory seed {seed} assignment payload hashes mismatch"
            )
        if binding.get("audit_config_sha256") != sha256_file(tracked_config):
            raise SchemaError(
                f"memory seed {seed} audit config differs from tracked source"
            )
        if binding.get("audit_source_sha256") != sha256_file(tracked_source):
            raise SchemaError(
                f"memory seed {seed} audit producer differs from tracked source"
            )
        if common is None:
            common = dict(binding)
        elif dict(binding) != common:
            raise SchemaError(
                "matched memory seeds cite different joint duplicate evidence"
            )
        items = read_jsonl(manifest_path.parent / "items.jsonl")
        if items != expected_items:
            raise SchemaError(
                f"memory seed {seed} item corpus differs from exact compact "
                "eligibility/dedup replay"
            )
        expected_manifest_fields = {
            "dataset_id": preparation_manifest["dataset_id"],
            "dataset_version": preparation_manifest["dataset_version"],
            "records_sha256": preparation_manifest["records_sha256"],
            "dataset_artifacts_sha256": preparation_manifest[
                "dataset_artifacts_sha256"
            ],
            "provenance_manifest_sha256": provenance_sha256,
            "duplicate_cluster_namespace": (
                expected_selection.duplicate_cluster_namespace
            ),
            "eligibility": expected_eligibility,
            "transition_report_sha256": canonical_sha256(
                preparation_audit["transition_report"]
            ),
            "item_count": len(expected_items),
        }
        for field, expected_value in expected_manifest_fields.items():
            if memory.get(field) != expected_value:
                raise SchemaError(
                    f"memory seed {seed} manifest differs from compact source "
                    f"closure at {field}"
                )
        verification_records = read_json(
            manifest_path.parent / "verification_evidence.json"
        )
        frozen_evidence_rows = verification_records.get("records")
        if not isinstance(frozen_evidence_rows, list):
            raise SchemaError(
                f"memory seed {seed} lacks frozen verification records"
            )
        if len(frozen_evidence_rows) != len(expected_evidence):
            raise SchemaError(
                f"memory seed {seed} frozen verification coverage differs"
            )
        for row in frozen_evidence_rows:
            if not isinstance(row, Mapping):
                raise SchemaError(
                    f"memory seed {seed} has malformed verification record"
                )
            memory_id = str(row.get("memory_id") or "")
            expected_record = expected_evidence.get(memory_id)
            if expected_record is None:
                raise SchemaError(
                    f"memory seed {seed} verification record is not eligible: "
                    f"{memory_id}"
                )
            for field in (
                "recovery_verification",
                "final_task_verification",
                "p4_label_review",
                "p4_label_review_evidence_sha256",
                "verification_evidence_sha256",
            ):
                if row.get(field) != expected_record.get(field):
                    raise SchemaError(
                        f"memory seed {seed} verification record {memory_id} "
                        f"differs from staged provenance at {field}"
                    )
        calibration_evidence = read_json(
            manifest_path.parent / "calibration_evidence.json"
        )
        descriptors = calibration_evidence.get("eligible_selection")
        if not isinstance(descriptors, list) or len(descriptors) != len(
            expected_items
        ):
            raise SchemaError(
                f"memory seed {seed} calibration selection coverage differs"
            )
        expected_items_by_id = {
            str(item["memory_id"]): item for item in expected_items
        }
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping):
                raise SchemaError(
                    f"memory seed {seed} calibration descriptor is malformed"
                )
            memory_id = str(descriptor.get("memory_id") or "")
            expected_item = expected_items_by_id.get(memory_id)
            if expected_item is None:
                raise SchemaError(
                    f"memory seed {seed} calibration cites an ineligible item"
                )
            source_id = str(expected_item["source_sample_id"])
            expected_provenance_record = provenance_records.get(source_id)
            if not isinstance(expected_provenance_record, Mapping):
                raise SchemaError(
                    f"memory seed {seed} calibration source lacks provenance"
                )
            if descriptor.get("provenance_record_sha256") != canonical_sha256(
                dict(expected_provenance_record)
            ):
                raise SchemaError(
                    f"memory seed {seed} calibration provenance digest differs "
                    f"for {source_id}"
                )
        seen_sources: set[str] = set()
        for index, item in enumerate(items):
            source_id = str(item.get("source_sample_id") or "")
            if not source_id or source_id in seen_sources:
                raise SchemaError(
                    f"memory seed {seed} has invalid source identity at item {index}"
                )
            seen_sources.add(source_id)
            assignment = candidate_assignments.get(source_id)
            provenance_record = provenance_records.get(source_id)
            if assignment is None or not isinstance(provenance_record, Mapping):
                raise SchemaError(
                    f"memory seed {seed} item {source_id} is absent from staged evidence"
                )
            expected_item_fields = {
                "exact_duplicate_key": assignment["exact_duplicate_key"],
                "duplicate_cluster_id": assignment[
                    "near_duplicate_cluster_id"
                ],
                "duplicate_cluster_namespace_id": (
                    assignment_package.namespace.namespace_id
                ),
                "source_task_id": provenance_record["canonical_task_id"],
                "source_episode_id": provenance_record["episode_id"],
            }
            for field, expected_value in expected_item_fields.items():
                if item.get(field) != expected_value:
                    raise SchemaError(
                        f"memory seed {seed} item {source_id} differs from staged "
                        f"assignment/provenance at {field}"
                    )
    return common


def _build_task_content_binding_manifest(
    normal_tasks: Sequence[Mapping[str, Any]],
    *,
    task_manifest_sha256: str,
    memory_by_seed: Mapping[int, Path],
    evidence_scope: str,
) -> dict[str, Any]:
    task_hashes: dict[str, str] = {}
    for row in normal_tasks:
        task_id = str(row["task_id"])
        if task_id in task_hashes:
            raise SchemaError(f"duplicate task ID in task-content binding: {task_id}")
        task_hashes[task_id] = sha256_json(dict(row))
    corpus_by_seed = {
        str(seed): _train_corpus_binding(read_json(path))
        for seed, path in sorted(memory_by_seed.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "FrozenTaskAndTrainCorpusBindings",
        "evidence_scope": evidence_scope,
        "task_hash_method": "canonical_json_sha256_v1",
        "task_manifest_sha256": task_manifest_sha256,
        "task_content_sha256_by_id": task_hashes,
        "train_corpus_by_seed": corpus_by_seed,
    }


def _validate_duplicate_audit_bindings(
    duplicate_audit_path: Path,
    *,
    task_content_manifest: Mapping[str, Any],
    require_verified_normal: bool,
) -> None:
    payload = read_json(duplicate_audit_path)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise SchemaError("duplicate-audit entries must be an array")
    task_hashes = task_content_manifest.get("task_content_sha256_by_id")
    corpus_by_seed = task_content_manifest.get("train_corpus_by_seed")
    if not isinstance(task_hashes, Mapping) or not isinstance(corpus_by_seed, Mapping):
        raise SchemaError("frozen task/corpus binding manifest is malformed")
    corpus_hashes = {
        str(value.get("corpus_binding_sha256"))
        for value in corpus_by_seed.values()
        if isinstance(value, Mapping)
    }
    if require_verified_normal:
        if not corpus_hashes:
            raise SchemaError("evaluation duplicate audit has no frozen train-corpus binding")
        if len(corpus_hashes) != 1:
            raise SchemaError(
                "matched seeds use different train corpora; one duplicate audit cannot bind all seeds"
            )
    expected_corpus = next(iter(corpus_hashes), None)
    try:
        from web_agent.runtime.duplicate_audit import (
            JointDuplicateClusterNamespace,
        )

        audit_namespace = JointDuplicateClusterNamespace.from_mapping(
            payload.get("duplicate_cluster_namespace"),
            require_hashes=require_verified_normal,
        ).to_dict()
    except (ImportError, TypeError, ValueError, RuntimeError) as exc:
        raise SchemaError(
            f"duplicate-audit cluster namespace is invalid: {exc}"
        ) from exc
    for seed, value in corpus_by_seed.items():
        if not isinstance(value, Mapping):
            raise SchemaError(f"train-corpus binding for seed {seed} is malformed")
        if value.get("duplicate_cluster_namespace") != audit_namespace:
            raise SchemaError(
                "duplicate audit and frozen memory use different cluster namespaces"
            )
        if value.get("duplicate_cluster_namespace_sha256") != sha256_json(
            audit_namespace
        ):
            raise SchemaError(
                f"train-corpus duplicate namespace hash is invalid for seed {seed}"
            )
    seen_verified: set[str] = set()
    for index, row in enumerate(entries):
        if not isinstance(row, Mapping) or row.get("task_partition") != "normal":
            continue
        task_id = str(row.get("task_id", ""))
        if row.get("status") != "VERIFIED":
            continue
        seen_verified.add(task_id)
        expected_content = task_hashes.get(task_id)
        if expected_content is None:
            raise SchemaError(
                f"verified duplicate-audit task is absent from frozen task map: {task_id}"
            )
        if row.get("content_sha256") != expected_content:
            raise SchemaError(
                f"duplicate-audit task content hash mismatch for {task_id}"
            )
        if expected_corpus is None or row.get("train_corpus_manifest_sha256") != expected_corpus:
            raise SchemaError(
                f"duplicate-audit train corpus hash mismatch for {task_id}"
            )
    if require_verified_normal and seen_verified != set(map(str, task_hashes)):
        raise SchemaError("not every frozen normal task has a corpus/content-bound VERIFIED audit")


def _expected_runner_runtime_identity(
    *,
    model_by_seed: Mapping[int, Path],
    memory_by_seed: Mapping[int, Path],
    protocol: Mapping[str, Any],
    prompt_sources: Mapping[str, Path],
    environment_path: Path,
    resolved_task_snapshot_path: Path,
    runtime_integration: Mapping[str, str],
    selection_evidence_path: Path | None = None,
    checkpoint_compatibility_receipt_path: Path | None = None,
) -> dict[str, Any]:
    checkpoints: dict[str, dict[str, str]] = {}
    backbones: dict[str, dict[str, str]] = {}
    for seed, path in sorted(model_by_seed.items()):
        value = read_json(path)
        checkpoints[str(seed)] = {
            "selected_checkpoint_sha256": str(value["selected_checkpoint_sha256"]),
            "resolved_config_sha256": str(value["resolved_config_sha256"]),
            "resolved_config_record_sha256": str(
                value["resolved_config_record_sha256"]
            ),
            "processor_contract_sha256": str(value["processor_contract_sha256"]),
            "model_evidence_bundle_sha256": str(
                value["model_evidence_bundle_sha256"]
            ),
        }
        backbones[str(seed)] = {
            "backbone_id": str(value["e0_backbone_id"]),
            "backbone_revision": str(value["e0_backbone_revision"]),
            "backbone_sha256": str(value["e0_backbone_sha256"]),
            "resolved_config_sha256": str(value["e0_resolved_config_sha256"]),
            "processor_contract_sha256": str(value["e0_processor_contract_sha256"]),
            "base_prompt_sha256": str(value["e0_base_prompt_sha256"]),
            "parser_id": str(value["e0_parser_id"]),
            "parser_version": str(value["e0_parser_version"]),
            "parser_sha256": str(value["e0_parser_sha256"]),
        }
    memories = {
        str(seed): {
            "model_seed": int(seed),
            "manifest_sha256": sha256_file(path),
            **_train_corpus_binding(read_json(path)),
            "joint_duplicate_audit_binding": dict(
                read_json(path)["joint_duplicate_audit_binding"]
            ),
        }
        for seed, path in sorted(memory_by_seed.items())
    }
    provider = protocol.get("parameter_provider", {})
    if not isinstance(provider, Mapping):
        raise SchemaError("protocol parameter_provider must be a mapping")
    decoding = provider.get("decoding_parameters", {})
    if not isinstance(decoding, Mapping):
        raise SchemaError("parameter-provider decoding_parameters must be a mapping")
    environment = read_json(environment_path)
    evaluator = environment.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise SchemaError("environment evaluator identity is absent")
    identity = {
        "checkpoint_systems": ["E1", "E2", "E3"],
        "selected_checkpoint_by_seed": checkpoints,
        "e0_unadapted_backbone_by_seed": backbones,
        "parameter_provider": {
            "prompt_sha256": sha256_file(prompt_sources["parameter_provider_v1.txt"]),
            "decoding_parameters": dict(decoding),
            "decoding_parameters_sha256": sha256_json(dict(decoding)),
        },
        "memory_by_seed": memories,
        "environment": {
            "manifest_sha256": sha256_file(environment_path),
            "benchmark": environment.get("benchmark"),
            "benchmark_version": environment.get("benchmark_version"),
            "benchmark_revision": environment.get("benchmark_revision"),
            "task_definition_version": environment.get("task_definition_version"),
            "browser": environment.get("browser"),
            "browser_version": environment.get("browser_version"),
            "playwright_version": environment.get("playwright_version"),
            "controller_version": environment.get("controller_version"),
            "environment_adapter_id": environment.get("environment_adapter_id"),
            "environment_adapter_version": environment.get(
                "environment_adapter_version"
            ),
            "environment_state_digester": environment.get(
                "environment_state_digester"
            ),
            "infrastructure_classifier": environment.get(
                "infrastructure_classifier"
            ),
            "page_settle_policy": environment.get("page_settle_policy"),
            "manual_rescue_guard": environment.get("manual_rescue_guard"),
            "container_digest": environment.get("container_digest"),
            "dependency_lock_sha256": environment.get("dependency_lock_sha256"),
            "dependency_lock_relative_path": environment.get(
                "dependency_lock_relative_path"
            ),
            "viewport": environment.get("viewport"),
            "pc01_live_deployment": environment.get(
                "pc01_live_deployment"
            ),
        },
        "evaluator": dict(evaluator),
        "resolved_task_snapshot": {
            "sha256": sha256_file(resolved_task_snapshot_path),
            "record_type": read_json(resolved_task_snapshot_path).get("record_type"),
        },
        "runtime_integration": dict(runtime_integration),
    }
    if selection_evidence_path is not None:
        identity["validation_selection_evidence"] = {
            "manifest_sha256": sha256_file(selection_evidence_path),
            "schema_version": read_json(selection_evidence_path).get(
                "schema_version"
            ),
        }
    if checkpoint_compatibility_receipt_path is not None:
        receipt = _read_canonical_pc01_checkpoint_compatibility_receipt(
            checkpoint_compatibility_receipt_path
        )
        identity[PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD] = (
            _pc01_checkpoint_compatibility_binding(
                checkpoint_compatibility_receipt_path,
                receipt,
            )
        )
    return identity


def _validate_frozen_runner_attestation(
    root: Path,
    campaign_manifest: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
) -> None:
    classify_campaign_profile(
        campaign_manifest,
        context="runner-attested campaign manifest",
        require_campaign_mode=True,
    )
    mode = campaign_manifest["campaign_mode"]
    attestation_path = root / "frozen" / "runner_attestation.json"
    source_root = root / "frozen" / "runner_source"
    if mode == "smoke":
        if campaign_manifest.get("runner_identity_scope") != ENGINEERING_SMOKE_SCOPE:
            raise SchemaError("smoke runner bypass lacks ENGINEERING_SMOKE_ONLY scope")
        if (
            campaign_manifest.get("runner_entrypoint") is not None
            or campaign_manifest.get("runner_attestation_sha256") is not None
            or attestation_path.exists()
            or source_root.exists()
        ):
            raise SchemaError("engineering smoke falsely claims frozen evaluation runner evidence")
        return
    if campaign_manifest.get("runner_identity_scope") != EVALUATION_RUNNER_SCOPE:
        raise SchemaError("evaluation campaign lacks frozen runner identity scope")
    if not attestation_path.is_file():
        raise SchemaError("evaluation campaign lacks frozen runner attestation")
    if sha256_file(attestation_path) != campaign_manifest.get("runner_attestation_sha256"):
        raise SchemaError("evaluation runner attestation hash mismatch")
    attestation = read_json(attestation_path)
    if attestation.get("runner_entrypoint") != campaign_manifest.get("runner_entrypoint"):
        raise SchemaError("campaign/attestation runner entrypoints differ")
    if attestation.get("repository_commit") != campaign_manifest.get("repository_commit"):
        raise SchemaError("campaign/attestation repository commits differ")
    model_by_seed = {
        int(path.stem.removeprefix("seed_")): path
        for path in sorted((root / "frozen" / "models").glob("seed_*.json"))
    }
    memory_by_seed = {
        int(path.parent.name.removeprefix("seed_")): path
        for path in sorted((root / "memory").glob("seed_*/manifest.json"))
    }
    expected = _expected_runner_runtime_identity(
        model_by_seed=model_by_seed,
        memory_by_seed=memory_by_seed,
        protocol=protocol,
        prompt_sources={
            "parameter_provider_v1.txt": (
                root / "frozen" / "prompts" / "parameter_provider_v1.txt"
            ),
            "e0_action_v1.txt": root / "frozen" / "prompts" / "e0_action_v1.txt",
        },
        environment_path=root / "frozen" / "environment.json",
        resolved_task_snapshot_path=root / "frozen" / "task_manifest.json",
        runtime_integration=campaign_manifest.get("runtime_integration", {}),
        selection_evidence_path=(
            root / "frozen" / "selection_evidence" / "manifest.json"
            if campaign_manifest.get("checkpoint_selection_evidence_sha256")
            is not None
            else None
        ),
        checkpoint_compatibility_receipt_path=(
            (
                root
                / "frozen"
                / PC01_CHECKPOINT_COMPATIBILITY_RECEIPT_RELATIVE_PATH
            )
            if campaign_manifest.get(
                PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD
            )
            is not None
            else None
        ),
    )
    if attestation.get("runtime_identity") != expected:
        raise SchemaError("frozen runner loaded-identity binding differs from artifacts")
    if attestation.get("runtime_integration_entrypoint") != campaign_manifest.get(
        "runtime_integration_entrypoint"
    ):
        raise SchemaError("campaign/attestation runtime integration entrypoints differ")
    rows = attestation.get("source_files")
    if not isinstance(rows, list) or not rows:
        raise SchemaError("frozen runner attestation has no source files")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise SchemaError("frozen runner source descriptor is malformed")
        relative = str(row.get("relative_path", ""))
        if not relative or relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise SchemaError("frozen runner source path is unsafe or duplicated")
        seen.add(relative)
        expected_hash = str(row.get("sha256", ""))
        source = source_root / relative
        if not _is_sha256(expected_hash) or not source.is_file() or sha256_file(source) != expected_hash:
            raise SchemaError(f"frozen runner source hash mismatch: {relative}")
        normalized.append({"relative_path": relative, "sha256": expected_hash})
    if str(attestation.get("primary_source_relative_path", "")) not in seen:
        raise SchemaError("frozen primary runner source is not registered")
    source_hashes = {row["relative_path"]: row["sha256"] for row in normalized}
    missing_control_sources = sorted(
        set(EVALUATION_CONTROL_SOURCE_RELATIVE_PATHS) - set(source_hashes)
    )
    if missing_control_sources:
        raise SchemaError(
            "frozen runner source set omits Table 2 freeze/live/analysis "
            "control source: " + ", ".join(missing_control_sources)
        )
    for label, identity in (
        ("runtime integration", expected.get("runtime_integration")),
        ("evaluator", expected.get("evaluator")),
    ):
        if not isinstance(identity, Mapping):
            raise SchemaError(f"frozen {label} identity is absent")
        relative = str(identity.get("source_relative_path", ""))
        if source_hashes.get(relative) != identity.get("source_sha256"):
            raise SchemaError(f"frozen {label} source is absent from runner source set")
    live_deployment = validate_bound_pc01_live_deployment(
        read_json(root / "frozen" / "environment.json"),
        artifact_root=root / "frozen",
        repository_root=source_root,
    )
    required_live_rows = list(
        live_deployment.binding["capability_source_files"]
    )
    live_validator_relative = "src/web_agent/eval/table2/live_deployment.py"
    required_live_rows.append(
        {
            "relative_path": live_validator_relative,
            "sha256": sha256_file(source_root / live_validator_relative),
        }
    )
    for row in required_live_rows:
        relative = str(row["relative_path"])
        if source_hashes.get(relative) != row["sha256"]:
            raise SchemaError(
                "frozen runner source set omits live capability implementation "
                f"source: {relative}"
            )
    if attestation.get("source_set_sha256") != sha256_json(normalized):
        raise SchemaError("frozen runner source-set hash mismatch")
    if attestation.get("runner_entrypoint") == PRODUCTION_RUNNER_ENTRYPOINT:
        validate_pc01_page_broker_security_binding(
            attestation.get(PC01_PAGE_BROKER_SECURITY_FIELD),
            source_hashes=source_hashes,
        )
    elif attestation.get(PC01_PAGE_BROKER_SECURITY_FIELD) is not None:
        raise SchemaError(
            "non-production runner attestation must not claim PC-01 broker security"
        )


def _payload_sha256(path: Path) -> str:
    if path.is_symlink():
        raise SchemaError(f"artifact payload must not be a symlink: {path}")
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    entries = list(path.rglob("*"))
    if any(candidate.is_symlink() for candidate in entries):
        raise SchemaError(f"artifact payload directory contains a symlink: {path}")
    files = sorted(candidate for candidate in entries if candidate.is_file())
    if not files:
        raise SchemaError(f"artifact payload directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _payload_size_and_count(path: Path) -> tuple[int, int]:
    if path.is_symlink():
        raise SchemaError(f"artifact payload must not be a symlink: {path}")
    if path.is_file():
        return path.stat().st_size, 1
    if not path.is_dir():
        raise FileNotFoundError(path)
    entries = list(path.rglob("*"))
    if any(candidate.is_symlink() for candidate in entries):
        raise SchemaError(f"artifact payload directory contains a symlink: {path}")
    files = sorted(candidate for candidate in entries if candidate.is_file())
    if not files:
        raise SchemaError(f"artifact payload directory is empty: {path}")
    return sum(item.stat().st_size for item in files), len(files)


def _stable_payload_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    """Return the fields which bind one opened payload inode and its bytes."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_stable_payload_component(
    parent_descriptor: int,
    name: str,
    *,
    directory: bool,
    label: str,
) -> tuple[int, os.stat_result]:
    """Open one path component without following it and bind its inode."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    else:
        # A concurrent replacement with a FIFO/device must not block validation.
        flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            raise SchemaError(f"{label} must not use symlink ancestry")
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except SchemaError:
        raise
    except OSError as exc:
        raise SchemaError(f"cannot securely open {label}") from exc
    try:
        opened = os.fstat(descriptor)
        after = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
        if (
            not expected_kind(opened.st_mode)
            or _stable_payload_metadata(before) != _stable_payload_metadata(opened)
            or _stable_payload_metadata(opened) != _stable_payload_metadata(after)
        ):
            raise SchemaError(f"{label} changed during secure open")
        if not directory and opened.st_nlink != 1:
            raise SchemaError(f"{label} must not be hard-linked")
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _hash_stable_payload_file(
    parent_descriptor: int,
    name: str,
    *,
    label: str,
    aggregate: Any | None = None,
) -> tuple[str | None, int, os.stat_result]:
    descriptor, before = _open_stable_payload_component(
        parent_descriptor,
        name,
        directory=False,
        label=label,
    )
    try:
        digest = aggregate if aggregate is not None else hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            digest.update(block)
        opened_after = os.fstat(descriptor)
        path_after = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            total != before.st_size
            or opened_after.st_nlink != 1
            or path_after.st_nlink != 1
            or _stable_payload_metadata(before)
            != _stable_payload_metadata(opened_after)
            or _stable_payload_metadata(opened_after)
            != _stable_payload_metadata(path_after)
        ):
            raise SchemaError(f"{label} changed during authenticated read")
        return (digest.hexdigest() if aggregate is None else None), total, opened_after
    except OSError as exc:
        raise SchemaError(f"cannot securely read {label}") from exc
    finally:
        os.close(descriptor)


def _scan_stable_payload_directory(
    descriptor: int,
    opened_before: os.stat_result,
    *,
    relative_prefix: Path,
    label: str,
    aggregate: Any,
) -> tuple[list[tuple[str, int]], os.stat_result]:
    """Scan a directory through its fd, returning ordered file identities."""

    try:
        names_before = sorted(os.listdir(descriptor))
    except OSError as exc:
        raise SchemaError(f"cannot securely enumerate {label}") from exc
    files: list[tuple[str, int]] = []
    for name in names_before:
        relative = relative_prefix / name
        try:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise SchemaError(f"{label} changed during enumeration") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SchemaError(f"{label} contains a symlink: {relative.as_posix()}")
        if stat.S_ISDIR(metadata.st_mode):
            child, child_before = _open_stable_payload_component(
                descriptor,
                name,
                directory=True,
                label=f"{label} directory {relative.as_posix()}",
            )
            try:
                nested, child_after = _scan_stable_payload_directory(
                    child,
                    child_before,
                    relative_prefix=relative,
                    label=label,
                    aggregate=aggregate,
                )
            finally:
                os.close(child)
            rebound = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if (
                _stable_payload_metadata(metadata)
                != _stable_payload_metadata(child_before)
                or _stable_payload_metadata(child_before)
                != _stable_payload_metadata(child_after)
                or _stable_payload_metadata(child_after)
                != _stable_payload_metadata(rebound)
            ):
                raise SchemaError(f"{label} directory changed during traversal")
            files.extend(nested)
        elif stat.S_ISREG(metadata.st_mode):
            relative_bytes = relative.as_posix().encode("utf-8")
            aggregate.update(len(relative_bytes).to_bytes(8, "big"))
            aggregate.update(relative_bytes)
            _, size, _ = _hash_stable_payload_file(
                descriptor,
                name,
                label=f"{label} file {relative.as_posix()}",
                aggregate=aggregate,
            )
            files.append((relative.as_posix(), size))
        else:
            raise SchemaError(
                f"{label} contains a non-regular entry: {relative.as_posix()}"
            )
    try:
        names_after = sorted(os.listdir(descriptor))
        opened_after = os.fstat(descriptor)
    except OSError as exc:
        raise SchemaError(f"{label} changed during enumeration") from exc
    if (
        names_after != names_before
        or _stable_payload_metadata(opened_before)
        != _stable_payload_metadata(opened_after)
    ):
        raise SchemaError(f"{label} changed during traversal")
    return files, opened_after


def _validated_campaign_relative_model_payload(
    campaign_root: Path,
    relative_path: object,
    *,
    label: str,
    stored_path: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Authenticate one campaign-relative payload without resolving aliases."""

    if (
        type(relative_path) is not str
        or not relative_path
        or relative_path != relative_path.strip()
        or "\\" in relative_path
        or "\x00" in relative_path
    ):
        raise SchemaError(f"{label} requires a canonical campaign-relative path")
    lexical = Path(relative_path)
    if (
        lexical.is_absolute()
        or ".." in lexical.parts
        or "." in lexical.parts
        or lexical.as_posix() != relative_path
    ):
        raise SchemaError(f"{label} requires a canonical campaign-relative path")
    root = Path(os.path.abspath(os.fspath(campaign_root)))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root_descriptor = os.open(root, directory_flags)
    except OSError as exc:
        raise SchemaError(f"cannot securely open {label} campaign root") from exc
    opened_components: list[tuple[int, str, int, os.stat_result]] = []
    descriptors = [root_descriptor]
    try:
        root_before = os.fstat(root_descriptor)
        parent_descriptor = root_descriptor
        for component in lexical.parts[:-1]:
            child, metadata = _open_stable_payload_component(
                parent_descriptor,
                component,
                directory=True,
                label=f"{label} ancestor",
            )
            opened_components.append(
                (parent_descriptor, component, child, metadata)
            )
            descriptors.append(child)
            parent_descriptor = child

        leaf = lexical.parts[-1]
        leaf_metadata = os.stat(
            leaf,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(leaf_metadata.st_mode):
            raise SchemaError(f"{label} must not use symlinks")
        if stat.S_ISREG(leaf_metadata.st_mode):
            digest, size_bytes, stable_leaf = _hash_stable_payload_file(
                parent_descriptor,
                leaf,
                label=label,
            )
            if _stable_payload_metadata(leaf_metadata) != _stable_payload_metadata(
                stable_leaf
            ):
                raise SchemaError(f"{label} changed before authenticated read")
            kind = "file"
            file_count = 1
        elif stat.S_ISDIR(leaf_metadata.st_mode):
            leaf_descriptor, opened_leaf = _open_stable_payload_component(
                parent_descriptor,
                leaf,
                directory=True,
                label=label,
            )
            descriptors.append(leaf_descriptor)
            aggregate = hashlib.sha256()
            files, leaf_after = _scan_stable_payload_directory(
                leaf_descriptor,
                opened_leaf,
                relative_prefix=Path(),
                label=label,
                aggregate=aggregate,
            )
            if not files:
                raise SchemaError(f"{label} directory is empty")
            rebound_leaf = os.stat(
                leaf,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                _stable_payload_metadata(leaf_metadata)
                != _stable_payload_metadata(opened_leaf)
                or _stable_payload_metadata(opened_leaf)
                != _stable_payload_metadata(leaf_after)
                or _stable_payload_metadata(leaf_after)
                != _stable_payload_metadata(rebound_leaf)
            ):
                raise SchemaError(f"{label} directory changed during traversal")
            digest = aggregate.hexdigest()
            size_bytes = sum(size for _, size in files)
            file_count = len(files)
            kind = "directory"
            stable_leaf = leaf_after
        else:
            raise SchemaError(f"{label} must be a regular file or directory")

        # Rebind every open component after all reads, then resolve only after
        # the no-follow walk has authenticated the complete lexical ancestry.
        for component_parent, component, child, opened in opened_components:
            current = os.stat(
                component,
                dir_fd=component_parent,
                follow_symlinks=False,
            )
            if (
                _stable_payload_metadata(opened)
                != _stable_payload_metadata(os.fstat(child))
                or _stable_payload_metadata(opened)
                != _stable_payload_metadata(current)
            ):
                raise SchemaError(f"{label} ancestry changed during validation")
        if _stable_payload_metadata(root_before) != _stable_payload_metadata(
            os.fstat(root_descriptor)
        ):
            raise SchemaError(f"{label} campaign root changed during validation")
        resolved = (root / lexical).resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:  # pragma: no cover - no-follow walk is primary
            raise SchemaError(f"{label} escaped its campaign root") from exc
        resolved_metadata = resolved.lstat()
        if (
            _stable_payload_metadata(stable_leaf)
            != _stable_payload_metadata(resolved_metadata)
            or _stable_payload_metadata(root_before)
            != _stable_payload_metadata(root.lstat())
        ):
            raise SchemaError(f"{label} identity changed during resolution")
        for component_parent, component, child, opened in opened_components:
            current = os.stat(
                component,
                dir_fd=component_parent,
                follow_symlinks=False,
            )
            if (
                _stable_payload_metadata(opened)
                != _stable_payload_metadata(os.fstat(child))
                or _stable_payload_metadata(opened)
                != _stable_payload_metadata(current)
            ):
                raise SchemaError(f"{label} ancestry changed during resolution")
        return resolved, {
            "path": stored_path if stored_path is not None else str(resolved),
            "kind": kind,
            "sha256": digest,
            "size_bytes": size_bytes,
            "file_count": file_count,
        }
    except OSError as exc:
        raise SchemaError(f"{label} changed during validation") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


_PUBLIC_MODEL_PAYLOAD_DESCRIPTOR_FIELDS = (
    "path",
    "kind",
    "sha256",
    "size_bytes",
    "file_count",
)


def _public_model_payload_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: value.get(field)
        for field in _PUBLIC_MODEL_PAYLOAD_DESCRIPTOR_FIELDS
    }


def _reauthenticate_campaign_relative_model_payloads(
    campaign_root: Path,
    rows: Mapping[str, Any],
    authenticated: Mapping[str, tuple[Path, Mapping[str, Any]]],
    *,
    roles: Sequence[str],
    label: str,
) -> None:
    """Close in-call cross-role drift before returning retained paths.

    Returned paths remain subject to the repository's documented same-UID
    limitation once these descriptors close; this is not a hostile-code boundary.
    """

    if set(rows) != set(roles) or set(authenticated) != set(roles):
        raise SchemaError(f"{label} role coverage changed before return")
    for role in roles:
        row = rows.get(role)
        retained = authenticated.get(role)
        if not isinstance(row, Mapping) or retained is None:
            raise SchemaError(f"{label} descriptor changed before return: {role}")
        retained_path, retained_descriptor = retained
        rebound_path, rebound_descriptor = _validated_campaign_relative_model_payload(
            campaign_root,
            row.get("path"),
            label=f"{label} {role} return authentication",
            stored_path=str(row.get("path")),
        )
        public = _public_model_payload_descriptor(row)
        if (
            rebound_path != retained_path
            or rebound_descriptor != dict(retained_descriptor)
            or rebound_descriptor != public
        ):
            raise SchemaError(f"{label} changed before return: {role}")


def _artifact_payload_descriptor(path: Path, *, stored_path: str | None = None) -> dict[str, Any]:
    candidate = path.absolute()
    if candidate.is_symlink():
        raise SchemaError(f"artifact payload must not be a symlink: {path}")
    resolved = candidate.resolve()
    size_bytes, file_count = _payload_size_and_count(resolved)
    return {
        "path": stored_path if stored_path is not None else str(resolved),
        "kind": "file" if resolved.is_file() else "directory",
        "sha256": _payload_sha256(resolved),
        "size_bytes": size_bytes,
        "file_count": file_count,
    }


def _processor_contract_from_payload(path: Path) -> Any:
    if not path.is_file() or path.suffix.lower() != ".json":
        raise SchemaError(f"processor contract must be a JSON file: {path}")
    value = read_json(path)
    required = (
        "processor_class",
        "processor_revision",
        "processor_config_sha256",
        "pre_action_field_mapping",
        "post_action_field_mapping",
    )
    require_keys(value, required, context=f"processor contract {path}")
    try:
        from web_agent.runtime.observation import ProcessorParityContract

        contract = ProcessorParityContract(**{key: value[key] for key in required})
    except ImportError as exc:
        raise SchemaError("processor parity contract implementation is unavailable") from exc
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"processor contract cannot reconstruct ProcessorParityContract: {path}") from exc
    canonical_payload = canonical_json_bytes(contract.to_dict())
    if value != contract.to_dict() or path.read_bytes() != canonical_payload:
        raise SchemaError(
            "processor contract payload must be exact canonical "
            "ProcessorParityContract.to_dict() JSON bytes"
        )
    if sha256_file(path) != contract.record_sha256:
        raise SchemaError("processor contract raw hash differs from record_sha256")
    return contract


def _model_evidence_bundle_value(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    producer_schema_version: str,
) -> dict[str, Any]:
    """Build the content-stable, evidence-only model bundle record.

    The bundle deliberately lives beside, rather than inside, the executable
    payload role set.  In particular, an export receipt whose
    ``runtime_ready`` value is false is authentication/parity evidence; it is
    never a substitute for the live deployment and checkpoint-backed runtime
    gates.
    """

    artifacts_value = {
        role: dict(artifacts[role]) for role in MODEL_EVIDENCE_ROLES
    }
    basis = {
        "schema_version": MODEL_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "producer_schema_version": producer_schema_version,
        "evidence_status": MODEL_EVIDENCE_STATUS,
        "runtime_readiness_claim": False,
        "artifacts": artifacts_value,
    }
    content_identity = {
        **{key: value for key, value in basis.items() if key != "artifacts"},
        "artifacts": {
            role: {
                key: artifacts_value[role][key]
                for key in ("kind", "sha256", "size_bytes", "file_count")
            }
            for role in MODEL_EVIDENCE_ROLES
        },
    }
    return {**basis, "bundle_sha256": sha256_json(content_identity)}


def _canonical_model_evidence_json(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file() or path.suffix.lower() != ".json":
        raise SchemaError(f"model evidence {role} must be one canonical JSON file")
    value = read_json(path)
    if path.read_bytes() != canonical_json_bytes(value):
        raise SchemaError(f"model evidence {role} is not canonical JSON bytes")
    return value


def _require_evidence_hash_link(
    value: Mapping[str, Any],
    field: str,
    expected: str,
    *,
    context: str,
) -> None:
    if value.get(field) != expected:
        raise SchemaError(f"{context}.{field} differs from the bound evidence bytes")


def _validate_registered_pc01_base_snapshot(
    evidence_path: Path,
    executable_backbone_path: Path,
) -> dict[str, Any]:
    """Rebuild the registered base manifest from executable E0 bytes."""

    try:
        supplied = load_pc01_base_snapshot_manifest(
            evidence_path,
            expected_directory_payload_sha256=PC01_EXPECTED_BASE_SNAPSHOT_SHA256,
        )
        rebuilt = build_pc01_base_snapshot_manifest(executable_backbone_path)
    except Exception as exc:
        raise SchemaError("registered PC-01 base-snapshot evidence is invalid") from exc
    if supplied != rebuilt:
        raise SchemaError(
            "PC-01 base-snapshot evidence does not exactly reproduce executable E0"
        )
    return supplied


def _validate_registered_pc01_export_manifest(evidence_path: Path) -> None:
    """Require the exact frozen PC-01 v3 export-manifest bytes."""

    if sha256_file(evidence_path) != PC01_EXPECTED_EXPORT_MANIFEST_SHA256:
        raise SchemaError(
            "PC-01 export-manifest evidence differs from the registered v3 identity"
        )


def _validate_registered_pc01_action_value_evidence(
    value: Mapping[str, Any],
    *,
    report_sha256: str,
    run_contract_sha256: str,
) -> dict[str, Any]:
    try:
        return validate_pc01_training_action_value_evidence(
            value,
            report_sha256=report_sha256,
            run_contract_sha256=run_contract_sha256,
        )
    except Exception as exc:
        raise SchemaError(
            "registered PC-01 training action-value evidence is invalid"
        ) from exc


def _validate_registered_pc01_processor_parity(
    value: Mapping[str, Any],
    **identities: str,
) -> dict[str, Any]:
    try:
        return validate_pc01_processor_parity_receipt(value, **identities)
    except Exception as exc:
        raise SchemaError("registered PC-01 processor parity evidence is invalid") from exc


def _validate_pc01_model_evidence_cross_consistency(
    *,
    model: Mapping[str, Any],
    evidence: Mapping[str, tuple[Path, dict[str, Any]]],
    executable: Mapping[str, tuple[Path, dict[str, Any]]] | None,
) -> None:
    if executable is None:
        raise SchemaError(
            "PC-01 evidence requires the executable model payloads for authentication"
        )
    expected_model_identity = {
        "selected_model_id": PC01_MODEL_ID,
        "model_seed": PC01_MODEL_SEED,
        "e0_backbone_revision": PC01_MODEL_REVISION,
    }
    for field, expected in expected_model_identity.items():
        if model.get(field) != expected:
            raise SchemaError(f"PC-01 model manifest changed {field}")
    values = {
        role: _canonical_model_evidence_json(path, role=role)
        for role, (path, _) in evidence.items()
    }
    _validate_registered_pc01_export_manifest(evidence["export_manifest"][0])
    export = values["export_manifest"]
    exact_export = {
        "schema_version": MODEL_EVIDENCE_PRODUCER_SCHEMA_VERSION,
        "model_id": PC01_MODEL_ID,
        "model_seed": PC01_MODEL_SEED,
        "model_revision": PC01_MODEL_REVISION,
        "checkpoint_sha256": model.get("selected_checkpoint_sha256"),
        "resolved_config_record_sha256": model.get(
            "resolved_config_record_sha256"
        ),
        "resolved_config_payload_sha256": model.get("resolved_config_sha256"),
        "processor_contract_sha256": model.get("processor_contract_sha256"),
        "selection_scope": "validation_only",
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "processor_parity_verified": True,
        "training_environment_source_sha256": (
            PC01_TRAINING_ENVIRONMENT_SHA256
        ),
        "network_access_used": False,
        "weight_updates_performed": False,
        # This is intentionally evidence that the compact exporter did not
        # execute a model forward pass.  It must never be promoted to runtime
        # readiness by the handoff/campaign machinery.
        "runtime_ready": False,
    }
    for field, expected in exact_export.items():
        if export.get(field) != expected:
            raise SchemaError(
                f"model export evidence changed {field}: expected {expected!r}, "
                f"got {export.get(field)!r}"
            )

    files = export.get("files")
    if not isinstance(files, Mapping):
        raise SchemaError("model export evidence lacks its file manifest")
    expected_export_files = {
        "resolved_config.json",
        "processor_contract.json",
        "e0_resolved_config.json",
        "e0_processor_contract.json",
        *(
            filename
            for role, filename in MODEL_EVIDENCE_FILENAMES.items()
            if role != "export_manifest"
        ),
    }
    if set(files) != expected_export_files:
        raise SchemaError("model export evidence file coverage changed")
    for role in MODEL_EVIDENCE_ROLES:
        if role == "export_manifest":
            continue
        filename = MODEL_EVIDENCE_FILENAMES[role]
        file_row = files.get(filename)
        descriptor = evidence[role][1]
        if not isinstance(file_row, Mapping):
            raise SchemaError(f"model export evidence omits {filename}")
        if file_row.get("sha256") != descriptor["sha256"] or file_row.get(
            "size_bytes"
        ) != descriptor["size_bytes"]:
            raise SchemaError(f"model export file manifest differs for {filename}")

    export_hash_fields = {
        "base_snapshot_manifest": "base_snapshot_manifest_sha256",
        "processor_artifact_manifest": "processor_artifact_manifest_sha256",
        "processor_parity_receipt": "processor_parity_receipt_sha256",
        "training_environment": "training_environment_record_sha256",
        "training_source_manifest": "training_source_manifest_sha256",
        "training_action_value_evidence": (
            "training_action_value_evidence_sha256"
        ),
    }
    for role, field in export_hash_fields.items():
        _require_evidence_hash_link(
            export,
            field,
            str(evidence[role][1]["sha256"]),
            context="model export evidence",
        )

    base = _validate_registered_pc01_base_snapshot(
        evidence["base_snapshot_manifest"][0],
        executable["e0_backbone"][0],
    )
    processor = values["processor_artifact_manifest"]
    parity = values["processor_parity_receipt"]
    training_environment = values["training_environment"]
    training_sources = values["training_source_manifest"]
    action_value = _validate_registered_pc01_action_value_evidence(
        values["training_action_value_evidence"],
        report_sha256=str(export.get("report_sha256", "")),
        run_contract_sha256=str(export.get("run_contract_sha256", "")),
    )
    if processor.get("schema_version") != "table2.pc01-processor-artifacts.v3":
        raise SchemaError("PC-01 processor-artifact evidence schema is not registered")
    if parity.get("schema_version") != "table2.pc01-processor-parity.v1":
        raise SchemaError("PC-01 processor-parity evidence schema is not registered")
    if training_sources.get("schema_version") != "table2.pc01-training-sources.v1":
        raise SchemaError("PC-01 training-source evidence schema is not registered")
    base_directory_sha256 = str(base.get("directory_payload_sha256", ""))
    cross_links = {
        "resolved_config_record_sha256": model[
            "resolved_config_record_sha256"
        ],
        "processor_contract_sha256": model["processor_contract_sha256"],
        "processor_artifact_manifest_sha256": evidence[
            "processor_artifact_manifest"
        ][1]["sha256"],
        "base_snapshot_directory_payload_sha256": base_directory_sha256,
        "training_environment_record_sha256": evidence["training_environment"][1][
            "sha256"
        ],
        "training_source_manifest_sha256": evidence["training_source_manifest"][1][
            "sha256"
        ],
        "training_action_value_evidence_sha256": evidence[
            "training_action_value_evidence"
        ][1]["sha256"],
        "training_environment_sha256": export[
            "training_environment_source_sha256"
        ],
        "run_contract_sha256": export["run_contract_sha256"],
    }
    for field, expected in cross_links.items():
        _require_evidence_hash_link(
            parity,
            field,
            str(expected),
            context="processor parity receipt",
        )
    parity = _validate_registered_pc01_processor_parity(
        parity,
        resolved_config_record_sha256=str(model["resolved_config_record_sha256"]),
        processor_contract_sha256=str(model["processor_contract_sha256"]),
        processor_artifact_manifest_sha256=str(
            evidence["processor_artifact_manifest"][1]["sha256"]
        ),
        base_snapshot_directory_payload_sha256=base_directory_sha256,
        training_environment_sha256=str(
            export["training_environment_source_sha256"]
        ),
        training_environment_record_sha256=str(
            evidence["training_environment"][1]["sha256"]
        ),
        training_source_manifest_sha256=str(
            evidence["training_source_manifest"][1]["sha256"]
        ),
        training_action_value_evidence_sha256=str(
            evidence["training_action_value_evidence"][1]["sha256"]
        ),
        run_contract_sha256=str(export["run_contract_sha256"]),
    )

    if export.get("base_snapshot_directory_payload_sha256") != base_directory_sha256:
        raise SchemaError("model export/base-snapshot directory identities differ")
    if export.get("base_snapshot_file_count") != base.get("file_count"):
        raise SchemaError("model export/base-snapshot file counts differ")
    if export.get("training_transformers_version") != training_environment.get(
        "transformers"
    ):
        raise SchemaError("training environment/processor version evidence differs")
    if processor.get("training_environment_sha256") != export.get(
        "training_environment_source_sha256"
    ):
        raise SchemaError("processor/export training-environment source identities differ")
    if processor.get("training_source_manifest_sha256") != evidence[
        "training_source_manifest"
    ][1]["sha256"]:
        raise SchemaError("processor/training-source evidence identities differ")
    if processor.get("training_action_value_evidence_sha256") != evidence[
        "training_action_value_evidence"
    ][1]["sha256"]:
        raise SchemaError("processor/action-value evidence identities differ")
    if processor.get("training_transformers_version") != training_environment.get(
        "transformers"
    ):
        raise SchemaError("processor/training Transformers versions differ")
    if processor.get("implementation_identity") != parity.get(
        "implementation_identity"
    ) or export.get("runtime_processor_implementation") != parity.get(
        "runtime_processor_implementation"
    ):
        raise SchemaError("processor implementation identities differ")
    for other_name, other in (("processor", processor), ("parity", parity)):
        if base.get("model_id") != other.get("model_id") or base.get(
            "revision"
        ) != other.get("revision"):
            raise SchemaError(
                f"base snapshot/{other_name} model identities differ"
            )
    if training_sources.get("git_commit") != training_environment.get("git_commit"):
        raise SchemaError("training source/environment Git commits differ")

    if executable["e0_backbone"][1]["sha256"] != base_directory_sha256:
        raise SchemaError("E0 backbone bytes differ from base-snapshot evidence")
    processor_manifest_sha256 = str(
        evidence["processor_artifact_manifest"][1]["sha256"]
    )
    for role in ("processor_contract", "e0_processor_contract"):
        contract = _processor_contract_from_payload(executable[role][0])
        if contract.processor_config_sha256 != processor_manifest_sha256:
            raise SchemaError(
                f"{role} does not bind the processor-artifact evidence"
            )
        implementation = parity["implementation_identity"]
        if (
            contract.processor_class != implementation["processor_class"]
            or contract.processor_revision != parity["revision"]
        ):
            raise SchemaError(
                f"{role} processor class/revision differs from parity evidence"
            )
    executable_files = {
        "resolved_config.json": "resolved_config",
        "processor_contract.json": "processor_contract",
        "e0_resolved_config.json": "e0_resolved_config",
        "e0_processor_contract.json": "e0_processor_contract",
    }
    for filename, role in executable_files.items():
        file_row = files.get(filename)
        if not isinstance(file_row, Mapping):
            raise SchemaError(f"model export evidence omits {filename}")
        descriptor = executable[role][1]
        if file_row.get("sha256") != descriptor["sha256"] or file_row.get(
            "size_bytes"
        ) != descriptor["size_bytes"]:
            raise SchemaError(
                f"model export evidence differs from executable {role} payload"
            )

    # Re-open every processor file named by the compact evidence against the
    # complete E0 snapshot.  A manifest copied from another snapshot therefore
    # cannot authenticate the executable base directory.
    backbone = executable["e0_backbone"][0]
    processor_files = processor.get("files")
    if not isinstance(processor_files, list) or not processor_files:
        raise SchemaError("processor artifact evidence has no file records")
    seen: set[str] = set()
    for row in processor_files:
        if not isinstance(row, Mapping):
            raise SchemaError("processor artifact evidence row is malformed")
        relative = str(row.get("path", ""))
        relative_path = Path(relative)
        if (
            not relative
            or relative in seen
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise SchemaError("processor artifact evidence path is unsafe or duplicated")
        seen.add(relative)
        target = (backbone / relative_path).resolve()
        if backbone != target and backbone not in target.parents:
            raise SchemaError("processor artifact evidence escaped the base snapshot")
        if not target.is_file() or target.is_symlink():
            raise SchemaError(f"processor artifact evidence file is absent: {relative}")
        if row.get("size_bytes") != target.stat().st_size or row.get(
            "sha256"
        ) != sha256_file(target):
            raise SchemaError(f"processor artifact evidence differs for {relative}")


def _validate_model_evidence_bundle(
    manifest_path: Path,
    model: Mapping[str, Any],
    *,
    path_base: Path | None = None,
    executable_payloads: Mapping[str, tuple[Path, dict[str, Any]]] | None = None,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    bundle = model.get("model_evidence_bundle")
    if not isinstance(bundle, Mapping):
        raise SchemaError("model manifest lacks a versioned model_evidence_bundle")
    expected_keys = {
        "schema_version",
        "producer_schema_version",
        "evidence_status",
        "runtime_readiness_claim",
        "artifacts",
        "bundle_sha256",
    }
    if set(bundle) != expected_keys:
        raise SchemaError("model evidence bundle fields differ from the registered schema")
    exact = {
        "schema_version": MODEL_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "producer_schema_version": MODEL_EVIDENCE_PRODUCER_SCHEMA_VERSION,
        "evidence_status": MODEL_EVIDENCE_STATUS,
        "runtime_readiness_claim": False,
    }
    for field, expected in exact.items():
        if bundle.get(field) != expected:
            raise SchemaError(f"model evidence bundle changed {field}")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        MODEL_EVIDENCE_ROLES
    ):
        raise SchemaError(
            "model evidence artifacts must contain exactly: "
            + ", ".join(MODEL_EVIDENCE_ROLES)
        )
    for role in MODEL_EVIDENCE_ROLES:
        row = artifacts[role]
        if not isinstance(row, Mapping) or any(
            key not in row
            for key in ("kind", "sha256", "size_bytes", "file_count")
        ):
            raise SchemaError(f"model evidence descriptor is malformed: {role}")
    content_identity = {
        **{
            key: bundle[key]
            for key in expected_keys
            if key not in {"bundle_sha256", "artifacts"}
        },
        "artifacts": {
            role: {
                key: bundle["artifacts"][role][key]
                for key in ("kind", "sha256", "size_bytes", "file_count")
            }
            for role in MODEL_EVIDENCE_ROLES
        },
    }
    if bundle.get("bundle_sha256") != sha256_json(content_identity):
        raise SchemaError("model evidence bundle digest differs from its record")
    if model.get("model_evidence_bundle_sha256") != bundle["bundle_sha256"]:
        raise SchemaError("model manifest evidence-bundle hash differs from its record")
    resolved: dict[str, tuple[Path, dict[str, Any]]] = {}
    authenticated: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for role in MODEL_EVIDENCE_ROLES:
        row = artifacts[role]
        if not isinstance(row, Mapping):
            raise SchemaError(f"model evidence descriptor is malformed: {role}")
        raw_path = Path(str(row.get("path", "")))
        if not str(raw_path):
            raise SchemaError(f"model evidence descriptor has no path: {role}")
        candidate = (
            raw_path
            if raw_path.is_absolute()
            else (path_base if path_base is not None else manifest_path.parent)
            / raw_path
        )
        if path_base is not None:
            source, descriptor = _validated_campaign_relative_model_payload(
                path_base,
                row.get("path"),
                label=f"model evidence {role}",
                stored_path=str(row.get("path")),
            )
        else:
            if candidate.is_symlink():
                raise SchemaError(f"model evidence {role} must not be a symlink")
            source = candidate.resolve()
            descriptor = _artifact_payload_descriptor(
                source, stored_path=str(row.get("path"))
            )
        if descriptor["kind"] != "file":
            raise SchemaError(f"model evidence {role} must be one file")
        if source.name != MODEL_EVIDENCE_FILENAMES[role]:
            raise SchemaError(f"model evidence {role} has an unregistered filename")
        for key in ("kind", "sha256", "size_bytes", "file_count"):
            if row.get(key) != descriptor[key]:
                raise SchemaError(
                    f"model evidence {role} {key} differs from payload bytes"
                )
        resolved[role] = (source, dict(row))
        authenticated[role] = (source, descriptor)
    producer = _canonical_model_evidence_json(
        resolved["export_manifest"][0], role="export_manifest"
    )
    if producer.get("schema_version") != bundle["producer_schema_version"]:
        raise SchemaError("model evidence producer schema differs from export bytes")
    _validate_pc01_model_evidence_cross_consistency(
        model=model,
        evidence=resolved,
        executable=executable_payloads,
    )
    if path_base is not None:
        _reauthenticate_campaign_relative_model_payloads(
            path_base,
            artifacts,
            authenticated,
            roles=MODEL_EVIDENCE_ROLES,
            label="model evidence",
        )
    return resolved


def _validate_model_artifact_payloads(
    manifest_path: Path,
    value: Mapping[str, Any],
    *,
    path_base: Path | None = None,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    payloads = value.get("artifact_payloads")
    if not isinstance(payloads, Mapping) or set(payloads) != set(MODEL_PAYLOAD_ROLES):
        raise SchemaError(
            "model manifest artifact_payloads must contain exactly: "
            + ", ".join(MODEL_PAYLOAD_ROLES)
        )
    resolved: dict[str, tuple[Path, dict[str, Any]]] = {}
    authenticated: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for role in MODEL_PAYLOAD_ROLES:
        row = payloads.get(role)
        if not isinstance(row, Mapping):
            raise SchemaError(f"model artifact descriptor is malformed: {role}")
        raw_path = Path(str(row.get("path", "")))
        if not str(raw_path):
            raise SchemaError(f"model artifact descriptor has no path: {role}")
        candidate = (
            raw_path
            if raw_path.is_absolute()
            else (path_base if path_base is not None else manifest_path.parent) / raw_path
        )
        if path_base is not None:
            source, descriptor = _validated_campaign_relative_model_payload(
                path_base,
                row.get("path"),
                label=f"model artifact {role}",
                stored_path=str(row.get("path")),
            )
        else:
            if candidate.is_symlink():
                raise SchemaError(f"model artifact {role} must not be a symlink")
            source = candidate.resolve()
            descriptor = _artifact_payload_descriptor(
                source, stored_path=str(row.get("path"))
            )
        for key in ("kind", "sha256", "size_bytes", "file_count"):
            if row.get(key) != descriptor[key]:
                raise SchemaError(f"model artifact {role} {key} differs from payload bytes")
        hash_field = MODEL_PAYLOAD_HASH_FIELDS[role]
        if value.get(hash_field) != descriptor["sha256"]:
            raise SchemaError(f"model manifest {hash_field} differs from {role} payload")
        resolved[role] = (source, dict(row))
        authenticated[role] = (source, descriptor)

    for role in ("processor_contract", "e0_processor_contract"):
        _processor_contract_from_payload(resolved[role][0])
    try:
        resolved_config_identity = load_resolved_config_identity(
            resolved["resolved_config"][0]
        )
    except ResolvedConfigIdentityError as exc:
        raise SchemaError("selected resolved-config payload is invalid") from exc
    if (
        value.get("resolved_config_record_sha256")
        != resolved_config_identity.record_sha256
    ):
        raise SchemaError(
            "model manifest resolved_config_record_sha256 differs from the "
            "canonical resolved-config mapping"
        )
    parser_source, parser_row = resolved["e0_parser"]
    if not parser_source.is_file():
        raise SchemaError("E0 parser payload must be a source file")
    for key in ("module", "attribute"):
        text = str(parser_row.get(key, "")).strip()
        if not text:
            raise SchemaError(f"E0 parser descriptor requires {key}")
    if value.get("e0_parser_module") != parser_row.get("module") or value.get(
        "e0_parser_attribute"
    ) != parser_row.get("attribute"):
        raise SchemaError("E0 parser module/attribute differs from its payload descriptor")
    if path_base is not None:
        _reauthenticate_campaign_relative_model_payloads(
            path_base,
            payloads,
            authenticated,
            roles=MODEL_PAYLOAD_ROLES,
            label="model artifact",
        )
    return resolved


def _copy_model_payloads(
    *,
    campaign_root: Path,
    seed: int,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    copied: list[Path],
    copy_file: Callable[[Path, Path], Path] | None = None,
) -> tuple[
    Path,
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    copy_one = copy_file or _copy_exact
    sources = _validate_model_artifact_payloads(manifest_path, manifest)
    evidence_sources = _validate_model_evidence_bundle(
        manifest_path,
        manifest,
        executable_payloads=sources,
    )
    frozen_descriptors: dict[str, dict[str, Any]] = {}
    for role in MODEL_PAYLOAD_ROLES:
        source, source_descriptor = sources[role]
        role_root = campaign_root / "frozen" / "model_payloads" / f"seed_{seed}" / role
        if source.is_file():
            destination = role_root / source.name
            copied.append(copy_one(source, destination))
        else:
            if role_root.exists():
                raise Table2Error(f"refusing to overwrite model payload directory: {role_root}")
            copied.extend(
                _copy_tree_exact(
                    source,
                    role_root,
                    copy_file=copy_one,
                )
            )
            destination = role_root
        relative = str(destination.relative_to(campaign_root))
        descriptor = _artifact_payload_descriptor(destination, stored_path=relative)
        for key in ("kind", "sha256", "size_bytes", "file_count"):
            if descriptor[key] != source_descriptor[key]:
                raise SchemaError(f"copied model artifact {role} differs from source")
        if role == "e0_parser":
            descriptor.update(
                {
                    "module": source_descriptor["module"],
                    "attribute": source_descriptor["attribute"],
                }
            )
        frozen_descriptors[role] = descriptor
    frozen_evidence_descriptors: dict[str, dict[str, Any]] = {}
    for role in MODEL_EVIDENCE_ROLES:
        source, source_descriptor = evidence_sources[role]
        destination = (
            campaign_root
            / "frozen"
            / "model_evidence"
            / f"seed_{seed}"
            / role
            / source.name
        )
        copied.append(copy_one(source, destination))
        descriptor = _artifact_payload_descriptor(
            destination,
            stored_path=str(destination.relative_to(campaign_root)),
        )
        for key in ("kind", "sha256", "size_bytes", "file_count"):
            if descriptor[key] != source_descriptor[key]:
                raise SchemaError(f"copied model evidence {role} differs from source")
        frozen_evidence_descriptors[role] = descriptor
    frozen_evidence_bundle = _model_evidence_bundle_value(
        frozen_evidence_descriptors,
        producer_schema_version=MODEL_EVIDENCE_PRODUCER_SCHEMA_VERSION,
    )
    frozen_manifest = dict(manifest)
    frozen_manifest["artifact_payloads"] = frozen_descriptors
    frozen_manifest["model_evidence_bundle"] = frozen_evidence_bundle
    frozen_manifest["model_evidence_bundle_sha256"] = frozen_evidence_bundle[
        "bundle_sha256"
    ]
    frozen_manifest_path = campaign_root / "frozen" / "models" / f"seed_{seed}.json"
    atomic_write_json(frozen_manifest_path, frozen_manifest)
    copied.append(frozen_manifest_path)
    frozen_executable = _validate_model_artifact_payloads(
        frozen_manifest_path,
        frozen_manifest,
        path_base=campaign_root,
    )
    _validate_model_evidence_bundle(
        frozen_manifest_path,
        frozen_manifest,
        path_base=campaign_root,
        executable_payloads=frozen_executable,
    )
    return frozen_manifest_path, frozen_descriptors, frozen_evidence_bundle


def _validate_seed_manifest_coverage(
    paths: Sequence[Path],
    matched_seeds: Sequence[int],
    *,
    kind: str,
    required: bool,
) -> dict[int, Path]:
    if kind not in {"model", "memory"}:
        raise ValueError(f"unknown seed manifest kind: {kind}")
    by_seed: dict[int, Path] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        value = read_json(path)
        raw_seed = value.get("model_seed")
        if type(raw_seed) is not int or int(raw_seed) < 0:
            raise SchemaError(f"{kind} manifest lacks a nonnegative integer model seed: {path}")
        seed = int(raw_seed)
        if seed in by_seed:
            raise SchemaError(f"duplicate {kind} manifest for matched seed {seed}")
        if kind == "model":
            checkpoint_hash = value.get("selected_checkpoint_sha256")
            if not _is_sha256(checkpoint_hash):
                raise SchemaError(f"model manifest lacks selected checkpoint SHA-256: {path}")
            for hash_key in (
                "resolved_config_sha256",
                "resolved_config_record_sha256",
                "processor_contract_sha256",
            ):
                if not _is_sha256(value.get(hash_key)):
                    raise SchemaError(f"model manifest lacks {hash_key}: {path}")
            for text_key in (
                "e0_backbone_id",
                "e0_backbone_revision",
                "e0_parser_id",
                "e0_parser_version",
                "e0_parser_module",
                "e0_parser_attribute",
            ):
                if not str(value.get(text_key) or "").strip():
                    raise SchemaError(f"model manifest lacks {text_key}: {path}")
            for hash_key in (
                "e0_backbone_sha256",
                "e0_resolved_config_sha256",
                "e0_processor_contract_sha256",
                "e0_base_prompt_sha256",
                "e0_parser_sha256",
            ):
                if not _is_sha256(value.get(hash_key)):
                    raise SchemaError(f"model manifest lacks {hash_key}: {path}")
            if (
                value.get("selection_scope") != "validation_only"
                or value.get("checkpoint_selection") != "validation_only"
                or type(value.get("validation_rows_read")) is not int
                or int(value["validation_rows_read"]) <= 0
            ):
                raise SchemaError(
                    "model manifest is not explicitly validation-only selected "
                    f"with positive validation evidence: {path}"
                )
            for field in ("test_rows_read", "locked_test_rows_read"):
                if type(value.get(field)) is not int or int(value[field]) != 0:
                    raise SchemaError(f"model manifest reports nonzero/missing {field}: {path}")
            if required:
                executable = _validate_model_artifact_payloads(path, value)
                _validate_model_evidence_bundle(
                    path,
                    value,
                    executable_payloads=executable,
                )
        else:
            if path.name != "manifest.json":
                raise SchemaError(
                    f"memory manifest must be the manifest.json inside a complete store: {path}"
                )
            _load_and_verify_frozen_memory_store(path.parent)
            if value.get("runtime_writes_allowed") is not False:
                raise SchemaError(f"memory manifest is not frozen read-only: {path}")
            if type(value.get("test_rows_read")) is not int or int(
                value["test_rows_read"]
            ) != 0:
                raise SchemaError(f"memory manifest reports test reads: {path}")
        by_seed[seed] = path
    expected = set(map(int, matched_seeds))
    actual = set(by_seed)
    if required and actual != expected:
        raise SchemaError(
            f"{kind} manifests must cover every matched seed exactly; "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    if actual - expected:
        raise SchemaError(f"{kind} manifests contain unregistered seeds: {sorted(actual - expected)}")
    return by_seed


def _load_and_verify_frozen_memory_store(root: Path) -> Any:
    """Fully load a frozen store so freeze cannot copy a manifest-only shell."""

    try:
        from web_agent.memory.frozen_store import FrozenMemoryStore

        return FrozenMemoryStore.load(root)
    except ImportError as exc:
        raise SchemaError("frozen-memory verifier dependencies are unavailable") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise SchemaError(f"invalid frozen memory store {root}: {exc}") from exc


def _validate_model_memory_source_bindings(
    models: Mapping[int, Path],
    memories: Mapping[int, Path],
    *,
    protocol_source: Path,
) -> None:
    expected_e0_prompt = sha256_file(
        protocol_source.parent / "prompts" / "e0_action_v1.txt"
    )
    for seed, path in sorted(models.items()):
        model = read_json(path)
        if model.get("e0_base_prompt_sha256") != expected_e0_prompt:
            raise SchemaError(f"E0 base-prompt hash differs from frozen prompt for seed {seed}")
    for seed in sorted(set(models).intersection(memories)):
        model = read_json(models[seed])
        memory = read_json(memories[seed])
        if memory.get("checkpoint_sha256") != model.get("selected_checkpoint_sha256"):
            raise SchemaError(f"memory/model checkpoint hashes disagree for seed {seed}")
        if memory.get("resolved_config_sha256") != model.get("resolved_config_sha256"):
            raise SchemaError(f"memory/model resolved-config hashes disagree for seed {seed}")
        if memory.get("resolved_config_record_sha256") != model.get(
            "resolved_config_record_sha256"
        ):
            raise SchemaError(
                "memory/model canonical resolved-config record hashes disagree "
                f"for seed {seed}"
            )
        if memory.get("protocol_sha256") != sha256_file(protocol_source):
            raise SchemaError(f"memory/protocol hashes disagree for seed {seed}")


def _write_ledger(
    path: Path,
    ledger_type: str,
    events: Sequence[Mapping[str, Any]],
) -> str:
    if path.exists() and path.stat().st_size:
        raise Table2Error(f"refusing to overwrite append-only ledger: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = "0" * 64
    with path.open("x" if not path.exists() else "w", encoding="utf-8") as handle:
        for sequence, event in enumerate(events, start=1):
            body = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "CampaignLedgerEvent",
                "ledger": ledger_type,
                "sequence": sequence,
                "previous_record_hash": previous,
                **dict(event),
            }
            record_hash = sha256_json(body)
            handle.write(canonical_json_bytes({**body, "record_hash": record_hash}).decode("utf-8") + "\n")
            previous = record_hash
        handle.flush()
        os.fsync(handle.fileno())
    return previous


def _verify_ledger(path: Path, ledger_type: str) -> tuple[int, str]:
    previous = "0" * 64
    records = read_jsonl(path)
    for sequence, record in enumerate(records, start=1):
        required = {
            "schema_version",
            "record_type",
            "ledger",
            "sequence",
            "previous_record_hash",
            "event_type",
            "timestamp_utc",
            "payload",
            "record_hash",
        }
        if not required.issubset(record):
            raise SchemaError(f"ledger record lacks required fields: {path}:{sequence}")
        if (
            record["schema_version"] != SCHEMA_VERSION
            or record["record_type"] != "CampaignLedgerEvent"
            or record["ledger"] != ledger_type
            or int(record["sequence"]) != sequence
            or record["previous_record_hash"] != previous
        ):
            raise SchemaError(f"ledger identity/chain mismatch: {path}:{sequence}")
        body = {key: value for key, value in record.items() if key != "record_hash"}
        expected = sha256_json(body)
        if record["record_hash"] != expected:
            raise SchemaError(f"ledger hash mismatch: {path}:{sequence}")
        previous = expected
    return len(records), previous


def append_campaign_ledger_event(
    campaign_dir: str | Path,
    *,
    ledger_type: str,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one verified campaign access/deviation record."""

    if ledger_type not in {"access", "deviation"}:
        raise ValueError("ledger_type must be access or deviation")
    root = Path(campaign_dir).resolve()
    path = root / f"{ledger_type}_ledger.jsonl"
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        count, previous = _verify_ledger(path, ledger_type)
        body = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "CampaignLedgerEvent",
            "ledger": ledger_type,
            "sequence": count + 1,
            "previous_record_hash": previous,
            "event_type": event_type,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "payload": dict(payload),
        }
        record = {**body, "record_hash": sha256_json(body)}
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(record).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return record


def _display_source(path: Path, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


def _is_locked_source(path: Path, repo: Path, protocol: Mapping[str, Any]) -> bool:
    locked_mount = str(protocol.get("benchmark", {}).get("locked_mount", "")).strip()
    if not locked_mount:
        return False
    locked_path = _resolve_input(repo, locked_mount)
    resolved = path.resolve()
    return resolved == locked_path or locked_path in resolved.parents


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_environment_manifest(
    value: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
) -> None:
    if value.get("schema_version") != "table2-environment-v2":
        raise SchemaError("environment manifest schema version is not registered")
    required_text = (
        "benchmark",
        "benchmark_version",
        "benchmark_repository",
        "benchmark_revision",
        "benchmark_license",
        "task_definition_version",
        "operating_system",
        "browser",
        "browser_version",
        "playwright_version",
        "controller_id",
        "controller_version",
        "environment_adapter_id",
        "environment_adapter_version",
        "container_digest",
        "hardware",
        "gpu",
        "driver_version",
        "credential_policy",
        "safety_policy_id",
        "safety_policy_version",
        "destructive_action_policy",
        "element_resolution_policy",
        "invalid_action_policy",
        "invalid_task_policy",
        "screenshot_timing_policy",
    )
    for key in required_text:
        text = str(value.get(key, "")).strip()
        if not text or any(marker in text.lower() for marker in ("placeholder", "tbd", "unknown")):
            raise SchemaError(f"environment manifest requires measured {key}")
    if not _is_sha256(value.get("dependency_lock_sha256")):
        raise SchemaError("environment manifest requires dependency_lock_sha256")
    if (
        value.get("dependency_lock_relative_path")
        != FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH
    ):
        raise SchemaError(
            "environment manifest requires the registered dependency-lock path"
        )
    captured_at = str(value.get("captured_at_utc", "")).strip()
    try:
        parsed_capture = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaError("environment manifest captured_at_utc must be ISO-8601") from exc
    if parsed_capture.tzinfo is None:
        raise SchemaError("environment manifest captured_at_utc must include a timezone")

    if protocol is None:
        raise SchemaError(
            "environment validation requires the frozen protocol locked-mount boundary"
        )
    try:
        validate_locked_mount_attestation(value, protocol)
    except LockedMountPreflightError as exc:
        raise SchemaError(f"environment locked-mount attestation is invalid: {exc}") from exc

    reset = value.get("reset")
    if not isinstance(reset, Mapping):
        raise SchemaError("environment manifest requires reset provenance")
    for key in ("implementation_id", "implementation_version"):
        text = str(reset.get(key, "")).strip()
        if not text or any(marker in text.lower() for marker in ("placeholder", "tbd", "unknown")):
            raise SchemaError(f"environment reset provenance requires measured {key}")
    if reset.get("deterministic_start_state") is not True:
        raise SchemaError("environment reset must register deterministic start states")

    state_digester = value.get("environment_state_digester")
    if not isinstance(state_digester, Mapping):
        raise SchemaError("environment manifest requires state-digester identity")
    for key in ("digester_id", "digester_version"):
        text = str(state_digester.get(key, "")).strip()
        if not text or any(
            marker in text.lower() for marker in ("placeholder", "tbd", "unknown")
        ):
            raise SchemaError(
                f"environment state digester requires measured {key}"
            )

    classifier = value.get("infrastructure_classifier")
    if not isinstance(classifier, Mapping):
        raise SchemaError("environment manifest requires infrastructure classifier")
    for key in ("classifier_id", "classifier_version"):
        text = str(classifier.get(key, "")).strip()
        if not text or any(
            marker in text.lower() for marker in ("placeholder", "tbd", "unknown")
        ):
            raise SchemaError(
                f"environment infrastructure classifier requires measured {key}"
            )
    rule_values = classifier.get("rules")
    if not isinstance(rule_values, list) or not rule_values:
        raise SchemaError(
            "environment infrastructure classifier requires exact frozen rules"
        )
    rules: list[WebArenaInfrastructureRule] = []
    for index, rule_value in enumerate(rule_values):
        if not isinstance(rule_value, Mapping) or set(rule_value) != {
            "operation",
            "exception_type",
            "reason_code",
            "failure_class",
        }:
            raise SchemaError(
                f"environment infrastructure rule {index} has an invalid schema"
            )
        try:
            rules.append(WebArenaInfrastructureRule(**dict(rule_value)))
        except (TypeError, ValueError) as exc:
            raise SchemaError(
                f"environment infrastructure rule {index} is invalid: {exc}"
            ) from exc
    keys = [(rule.operation, rule.exception_type) for rule in rules]
    if len(keys) != len(set(keys)):
        raise SchemaError(
            "environment infrastructure rules repeat an operation/exception pair"
        )
    if classifier.get("rules_sha256") != sha256_json(
        [rule.to_dict() for rule in rules]
    ):
        raise SchemaError("environment infrastructure classifier rule hash differs")

    page_settle = value.get("page_settle_policy")
    if not isinstance(page_settle, Mapping):
        raise SchemaError("environment manifest requires a page-settle policy")
    if not str(page_settle.get("policy_id", "")).strip():
        raise SchemaError("environment page-settle policy requires policy_id")
    if type(page_settle.get("network_idle_required")) is not bool:
        raise SchemaError("environment page-settle network-idle flag must be boolean")
    _positive_finite_number(
        page_settle.get("settle_timeout_seconds"),
        "environment.page_settle_policy.settle_timeout_seconds",
    )
    manual_rescue = value.get("manual_rescue_guard")
    if not isinstance(manual_rescue, Mapping):
        raise SchemaError("environment manifest requires a manual-rescue guard")
    for key in ("guard_id", "guard_version"):
        text = str(manual_rescue.get(key, "")).strip()
        if not text or any(
            marker in text.lower() for marker in ("placeholder", "tbd", "unknown")
        ):
            raise SchemaError(
                f"environment manual-rescue guard requires measured {key}"
            )
    protocol_manual_rescue = protocol.get("manual_rescue")
    if not isinstance(protocol_manual_rescue, Mapping):
        raise SchemaError("protocol requires its frozen manual-rescue boundary")
    if manual_rescue.get("evidence_mode") != protocol_manual_rescue.get(
        "evidence_mode"
    ):
        raise SchemaError(
            "environment manual-rescue evidence mode differs from protocol"
        )
    _positive_finite_number(
        value.get("model_call_timeout_seconds"),
        "environment.model_call_timeout_seconds",
    )

    evaluator = value.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise SchemaError("environment manifest requires evaluator provenance")
    for key in ("evaluator_id", "evaluator_version"):
        text = str(evaluator.get(key, "")).strip()
        if not text or any(marker in text.lower() for marker in ("placeholder", "tbd", "unknown")):
            raise SchemaError(f"environment evaluator requires measured {key}")
    evaluator_entrypoint = str(evaluator.get("entrypoint") or "").strip()
    if ":" not in evaluator_entrypoint:
        raise SchemaError(
            "environment evaluator requires an independent module:factory entrypoint"
        )
    if not _is_sha256(evaluator.get("source_sha256")):
        raise SchemaError("environment evaluator requires source_sha256")
    evaluator_source = Path(str(evaluator.get("source_relative_path", "")))
    if (
        not str(evaluator_source)
        or evaluator_source.is_absolute()
        or ".." in evaluator_source.parts
    ):
        raise SchemaError("environment evaluator requires a safe source_relative_path")
    evaluator_module = evaluator_entrypoint.split(":", 1)[0]
    module_stem = evaluator_module.replace(".", "/")
    if evaluator_source.as_posix() not in {
        f"{module_stem}.py",
        f"{module_stem}/__init__.py",
    }:
        raise SchemaError(
            "environment evaluator entrypoint differs from attested source"
        )
    if not _is_sha256(evaluator.get("oracle_rules_sha256")):
        raise SchemaError("environment evaluator requires oracle_rules_sha256")
    if type(evaluator.get("model_based")) is not bool:
        raise SchemaError("environment evaluator model_based must be an exact boolean")
    prompt_hash = evaluator.get("prompt_sha256")
    if evaluator["model_based"] is True:
        if not _is_sha256(prompt_hash):
            raise SchemaError("model-based evaluator requires a frozen prompt_sha256")
    elif prompt_hash is not None:
        raise SchemaError("non-model evaluator must set prompt_sha256 to null")

    if value.get("execution_order_algorithm") != "sha256_offset_permutation_cycle_v1":
        raise SchemaError("environment execution-order algorithm differs from registration")
    taxonomy = value.get("failure_taxonomy")
    registered_taxonomy = (
        "BROWSER_CONTROLLER_ERROR",
        "TASK_RESET_FAILURE",
        "NETWORK_HTTP_INFRASTRUCTURE_FAILURE",
        "SITE_UNAVAILABLE",
        "CAPTCHA_LOGIN_CREDENTIAL_GATE",
        "EVALUATOR_FAILURE",
    )
    if not isinstance(taxonomy, list) or tuple(taxonomy) != registered_taxonomy:
        raise SchemaError("environment failure taxonomy differs from registration")
    viewport = value.get("viewport")
    if not isinstance(viewport, Mapping):
        raise SchemaError("environment manifest requires frozen viewport metadata")
    for key in ("width", "height", "device_scale_factor"):
        if key not in viewport:
            raise SchemaError(f"environment viewport is missing {key}")
        _positive_finite_number(viewport[key], f"environment.viewport.{key}")


def _positive_finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{context} must be a JSON number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise SchemaError(f"{context} must be finite and positive")
    return number


def _validate_registered_protocol(value: Mapping[str, Any]) -> None:
    """Delegate to the runtime's single fail-closed protocol contract."""

    try:
        from web_agent.runtime.protocol import validate_frozen_protocol_mapping

        validate_frozen_protocol_mapping(value)
    except ImportError as exc:
        raise SchemaError("registered runtime protocol validator is unavailable") from exc
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"frozen Table 2 protocol is not registered: {exc}") from exc


def _protocol_selection_mode(protocol: Mapping[str, Any]) -> str:
    selection = protocol.get("selection")
    if not isinstance(selection, Mapping):
        raise SchemaError("frozen protocol lacks selection configuration")
    mode = str(selection.get("mode") or "")
    if mode not in {"pc01_provisional", "three_candidate_final"}:
        raise SchemaError("frozen protocol has an unregistered selection mode")
    return mode


def _validate_protocol_access_boundary(
    protocol: Mapping[str, Any],
    campaign: Mapping[str, Any],
    *,
    pilot_only: bool,
) -> None:
    """Keep runtime oracle blindness separate from one final data opening."""

    campaign_profile = classify_campaign_profile(
        campaign,
        context="protocol-bound campaign",
        default_campaign_mode="evaluation",
    )
    if type(pilot_only) is not bool or pilot_only is not (
        campaign_profile == CAMPAIGN_PROFILE_PILOT
    ):
        raise SchemaError(
            "protocol campaign classification differs from the registered profile"
        )

    benchmark = protocol.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise SchemaError("frozen protocol lacks its benchmark boundary")
    if benchmark.get("allow_locked_reads") is not False:
        raise SchemaError(
            "runtime protocol must keep allow_locked_reads=false; locked benchmark "
            "access belongs only to the frozen campaign access boundary"
        )
    manual_rescue = protocol.get("manual_rescue")
    if not isinstance(manual_rescue, Mapping) or manual_rescue.get(
        "policy"
    ) != "forbidden":
        raise SchemaError("runtime protocol must forbid manual rescue")
    if campaign.get("manual_rescue") != "forbidden":
        raise SchemaError("campaign must set manual_rescue=forbidden")
    if pilot_only:
        expected = {
            "protocol_status": "DEVELOPMENT_FROZEN",
            "evidence_label": PILOT_EVIDENCE_LABEL,
            "paper_table_status": "N/R",
        }
        for key, value in expected.items():
            if protocol.get(key) != value:
                raise SchemaError(f"pilot protocol requires {key}={value}")
        if campaign.get("evidence_label") != PILOT_EVIDENCE_LABEL:
            raise SchemaError("pilot campaign lacks PILOT_ONLY evidence boundary")
        if campaign.get("locked_test_access", "forbidden") != "forbidden":
            raise SchemaError("pilot campaign must remain locked-test blind")
        return

    if (
        protocol.get("protocol_status") == "AWAITING_MODEL_PROMOTION"
        and protocol.get("evidence_label") == "FINAL_TEMPLATE_ONLY"
    ):
        raise SchemaError(
            "the checked final template cannot authorize a campaign; first finish "
            "the registered three-candidate validation comparison, promote its "
            "winner, freeze the eligible final task list, and materialize a final "
            "protocol"
        )
    raise SchemaError(
        "no runnable locked-final protocol is registered in this engineering source"
    )


def _validate_system_overlay(
    value: Mapping[str, Any], system_id: str, source: str | Path
) -> None:
    expected_features = {
        "E0": {
            "trained_pre_action_policy": False,
            "post_action_diagnosis": False,
            "recovery_controller": False,
            "memory_query": False,
            "memory_intervention": False,
            "evaluation_memory_write": False,
        },
        "E1": {
            "trained_pre_action_policy": True,
            "post_action_diagnosis": False,
            "recovery_controller": False,
            "memory_query": False,
            "memory_intervention": False,
            "evaluation_memory_write": False,
        },
        "E2": {
            "trained_pre_action_policy": True,
            "post_action_diagnosis": True,
            "recovery_controller": True,
            "memory_query": False,
            "memory_intervention": False,
            "evaluation_memory_write": False,
        },
        "E3": {
            "trained_pre_action_policy": True,
            "post_action_diagnosis": True,
            "recovery_controller": True,
            "memory_query": True,
            "memory_intervention": True,
            "evaluation_memory_write": False,
        },
    }[system_id]
    if str(value.get("system_id")) != system_id:
        raise SchemaError(f"system overlay {source} does not identify {system_id}")
    if value.get("features") != expected_features:
        raise SchemaError(f"system overlay {source} breaks the registered mechanism matrix")
    expected_policy = "selected_backbone_unadapted" if system_id == "E0" else "selected_checkpoint"
    if value.get("policy_source") != expected_policy:
        raise SchemaError(f"system overlay {source} has wrong policy_source")


def _reject_locked_mount_path(
    source: Path,
    repo: Path,
    protocol: Mapping[str, Any],
    *,
    pilot_only: bool,
) -> None:
    if not pilot_only:
        return
    locked_mount = str(protocol["benchmark"].get("locked_mount", "")).strip()
    if not locked_mount:
        raise SchemaError("protocol must name its locked benchmark mount")
    locked_path = _resolve_input(repo, locked_mount)
    resolved = source.resolve()
    if resolved == locked_path or locked_path in resolved.parents:
        raise SchemaError(f"pilot freeze attempted to read below locked_benchmark_mount: {source}")
    if locked_mount in source.parts or locked_mount in str(source):
        raise SchemaError(f"pilot freeze path references locked_benchmark_mount: {source}")


def _copy_manifests(paths: Sequence[str | Path], repo: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    if paths:
        destination.mkdir(parents=True, exist_ok=True)
    names: set[str] = set()
    for value in paths:
        source = _resolve_input(repo, value)
        if source.name in names:
            raise SchemaError(f"duplicate manifest filename: {source.name}")
        names.add(source.name)
        copied.append(_copy_exact(source, destination / source.name))
    return copied


def _copy_exact(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _copy_tree_exact(
    source: Path,
    destination: Path,
    *,
    copy_file: Callable[[Path, Path], Path],
) -> list[Path]:
    """Copy a directory without following links and return copied files."""

    if source.is_symlink() or not source.is_dir():
        raise SchemaError(f"model payload directory is missing or symlinked: {source}")
    if destination.exists():
        raise Table2Error(
            f"refusing to overwrite model payload directory: {destination}"
        )
    destination.mkdir(parents=True)
    copied: list[Path] = []
    for candidate in sorted(source.rglob("*")):
        if candidate.is_symlink():
            raise SchemaError(
                f"model payload directory contains a symlink: {candidate}"
            )
        relative = candidate.relative_to(source)
        if candidate.is_dir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
        elif candidate.is_file():
            copied.append(copy_file(candidate, destination / relative))
        else:
            raise SchemaError(
                f"model payload directory contains a non-regular entry: {candidate}"
            )
    return copied


def _resolve_input(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else repo / path).resolve()


def _git_identity(repo: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "NO_GIT_COMMIT", True


def _block_base(root: Path, row: Mapping[str, Any]) -> Path:
    return (
        root
        / "paired_blocks"
        / f"seed_{int(row['matched_model_seed'])}"
        / _path_id(str(row["task_id"]))
        / f"repeat_{int(row['repeat_id'])}"
    )


def _system_package(base: Path, attempt_id: int, system_id: str) -> Path:
    return base / f"rerun_{int(attempt_id)}" / system_id


def _path_id(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    cleaned = cleaned.strip("-")
    if not cleaned:
        raise SchemaError(f"unsafe empty artifact path ID: {value!r}")
    return cleaned[:160]
