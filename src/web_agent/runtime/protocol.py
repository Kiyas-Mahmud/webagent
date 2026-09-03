"""Frozen Table 2 system switches, budgets, and deterministic RNG streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import random
from pathlib import Path
from typing import Any, Mapping

from web_agent.runtime.contracts import (
    JsonValue,
    SystemID,
    VersionedRecord,
    canonical_json,
)
from web_agent.runtime.model_calls import REGISTERED_MAX_MODEL_CALLS_PER_EPISODE


REGISTERED_MAX_EXECUTOR_STEPS = 30
REGISTERED_MAX_RECOVERY_ATTEMPTS_PER_INCIDENT = 2
REGISTERED_MAX_RECOVERY_ATTEMPTS_PER_EPISODE = 4
REGISTERED_EPISODE_TIMEOUT_SECONDS = 600.0
REGISTERED_PRE_BROWSER_SETUP_TIMEOUT_SECONDS = 120.0
REGISTERED_RECOVERY_AT_K = 2
REGISTERED_WHOLE_BLOCK_INFRASTRUCTURE_RERUNS = 1
REGISTERED_LOCKED_MOUNT = "locked_benchmark_mount"
REGISTERED_INFRASTRUCTURE_INVALID_REASONS = (
    "BENCHMARK_SERVICE_UNAVAILABLE",
    "BROWSER_CONTROLLER_DISCONNECTED",
    "ENVIRONMENT_RESET_FAILED",
    "FROZEN_DEPENDENCY_UNAVAILABLE",
)

REGISTERED_PARAMETER_PROVIDER_MODE = "deterministic_then_frozen_base_fallback"
REGISTERED_PARAMETER_PROVIDER_FALLBACK = "selected_backbone_unadapted"
REGISTERED_INVALID_PROVIDER_OUTPUT_POLICY = "reject_and_consume_executor_step"
REGISTERED_PARAMETER_PROVIDER_PROMPT = (
    "configs/eval/table2/prompts/parameter_provider_v1.txt"
)
REGISTERED_PARAMETER_PROVIDER_DECODING_PARAMETERS: dict[str, JsonValue] = {
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 128,
}

REGISTERED_MEMORY_RUNTIME_MODE = "immutable_read_only"
REGISTERED_MEMORY_SOURCE_SPLIT = "train"
REGISTERED_MEMORY_EMBEDDING_STAGE = "post_action_memory_task_adapter"
REGISTERED_MEMORY_EMBEDDING_DIMENSION = 768
REGISTERED_MEMORY_NORMALIZATION = "l2"
REGISTERED_MEMORY_SIMILARITY = "cosine"
REGISTERED_MEMORY_TOP_K = 3
REGISTERED_MEMORY_TIE_BREAK = "memory_id_ascending"
REGISTERED_MEMORY_THRESHOLD_SOURCE = "train_only_calibration"
REGISTERED_RNG_ALGORITHM = "sha256_stage_keyed_v1"
REGISTERED_RNG_NAMESPACE_FIELDS = (
    "protocol_id",
    "campaign_seed",
)
REGISTERED_RNG_KEY_FIELDS = (
    "schema_version",
    "record_type",
    "campaign_id",
    "task_id",
    "repeat_id",
    "matched_seed",
    "stage",
    "decision_index",
    "incident_index",
    "attempt_index",
    "stream",
)
REGISTERED_RNG_SYSTEM_ID_IN_KEY = False
REGISTERED_PROTOCOL_EVIDENCE_LABEL = "PILOT_ONLY"
REGISTERED_PAPER_TABLE_STATUS = "N/R"
REGISTERED_PC01_PILOT_PROTOCOL_ID = "table2-pc01-pilot-v1"
REGISTERED_FINAL_PROTOCOL_ID = "table2-final-template-v1"
REGISTERED_PC01_SELECTION_MODE = "pc01_provisional"
REGISTERED_FINAL_SELECTION_MODE = "three_candidate_final"
REGISTERED_PC01_CANDIDATE_IDS = ("qwen2vl_2b_gold_v2_8_dgx",)
REGISTERED_FINAL_CANDIDATE_IDS = (
    "qwen2vl_2b_gold_v2_8_dgx",
    "qwen25vl_7b_gold_v2_8_dgx",
    "internvl35_8b_gold_v2_8_dgx",
)
REGISTERED_BENCHMARK_NAME = "webarena"
REGISTERED_BENCHMARK_ADAPTER = "web_agent.benchmarks.webarena.WebArenaAdapter"
REGISTERED_PC01_BENCHMARK_TASK_MANIFEST = (
    "benchmarks/table2/pilot/task_manifest.json"
)
REGISTERED_FINAL_BENCHMARK_TASK_MANIFEST = (
    "locked_benchmark_mount/table2/final/task_manifest.json"
)
REGISTERED_CAUSAL_PRE_ACTION_INPUTS = (
    "task_instruction",
    "current_observation",
    "causal_history",
)
REGISTERED_CAUSAL_POST_ACTION_INPUTS = (
    "pre_action_observation",
    "executed_action",
    "post_action_observation",
    "executor_status",
)
REGISTERED_FORBIDDEN_RUNTIME_INPUTS = (
    "oracle_success",
    "oracle_failure",
    "oracle_progress",
    "relevance_label",
    "reference_trajectory",
)
REGISTERED_SEALED_VERIFIER_RETURN = "opaque_terminal_signal"
REGISTERED_ARTIFACT_ROOT = "artifacts/table2"
REGISTERED_RESULT_ROOT = "results/table2"
REGISTERED_ARTIFACT_HASH_ALGORITHM = "sha256"
REGISTERED_FAILURE_PROBABILITY_THRESHOLD = 0.5
REGISTERED_NEEDS_RECOVERY_PROBABILITY_THRESHOLD = 0.5
REGISTERED_RECOVERY_PROBABILITY_COMPARATOR = "greater_than_or_equal"
REGISTERED_RECOVERY_POLICY_SIGNAL_ORDER = (
    "predicted_failure",
    "failure_probability_at_or_above_threshold",
    "needs_recovery",
    "needs_recovery_probability_at_or_above_threshold",
)
REGISTERED_RECOVERY_POLICY_TRUTH_TABLE = tuple(
    (f"{value:04b}", value != 0) for value in range(16)
)
REGISTERED_EXECUTOR_TRIGGER_RULE = "trigger_when_status_not_executed"
REGISTERED_LOOP_TRIGGER_RULE = "trigger_when_loop_detected"
REGISTERED_PAIRED_UNIT = "task_id_x_repeat_id_x_matched_seed"
REGISTERED_BOOTSTRAP_UNIT = "task_id"
REGISTERED_BOOTSTRAP_SAMPLES = 10_000
REGISTERED_CONFIDENCE_LEVEL = 0.95
REGISTERED_CAMPAIGN_SEED = 20_250_831
REGISTERED_LOOP_STATE_FINGERPRINT_ALGORITHM = "canonical_sha256_v1"
REGISTERED_LOOP_STATE_FINGERPRINT_FIELDS = (
    "screenshot_sha256",
    "url",
    "title",
    "page_state",
)
REGISTERED_LOOP_ACTION_FINGERPRINT_ALGORITHM = "canonical_sha256_v1"
REGISTERED_LOOP_ACTION_FINGERPRINT_FIELDS = (
    "action_type",
    "parameters",
    "bbox",
)
REGISTERED_LOOP_SIMILARITY_METRIC = "exact_hash_match"
REGISTERED_LOOP_SIMILARITY_THRESHOLD = 1.0
REGISTERED_LOOP_ROLLING_WINDOW = 6
REGISTERED_LOOP_REPETITION_COUNT = 3
REGISTERED_LOOP_DETECT_ABAB = True
REGISTERED_EXECUTION_ORDER_ALGORITHM = "sha256_offset_permutation_cycle_v1"
REGISTERED_EXECUTION_ORDER_KEY_FIELDS = ("campaign_seed", "block_number")
REGISTERED_MANUAL_RESCUE_POLICY = "forbidden"
REGISTERED_MANUAL_RESCUE_EVIDENCE_MODE = "exclusive_controller_input_audit_v1"
REGISTERED_RECOVERY_EVIDENCE_SOURCES = (
    "verified_normal_task_failure_incidents",
    "preregistered_deterministic_scenarios",
)
REGISTERED_HEADLINE_RECOVERY_PARTITION = "normal"
REGISTERED_CONTROLLED_RECOVERY_ROLE = "pilot_diagnostic_only"


class RuntimeStage(str, Enum):
    RESET = "reset"
    PRE_ACTION = "pre_action"
    ACTION_PARAMETERS = "action_parameters"
    POST_ACTION_ASSESSMENT = "post_action_assessment"
    RECOVERY_SHADOW = "recovery_shadow"
    MEMORY_QUERY = "memory_query"
    MEMORY_INTERVENTION = "memory_intervention"
    RECOVERY_RESOLUTION = "recovery_resolution"
    RECOVERY_ASSESSMENT = "recovery_assessment"


@dataclass(frozen=True, slots=True)
class RuntimeBudgets(VersionedRecord):
    max_executor_steps: int = REGISTERED_MAX_EXECUTOR_STEPS
    max_recovery_attempts_per_incident: int = (
        REGISTERED_MAX_RECOVERY_ATTEMPTS_PER_INCIDENT
    )
    max_recovery_attempts_per_episode: int = (
        REGISTERED_MAX_RECOVERY_ATTEMPTS_PER_EPISODE
    )
    max_model_calls_per_episode: int = REGISTERED_MAX_MODEL_CALLS_PER_EPISODE
    episode_timeout_seconds: float = REGISTERED_EPISODE_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        registered = (
            REGISTERED_MAX_EXECUTOR_STEPS,
            REGISTERED_MAX_RECOVERY_ATTEMPTS_PER_INCIDENT,
            REGISTERED_MAX_RECOVERY_ATTEMPTS_PER_EPISODE,
            REGISTERED_MAX_MODEL_CALLS_PER_EPISODE,
            REGISTERED_EPISODE_TIMEOUT_SECONDS,
        )
        actual = (
            self.max_executor_steps,
            self.max_recovery_attempts_per_incident,
            self.max_recovery_attempts_per_episode,
            self.max_model_calls_per_episode,
            float(self.episode_timeout_seconds),
        )
        if actual != registered:
            raise ValueError(
                "Table 2 budgets are registered and immutable: "
                f"expected {registered}, got {actual}"
            )


REGISTERED_BUDGETS = RuntimeBudgets()


def validate_frozen_protocol_mapping(mapping: Mapping[str, Any]) -> None:
    """Fail closed unless a frozen YAML mapping preserves Table 2 semantics.

    This validation deliberately checks scientific behaviour rather than only
    the four integer runtime budgets.  It is safe for the campaign freezer and
    package validator to call before importing browser/model dependencies.
    Unknown additional documentation fields are permitted, but every required
    registered field must be present with its exact type and value.
    """

    if not isinstance(mapping, Mapping):
        raise TypeError("frozen Table 2 protocol must be a mapping")

    def section(name: str) -> Mapping[str, Any]:
        value = mapping.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"frozen protocol requires mapping section {name!r}")
        return value

    def require_exact(
        values: Mapping[str, Any],
        expected: Mapping[str, Any],
        *,
        location: str,
    ) -> None:
        for key, registered in expected.items():
            if key not in values:
                raise ValueError(f"frozen protocol is missing {location}.{key}")
            actual = values[key]
            # Bool is an int subclass, so require identical concrete types for
            # all scalar registration fields before comparing values.
            if type(actual) is not type(registered) or actual != registered:
                raise ValueError(
                    f"frozen protocol {location}.{key} differs from registration: "
                    f"expected {registered!r}, got {actual!r}"
                )

    protocol_id = mapping.get("protocol_id")
    if protocol_id == REGISTERED_PC01_PILOT_PROTOCOL_ID:
        protocol_profile = {
            "schema_version": "1.0",
            "protocol_status": "DEVELOPMENT_FROZEN",
            "evidence_label": REGISTERED_PROTOCOL_EVIDENCE_LABEL,
            "paper_table_status": REGISTERED_PAPER_TABLE_STATUS,
        }
        selection_profile = {
            "mode": REGISTERED_PC01_SELECTION_MODE,
            "required_candidate_count": 1,
            "checkpoint_selection": "validation_only",
            "locked_task_eligible": False,
        }
        candidate_ids = REGISTERED_PC01_CANDIDATE_IDS
        benchmark_task_manifest = REGISTERED_PC01_BENCHMARK_TASK_MANIFEST
    elif protocol_id == REGISTERED_FINAL_PROTOCOL_ID:
        protocol_profile = {
            "schema_version": "1.0",
            "protocol_status": "AWAITING_MODEL_PROMOTION",
            "evidence_label": "FINAL_TEMPLATE_ONLY",
            # This checked template cannot authorize a locked campaign.  A
            # materialized final protocol is frozen only after the registered
            # three-candidate validation comparison and task eligibility pass.
            "paper_table_status": REGISTERED_PAPER_TABLE_STATUS,
        }
        selection_profile = {
            "mode": REGISTERED_FINAL_SELECTION_MODE,
            "required_candidate_count": 3,
            "checkpoint_selection": "validation_only",
            "locked_task_eligible": False,
        }
        candidate_ids = REGISTERED_FINAL_CANDIDATE_IDS
        benchmark_task_manifest = REGISTERED_FINAL_BENCHMARK_TASK_MANIFEST
    else:
        raise ValueError(
            "frozen protocol protocol_id must identify the registered PC-01 "
            "pilot or not-yet-runnable final-selection template"
        )
    require_exact(mapping, protocol_profile, location="protocol")
    selection = section("selection")
    require_exact(selection, selection_profile, location="selection")
    configured_candidate_ids = selection.get("candidate_ids")
    if (
        not isinstance(configured_candidate_ids, list)
        or tuple(configured_candidate_ids) != candidate_ids
    ):
        raise ValueError(
            "frozen protocol selection.candidate_ids differs from the registered "
            f"{selection_profile['mode']} candidate gate"
        )
    expected_selection_keys = {*selection_profile, "candidate_ids"}
    if set(selection) != expected_selection_keys:
        raise ValueError(
            "frozen protocol selection must contain exactly the registered keys"
        )
    causal = section("causal_boundary")
    for key, registered in (
        ("pre_action_inputs", REGISTERED_CAUSAL_PRE_ACTION_INPUTS),
        ("post_action_inputs", REGISTERED_CAUSAL_POST_ACTION_INPUTS),
        ("forbidden_runtime_inputs", REGISTERED_FORBIDDEN_RUNTIME_INPUTS),
    ):
        actual = causal.get(key)
        if not isinstance(actual, list) or tuple(actual) != registered:
            raise ValueError(
                f"frozen protocol causal_boundary.{key} differs from registration"
            )
    require_exact(
        causal,
        {"sealed_verifier_returns": REGISTERED_SEALED_VERIFIER_RETURN},
        location="causal_boundary",
    )
    require_exact(
        section("budgets"),
        {
            "executor_requests_per_episode": REGISTERED_MAX_EXECUTOR_STEPS,
            "recovery_attempts_per_incident": (
                REGISTERED_MAX_RECOVERY_ATTEMPTS_PER_INCIDENT
            ),
            "recovery_attempts_per_episode": (
                REGISTERED_MAX_RECOVERY_ATTEMPTS_PER_EPISODE
            ),
            "model_calls_per_episode": REGISTERED_MAX_MODEL_CALLS_PER_EPISODE,
            "recovery_at_k": REGISTERED_RECOVERY_AT_K,
            "task_timeout_seconds": int(REGISTERED_EPISODE_TIMEOUT_SECONDS),
            "pre_browser_setup_timeout_seconds": int(
                REGISTERED_PRE_BROWSER_SETUP_TIMEOUT_SECONDS
            ),
            "whole_block_infrastructure_reruns": (
                REGISTERED_WHOLE_BLOCK_INFRASTRUCTURE_RERUNS
            ),
            "abort_counts_as_recovery_attempt": True,
            "abort_executor_steps": 0,
        },
        location="budgets",
    )
    if tuple(section("budgets").get("infrastructure_invalid_reasons", ())) != (
        REGISTERED_INFRASTRUCTURE_INVALID_REASONS
    ):
        raise ValueError(
            "frozen protocol infrastructure-invalid reasons differ from registration"
        )
    # Validate the locked-boundary declaration before resolving any protocol
    # asset path.  A protocol that tries to relocate or relax that boundary
    # must fail for the boundary drift itself, not for a later asset lookup.
    require_exact(
        section("benchmark"),
        {
            "name": REGISTERED_BENCHMARK_NAME,
            "adapter": REGISTERED_BENCHMARK_ADAPTER,
            "task_manifest": benchmark_task_manifest,
            "locked_mount": REGISTERED_LOCKED_MOUNT,
            "allow_locked_reads": False,
            "invalidate_if_successful_after_reset": True,
        },
        location="benchmark",
    )
    require_exact(
        section("manual_rescue"),
        {
            "policy": REGISTERED_MANUAL_RESCUE_POLICY,
            "evidence_mode": REGISTERED_MANUAL_RESCUE_EVIDENCE_MODE,
        },
        location="manual_rescue",
    )
    if set(section("manual_rescue")) != {"policy", "evidence_mode"}:
        raise ValueError(
            "frozen protocol manual_rescue must contain exactly the registered keys"
        )
    require_exact(
        section("parameter_provider"),
        {
            "mode": REGISTERED_PARAMETER_PROVIDER_MODE,
            "deterministic_first": True,
            "fallback_policy_source": REGISTERED_PARAMETER_PROVIDER_FALLBACK,
            "prompt": REGISTERED_PARAMETER_PROVIDER_PROMPT,
            "shared_across_systems": True,
            "invalid_output_policy": REGISTERED_INVALID_PROVIDER_OUTPUT_POLICY,
        },
        location="parameter_provider",
    )
    decoding_parameters = section("parameter_provider").get(
        "decoding_parameters"
    )
    if not isinstance(decoding_parameters, Mapping):
        raise ValueError(
            "frozen protocol requires mapping section "
            "'parameter_provider.decoding_parameters'"
        )
    require_exact(
        decoding_parameters,
        REGISTERED_PARAMETER_PROVIDER_DECODING_PARAMETERS,
        location="parameter_provider.decoding_parameters",
    )
    if set(decoding_parameters) != set(
        REGISTERED_PARAMETER_PROVIDER_DECODING_PARAMETERS
    ):
        raise ValueError(
            "frozen protocol parameter_provider.decoding_parameters must contain "
            "exactly the registered keys"
        )
    require_exact(
        section("loop_rule"),
        {
            "state_fingerprint_algorithm": REGISTERED_LOOP_STATE_FINGERPRINT_ALGORITHM,
            "action_target_fingerprint_algorithm": REGISTERED_LOOP_ACTION_FINGERPRINT_ALGORITHM,
            "perceptual_similarity_metric": REGISTERED_LOOP_SIMILARITY_METRIC,
            "perceptual_similarity_threshold": REGISTERED_LOOP_SIMILARITY_THRESHOLD,
            "rolling_window": REGISTERED_LOOP_ROLLING_WINDOW,
            "equivalent_repetition_count": REGISTERED_LOOP_REPETITION_COUNT,
            "detect_abab_cycle": REGISTERED_LOOP_DETECT_ABAB,
        },
        location="loop_rule",
    )
    loop = section("loop_rule")
    if tuple(loop.get("state_fingerprint_fields", ())) != (
        REGISTERED_LOOP_STATE_FINGERPRINT_FIELDS
    ):
        raise ValueError("frozen loop state fingerprint fields differ from registration")
    if tuple(loop.get("action_target_fingerprint_fields", ())) != (
        REGISTERED_LOOP_ACTION_FINGERPRINT_FIELDS
    ):
        raise ValueError("frozen loop action fingerprint fields differ from registration")
    require_exact(
        section("execution_order"),
        {
            "algorithm": REGISTERED_EXECUTION_ORDER_ALGORITHM,
            "systems_once_per_block": True,
        },
        location="execution_order",
    )
    if tuple(section("execution_order").get("key_fields", ())) != (
        REGISTERED_EXECUTION_ORDER_KEY_FIELDS
    ):
        raise ValueError("frozen execution-order key fields differ from registration")
    require_exact(
        section("memory"),
        {
            "runtime_mode": REGISTERED_MEMORY_RUNTIME_MODE,
            "source_split": REGISTERED_MEMORY_SOURCE_SPLIT,
            "embedding_stage": REGISTERED_MEMORY_EMBEDDING_STAGE,
            "embedding_dimension": REGISTERED_MEMORY_EMBEDDING_DIMENSION,
            "normalization": REGISTERED_MEMORY_NORMALIZATION,
            "similarity": REGISTERED_MEMORY_SIMILARITY,
            "top_k": REGISTERED_MEMORY_TOP_K,
            "tie_break": REGISTERED_MEMORY_TIE_BREAK,
            "admission_threshold_source": REGISTERED_MEMORY_THRESHOLD_SOURCE,
            "same_task_exclusion": True,
            "duplicate_exclusion": True,
            "e3_shadow_decision": "required_before_retrieval",
            "e2_queries": "forbidden",
            "e3_writes": "forbidden",
        },
        location="memory",
    )
    recovery_trigger = section("recovery_trigger")
    require_exact(
        recovery_trigger,
        {
            "failure_probability_threshold": (
                REGISTERED_FAILURE_PROBABILITY_THRESHOLD
            ),
            "needs_recovery_probability_threshold": (
                REGISTERED_NEEDS_RECOVERY_PROBABILITY_THRESHOLD
            ),
            "probability_comparator": REGISTERED_RECOVERY_PROBABILITY_COMPARATOR,
            "executor_status_rule": REGISTERED_EXECUTOR_TRIGGER_RULE,
            "loop_rule": REGISTERED_LOOP_TRIGGER_RULE,
        },
        location="recovery_trigger",
    )
    policy_signal_order = recovery_trigger.get("policy_signal_order")
    if (
        not isinstance(policy_signal_order, list)
        or tuple(policy_signal_order) != REGISTERED_RECOVERY_POLICY_SIGNAL_ORDER
    ):
        raise ValueError(
            "frozen protocol recovery_trigger.policy_signal_order differs from "
            "registration"
        )
    truth_table = recovery_trigger.get("policy_truth_table")
    registered_truth_table = dict(REGISTERED_RECOVERY_POLICY_TRUTH_TABLE)
    if not isinstance(truth_table, Mapping) or set(truth_table) != set(
        registered_truth_table
    ):
        raise ValueError(
            "frozen protocol recovery_trigger.policy_truth_table keys differ from "
            "registration"
        )
    for key, registered in registered_truth_table.items():
        actual = truth_table[key]
        if type(actual) is not bool or actual is not registered:
            raise ValueError(
                "frozen protocol recovery_trigger.policy_truth_table differs from "
                f"registration at {key}"
            )
    expected_recovery_keys = {
        "failure_probability_threshold",
        "needs_recovery_probability_threshold",
        "probability_comparator",
        "policy_signal_order",
        "policy_truth_table",
        "executor_status_rule",
        "loop_rule",
    }
    if set(recovery_trigger) != expected_recovery_keys:
        raise ValueError(
            "frozen protocol recovery_trigger must contain exactly the registered keys"
        )
    require_exact(
        section("verification"),
        {
            "attach_labels_after_action_completion": True,
            "runtime_can_read_full_labels": False,
        },
        location="verification",
    )
    require_exact(
        section("verification"),
        {
            "final_success_source": "official_webarena_evaluator",
            "headline_recovery_partition": REGISTERED_HEADLINE_RECOVERY_PARTITION,
            "controlled_recovery_role": REGISTERED_CONTROLLED_RECOVERY_ROLE,
        },
        location="verification",
    )
    recovery_sources = section("verification").get("recovery_evidence_sources")
    if (
        not isinstance(recovery_sources, list)
        or tuple(recovery_sources) != REGISTERED_RECOVERY_EVIDENCE_SOURCES
    ):
        raise ValueError(
            "frozen protocol verification.recovery_evidence_sources differs "
            "from registration"
        )
    if set(section("verification")) != {
        "final_success_source",
        "recovery_evidence_sources",
        "headline_recovery_partition",
        "controlled_recovery_role",
        "attach_labels_after_action_completion",
        "runtime_can_read_full_labels",
    }:
        raise ValueError(
            "frozen protocol verification must contain exactly the registered keys"
        )
    require_exact(
        section("randomness"),
        {
            "algorithm": REGISTERED_RNG_ALGORITHM,
            "system_id_in_key": REGISTERED_RNG_SYSTEM_ID_IN_KEY,
            "shared_across_systems_before_intervention": True,
        },
        location="randomness",
    )
    key_fields = section("randomness").get("key_fields")
    if not isinstance(key_fields, list) or tuple(key_fields) != REGISTERED_RNG_KEY_FIELDS:
        raise ValueError("frozen protocol randomness.key_fields differ from registration")
    namespace_fields = section("randomness").get("namespace_fields")
    if (
        not isinstance(namespace_fields, list)
        or tuple(namespace_fields) != REGISTERED_RNG_NAMESPACE_FIELDS
    ):
        raise ValueError(
            "frozen protocol randomness.namespace_fields differ from registration"
        )
    require_exact(
        section("artifacts"),
        {
            "root": REGISTERED_ARTIFACT_ROOT,
            "paired_block_first": True,
            "append_only_jsonl": True,
            "hash_algorithm": REGISTERED_ARTIFACT_HASH_ALGORITHM,
            "raw_artifacts_in_git": "forbidden",
            "result_root": REGISTERED_RESULT_ROOT,
        },
        location="artifacts",
    )
    require_exact(
        section("statistics"),
        {
            "paired_unit": REGISTERED_PAIRED_UNIT,
            "bootstrap_unit": REGISTERED_BOOTSTRAP_UNIT,
            "bootstrap_samples": REGISTERED_BOOTSTRAP_SAMPLES,
            "confidence_level": REGISTERED_CONFIDENCE_LEVEL,
            "seed": REGISTERED_CAMPAIGN_SEED,
        },
        location="statistics",
    )


@dataclass(frozen=True, slots=True)
class SystemSwitches(VersionedRecord):
    system_id: SystemID
    trained_pre_action_policy: bool
    post_action_diagnosis: bool
    recovery_controller: bool
    memory_query: bool
    memory_intervention: bool
    evaluation_memory_write: bool = False

    def __post_init__(self) -> None:
        expected = SYSTEM_SWITCH_MATRIX.get(self.system_id)
        if expected is None:
            raise ValueError(f"unregistered Table 2 system: {self.system_id}")
        actual = (
            self.trained_pre_action_policy,
            self.post_action_diagnosis,
            self.recovery_controller,
            self.memory_query,
            self.memory_intervention,
            self.evaluation_memory_write,
        )
        if actual != expected:
            raise ValueError(
                f"{self.system_id.value} switches differ from the registered matrix: "
                f"expected {expected}, got {actual}"
            )


SYSTEM_SWITCH_MATRIX: dict[SystemID, tuple[bool, bool, bool, bool, bool, bool]] = {
    SystemID.E0: (False, False, False, False, False, False),
    SystemID.E1: (True, False, False, False, False, False),
    SystemID.E2: (True, True, True, False, False, False),
    SystemID.E3: (True, True, True, True, True, False),
}


def switches_for(system_id: SystemID | str) -> SystemSwitches:
    resolved_id = SystemID(system_id)
    values = SYSTEM_SWITCH_MATRIX[resolved_id]
    return SystemSwitches(
        system_id=resolved_id,
        trained_pre_action_policy=values[0],
        post_action_diagnosis=values[1],
        recovery_controller=values[2],
        memory_query=values[3],
        memory_intervention=values[4],
        evaluation_memory_write=values[5],
    )


@dataclass(frozen=True, slots=True)
class LoopRule(VersionedRecord):
    state_fingerprint_algorithm: str = REGISTERED_LOOP_STATE_FINGERPRINT_ALGORITHM
    state_fingerprint_fields: tuple[str, ...] = REGISTERED_LOOP_STATE_FINGERPRINT_FIELDS
    action_target_fingerprint_algorithm: str = (
        REGISTERED_LOOP_ACTION_FINGERPRINT_ALGORITHM
    )
    action_target_fingerprint_fields: tuple[str, ...] = (
        REGISTERED_LOOP_ACTION_FINGERPRINT_FIELDS
    )
    perceptual_similarity_metric: str = REGISTERED_LOOP_SIMILARITY_METRIC
    perceptual_similarity_threshold: float = REGISTERED_LOOP_SIMILARITY_THRESHOLD
    rolling_window: int = REGISTERED_LOOP_ROLLING_WINDOW
    equivalent_repetition_count: int = REGISTERED_LOOP_REPETITION_COUNT
    detect_abab_cycle: bool = REGISTERED_LOOP_DETECT_ABAB

    def __post_init__(self) -> None:
        if (
            type(self.state_fingerprint_algorithm) is not str
            or type(self.state_fingerprint_fields) is not tuple
            or any(type(value) is not str for value in self.state_fingerprint_fields)
            or type(self.action_target_fingerprint_algorithm) is not str
            or type(self.action_target_fingerprint_fields) is not tuple
            or any(
                type(value) is not str
                for value in self.action_target_fingerprint_fields
            )
            or type(self.perceptual_similarity_metric) is not str
            or type(self.perceptual_similarity_threshold) is not float
            or type(self.rolling_window) is not int
            or type(self.equivalent_repetition_count) is not int
            or type(self.detect_abab_cycle) is not bool
        ):
            raise TypeError("Table 2 loop-rule fields require their exact registered types")
        registered = (
            REGISTERED_LOOP_STATE_FINGERPRINT_ALGORITHM,
            REGISTERED_LOOP_STATE_FINGERPRINT_FIELDS,
            REGISTERED_LOOP_ACTION_FINGERPRINT_ALGORITHM,
            REGISTERED_LOOP_ACTION_FINGERPRINT_FIELDS,
            REGISTERED_LOOP_SIMILARITY_METRIC,
            REGISTERED_LOOP_SIMILARITY_THRESHOLD,
            REGISTERED_LOOP_ROLLING_WINDOW,
            REGISTERED_LOOP_REPETITION_COUNT,
            REGISTERED_LOOP_DETECT_ABAB,
        )
        actual = (
            self.state_fingerprint_algorithm,
            tuple(self.state_fingerprint_fields),
            self.action_target_fingerprint_algorithm,
            tuple(self.action_target_fingerprint_fields),
            self.perceptual_similarity_metric,
            float(self.perceptual_similarity_threshold),
            self.rolling_window,
            self.equivalent_repetition_count,
            self.detect_abab_cycle,
        )
        if actual != registered:
            raise ValueError(
                "Table 2 loop semantics are registered and immutable: "
                f"expected {registered}, got {actual}"
            )


@dataclass(frozen=True, slots=True)
class RecoveryTriggerConfig(VersionedRecord):
    """Typed, immutable policy/executor/loop trigger semantics."""

    failure_probability_threshold: float = REGISTERED_FAILURE_PROBABILITY_THRESHOLD
    needs_recovery_probability_threshold: float = (
        REGISTERED_NEEDS_RECOVERY_PROBABILITY_THRESHOLD
    )
    probability_comparator: str = REGISTERED_RECOVERY_PROBABILITY_COMPARATOR
    policy_signal_order: tuple[str, ...] = REGISTERED_RECOVERY_POLICY_SIGNAL_ORDER
    policy_truth_table: tuple[tuple[str, bool], ...] = (
        REGISTERED_RECOVERY_POLICY_TRUTH_TABLE
    )
    executor_status_rule: str = REGISTERED_EXECUTOR_TRIGGER_RULE
    loop_rule: str = REGISTERED_LOOP_TRIGGER_RULE

    def __post_init__(self) -> None:
        if (
            type(self.failure_probability_threshold) is not float
            or type(self.needs_recovery_probability_threshold) is not float
            or type(self.probability_comparator) is not str
            or type(self.policy_signal_order) is not tuple
            or any(type(value) is not str for value in self.policy_signal_order)
            or type(self.policy_truth_table) is not tuple
            or any(
                type(row) is not tuple
                or len(row) != 2
                or type(row[0]) is not str
                or type(row[1]) is not bool
                for row in self.policy_truth_table
            )
            or type(self.executor_status_rule) is not str
            or type(self.loop_rule) is not str
        ):
            raise TypeError(
                "Table 2 recovery-trigger fields require their exact registered types"
            )
        registered = (
            REGISTERED_FAILURE_PROBABILITY_THRESHOLD,
            REGISTERED_NEEDS_RECOVERY_PROBABILITY_THRESHOLD,
            REGISTERED_RECOVERY_PROBABILITY_COMPARATOR,
            REGISTERED_RECOVERY_POLICY_SIGNAL_ORDER,
            REGISTERED_RECOVERY_POLICY_TRUTH_TABLE,
            REGISTERED_EXECUTOR_TRIGGER_RULE,
            REGISTERED_LOOP_TRIGGER_RULE,
        )
        actual = (
            self.failure_probability_threshold,
            self.needs_recovery_probability_threshold,
            self.probability_comparator,
            self.policy_signal_order,
            self.policy_truth_table,
            self.executor_status_rule,
            self.loop_rule,
        )
        if actual != registered:
            raise ValueError(
                "Table 2 recovery-trigger semantics are registered and immutable"
            )

    def policy_trigger(self, signals: tuple[bool, bool, bool, bool]) -> bool:
        if type(signals) is not tuple or len(signals) != len(self.policy_signal_order):
            raise TypeError("recovery-trigger policy signals must use the registered tuple")
        if any(type(value) is not bool for value in signals):
            raise TypeError("recovery-trigger policy signals must be exact booleans")
        key = "".join("1" if value else "0" for value in signals)
        truth_table = dict(self.policy_truth_table)
        if key not in truth_table:  # pragma: no cover - constructor closes the table
            raise RuntimeError("registered recovery-trigger truth table is incomplete")
        return truth_table[key]


REGISTERED_RECOVERY_TRIGGER_CONFIG = RecoveryTriggerConfig()


@dataclass(frozen=True, slots=True)
class RuntimeProtocol(VersionedRecord):
    protocol_id: str
    campaign_id: str
    campaign_seed: int = REGISTERED_CAMPAIGN_SEED
    budgets: RuntimeBudgets = field(default_factory=RuntimeBudgets)
    loop_rule: LoopRule = field(default_factory=LoopRule)
    recovery_trigger: RecoveryTriggerConfig = field(
        default_factory=RecoveryTriggerConfig
    )
    oracle_blind_recovery: bool = True
    pre_action_memory_enabled: bool = False
    evaluation_memory_writes_enabled: bool = False
    require_frozen_stage_seed_plan: bool = False
    provider_id: str = "unregistered-provider"
    provider_version: str = "unregistered"
    benchmark_id: str = "unregistered-benchmark"
    benchmark_version: str = "unregistered"
    evaluation_mode: bool = False
    provider_prompt_sha256: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.protocol_id.strip() or not self.campaign_id.strip():
            raise ValueError("protocol_id and campaign_id must be non-empty")
        if not self.oracle_blind_recovery:
            raise ValueError("primary E0-E3 recovery must be oracle blind")
        if self.pre_action_memory_enabled:
            raise ValueError("primary E3 memory is post-failure only")
        if self.evaluation_memory_writes_enabled:
            raise ValueError("primary E3 memory is immutable/read-only")
        # RuntimeBudgets validates exact registered values on construction.
        if self.budgets != REGISTERED_BUDGETS:
            raise ValueError("runtime budgets differ from the registered Table 2 budgets")
        if self.recovery_trigger != REGISTERED_RECOVERY_TRIGGER_CONFIG:
            raise ValueError(
                "runtime recovery trigger differs from the registered Table 2 semantics"
            )
        if self.evaluation_mode:
            if self.provider_id != REGISTERED_PARAMETER_PROVIDER_MODE:
                raise ValueError("evaluation runtime provider mode is not registered")
            if self.provider_prompt_sha256 is None or len(self.provider_prompt_sha256) != 64:
                raise ValueError("evaluation runtime requires the frozen provider prompt hash")

    def switches_for(self, system_id: SystemID | str) -> SystemSwitches:
        return switches_for(system_id)

    def rng_factory(self) -> "StageRNGFactory":
        return StageRNGFactory(
            protocol_id=self.protocol_id,
            campaign_id=self.campaign_id,
            campaign_seed=self.campaign_seed,
        )


@dataclass(frozen=True, slots=True)
class StageRNGKey(VersionedRecord):
    campaign_id: str
    task_id: str
    repeat_id: int
    matched_seed: int
    stage: RuntimeStage
    decision_index: int
    incident_index: int = 0
    attempt_index: int = 0
    stream: str = "default"

    def __post_init__(self) -> None:
        if not self.campaign_id or not self.task_id or not self.stream:
            raise ValueError("stage RNG identifiers must be non-empty")
        if min(self.repeat_id, self.decision_index, self.incident_index, self.attempt_index) < 0:
            raise ValueError("stage RNG indices cannot be negative")


@dataclass(frozen=True, slots=True)
class StageRNGFactory:
    """Order-independent RNG factory shared by matched E0-E3 runs.

    System ID is intentionally absent from the key.  Therefore E1/E2/E3 draw
    identical randomness at every shared stage until an enabled feature causes
    their observable histories to diverge.
    """

    protocol_id: str
    campaign_id: str
    campaign_seed: int

    def key(
        self,
        *,
        task_id: str,
        repeat_id: int,
        matched_seed: int,
        stage: RuntimeStage | str,
        step_index: int | None = None,
        decision_index: int | None = None,
        incident_index: int = 0,
        attempt_index: int = 0,
        stream: str = "default",
    ) -> StageRNGKey:
        if decision_index is None:
            decision_index = step_index
        elif step_index is not None and decision_index != step_index:
            raise ValueError("step_index alias disagrees with decision_index")
        if decision_index is None:
            raise TypeError("decision_index (or legacy step_index alias) is required")
        return StageRNGKey(
            campaign_id=self.campaign_id,
            task_id=task_id,
            repeat_id=repeat_id,
            matched_seed=matched_seed,
            stage=RuntimeStage(stage),
            decision_index=decision_index,
            incident_index=incident_index,
            attempt_index=attempt_index,
            stream=stream,
        )

    def seed_for(self, **kwargs: object) -> int:
        key = self.key(**kwargs)  # type: ignore[arg-type]
        return self.seed_for_key(key)

    def seed_for_key(self, key: StageRNGKey) -> int:
        if key.campaign_id != self.campaign_id:
            raise ValueError("stage RNG key belongs to another campaign")
        namespace = {
            "protocol_id": self.protocol_id,
            "campaign_seed": self.campaign_seed,
        }
        key_payload = key.to_dict()
        if tuple(namespace) != REGISTERED_RNG_NAMESPACE_FIELDS:
            raise RuntimeError("stage RNG namespace serialization drifted")
        if tuple(key_payload) != REGISTERED_RNG_KEY_FIELDS:
            raise RuntimeError("StageRNGKey serialization drifted from registration")
        if "system_id" in key_payload:
            raise RuntimeError("StageRNGKey must remain system-independent")
        digest = hashlib.sha256(
            canonical_json(
                {
                    "algorithm": REGISTERED_RNG_ALGORITHM,
                    "namespace": namespace,
                    "key": key_payload,
                }
            ).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def random(self, **kwargs: object) -> random.Random:
        return random.Random(self.seed_for(**kwargs))

    def seed_plan(
        self,
        *,
        task_id: str,
        repeat_id: int,
        matched_seed: int,
        decision_index: int = 0,
    ) -> "StageSeedPlan":
        seeds = {
            stage.value: self.seed_for(
                task_id=task_id,
                repeat_id=repeat_id,
                matched_seed=matched_seed,
                stage=stage,
                decision_index=decision_index,
            )
            for stage in RuntimeStage
        }
        return StageSeedPlan(
            protocol_id=self.protocol_id,
            campaign_id=self.campaign_id,
            campaign_seed=self.campaign_seed,
            task_id=task_id,
            repeat_id=repeat_id,
            matched_seed=matched_seed,
            decision_index=decision_index,
            stage_seeds=seeds,
        )

    def validate_seed_plan(self, plan: "StageSeedPlan") -> None:
        expected = self.seed_plan(
            task_id=plan.task_id,
            repeat_id=plan.repeat_id,
            matched_seed=plan.matched_seed,
            decision_index=plan.decision_index,
        )
        if plan != expected:
            raise ValueError("frozen stage-seed plan differs from runtime derivation")


@dataclass(frozen=True, slots=True)
class StageSeedPlan(VersionedRecord):
    """Schedule-level decision-index anchor for the exact RNG namespace."""

    protocol_id: str
    campaign_id: str
    campaign_seed: int
    task_id: str
    repeat_id: int
    matched_seed: int
    decision_index: int
    stage_seeds: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.protocol_id or not self.campaign_id or not self.task_id:
            raise ValueError("stage-seed plan identities must be non-empty")
        if self.repeat_id < 0 or self.decision_index < 0:
            raise ValueError("stage-seed plan indices cannot be negative")
        if set(self.stage_seeds) != {stage.value for stage in RuntimeStage}:
            raise ValueError("stage-seed plan must contain every registered stage")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.stage_seeds.values()
        ):
            raise TypeError("stage-seed plan values must be integers")


@dataclass(frozen=True, slots=True)
class StageSeedUse(VersionedRecord):
    """Evidence binding one consumed RNG seed to its key and namespace."""

    protocol_id: str
    campaign_seed: int
    key: StageRNGKey
    seed: int

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise ValueError("RNG use requires a protocol identity")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("RNG seed must be an integer")


@dataclass(frozen=True, slots=True)
class SystemOverlay(VersionedRecord):
    system_id: SystemID
    display_name: str
    policy_source: str
    switches: SystemSwitches


@dataclass(frozen=True, slots=True)
class ProtocolBundle(VersionedRecord):
    protocol: RuntimeProtocol
    systems: tuple[SystemOverlay, ...]
    source_path: str
    source_sha256: str

    def system(self, system_id: SystemID | str) -> SystemOverlay:
        resolved = SystemID(system_id)
        return next(item for item in self.systems if item.system_id is resolved)


_OVERLAY_PATHS = {
    "system_id",
    "display_name",
    "policy_source",
    "features.trained_pre_action_policy",
    "features.post_action_diagnosis",
    "features.recovery_controller",
    "features.memory_query",
    "features.memory_intervention",
    "features.evaluation_memory_write",
}
_PROTOCOL_TOP_KEYS = {
    "schema_version",
    "protocol_id",
    "protocol_status",
    "evidence_label",
    "paper_table_status",
    "selection",
    "benchmark",
    "manual_rescue",
    "systems",
    "causal_boundary",
    "recovery_trigger",
    "parameter_provider",
    "budgets",
    "loop_rule",
    "execution_order",
    "randomness",
    "memory",
    "verification",
    "artifacts",
    "statistics",
}


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is project-standard
        raise RuntimeError(
            "PyYAML is required to load Table 2 protocol files; install the "
            "project dependencies before running the runtime"
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _flatten_overlay(payload: Mapping[str, Any]) -> set[str]:
    paths: set[str] = set()
    for key, value in payload.items():
        if key == "schema_version":
            continue
        if isinstance(value, Mapping):
            paths.update(f"{key}.{nested}" for nested in value)
        else:
            paths.add(str(key))
    return paths


def _resolve_protocol_asset(
    protocol_path: Path,
    configured_path: str | Path,
    *,
    frozen_relative_path: Path,
) -> Path:
    """Resolve a protocol dependency without escaping a frozen campaign.

    Campaign freezing rewrites the dependency closure beside ``protocol.yaml``
    under ``frozen/`` while retaining the repository-relative strings in the
    registered YAML.  A runtime loading that frozen copy must therefore prefer
    the sibling closure and must never fall back to a mutable repository file.
    The repository-relative fallback is retained only for source configuration
    files that are not themselves inside a canonical ``frozen`` directory.
    """

    sibling = protocol_path.parent / frozen_relative_path
    if sibling.exists():
        return sibling
    if protocol_path.parent.name == "frozen":
        raise FileNotFoundError(
            "frozen protocol dependency is missing from the campaign closure: "
            f"{sibling}"
        )
    configured = Path(configured_path)
    if configured.is_absolute():
        return configured
    # Paths in the source protocol are repository-root relative.  This fallback
    # is deliberately unavailable to a campaign/frozen/protocol.yaml load.
    return protocol_path.resolve().parents[3] / configured


def load_protocol_bundle(
    protocol_path: str | Path,
    *,
    campaign_id: str,
    campaign_seed: int | None = None,
) -> ProtocolBundle:
    """Load and fail-closed validate the canonical protocol plus E0-E3 overlays."""
    path = Path(protocol_path)
    payload = _load_yaml(path)
    unknown_top = set(payload) - _PROTOCOL_TOP_KEYS
    if unknown_top:
        raise ValueError(f"unregistered protocol keys: {sorted(unknown_top)}")
    if str(payload.get("schema_version")) != "1.0":
        raise ValueError("Table 2 YAML schema_version must be '1.0'")
    validate_frozen_protocol_mapping(payload)
    registered_campaign_seed = int(payload["statistics"]["seed"])
    if campaign_seed is None:
        campaign_seed = registered_campaign_seed
    elif campaign_seed != registered_campaign_seed:
        raise ValueError(
            "explicit campaign seed differs from frozen protocol statistics.seed"
        )

    budgets = payload.get("budgets")
    if not isinstance(budgets, Mapping):
        raise ValueError("protocol budgets must be a mapping")
    expected_budget_values = {
        "executor_requests_per_episode": REGISTERED_MAX_EXECUTOR_STEPS,
        "recovery_attempts_per_incident": REGISTERED_MAX_RECOVERY_ATTEMPTS_PER_INCIDENT,
        "recovery_attempts_per_episode": REGISTERED_MAX_RECOVERY_ATTEMPTS_PER_EPISODE,
        "model_calls_per_episode": REGISTERED_MAX_MODEL_CALLS_PER_EPISODE,
        "task_timeout_seconds": int(REGISTERED_EPISODE_TIMEOUT_SECONDS),
        "pre_browser_setup_timeout_seconds": int(
            REGISTERED_PRE_BROWSER_SETUP_TIMEOUT_SECONDS
        ),
        "abort_counts_as_recovery_attempt": True,
        "abort_executor_steps": 0,
    }
    for key, expected in expected_budget_values.items():
        if budgets.get(key) != expected:
            raise ValueError(
                f"registered protocol budget {key!r} must be {expected!r}, "
                f"got {budgets.get(key)!r}"
            )

    randomness = payload.get("randomness")
    if not isinstance(randomness, Mapping):
        raise ValueError("protocol randomness must be a mapping")
    if randomness.get("algorithm") != REGISTERED_RNG_ALGORITHM:
        raise ValueError(f"runtime requires {REGISTERED_RNG_ALGORITHM}")
    if tuple(randomness.get("namespace_fields", ())) != (
        REGISTERED_RNG_NAMESPACE_FIELDS
    ):
        raise ValueError("runtime RNG namespace fields differ from registration")
    if tuple(randomness.get("key_fields", ())) != REGISTERED_RNG_KEY_FIELDS:
        raise ValueError("runtime RNG key fields differ from the registered order")
    if randomness.get("system_id_in_key") is not REGISTERED_RNG_SYSTEM_ID_IN_KEY:
        raise ValueError("runtime RNG key must remain system-independent")
    if randomness.get("shared_across_systems_before_intervention") is not True:
        raise ValueError("matched systems must share stage RNG before intervention")

    causal = payload.get("causal_boundary")
    memory = payload.get("memory")
    verification = payload.get("verification")
    if not all(isinstance(value, Mapping) for value in (causal, memory, verification)):
        raise ValueError("causal, memory, and verification sections are required")
    assert isinstance(causal, Mapping) and isinstance(memory, Mapping)
    assert isinstance(verification, Mapping)
    forbidden = set(causal.get("forbidden_runtime_inputs", []))
    required_forbidden = {
        "oracle_success",
        "oracle_failure",
        "oracle_progress",
        "relevance_label",
        "reference_trajectory",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("protocol weakened the oracle-blind input boundary")
    if causal.get("sealed_verifier_returns") != "opaque_terminal_signal":
        raise ValueError("sealed verifier must return only an opaque terminal signal")
    if memory.get("runtime_mode") != "immutable_read_only":
        raise ValueError("primary memory must be immutable/read-only")
    if memory.get("source_split") != "train":
        raise ValueError("primary memory must be train-only")
    if memory.get("e3_shadow_decision") != "required_before_retrieval":
        raise ValueError("E3 no-memory shadow decision is mandatory")
    if memory.get("e2_queries") != "forbidden" or memory.get("e3_writes") != "forbidden":
        raise ValueError("protocol permits a prohibited memory operation")
    if verification.get("runtime_can_read_full_labels") is not False:
        raise ValueError("runtime cannot read full verifier labels")

    provider = payload.get("parameter_provider")
    benchmark = payload.get("benchmark")
    systems_cfg = payload.get("systems")
    if not all(isinstance(value, Mapping) for value in (provider, benchmark, systems_cfg)):
        raise ValueError("provider, benchmark, and systems sections are required")
    assert isinstance(provider, Mapping) and isinstance(benchmark, Mapping)
    assert isinstance(systems_cfg, Mapping)
    if provider.get("shared_across_systems") is not True:
        raise ValueError("action-parameter provider must be shared across E0-E3")
    if benchmark.get("invalidate_if_successful_after_reset") is not True:
        raise ValueError("already-successful reset invalidation cannot be disabled")

    recovery_cfg = payload["recovery_trigger"]
    assert isinstance(recovery_cfg, Mapping)
    recovery_truth_table = recovery_cfg["policy_truth_table"]
    assert isinstance(recovery_truth_table, Mapping)
    protocol = RuntimeProtocol(
        protocol_id=str(payload["protocol_id"]),
        campaign_id=campaign_id,
        campaign_seed=campaign_seed,
        loop_rule=LoopRule(
            state_fingerprint_algorithm=str(
                payload["loop_rule"]["state_fingerprint_algorithm"]
            ),
            state_fingerprint_fields=tuple(
                str(value) for value in payload["loop_rule"]["state_fingerprint_fields"]
            ),
            action_target_fingerprint_algorithm=str(
                payload["loop_rule"]["action_target_fingerprint_algorithm"]
            ),
            action_target_fingerprint_fields=tuple(
                str(value)
                for value in payload["loop_rule"]["action_target_fingerprint_fields"]
            ),
            perceptual_similarity_metric=str(
                payload["loop_rule"]["perceptual_similarity_metric"]
            ),
            perceptual_similarity_threshold=float(
                payload["loop_rule"]["perceptual_similarity_threshold"]
            ),
            rolling_window=int(payload["loop_rule"]["rolling_window"]),
            equivalent_repetition_count=int(
                payload["loop_rule"]["equivalent_repetition_count"]
            ),
            detect_abab_cycle=bool(payload["loop_rule"]["detect_abab_cycle"]),
        ),
        recovery_trigger=RecoveryTriggerConfig(
            failure_probability_threshold=float(
                recovery_cfg["failure_probability_threshold"]
            ),
            needs_recovery_probability_threshold=float(
                recovery_cfg["needs_recovery_probability_threshold"]
            ),
            probability_comparator=str(recovery_cfg["probability_comparator"]),
            policy_signal_order=tuple(
                str(value) for value in recovery_cfg["policy_signal_order"]
            ),
            policy_truth_table=tuple(
                (key, recovery_truth_table[key])
                for key, _ in REGISTERED_RECOVERY_POLICY_TRUTH_TABLE
            ),
            executor_status_rule=str(recovery_cfg["executor_status_rule"]),
            loop_rule=str(recovery_cfg["loop_rule"]),
        ),
        require_frozen_stage_seed_plan=True,
        provider_id=str(provider.get("mode", "unregistered-provider")),
        provider_version=str(payload["schema_version"]),
        benchmark_id=str(benchmark.get("name", "unregistered-benchmark")),
        benchmark_version=str(payload["schema_version"]),
        evaluation_mode=True,
        provider_prompt_sha256=hashlib.sha256(
            _resolve_protocol_asset(
                path,
                str(provider["prompt"]),
                frozen_relative_path=(
                    Path("prompts") / Path(str(provider["prompt"])).name
                ),
            ).read_bytes()
        ).hexdigest(),
        metadata={
            "protocol_status": str(payload.get("protocol_status", "")),
            "evidence_label": str(payload.get("evidence_label", "")),
            "paper_table_status": str(payload.get("paper_table_status", "")),
            "selection_mode": str(payload["selection"]["mode"]),
            "selection_candidate_ids": list(payload["selection"]["candidate_ids"]),
            "selection_locked_task_eligible": bool(
                payload["selection"]["locked_task_eligible"]
            ),
        },
    )

    allowed = set(systems_cfg.get("allowed_overlay_keys", []))
    if allowed != _OVERLAY_PATHS:
        raise ValueError("allowed overlay keys differ from the canonical set")
    registered_ids = tuple(SystemID(value) for value in systems_cfg.get("registered_ids", []))
    if registered_ids != tuple(SystemID):
        raise ValueError("registered system IDs must be exactly E0, E1, E2, E3")
    configured_overlay_dir = systems_cfg.get("overlay_directory")
    if configured_overlay_dir is None:
        raise ValueError("protocol systems.overlay_directory is required")
    overlay_dir = _resolve_protocol_asset(
        path,
        str(configured_overlay_dir),
        frozen_relative_path=Path("systems"),
    )

    overlays = tuple(_load_system_overlay(overlay_dir / f"{item.value.lower()}.yaml") for item in SystemID)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return ProtocolBundle(
        protocol=protocol,
        systems=overlays,
        source_path=str(path),
        source_sha256=source_hash,
    )


def _load_system_overlay(path: Path) -> SystemOverlay:
    payload = _load_yaml(path)
    if str(payload.get("schema_version")) != "1.0":
        raise ValueError(f"system overlay schema mismatch: {path}")
    actual_paths = _flatten_overlay(payload)
    if actual_paths != _OVERLAY_PATHS:
        missing = sorted(_OVERLAY_PATHS - actual_paths)
        extra = sorted(actual_paths - _OVERLAY_PATHS)
        raise ValueError(f"invalid system overlay {path}: missing={missing}, extra={extra}")
    system_id = SystemID(payload["system_id"])
    features = payload["features"]
    if not isinstance(features, Mapping):
        raise ValueError(f"system features must be a mapping: {path}")
    switch_names = (
        "trained_pre_action_policy",
        "post_action_diagnosis",
        "recovery_controller",
        "memory_query",
        "memory_intervention",
        "evaluation_memory_write",
    )
    if any(type(features[name]) is not bool for name in switch_names):
        raise TypeError(f"system mechanism switches must be booleans: {path}")
    configured_switches = tuple(features[name] for name in switch_names)
    if configured_switches != SYSTEM_SWITCH_MATRIX[system_id]:
        raise ValueError(
            f"{system_id.value} overlay breaks registered E0-E3 switches: "
            f"{configured_switches}"
        )
    expected_policy = (
        "selected_backbone_unadapted"
        if system_id is SystemID.E0
        else "selected_checkpoint"
    )
    if payload["policy_source"] != expected_policy:
        raise ValueError(f"{system_id.value} has invalid policy_source")
    return SystemOverlay(
        system_id=system_id,
        display_name=str(payload["display_name"]),
        policy_source=str(payload["policy_source"]),
        switches=switches_for(system_id),
    )
