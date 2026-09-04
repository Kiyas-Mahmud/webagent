"""Attested production bridge for the frozen Table 2 evaluation campaign.

This module contains the stable runner named by the campaign attestation.  It
does not guess a model family, BrowserGym API, evaluator implementation, or
credential layout.  Instead, the post-training handoff freezes one source-
attested integration factory.  That factory must return the exact typed
bindings below; this runner then verifies all frozen bytes and constructs the
registered E0--E3 systems itself.

The distinction is important: model/browser-specific code is necessarily
written after validation-only checkpoint selection, while the scientific
switch matrix, causal boundary, memory reader, budgets, and evidence layout
remain owned by this repository.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

from web_agent.benchmarks.recovery_fixture import RecoveryFixtureCampaignRunner
from web_agent.benchmarks.browsergym_webarena import (
    BrowserGymEpisodeAbortReceipt,
)
from web_agent.benchmarks.webarena import (
    ActionMapper,
    EnvironmentFactory,
    FrozenWebArenaActionSafetyPolicy,
    FrozenWebArenaEnvironmentStateDigester,
    FrozenWebArenaInfrastructureFaultClassifier,
    FrozenWebArenaManualRescueGuard,
    FrozenWebArenaPageSettlePolicy,
    ObservationMapper,
    ScreenshotBytesProvider,
    TerminalSignalMapper,
    WebArenaAdapter,
)
from web_agent.eval.table2.campaign import load_entrypoint
from web_agent.eval.table2.common import (
    SchemaError,
    Table2Error,
    read_json,
    safe_relative_path,
    sha256_file,
    sha256_json,
)
from web_agent.eval.table2.execution_guard import (
    InfrastructureInvalidError,
    PC01_PAGE_BROKER_SECURITY_FIELD,
    PC01_PROVIDER_BOOTSTRAP_BINDING_FIELD,
    assert_pc01_page_broker_production_authorized,
    assert_clean_git_checkout,
    attested_source_hashes,
    validate_attested_callable_source,
    validate_runner_attestation_payload,
)
from web_agent.eval.table2.locked_mount_preflight import (
    LockedMountPreflightError,
    assert_production_locked_mount_preflight,
)
from web_agent.eval.table2.live_deployment import (
    LIVE_DEPLOYMENT_BINDING_FIELD,
    ValidatedPC01LiveDeployment,
    validate_bound_pc01_live_deployment,
    validate_evaluator_requirements_resolved_snapshot_binding,
)
from web_agent.eval.table2.package_validator import (
    MODEL_EVIDENCE_ROLES,
    MODEL_PAYLOAD_HASH_FIELDS,
    MODEL_PAYLOAD_ROLES,
    PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD,
    PC01_CHECKPOINT_COMPATIBILITY_RECEIPT_RELATIVE_PATH,
    _validate_pc01_checkpoint_compatibility_readiness,
    _validate_model_evidence_bundle,
    require_pc01_provider_installation_ledger,
    validate_campaign,
)
from web_agent.eval.table2.resolved_config import (
    ResolvedConfigIdentityError,
    load_resolved_config_identity,
)
from web_agent.eval.table2.sealed_verifier import (
    SealedVerifierWriter,
    assert_no_verifier_evidence,
)
from web_agent.eval.table2.webarena_preflight_binding import (
    validate_bound_deployment_preflight,
)
from web_agent.memory.frozen_store import FrozenMemoryStore
from web_agent.runtime.action_parameters import (
    ProviderCallable,
    build_registered_hybrid_parameter_provider,
)
from web_agent.runtime.checkpoint_inference import (
    LoadedSelectedBackboneBackend,
    LoadedSelectedCheckpointBackend,
    SelectedBackbonePolicyAdapter,
    SelectedCheckpointPolicyAdapter,
    ValidationSelectedBackbone,
    ValidationSelectedCheckpoint,
)
from web_agent.runtime.contracts import (
    EpisodeSummary,
    OpaqueTerminalSignal,
    RuntimeStartState,
    SystemID,
    TaskSpecification,
    canonical_sha256,
)
from web_agent.runtime.duplicate_audit import FrozenDuplicateAuditManifest
from web_agent.runtime.episode import EpisodeRunner
from web_agent.runtime.event_log import EpisodeEventLogs
from web_agent.runtime.executor import EpisodeTimeout, Executor
from web_agent.runtime.memory_adapter import (
    CallablePostFailureEmbeddingProvider,
    FrozenStoreMemoryReader,
    MemoryAdapter,
)
from web_agent.runtime.observation import (
    ObservationBuilder,
    ProcessorParityContract,
    assert_oracle_blind_mapping,
    validate_processor_parity,
)
from web_agent.runtime.policy import SystemPolicy
from web_agent.runtime.protocol import ProtocolBundle, load_protocol_bundle
from web_agent.runtime.protocol import REGISTERED_PRE_BROWSER_SETUP_TIMEOUT_SECONDS
from web_agent.runtime.recovery.controller import (
    RecoveryActionPlanner,
    RecoveryController,
)
from web_agent.runtime.state_reset import (
    EpisodeStateResetEvidence,
    EpisodeStateResetRequest,
    EpisodeStateResetter,
    FrozenWebArenaResetStateAttester,
    PreBrowserSetupDeadline,
)


PRODUCTION_RUNNER_ENTRYPOINT = (
    "web_agent.eval.table2.production_runner:create_runner"
)
_PROCESSOR_FIELDS = (
    "processor_class",
    "processor_revision",
    "processor_config_sha256",
    "pre_action_field_mapping",
    "post_action_field_mapping",
)
_OPTIONAL_EFFICIENCY_FIELDS = frozenset(
    {
        "input_token_count",
        "output_token_count",
        "model_parameter_count",
        "trainable_parameter_count",
        "peak_gpu_memory_mb",
        "peak_system_memory_mb",
        "training_gpu_hours",
    }
)

RUNTIME_CAPABILITY_AUTHORITY_SCHEMA_VERSION = (
    "table2-pc01-runtime-capability-authority-v1"
)
RUNTIME_LIVE_CAPABILITY_IDS = (
    "deterministic_reset",
    "exclusive_input_audit",
    "oracle_blind_browser_mapping",
    "action_safety_fault_classification",
    "recovery_action_planner",
    "efficiency_measurement",
)
_RUNTIME_CAPABILITY_IDENTITY_FIELDS = frozenset(
    {
        "capability_id",
        "implementation_id",
        "implementation_version",
        "source_relative_path",
        "source_sha256",
        "deployment_state_sha256",
        "readiness_evidence_sha256",
        "runtime_return_contract",
    }
)
_RUNTIME_CONTEXT_IDENTITY_FIELDS = frozenset(
    {
        "checkpoint_systems",
        "selected_checkpoint_by_seed",
        "e0_unadapted_backbone_by_seed",
        "parameter_provider",
        "memory_by_seed",
        "environment",
        "runtime_integration",
        "validation_selection_evidence",
        PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD,
    }
)
_RUNTIME_CONTEXT_ENVIRONMENT_FIELDS = frozenset(
    {
        "benchmark",
        "benchmark_version",
        "benchmark_revision",
        "task_definition_version",
        "browser",
        "browser_version",
        "playwright_version",
        "controller_version",
        "environment_adapter_id",
        "environment_adapter_version",
        "environment_state_digester",
        "infrastructure_classifier",
        "page_settle_policy",
        "manual_rescue_guard",
        "container_digest",
        "dependency_lock_sha256",
        "dependency_lock_relative_path",
        "viewport",
    }
)


class ProductionRunnerError(Table2Error):
    """A frozen production binding is absent, inconsistent, or unsafe."""


def _require_runtime_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ProductionRunnerError(f"{field} must be a lowercase SHA-256")
    return value


def runtime_context_identity(
    attested_runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the full runner attestation onto the oracle-blind runtime plane.

    The runner attestation legitimately authenticates the sealed evaluator and
    the resolved task package.  The model integration does not need either, so
    neither their records nor the live-deployment package binding are copied
    into ``FrozenRuntimeContext``.  An explicit allow-list makes a newly added
    attestation field fail closed instead of silently crossing the boundary.
    """

    if not isinstance(attested_runtime_identity, Mapping):
        raise ProductionRunnerError("runner attestation runtime identity is malformed")
    unexpected = set(attested_runtime_identity) - (
        _RUNTIME_CONTEXT_IDENTITY_FIELDS | {"evaluator", "resolved_task_snapshot"}
    )
    if unexpected:
        raise ProductionRunnerError(
            "runner attestation has unclassified runtime/sealed identity fields: "
            f"{sorted(unexpected)}"
        )
    result = {
        key: value
        for key, value in attested_runtime_identity.items()
        if key in _RUNTIME_CONTEXT_IDENTITY_FIELDS
    }
    environment = result.get("environment")
    if isinstance(environment, Mapping):
        unexpected_environment = set(environment) - (
            _RUNTIME_CONTEXT_ENVIRONMENT_FIELDS
            | {"manifest_sha256", LIVE_DEPLOYMENT_BINDING_FIELD}
        )
        if unexpected_environment:
            raise ProductionRunnerError(
                "runner environment has unclassified runtime/sealed fields: "
                f"{sorted(unexpected_environment)}"
            )
        result["environment"] = {
            key: value
            for key, value in environment.items()
            if key in _RUNTIME_CONTEXT_ENVIRONMENT_FIELDS
        }
    elif environment is not None:
        raise ProductionRunnerError("runner runtime environment identity is malformed")
    # Produce a detached JSON value before it is handed to deployment code.
    projected = json.loads(json.dumps(result, sort_keys=True))
    try:
        assert_oracle_blind_mapping(
            projected,
            location="frozen_runtime_context.runtime_identity",
        )
    except ValueError as exc:
        raise ProductionRunnerError(
            "runtime identity projection contains sealed/oracle data"
        ) from exc
    return projected


def validate_runtime_capability_authority(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the evaluator-free authority presented to live runtime code."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "deployment_preflight_binding_sha256",
        "expected_provider_public_contract_sha256",
        "capabilities",
        "capability_set_sha256",
    }:
        raise ProductionRunnerError(
            "runtime capability authority has extra/missing fields"
        )
    if value.get("schema_version") != RUNTIME_CAPABILITY_AUTHORITY_SCHEMA_VERSION:
        raise ProductionRunnerError("runtime capability authority version changed")
    _require_runtime_sha256(
        value.get("deployment_preflight_binding_sha256"),
        field="runtime capability authority deployment-preflight identity",
    )
    _require_runtime_sha256(
        value.get("expected_provider_public_contract_sha256"),
        field="runtime capability authority provider public contract",
    )
    capabilities = value.get("capabilities")
    if not isinstance(capabilities, Mapping) or set(capabilities) != set(
        RUNTIME_LIVE_CAPABILITY_IDS
    ):
        raise ProductionRunnerError(
            "runtime capability authority must contain the six runtime capabilities"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for capability_id in RUNTIME_LIVE_CAPABILITY_IDS:
        row = capabilities[capability_id]
        if not isinstance(row, Mapping) or set(row) != _RUNTIME_CAPABILITY_IDENTITY_FIELDS:
            raise ProductionRunnerError(
                f"runtime capability authority row is malformed: {capability_id}"
            )
        item = dict(row)
        if item.get("capability_id") != capability_id:
            raise ProductionRunnerError(
                f"runtime capability authority ID differs: {capability_id}"
            )
        for field in (
            "implementation_id",
            "implementation_version",
            "source_relative_path",
            "runtime_return_contract",
        ):
            if type(item.get(field)) is not str or not str(item[field]).strip():
                raise ProductionRunnerError(
                    f"runtime capability authority lacks {capability_id}.{field}"
                )
        relative = safe_relative_path(str(item["source_relative_path"])).as_posix()
        if relative != item["source_relative_path"]:
            raise ProductionRunnerError(
                f"runtime capability authority source path is not canonical: {capability_id}"
            )
        for field in (
            "source_sha256",
            "deployment_state_sha256",
            "readiness_evidence_sha256",
        ):
            _require_runtime_sha256(
                item.get(field),
                field=f"runtime capability {capability_id}.{field}",
            )
        normalized[capability_id] = item
    ordered_rows = [normalized[item] for item in RUNTIME_LIVE_CAPABILITY_IDS]
    if sha256_json(ordered_rows) != value.get("capability_set_sha256"):
        raise ProductionRunnerError(
            "runtime capability authority capability-set hash differs"
        )
    result = {
        "schema_version": RUNTIME_CAPABILITY_AUTHORITY_SCHEMA_VERSION,
        "deployment_preflight_binding_sha256": str(
            value["deployment_preflight_binding_sha256"]
        ),
        "expected_provider_public_contract_sha256": str(
            value["expected_provider_public_contract_sha256"]
        ),
        "capabilities": normalized,
        "capability_set_sha256": sha256_json(ordered_rows),
    }
    try:
        assert_oracle_blind_mapping(
            result,
            location="frozen_runtime_context.runtime_capability_authority",
        )
    except ValueError as exc:
        raise ProductionRunnerError(
            "runtime capability authority contains sealed/oracle data"
        ) from exc
    return result


def runtime_deployment_preflight_view(deployment_preflight: Any) -> dict[str, Any]:
    """Project validated preflight evidence onto a path/content-free runtime view."""

    binding = getattr(deployment_preflight, "binding", None)
    evidence = getattr(deployment_preflight, "evidence", None)
    urls = getattr(deployment_preflight, "service_url_map", None)
    if (
        not isinstance(binding, Mapping)
        or not isinstance(evidence, Mapping)
        or not isinstance(urls, Mapping)
    ):
        raise ProductionRunnerError("validated deployment preflight is malformed")
    required = (
        "schema_version",
        "deployment_topology",
        "validator_contract",
        "expected_live_reset_task_index",
        "preflight_content_sha256",
        "service_url_map_content_sha256",
        "expected_dgx_model_runtime_identity_sha256",
        "expected_bridge_identity_sha256",
    )
    if any(key not in binding for key in required):
        raise ProductionRunnerError("deployment preflight lacks runtime view fields")
    view = {
        "schema_version": "table2-pc01-runtime-preflight-view-v1",
        "source_binding_sha256": sha256_json(dict(binding)),
        "deployment_topology": binding["deployment_topology"],
        "validator_contract": binding["validator_contract"],
        "expected_live_reset_task_index": binding[
            "expected_live_reset_task_index"
        ],
        "preflight_content_sha256": binding["preflight_content_sha256"],
        "service_url_map_content_sha256": binding[
            "service_url_map_content_sha256"
        ],
        # Split-host identities remain commitments only.  Their content, and
        # therefore any model/bridge path within it, never reaches a provider.
        "expected_dgx_runtime_identity_sha256": binding[
            "expected_dgx_model_runtime_identity_sha256"
        ],
        "expected_bridge_identity_sha256": binding[
            "expected_bridge_identity_sha256"
        ],
        "preflight_status": evidence.get("status"),
    }
    if view["preflight_status"] != "PASS":
        raise ProductionRunnerError("deployment preflight runtime view did not pass")
    if sha256_json(dict(urls)) != view["service_url_map_content_sha256"]:
        raise ProductionRunnerError("deployment preflight runtime URL commitment differs")
    try:
        assert_oracle_blind_mapping(view, location="provider_bootstrap.preflight_view")
    except ValueError as exc:
        raise ProductionRunnerError(
            "deployment preflight runtime view contains sealed/oracle data"
        ) from exc
    return json.loads(json.dumps(view, sort_keys=True))


def build_runtime_capability_authority(
    live_deployment: ValidatedPC01LiveDeployment,
    deployment_preflight: Any,
    *,
    expected_provider_public_contract_sha256: str,
) -> dict[str, Any]:
    """Derive six runtime identities from already-validated frozen evidence."""

    capabilities = live_deployment.manifest.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise ProductionRunnerError("live deployment capabilities are malformed")
    rows: dict[str, dict[str, Any]] = {}
    for capability_id in RUNTIME_LIVE_CAPABILITY_IDS:
        capability = capabilities.get(capability_id)
        if not isinstance(capability, Mapping):
            raise ProductionRunnerError(
                f"live deployment lacks runtime capability: {capability_id}"
            )
        readiness_relative = safe_relative_path(
            str(capability.get("readiness_evidence_path") or "")
        )
        readiness_path = live_deployment.package_root / readiness_relative
        readiness = read_json(readiness_path)
        if sha256_file(readiness_path) != capability.get("readiness_evidence_sha256"):
            raise ProductionRunnerError(
                f"runtime capability readiness bytes changed: {capability_id}"
            )
        rows[capability_id] = {
            "capability_id": capability_id,
            "implementation_id": capability.get("implementation_id"),
            "implementation_version": capability.get("implementation_version"),
            "source_relative_path": capability.get("source_relative_path"),
            "source_sha256": capability.get("source_sha256"),
            "deployment_state_sha256": readiness.get("deployment_state_sha256"),
            "readiness_evidence_sha256": capability.get(
                "readiness_evidence_sha256"
            ),
            "runtime_return_contract": capability.get("runtime_return_contract"),
        }
    preflight_view = runtime_deployment_preflight_view(deployment_preflight)
    ordered_rows = [rows[item] for item in RUNTIME_LIVE_CAPABILITY_IDS]
    return validate_runtime_capability_authority(
        {
            "schema_version": RUNTIME_CAPABILITY_AUTHORITY_SCHEMA_VERSION,
            "deployment_preflight_binding_sha256": sha256_json(preflight_view),
            "expected_provider_public_contract_sha256": (
                expected_provider_public_contract_sha256
            ),
            "capabilities": rows,
            "capability_set_sha256": sha256_json(ordered_rows),
        }
    )


@dataclass(frozen=True, slots=True)
class FrozenRuntimeContext:
    """Minimum model-loading capability supplied to the attested integration.

    It intentionally contains no campaign directory, task snapshot, evaluator
    configuration, memory-store path, sealed path, verifier result, relevance
    label, or reference trajectory.  Memory is opened only by this canonical
    runner for E3.  Evaluators receive a separate write-only verifier
    capability when an individual episode is created.
    """

    model_manifest_paths: Mapping[int, Path]
    model_payload_paths: Mapping[int, Mapping[str, Path]]
    model_evidence_paths: Mapping[int, Mapping[str, Path]]
    runtime_identity: Mapping[str, Any]
    runtime_capability_authority: Mapping[str, Any]

    def __post_init__(self) -> None:
        if {"evaluator", "resolved_task_snapshot"} & set(self.runtime_identity):
            raise ValueError(
                "frozen runtime identity must exclude evaluator/task-snapshot metadata"
            )
        environment = self.runtime_identity.get("environment")
        if isinstance(environment, Mapping) and (
            LIVE_DEPLOYMENT_BINDING_FIELD in environment
            or "manifest_sha256" in environment
        ):
            raise ValueError(
                "frozen runtime identity must exclude sealed live-package metadata"
            )
        try:
            assert_oracle_blind_mapping(
                self.runtime_identity,
                location="frozen_runtime_context.runtime_identity",
            )
        except ValueError as exc:
            raise ValueError(
                "frozen runtime identity must contain no sealed/oracle data"
            ) from exc
        validate_runtime_capability_authority(self.runtime_capability_authority)


BeginMeasurement = Callable[[TaskSpecification, SystemID], Any]
FinishMeasurement = Callable[[Any, EpisodeSummary], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class SeedRuntimeBinding:
    """Already-loaded, frozen runtime objects for one matched model seed."""

    model_seed: int
    episode_state_resetter: EpisodeStateResetter
    selected_checkpoint_backend: LoadedSelectedCheckpointBackend
    selected_backbone_backend: LoadedSelectedBackboneBackend
    e0_parser_sha256: str
    parameter_fallback_resolver: ProviderCallable
    parameter_fallback_backbone_sha256: str
    recovery_action_planner: RecoveryActionPlanner
    begin_measurement: BeginMeasurement
    finish_measurement: FinishMeasurement
    frozen: bool = True

    def __post_init__(self) -> None:
        if type(self.model_seed) is not int or self.model_seed < 0:
            raise ValueError("seed runtime binding requires a nonnegative integer seed")
        if not isinstance(self.episode_state_resetter, EpisodeStateResetter):
            raise TypeError("seed runtime binding requires a typed episode state resetter")
        if (
            self.episode_state_resetter.frozen is not True
            or type(self.episode_state_resetter.frozen) is not bool
        ):
            raise ValueError("episode state resetter must be frozen")
        if not isinstance(
            self.selected_checkpoint_backend, LoadedSelectedCheckpointBackend
        ):
            raise TypeError("selected checkpoint binding has the wrong contract")
        if not isinstance(
            self.selected_backbone_backend, LoadedSelectedBackboneBackend
        ):
            raise TypeError("selected backbone binding has the wrong contract")
        if not callable(self.parameter_fallback_resolver):
            raise TypeError("parameter fallback resolver must be callable")
        for name in (
            "e0_parser_sha256",
            "parameter_fallback_backbone_sha256",
        ):
            digest = str(getattr(self, name))
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"seed runtime binding {name} must be SHA-256")
        if not isinstance(self.recovery_action_planner, RecoveryActionPlanner):
            raise TypeError("recovery action planner must implement the frozen contract")
        if not self.recovery_action_planner.frozen:
            raise ValueError("recovery action planner must be frozen")
        if not callable(self.begin_measurement) or not callable(self.finish_measurement):
            raise TypeError("evaluation efficiency measurement callbacks are required")
        if not self.frozen:
            raise ValueError("seed runtime binding must be frozen")


FinalEvidenceWriter = Callable[
    [EpisodeSummary, SealedVerifierWriter, Path],
    OpaqueTerminalSignal,
]
AbortBrowserEpisode = Callable[[str, str], BrowserGymEpisodeAbortReceipt]


@dataclass(frozen=True, slots=True)
class WebArenaRuntimeBinding:
    """Oracle-blind browser callbacks created from ``TaskSpecification`` only."""

    benchmark_version: str
    environment_adapter_id: str
    environment_adapter_version: str
    dependency_module: str
    environment_factory: EnvironmentFactory
    observation_mapper: ObservationMapper
    action_mapper: ActionMapper
    screenshot_bytes_provider: ScreenshotBytesProvider
    environment_state_digester: FrozenWebArenaEnvironmentStateDigester
    infrastructure_fault_classifier: FrozenWebArenaInfrastructureFaultClassifier
    reset_state_attester: FrozenWebArenaResetStateAttester
    page_settle_policy: FrozenWebArenaPageSettlePolicy
    action_safety_policy: FrozenWebArenaActionSafetyPolicy
    manual_rescue_guard: FrozenWebArenaManualRescueGuard
    abort_episode: AbortBrowserEpisode
    frozen: bool = True

    def __post_init__(self) -> None:
        for name in (
            "benchmark_version",
            "environment_adapter_id",
            "environment_adapter_version",
            "dependency_module",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"WebArena runtime binding requires {name}")
        for name in (
            "environment_factory",
            "observation_mapper",
            "action_mapper",
            "screenshot_bytes_provider",
            "abort_episode",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"WebArena runtime binding {name} must be callable")
        if not isinstance(
            self.environment_state_digester,
            FrozenWebArenaEnvironmentStateDigester,
        ):
            raise TypeError("WebArena runtime needs the typed frozen state digester")
        if self.environment_state_digester.benchmark_version != self.benchmark_version:
            raise ValueError("WebArena state digester/runtime versions differ")
        if not isinstance(
            self.infrastructure_fault_classifier,
            FrozenWebArenaInfrastructureFaultClassifier,
        ):
            raise TypeError("WebArena runtime needs the typed frozen fault classifier")
        if (
            self.infrastructure_fault_classifier.benchmark_version
            != self.benchmark_version
        ):
            raise ValueError("WebArena classifier/runtime versions differ")
        if not isinstance(
            self.reset_state_attester,
            FrozenWebArenaResetStateAttester,
        ):
            raise TypeError("WebArena runtime needs the typed frozen reset-state attester")
        if type(self.page_settle_policy) is not FrozenWebArenaPageSettlePolicy:
            raise TypeError("WebArena runtime needs the typed frozen page-settle policy")
        if self.page_settle_policy.benchmark_version != self.benchmark_version:
            raise ValueError("WebArena page-settle/runtime versions differ")
        if type(self.action_safety_policy) is not FrozenWebArenaActionSafetyPolicy:
            raise TypeError("WebArena runtime needs the typed frozen action-safety policy")
        if self.action_safety_policy.benchmark_version != self.benchmark_version:
            raise ValueError("WebArena action-safety/runtime versions differ")
        if type(self.manual_rescue_guard) is not FrozenWebArenaManualRescueGuard:
            raise TypeError("WebArena runtime needs the typed frozen manual-rescue guard")
        if self.manual_rescue_guard.benchmark_version != self.benchmark_version:
            raise ValueError("WebArena manual-rescue/runtime versions differ")
        if self.frozen is not True or type(self.frozen) is not bool:
            raise ValueError("WebArena runtime binding must be frozen")


@dataclass(frozen=True, slots=True)
class SealedEvaluatorBinding:
    """Evaluator-only callbacks with no policy, provider, or memory capability.

    ``terminal_signal_mapper`` receives only a detached, deep-frozen observation
    snapshot; it must write full transition evidence directly to its captured
    sealed sink and return only ``OpaqueTerminalSignal``.  The typed state
    digester in the runtime binding commits the live environment immediately
    before and after that call. The final writer has the same opaque-return
    interface rule for episode-level evidence. These are reviewed-code
    dataflow constraints; they do not assert process isolation.
    """

    benchmark_version: str
    evaluator_id: str
    evaluator_version: str
    terminal_signal_mapper: TerminalSignalMapper
    finalize_episode_evidence: FinalEvidenceWriter
    frozen: bool = True
    oracle_labels_exposed_to_runtime: bool = False

    def __post_init__(self) -> None:
        for name in (
            "benchmark_version",
            "evaluator_id",
            "evaluator_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"sealed evaluator binding requires {name}")
        for name in (
            "terminal_signal_mapper",
            "finalize_episode_evidence",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"sealed evaluator binding {name} must be callable")
        if not self.frozen or self.oracle_labels_exposed_to_runtime:
            raise ValueError(
                "sealed evaluator binding must be frozen and opaque-return only"
            )


RuntimeEpisodeFactory = Callable[
    [TaskSpecification],
    WebArenaRuntimeBinding,
]
EvaluatorEpisodeFactory = Callable[
    [Mapping[str, Any], SealedVerifierWriter],
    SealedEvaluatorBinding,
]


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeBinding:
    """Complete return value required from the source-attested integration."""

    runtime_identity: Mapping[str, Any]
    seed_bindings: Mapping[int, SeedRuntimeBinding]
    create_webarena_runtime: RuntimeEpisodeFactory
    frozen: bool = True
    oracle_labels_exposed_to_runtime: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_identity, Mapping) or not self.runtime_identity:
            raise ValueError("evaluation integration requires a runtime identity")
        if not self.seed_bindings:
            raise ValueError("evaluation integration requires matched-seed bindings")
        for seed, binding in self.seed_bindings.items():
            if type(seed) is not int or type(binding) is not SeedRuntimeBinding:
                raise TypeError("seed bindings must map exact integers to typed bindings")
            if binding.model_seed != seed:
                raise ValueError("seed binding key and model_seed differ")
        if not callable(self.create_webarena_runtime):
            raise TypeError("oracle-blind WebArena runtime factory must be callable")
        if not self.frozen or self.oracle_labels_exposed_to_runtime:
            raise ValueError("evaluation integration must be frozen and oracle-blind")


@dataclass(slots=True)
class _VerifiedSeedState:
    trained_policy: SelectedCheckpointPolicyAdapter
    base_policy: SelectedBackbonePolicyAdapter
    training_processor: ProcessorParityContract
    e0_training_processor: ProcessorParityContract
    provider: Any
    memory_store: FrozenMemoryStore
    post_failure_embedding_provider: CallablePostFailureEmbeddingProvider
    binding: SeedRuntimeBinding


@dataclass(slots=True)
class _ActiveWebArenaSession:
    evaluator: SealedEvaluatorBinding
    resetter: EpisodeStateResetter
    event_logs: EpisodeEventLogs
    task_id: str
    abort_episode: AbortBrowserEpisode
    guard_index: int = 0


class ProductionTable2Runner:
    """Construct and execute the registered systems from frozen inputs only."""

    def __init__(self, *, campaign_dir: str | Path) -> None:
        # Fail before campaign construction or any external integration import.
        self.repository_root = Path(__file__).resolve().parents[4]
        self._assert_clean_source_checkout()
        self.root = Path(campaign_dir).resolve()
        report = validate_campaign(
            self.root,
            require_complete=False,
            require_aggregates=False,
        )
        if not report.passed:
            raise ProductionRunnerError(
                "production campaign preflight failed: " + "; ".join(report.errors)
            )
        self.manifest = read_json(self.root / "campaign_manifest.json")
        if self.manifest.get("campaign_mode") != "evaluation":
            raise ProductionRunnerError(
                "production runner is forbidden for engineering-smoke campaigns"
            )
        if self.manifest.get("evidence_label") != "PILOT_ONLY":
            raise ProductionRunnerError(
                "this registered runner implements the 50+15 development pilot only; "
                "a later final campaign requires its own preregistered freeze"
            )
        self.attestation = read_json(self.root / "frozen" / "runner_attestation.json")
        try:
            assert_pc01_page_broker_production_authorized(
                self.attestation.get(PC01_PAGE_BROKER_SECURITY_FIELD)
            )
        except SchemaError as exc:
            raise ProductionRunnerError(
                f"production page-broker isolation gate failed: {exc}"
            ) from exc
        if self.attestation.get("runner_entrypoint") != PRODUCTION_RUNNER_ENTRYPOINT:
            raise ProductionRunnerError(
                "campaign is not attested to the canonical production runner entrypoint"
            )
        if self.manifest.get("runner_entrypoint") != PRODUCTION_RUNNER_ENTRYPOINT:
            raise ProductionRunnerError("campaign/production runner entrypoint mismatch")
        try:
            self.provider_installation_receipt = (
                require_pc01_provider_installation_ledger(
                    self.root,
                    self.manifest,
                )
            )
        except (OSError, ValueError, SchemaError) as exc:
            raise ProductionRunnerError(
                f"production provider installation evidence failed: {exc}"
            ) from exc

        # The CLI constructs a runner before CampaignRunner performs its own
        # execution guard.  Verify the source set here as well, before importing
        # or calling the external integration factory.
        repository_root = self.repository_root
        repository_commit = _git_commit(repository_root)
        validate_runner_attestation_payload(
            self.attestation,
            repository_root=repository_root,
            repository_commit=repository_commit,
            expected_runtime_identity=self.attestation["runtime_identity"],
        )
        if self.manifest.get("repository_commit") != repository_commit:
            raise ProductionRunnerError(
                "live repository commit differs from the frozen campaign"
            )
        self._attested_source_hashes = attested_source_hashes(self.attestation)

        self.environment = read_json(self.root / "frozen" / "environment.json")
        self.protocol_mapping = _read_json_or_yaml_mapping(
            self.root / "frozen" / "protocol.yaml"
        )
        try:
            deployment_preflight = validate_bound_deployment_preflight(
                self.environment,
                artifact_root=self.root / "frozen",
            )
        except SchemaError as exc:
            raise ProductionRunnerError(
                f"production WebArena deployment preflight failed: {exc}"
            ) from exc
        preflight_manifest_fields = {
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
        for field, expected in preflight_manifest_fields.items():
            if self.manifest.get(field) != expected:
                raise ProductionRunnerError(
                    f"production campaign {field} differs from deployment preflight"
                )
        self.deployment_preflight = deployment_preflight
        self._assert_locked_mount_absent()
        self.bundle = load_protocol_bundle(
            self.root / "frozen" / "protocol.yaml",
            campaign_id=str(self.manifest["campaign_id"]),
            campaign_seed=int(self.manifest["campaign_seed"]),
        )
        (
            self.model_manifests,
            payload_paths,
            evidence_paths,
        ) = self._verify_model_payloads()
        self._revalidate_pc01_checkpoint_compatibility()
        self.memory_store_paths = {
            seed: self.root / "memory" / f"seed_{seed}"
            for seed in self.model_manifests
        }
        task_snapshot = _single_task_snapshot(self.root / "frozen")
        self.duplicate_audit = FrozenDuplicateAuditManifest.from_path(
            self.root / "frozen" / "benchmark" / "duplicate_audit_manifest.json"
        )
        normal_ids, recovery_ids = _frozen_task_ids(self.root, task_snapshot)
        self.duplicate_audit.require_exact_coverage((*normal_ids, *recovery_ids))
        for task_id in normal_ids:
            self.duplicate_audit.clusters_for(task_id, task_partition="normal")

        runtime_identity = self.attestation.get("runtime_identity")
        if not isinstance(runtime_identity, Mapping):
            raise ProductionRunnerError("runner attestation runtime identity is malformed")
        integration_spec = str(self.manifest.get("runtime_integration_entrypoint") or "")
        if integration_spec != self.attestation.get("runtime_integration_entrypoint"):
            raise ProductionRunnerError(
                "campaign/attestation runtime integration entrypoint mismatch"
            )
        # Keep this adjacent to import to close the construction-time race:
        # validation/model hashing above may be slow on the DGX filesystem.
        self._assert_clean_source_checkout()
        self._assert_locked_mount_absent()
        self.live_deployment = self._revalidate_live_deployment()
        try:
            validate_evaluator_requirements_resolved_snapshot_binding(
                self.live_deployment.evaluator_requirements,
                task_export=read_json(
                    self.root
                    / "frozen"
                    / "joint_duplicate_evidence"
                    / "resolved_task_export.json"
                ),
                resolved_task_snapshot=read_json(task_snapshot),
            )
        except (OSError, ValueError, SchemaError) as exc:
            raise ProductionRunnerError(
                "production evaluator/task authority binding failed: "
                f"{exc}"
            ) from exc
        runtime_identity_projection = runtime_context_identity(runtime_identity)
        runtime_capability_authority = build_runtime_capability_authority(
            self.live_deployment,
            self.deployment_preflight,
            expected_provider_public_contract_sha256=str(
                self.attestation[PC01_PROVIDER_BOOTSTRAP_BINDING_FIELD][
                    "expected_provider_public_contract_sha256"
                ]
            ),
        )
        context = FrozenRuntimeContext(
            model_manifest_paths={
                seed: self.root / "frozen" / "models" / f"seed_{seed}.json"
                for seed in self.model_manifests
            },
            model_payload_paths=payload_paths,
            model_evidence_paths=evidence_paths,
            runtime_identity=runtime_identity_projection,
            runtime_capability_authority=runtime_capability_authority,
        )
        integration_factory = load_entrypoint(integration_spec)
        if not callable(integration_factory):
            raise ProductionRunnerError("runtime integration entrypoint is not callable")
        integration_identity = runtime_identity.get("runtime_integration")
        if not isinstance(integration_identity, Mapping):
            raise ProductionRunnerError("runtime integration source identity is absent")
        self._validate_injected_callable(
            "runtime_integration_factory",
            integration_factory,
            expected_relative_path=str(
                integration_identity.get("source_relative_path", "")
            ),
        )
        integration = integration_factory(context)
        if type(integration) is not EvaluationRuntimeBinding:
            raise ProductionRunnerError(
                "runtime integration must return exact EvaluationRuntimeBinding"
            )
        if dict(integration.runtime_identity) != runtime_identity_projection:
            raise ProductionRunnerError(
                "integration-reported runtime identity differs from runtime-only "
                "attestation projection"
            )
        expected_seeds = set(self.model_manifests)
        if set(integration.seed_bindings) != expected_seeds:
            raise ProductionRunnerError(
                "runtime integration does not cover every matched seed exactly"
            )
        self._validate_injected_callable(
            "runtime_integration.create_webarena_runtime",
            integration.create_webarena_runtime,
        )
        for seed, binding in sorted(integration.seed_bindings.items()):
            self._validate_seed_binding_sources(seed, binding)
        evaluator_identity = self.environment.get("evaluator")
        if not isinstance(evaluator_identity, Mapping):
            raise ProductionRunnerError("frozen evaluator identity is malformed")
        if runtime_identity.get("evaluator") != dict(evaluator_identity):
            raise ProductionRunnerError(
                "runner attestation evaluator identity differs from environment"
            )
        evaluator_entrypoint = str(evaluator_identity.get("entrypoint") or "")
        if ":" not in evaluator_entrypoint:
            raise ProductionRunnerError(
                "frozen evaluator requires an independently attested entrypoint"
            )
        self._assert_locked_mount_absent()
        self._assert_clean_source_checkout()
        evaluator_factory = load_entrypoint(evaluator_entrypoint)
        if not callable(evaluator_factory):
            raise ProductionRunnerError("sealed evaluator entrypoint is not callable")
        self._validate_injected_callable(
            "sealed_evaluator_factory",
            evaluator_factory,
            expected_relative_path=str(
                evaluator_identity.get("source_relative_path", "")
            ),
        )
        self.evaluator_factory: EvaluatorEpisodeFactory = evaluator_factory
        self.integration = integration
        self.seed_states = {
            seed: self._build_seed_state(
                seed,
                model_manifest=self.model_manifests[seed],
                payloads=payload_paths[seed],
                binding=integration.seed_bindings[seed],
            )
            for seed in sorted(expected_seeds)
        }
        self._active_webarena: dict[str, _ActiveWebArenaSession] = {}
        self._efficiency: dict[str, dict[str, Any]] = {}
        self._contract_validation_receipts: dict[str, Mapping[str, Any]] = {}
        self._recovery_runner = RecoveryFixtureCampaignRunner(
            episode_executor=self._execute_recovery_episode,
            duplicate_audit=self.duplicate_audit,
            scenario_path=(
                self.root / "frozen" / "benchmark" / "recovery_scenarios.json"
            ),
            rule_path=(
                self.root / "frozen" / "benchmark" / "recovery_oracle_rules.json"
            ),
        )

    def _assert_locked_mount_absent(self) -> None:
        """Enforce the pilot's live locked-test boundary before external code."""

        try:
            assert_production_locked_mount_preflight(
                repository_root=self.repository_root,
                protocol=self.protocol_mapping,
                environment=self.environment,
            )
        except LockedMountPreflightError as exc:
            raise ProductionRunnerError(
                f"production locked-mount preflight failed: {exc}"
            ) from exc

    def _assert_clean_source_checkout(self) -> None:
        try:
            assert_clean_git_checkout(self.repository_root)
        except SchemaError as exc:
            raise ProductionRunnerError(
                f"production source checkout preflight failed: {exc}"
            ) from exc

    def _revalidate_live_deployment(self) -> ValidatedPC01LiveDeployment:
        """Reopen all measured capability bytes before external code can run."""

        try:
            validated = validate_bound_pc01_live_deployment(
                self.environment,
                artifact_root=self.root / "frozen",
                repository_root=self.repository_root,
            )
        except (OSError, ValueError, SchemaError) as exc:
            raise ProductionRunnerError(
                f"production PC-01 live-deployment preflight failed: {exc}"
            ) from exc
        if self.manifest.get(LIVE_DEPLOYMENT_BINDING_FIELD) != validated.binding:
            raise ProductionRunnerError(
                "campaign PC-01 live-deployment binding differs from frozen evidence"
            )
        required_rows = list(validated.binding["capability_source_files"])
        validator_relative = "src/web_agent/eval/table2/live_deployment.py"
        required_rows.append(
            {
                "relative_path": validator_relative,
                "sha256": sha256_file(self.repository_root / validator_relative),
            }
        )
        for row in required_rows:
            relative = str(row["relative_path"])
            if self._attested_source_hashes.get(relative) != row["sha256"]:
                raise ProductionRunnerError(
                    "runner attestation omits live capability implementation "
                    f"source: {relative}"
                )
        return validated

    def _validate_injected_callable(
        self,
        field: str,
        callback: Any,
        *,
        expected_relative_path: str | None = None,
    ) -> None:
        try:
            validate_attested_callable_source(
                callback,
                repository_root=self.repository_root,
                source_hashes=self._attested_source_hashes,
                field=field,
                expected_relative_path=expected_relative_path,
            )
        except SchemaError as exc:
            raise ProductionRunnerError(
                f"production callback source preflight failed: {exc}"
            ) from exc

    def _validate_seed_binding_sources(
        self,
        seed: int,
        binding: SeedRuntimeBinding,
    ) -> None:
        resetter = binding.episode_state_resetter
        reset_callbacks = (
            ("reset", getattr(resetter, "callback", resetter.reset)),
            (
                "digest_backend_state",
                getattr(resetter, "digest_callback", resetter.digest_backend_state),
            ),
        )
        recovery = binding.recovery_action_planner
        callbacks = (
            *reset_callbacks,
            (
                "selected_checkpoint.action_predictor",
                binding.selected_checkpoint_backend.action_predictor,
            ),
            (
                "selected_checkpoint.transition_predictor",
                binding.selected_checkpoint_backend.transition_predictor,
            ),
            (
                "selected_checkpoint.recovery_predictor",
                binding.selected_checkpoint_backend.recovery_predictor,
            ),
            (
                "selected_backbone.action_predictor",
                binding.selected_backbone_backend.action_predictor,
            ),
            ("parameter_fallback_resolver", binding.parameter_fallback_resolver),
            ("recovery_action_planner", getattr(recovery, "callback", recovery.plan)),
            (
                "selected_checkpoint.memory_embedding",
                binding.selected_checkpoint_backend.memory_embedding,
            ),
            ("begin_measurement", binding.begin_measurement),
            ("finish_measurement", binding.finish_measurement),
        )
        for name, callback in callbacks:
            self._validate_injected_callable(
                f"seed_bindings[{seed}].{name}",
                callback,
            )

    def _validate_episode_binding_sources(
        self,
        runtime: WebArenaRuntimeBinding,
        evaluator: SealedEvaluatorBinding,
    ) -> None:
        callbacks = (
            ("runtime.environment_factory", runtime.environment_factory),
            ("runtime.observation_mapper", runtime.observation_mapper),
            ("runtime.action_mapper", runtime.action_mapper),
            (
                "runtime.screenshot_bytes_provider",
                runtime.screenshot_bytes_provider,
            ),
            ("runtime.abort_episode", runtime.abort_episode),
            (
                "runtime.environment_state_digester",
                runtime.environment_state_digester.callback,
            ),
            (
                "runtime.infrastructure_fault_classifier",
                runtime.infrastructure_fault_classifier.callback,
            ),
            ("runtime.page_settle_policy", runtime.page_settle_policy.callback),
            ("evaluator.terminal_signal_mapper", evaluator.terminal_signal_mapper),
            (
                "evaluator.finalize_episode_evidence",
                evaluator.finalize_episode_evidence,
            ),
        )
        for name, callback in callbacks:
            self._validate_injected_callable(name, callback)
        integration_identity = self.integration.runtime_identity.get(
            "runtime_integration"
        )
        integration_source_path = (
            str(integration_identity.get("source_relative_path") or "")
            if isinstance(integration_identity, Mapping)
            else ""
        )
        if not integration_source_path:
            raise ProductionRunnerError(
                "runtime integration lacks an attested reset-state source path"
            )
        self._validate_injected_callable(
            "runtime.reset_state_attester",
            runtime.reset_state_attester.callback,
            expected_relative_path=integration_source_path,
        )
        self._validate_injected_callable(
            "runtime.action_safety_policy",
            runtime.action_safety_policy.callback,
            expected_relative_path=integration_source_path,
        )
        self._validate_injected_callable(
            "runtime.manual_rescue_guard",
            runtime.manual_rescue_guard.callback,
            expected_relative_path=integration_source_path,
        )

    def evaluation_runner_attestation(self) -> Mapping[str, Any]:
        """Return identities verified against actual loaded backend objects."""

        # CampaignRunner compares this with the complete runner attestation,
        # including the separately loaded sealed evaluator.  The integration
        # itself receives and reports only the runtime-plane projection.
        return dict(self.attestation["runtime_identity"])

    def run(
        self,
        *,
        task: Mapping[str, Any],
        repeat_id: int,
        model_seed: int,
        system_id: str,
        system_config: Mapping[str, Any],
        protocol: Mapping[str, Any],
        stage_seeds: Mapping[str, int],
        runtime_dir: str | Path,
        event_logs: Any,
        verifier_writer: SealedVerifierWriter,
        episode_id: str,
        **context: Any,
    ) -> Mapping[str, Any]:
        del context
        partition = str(task.get("task_partition") or "")
        setup_deadline = (
            PreBrowserSetupDeadline(
                REGISTERED_PRE_BROWSER_SETUP_TIMEOUT_SECONDS
            )
            if partition == "normal"
            else None
        )
        # Repeat immediately before each episode: a mount appearing after
        # construction must still be barred before any adapter reset.
        self._assert_locked_mount_absent()
        if dict(protocol) != self.protocol_mapping:
            raise ProductionRunnerError("runner received a mutated frozen protocol")
        resolved_system = SystemID(system_id)
        overlay = self.bundle.system(resolved_system)
        if str(system_config.get("system_id")) != resolved_system.value:
            raise ProductionRunnerError("runner system overlay identity mismatch")
        if model_seed not in self.seed_states:
            raise ProductionRunnerError("runner received an unregistered matched seed")
        expected_episode = (
            f"{self.manifest['campaign_id']}:{resolved_system.value}:"
            f"{task.get('task_id')}:repeat-{repeat_id}:seed-{model_seed}"
        )
        if episode_id != expected_episode:
            raise ProductionRunnerError("runner episode identity differs from schedule")
        destination = Path(runtime_dir).resolve()
        if self.root not in destination.parents:
            raise ProductionRunnerError("runtime output directory escaped the campaign")
        if not isinstance(verifier_writer, SealedVerifierWriter):
            raise ProductionRunnerError(
                "production runner needs the exact write-only verifier capability"
            )

        if partition == "recovery_diagnostic":
            summary = self._recovery_runner.run(
                task=dict(task),
                repeat_id=repeat_id,
                model_seed=model_seed,
                system_id=resolved_system.value,
                protocol=dict(protocol),
                stage_seeds=dict(stage_seeds),
                runtime_dir=destination,
                event_logs=event_logs,
                verifier_sink=verifier_writer,
                episode_id=episode_id,
                system_config=dict(system_config),
            )
            efficiency = self._efficiency.pop(episode_id)
            receipt = self._contract_validation_receipts.pop(episode_id)
            self._append_contract_validation_receipt(event_logs, receipt)
            return {**summary.to_dict(), **efficiency, "completed": True}
        if partition != "normal":
            raise ProductionRunnerError("frozen task has an unregistered partition")

        assert setup_deadline is not None
        # Evidence may live on a shared filesystem.  Reopen it immediately
        # before constructing the browser binding so post-construction
        # tampering cannot silently authorize a launch.
        self.live_deployment = self._revalidate_live_deployment()
        task_spec = setup_deadline.run_blocking(
            "oracle-blind task projection",
            lambda: _normal_task_specification(task, self.environment),
        )
        if not isinstance(event_logs, EpisodeEventLogs):
            raise ProductionRunnerError(
                "ordinary WebArena episode requires canonical event logs"
            )
        runtime_binding = setup_deadline.run_blocking(
            "WebArena runtime binding construction",
            lambda: self.integration.create_webarena_runtime(task_spec),
        )
        if type(runtime_binding) is not WebArenaRuntimeBinding:
            raise ProductionRunnerError(
                "runtime factory returned the wrong WebArena binding contract"
            )
        session: _ActiveWebArenaSession | None = None
        try:
            evaluator_binding = setup_deadline.run_blocking(
                "sealed evaluator binding construction",
                lambda: self.evaluator_factory(dict(task), verifier_writer),
            )
            if type(evaluator_binding) is not SealedEvaluatorBinding:
                raise ProductionRunnerError(
                    "evaluator factory returned the wrong sealed binding contract"
                )
            self._validate_episode_binding_sources(runtime_binding, evaluator_binding)
            setup_deadline.run_blocking(
                "WebArena binding verification",
                lambda: self._verify_episode_bindings(
                    runtime_binding,
                    evaluator_binding,
                ),
            )
            session = _ActiveWebArenaSession(
                evaluator=evaluator_binding,
                resetter=(
                    self.seed_states[model_seed].binding.episode_state_resetter
                ),
                event_logs=event_logs,
                task_id=task_spec.task_id,
                abort_episode=runtime_binding.abort_episode,
            )
            self._active_webarena[episode_id] = session

            def guarded_terminal_signal(
                *args: Any,
                **kwargs: Any,
            ) -> OpaqueTerminalSignal:
                assert session is not None
                return self._invoke_guarded_evaluator(
                    session,
                    callback_kind="transition_verifier",
                    callback=lambda: evaluator_binding.terminal_signal_mapper(
                        *args, **kwargs
                    ),
                )

            adapter = WebArenaAdapter(
                benchmark_version=runtime_binding.benchmark_version,
                adapter_id=runtime_binding.environment_adapter_id,
                adapter_version=runtime_binding.environment_adapter_version,
                dependency_module=runtime_binding.dependency_module,
                environment_factory=runtime_binding.environment_factory,
                observation_mapper=runtime_binding.observation_mapper,
                action_mapper=runtime_binding.action_mapper,
                screenshot_bytes_provider=runtime_binding.screenshot_bytes_provider,
                episode_runtime_dir=destination,
                terminal_signal_mapper=guarded_terminal_signal,
                environment_state_digester=(
                    runtime_binding.environment_state_digester
                ),
                infrastructure_fault_classifier=(
                    runtime_binding.infrastructure_fault_classifier
                ),
                reset_state_attester=runtime_binding.reset_state_attester,
                require_reset_state_receipt=True,
                page_settle_policy=runtime_binding.page_settle_policy,
                require_page_settle_policy=True,
                action_safety_policy=runtime_binding.action_safety_policy,
                require_action_safety_policy=True,
                manual_rescue_guard=runtime_binding.manual_rescue_guard,
                manual_rescue_evidence_sink=lambda check, receipt: event_logs.append(
                    "environment_events",
                    "manual_rescue_guard",
                    {
                        "check": check.to_dict(),
                        "check_sha256": check.record_sha256,
                        "receipt": receipt.to_dict(),
                        "receipt_sha256": receipt.record_sha256,
                    },
                ),
                require_manual_rescue_guard=True,
            )
            summary, efficiency, contract_receipt = self._execute_episode(
                task=task_spec,
                adapter=adapter,
                system_id=resolved_system,
                repeat_id=repeat_id,
                model_seed=model_seed,
                stage_seeds=stage_seeds,
                event_logs=event_logs,
                pre_browser_setup_deadline=setup_deadline,
            )
            if summary.reset_already_success:
                # This is a reset failure for the complete paired block, never
                # a free success for the system that happened to run first.
                raise InfrastructureInvalidError(
                    reason_code="ENVIRONMENT_RESET_FAILED",
                    adapter_id=str(self.environment["environment_adapter_id"]),
                    adapter_version=str(
                        self.environment["environment_adapter_version"]
                    ),
                    operation="post_reset_terminal_check",
                    adapter_evidence={
                        "adapter_event_id": f"{episode_id}:reset-already-complete",
                        "failure_class": "TASK_ALREADY_COMPLETE_AFTER_RESET",
                        "episode_id": episode_id,
                        "task_id": task_spec.task_id,
                        "terminal_reason": summary.terminal_reason.value,
                        "diagnostic_sha256": canonical_sha256(
                            {
                                "episode_id": episode_id,
                                "task_id": task_spec.task_id,
                                "terminal_reason": summary.terminal_reason.value,
                            }
                        ),
                        "retryable": True,
                    },
                )
            self._append_contract_validation_receipt(event_logs, contract_receipt)
            return {**summary.to_dict(), **efficiency, "completed": True}
        except BaseException:
            self._active_webarena.pop(episode_id, None)
            try:
                self._abort_browser_episode(
                    abort_episode=runtime_binding.abort_episode,
                    episode_id=episode_id,
                    task_id=task_spec.task_id,
                    event_logs=event_logs,
                )
            except BaseException as cleanup_error:
                raise ProductionRunnerError(
                    "WebArena episode failed and exceptional browser cleanup failed"
                ) from cleanup_error
            raise

    def finalize_episode(
        self,
        *,
        result: Mapping[str, Any] | EpisodeSummary,
        verifier_writer: SealedVerifierWriter,
        episode_id: str,
        runtime_dir: str | Path,
        **_: Any,
    ) -> OpaqueTerminalSignal:
        """Seal final ordinary-task evidence after every runtime action ends."""

        try:
            session = self._active_webarena.pop(episode_id)
        except KeyError as exc:
            raise ProductionRunnerError(
                "no active WebArena evaluator session exists for finalization"
            ) from exc
        try:
            summary = _summary_from_result(result)
            if not isinstance(verifier_writer, SealedVerifierWriter):
                raise ProductionRunnerError(
                    "final evaluator requires the exact write-only verifier capability"
                )
            signal = self._invoke_guarded_evaluator(
                session,
                callback_kind="episode_final_verifier",
                callback=lambda: session.evaluator.finalize_episode_evidence(
                    summary,
                    verifier_writer,
                    Path(runtime_dir).resolve(),
                ),
            )
            if not isinstance(signal, OpaqueTerminalSignal):
                raise ProductionRunnerError(
                    "final evaluator returned a non-opaque result"
                )
            return signal
        except BaseException:
            try:
                self._abort_browser_episode(
                    abort_episode=session.abort_episode,
                    episode_id=episode_id,
                    task_id=session.task_id,
                    event_logs=session.event_logs,
                )
            except BaseException as cleanup_error:
                raise ProductionRunnerError(
                    "sealed final evaluation failed and browser cleanup failed"
                ) from cleanup_error
            raise

    @staticmethod
    def _abort_browser_episode(
        *,
        abort_episode: AbortBrowserEpisode,
        episode_id: str,
        task_id: str,
        event_logs: EpisodeEventLogs,
    ) -> BrowserGymEpisodeAbortReceipt:
        """Invoke and record the single source-attested exceptional cleanup hook."""

        receipt = abort_episode(episode_id, task_id)
        if type(receipt) is not BrowserGymEpisodeAbortReceipt:
            raise ProductionRunnerError(
                "WebArena abort hook returned the wrong typed receipt"
            )
        if receipt.episode_id != episode_id or receipt.task_id != task_id:
            raise ProductionRunnerError(
                "WebArena abort receipt identity differs from active episode"
            )
        event_logs.append(
            "environment_events",
            "runtime_browser_abort",
            {
                "receipt": receipt.to_dict(),
                "receipt_sha256": receipt.record_sha256,
            },
        )
        return receipt

    def _revalidate_pc01_checkpoint_compatibility(self) -> None:
        """Reopen the DGX gate before any integration import or browser reset."""

        receipt_path = (
            self.root
            / "frozen"
            / PC01_CHECKPOINT_COMPATIBILITY_RECEIPT_RELATIVE_PATH
        )
        try:
            receipt, binding, _ = (
                _validate_pc01_checkpoint_compatibility_readiness(
                    receipt_path,
                    repository_root=self.repository_root,
                    expected_source_commit=str(
                        self.manifest["repository_commit"]
                    ),
                    model_manifest_path=(
                        self.root / "frozen" / "models" / "seed_42.json"
                    ),
                    selection_evidence_path=(
                        self.root
                        / "frozen"
                        / "selection_evidence"
                        / "manifest.json"
                    ),
                    path_base=self.root,
                )
            )
        except (SchemaError, FileNotFoundError, OSError) as exc:
            raise ProductionRunnerError(
                "PC-01 checkpoint compatibility revalidation failed"
            ) from exc
        runtime_identity = self.attestation.get("runtime_identity")
        if not isinstance(runtime_identity, Mapping):
            raise ProductionRunnerError(
                "runner checkpoint compatibility runtime identity is malformed"
            )
        if (
            self.manifest.get(
                "pc01_checkpoint_compatibility_receipt_sha256"
            )
            != binding["receipt_sha256"]
            or self.manifest.get(
                PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD
            )
            != binding
            or runtime_identity.get(
                PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD
            )
            != binding
        ):
            raise ProductionRunnerError(
                "campaign/runner checkpoint compatibility bindings differ"
            )
        self.checkpoint_compatibility_receipt = receipt
        self.checkpoint_compatibility_binding = binding

    def _invoke_guarded_evaluator(
        self,
        session: _ActiveWebArenaSession,
        *,
        callback_kind: str,
        callback: Callable[[], OpaqueTerminalSignal],
    ) -> OpaqueTerminalSignal:
        """Prove that an in-process trusted evaluator did not mutate backends."""

        try:
            before = dict(session.resetter.digest_backend_state())
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ProductionRunnerError(
                "cannot digest runtime backends before sealed evaluator callback"
            ) from exc
        callback_error: BaseException | None = None
        signal: OpaqueTerminalSignal | None = None
        try:
            signal = callback()
        except BaseException as exc:  # preserve interruption after the mutation check
            callback_error = exc
        try:
            after = dict(session.resetter.digest_backend_state())
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ProductionRunnerError(
                "cannot digest runtime backends after sealed evaluator callback"
            ) from exc
        session.guard_index += 1
        unchanged = before == after
        session.event_logs.append(
            "environment_events",
            "sealed_evaluator_backend_guard",
            {
                "guard_index": session.guard_index,
                "callback_kind": callback_kind,
                "before_sha256": canonical_sha256(before),
                "after_sha256": canonical_sha256(after),
                "unchanged": unchanged,
            },
        )
        if not unchanged:
            raise ProductionRunnerError(
                "sealed evaluator mutated a registered runtime backend"
            ) from callback_error
        if callback_error is not None:
            raise callback_error
        if signal is None:
            raise ProductionRunnerError(
                "sealed evaluator callback returned no opaque signal"
            )
        return signal

    def _execute_recovery_episode(
        self,
        request: Mapping[str, Any],
    ) -> EpisodeSummary:
        scenario = request["scenario"]
        task = TaskSpecification(
            task_id=str(scenario.scenario_id),
            goal=f"resolve the observable failure {scenario.failure_kind}",
            benchmark_id="recovery_fixture",
            benchmark_version="v1",
            start_state_id="failure-present",
            metadata={"task_partition": "recovery_diagnostic"},
        )
        summary, efficiency, contract_receipt = self._execute_episode(
            task=task,
            adapter=request["adapter"],
            system_id=SystemID(request["system_id"]),
            repeat_id=int(request["repeat_id"]),
            model_seed=int(request["model_seed"]),
            stage_seeds=request["stage_seeds"],
            event_logs=request["event_logs"],
        )
        self._efficiency[summary.episode_id] = efficiency
        self._contract_validation_receipts[summary.episode_id] = contract_receipt
        return summary

    def _execute_episode(
        self,
        *,
        task: TaskSpecification,
        adapter: Any,
        system_id: SystemID,
        repeat_id: int,
        model_seed: int,
        stage_seeds: Mapping[str, int],
        event_logs: Any,
        pre_browser_setup_deadline: PreBrowserSetupDeadline | None = None,
    ) -> tuple[EpisodeSummary, dict[str, Any], Mapping[str, Any]]:
        state = self.seed_states[model_seed]
        if pre_browser_setup_deadline is None:
            measurement = state.binding.begin_measurement(task, system_id)
        else:
            measurement = pre_browser_setup_deadline.run_blocking(
                "efficiency measurement setup",
                lambda: state.binding.begin_measurement(task, system_id),
            )
        started = perf_counter()
        reset_kwargs = {
            "state": state,
            "system_id": system_id,
            "repeat_id": repeat_id,
            "model_seed": model_seed,
            "stage_seeds": stage_seeds,
            "event_logs": event_logs,
        }
        if pre_browser_setup_deadline is None:
            self._reset_episode_backends(**reset_kwargs)
        else:
            pre_browser_setup_deadline.run_blocking(
                "shared model-backend reset",
                lambda: self._reset_episode_backends(**reset_kwargs),
            )
        switches = self.bundle.system(system_id).switches
        policy = state.base_policy if system_id is SystemID.E0 else state.trained_policy
        training_processor = (
            state.e0_training_processor
            if system_id is SystemID.E0
            else state.training_processor
        )
        controller = None
        if switches.recovery_controller:
            controller = RecoveryController(
                self.bundle.protocol.budgets,
                action_planner=state.binding.recovery_action_planner,
            )
        memory = None
        if system_id is SystemID.E3:
            reader = FrozenStoreMemoryReader(
                state.memory_store,
                embedding_provider=state.post_failure_embedding_provider,
                expected_model_seed=model_seed,
            )
            memory = MemoryAdapter(
                switches,
                reader=reader,
                admission_threshold=state.memory_store.admission_threshold,
                evaluation_mode=True,
                expected_store_manifest_sha256=state.memory_store.manifest_sha256,
            )
        executor = Executor(adapter, budgets=self.bundle.protocol.budgets)
        runner = EpisodeRunner(
            protocol=self.bundle.protocol,
            system_policy=SystemPolicy(policy, switches),
            provider=state.provider,
            executor=executor,
            recovery_controller=controller,
            memory_adapter=memory,
            observation_builder=ObservationBuilder(
                processor_contract=policy.runtime_processor_contract
            ),
            training_processor_contract=training_processor,
            duplicate_audit_registry=(
                self.duplicate_audit if system_id is SystemID.E3 else None
            ),
            event_logs=event_logs,
            before_environment_reset=(
                None
                if pre_browser_setup_deadline is None
                else lambda: pre_browser_setup_deadline.complete(
                    episode_id=event_logs.episode_id,
                    system_id=system_id,
                )
            ),
            defer_contract_validation_receipt=True,
        )
        summary = runner.run(
            task,
            repeat_id=repeat_id,
            model_seed=model_seed,
            stage_seeds=stage_seeds,
        )
        wall_clock = perf_counter() - started
        supplied = _finish_measurement_under_deadline(
            executor,
            state.binding.finish_measurement,
            measurement,
            summary,
        )
        efficiency = _validated_efficiency(
            supplied,
            task_wall_clock_seconds=wall_clock,
            memory_index_size=(
                _payload_size(self.memory_store_paths[model_seed])[0]
                if system_id is SystemID.E3
                else 0
            ),
            expected_model_call_count=summary.model_call_count,
        )
        contract_receipt = runner.contract_validation_receipt
        if contract_receipt is None:
            raise ProductionRunnerError(
                "EpisodeRunner returned without an in-memory contract validation receipt"
            )
        return summary, efficiency, contract_receipt

    @staticmethod
    def _append_contract_validation_receipt(
        event_logs: Any,
        receipt: Mapping[str, Any],
    ) -> None:
        if not isinstance(event_logs, EpisodeEventLogs):
            raise ProductionRunnerError(
                "production contract validation requires canonical episode event logs"
            )
        event_logs.append(
            "environment_events",
            "episode_contract_validation",
            dict(receipt),
        )

    def _reset_episode_backends(
        self,
        *,
        state: _VerifiedSeedState,
        system_id: SystemID,
        repeat_id: int,
        model_seed: int,
        stage_seeds: Mapping[str, int],
        event_logs: Any,
    ) -> EpisodeStateResetEvidence:
        """Reset shared model state once, immediately before runtime construction."""

        if not isinstance(event_logs, EpisodeEventLogs):
            raise ProductionRunnerError(
                "production backend reset requires canonical episode event logs"
            )
        episode_id = event_logs.episode_id
        identity_prefix = f"{self.manifest['campaign_id']}:{system_id.value}:"
        identity_suffix = f":repeat-{repeat_id}:seed-{model_seed}"
        if not (
            episode_id.startswith(identity_prefix)
            and episode_id.endswith(identity_suffix)
        ):
            raise ProductionRunnerError(
                "backend reset event log belongs to another episode"
            )
        environment_log = event_logs.logs.get("environment_events")
        if environment_log is None or environment_log.records != 0:
            raise ProductionRunnerError(
                "backend state reset must be the first environment event"
            )
        reset_seed = stage_seeds.get("reset")
        if isinstance(reset_seed, bool) or not isinstance(reset_seed, int):
            raise ProductionRunnerError(
                "backend reset requires the registered exact reset-stage seed"
            )
        request = EpisodeStateResetRequest(
            episode_id=episode_id,
            system_id=system_id,
            reset_stage_seed=reset_seed,
        )
        resetter = state.binding.episode_state_resetter
        evidence = resetter.reset(request)
        if type(evidence) is not EpisodeStateResetEvidence:
            raise ProductionRunnerError(
                "backend resetter returned the wrong evidence contract"
            )
        if (
            evidence.episode_id != request.episode_id
            or evidence.system_id is not request.system_id
            or evidence.reset_stage_seed != request.reset_stage_seed
            or evidence.resetter_id != resetter.resetter_id
            or evidence.resetter_version != resetter.resetter_version
            or evidence.resetter_source_sha256 != resetter.source_sha256
        ):
            raise ProductionRunnerError(
                "backend reset evidence is not bound to this episode request"
            )
        event_logs.append(
            "environment_events",
            "episode_state_reset",
            {
                "request": request.to_dict(),
                "request_sha256": request.record_sha256,
                "evidence": evidence.to_dict(),
                "evidence_sha256": evidence.record_sha256,
            },
        )
        return evidence

    def _verify_model_payloads(
        self,
    ) -> tuple[
        dict[int, dict[str, Any]],
        dict[int, dict[str, Path]],
        dict[int, dict[str, Path]],
    ]:
        expected = {int(seed) for seed in self.manifest.get("matched_seeds", [])}
        descriptors_by_seed = self.manifest.get("model_payloads_by_seed")
        if not isinstance(descriptors_by_seed, Mapping):
            raise ProductionRunnerError("campaign lacks frozen model payload descriptors")
        manifests: dict[int, dict[str, Any]] = {}
        paths: dict[int, dict[str, Path]] = {}
        evidence_paths: dict[int, dict[str, Path]] = {}
        for seed in sorted(expected):
            path = self.root / "frozen" / "models" / f"seed_{seed}.json"
            value = read_json(path)
            if value.get("model_seed") != seed:
                raise ProductionRunnerError("model manifest seed mismatch")
            rows = value.get("artifact_payloads")
            if not isinstance(rows, Mapping) or set(rows) != set(MODEL_PAYLOAD_ROLES):
                raise ProductionRunnerError("model manifest payload role coverage differs")
            if descriptors_by_seed.get(str(seed)) != rows:
                raise ProductionRunnerError("campaign/model payload descriptors differ")
            seed_paths: dict[str, Path] = {}
            for role in MODEL_PAYLOAD_ROLES:
                row = rows[role]
                if not isinstance(row, Mapping):
                    raise ProductionRunnerError(f"malformed model payload: {role}")
                relative = safe_relative_path(str(row.get("path", "")))
                payload = (self.root / relative).resolve()
                if self.root not in payload.parents or payload.is_symlink():
                    raise ProductionRunnerError(f"unsafe model payload path: {role}")
                actual_kind = "file" if payload.is_file() else "directory"
                if actual_kind != row.get("kind"):
                    raise ProductionRunnerError(f"model payload kind differs: {role}")
                size, count = _payload_size(payload)
                digest = _payload_sha256(payload)
                expected_values = {
                    "size_bytes": size,
                    "file_count": count,
                    "sha256": digest,
                }
                for key, actual in expected_values.items():
                    if row.get(key) != actual:
                        raise ProductionRunnerError(
                            f"model payload {role} {key} differs from frozen bytes"
                        )
                if value.get(MODEL_PAYLOAD_HASH_FIELDS[role]) != digest:
                    raise ProductionRunnerError(
                        f"model manifest hash differs from {role} bytes"
                    )
                seed_paths[role] = payload
            try:
                executable = {
                    role: (seed_paths[role], dict(rows[role]))
                    for role in MODEL_PAYLOAD_ROLES
                }
                evidence = _validate_model_evidence_bundle(
                    path,
                    value,
                    path_base=self.root,
                    executable_payloads=executable,
                )
            except (SchemaError, FileNotFoundError) as exc:
                raise ProductionRunnerError(
                    "frozen model evidence bundle failed authentication"
                ) from exc
            evidence_by_seed = self.manifest.get("model_evidence_by_seed")
            if not isinstance(evidence_by_seed, Mapping) or evidence_by_seed.get(
                str(seed)
            ) != value.get("model_evidence_bundle"):
                raise ProductionRunnerError(
                    "campaign/model evidence bundle descriptors differ"
                )
            if set(evidence) != set(MODEL_EVIDENCE_ROLES):
                raise ProductionRunnerError(
                    "model evidence bundle role coverage differs"
                )
            evidence_paths[seed] = {
                role: source for role, (source, _) in evidence.items()
            }
            try:
                resolved_config_identity = load_resolved_config_identity(
                    seed_paths["resolved_config"]
                )
            except ResolvedConfigIdentityError as exc:
                raise ProductionRunnerError(
                    "frozen selected resolved-config payload is invalid"
                ) from exc
            if value.get("resolved_config_record_sha256") != (
                resolved_config_identity.record_sha256
            ):
                raise ProductionRunnerError(
                    "model canonical resolved-config identity differs from payload"
                )
            manifests[seed] = value
            paths[seed] = seed_paths
        if set(map(int, descriptors_by_seed)) != expected:
            raise ProductionRunnerError("campaign payload seeds differ from schedule")
        return manifests, paths, evidence_paths

    def _build_seed_state(
        self,
        seed: int,
        *,
        model_manifest: Mapping[str, Any],
        payloads: Mapping[str, Path],
        binding: SeedRuntimeBinding,
    ) -> _VerifiedSeedState:
        trained_processor = _load_processor_contract(payloads["processor_contract"])
        e0_processor = _load_processor_contract(payloads["e0_processor_contract"])
        selected = ValidationSelectedCheckpoint.from_mapping(
            model_manifest,
            checkpoint_path=payloads["selected_checkpoint"],
        )
        base = ValidationSelectedBackbone(
            manifest_id=str(model_manifest.get("manifest_id") or f"seed-{seed}-e0"),
            backbone_id=str(model_manifest["e0_backbone_id"]),
            backbone_revision=str(model_manifest["e0_backbone_revision"]),
            backbone_path=payloads["e0_backbone"],
            backbone_sha256=str(model_manifest["e0_backbone_sha256"]),
            resolved_config_sha256=str(model_manifest["e0_resolved_config_sha256"]),
            processor_contract_sha256=str(
                model_manifest["e0_processor_contract_sha256"]
            ),
            base_prompt_sha256=str(model_manifest["e0_base_prompt_sha256"]),
            parser_id=str(model_manifest["e0_parser_id"]),
            parser_version=str(model_manifest["e0_parser_version"]),
            validation_rows_read=int(model_manifest["validation_rows_read"]),
            selection_scope=str(model_manifest["selection_scope"]),
            test_rows_read=int(model_manifest["test_rows_read"]),
            locked_test_rows_read=int(model_manifest["locked_test_rows_read"]),
        )
        if selected.model_seed != seed or binding.model_seed != seed:
            raise ProductionRunnerError("selected checkpoint/binding seed mismatch")
        runtime_integration = self.integration.runtime_identity.get(
            "runtime_integration"
        )
        resetter_source_sha256 = (
            runtime_integration.get("source_sha256")
            if isinstance(runtime_integration, Mapping)
            else None
        )
        if binding.episode_state_resetter.source_sha256 != resetter_source_sha256:
            raise ProductionRunnerError(
                "episode state resetter source differs from the frozen runtime integration"
            )
        if binding.e0_parser_sha256 != model_manifest["e0_parser_sha256"]:
            raise ProductionRunnerError(
                "loaded E0 parser identity differs from the frozen payload"
            )
        if (
            binding.parameter_fallback_backbone_sha256
            != base.backbone_sha256
        ):
            raise ProductionRunnerError(
                "common parameter fallback is not bound to the selected E0 backbone"
            )
        trained_policy = SelectedCheckpointPolicyAdapter(
            selection=selected,
            backend_factory=lambda _selection: binding.selected_checkpoint_backend,
            runtime_processor_contract=trained_processor,
        )
        base_policy = SelectedBackbonePolicyAdapter(
            selection=base,
            backend_factory=lambda _selection: binding.selected_backbone_backend,
            runtime_processor_contract=e0_processor,
        )
        # Force byte checks and loaded-backend identity checks before the runner
        # is allowed to attest itself to CampaignRunner.
        trained_policy._load()
        base_policy._load()
        validate_processor_parity(
            binding.selected_checkpoint_backend.processor_contract,
            trained_processor,
        )
        validate_processor_parity(
            binding.selected_backbone_backend.processor_contract,
            e0_processor,
        )
        embedding_provider = CallablePostFailureEmbeddingProvider(
            embedder=binding.selected_checkpoint_backend.memory_embedding,
            provider_id="selected-checkpoint-memory-embedding",
            provider_version="v1",
            checkpoint_sha256=selected.selected_checkpoint_sha256,
            processor_contract_sha256=trained_processor.record_sha256,
        )
        prompt = (self.root / "frozen" / "prompts" / "parameter_provider_v1.txt")
        provider_cfg = self.protocol_mapping.get("parameter_provider")
        if not isinstance(provider_cfg, Mapping):
            raise ProductionRunnerError("frozen parameter-provider config is malformed")
        decoding = provider_cfg.get("decoding_parameters", {})
        if not isinstance(decoding, Mapping):
            raise ProductionRunnerError("parameter-provider decoding config is malformed")
        provider = build_registered_hybrid_parameter_provider(
            fallback_resolver=binding.parameter_fallback_resolver,
            fallback_policy_id=str(model_manifest["e0_parser_id"]),
            fallback_policy_version=str(model_manifest["e0_parser_version"]),
            prompt_bytes=prompt.read_bytes(),
            decoding_parameters=dict(decoding),
        )
        memory_store = FrozenMemoryStore.load(self.memory_store_paths[seed])
        if memory_store.model_seed != seed:
            raise ProductionRunnerError("frozen memory store seed mismatch")
        if memory_store.manifest.get("checkpoint_sha256") != selected.selected_checkpoint_sha256:
            raise ProductionRunnerError("frozen memory/checkpoint identity mismatch")
        if memory_store.manifest.get("resolved_config_sha256") != selected.resolved_config_sha256:
            raise ProductionRunnerError("frozen memory/config identity mismatch")
        if memory_store.manifest.get("resolved_config_record_sha256") != (
            model_manifest.get("resolved_config_record_sha256")
        ):
            raise ProductionRunnerError(
                "frozen memory/canonical config-record identity mismatch"
            )
        memory_identity = self.integration.runtime_identity.get("memory_by_seed")
        if not isinstance(memory_identity, Mapping):
            raise ProductionRunnerError("runtime attestation lacks memory identities")
        frozen_memory_identity = memory_identity.get(str(seed))
        if (
            not isinstance(frozen_memory_identity, Mapping)
            or frozen_memory_identity.get("manifest_sha256")
            != memory_store.manifest_sha256
        ):
            raise ProductionRunnerError(
                "loaded frozen memory store differs from runner attestation"
            )
        return _VerifiedSeedState(
            trained_policy=trained_policy,
            base_policy=base_policy,
            training_processor=trained_processor,
            e0_training_processor=e0_processor,
            provider=provider,
            memory_store=memory_store,
            post_failure_embedding_provider=embedding_provider,
            binding=binding,
        )

    def _verify_episode_bindings(
        self,
        runtime: WebArenaRuntimeBinding,
        evaluator_binding: SealedEvaluatorBinding,
    ) -> None:
        evaluator = self.environment.get("evaluator")
        if not isinstance(evaluator, Mapping):
            raise ProductionRunnerError("frozen environment evaluator is malformed")
        state_digester = self.environment.get("environment_state_digester")
        classifier = self.environment.get("infrastructure_classifier")
        page_settle = self.environment.get("page_settle_policy")
        manual_rescue = self.environment.get("manual_rescue_guard")
        if not isinstance(state_digester, Mapping) or not isinstance(
            classifier, Mapping
        ) or not isinstance(page_settle, Mapping) or not isinstance(
            manual_rescue, Mapping
        ):
            raise ProductionRunnerError(
                "frozen environment lacks digester/classifier/page-settle/manual-rescue identities"
            )
        runtime_exact = {
            "benchmark_version": self.environment.get("benchmark_version"),
            "environment_adapter_id": self.environment.get("environment_adapter_id"),
            "environment_adapter_version": self.environment.get(
                "environment_adapter_version"
            ),
        }
        evaluator_exact = {
            "benchmark_version": self.environment.get("benchmark_version"),
            "evaluator_id": evaluator.get("evaluator_id"),
            "evaluator_version": evaluator.get("evaluator_version"),
        }
        for name, expected in runtime_exact.items():
            if getattr(runtime, name) != expected:
                raise ProductionRunnerError(
                    f"live WebArena runtime {name} differs from frozen environment"
                )
        digester_exact = {
            "digester_id": state_digester.get("digester_id"),
            "digester_version": state_digester.get("digester_version"),
        }
        for name, expected in digester_exact.items():
            if getattr(runtime.environment_state_digester, name) != expected:
                raise ProductionRunnerError(
                    f"live WebArena state digester {name} differs from frozen environment"
                )
        classifier_exact = {
            "classifier_id": classifier.get("classifier_id"),
            "classifier_version": classifier.get("classifier_version"),
            "rules_sha256": classifier.get("rules_sha256"),
        }
        for name, expected in classifier_exact.items():
            if getattr(runtime.infrastructure_fault_classifier, name) != expected:
                raise ProductionRunnerError(
                    f"live WebArena classifier {name} differs from frozen environment"
                )
        page_settle_exact = {
            "policy_id": page_settle.get("policy_id"),
            "network_idle_required": page_settle.get("network_idle_required"),
            "settle_timeout_seconds": page_settle.get("settle_timeout_seconds"),
        }
        for name, expected in page_settle_exact.items():
            if getattr(runtime.page_settle_policy, name) != expected:
                raise ProductionRunnerError(
                    f"live WebArena page-settle {name} differs from frozen environment"
                )
        action_safety_exact = {
            "safety_policy_id": self.environment.get("safety_policy_id"),
            "safety_policy_version": self.environment.get("safety_policy_version"),
            "destructive_action_policy": self.environment.get(
                "destructive_action_policy"
            ),
        }
        for name, expected in action_safety_exact.items():
            if getattr(runtime.action_safety_policy, name) != expected:
                raise ProductionRunnerError(
                    f"live WebArena action-safety {name} differs from frozen environment"
                )
        manual_rescue_exact = {
            "guard_id": manual_rescue.get("guard_id"),
            "guard_version": manual_rescue.get("guard_version"),
            "evidence_mode": manual_rescue.get("evidence_mode"),
        }
        for name, expected in manual_rescue_exact.items():
            if getattr(runtime.manual_rescue_guard, name) != expected:
                raise ProductionRunnerError(
                    f"live WebArena manual-rescue {name} differs from frozen environment"
                )
        for name, expected in evaluator_exact.items():
            if getattr(evaluator_binding, name) != expected:
                raise ProductionRunnerError(
                    f"sealed evaluator {name} differs from frozen environment"
                )
        runtime_integration = self.integration.runtime_identity.get(
            "runtime_integration"
        )
        integration_source = (
            runtime_integration.get("source_sha256")
            if isinstance(runtime_integration, Mapping)
            else None
        )
        if runtime.reset_state_attester.source_sha256 != integration_source:
            raise ProductionRunnerError(
                "WebArena reset-state attester source differs from the frozen "
                "runtime integration"
            )
        if runtime.action_safety_policy.source_sha256 != integration_source:
            raise ProductionRunnerError(
                "WebArena action-safety source differs from the frozen runtime "
                "integration"
            )
        if runtime.manual_rescue_guard.source_sha256 != integration_source:
            raise ProductionRunnerError(
                "WebArena manual-rescue source differs from the frozen runtime "
                "integration"
            )


def create_runner(*, campaign_dir: str | Path) -> ProductionTable2Runner:
    """Canonical factory used by ``run_table2_evaluation.py --runner-factory``."""

    return ProductionTable2Runner(campaign_dir=campaign_dir)


def _load_processor_contract(path: Path) -> ProcessorParityContract:
    value = read_json(path)
    try:
        contract = ProcessorParityContract(
            **{name: value[name] for name in _PROCESSOR_FIELDS}
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProductionRunnerError(
            f"cannot reconstruct frozen processor contract: {path}"
        ) from exc
    # The model manifest and payload descriptor use a byte-level digest, while
    # the runtime adapter uses record_sha256.  Requiring equality makes their
    # identity one and the same and catches harmless-looking JSON reformatting.
    if sha256_file(path) != contract.record_sha256:
        raise ProductionRunnerError(
            "processor payload must be the exact canonical ProcessorParityContract "
            f"record bytes: {path}"
        )
    return contract


def _normal_task_specification(
    row: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> TaskSpecification:
    """Project a resolved task to a strictly oracle-blind runtime view."""

    start = row.get("start_state")
    if not isinstance(start, Mapping):
        raise ProductionRunnerError("resolved WebArena task lacks start_state")
    try:
        runtime_start_state = RuntimeStartState.from_webarena_mapping(start)
    except (TypeError, ValueError) as exc:
        raise ProductionRunnerError(
            "resolved WebArena task has an invalid oracle-blind runtime start_state"
        ) from exc
    safe_metadata = {
        "task_partition": "normal",
        "upstream_index": row.get("upstream_index"),
        "benchmark_task_id": row.get("benchmark_task_id"),
        "source_content_sha256": row.get("source_content_sha256"),
    }
    assert_oracle_blind_mapping(safe_metadata, location="frozen_task.runtime_metadata")
    return TaskSpecification(
        task_id=str(row.get("task_id") or ""),
        goal=str(row.get("instruction") or ""),
        benchmark_id=str(environment.get("benchmark") or ""),
        benchmark_version=str(environment.get("benchmark_version") or ""),
        start_state_id=runtime_start_state.start_state_sha256,
        site=runtime_start_state.sites[0],
        start_url=runtime_start_state.start_url,
        development_partition=True,
        destructive_actions_allowed=False,
        metadata=safe_metadata,
        runtime_start_state=runtime_start_state,
    )


def _validated_efficiency(
    value: Mapping[str, Any],
    *,
    task_wall_clock_seconds: float,
    memory_index_size: int,
    expected_model_call_count: int | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionRunnerError("efficiency probe must return a mapping")
    assert_no_verifier_evidence(value, context="efficiency probe")
    unknown = set(value) - (_OPTIONAL_EFFICIENCY_FIELDS | {"model_call_count"})
    if unknown:
        raise ProductionRunnerError(
            f"efficiency probe returned unregistered fields: {sorted(unknown)}"
        )
    calls = value.get("model_call_count")
    if type(calls) is not int or calls < 0:
        raise ProductionRunnerError(
            "efficiency probe requires an exact nonnegative model_call_count"
        )
    if expected_model_call_count is not None:
        if type(expected_model_call_count) is not int or expected_model_call_count < 0:
            raise ProductionRunnerError("runner-owned model-call count is invalid")
        if calls != expected_model_call_count:
            raise ProductionRunnerError(
                "backend model-call receipt differs from runner-owned dispatch ledger"
            )
    missing = _OPTIONAL_EFFICIENCY_FIELDS - set(value)
    if missing:
        raise ProductionRunnerError(
            "production efficiency probe is incomplete: "
            + ", ".join(sorted(missing))
        )
    if not math.isfinite(task_wall_clock_seconds) or task_wall_clock_seconds < 0:
        raise ProductionRunnerError("measured task wall-clock is invalid")
    output = {
        **dict(value),
        "model_call_count": calls,
        "task_wall_clock_seconds": float(task_wall_clock_seconds),
        "memory_index_size": memory_index_size,
    }
    for name in (
        "input_token_count",
        "output_token_count",
        "model_parameter_count",
        "trainable_parameter_count",
    ):
        item = output.get(name)
        if item is not None and (type(item) is not int or item < 0):
            raise ProductionRunnerError(f"efficiency field {name} is invalid")
    for name in (
        "peak_gpu_memory_mb",
        "peak_system_memory_mb",
        "training_gpu_hours",
    ):
        item = output.get(name)
        if item is None:
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ProductionRunnerError(f"efficiency field {name} is invalid")
        numeric = float(item)
        if not math.isfinite(numeric) or numeric < 0:
            raise ProductionRunnerError(f"efficiency field {name} is invalid")
        output[name] = numeric
    return output


def _finish_measurement_under_deadline(
    executor: Executor,
    callback: FinishMeasurement,
    measurement: Any,
    summary: EpisodeSummary,
) -> Mapping[str, Any]:
    """Finalize efficiency evidence without a post-episode blocking gap.

    The callback runs inside the unspent portion of the same 600-second
    episode clock.  A callback that blocks is therefore interrupted and fails
    the episode package closed.  If the runtime itself already reached the
    deadline, no callback can legally run; registered efficiency fields are
    emitted as unavailable while the runner-owned model-call count remains
    independently reproducible from the dispatch ledger.
    """

    if not isinstance(executor, Executor):
        raise TypeError("efficiency finalization requires the canonical executor")
    if not callable(callback):
        raise TypeError("efficiency finalization callback must be callable")
    if not isinstance(summary, EpisodeSummary):
        raise TypeError("efficiency finalization requires an episode summary")
    if executor.deadline_expired:
        return {
            "model_call_count": summary.model_call_count,
            **{field: None for field in _OPTIONAL_EFFICIENCY_FIELDS},
        }
    try:
        return executor.run_blocking(
            "efficiency measurement finalization",
            lambda: callback(measurement, summary),
        )
    except EpisodeTimeout as exc:
        # This is a post-decision evidence-probe defect, not an agent-caused
        # task timeout and not a registered infrastructure-rerun reason.
        raise ProductionRunnerError(
            "efficiency measurement finalization exhausted the registered "
            "episode deadline"
        ) from exc


def _summary_from_result(value: Mapping[str, Any] | EpisodeSummary) -> EpisodeSummary:
    if isinstance(value, EpisodeSummary):
        return value
    if not isinstance(value, Mapping):
        raise ProductionRunnerError("episode finalizer received an invalid summary")
    # ProductionTable2Runner.run returns the untouched runtime record fields
    # plus efficiency fields.  Reconstruct only the EpisodeSummary fields.
    record = {
        key: item
        for key, item in value.items()
        if key in {
            "schema_version",
            "record_type",
            "episode_id",
            "protocol_id",
            "system_id",
            "task_id",
            "repeat_id",
            "model_seed",
            "valid_for_primary",
            "terminal_reason",
            "executor_steps",
            "normal_actions",
            "recovery_actions",
            "recovery_attempts",
            "failure_incidents",
            "memory_queries",
            "memory_interventions",
            "elapsed_seconds",
            "reset_already_success",
            "environment_failure",
            "verifier_event_id",
            "verifier_token_sha256",
            "event_log_sha256",
        }
    }
    try:
        result = EpisodeSummary.from_dict(record)
    except (TypeError, ValueError) as exc:
        raise ProductionRunnerError("cannot reconstruct runtime episode summary") from exc
    assert isinstance(result, EpisodeSummary)
    return result


def _single_task_snapshot(frozen: Path) -> Path:
    candidates = sorted(frozen.glob("task_manifest.*"))
    if len(candidates) != 1:
        raise ProductionRunnerError(
            "frozen campaign must contain exactly one resolved task snapshot"
        )
    return candidates[0]


def _frozen_task_ids(root: Path, task_snapshot: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    payload = json.loads(task_snapshot.read_text(encoding="utf-8"))
    normal_rows = payload if isinstance(payload, list) else payload.get("tasks", [])
    if not isinstance(normal_rows, list):
        raise ProductionRunnerError("resolved task snapshot has no task rows")
    normal = tuple(str(row["task_id"]) for row in normal_rows)
    recovery_payload = read_json(
        root / "frozen" / "benchmark" / "recovery_scenarios.json"
    )
    recovery_rows = recovery_payload.get("scenarios")
    if not isinstance(recovery_rows, list):
        raise ProductionRunnerError("frozen recovery scenario manifest is malformed")
    recovery = tuple(str(row["scenario_id"]) for row in recovery_rows)
    if len(normal) != 50 or len(recovery) != 15:
        raise ProductionRunnerError("production pilot requires exact 50+15 coverage")
    return normal, recovery


def _read_json_or_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependency
        raise ProductionRunnerError("PyYAML is required for frozen protocol loading") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProductionRunnerError(f"expected mapping: {path}")
    return value


def _payload_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        if path.is_symlink():
            raise ProductionRunnerError(f"payload is a symlink: {path}")
        return (path,)
    if not path.is_dir():
        raise ProductionRunnerError(f"payload is absent: {path}")
    files = tuple(sorted(candidate for candidate in path.rglob("*") if candidate.is_file()))
    if not files or any(candidate.is_symlink() for candidate in files):
        raise ProductionRunnerError(f"payload directory is empty or contains symlinks: {path}")
    return files


def _payload_size(path: Path) -> tuple[int, int]:
    files = _payload_files(path)
    return sum(item.stat().st_size for item in files), len(files)


def _payload_sha256(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    import hashlib

    digest = hashlib.sha256()
    for item in _payload_files(path):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _git_commit(repository_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProductionRunnerError(
            "production runner source is not in the frozen Git checkout"
        ) from exc
