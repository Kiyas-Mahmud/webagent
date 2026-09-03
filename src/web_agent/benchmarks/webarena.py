"""Lazy, fail-closed WebArena/BrowserGym adapter.

This module deliberately has no import-time dependency on WebArena,
BrowserGym, AgentLab, Gymnasium, or Playwright.  A campaign must supply a
version-pinned environment factory and explicit causal/oracle mappers.  That
keeps external API drift from being guessed silently.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import stat
from time import perf_counter
from types import MappingProxyType
from typing import Any

from web_agent.benchmarks.base import (
    AdapterExecution,
    BenchmarkAdapter,
    BenchmarkStateError,
    BenchmarkUnavailableError,
)
from web_agent.runtime.contracts import (
    ConcreteAction,
    ControllerCommandCommitment,
    ExecutionEvidence,
    ExecutionSafetyReceipt,
    ExecutionStatus,
    Observation,
    ObservationStage,
    OpaqueTerminalSignal,
    TaskSpecification,
    VersionedRecord,
    VerifierReceiptBinding,
    canonical_sha256,
    canonical_json,
    detached_record_copy,
)
from web_agent.runtime.observation import assert_oracle_blind_mapping
from web_agent.runtime.state_reset import (
    FrozenWebArenaResetStateAttester,
    WebArenaResetStateReceipt,
    WebArenaResetStateRequest,
)


EnvironmentFactory = Callable[[TaskSpecification, int], Any]
ObservationMapper = Callable[
    [Any, str, ObservationStage, str | None],
    Observation,
]
ActionMapper = Callable[[ConcreteAction], Any]
ScreenshotBytesProvider = Callable[
    [Any, Observation],
    bytes | bytearray | memoryview,
]
EnvironmentStateDigestCallable = Callable[[Any], str]
TerminalSignalMapper = Callable[
    ["FrozenWebArenaObservationSnapshot", TaskSpecification, VerifierReceiptBinding | None],
    OpaqueTerminalSignal,
]
PageSettleCallable = Callable[
    [Any, Any, ObservationStage, float, bool],
    Any,
]
ActionSafetyCallable = Callable[
    [TaskSpecification, Observation, ConcreteAction],
    "WebArenaActionSafetyDecision",
]
ManualRescueGuardCallable = Callable[
    [Any, "WebArenaManualRescueCheck"],
    "WebArenaManualRescueReceipt",
]
ManualRescueEvidenceSink = Callable[
    ["WebArenaManualRescueCheck", "WebArenaManualRescueReceipt"], None
]


REGISTERED_MANUAL_RESCUE_EVIDENCE_MODE = (
    "exclusive_controller_input_audit_v1"
)
REGISTERED_MANUAL_RESCUE_CHECK_STAGES = frozenset(
    {"after_reset", "before_step", "after_step", "before_terminal"}
)


REGISTERED_WEBARENA_INFRASTRUCTURE_REASONS = frozenset(
    {
        "BENCHMARK_SERVICE_UNAVAILABLE",
        "BROWSER_CONTROLLER_DISCONNECTED",
        "ENVIRONMENT_RESET_FAILED",
        "FROZEN_DEPENDENCY_UNAVAILABLE",
    }
)
REGISTERED_WEBARENA_INFRASTRUCTURE_OPERATIONS = frozenset(
    {
        "environment_factory",
        "reset",
        "step",
        "screenshot_bytes_reset",
        "screenshot_bytes_pre_action",
        "screenshot_bytes_post_action",
        "screenshot_bytes_post_recovery",
    }
)
REGISTERED_WEBARENA_OPERATION_REASONS = {
    "environment_factory": frozenset({"BENCHMARK_SERVICE_UNAVAILABLE"}),
    "reset": frozenset(
        {"BENCHMARK_SERVICE_UNAVAILABLE", "ENVIRONMENT_RESET_FAILED"}
    ),
    "step": frozenset(
        {"BENCHMARK_SERVICE_UNAVAILABLE", "BROWSER_CONTROLLER_DISCONNECTED"}
    ),
    "screenshot_bytes_reset": frozenset({"BROWSER_CONTROLLER_DISCONNECTED"}),
    "screenshot_bytes_pre_action": frozenset(
        {"BROWSER_CONTROLLER_DISCONNECTED"}
    ),
    "screenshot_bytes_post_action": frozenset(
        {"BROWSER_CONTROLLER_DISCONNECTED"}
    ),
    "screenshot_bytes_post_recovery": frozenset(
        {"BROWSER_CONTROLLER_DISCONNECTED"}
    ),
}


def _require_lowercase_sha256(name: str, value: object) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deep_freeze_json(value: Any) -> Any:
    """Copy JSON-compatible policy state into recursively immutable containers."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze_json(item) for item in value)
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise TypeError(
        "WebArena observation snapshot contains a non-JSON value: "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class FrozenWebArenaObservationSnapshot(VersionedRecord):
    """Deep-frozen, oracle-blind state visible to the sealed terminal mapper.

    The callback receives this detached value instead of the live browser
    environment.  ``environment_state_sha256`` is produced by the registered
    state digester immediately before the callback and checked again after it.
    """

    observation: Observation
    environment_state_digester_id: str
    environment_state_digester_version: str
    environment_state_sha256: str

    def __post_init__(self) -> None:
        if type(self.observation) is not Observation:
            raise TypeError("snapshot observation must use the exact Observation contract")
        if not self.environment_state_digester_id.strip():
            raise ValueError("environment-state digester identity is required")
        if not self.environment_state_digester_version.strip():
            raise ValueError("environment-state digester version is required")
        _require_lowercase_sha256(
            "environment_state_sha256", self.environment_state_sha256
        )


@dataclass(frozen=True, slots=True)
class FrozenWebArenaEnvironmentStateDigester:
    """Version-pinned digest of the live environment's evaluation state."""

    digester_id: str
    digester_version: str
    benchmark_version: str
    callback: EnvironmentStateDigestCallable
    frozen: bool = True

    def __post_init__(self) -> None:
        if not self.digester_id.strip() or not self.digester_version.strip():
            raise ValueError("environment-state digester identity/version are required")
        if not self.benchmark_version.strip() or self.benchmark_version == "unregistered":
            raise ValueError("environment-state digester benchmark version must be pinned")
        if not callable(self.callback):
            raise TypeError("environment-state digester callback must be callable")
        if self.frozen is not True:
            raise ValueError("evaluation environment-state digester must be frozen")

    def digest(self, environment: Any) -> str:
        return _require_lowercase_sha256(
            "registered environment-state digest",
            self.callback(environment),
        )


@dataclass(frozen=True, slots=True)
class FrozenWebArenaPageSettlePolicy:
    """Version-pinned browser settle operation applied before observation mapping.

    The integration callback receives only the live environment, the causal raw
    observation slot, the current causal stage, and the frozen settle settings.
    It must return the raw observation that is safe to map after waiting for the
    registered browser condition.  Reward, done, info, and verifier evidence are
    never arguments to this callback.
    """

    policy_id: str
    benchmark_version: str
    network_idle_required: bool
    settle_timeout_seconds: float
    callback: PageSettleCallable
    frozen: bool = True

    def __post_init__(self) -> None:
        if type(self.policy_id) is not str or not self.policy_id.strip():
            raise ValueError("page-settle policy identity is required")
        if (
            type(self.benchmark_version) is not str
            or not self.benchmark_version.strip()
            or self.benchmark_version == "unregistered"
        ):
            raise ValueError("page-settle benchmark version must be pinned")
        if type(self.network_idle_required) is not bool:
            raise TypeError("page-settle network-idle setting must be an exact boolean")
        if (
            isinstance(self.settle_timeout_seconds, bool)
            or not isinstance(self.settle_timeout_seconds, (int, float))
            or not math.isfinite(float(self.settle_timeout_seconds))
            or float(self.settle_timeout_seconds) <= 0.0
        ):
            raise ValueError("page-settle timeout must be positive and finite")
        if not callable(self.callback):
            raise TypeError("page-settle callback must be callable")
        if self.frozen is not True or type(self.frozen) is not bool:
            raise ValueError("evaluation page-settle policy must be frozen")

    def settle(
        self,
        environment: Any,
        raw_observation: Any,
        stage: ObservationStage,
    ) -> Any:
        if type(stage) is not ObservationStage:
            raise TypeError("page-settle stage must use ObservationStage")
        settled = self.callback(
            environment,
            raw_observation,
            stage,
            float(self.settle_timeout_seconds),
            self.network_idle_required,
        )
        if settled is None:
            raise ValueError("page-settle callback returned no causal observation")
        return settled


@dataclass(frozen=True, slots=True)
class WebArenaActionSafetyDecision(VersionedRecord):
    """Strict causal result from the registered live-action safety policy.

    The callback must attribute its result to the frozen policy and exact
    concrete action.  A compact reason code is retained instead of arbitrary
    free text so task content, credentials, or browser state cannot be leaked
    into the execution ledger through a denial message.
    """

    safety_policy_id: str
    safety_policy_version: str
    action_id: str
    allowed: bool
    reason_code: str

    def __post_init__(self) -> None:
        for name in (
            "safety_policy_id",
            "safety_policy_version",
            "action_id",
            "reason_code",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"WebArena safety decision requires {name}")
        if type(self.allowed) is not bool:
            raise TypeError(
                "WebArena safety decision allowed must be an exact boolean"
            )
        if len(self.reason_code) > 96 or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
            for character in self.reason_code
        ):
            raise ValueError(
                "WebArena safety reason_code must be a compact uppercase identifier"
            )
        expected_prefix = "ALLOW_" if self.allowed else "DENY_"
        if not self.reason_code.startswith(expected_prefix):
            raise ValueError(
                "WebArena safety reason_code contradicts its allow/deny result"
            )


@dataclass(frozen=True, slots=True)
class FrozenWebArenaActionSafetyPolicy:
    """Source-attested, oracle-blind policy applied before every live step.

    Allowed sites/origins are deliberately not represented here.  The frozen
    integration callback decides from only the registered task, current causal
    observation, and proposed action.  All three inputs are strictly detached
    and hash-guarded so a callback cannot rewrite runtime-owned evidence.
    """

    safety_policy_id: str
    safety_policy_version: str
    destructive_action_policy: str
    benchmark_version: str
    source_sha256: str
    callback: ActionSafetyCallable
    frozen: bool = True
    oracle_labels_exposed_to_runtime: bool = False

    def __post_init__(self) -> None:
        for name in (
            "safety_policy_id",
            "safety_policy_version",
            "destructive_action_policy",
            "benchmark_version",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"WebArena action-safety policy requires {name}")
        if self.benchmark_version == "unregistered":
            raise ValueError("action-safety benchmark version must be pinned")
        _require_lowercase_sha256("action-safety source_sha256", self.source_sha256)
        if not callable(self.callback):
            raise TypeError("action-safety callback must be callable")
        if self.frozen is not True or type(self.frozen) is not bool:
            raise ValueError("evaluation action-safety policy must be frozen")
        if (
            self.oracle_labels_exposed_to_runtime is not False
            or type(self.oracle_labels_exposed_to_runtime) is not bool
        ):
            raise ValueError("action-safety policy must be oracle-blind")

    def evaluate(
        self,
        task: TaskSpecification,
        observation: Observation,
        action: ConcreteAction,
    ) -> WebArenaActionSafetyDecision:
        if type(task) is not TaskSpecification:
            raise TypeError("action-safety task must use TaskSpecification")
        if type(observation) is not Observation:
            raise TypeError("action-safety observation must use Observation")
        if type(action) is not ConcreteAction:
            raise TypeError("action-safety action must use ConcreteAction")
        assert_oracle_blind_mapping(
            task.metadata,
            location="webarena_action_safety.task_metadata",
        )
        assert_oracle_blind_mapping(
            observation.page_state,
            location="webarena_action_safety.current_observation",
        )

        originals = (task, observation, action)
        original_hashes = tuple(record.record_sha256 for record in originals)
        detached = tuple(detached_record_copy(record) for record in originals)
        detached_hashes = tuple(record.record_sha256 for record in detached)
        callback_error: BaseException | None = None
        decision: object | None = None
        try:
            decision = self.callback(
                detached[0],
                detached[1],
                detached[2],
            )
        except BaseException as exc:
            callback_error = exc

        mutated: list[str] = []
        for label, records, expected_hashes in (
            ("runtime", originals, original_hashes),
            ("callback", detached, detached_hashes),
        ):
            for index, (record, expected) in enumerate(
                zip(records, expected_hashes, strict=True)
            ):
                try:
                    actual = record.record_sha256
                except Exception:
                    actual = "<unhashable>"
                if actual != expected:
                    mutated.append(f"{label}[{index}]")
        if mutated:
            error = ValueError(
                "action-safety callback mutated protected causal input: "
                + ", ".join(mutated)
            )
            if callback_error is not None:
                raise error from callback_error
            raise error
        if callback_error is not None:
            raise callback_error
        if type(decision) is not WebArenaActionSafetyDecision:
            raise TypeError(
                "action-safety callback must return WebArenaActionSafetyDecision"
            )
        if (
            decision.safety_policy_id != self.safety_policy_id
            or decision.safety_policy_version != self.safety_policy_version
        ):
            raise ValueError(
                "action-safety decision identity differs from the frozen policy"
            )
        if decision.action_id != action.action_id:
            raise ValueError(
                "action-safety decision is bound to a different concrete action"
            )
        return decision


@dataclass(frozen=True, slots=True)
class WebArenaManualRescueCheck(VersionedRecord):
    """Causal request for one exclusive-controller input-audit receipt."""

    episode_id: str
    task_id: str
    task_sha256: str
    guard_id: str
    guard_version: str
    evidence_mode: str
    check_index: int
    stage: str
    expected_registered_browser_steps: int
    observation_id: str
    observation_sha256: str
    action_id: str | None
    action_sha256: str | None
    previous_receipt_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "task_id",
            "guard_id",
            "guard_version",
            "observation_id",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                raise ValueError(f"manual-rescue check requires {name}")
        for name in (
            "task_sha256",
            "observation_sha256",
            "previous_receipt_sha256",
        ):
            _require_lowercase_sha256(f"manual-rescue {name}", getattr(self, name))
        if self.evidence_mode != REGISTERED_MANUAL_RESCUE_EVIDENCE_MODE:
            raise ValueError("manual-rescue evidence mode is not registered")
        if type(self.check_index) is not int or self.check_index < 1:
            raise ValueError("manual-rescue check_index must be a positive integer")
        if self.stage not in REGISTERED_MANUAL_RESCUE_CHECK_STAGES:
            raise ValueError("manual-rescue check stage is not registered")
        if (
            type(self.expected_registered_browser_steps) is not int
            or self.expected_registered_browser_steps < 0
        ):
            raise ValueError(
                "manual-rescue registered browser-step count must be nonnegative"
            )
        if (self.action_id is None) is not (self.action_sha256 is None):
            raise ValueError("manual-rescue action ID/hash must be jointly present")
        if self.action_sha256 is not None:
            if type(self.action_id) is not str or not self.action_id.strip():
                raise ValueError("manual-rescue action identity is empty")
            _require_lowercase_sha256(
                "manual-rescue action_sha256", self.action_sha256
            )
        if self.stage in {"before_step", "after_step"} and self.action_id is None:
            raise ValueError("manual-rescue step check requires an action binding")
        if self.stage == "after_reset" and self.action_id is not None:
            raise ValueError("manual-rescue reset check cannot cite an action")


@dataclass(frozen=True, slots=True)
class WebArenaManualRescueReceipt(VersionedRecord):
    """Hashes-only result from the frozen browser-controller input audit."""

    guard_id: str
    guard_version: str
    source_sha256: str
    evidence_mode: str
    check_sha256: str
    check_index: int
    stage: str
    expected_registered_browser_steps: int
    observed_registered_browser_steps: int
    automation_session_sha256: str
    controller_input_audit_sha256: str
    exclusive_automation_control: bool
    non_agent_input_event_count: int
    manual_rescue_detected: bool
    oracle_labels_observed: bool = False

    def __post_init__(self) -> None:
        for name in ("guard_id", "guard_version"):
            if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                raise ValueError(f"manual-rescue receipt requires {name}")
        for name in (
            "source_sha256",
            "check_sha256",
            "automation_session_sha256",
            "controller_input_audit_sha256",
        ):
            _require_lowercase_sha256(f"manual-rescue {name}", getattr(self, name))
        if self.evidence_mode != REGISTERED_MANUAL_RESCUE_EVIDENCE_MODE:
            raise ValueError("manual-rescue receipt evidence mode is not registered")
        if type(self.check_index) is not int or self.check_index < 1:
            raise ValueError("manual-rescue receipt index must be positive")
        if self.stage not in REGISTERED_MANUAL_RESCUE_CHECK_STAGES:
            raise ValueError("manual-rescue receipt stage is not registered")
        for name in (
            "expected_registered_browser_steps",
            "observed_registered_browser_steps",
            "non_agent_input_event_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"manual-rescue receipt {name} must be nonnegative")
        for name in (
            "exclusive_automation_control",
            "manual_rescue_detected",
            "oracle_labels_observed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"manual-rescue receipt {name} must be boolean")
        detected = (
            not self.exclusive_automation_control
            or self.non_agent_input_event_count != 0
            or self.observed_registered_browser_steps
            != self.expected_registered_browser_steps
        )
        if self.manual_rescue_detected is not detected:
            raise ValueError(
                "manual-rescue receipt detection flag contradicts its audit evidence"
            )
        if self.oracle_labels_observed is not False:
            raise ValueError("manual-rescue guard must remain oracle-blind")


@dataclass(frozen=True, slots=True)
class FrozenWebArenaManualRescueGuard:
    """Source-attested audit of the exclusive live browser-control surface."""

    guard_id: str
    guard_version: str
    benchmark_version: str
    evidence_mode: str
    source_sha256: str
    callback: ManualRescueGuardCallable
    frozen: bool = True
    oracle_labels_exposed_to_runtime: bool = False

    def __post_init__(self) -> None:
        for name in ("guard_id", "guard_version", "benchmark_version"):
            if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                raise ValueError(f"manual-rescue guard requires {name}")
        if self.benchmark_version == "unregistered":
            raise ValueError("manual-rescue guard benchmark version must be pinned")
        if self.evidence_mode != REGISTERED_MANUAL_RESCUE_EVIDENCE_MODE:
            raise ValueError("manual-rescue guard evidence mode is not registered")
        _require_lowercase_sha256("manual-rescue source_sha256", self.source_sha256)
        if not callable(self.callback):
            raise TypeError("manual-rescue guard callback must be callable")
        if self.frozen is not True or type(self.frozen) is not bool:
            raise ValueError("manual-rescue guard must be frozen")
        if (
            self.oracle_labels_exposed_to_runtime is not False
            or type(self.oracle_labels_exposed_to_runtime) is not bool
        ):
            raise ValueError("manual-rescue guard must be oracle-blind")

    def attest(
        self,
        environment: Any,
        check: WebArenaManualRescueCheck,
    ) -> WebArenaManualRescueReceipt:
        if type(check) is not WebArenaManualRescueCheck:
            raise TypeError("manual-rescue guard requires its typed check")
        if (
            check.guard_id != self.guard_id
            or check.guard_version != self.guard_version
            or check.evidence_mode != self.evidence_mode
        ):
            raise ValueError("manual-rescue check identity differs from frozen guard")
        original_sha256 = check.record_sha256
        callback_check = detached_record_copy(check)
        callback_sha256 = callback_check.record_sha256
        callback_error: BaseException | None = None
        receipt: object | None = None
        try:
            receipt = self.callback(environment, callback_check)
        except BaseException as exc:
            callback_error = exc
        mutated = []
        for label, protected, expected in (
            ("runtime", check, original_sha256),
            ("callback", callback_check, callback_sha256),
        ):
            try:
                actual = protected.record_sha256
            except Exception:
                actual = "<unhashable>"
            if actual != expected:
                mutated.append(label)
        if mutated:
            error = ValueError(
                "manual-rescue callback mutated protected check input: "
                + ", ".join(mutated)
            )
            if callback_error is not None:
                raise error from callback_error
            raise error
        if callback_error is not None:
            raise callback_error
        if type(receipt) is not WebArenaManualRescueReceipt:
            raise TypeError(
                "manual-rescue callback must return WebArenaManualRescueReceipt"
            )
        if (
            receipt.guard_id != self.guard_id
            or receipt.guard_version != self.guard_version
            or receipt.source_sha256 != self.source_sha256
            or receipt.evidence_mode != self.evidence_mode
            or receipt.check_sha256 != check.record_sha256
            or receipt.check_index != check.check_index
            or receipt.stage != check.stage
            or receipt.expected_registered_browser_steps
            != check.expected_registered_browser_steps
        ):
            raise ValueError("manual-rescue receipt is not bound to its frozen check")
        return receipt


@dataclass(frozen=True, slots=True)
class WebArenaInfrastructureClassification(VersionedRecord):
    """One explicit preregistered classification, never an inferred retry."""

    reason_code: str
    failure_class: str
    retryable: bool = True

    def __post_init__(self) -> None:
        if self.reason_code not in REGISTERED_WEBARENA_INFRASTRUCTURE_REASONS:
            raise ValueError(
                "WebArena infrastructure reason is not preregistered"
            )
        if not self.failure_class.strip():
            raise ValueError("WebArena infrastructure failure_class is required")
        if self.retryable is not True:
            raise ValueError("classified WebArena infrastructure fault must be retryable")


@dataclass(frozen=True, slots=True)
class WebArenaInfrastructureRule(VersionedRecord):
    """One exact operation/exception pair allowed to authorize a rerun."""

    operation: str
    exception_type: str
    reason_code: str
    failure_class: str

    def __post_init__(self) -> None:
        if self.operation not in REGISTERED_WEBARENA_INFRASTRUCTURE_OPERATIONS:
            raise ValueError("WebArena infrastructure rule operation is not registered")
        if "." not in self.exception_type or not self.exception_type.strip():
            raise ValueError(
                "WebArena infrastructure rule requires a qualified exception type"
            )
        if self.reason_code not in REGISTERED_WEBARENA_INFRASTRUCTURE_REASONS:
            raise ValueError("WebArena infrastructure rule reason is not registered")
        if self.reason_code not in REGISTERED_WEBARENA_OPERATION_REASONS[
            self.operation
        ]:
            raise ValueError(
                "WebArena infrastructure reason is invalid for this operation"
            )
        if not self.failure_class.strip():
            raise ValueError("WebArena infrastructure rule failure_class is required")


InfrastructureClassifierCallable = Callable[
    [Exception, str],
    WebArenaInfrastructureClassification | None,
]


@dataclass(frozen=True, slots=True)
class FrozenWebArenaInfrastructureFaultClassifier:
    """Version-pinned allow-list classifier supplied by the integration.

    Returning ``None`` is fatal and never authorizes a block rerun.  The
    callback must return the typed classification above; arbitrary truthy
    mappings and broad exception-to-retry coercions are deliberately rejected.
    """

    classifier_id: str
    classifier_version: str
    benchmark_version: str
    callback: InfrastructureClassifierCallable
    rules: tuple[WebArenaInfrastructureRule, ...]
    frozen: bool = True

    def __post_init__(self) -> None:
        if not self.classifier_id.strip() or not self.classifier_version.strip():
            raise ValueError("infrastructure classifier identity/version are required")
        if not self.benchmark_version.strip() or self.benchmark_version == "unregistered":
            raise ValueError("infrastructure classifier benchmark version must be pinned")
        if type(self.rules) is not tuple or not self.rules or any(
            type(rule) is not WebArenaInfrastructureRule for rule in self.rules
        ):
            raise ValueError(
                "infrastructure classifier requires frozen typed exact-match rules"
            )
        keys = [(rule.operation, rule.exception_type) for rule in self.rules]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "infrastructure classifier rules repeat an operation/exception pair"
            )
        if not callable(self.callback):
            raise TypeError("infrastructure classifier callback must be callable")
        if self.frozen is not True:
            raise ValueError("evaluation infrastructure classifier must be frozen")

    @property
    def rules_sha256(self) -> str:
        return canonical_sha256([rule.to_dict() for rule in self.rules])

    def classify(
        self,
        error: Exception,
        operation: str,
    ) -> WebArenaInfrastructureClassification | None:
        if operation not in REGISTERED_WEBARENA_INFRASTRUCTURE_OPERATIONS:
            return None
        exception_type = f"{type(error).__module__}.{type(error).__qualname__}"
        matching = [
            rule
            for rule in self.rules
            if rule.operation == operation and rule.exception_type == exception_type
        ]
        if not matching:
            return None
        result = self.callback(error, operation)
        if result is None:
            return None
        if type(result) is not WebArenaInfrastructureClassification:
            raise TypeError(
                "infrastructure classifier must return the registered typed contract"
            )
        rule = matching[0]
        if (
            result.reason_code != rule.reason_code
            or result.failure_class != rule.failure_class
        ):
            raise ValueError(
                "infrastructure classifier output differs from its exact frozen rule"
            )
        return result


class WebArenaAdapter(BenchmarkAdapter):
    benchmark_id = "webarena"

    def __init__(
        self,
        *,
        benchmark_version: str = "unregistered",
        adapter_id: str | None = None,
        adapter_version: str | None = None,
        dependency_module: str = "browsergym",
        environment_factory: EnvironmentFactory | None = None,
        observation_mapper: ObservationMapper | None = None,
        action_mapper: ActionMapper | None = None,
        screenshot_bytes_provider: ScreenshotBytesProvider | None = None,
        episode_runtime_dir: str | Path | None = None,
        terminal_signal_mapper: TerminalSignalMapper | None = None,
        environment_state_digester: (
            FrozenWebArenaEnvironmentStateDigester | None
        ) = None,
        infrastructure_fault_classifier: (
            FrozenWebArenaInfrastructureFaultClassifier | None
        ) = None,
        reset_state_attester: FrozenWebArenaResetStateAttester | None = None,
        require_reset_state_receipt: bool = False,
        page_settle_policy: FrozenWebArenaPageSettlePolicy | None = None,
        require_page_settle_policy: bool = False,
        action_safety_policy: FrozenWebArenaActionSafetyPolicy | None = None,
        require_action_safety_policy: bool = False,
        manual_rescue_guard: FrozenWebArenaManualRescueGuard | None = None,
        manual_rescue_evidence_sink: ManualRescueEvidenceSink | None = None,
        require_manual_rescue_guard: bool = False,
    ) -> None:
        self.benchmark_version = benchmark_version
        self._adapter_id = adapter_id
        self._adapter_version = adapter_version
        self._dependency_module = dependency_module
        self._environment_factory = environment_factory
        self._observation_mapper = observation_mapper
        self._action_mapper = action_mapper
        self._screenshot_bytes_provider = screenshot_bytes_provider
        self._episode_runtime_dir = (
            Path(episode_runtime_dir) if episode_runtime_dir is not None else None
        )
        self._terminal_signal_mapper = terminal_signal_mapper
        if (
            environment_state_digester is not None
            and environment_state_digester.benchmark_version != benchmark_version
        ):
            raise ValueError(
                "environment-state digester and WebArena benchmark versions differ"
            )
        self._environment_state_digester = environment_state_digester
        if (
            infrastructure_fault_classifier is not None
            and infrastructure_fault_classifier.benchmark_version
            != benchmark_version
        ):
            raise ValueError(
                "infrastructure classifier and WebArena benchmark versions differ"
            )
        self._infrastructure_fault_classifier = infrastructure_fault_classifier
        if infrastructure_fault_classifier is not None and (
            not str(adapter_id or "").strip()
            or not str(adapter_version or "").strip()
        ):
            raise ValueError(
                "classified WebArena faults require the frozen adapter identity"
            )
        self._reset_state_attester = reset_state_attester
        if type(require_reset_state_receipt) is not bool:
            raise TypeError("require_reset_state_receipt must be an exact boolean")
        self._require_reset_state_receipt = require_reset_state_receipt
        if (
            page_settle_policy is not None
            and type(page_settle_policy) is not FrozenWebArenaPageSettlePolicy
        ):
            raise TypeError("page-settle policy has the wrong typed contract")
        if (
            page_settle_policy is not None
            and page_settle_policy.benchmark_version != benchmark_version
        ):
            raise ValueError(
                "page-settle policy and WebArena benchmark versions differ"
            )
        if type(require_page_settle_policy) is not bool:
            raise TypeError("require_page_settle_policy must be an exact boolean")
        self._page_settle_policy = page_settle_policy
        self._require_page_settle_policy = require_page_settle_policy
        if (
            action_safety_policy is not None
            and type(action_safety_policy) is not FrozenWebArenaActionSafetyPolicy
        ):
            raise TypeError("action-safety policy has the wrong typed contract")
        if (
            action_safety_policy is not None
            and action_safety_policy.benchmark_version != benchmark_version
        ):
            raise ValueError(
                "action-safety policy and WebArena benchmark versions differ"
            )
        if type(require_action_safety_policy) is not bool:
            raise TypeError("require_action_safety_policy must be an exact boolean")
        self._action_safety_policy = action_safety_policy
        self._require_action_safety_policy = require_action_safety_policy
        if (
            manual_rescue_guard is not None
            and type(manual_rescue_guard) is not FrozenWebArenaManualRescueGuard
        ):
            raise TypeError("manual-rescue guard has the wrong typed contract")
        if (
            manual_rescue_guard is not None
            and manual_rescue_guard.benchmark_version != benchmark_version
        ):
            raise ValueError(
                "manual-rescue guard and WebArena benchmark versions differ"
            )
        if manual_rescue_evidence_sink is not None and not callable(
            manual_rescue_evidence_sink
        ):
            raise TypeError("manual-rescue evidence sink must be callable")
        if type(require_manual_rescue_guard) is not bool:
            raise TypeError("require_manual_rescue_guard must be an exact boolean")
        self._manual_rescue_guard = manual_rescue_guard
        self._manual_rescue_evidence_sink = manual_rescue_evidence_sink
        self._require_manual_rescue_guard = require_manual_rescue_guard
        self._environment: Any | None = None
        self._raw_observation: Any | None = None
        self._episode_id: str | None = None
        self._task: TaskSpecification | None = None
        self._last_observation: Observation | None = None
        self._pending_observation: Observation | None = None
        self._observation_index = 0
        self._closed = False
        self._infrastructure_event_index = 0
        self._reset_state_receipt: WebArenaResetStateReceipt | None = None
        self._manual_rescue_check_index = 0
        self._manual_rescue_previous_receipt_sha256 = "0" * 64
        self._registered_browser_steps = 0

    @classmethod
    def dependency_available(cls, module: str = "browsergym") -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def _require_contract(self) -> None:
        missing = [
            name
            for name, value in (
                ("environment_factory", self._environment_factory),
                ("observation_mapper", self._observation_mapper),
                ("action_mapper", self._action_mapper),
                ("screenshot_bytes_provider", self._screenshot_bytes_provider),
                ("episode_runtime_dir", self._episode_runtime_dir),
                ("terminal_signal_mapper", self._terminal_signal_mapper),
                ("environment_state_digester", self._environment_state_digester),
                *(
                    (("reset_state_attester", self._reset_state_attester),)
                    if self._require_reset_state_receipt
                    else ()
                ),
                *(
                    (("page_settle_policy", self._page_settle_policy),)
                    if self._require_page_settle_policy
                    else ()
                ),
                *(
                    (("action_safety_policy", self._action_safety_policy),)
                    if self._require_action_safety_policy
                    else ()
                ),
                *(
                    (
                        ("manual_rescue_guard", self._manual_rescue_guard),
                        (
                            "manual_rescue_evidence_sink",
                            self._manual_rescue_evidence_sink,
                        ),
                    )
                    if self._require_manual_rescue_guard
                    else ()
                ),
            )
            if value is None
        ]
        if not self.dependency_available(self._dependency_module):
            raise BenchmarkUnavailableError(
                f"WebArena dependency {self._dependency_module!r} is unavailable; "
                "install the frozen campaign environment before selecting this adapter"
            )
        if missing:
            raise BenchmarkUnavailableError(
                "WebArena automatic API guessing is disabled; supply explicit "
                f"version-pinned integration bindings: {', '.join(missing)}"
            )
        if self.benchmark_version == "unregistered":
            raise BenchmarkUnavailableError("WebArena benchmark_version must be frozen")

    def reset(
        self,
        task: TaskSpecification,
        *,
        episode_id: str,
        seed: int,
    ) -> Observation:
        if self._closed:
            raise BenchmarkStateError("WebArena adapter is closed")
        self._require_contract()
        if task.benchmark_id.lower() != self.benchmark_id:
            raise BenchmarkStateError("task benchmark does not match WebArena adapter")
        if task.benchmark_version != self.benchmark_version:
            raise BenchmarkStateError(
                "task benchmark version differs from the pinned adapter version"
            )
        assert self._environment_factory is not None
        # Bind diagnostic identity before factory/reset so a classified reset
        # fault still cites the exact episode and registered task.
        self._episode_id = episode_id
        self._task = task
        self._last_observation = None
        self._pending_observation = None
        self._observation_index = 0
        self._reset_state_receipt = None
        self._manual_rescue_check_index = 0
        self._manual_rescue_previous_receipt_sha256 = "0" * 64
        self._registered_browser_steps = 0
        screenshot_root = self._require_screenshot_root()
        if any(screenshot_root.iterdir()):
            raise BenchmarkStateError(
                "WebArena screenshot artifact directory must be fresh for the episode"
            )
        try:
            self._environment = self._environment_factory(task, seed)
        except Exception as exc:
            self._raise_classified_or_fatal(
                exc,
                operation="environment_factory",
                fatal_message=(
                    "version-pinned WebArena environment factory failed; "
                    "runtime will not guess another API"
                ),
            )
        try:
            reset_result = self._environment.reset(seed=seed)
        except Exception as exc:
            self._raise_classified_or_fatal(
                exc,
                operation="reset",
                fatal_message=(
                    "version-pinned WebArena reset(seed=...) contract failed; "
                    "runtime will not guess another API"
                ),
            )
        self._raw_observation = (
            reset_result[0]
            if isinstance(reset_result, tuple) and reset_result
            else reset_result
        )
        self._raw_observation = self._settle_observation(
            self._raw_observation,
            ObservationStage.RESET,
        )
        if self._reset_state_attester is not None:
            request = WebArenaResetStateRequest(
                episode_id=episode_id,
                task_id=task.task_id,
                benchmark_version=task.benchmark_version,
                start_state_id=task.start_state_id,
                reset_stage_seed=seed,
            )
            try:
                self._reset_state_receipt = self._reset_state_attester.attest(request)
            except Exception as exc:
                raise BenchmarkStateError(
                    "source-attested WebArena reset-state receipt failed"
                ) from exc
        observation = self._map_observation(ObservationStage.RESET, None)
        self._check_manual_rescue(
            stage="after_reset",
            observation=observation,
        )
        return observation

    def reset_state_receipt(self) -> WebArenaResetStateReceipt | None:
        """Return the immutable receipt produced immediately after reset."""

        return self._reset_state_receipt

    def observe(
        self,
        *,
        stage: ObservationStage,
        prior_action_id: str | None = None,
    ) -> Observation:
        if self._environment is None or self._raw_observation is None:
            raise BenchmarkStateError("WebArena task has not been reset")
        if self._pending_observation is not None:
            pending = self._pending_observation
            if pending.stage is not stage or pending.prior_action_id != prior_action_id:
                raise BenchmarkStateError(
                    "WebArena pending post-action observation does not match request"
                )
            self._pending_observation = None
            self._last_observation = pending
            return pending
        return self._map_observation(stage, prior_action_id)

    def _map_observation(
        self,
        stage: ObservationStage,
        prior_action_id: str | None,
        *,
        register: bool = True,
    ) -> Observation:
        assert self._observation_mapper is not None
        try:
            observation = self._observation_mapper(
                self._raw_observation,
                self._episode_id or "uninitialised",
                stage,
                prior_action_id,
            )
        except Exception as exc:
            raise BenchmarkStateError(
                "version-pinned WebArena observation mapper failed; mapper "
                "errors are fatal integration defects, never infrastructure reruns"
            ) from exc
        if not isinstance(observation, Observation):
            raise BenchmarkStateError("WebArena observation mapper returned invalid data")
        assert_oracle_blind_mapping(
            observation.page_state,
            location="webarena_policy_observation",
        )
        if observation.stage is not stage:
            raise BenchmarkStateError("WebArena observation mapper changed causal stage")
        if observation.episode_id != self._episode_id:
            raise BenchmarkStateError("WebArena observation mapper changed episode ID")
        if observation.prior_action_id != prior_action_id:
            raise BenchmarkStateError("WebArena observation mapper changed prior action ID")
        if self._page_settle_policy is not None and observation.page_settled is not True:
            raise BenchmarkStateError(
                "WebArena observation mapper did not attest the registered settled state"
            )
        observation = self._materialize_screenshot(observation, stage=stage)
        # Detach the policy/evaluator view from mapper-owned mutable objects.
        # A frozen dataclass alone is only shallowly immutable, so every nested
        # list/mapping must be copied into immutable containers as well.
        try:
            frozen_observation = replace(
                observation,
                page_state=_deep_freeze_json(observation.page_state),
            )
        except (TypeError, ValueError) as exc:
            raise BenchmarkStateError(
                "WebArena observation cannot be deep-frozen deterministically"
            ) from exc
        if register:
            self._last_observation = frozen_observation
        return frozen_observation

    def _require_screenshot_root(self) -> Path:
        runtime = self._episode_runtime_dir
        if runtime is None:
            raise BenchmarkStateError("WebArena episode runtime directory is required")
        if runtime.is_symlink() or not runtime.is_dir():
            raise BenchmarkStateError(
                "WebArena episode runtime directory must be an existing non-symlink directory"
            )
        resolved_runtime = runtime.resolve()
        screenshot_root = resolved_runtime / "screenshots"
        try:
            screenshot_root.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc:
            raise BenchmarkStateError(
                "WebArena screenshot artifact directory cannot be created"
            ) from exc
        if screenshot_root.is_symlink() or not screenshot_root.is_dir():
            raise BenchmarkStateError(
                "WebArena screenshot artifact directory must not be a symlink"
            )
        if screenshot_root.resolve() != screenshot_root:
            raise BenchmarkStateError("WebArena screenshot directory escaped runtime")
        return screenshot_root

    def _materialize_screenshot(
        self,
        observation: Observation,
        *,
        stage: ObservationStage,
    ) -> Observation:
        """Own the exact immutable screenshot bytes consumed by policy code."""

        if observation.screenshot_path is not None:
            raise BenchmarkStateError(
                "WebArena observation mapper must not supply a screenshot path; "
                "adapter-owned byte materialization is mandatory"
            )
        assert self._screenshot_bytes_provider is not None
        try:
            captured = self._screenshot_bytes_provider(
                self._raw_observation,
                observation,
            )
        except Exception as exc:
            self._raise_classified_or_fatal(
                exc,
                operation=f"screenshot_bytes_{stage.value}",
                fatal_message=(
                    "version-pinned WebArena screenshot-byte provider failed; "
                    "runtime will not trust a mapper path or hash"
                ),
            )
        if not isinstance(captured, (bytes, bytearray, memoryview)):
            raise BenchmarkStateError(
                "WebArena screenshot-byte provider returned a non-bytes value"
            )
        screenshot_bytes = bytes(captured)
        if not screenshot_bytes:
            raise BenchmarkStateError("WebArena screenshot-byte provider returned empty bytes")
        digest = hashlib.sha256(screenshot_bytes).hexdigest()
        try:
            claimed = _require_lowercase_sha256(
                "mapper screenshot_sha256",
                observation.screenshot_sha256,
            )
        except ValueError as exc:
            raise BenchmarkStateError(
                "WebArena observation mapper supplied an invalid screenshot hash"
            ) from exc
        if claimed != digest:
            raise BenchmarkStateError(
                "WebArena observation mapper screenshot hash differs from captured bytes"
            )

        screenshot_root = self._require_screenshot_root()
        self._observation_index += 1
        destination = screenshot_root / (
            f"{self._observation_index:06d}-{digest}.png"
        )
        if destination.is_symlink() or destination.exists():
            raise BenchmarkStateError(
                "WebArena screenshot artifact path is not a fresh regular file"
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor: int | None = None
        try:
            descriptor = os.open(destination, flags, 0o400)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(screenshot_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            destination.unlink(missing_ok=True)
            raise BenchmarkStateError(
                "WebArena immutable screenshot artifact could not be materialized"
            ) from exc
        metadata = destination.lstat()
        if (
            destination.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or bool(
                metadata.st_mode
                & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            )
            or hashlib.sha256(destination.read_bytes()).hexdigest() != digest
        ):
            raise BenchmarkStateError(
                "WebArena materialized screenshot failed its byte-integrity check"
            )
        return replace(
            observation,
            screenshot_sha256=digest,
            screenshot_path=str(destination),
        )

    def execute(self, action: ConcreteAction) -> AdapterExecution:
        if self._environment is None or self._last_observation is None:
            raise BenchmarkStateError("WebArena task has not been reset")
        if self._pending_observation is not None:
            raise BenchmarkStateError(
                "WebArena pending post-action observation must be consumed first"
            )
        assert self._action_mapper is not None
        assert self._task is not None
        previous_observation = self._last_observation
        start = perf_counter()
        started_at_utc = _utc_now()
        self._check_manual_rescue(
            stage="before_step",
            observation=previous_observation,
            action=action,
        )
        safety_policy = self._action_safety_policy
        safety_decision: WebArenaActionSafetyDecision | None = None
        safety_receipt: ExecutionSafetyReceipt | None = None
        if safety_policy is not None:
            try:
                safety_decision = safety_policy.evaluate(
                    self._task,
                    previous_observation,
                    action,
                )
            except BaseException as exc:
                raise BenchmarkStateError(
                    "version-pinned WebArena action-safety policy failed before "
                    "browser action mapping"
                ) from exc
            safety_receipt = ExecutionSafetyReceipt(
                safety_policy_id=safety_decision.safety_policy_id,
                safety_policy_version=safety_decision.safety_policy_version,
                action_id=safety_decision.action_id,
                allowed=safety_decision.allowed,
                reason_code=safety_decision.reason_code,
                source_record_type=type(safety_decision).__name__,
                source_record_sha256=safety_decision.record_sha256,
            )
            if safety_decision.allowed is not True:
                self._check_manual_rescue(
                    stage="after_step",
                    observation=previous_observation,
                    action=action,
                )
                ended_at_utc = _utc_now()
                return AdapterExecution(
                    status=ExecutionStatus.REJECTED,
                    state_changed=False,
                    environment_error=False,
                    error_kind="registered_safety_denial",
                    message=(
                        "registered action-safety denial: "
                        f"{safety_decision.reason_code}"
                    ),
                    internal_retry_count=0,
                    latency_ms=(perf_counter() - start) * 1000.0,
                    evidence=ExecutionEvidence(
                        action_id=action.action_id,
                        started_at_utc=started_at_utc,
                        ended_at_utc=ended_at_utc,
                        status=ExecutionStatus.REJECTED,
                        safety_receipt=safety_receipt,
                    ),
                )
        try:
            native_action = self._action_mapper(action)
        except Exception as exc:
            error = BenchmarkStateError(
                "version-pinned WebArena action mapper failed; mapper errors "
                "are fatal integration defects, never infrastructure reruns"
            )
            setattr(
                error,
                "adapter_execution",
                AdapterExecution(
                    status=ExecutionStatus.ERROR,
                    state_changed=False,
                    environment_error=False,
                    error_kind="action_mapper_failure",
                    message="registered WebArena action mapper failed",
                    latency_ms=(perf_counter() - start) * 1000.0,
                    evidence=ExecutionEvidence(
                        action_id=action.action_id,
                        started_at_utc=started_at_utc,
                        ended_at_utc=_utc_now(),
                        status=ExecutionStatus.ERROR,
                        safety_receipt=safety_receipt,
                    ),
                ),
            )
            raise error from exc
        try:
            native_canonical = canonical_json(native_action).encode("utf-8")
        except (TypeError, ValueError) as exc:
            error = BenchmarkStateError(
                "version-pinned WebArena action mapper returned a command that "
                "cannot be committed with canonical JSON"
            )
            setattr(
                error,
                "adapter_execution",
                AdapterExecution(
                    status=ExecutionStatus.ERROR,
                    state_changed=False,
                    environment_error=False,
                    error_kind="uncommittable_controller_command",
                    message="native controller command commitment failed",
                    latency_ms=(perf_counter() - start) * 1000.0,
                    evidence=ExecutionEvidence(
                        action_id=action.action_id,
                        started_at_utc=started_at_utc,
                        ended_at_utc=_utc_now(),
                        status=ExecutionStatus.ERROR,
                        safety_receipt=safety_receipt,
                    ),
                ),
            )
            raise error from exc
        command_commitment = ControllerCommandCommitment(
            action_id=action.action_id,
            adapter_id=str(self._adapter_id or "unregistered-adapter"),
            adapter_version=str(self._adapter_version or "unregistered-version"),
            command_type=(
                f"{type(native_action).__module__}.{type(native_action).__qualname__}"
            ),
            command_sha256=hashlib.sha256(native_canonical).hexdigest(),
            canonical_byte_length=len(native_canonical),
        )
        try:
            step_result = self._environment.step(native_action)
        except Exception as exc:
            self._raise_classified_or_fatal(
                exc,
                operation="step",
                fatal_message=(
                    "version-pinned WebArena step contract failed; runtime will "
                    "not guess execution status or infrastructure taxonomy"
                ),
                execution_evidence=AdapterExecution(
                    status=ExecutionStatus.ERROR,
                    state_changed=False,
                    environment_error=True,
                    error_kind="controller_step_interruption",
                    message="native controller step did not return",
                    latency_ms=(perf_counter() - start) * 1000.0,
                    evidence=ExecutionEvidence(
                        action_id=action.action_id,
                        started_at_utc=started_at_utc,
                        ended_at_utc=_utc_now(),
                        status=ExecutionStatus.ERROR,
                        controller_command=command_commitment,
                        safety_receipt=safety_receipt,
                    ),
                ),
            )
        self._registered_browser_steps += 1
        post_stage = (
            ObservationStage.POST_RECOVERY
            if action.recovery_attempt_id is not None
            else ObservationStage.POST_ACTION
        )
        self._raw_observation = self._settle_observation(
            self._trusted_post_observation(step_result),
            post_stage,
        )
        self._last_observation = None
        post_observation = self._map_observation(
            post_stage,
            action.action_id,
            register=False,
        )
        self._check_manual_rescue(
            stage="after_step",
            observation=post_observation,
            action=action,
        )
        state_changed = canonical_sha256(
            self._observable_execution_state(previous_observation)
        ) != canonical_sha256(self._observable_execution_state(post_observation))
        environment_error = post_observation.environment_error
        status = (
            ExecutionStatus.ERROR
            if environment_error
            else ExecutionStatus.EXECUTED
        )
        self._pending_observation = post_observation
        # Sealed verification is forbidden until EpisodeRunner consumes and
        # logs this exact pending observation through ``observe``.
        ended_at_utc = _utc_now()
        return AdapterExecution(
            status=status,
            state_changed=state_changed,
            environment_error=environment_error,
            error_kind=(
                "observable_environment_error" if environment_error else None
            ),
            message=(
                "mapped observation reports browser-local environment error"
                if environment_error
                else "browser request completed"
            )
            + (
                f"; safety={safety_decision.reason_code}"
                if safety_decision is not None
                else ""
            ),
            internal_retry_count=0,
            latency_ms=(perf_counter() - start) * 1000.0,
            evidence=ExecutionEvidence(
                action_id=action.action_id,
                started_at_utc=started_at_utc,
                ended_at_utc=ended_at_utc,
                status=status,
                controller_command=command_commitment,
                safety_receipt=safety_receipt,
            ),
        )

    def _settle_observation(
        self,
        raw_observation: Any,
        stage: ObservationStage,
    ) -> Any:
        policy = self._page_settle_policy
        if policy is None:
            return raw_observation
        if self._environment is None:
            raise BenchmarkStateError("WebArena page settlement requires an environment")
        try:
            return policy.settle(self._environment, raw_observation, stage)
        except Exception as exc:
            raise BenchmarkStateError(
                "version-pinned WebArena page-settle policy failed; runtime will "
                "not map an unsettled browser observation"
            ) from exc

    @staticmethod
    def _trusted_post_observation(step_result: Any) -> Any:
        """Extract only Gym's observation slot; discard all oracle-bearing slots."""

        if type(step_result) is not tuple or len(step_result) not in {4, 5}:
            raise BenchmarkStateError(
                "WebArena step must return the pinned Gym 4/5-tuple contract"
            )
        post_raw = step_result[0]
        if post_raw is None:
            raise BenchmarkStateError("WebArena step returned no post observation")
        # Indices 1..4 (reward, terminated/done, truncated, info) are neither
        # inspected nor forwarded to any mapper, callback, runtime record, or
        # recovery/memory decision.
        return post_raw

    @staticmethod
    def _observable_execution_state(observation: Observation) -> Mapping[str, Any]:
        """Fields allowed to influence runtime execution/recovery evidence."""

        return {
            "screenshot_sha256": observation.screenshot_sha256,
            "width": observation.width,
            "height": observation.height,
            "url": observation.url,
            "title": observation.title,
            "page_state": observation.page_state,
            "page_settled": observation.page_settled,
            "environment_error": observation.environment_error,
        }

    def terminal_signal(
        self,
        task: TaskSpecification,
        binding: VerifierReceiptBinding | None = None,
    ) -> OpaqueTerminalSignal:
        if self._environment is None or self._task is None:
            raise BenchmarkStateError("WebArena task has not been reset")
        if task.task_id != self._task.task_id:
            raise BenchmarkStateError("verification task differs from active WebArena task")
        if self._last_observation is None:
            raise BenchmarkStateError("WebArena has no registered observation snapshot")
        if binding is not None and (
            binding.observation_id != self._last_observation.observation_id
            or binding.observation_sha256 != self._last_observation.record_sha256
        ):
            raise BenchmarkStateError(
                "verifier receipt binding differs from the registered observation snapshot"
            )
        self._check_manual_rescue(
            stage="before_terminal",
            observation=self._last_observation,
            action_id=(binding.action_id if binding is not None else None),
            action_sha256=(binding.action_sha256 if binding is not None else None),
        )
        assert self._terminal_signal_mapper is not None
        assert self._environment_state_digester is not None
        try:
            environment_digest_before = self._environment_state_digester.digest(
                self._environment
            )
        except Exception as exc:
            raise BenchmarkStateError(
                "registered environment-state digest failed before terminal mapping"
            ) from exc
        snapshot = FrozenWebArenaObservationSnapshot(
            observation=replace(
                self._last_observation,
                page_state=_deep_freeze_json(self._last_observation.page_state),
            ),
            environment_state_digester_id=(
                self._environment_state_digester.digester_id
            ),
            environment_state_digester_version=(
                self._environment_state_digester.digester_version
            ),
            environment_state_sha256=environment_digest_before,
        )
        snapshot_digest_before = snapshot.record_sha256
        mapper_error: Exception | None = None
        result: object | None = None
        try:
            result = self._terminal_signal_mapper(snapshot, task, binding)
        except Exception as exc:
            mapper_error = exc
        try:
            environment_digest_after = self._environment_state_digester.digest(
                self._environment
            )
        except Exception as exc:
            raise BenchmarkStateError(
                "registered environment-state digest failed after terminal mapping"
            ) from exc
        if snapshot.record_sha256 != snapshot_digest_before:
            raise BenchmarkStateError(
                "terminal signal mapper mutated its immutable observation snapshot"
            ) from mapper_error
        if environment_digest_after != environment_digest_before:
            raise BenchmarkStateError(
                "terminal signal mapper changed live environment state"
            ) from mapper_error
        if mapper_error is not None:
            raise BenchmarkStateError(
                "terminal signal mapper rejected the immutable observation contract"
            ) from mapper_error
        if not isinstance(result, OpaqueTerminalSignal):
            raise BenchmarkStateError(
                "WebArena sealed evaluator returned more than an opaque terminal signal"
            )
        return result

    def _check_manual_rescue(
        self,
        *,
        stage: str,
        observation: Observation,
        action: ConcreteAction | None = None,
        action_id: str | None = None,
        action_sha256: str | None = None,
    ) -> None:
        guard = self._manual_rescue_guard
        if guard is None:
            return
        if self._environment is None or self._task is None or self._episode_id is None:
            raise BenchmarkStateError("manual-rescue guard lacks an active WebArena task")
        if action is not None:
            if action_id is not None or action_sha256 is not None:
                raise BenchmarkStateError(
                    "manual-rescue guard received duplicate action binding"
                )
            action_id = action.action_id
            action_sha256 = action.record_sha256
        self._manual_rescue_check_index += 1
        check = WebArenaManualRescueCheck(
            episode_id=self._episode_id,
            task_id=self._task.task_id,
            task_sha256=self._task.record_sha256,
            guard_id=guard.guard_id,
            guard_version=guard.guard_version,
            evidence_mode=guard.evidence_mode,
            check_index=self._manual_rescue_check_index,
            stage=stage,
            expected_registered_browser_steps=self._registered_browser_steps,
            observation_id=observation.observation_id,
            observation_sha256=observation.record_sha256,
            action_id=action_id,
            action_sha256=action_sha256,
            previous_receipt_sha256=self._manual_rescue_previous_receipt_sha256,
        )
        assert self._environment_state_digester is not None
        try:
            before = self._environment_state_digester.digest(self._environment)
            receipt = guard.attest(self._environment, check)
            after = self._environment_state_digester.digest(self._environment)
        except BaseException as exc:
            raise BenchmarkStateError(
                "source-attested manual-rescue guard failed"
            ) from exc
        if after != before:
            raise BenchmarkStateError(
                "manual-rescue guard changed live environment state"
            )
        sink = self._manual_rescue_evidence_sink
        if sink is None:
            if self._require_manual_rescue_guard:
                raise BenchmarkStateError(
                    "manual-rescue guard lacks its canonical evidence sink"
                )
        else:
            sink(check, receipt)
        self._manual_rescue_previous_receipt_sha256 = receipt.record_sha256
        if receipt.manual_rescue_detected:
            raise BenchmarkStateError(
                "manual browser intervention detected by frozen guard"
            )

    def _raise_classified_or_fatal(
        self,
        error: Exception,
        *,
        operation: str,
        fatal_message: str,
        execution_evidence: AdapterExecution | None = None,
    ) -> None:
        classifier = self._infrastructure_fault_classifier
        classification = (
            classifier.classify(error, operation) if classifier is not None else None
        )
        if classification is None:
            fatal = BenchmarkStateError(fatal_message)
            if execution_evidence is not None:
                setattr(fatal, "adapter_execution", execution_evidence)
            raise fatal from error

        # Import lazily so ordinary runtime/fixture use does not import the
        # evaluation package or create a benchmark<->evaluator import cycle.
        from web_agent.eval.table2.execution_guard import InfrastructureInvalidError

        self._infrastructure_event_index += 1
        exception_type = (
            f"{type(error).__module__}.{type(error).__qualname__}"
        )
        diagnostic_identity = {
            "classifier_id": classifier.classifier_id,
            "classifier_version": classifier.classifier_version,
            "classifier_rules_sha256": classifier.rules_sha256,
            "benchmark_version": self.benchmark_version,
            "operation": operation,
            "failure_class": classification.failure_class,
            "exception_type": exception_type,
            "episode_id": self._episode_id,
            "task_id": self._task.task_id if self._task is not None else None,
        }
        invalid = InfrastructureInvalidError(
            reason_code=classification.reason_code,
            adapter_id=str(self._adapter_id),
            adapter_version=str(self._adapter_version),
            operation=operation,
            adapter_evidence={
                "adapter_event_id": (
                    f"{self._episode_id or 'pre-reset'}:infrastructure:"
                    f"{self._infrastructure_event_index}"
                ),
                "failure_class": classification.failure_class,
                "diagnostic_sha256": canonical_sha256(diagnostic_identity),
                "retryable": True,
                **diagnostic_identity,
            },
        )
        if execution_evidence is not None:
            setattr(invalid, "adapter_execution", execution_evidence)
        raise invalid from error

    def close(self) -> None:
        environment = self._environment
        self._environment = None
        self._raw_observation = None
        self._task = None
        self._last_observation = None
        self._pending_observation = None
        self._closed = True
        if environment is not None and hasattr(environment, "close"):
            environment.close()
