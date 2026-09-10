"""Source-attested, fail-closed live integration for the PC-01 Table 2 pilot.

The production runner deliberately gives its integration entrypoint only model
payloads and a frozen runtime identity.  Deployment-owned browser credentials,
reset operations, controller transport, recovery planning, and measurement are
therefore installed through the immutable registry in this module *before* the
runner imports the entrypoint.  Nothing in this module supplies a default,
fixture, heuristic, or synthetic replacement for a missing live capability.

The sealed evaluator is intentionally absent from every contract below.  A
registered browser factory receives only an oracle-blind ``TaskSpecification``;
model callbacks receive the still smaller ``RuntimeTaskView`` in the canonical
runner.  Reward, Gym termination/info slots, verifier truth, relevance labels,
and reference trajectories never enter this integration boundary.

Those typed contracts are reviewed-code engineering controls, not a process
sandbox.  The current browser factory uses the same-process broker fixture, in
whose interpreter the sealed capability remains importable.  Canonical live
launch is therefore blocked before this provider module or any external factory
is imported; these classes remain constructible only for deterministic tests
until a separately authenticated process-isolated broker is registered.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import functools
import hashlib
import inspect
import os
from pathlib import Path
import stat
from threading import Lock
from types import MappingProxyType
from typing import Any

from web_agent.benchmarks.browsergym_webarena import (
    BrowserGymWebArenaRuntimeFactory,
    FrozenBrowserGymTaskStateResetter,
)
from web_agent.benchmarks.webarena import (
    FrozenWebArenaActionSafetyPolicy,
    FrozenWebArenaManualRescueGuard,
    WebArenaActionSafetyDecision,
    WebArenaManualRescueCheck,
    WebArenaManualRescueReceipt,
)
from web_agent.eval.table2.common import read_json, sha256_file, sha256_json
from web_agent.eval.table2.live_page_broker_assembly import (
    assert_process_wide_broker_assembly,
    process_runtime_page_publisher,
)
from web_agent.eval.table2.pc01_artifacts import (
    PC01_EXPECTED_BASE_SNAPSHOT_SHA256,
    PC01_EXPECTED_CHECKPOINT_SHA256,
    PC01_EXPECTED_CONFIG_SHA256,
    PC01_MODEL_REVISION,
)
from web_agent.eval.table2.process_broker_episode_factory import (
    CredentialFreeProcessIsolatedWebArenaEpisodeFactory,
)
from web_agent.eval.table2.execution_guard import (
    PC01_PROVIDER_BOOTSTRAP_SCHEMA_VERSION,
    PC01ProviderInstallationReceipt,
    PC01_PROVIDER_PUBLIC_CONTRACT_SCHEMA_VERSION,
)
from web_agent.eval.table2.webarena_preflight import (
    PINNED_WEBARENA_SERVICE_URL_KEYS,
)
from web_agent.eval.table2.production_runner import (
    EvaluationRuntimeBinding,
    FrozenRuntimeContext,
    RUNTIME_CAPABILITY_AUTHORITY_SCHEMA_VERSION,
    RUNTIME_LIVE_CAPABILITY_IDS,
    SeedRuntimeBinding,
    WebArenaRuntimeBinding,
    validate_runtime_capability_authority,
)
from web_agent.runtime.contracts import TaskSpecification
from web_agent.runtime.observation import assert_oracle_blind_mapping
from web_agent.runtime.qwen2vl_pc01 import (
    PC01EvaluationRuntimeFactory,
    PC01SeedOperationalServices,
)
from web_agent.runtime.recovery.controller import RecoveryActionPlanner
from web_agent.runtime.state_reset import (
    CallableEpisodeStateResetter,
    EpisodeStateResetEvidence,
    EpisodeStateResetRequest,
    EpisodeStateResetter,
    FrozenWebArenaResetStateAttester,
    WebArenaResetStateReceipt,
    WebArenaResetStateRequest,
    evidence_from_backend_digests,
    webarena_reset_receipt_from_commitments,
)


PC01_LIVE_INTEGRATION_ENTRYPOINT = (
    "web_agent.runtime.pc01_live_integration:create_pc01_live_runtime"
)
PC01_LIVE_INTEGRATION_SOURCE = "src/web_agent/runtime/pc01_live_integration.py"
PC01_PARAMETER_FALLBACK_CONTRACT = (
    "unadapted-e0-backbone-strict-reject-no-repair-v1"
)
PC01_CHECKPOINT_SYSTEMS = ("E1", "E2", "E3")
PC01_LIVE_MODEL_SEED = 42


class PC01LiveIntegrationError(RuntimeError):
    """A live capability is absent, mutable, mismatched, or crosses a boundary."""


def _lexical_absolute_path(value: str | Path) -> Path:
    """Return an absolute path without following any symlink component."""

    return Path(os.path.abspath(os.fspath(value)))


def _assert_no_symlink_components(path: Path, *, label: str) -> None:
    """Reject a symlink at the leaf or in any existing parent component."""

    absolute = _lexical_absolute_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            # Credential roots must exist, so the caller will reject this path.
            # Stopping here also avoids accidentally resolving a later component.
            break
        if stat.S_ISLNK(mode):
            raise PC01LiveIntegrationError(
                f"{label} must not contain a symlink component: {current}"
            )


def validate_external_credential_capability_root(
    root: str | Path,
    *,
    forbidden_roots: tuple[str | Path, ...] = (),
) -> Path:
    """Return one existing credential directory disjoint from protected trees.

    Disjointness is deliberately symmetric: a credential root may be neither
    inside a protected tree nor an ancestor from which that tree is reachable.
    """

    unresolved = _lexical_absolute_path(root)
    if unresolved == Path(unresolved.anchor):
        raise PC01LiveIntegrationError(
            "credential capability root cannot be a filesystem root"
        )
    _assert_no_symlink_components(unresolved, label="credential capability root")
    if not unresolved.is_dir():
        raise PC01LiveIntegrationError(
            "credential capability root must be an external regular directory"
        )
    resolved = unresolved.resolve(strict=True)
    protected = (_repository_root(), *forbidden_roots)
    for candidate in protected:
        protected_root = _lexical_absolute_path(candidate).resolve()
        if (
            resolved == protected_root
            or resolved in protected_root.parents
            or protected_root in resolved.parents
        ):
            raise PC01LiveIntegrationError(
                "credential capability root must be tree-disjoint from source "
                "and campaign roots"
            )
    return resolved


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _reject_bootstrap_paths(value: Any, *, location: str) -> None:
    if isinstance(value, Path):
        raise PC01LiveIntegrationError(
            f"provider bootstrap contains a filesystem path at {location}"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_bootstrap_paths(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_bootstrap_paths(item, location=f"{location}[{index}]")


@dataclass(frozen=True, slots=True)
class PC01ExternalCredentialCapability:
    """Non-secret handle to an operator-owned credential directory."""

    capability_id: str
    capability_version: str
    root: Path
    credential_material_embedded: bool = False
    frozen: bool = True

    def __post_init__(self) -> None:
        if not self.capability_id.strip() or not self.capability_version.strip():
            raise PC01LiveIntegrationError("credential capability identity is required")
        if self.credential_material_embedded is not False or self.frozen is not True:
            raise PC01LiveIntegrationError(
                "credential capability must be non-secret and frozen"
            )
        resolved = validate_external_credential_capability_root(self.root)
        object.__setattr__(self, "root", resolved)

    @property
    def public_identity(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "capability_id": self.capability_id,
                "capability_version": self.capability_version,
                "root_path_sha256": hashlib.sha256(
                    str(self.root).encode("utf-8")
                ).hexdigest(),
                "credential_material_embedded": False,
                "frozen": True,
            }
        )


@dataclass(frozen=True, slots=True)
class PC01ProviderBootstrapContext:
    """Detached oracle-free input accepted by the provider factory."""

    schema_version: str
    protocol_id: str
    model_seed: int
    repository_commit: str
    runtime_identity_sha256: str
    runtime_capability_authority: Mapping[str, Any]
    runtime_environment: Mapping[str, Any]
    deployment_preflight_view: Mapping[str, Any]
    service_url_map: Mapping[str, str]
    credential_capability: PC01ExternalCredentialCapability

    def __post_init__(self) -> None:
        if self.schema_version != PC01_PROVIDER_BOOTSTRAP_SCHEMA_VERSION:
            raise PC01LiveIntegrationError("provider bootstrap schema version changed")
        if self.protocol_id != "table2-pc01-pilot-v1":
            raise PC01LiveIntegrationError("provider bootstrap protocol is not registered")
        if self.model_seed != PC01_LIVE_MODEL_SEED or type(self.model_seed) is not int:
            raise PC01LiveIntegrationError("provider bootstrap must bind seed 42")
        if (
            type(self.repository_commit) is not str
            or len(self.repository_commit) != 40
            or any(character not in "0123456789abcdef" for character in self.repository_commit)
        ):
            raise PC01LiveIntegrationError("provider bootstrap Git commit is invalid")
        _require_sha256(
            self.runtime_identity_sha256,
            field="provider bootstrap runtime identity",
        )
        if type(self.credential_capability) is not PC01ExternalCredentialCapability:
            raise PC01LiveIntegrationError(
                "provider bootstrap requires the exact credential capability"
            )
        if set(self.service_url_map) != set(PINNED_WEBARENA_SERVICE_URL_KEYS):
            raise PC01LiveIntegrationError(
                "provider bootstrap requires the seven registered service URLs"
            )
        for key, value in self.service_url_map.items():
            if not isinstance(key, str) or not isinstance(value, str) or not value:
                raise PC01LiveIntegrationError("provider bootstrap service URL is invalid")
        for field in (
            "runtime_capability_authority",
            "runtime_environment",
            "deployment_preflight_view",
        ):
            value = getattr(self, field)
            if not isinstance(value, Mapping):
                raise PC01LiveIntegrationError(f"provider bootstrap {field} is invalid")
            oracle_guard_value: Mapping[str, Any] = value
            if field == "runtime_capability_authority" and isinstance(
                value.get("capabilities"), Mapping
            ):
                # Capability map keys are separately constrained by the exact
                # six-item registry. Guard the fixed row schemas without
                # globally exempting the safety capability whose registered
                # ID contains the word ``oracle``.
                oracle_guard_value = {
                    **value,
                    "capabilities": list(value["capabilities"].values()),
                }
            assert_oracle_blind_mapping(
                oracle_guard_value,
                location=f"provider_bootstrap.{field}",
            )
            _reject_bootstrap_paths(value, location=f"provider_bootstrap.{field}")
            object.__setattr__(self, field, _deep_freeze(value))
        if (
            self.deployment_preflight_view.get("schema_version")
            != "table2-pc01-runtime-preflight-view-v1"
            or self.deployment_preflight_view.get("preflight_status") != "PASS"
            or self.deployment_preflight_view.get(
                "service_url_map_content_sha256"
            )
            != sha256_json(dict(self.service_url_map))
        ):
            raise PC01LiveIntegrationError(
                "provider bootstrap preflight view/URL commitment differs"
            )
        authority = validate_runtime_capability_authority(
            self.runtime_capability_authority
        )
        if authority["deployment_preflight_binding_sha256"] != sha256_json(
            dict(self.deployment_preflight_view)
        ):
            raise PC01LiveIntegrationError(
                "provider bootstrap preflight view differs from capability authority"
            )
        object.__setattr__(self, "service_url_map", _deep_freeze(self.service_url_map))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def integration_source_sha256() -> str:
    """Return the exact bytes that a handoff must attest for this entrypoint."""

    source = Path(__file__).absolute()
    if source.is_symlink() or not source.is_file():
        raise PC01LiveIntegrationError(
            "PC-01 live integration source must be a regular non-symlink file"
        )
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PC01LiveIntegrationError(f"{field} must be a lowercase SHA-256")
    return value


def _callable_source(callback: Callable[..., Any]) -> tuple[str, str]:
    """Resolve and hash one executable callable inside the current repository."""

    target: Any = callback.func if isinstance(callback, functools.partial) else callback
    if inspect.ismethod(target):
        target = target.__func__
    elif not (inspect.isfunction(target) or inspect.isclass(target)):
        target = getattr(type(target), "__call__", None)
    if target is None:
        raise PC01LiveIntegrationError("cannot resolve a live operation's source")
    try:
        source_name = inspect.getsourcefile(target) or inspect.getfile(target)
    except (OSError, TypeError) as exc:
        raise PC01LiveIntegrationError(
            "cannot resolve a live operation's source"
        ) from exc
    if not source_name:
        raise PC01LiveIntegrationError("live operation has no source file")
    unresolved = Path(source_name).absolute()
    if unresolved.is_symlink():
        raise PC01LiveIntegrationError("live operation source must not be a symlink")
    source = unresolved.resolve()
    root = _repository_root()
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise PC01LiveIntegrationError(
            f"live operation source is outside the repository: {source}"
        ) from exc
    if not source.is_file():
        raise PC01LiveIntegrationError("live operation source is missing")
    return relative, sha256_file(source)


def _operation_identity(callback: Callable[..., Any], *, operation_id: str) -> dict[str, str]:
    """Return the exact executable identity, including its non-substitutable qualname."""

    if isinstance(callback, functools.partial):
        raise PC01LiveIntegrationError(
            f"provider operation cannot be functools.partial: {operation_id}"
        )
    target: Any = callback
    if inspect.ismethod(target):
        target = target.__func__
    if not inspect.isfunction(target):
        raise PC01LiveIntegrationError(
            f"provider operation must be an exact function/method: {operation_id}"
        )
    relative, digest = _callable_source(target)
    return {
        "operation_id": operation_id,
        "module": str(target.__module__),
        "qualname": str(target.__qualname__),
        "source_relative_path": relative,
        "source_sha256": digest,
    }


def _guard_registered_delegate(callback: Callable[[], Any]) -> Any:
    """Detect provider mutation during the operation that caused it.

    The dispatchers are also useful while assembling/testing an unregistered
    provider, so the guard is conditional until registration.  Once installed,
    both successful and failing delegates are followed by an immediate registry
    fingerprint check; mutation takes precedence over a delegate error.
    """

    registry = globals().get("_OPERATIONS_REGISTRY")
    before = (
        registry.fingerprint_if_registered()
        if isinstance(registry, _ImmutableOperationsRegistry)
        else None
    )
    delegate_error: BaseException | None = None
    result: Any = None
    try:
        result = callback()
    except BaseException as exc:
        delegate_error = exc
    if before is not None:
        try:
            after = registry.require().contract_sha256
            if after != before:
                raise PC01LiveIntegrationError(
                    "registered PC-01 live operations changed during delegate call"
                )
        except BaseException as mutation:
            if delegate_error is not None:
                raise mutation from delegate_error
            raise
    if delegate_error is not None:
        raise delegate_error
    return result


@dataclass(frozen=True, slots=True)
class _EpisodeResetDispatch:
    delegate: EpisodeStateResetter

    def __call__(self, request: EpisodeStateResetRequest) -> EpisodeStateResetEvidence:
        """Delegate a real reset, then re-attest only its measured state digests."""

        evidence, measured_after = _guard_registered_delegate(
            lambda: (self.delegate.reset(request), self.delegate.digest_backend_state())
        )
        if dict(measured_after) != dict(evidence.backend_state_sha256):
            raise PC01LiveIntegrationError(
                "episode reset evidence differs from the immediate backend digest"
            )
        return evidence_from_backend_digests(
            request,
            resetter_id=self.delegate.resetter_id,
            resetter_version=self.delegate.resetter_version,
            resetter_source_sha256=integration_source_sha256(),
            backend_state_sha256=measured_after,
        )


@dataclass(frozen=True, slots=True)
class _EpisodeDigestDispatch:
    delegate: EpisodeStateResetter

    def __call__(self) -> Mapping[str, str]:
        return _guard_registered_delegate(self.delegate.digest_backend_state)


@dataclass(frozen=True, slots=True)
class _WebArenaResetDispatch:
    delegate: FrozenWebArenaResetStateAttester

    def __call__(self, request: WebArenaResetStateRequest) -> WebArenaResetStateReceipt:
        """Preserve measured commitments while rebinding the integration source."""

        receipt = _guard_registered_delegate(lambda: self.delegate.attest(request))
        return webarena_reset_receipt_from_commitments(
            request,
            attester_id=self.delegate.attester_id,
            attester_version=self.delegate.attester_version,
            attester_source_sha256=integration_source_sha256(),
            commitments_sha256=receipt.commitments_sha256,
        )


@dataclass(frozen=True, slots=True)
class _ActionSafetyDispatch:
    delegate: FrozenWebArenaActionSafetyPolicy

    def __call__(
        self,
        task: Any,
        observation: Any,
        action: Any,
    ) -> WebArenaActionSafetyDecision:
        return _guard_registered_delegate(
            lambda: self.delegate.evaluate(task, observation, action)
        )


@dataclass(frozen=True, slots=True)
class _ManualRescueDispatch:
    delegate: FrozenWebArenaManualRescueGuard

    def __call__(
        self,
        environment: Any,
        check: WebArenaManualRescueCheck,
    ) -> WebArenaManualRescueReceipt:
        receipt = _guard_registered_delegate(
            lambda: self.delegate.attest(environment, check)
        )
        return WebArenaManualRescueReceipt(
            guard_id=receipt.guard_id,
            guard_version=receipt.guard_version,
            source_sha256=integration_source_sha256(),
            evidence_mode=receipt.evidence_mode,
            check_sha256=receipt.check_sha256,
            check_index=receipt.check_index,
            stage=receipt.stage,
            expected_registered_browser_steps=(
                receipt.expected_registered_browser_steps
            ),
            observed_registered_browser_steps=(
                receipt.observed_registered_browser_steps
            ),
            automation_session_sha256=receipt.automation_session_sha256,
            controller_input_audit_sha256=(
                receipt.controller_input_audit_sha256
            ),
            exclusive_automation_control=receipt.exclusive_automation_control,
            non_agent_input_event_count=receipt.non_agent_input_event_count,
            manual_rescue_detected=receipt.manual_rescue_detected,
            oracle_labels_observed=False,
        )


@dataclass(frozen=True, slots=True)
class _RecoveryPlannerDispatch(RecoveryActionPlanner):
    delegate: RecoveryActionPlanner
    planner_id: str
    planner_version: str
    frozen: bool = True

    def plan(
        self,
        task: Any,
        post_failure_observation: Any,
        decision: Any,
        failed_action: Any,
        *,
        rng: Any,
    ) -> Any:
        return _guard_registered_delegate(
            lambda: self.delegate.plan(
                task,
                post_failure_observation,
                decision,
                failed_action,
                rng=rng,
            )
        )


@dataclass(frozen=True, slots=True)
class _MeasurementDispatch:
    delegate: Callable[..., Any]
    operation_id: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _guard_registered_delegate(lambda: self.delegate(*args, **kwargs))


def bind_episode_state_resetter(
    delegate: EpisodeStateResetter,
) -> CallableEpisodeStateResetter:
    """Bind a measured external reset capability to this attested integration."""

    if not isinstance(delegate, EpisodeStateResetter) or delegate.frozen is not True:
        raise PC01LiveIntegrationError(
            "episode reset delegate must be a frozen typed capability"
        )
    return CallableEpisodeStateResetter(
        resetter_id=delegate.resetter_id,
        resetter_version=delegate.resetter_version,
        source_sha256=integration_source_sha256(),
        callback=_EpisodeResetDispatch(delegate),
        digest_callback=_EpisodeDigestDispatch(delegate),
        frozen=True,
    )


def bind_webarena_reset_state_attester(
    delegate: FrozenWebArenaResetStateAttester,
) -> FrozenWebArenaResetStateAttester:
    """Bind real service/account/database commitments to the runtime source."""

    if (
        type(delegate) is not FrozenWebArenaResetStateAttester
        or delegate.frozen is not True
        or delegate.oracle_labels_exposed is not False
    ):
        raise PC01LiveIntegrationError(
            "WebArena reset delegate must be frozen and oracle-blind"
        )
    return FrozenWebArenaResetStateAttester(
        attester_id=delegate.attester_id,
        attester_version=delegate.attester_version,
        source_sha256=integration_source_sha256(),
        callback=_WebArenaResetDispatch(delegate),
        frozen=True,
        oracle_labels_exposed=False,
    )


def bind_action_safety_policy(
    delegate: FrozenWebArenaActionSafetyPolicy,
) -> FrozenWebArenaActionSafetyPolicy:
    """Bind a real, oracle-blind denial policy without changing its decisions."""

    if (
        type(delegate) is not FrozenWebArenaActionSafetyPolicy
        or delegate.frozen is not True
        or delegate.oracle_labels_exposed_to_runtime is not False
    ):
        raise PC01LiveIntegrationError(
            "action-safety delegate must be frozen and oracle-blind"
        )
    return FrozenWebArenaActionSafetyPolicy(
        safety_policy_id=delegate.safety_policy_id,
        safety_policy_version=delegate.safety_policy_version,
        destructive_action_policy=delegate.destructive_action_policy,
        benchmark_version=delegate.benchmark_version,
        source_sha256=integration_source_sha256(),
        callback=_ActionSafetyDispatch(delegate),
        frozen=True,
        oracle_labels_exposed_to_runtime=False,
    )


def bind_manual_rescue_guard(
    delegate: FrozenWebArenaManualRescueGuard,
) -> FrozenWebArenaManualRescueGuard:
    """Bind real exclusive-input measurements without inventing a clean receipt."""

    if (
        type(delegate) is not FrozenWebArenaManualRescueGuard
        or delegate.frozen is not True
        or delegate.oracle_labels_exposed_to_runtime is not False
    ):
        raise PC01LiveIntegrationError(
            "manual-rescue delegate must be frozen and oracle-blind"
        )
    return FrozenWebArenaManualRescueGuard(
        guard_id=delegate.guard_id,
        guard_version=delegate.guard_version,
        benchmark_version=delegate.benchmark_version,
        evidence_mode=delegate.evidence_mode,
        source_sha256=integration_source_sha256(),
        callback=_ManualRescueDispatch(delegate),
        frozen=True,
        oracle_labels_exposed_to_runtime=False,
    )


def _dispatch_delegate(value: object, expected: type, *, field: str) -> Any:
    if type(value) is not expected:
        raise PC01LiveIntegrationError(
            f"{field} must use the source-bound PC-01 integration dispatcher"
        )
    delegate = getattr(value, "delegate", None)
    if delegate is None:
        raise PC01LiveIntegrationError(f"{field} has no external delegate")
    return delegate


@dataclass(frozen=True, slots=True)
class PC01LiveOperationsProvider:
    """All deployment-owned capabilities required before model construction.

    The record contains no evaluator, verifier, reward, relevance, or reference
    data.  Its expected identities are produced by the already-frozen handoff.
    Registration is process-local and immutable; it is not serialized into raw
    artifacts and never contains credential bytes.  Construction exercises the
    same-process engineering fixture only and cannot authorize live dispatch.
    """

    provider_id: str
    provider_version: str
    expected_runtime_identity_sha256: str
    efficiency_measurement_id: str
    efficiency_measurement_version: str
    credential_capability: PC01ExternalCredentialCapability
    browser_runtime_factory: BrowserGymWebArenaRuntimeFactory
    task_state_resetter: FrozenBrowserGymTaskStateResetter
    episode_state_resetter: CallableEpisodeStateResetter
    recovery_action_planner: RecoveryActionPlanner
    begin_measurement: Callable[..., Any]
    finish_measurement: Callable[..., Mapping[str, Any]]
    parameter_fallback_contract: str
    parameter_fallback_source_sha256: str
    create_process_isolated_webarena: (
        CredentialFreeProcessIsolatedWebArenaEpisodeFactory | None
    ) = None
    model_seed: int = PC01_LIVE_MODEL_SEED
    frozen: bool = True
    oracle_labels_exposed_to_runtime: bool = False
    reward_slots_exposed_to_runtime: bool = False
    sealed_evaluator_capability_present: bool = False

    def __post_init__(self) -> None:
        for name in (
            "provider_id",
            "provider_version",
            "efficiency_measurement_id",
            "efficiency_measurement_version",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise PC01LiveIntegrationError(f"live provider requires {name}")
        _require_sha256(
            self.expected_runtime_identity_sha256,
            field="expected runtime identity",
        )
        if self.model_seed != PC01_LIVE_MODEL_SEED or type(self.model_seed) is not int:
            raise PC01LiveIntegrationError("live provider must bind seed 42 exactly")
        if (
            self.frozen is not True
            or self.oracle_labels_exposed_to_runtime is not False
            or self.reward_slots_exposed_to_runtime is not False
            or self.sealed_evaluator_capability_present is not False
        ):
            raise PC01LiveIntegrationError(
                "live provider must be immutable and contain no oracle/reward/evaluator capability"
            )
        if type(self.credential_capability) is not PC01ExternalCredentialCapability:
            raise PC01LiveIntegrationError(
                "live provider requires the exact external credential capability"
            )
        if type(self.browser_runtime_factory) is not BrowserGymWebArenaRuntimeFactory:
            raise PC01LiveIntegrationError(
                "live provider requires the pinned BrowserGym WebArena factory"
            )
        if type(self.task_state_resetter) is not FrozenBrowserGymTaskStateResetter:
            raise PC01LiveIntegrationError(
                "live provider requires the typed task-state reset capability"
            )
        if self.browser_runtime_factory.credential_bundle_root != self.credential_capability.root:
            raise PC01LiveIntegrationError(
                "browser runtime uses a different credential capability"
            )
        if self.browser_runtime_factory.task_state_resetter is not self.task_state_resetter:
            raise PC01LiveIntegrationError(
                "browser runtime uses a different task-state reset capability"
            )
        assert_process_wide_broker_assembly()
        if (
            self.browser_runtime_factory.runtime_page_publisher
            is not process_runtime_page_publisher()
        ):
            raise PC01LiveIntegrationError(
                "browser runtime is disconnected from the process-wide sealed "
                "evaluator broker"
            )
        if type(self.episode_state_resetter) is not CallableEpisodeStateResetter:
            raise PC01LiveIntegrationError(
                "live provider requires a source-bound episode-state resetter"
            )
        episode_delegate = _dispatch_delegate(
            self.episode_state_resetter.callback,
            _EpisodeResetDispatch,
            field="episode-state reset callback",
        )
        digest_delegate = _dispatch_delegate(
            self.episode_state_resetter.digest_callback,
            _EpisodeDigestDispatch,
            field="episode-state digest callback",
        )
        if episode_delegate is not digest_delegate:
            raise PC01LiveIntegrationError(
                "episode reset and digest callbacks use different capabilities"
            )
        episode_reset_sources = {
            _callable_source(
                episode_delegate.callback
                if hasattr(episode_delegate, "callback")
                else episode_delegate.reset
            )[1],
            _callable_source(
                episode_delegate.digest_callback
                if hasattr(episode_delegate, "digest_callback")
                else episode_delegate.digest_backend_state
            )[1],
        }
        if episode_reset_sources != {episode_delegate.source_sha256}:
            raise PC01LiveIntegrationError(
                "episode reset delegate source declaration differs from its callbacks"
            )
        integration_sha256 = integration_source_sha256()
        if self.episode_state_resetter.source_sha256 != integration_sha256:
            raise PC01LiveIntegrationError(
                "episode-state resetter is not bound to this integration source"
            )
        factory = self.browser_runtime_factory
        reset_delegate = _dispatch_delegate(
            factory.reset_state_attester.callback,
            _WebArenaResetDispatch,
            field="WebArena reset-state callback",
        )
        safety_delegate = _dispatch_delegate(
            factory.action_safety_policy.callback,
            _ActionSafetyDispatch,
            field="action-safety callback",
        )
        manual_delegate = _dispatch_delegate(
            factory.manual_rescue_guard.callback,
            _ManualRescueDispatch,
            field="manual-rescue callback",
        )
        if not isinstance(reset_delegate, FrozenWebArenaResetStateAttester):
            raise PC01LiveIntegrationError("WebArena reset dispatcher has no typed delegate")
        if not isinstance(safety_delegate, FrozenWebArenaActionSafetyPolicy):
            raise PC01LiveIntegrationError("action-safety dispatcher has no typed delegate")
        if not isinstance(manual_delegate, FrozenWebArenaManualRescueGuard):
            raise PC01LiveIntegrationError("manual-rescue dispatcher has no typed delegate")
        for name, delegate in (
            ("WebArena reset-state", reset_delegate),
            ("action-safety", safety_delegate),
            ("manual-rescue", manual_delegate),
        ):
            if _callable_source(delegate.callback)[1] != delegate.source_sha256:
                raise PC01LiveIntegrationError(
                    f"{name} delegate source declaration differs from its callback"
                )
        for name, source_sha256 in (
            ("WebArena reset-state", factory.reset_state_attester.source_sha256),
            ("action-safety", factory.action_safety_policy.source_sha256),
            ("manual-rescue", factory.manual_rescue_guard.source_sha256),
        ):
            if source_sha256 != integration_sha256:
                raise PC01LiveIntegrationError(
                    f"{name} capability is not bound to this integration source"
                )
        if not isinstance(self.recovery_action_planner, RecoveryActionPlanner):
            raise PC01LiveIntegrationError(
                "live provider requires a typed recovery action planner"
            )
        if self.recovery_action_planner.frozen is not True:
            raise PC01LiveIntegrationError("recovery action planner must be frozen")
        if not callable(self.begin_measurement) or not callable(self.finish_measurement):
            raise PC01LiveIntegrationError(
                "live provider requires begin/finish efficiency measurement"
            )
        if self.parameter_fallback_contract != PC01_PARAMETER_FALLBACK_CONTRACT:
            raise PC01LiveIntegrationError(
                "parameter fallback must be the registered unadapted E0 contract"
            )
        qwen_source = Path(__file__).with_name("qwen2vl_pc01.py")
        actual_qwen_sha256 = sha256_file(qwen_source)
        if self.parameter_fallback_source_sha256 != actual_qwen_sha256:
            raise PC01LiveIntegrationError(
                "parameter fallback source differs from the PC-01 model bridge"
            )
        if self.create_process_isolated_webarena is not None and type(
            self.create_process_isolated_webarena
        ) is not CredentialFreeProcessIsolatedWebArenaEpisodeFactory:
            raise PC01LiveIntegrationError(
                "process-isolated episode factory must use the exact "
                "credential-free source-attested implementation"
            )
        # Resolve every external executable now.  This catches lambda/code
        # injection outside the attested repository without invoking it.
        self.external_operation_sources()
        self.public_contract()

    @property
    def credential_bundle_root(self) -> Path:
        return self.credential_capability.root

    def _operations_by_capability(
        self,
    ) -> Mapping[str, tuple[tuple[str, Callable[..., Any]], ...]]:
        factory = self.browser_runtime_factory
        episode_dispatch = self.episode_state_resetter.callback
        reset_dispatch = factory.reset_state_attester.callback
        safety_dispatch = factory.action_safety_policy.callback
        manual_dispatch = factory.manual_rescue_guard.callback
        recovery_callback = getattr(
            self.recovery_action_planner,
            "callback",
            self.recovery_action_planner.plan,
        )
        return {
            "deterministic_reset": (
                ("task_state_reset", self.task_state_resetter.callback),
                (
                    "episode_state_reset",
                    episode_dispatch.delegate.callback
                    if hasattr(episode_dispatch.delegate, "callback")
                    else episode_dispatch.delegate.reset,
                ),
                (
                    "episode_state_digest",
                    episode_dispatch.delegate.digest_callback
                    if hasattr(episode_dispatch.delegate, "digest_callback")
                    else episode_dispatch.delegate.digest_backend_state,
                ),
                ("webarena_reset_attestation", reset_dispatch.delegate.callback),
            ),
            "exclusive_input_audit": (
                ("manual_rescue_audit", manual_dispatch.delegate.callback),
            ),
            "oracle_blind_browser_mapping": (
                ("browser_runtime_factory", factory.__call__),
            ),
            "action_safety_fault_classification": (
                ("action_safety", safety_dispatch.delegate.callback),
                (
                    "infrastructure_fault_classification",
                    factory.infrastructure_fault_classifier.callback,
                ),
            ),
            "recovery_action_planner": (("recovery_plan", recovery_callback),),
            "efficiency_measurement": (
                ("begin_measurement", self.begin_measurement),
                ("finish_measurement", self.finish_measurement),
            ),
        }

    def operation_sources_by_capability(
        self,
    ) -> Mapping[str, tuple[tuple[str, str], ...]]:
        """Bind each measured runtime capability to its executable source."""

        return MappingProxyType(
            {
                capability_id: tuple(
                    sorted(
                        {
                            _callable_source(callback)
                            for _, callback in operations
                        }
                    )
                )
                for capability_id, operations in self._operations_by_capability().items()
            }
        )

    def external_operation_sources(self) -> tuple[tuple[str, str], ...]:
        """Return the deterministic, deduplicated sources of real operations."""

        factory = self.browser_runtime_factory
        episode_dispatch = self.episode_state_resetter.callback
        reset_dispatch = factory.reset_state_attester.callback
        safety_dispatch = factory.action_safety_policy.callback
        manual_dispatch = factory.manual_rescue_guard.callback
        recovery_callback = getattr(
            self.recovery_action_planner,
            "callback",
            self.recovery_action_planner.plan,
        )
        callbacks: tuple[Callable[..., Any], ...] = (
            factory,
            self.task_state_resetter.callback,
            episode_dispatch.delegate.callback
            if hasattr(episode_dispatch.delegate, "callback")
            else episode_dispatch.delegate.reset,
            episode_dispatch.delegate.digest_callback
            if hasattr(episode_dispatch.delegate, "digest_callback")
            else episode_dispatch.delegate.digest_backend_state,
            reset_dispatch.delegate.callback,
            safety_dispatch.delegate.callback,
            manual_dispatch.delegate.callback,
            factory.infrastructure_fault_classifier.callback,
            recovery_callback,
            self.begin_measurement,
            self.finish_measurement,
        )
        return tuple(sorted({_callable_source(callback) for callback in callbacks}))

    def _capability_identity_and_state(
        self,
        capability_id: str,
    ) -> tuple[str, str, dict[str, Any]]:
        """Derive one public capability contract from the objects actually used.

        In particular, none of these values are accepted from a deployment
        identity mapping supplied beside the provider.  The live-deployment
        evidence is an authority to compare against, not a way for the
        provider to describe itself.
        """

        factory = self.browser_runtime_factory
        if capability_id == "deterministic_reset":
            return (
                self.episode_state_resetter.resetter_id,
                self.episode_state_resetter.resetter_version,
                {
                    "episode_resetter": {
                        "id": self.episode_state_resetter.resetter_id,
                        "version": self.episode_state_resetter.resetter_version,
                        "bound_source_sha256": self.episode_state_resetter.source_sha256,
                    },
                    "task_resetter": {
                        "id": self.task_state_resetter.resetter_id,
                        "version": self.task_state_resetter.resetter_version,
                        "declared_source_sha256": self.task_state_resetter.source_sha256,
                        "service_url_map_sha256": (
                            self.task_state_resetter.service_url_map_sha256
                        ),
                    },
                    "webarena_reset_attester": {
                        "id": factory.reset_state_attester.attester_id,
                        "version": factory.reset_state_attester.attester_version,
                        "bound_source_sha256": (
                            factory.reset_state_attester.source_sha256
                        ),
                    },
                },
            )
        if capability_id == "exclusive_input_audit":
            guard = factory.manual_rescue_guard
            return (
                guard.guard_id,
                guard.guard_version,
                {
                    "benchmark_version": guard.benchmark_version,
                    "evidence_mode": guard.evidence_mode,
                    "bound_source_sha256": guard.source_sha256,
                    "oracle_labels_exposed_to_runtime": False,
                },
            )
        if capability_id == "oracle_blind_browser_mapping":
            return (
                factory.environment_adapter_id,
                factory.environment_adapter_version,
                {
                    "benchmark_version": factory.benchmark_version,
                    "wrapper_source_sha256": factory.wrapper_source_sha256,
                    "configuration": factory.configuration.to_dict(),
                    "deployment_preflight_binding_sha256": sha256_json(
                        dict(factory.validated_preflight.binding)
                    ),
                    "service_url_map_sha256": sha256_json(
                        dict(factory.validated_preflight.service_url_map)
                    ),
                    "environment_state_digester": {
                        "id": factory.environment_state_digester_id,
                        "version": factory.environment_state_digester_version,
                    },
                    "page_settle_policy_id": factory.page_settle_policy_id,
                },
            )
        if capability_id == "action_safety_fault_classification":
            safety = factory.action_safety_policy
            classifier = factory.infrastructure_fault_classifier
            return (
                safety.safety_policy_id,
                safety.safety_policy_version,
                {
                    "benchmark_version": safety.benchmark_version,
                    "destructive_action_policy": safety.destructive_action_policy,
                    "bound_source_sha256": safety.source_sha256,
                    "infrastructure_classifier": {
                        "id": classifier.classifier_id,
                        "version": classifier.classifier_version,
                        "benchmark_version": classifier.benchmark_version,
                        "rules_sha256": classifier.rules_sha256,
                    },
                    "oracle_labels_exposed_to_runtime": False,
                },
            )
        if capability_id == "recovery_action_planner":
            planner = self.recovery_action_planner
            return (
                planner.planner_id,
                planner.planner_version,
                {"frozen": planner.frozen, "oracle_inputs_observed": False},
            )
        if capability_id == "efficiency_measurement":
            return (
                self.efficiency_measurement_id,
                self.efficiency_measurement_version,
                {
                    "registered_measurement_pair": True,
                    "oracle_inputs_observed": False,
                },
            )
        raise PC01LiveIntegrationError(
            f"unregistered provider capability: {capability_id}"
        )

    def capability_public_contracts(self) -> dict[str, dict[str, Any]]:
        """Return six operation/config-derived public capability contracts."""

        operations_by_capability = self._operations_by_capability()
        if set(operations_by_capability) != set(RUNTIME_LIVE_CAPABILITY_IDS):
            raise PC01LiveIntegrationError(
                "provider operation map does not cover six runtime capabilities"
            )
        result: dict[str, dict[str, Any]] = {}
        for capability_id in RUNTIME_LIVE_CAPABILITY_IDS:
            implementation_id, implementation_version, public_state = (
                self._capability_identity_and_state(capability_id)
            )
            operations = [
                _operation_identity(callback, operation_id=operation_id)
                for operation_id, callback in operations_by_capability[capability_id]
            ]
            operation_sources = {
                (item["source_relative_path"], item["source_sha256"])
                for item in operations
            }
            if len(operation_sources) != 1:
                raise PC01LiveIntegrationError(
                    "one capability's operations span multiple source authorities: "
                    f"{capability_id}"
                )
            source_relative_path, source_sha256 = next(iter(operation_sources))
            result[capability_id] = {
                "capability_id": capability_id,
                "implementation_id": implementation_id,
                "implementation_version": implementation_version,
                "source_relative_path": source_relative_path,
                "source_sha256": source_sha256,
                "operations": operations,
                "operations_sha256": sha256_json(operations),
                "public_state": public_state,
                "public_state_sha256": sha256_json(public_state),
            }
        return result

    def public_contract(self) -> dict[str, Any]:
        """Derive the exact provider identity without credential contents."""

        capabilities = self.capability_public_contracts()
        ordered = [capabilities[item] for item in RUNTIME_LIVE_CAPABILITY_IDS]
        return {
            "schema_version": PC01_PROVIDER_PUBLIC_CONTRACT_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "expected_runtime_identity_sha256": (
                self.expected_runtime_identity_sha256
            ),
            "model_seed": self.model_seed,
            "credential_capability": dict(self.credential_capability.public_identity),
            "parameter_fallback_contract": self.parameter_fallback_contract,
            "parameter_fallback_source_sha256": (
                self.parameter_fallback_source_sha256
            ),
            "capabilities": capabilities,
            "capability_set_sha256": sha256_json(ordered),
            "oracle_labels_exposed_to_runtime": False,
            "reward_slots_exposed_to_runtime": False,
            "sealed_evaluator_capability_present": False,
        }

    @property
    def public_contract_sha256(self) -> str:
        return sha256_json(self.public_contract())

    @property
    def contract_sha256(self) -> str:
        """Registry fingerprint; identical to the frozen public contract hash."""

        return self.public_contract_sha256


class _ImmutableOperationsRegistry:
    """One-install registry; reads revalidate the provider fingerprint."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._provider: PC01LiveOperationsProvider | None = None
        self._fingerprint: str | None = None
        self._installation_receipt: PC01ProviderInstallationReceipt | None = None
        self._installation_receipt_sha256: str | None = None

    def register(
        self,
        provider: PC01LiveOperationsProvider,
        installation_receipt: PC01ProviderInstallationReceipt | None,
    ) -> str:
        if type(provider) is not PC01LiveOperationsProvider:
            raise PC01LiveIntegrationError(
                "live operations registry accepts only PC01LiveOperationsProvider"
            )
        if type(installation_receipt) is not PC01ProviderInstallationReceipt:
            raise PC01LiveIntegrationError(
                "live operations registry requires an exact post-installation receipt"
            )
        fingerprint = provider.contract_sha256
        try:
            installation_receipt.validate_provider_contract(fingerprint)
        except Exception as exc:
            raise PC01LiveIntegrationError(
                "provider installation receipt differs from the actual provider"
            ) from exc
        receipt_sha256 = installation_receipt.receipt_sha256
        with self._lock:
            if self._provider is not None:
                raise PC01LiveIntegrationError(
                    "PC-01 live operations are already registered for this process"
                )
            self._provider = provider
            self._fingerprint = fingerprint
            self._installation_receipt = installation_receipt
            self._installation_receipt_sha256 = receipt_sha256
        return fingerprint

    def require(self) -> PC01LiveOperationsProvider:
        with self._lock:
            provider = self._provider
            fingerprint = self._fingerprint
            installation_receipt = self._installation_receipt
            installation_receipt_sha256 = self._installation_receipt_sha256
        if (
            provider is None
            or fingerprint is None
            or installation_receipt is None
            or installation_receipt_sha256 is None
        ):
            raise PC01LiveIntegrationError(
                "PC-01 live operations provider is not registered; no live default exists"
            )
        if provider.contract_sha256 != fingerprint:
            raise PC01LiveIntegrationError(
                "registered PC-01 live operations changed after installation"
            )
        try:
            installation_receipt.validate_provider_contract(fingerprint)
        except Exception as exc:
            raise PC01LiveIntegrationError(
                "registered provider installation receipt no longer validates"
            ) from exc
        if installation_receipt.receipt_sha256 != installation_receipt_sha256:
            raise PC01LiveIntegrationError(
                "registered provider installation receipt changed after installation"
            )
        return provider

    def require_installation_receipt(self) -> PC01ProviderInstallationReceipt:
        """Return the receipt only after revalidating provider and receipt together."""

        self.require()
        with self._lock:
            receipt = self._installation_receipt
        if receipt is None:  # pragma: no cover - guarded by require()
            raise PC01LiveIntegrationError("provider installation receipt is absent")
        return receipt

    def fingerprint_if_registered(self) -> str | None:
        """Return a revalidated snapshot without requiring prior installation."""

        with self._lock:
            installed = self._provider is not None
        if not installed:
            return None
        return self.require().contract_sha256


_OPERATIONS_REGISTRY = _ImmutableOperationsRegistry()


def register_pc01_live_operations(
    provider: PC01LiveOperationsProvider,
    installation_receipt: PC01ProviderInstallationReceipt | None = None,
) -> str:
    """Install exactly one frozen provider before constructing the runner."""

    return _OPERATIONS_REGISTRY.register(provider, installation_receipt)


def validate_provider_public_contract(
    provider: PC01LiveOperationsProvider,
    authority_value: Mapping[str, Any],
    *,
    expected_preflight_view: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Compare actual operation/config identities with the frozen authority."""

    if type(provider) is not PC01LiveOperationsProvider:
        raise PC01LiveIntegrationError("provider public-contract type is invalid")
    try:
        authority = validate_runtime_capability_authority(authority_value)
    except Exception as exc:
        raise PC01LiveIntegrationError(
            "frozen runtime capability authority is invalid"
        ) from exc
    preflight_binding = dict(provider.browser_runtime_factory.validated_preflight.binding)
    if expected_preflight_view is not None and preflight_binding != dict(
        expected_preflight_view
    ):
        raise PC01LiveIntegrationError(
            "browser runtime preflight view differs from provider bootstrap"
        )
    if sha256_json(preflight_binding) != authority[
        "deployment_preflight_binding_sha256"
    ]:
        raise PC01LiveIntegrationError(
            "browser runtime deployment preflight differs from frozen authority"
        )
    provider_capabilities = provider.capability_public_contracts()
    expected_capabilities = authority["capabilities"]
    for capability_id in RUNTIME_LIVE_CAPABILITY_IDS:
        actual = provider_capabilities[capability_id]
        expected = expected_capabilities[capability_id]
        comparisons = {
            "implementation_id": actual["implementation_id"],
            "implementation_version": actual["implementation_version"],
            "source_relative_path": actual["source_relative_path"],
            "source_sha256": actual["source_sha256"],
            "deployment_state_sha256": actual["public_state_sha256"],
        }
        for field, value in comparisons.items():
            if expected.get(field) != value:
                raise PC01LiveIntegrationError(
                    "live provider capability public contract differs from frozen "
                    f"authority: {capability_id}.{field}"
                )
    if (
        provider.public_contract_sha256
        != authority["expected_provider_public_contract_sha256"]
    ):
        raise PC01LiveIntegrationError(
            "live provider public-contract hash differs from frozen authority"
        )
    source_rows: dict[str, str] = {}
    for capability_id in RUNTIME_LIVE_CAPABILITY_IDS:
        row = expected_capabilities[capability_id]
        relative = str(row["source_relative_path"])
        digest = str(row["source_sha256"])
        previous = source_rows.get(relative)
        if previous is not None and previous != digest:
            raise PC01LiveIntegrationError(
                "runtime capabilities repeat one source path with different hashes"
            )
        source_rows[relative] = digest
    for relative, digest in provider.external_operation_sources():
        if source_rows.get(relative) != digest:
            raise PC01LiveIntegrationError(
                f"live operation source is absent or changed in authority: {relative}"
            )
    return source_rows


def _validate_context(
    context: FrozenRuntimeContext,
    provider: PC01LiveOperationsProvider,
) -> dict[str, str]:
    if type(context) is not FrozenRuntimeContext:
        raise PC01LiveIntegrationError(
            "PC-01 live entrypoint requires exact FrozenRuntimeContext"
        )
    runtime_identity = context.runtime_identity
    if sha256_json(dict(runtime_identity)) != provider.expected_runtime_identity_sha256:
        raise PC01LiveIntegrationError(
            "runtime identity differs from the registered live provider"
        )
    source_rows = validate_provider_public_contract(
        provider,
        context.runtime_capability_authority,
    )
    integration = runtime_identity.get("runtime_integration")
    expected_integration = {
        "entrypoint": PC01_LIVE_INTEGRATION_ENTRYPOINT,
        "source_relative_path": PC01_LIVE_INTEGRATION_SOURCE,
        "source_sha256": integration_source_sha256(),
    }
    if not isinstance(integration, Mapping) or dict(integration) != expected_integration:
        raise PC01LiveIntegrationError(
            "runtime integration entrypoint/source identity is not this module"
        )
    if runtime_identity.get("checkpoint_systems") != list(PC01_CHECKPOINT_SYSTEMS):
        raise PC01LiveIntegrationError(
            "PC-01 checkpoint must be bound to E1, E2, and E3 exactly"
        )
    checkpoint_by_seed = runtime_identity.get("selected_checkpoint_by_seed")
    backbone_by_seed = runtime_identity.get("e0_unadapted_backbone_by_seed")
    if (
        not isinstance(checkpoint_by_seed, Mapping)
        or set(checkpoint_by_seed) != {"42"}
        or not isinstance(backbone_by_seed, Mapping)
        or set(backbone_by_seed) != {"42"}
    ):
        raise PC01LiveIntegrationError(
            "PC-01 runtime identity must contain exactly seed 42 checkpoint/base"
        )
    expected_seeds = {PC01_LIVE_MODEL_SEED}
    if (
        set(context.model_manifest_paths) != expected_seeds
        or set(context.model_payload_paths) != expected_seeds
        or set(context.model_evidence_paths) != expected_seeds
    ):
        raise PC01LiveIntegrationError(
            "model manifests, payloads, and evidence must cover seed 42 exactly"
        )
    manifest = read_json(context.model_manifest_paths[PC01_LIVE_MODEL_SEED])
    selected_identity = checkpoint_by_seed["42"]
    base_identity = backbone_by_seed["42"]
    if not isinstance(selected_identity, Mapping) or not isinstance(
        base_identity, Mapping
    ):
        raise PC01LiveIntegrationError("PC-01 model runtime identity is malformed")
    selected_fields = {
        "selected_checkpoint_sha256": "selected_checkpoint_sha256",
        "resolved_config_sha256": "resolved_config_sha256",
        "resolved_config_record_sha256": "resolved_config_record_sha256",
        "processor_contract_sha256": "processor_contract_sha256",
        "model_evidence_bundle_sha256": "model_evidence_bundle_sha256",
    }
    base_fields = {
        "backbone_id": "e0_backbone_id",
        "backbone_revision": "e0_backbone_revision",
        "backbone_sha256": "e0_backbone_sha256",
        "resolved_config_sha256": "e0_resolved_config_sha256",
        "processor_contract_sha256": "e0_processor_contract_sha256",
        "base_prompt_sha256": "e0_base_prompt_sha256",
        "parser_id": "e0_parser_id",
        "parser_version": "e0_parser_version",
        "parser_sha256": "e0_parser_sha256",
    }
    for identity_field, manifest_field in selected_fields.items():
        if selected_identity.get(identity_field) != manifest.get(manifest_field):
            raise PC01LiveIntegrationError(
                f"selected checkpoint manifest differs at {manifest_field}"
            )
    for identity_field, manifest_field in base_fields.items():
        if base_identity.get(identity_field) != manifest.get(manifest_field):
            raise PC01LiveIntegrationError(
                f"unadapted E0 manifest differs at {manifest_field}"
            )
    registered_pc01 = {
        "selected_checkpoint_sha256": PC01_EXPECTED_CHECKPOINT_SHA256,
        "resolved_config_record_sha256": PC01_EXPECTED_CONFIG_SHA256,
        "e0_backbone_id": "Qwen/Qwen2-VL-2B-Instruct",
        "e0_backbone_revision": PC01_MODEL_REVISION,
        "e0_backbone_sha256": PC01_EXPECTED_BASE_SNAPSHOT_SHA256,
    }
    for field, expected in registered_pc01.items():
        if manifest.get(field) != expected:
            raise PC01LiveIntegrationError(
                f"model manifest is not registered PC-01 epoch 6 seed 42: {field}"
            )
    return source_rows


@dataclass(frozen=True, slots=True)
class _OracleBlindWebArenaRuntimeFactory:
    delegate: BrowserGymWebArenaRuntimeFactory
    allowed_source_rows: Mapping[str, str]
    integration_sha256: str

    def __post_init__(self) -> None:
        rows = dict(self.allowed_source_rows)
        if not rows:
            raise PC01LiveIntegrationError(
                "browser runtime factory requires frozen capability sources"
            )
        object.__setattr__(self, "allowed_source_rows", MappingProxyType(rows))

    def __call__(self, task: TaskSpecification) -> WebArenaRuntimeBinding:
        provider = _OPERATIONS_REGISTRY.require()
        if provider.browser_runtime_factory is not self.delegate:
            raise PC01LiveIntegrationError(
                "registered browser operations changed after integration construction"
            )
        if type(task) is not TaskSpecification:
            raise PC01LiveIntegrationError(
                "browser runtime accepts only the projected TaskSpecification"
            )
        if task.development_partition is not True:
            raise PC01LiveIntegrationError(
                "PC-01 pilot browser runtime accepts development tasks only"
            )
        if task.destructive_actions_allowed is not False:
            raise PC01LiveIntegrationError(
                "PC-01 pilot browser runtime forbids destructive tasks"
            )
        if task.runtime_start_state is None:
            raise PC01LiveIntegrationError(
                "browser task lacks its public six-field start state"
            )
        assert_oracle_blind_mapping(
            task.metadata,
            location="pc01_live_integration.task_metadata",
        )
        result = _guard_registered_delegate(lambda: self.delegate(task))
        if type(result) is not WebArenaRuntimeBinding:
            raise PC01LiveIntegrationError(
                "browser operations returned the wrong runtime binding"
            )
        if result.frozen is not True:
            raise PC01LiveIntegrationError("browser runtime is not frozen")
        for name, value in (
            ("reset-state", result.reset_state_attester.source_sha256),
            ("action-safety", result.action_safety_policy.source_sha256),
            ("manual-rescue", result.manual_rescue_guard.source_sha256),
        ):
            if value != self.integration_sha256:
                raise PC01LiveIntegrationError(
                    f"browser {name} capability changed source identity"
                )
        callbacks = (
            result.environment_factory,
            result.observation_mapper,
            result.action_mapper,
            result.screenshot_bytes_provider,
            result.environment_state_digester.callback,
            result.infrastructure_fault_classifier.callback,
            result.reset_state_attester.callback,
            result.page_settle_policy.callback,
            result.action_safety_policy.callback,
            result.manual_rescue_guard.callback,
            result.abort_episode,
        )
        integration_row = (PC01_LIVE_INTEGRATION_SOURCE, self.integration_sha256)
        for callback in callbacks:
            source = _callable_source(callback)
            if source != integration_row and self.allowed_source_rows.get(
                source[0]
            ) != source[1]:
                raise PC01LiveIntegrationError(
                    f"browser callback source is not frozen: {source[0]}"
                )
        return result


def _validate_runtime_binding(
    result: EvaluationRuntimeBinding,
    context: FrozenRuntimeContext,
    provider: PC01LiveOperationsProvider,
) -> None:
    if type(result) is not EvaluationRuntimeBinding:
        raise PC01LiveIntegrationError(
            "PC-01 model factory returned the wrong evaluation binding"
        )
    if dict(result.runtime_identity) != dict(context.runtime_identity):
        raise PC01LiveIntegrationError(
            "PC-01 evaluation binding changed the frozen runtime identity"
        )
    if (
        result.frozen is not True
        or result.oracle_labels_exposed_to_runtime is not False
        or set(result.seed_bindings) != {PC01_LIVE_MODEL_SEED}
    ):
        raise PC01LiveIntegrationError(
            "PC-01 evaluation binding is mutable, oracle-bearing, or seed-mismatched"
        )
    if (
        result.create_process_isolated_webarena
        is not provider.create_process_isolated_webarena
    ):
        raise PC01LiveIntegrationError(
            "PC-01 evaluation binding replaced the process-isolated episode factory"
        )
    binding = result.seed_bindings[PC01_LIVE_MODEL_SEED]
    if type(binding) is not SeedRuntimeBinding:
        raise PC01LiveIntegrationError("PC-01 seed binding has the wrong type")
    selected_identity = context.runtime_identity["selected_checkpoint_by_seed"]["42"]
    base_identity = context.runtime_identity["e0_unadapted_backbone_by_seed"]["42"]
    selected = binding.selected_checkpoint_backend
    base = binding.selected_backbone_backend
    if (
        selected.checkpoint_sha256
        != selected_identity["selected_checkpoint_sha256"]
        or selected.resolved_config_sha256
        != selected_identity["resolved_config_sha256"]
        or selected.frozen is not True
        or selected.training is not False
    ):
        raise PC01LiveIntegrationError(
            "E1-E3 are not bound to the frozen selected PC-01 checkpoint"
        )
    if (
        base.backbone_sha256 != base_identity["backbone_sha256"]
        or base.resolved_config_sha256
        != base_identity["resolved_config_sha256"]
        or base.frozen is not True
        or base.training is not False
        or base.adaptation_loaded is not False
        or base.task_heads_loaded is not False
    ):
        raise PC01LiveIntegrationError(
            "E0 is not the frozen unadapted base without trained task heads"
        )
    if binding.parameter_fallback_backbone_sha256 != base.backbone_sha256:
        raise PC01LiveIntegrationError(
            "shared parameter fallback is not bound to the E0 backbone"
        )
    fallback_source = _callable_source(binding.parameter_fallback_resolver)
    qwen_relative = "src/web_agent/runtime/qwen2vl_pc01.py"
    if fallback_source != (
        qwen_relative,
        provider.parameter_fallback_source_sha256,
    ):
        raise PC01LiveIntegrationError(
            "shared parameter fallback is not the source-attested PC-01 E0 resolver"
        )
    if binding.episode_state_resetter is not provider.episode_state_resetter:
        raise PC01LiveIntegrationError(
            "seed binding replaced the registered episode reset capability"
        )
    if (
        type(binding.recovery_action_planner) is not _RecoveryPlannerDispatch
        or binding.recovery_action_planner.delegate
        is not provider.recovery_action_planner
    ):
        raise PC01LiveIntegrationError(
            "seed binding replaced the registered recovery planner"
        )
    if (
        type(binding.begin_measurement) is not _MeasurementDispatch
        or binding.begin_measurement.delegate is not provider.begin_measurement
        or type(binding.finish_measurement) is not _MeasurementDispatch
        or binding.finish_measurement.delegate is not provider.finish_measurement
    ):
        raise PC01LiveIntegrationError(
            "seed binding replaced the registered efficiency measurement"
        )


def create_pc01_live_runtime(context: FrozenRuntimeContext) -> EvaluationRuntimeBinding:
    """Create the real PC-01 binding, or fail before any browser/model action.

    This exact ``(FrozenRuntimeContext) -> EvaluationRuntimeBinding`` function is
    the source-attested campaign entrypoint.  The provider receives neither the
    context nor any evaluator capability.
    """

    provider = _OPERATIONS_REGISTRY.require()
    source_rows = _validate_context(context, provider)
    browser_factory = _OracleBlindWebArenaRuntimeFactory(
        delegate=provider.browser_runtime_factory,
        allowed_source_rows=dict(source_rows),
        integration_sha256=integration_source_sha256(),
    )
    recovery_planner = _RecoveryPlannerDispatch(
        delegate=provider.recovery_action_planner,
        planner_id=provider.recovery_action_planner.planner_id,
        planner_version=provider.recovery_action_planner.planner_version,
    )
    begin_measurement = _MeasurementDispatch(
        delegate=provider.begin_measurement,
        operation_id="begin_measurement",
    )
    finish_measurement = _MeasurementDispatch(
        delegate=provider.finish_measurement,
        operation_id="finish_measurement",
    )
    services = PC01SeedOperationalServices(
        episode_state_resetter=provider.episode_state_resetter,
        recovery_action_planner=recovery_planner,
        begin_measurement=begin_measurement,
        finish_measurement=finish_measurement,
    )
    factory = PC01EvaluationRuntimeFactory(
        create_webarena_runtime=browser_factory,
        seed_services={PC01_LIVE_MODEL_SEED: services},
        create_process_isolated_webarena=(
            provider.create_process_isolated_webarena
        ),
    )
    result = factory(context)
    _validate_runtime_binding(result, context, provider)
    return result
