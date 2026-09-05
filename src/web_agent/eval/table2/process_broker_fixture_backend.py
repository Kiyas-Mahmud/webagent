"""Deterministic engineering-only backend for isolated-broker tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from web_agent.benchmarks.base import AdapterExecution, EnvironmentAdapter
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    ControllerCommandCommitment,
    ExecutionEvidence,
    ExecutionResult,
    ExecutionSafetyReceipt,
    ExecutionStatus,
    EpisodeSummary,
    Observation,
    ObservationStage,
    OpaqueTerminalSignal,
    TaskSpecification,
    TerminalReason,
    VerifierReceiptBinding,
)
from web_agent.runtime.state_reset import (
    REGISTERED_WEBARENA_RESET_COMMITMENTS,
    WebArenaResetStateReceipt,
    webarena_reset_state_digest,
)
from web_agent.eval.table2.common import canonical_json_bytes, sha256_json
from web_agent.eval.table2.process_broker_protocol import (
    ProcessBrokerInfrastructureInvalidMixin,
    registered_browser_error_observation_url,
)
from web_agent.eval.table2.sealed_verifier import SealedVerifierWriter


_FIXTURE_SHA256 = "a" * 64
_FIXTURE_TIMESTAMP = "2026-01-01T00:00:00+00:00"


class FixtureInfrastructureInvalidError(
    RuntimeError,
    ProcessBrokerInfrastructureInvalidMixin,
):
    def __init__(
        self,
        episode_id: str,
        *,
        action: ConcreteAction | None = None,
        evidence_action_id: str | None = None,
    ) -> None:
        self.reason_code = "BROWSER_CONTROLLER_DISCONNECTED"
        self.adapter_id = "fixture-process-webarena"
        self.adapter_version = "v1"
        self.operation = "step"
        self.adapter_evidence = {
            "adapter_event_id": f"{episode_id}:infrastructure:1",
            "failure_class": "fixture_controller_disconnected",
            "diagnostic_sha256": "c" * 64,
            "retryable": True,
            # Deliberate child-only extras prove the IPC sanitizer projects to
            # the four registered runtime fields instead of serializing an
            # exception object or arbitrary diagnostic context.
            "exception_type": "fixture.private.RawControllerException",
            "raw_exception_text": "must never cross the broker",
        }
        if action is not None:
            committed_action_id = evidence_action_id or action.action_id
            command_bytes = canonical_json_bytes(action.to_dict())
            self.adapter_execution = AdapterExecution(
                status=ExecutionStatus.ERROR,
                state_changed=False,
                environment_error=True,
                error_kind="controller_step_interruption",
                message="native controller step did not return",
                internal_retry_count=0,
                latency_ms=1.25,
                evidence=ExecutionEvidence(
                    action_id=committed_action_id,
                    started_at_utc=_FIXTURE_TIMESTAMP,
                    ended_at_utc=_FIXTURE_TIMESTAMP,
                    status=ExecutionStatus.ERROR,
                    controller_command=ControllerCommandCommitment(
                        action_id=committed_action_id,
                        adapter_id=self.adapter_id,
                        adapter_version=self.adapter_version,
                        command_type="fixture.NativeCommand",
                        command_sha256=hashlib.sha256(command_bytes).hexdigest(),
                        canonical_byte_length=len(command_bytes),
                    ),
                    safety_receipt=ExecutionSafetyReceipt(
                        safety_policy_id="fixture-action-safety",
                        safety_policy_version="v1",
                        action_id=committed_action_id,
                        allowed=True,
                        reason_code="ALLOW_REGISTERED_FIXTURE_ACTION",
                        source_record_type="FixtureActionSafetyDecision",
                        source_record_sha256=_FIXTURE_SHA256,
                    ),
                ),
            )
        super().__init__("private fixture infrastructure detail")


class _RawFixturePage:
    def __init__(self) -> None:
        self.url = "https://fixture.invalid/start"
        self.actions = 0
        self.browser_error_kind: str | None = None
        self.closed = False


class FixtureSealedBackend:
    """Owns the raw fixture page; no runtime message returns this object."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        if config.get("schema_version") != (
            "table2-process-broker-fixture-backend-v1"
        ):
            raise ValueError("fixture backend config version differs")
        self._raw_page = _RawFixturePage()
        self._terminal_events = 0
        screenshot_root = config.get("screenshot_root")
        self._screenshot_root = (
            Path(str(screenshot_root)) if screenshot_root is not None else None
        )
        self._observation_count = 0

    def runtime_reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        episode_id = str(payload["episode_id"])
        task_id = str(payload["task_id"])
        benchmark_version = str(payload["benchmark_version"])
        start_state_id = str(payload["start_state_id"])
        reset_stage_seed = int(payload["reset_stage_seed"])
        commitments = {
            role: _FIXTURE_SHA256
            for role in REGISTERED_WEBARENA_RESET_COMMITMENTS
        }
        reset_state_sha256 = webarena_reset_state_digest(
            attester_id="fixture-reset-attester",
            attester_version="1.0",
            attester_source_sha256=_FIXTURE_SHA256,
            task_id=task_id,
            benchmark_version=benchmark_version,
            start_state_id=start_state_id,
            reset_stage_seed=reset_stage_seed,
            commitments_sha256=commitments,
        )
        observation = self.runtime_observe(
            {
                "episode_id": episode_id,
                "task_id": task_id,
                "stage": "reset",
                "prior_action_id": None,
            }
        )["observation"]
        receipt = WebArenaResetStateReceipt(
            episode_id=episode_id,
            task_id=task_id,
            benchmark_version=benchmark_version,
            start_state_id=start_state_id,
            reset_stage_seed=reset_stage_seed,
            attester_id="fixture-reset-attester",
            attester_version="1.0",
            attester_source_sha256=_FIXTURE_SHA256,
            commitments_sha256=commitments,
            reset_state_sha256=reset_state_sha256,
            reset_applied=True,
            oracle_labels_observed=False,
        )
        return {
            "observation": observation,
            "reset_state_receipt": receipt.to_dict(),
        }

    def runtime_observe(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        episode_id = str(payload["episode_id"])
        task_id = str(payload["task_id"])
        stage = ObservationStage(str(payload["stage"]))
        prior_action_id = payload["prior_action_id"]
        browser_error_kind = self._raw_page.browser_error_kind
        self._observation_count += 1
        observation_id = f"{episode_id}:{stage.value}:{self._observation_count}"
        observation = Observation(
                observation_id=observation_id,
                episode_id=episode_id,
                stage=stage,
                screenshot_sha256=_FIXTURE_SHA256,
                screenshot_path=None,
                width=1280,
                height=720,
                url=self._raw_page.url,
                title="Fixture",
                page_state={
                    "schema_version": "table2-browsergym-causal-observation-v1",
                    "visible_text": (
                        f"fixture action count {self._raw_page.actions}"
                    ),
                    "visible_controls": [],
                    "has_browser_error": browser_error_kind is not None,
                    "browser_error_kind": browser_error_kind,
                    "observable_select_controls": [],
                    "recovery_target_evidence": {
                        "schema_version": "oracle-blind-visible-targets-v1",
                        "observation_id": observation_id,
                        "task_id": task_id,
                        "task_goal_sha256": _FIXTURE_SHA256,
                        "registered_visible_targets": [],
                    },
                },
                page_settled=True,
                environment_error=browser_error_kind is not None,
                prior_action_id=(
                    str(prior_action_id) if prior_action_id is not None else None
                ),
            ).to_dict()
        if self._screenshot_root is not None:
            screenshot_bytes = (
                b"\x89PNG\r\n\x1a\nfixture-" + observation_id.encode("utf-8")
            )
            digest = hashlib.sha256(screenshot_bytes).hexdigest()
            screenshot_path = self._screenshot_root / (
                f"{self._observation_count:06d}-{digest}.png"
            )
            screenshot_path.write_bytes(screenshot_bytes)
            screenshot_path.chmod(0o400)
            observation["screenshot_sha256"] = digest
            observation["screenshot_path"] = str(screenshot_path)
        return {"observation": observation}

    def runtime_execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        action_value = payload.get("action")
        if not isinstance(action_value, Mapping):
            raise ValueError("fixture execution requires an action object")
        action = ConcreteAction.from_dict(action_value)
        if type(action) is not ConcreteAction:
            raise ValueError("fixture execution requires ConcreteAction")
        if action.action_id == "late-unregistered-import":
            # The worker must enforce its source closure after each backend
            # operation, not only while publishing readiness.
            from web_agent.runtime import action_parameters as _late  # noqa: F401
        self._raw_page.actions += 1
        if action.action_id == "emit-browser-error":
            self._raw_page.browser_error_kind = "TIMEOUTERROR"
            self._raw_page.url = registered_browser_error_observation_url(
                self._raw_page.browser_error_kind
            )
        elif action.action_type is ActionType.NAVIGATE:
            self._raw_page.url = str(action.parameters["url"])
        if action.action_id == "aliased-result":
            return {"execution": {"score": 1}}
        return {
            "execution": AdapterExecution(
                status=ExecutionStatus.EXECUTED,
                state_changed=True,
                environment_error=False,
                error_kind=None,
                message="browser request completed",
                internal_retry_count=0,
                latency_ms=0.0,
                evidence=ExecutionEvidence(
                    action_id=action.action_id,
                    started_at_utc=_FIXTURE_TIMESTAMP,
                    ended_at_utc=_FIXTURE_TIMESTAMP,
                    status=ExecutionStatus.EXECUTED,
                ),
            ).to_dict()
        }

    def runtime_register_rejected(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        action_value = payload.get("action")
        execution_value = payload.get("execution")
        if not isinstance(action_value, Mapping) or not isinstance(
            execution_value, Mapping
        ):
            raise ValueError("fixture rejected registration requires exact records")
        action = ConcreteAction.from_dict(action_value)
        execution = ExecutionResult.from_dict(execution_value)
        if (
            execution.action_id != action.action_id
            or execution.status is not ExecutionStatus.REJECTED
            or execution.state_changed is not False
            or execution.environment_error is not False
        ):
            raise ValueError("fixture rejected registration differs")
        # Deliberately do not change ``_raw_page.actions``: no controller step
        # was requested for an executor-local rejection.
        return {"registered": True}

    def runtime_terminal(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        binding = VerifierReceiptBinding.from_dict(payload["receipt_binding"])
        self._terminal_events += 1
        signal = OpaqueTerminalSignal(
            event_id=f"fixture-terminal-{self._terminal_events}",
            token_sha256=sha256_json(binding.to_dict()),
            terminate=self._raw_page.actions >= 1,
        )
        return {"opaque_terminal_signal": signal.to_dict()}

    def runtime_close(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        del payload
        self._raw_page.closed = True
        return {"closed": True}

    def shutdown(self) -> None:
        self._raw_page.closed = True


class FixtureWebArenaEnvironmentAdapter(EnvironmentAdapter):
    """EnvironmentAdapter fake used to exercise the production backend bridge."""

    benchmark_id = "webarena"

    def __init__(self, task: TaskSpecification) -> None:
        self.benchmark_version = str(task.benchmark_version)
        self._adapter_id = "fixture-process-webarena"
        self._adapter_version = "v1"
        self._task = task
        self._backend = FixtureSealedBackend(
            {"schema_version": "table2-process-broker-fixture-backend-v1"}
        )
        self._episode_id: str | None = None
        self._receipt: WebArenaResetStateReceipt | None = None
        self._closed = False

    def reset(
        self,
        task: TaskSpecification,
        *,
        episode_id: str,
        seed: int,
    ) -> Observation:
        if task.record_sha256 != self._task.record_sha256:
            raise ValueError("fixture adapter task identity differs")
        result = self._backend.runtime_reset(
            {
                "episode_id": episode_id,
                "task_id": task.task_id,
                "benchmark_version": task.benchmark_version,
                "start_state_id": task.start_state_id,
                "reset_stage_seed": seed,
            }
        )
        self._episode_id = episode_id
        self._receipt = WebArenaResetStateReceipt.from_dict(
            result["reset_state_receipt"]
        )
        return Observation.from_dict(result["observation"])

    def reset_state_receipt(self) -> WebArenaResetStateReceipt | None:
        return self._receipt

    def observe(
        self,
        *,
        stage: ObservationStage,
        prior_action_id: str | None = None,
    ) -> Observation:
        if self._episode_id is None:
            raise ValueError("fixture adapter has not reset")
        return Observation.from_dict(
            self._backend.runtime_observe(
                {
                    "episode_id": self._episode_id,
                    "task_id": self._task.task_id,
                    "stage": stage.value,
                    "prior_action_id": prior_action_id,
                }
            )["observation"]
        )

    def execute(self, action: ConcreteAction) -> AdapterExecution:
        if self._episode_id is None:
            raise ValueError("fixture adapter has not reset")
        if action.action_id == "emit-infrastructure-invalid":
            raise FixtureInfrastructureInvalidError(
                self._episode_id,
                action=action,
            )
        if action.action_id == "emit-infrastructure-invalid-wrong-action":
            raise FixtureInfrastructureInvalidError(
                self._episode_id,
                action=action,
                evidence_action_id="another-action",
            )
        return AdapterExecution.from_dict(
            self._backend.runtime_execute(
                {
                    "episode_id": self._episode_id,
                    "task_id": self._task.task_id,
                    "action": action.to_dict(),
                }
            )["execution"]
        )

    def terminal_signal(
        self,
        task: TaskSpecification,
        binding: VerifierReceiptBinding | None = None,
    ) -> OpaqueTerminalSignal:
        if task.record_sha256 != self._task.record_sha256:
            raise ValueError("fixture adapter terminal task identity differs")
        if type(binding) is not VerifierReceiptBinding:
            raise TypeError("fixture adapter requires verifier receipt binding")
        if task.goal == "fixture forged terminal infrastructure operation":
            # Deliberately claim a globally registered step fault from the
            # terminal callback. The bridge/worker must reject the cross-call
            # attribution instead of granting the typed rerun channel.
            raise FixtureInfrastructureInvalidError(str(self._episode_id))
        return OpaqueTerminalSignal.from_dict(
            self._backend.runtime_terminal(
                {
                    "episode_id": str(self._episode_id),
                    "task_id": self._task.task_id,
                    "receipt_binding": binding.to_dict(),
                }
            )["opaque_terminal_signal"]
        )

    def close(self) -> None:
        if self._closed:
            return None
        if self._episode_id is not None:
            self._backend.runtime_close(
                {
                    "episode_id": self._episode_id,
                    "task_id": self._task.task_id,
                }
            )
        self._closed = True
        return None


def create_environment_adapter(
    *, task: TaskSpecification, episode_runtime_dir: Path
) -> FixtureWebArenaEnvironmentAdapter:
    if not episode_runtime_dir.is_dir():
        raise ValueError("fixture adapter runtime directory is unavailable")
    return FixtureWebArenaEnvironmentAdapter(task)


def finalize_environment_episode(
    *,
    task: TaskSpecification,
    adapter: EnvironmentAdapter,
    episode_summary: EpisodeSummary,
    episode_runtime_dir: Path,
    evidence_writer: SealedVerifierWriter,
) -> OpaqueTerminalSignal:
    """Deterministic child-only finalizer for orchestration fixture tests."""

    if (
        type(task) is not TaskSpecification
        or type(episode_summary) is not EpisodeSummary
        or type(evidence_writer) is not SealedVerifierWriter
        or not episode_runtime_dir.is_dir()
    ):
        raise TypeError("fixture sealed finalizer received the wrong contract")
    if "finalizer-error" in task.goal:
        raise RuntimeError("private fixture finalizer detail")
    raw_backend = getattr(adapter, "_backend", None)
    raw_page = getattr(raw_backend, "_raw_page", None)
    action_count = int(getattr(raw_page, "actions", 0))
    task_success = (
        action_count > 0
        and episode_summary.terminal_reason
        is TerminalReason.OPAQUE_VERIFIER_TERMINAL
    )
    evidence = {
        "task_success": task_success,
        "terminal_reason": episode_summary.terminal_reason.value,
        "loop_detected": (
            episode_summary.terminal_reason is TerminalReason.LOOP
        ),
        "environment_failure": episode_summary.environment_failure,
        "failure_incidents": [],
        "recovery_verifications": [],
        "verified_failure_event_count": 0,
        "repeated_error_event_count": 0,
        "memory_relevance": {
            "query_count": episode_summary.memory_queries,
            "intervention_count": episode_summary.memory_interventions,
        },
    }
    if task.goal == "fixture finalizer no write":
        return OpaqueTerminalSignal(
            event_id="fixture-forged-final",
            token_sha256="f" * 64,
            terminate=True,
        )
    signal = evidence_writer.record_episode_final(evidence)
    if task.goal == "fixture finalizer double write":
        evidence_writer.record_episode_final(evidence)
    if task.goal == "fixture finalizer write then raise":
        raise RuntimeError("private fixture finalizer after-write detail")
    if task.goal == "fixture finalizer forged signal":
        return OpaqueTerminalSignal(
            event_id=signal.event_id,
            token_sha256="f" * 64,
            terminate=True,
        )
    return signal


def evaluate_environment_transition(
    *,
    task: TaskSpecification,
    adapter: EnvironmentAdapter,
    receipt_binding: VerifierReceiptBinding,
    evidence_writer: SealedVerifierWriter,
) -> OpaqueTerminalSignal:
    """Emit full transition evidence only through the child writer."""

    if (
        type(task) is not TaskSpecification
        or type(receipt_binding) is not VerifierReceiptBinding
        or type(evidence_writer) is not SealedVerifierWriter
    ):
        raise TypeError("fixture transition evaluator received the wrong contract")
    raw_backend = getattr(adapter, "_backend", None)
    raw_page = getattr(raw_backend, "_raw_page", None)
    action_count = int(getattr(raw_page, "actions", 0))
    if task.goal == "fixture transition no write":
        return OpaqueTerminalSignal(
            event_id="fixture-forged-transition",
            token_sha256="f" * 64,
            terminate=False,
        )
    signal = evidence_writer.record_bound_receipt(
        receipt_binding,
        {
            "fixture_page_action_count": action_count,
            "fixture_page_url_sha256": hashlib.sha256(
                str(getattr(raw_page, "url", "")).encode("utf-8")
            ).hexdigest(),
        },
        should_terminate=action_count >= 1,
    )
    if task.goal == "fixture transition double write":
        evidence_writer.record_bound_receipt(
            receipt_binding,
            {"fixture_duplicate_transition": True},
            should_terminate=action_count >= 1,
        )
    if task.goal == "fixture transition write then raise":
        raise RuntimeError("private fixture transition after-write detail")
    if task.goal == "fixture transition forged signal":
        return OpaqueTerminalSignal(
            event_id=signal.event_id,
            token_sha256="f" * 64,
            terminate=signal.terminate,
        )
    return signal
def create_backend(config: Mapping[str, Any]) -> FixtureSealedBackend:
    return FixtureSealedBackend(config)


def create_failing_backend(config: Mapping[str, Any]) -> FixtureSealedBackend:
    del config
    raise RuntimeError("fixture startup failure")


def create_unregistered_import_backend(
    config: Mapping[str, Any],
) -> FixtureSealedBackend:
    # Deliberately expands the worker's repository-local import graph.  The
    # readiness closure test requires the broker to reject this factory before
    # publishing READY because this source is not in the launch manifest.
    from web_agent.runtime import action_parameters as _unregistered  # noqa: F401

    return FixtureSealedBackend(config)
