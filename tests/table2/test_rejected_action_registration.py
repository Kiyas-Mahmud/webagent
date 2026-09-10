from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from web_agent.benchmarks.fixture import DeterministicFixtureAdapter
from web_agent.runtime.action_parameters import (
    CallableActionParameterProvider,
    DeterministicParameterProvider,
    HybridParameterProvider,
)
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    ExecutionResult,
    ExecutionStatus,
    MAX_EXECUTION_MESSAGE_CHARS,
    PreActionDecision,
    SystemID,
    TaskSpecification,
    canonical_sha256,
    probability_map,
)
from web_agent.runtime.episode import EpisodeRunner
from web_agent.runtime.event_log import EpisodeEventLogs
from web_agent.runtime.executor import (
    EpisodeTimeout,
    Executor,
    ExecutorInputMutation,
    RejectedActionRegistrationError,
)
from web_agent.runtime.policy import CallablePolicyAdapter, PolicyKind, SystemPolicy
from web_agent.runtime.protocol import RuntimeProtocol, switches_for


SHA = "a" * 64
BBOX = (0.10, 0.10, 0.20, 0.10)


class _RegistrationBaseFailure(BaseException):
    pass


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _RegistrationFaultAdapter(DeterministicFixtureAdapter):
    def __init__(self, mode: str, clock: _Clock) -> None:
        super().__init__()
        self.mode = mode
        self.clock = clock

    def register_rejected_action(self, action, execution) -> None:
        if self.mode == "failure":
            raise RuntimeError("injected rejected-action registration failure")
        if self.mode == "mutation":
            action.parameters["mutated"] = True
            return None
        if self.mode == "timeout":
            self.clock.now += 600.0
            return None
        if self.mode == "base_exception":
            raise _RegistrationBaseFailure("injected BaseException")
        raise AssertionError(f"unknown registration fault mode: {self.mode}")


_EXPECTED_CAUSE = {
    "failure": RuntimeError,
    "mutation": ExecutorInputMutation,
    "timeout": EpisodeTimeout,
    "base_exception": _RegistrationBaseFailure,
}


def _task() -> TaskSpecification:
    return TaskSpecification(
        task_id="fixture-six-actions",
        goal="exercise rejected-action registration",
        benchmark_id="fixture",
        benchmark_version="v1",
        start_state_id="s0",
    )


def _policy() -> SystemPolicy:
    policy_id = "rejected-registration-policy"

    def predict(task, observation, rng):
        del task, rng
        return PreActionDecision(
            decision_id=f"decision:{observation.observation_id}",
            observation_id=observation.observation_id,
            action_type=ActionType.TYPE,
            action_probabilities=probability_map(
                tuple(item.value for item in ActionType),
                ActionType.TYPE.value,
            ),
            bbox=BBOX,
            grounding_confidence=1.0,
            confidence_before=1.0,
            input_observation_ids=(observation.observation_id,),
            policy_id=policy_id,
            policy_version="v1",
            # Missing text forces the registered unresolved-parameter path.
            parameter_hints={},
        )

    return SystemPolicy(
        CallablePolicyAdapter(
            policy_id=policy_id,
            policy_version="v1",
            kind=PolicyKind.TRAINED,
            checkpoint_sha256=SHA,
            action_predictor=predict,
        ),
        switches_for(SystemID.E1),
    )


def _provider(kind: str):
    deterministic = DeterministicParameterProvider()
    if kind == "parameter":
        return deterministic
    if kind != "hybrid":
        raise AssertionError(f"unknown provider kind: {kind}")

    def reject_fallback(task, observation, decision, rng):
        del task, observation, decision, rng
        return {}

    fallback = CallableActionParameterProvider(
        provider_id="frozen-base-parameter-provider",
        provider_version="v1",
        prompt_sha256=canonical_sha256("frozen fallback prompt"),
        resolver=reject_fallback,
    )
    return HybridParameterProvider(
        deterministic=deterministic,
        frozen_base_fallback=fallback,
    )


def _read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.parametrize("mode", tuple(_EXPECTED_CAUSE))
def test_executor_wraps_every_rejected_registration_fault_after_exact_charge(
    mode: str,
) -> None:
    clock = _Clock()
    executor = Executor(
        _RegistrationFaultAdapter(mode, clock),
        budgets=RuntimeProtocol(
            protocol_id="rejected-registration-protocol",
            campaign_id="rejected-registration-campaign",
            provider_id="deterministic-parameter-provider",
            provider_version="v1",
        ).budgets,
        clock=clock,
    )
    executor.reset(_task(), episode_id="rejected-registration-episode", seed=42)
    action = ConcreteAction(
        action_id="unresolved-action",
        source_decision_id="decision-1",
        action_type=ActionType.TYPE,
        parameters={},
        bbox=BBOX,
    )

    with pytest.raises(RejectedActionRegistrationError) as caught:
        executor.reject_unresolved_request(
            action=action,
            reason="provider could not resolve parameters",
        )

    assert type(caught.value.__cause__) is _EXPECTED_CAUSE[mode]
    execution = caught.value.execution_result
    assert type(execution) is ExecutionResult
    assert execution.action_id == action.action_id
    assert execution.status is ExecutionStatus.REJECTED
    assert execution.error_kind == "parameter_resolution_rejected"
    assert execution.executor_step == executor.steps_used == 1
    assert execution.evidence is not None
    assert execution.evidence.action_id == action.action_id
    executor.close()


def test_long_unresolved_reason_is_hash_bounded_before_process_registration() -> None:
    executor = Executor(
        DeterministicFixtureAdapter(),
        budgets=RuntimeProtocol(
            protocol_id="bounded-rejection-protocol",
            campaign_id="bounded-rejection-campaign",
            provider_id="deterministic-parameter-provider",
            provider_version="v1",
        ).budgets,
    )
    executor.reset(_task(), episode_id="bounded-rejection-episode", seed=42)
    action = ConcreteAction(
        action_id="bounded-unresolved-action",
        source_decision_id="decision-1",
        action_type=ActionType.TYPE,
        parameters={},
        bbox=BBOX,
    )
    reason = "x" * (MAX_EXECUTION_MESSAGE_CHARS + 1)

    execution = executor.reject_unresolved_request(action=action, reason=reason)

    assert execution.executor_step == executor.steps_used == 1
    assert len(execution.message) <= MAX_EXECUTION_MESSAGE_CHARS
    assert hashlib.sha256(reason.encode("utf-8")).hexdigest() in execution.message
    executor.close()


@pytest.mark.parametrize("provider_kind", ("parameter", "hybrid"))
@pytest.mark.parametrize("mode", tuple(_EXPECTED_CAUSE))
def test_unresolved_registration_fault_is_logged_before_propagation(
    tmp_path: Path,
    provider_kind: str,
    mode: str,
) -> None:
    clock = _Clock()
    provider = _provider(provider_kind)
    protocol = RuntimeProtocol(
        protocol_id="rejected-registration-protocol",
        campaign_id="rejected-registration-campaign",
        provider_id=provider.provider_id,
        provider_version=provider.provider_version,
    )
    task = _task()
    episode_id = (
        f"{protocol.campaign_id}:E1:{task.task_id}:repeat-0:seed-42"
    )
    logs = EpisodeEventLogs(
        tmp_path,
        episode_id=episode_id,
        include_memory=False,
        system_id="E1",
        task_id=task.task_id,
        repeat_id=0,
        matched_seed=42,
        timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
    )
    runner = EpisodeRunner(
        protocol=protocol,
        system_policy=_policy(),
        provider=provider,
        executor=Executor(
            _RegistrationFaultAdapter(mode, clock),
            budgets=protocol.budgets,
            clock=clock,
        ),
        event_logs=logs,
    )

    with pytest.raises(RejectedActionRegistrationError) as caught:
        runner.run(task, repeat_id=0, model_seed=42)

    assert type(caught.value.__cause__) is _EXPECTED_CAUSE[mode]
    partial = caught.value.partial_episode_summary
    assert partial["executor_steps"] == partial["normal_actions"] == 1
    assert partial["recovery_actions"] == 0
    action_rows = _read_rows(tmp_path / "actions.jsonl")
    assert len(action_rows) == 1
    payload = action_rows[0]["payload"]
    assert action_rows[0]["event_type"] == "normal_action"
    assert payload["interrupted"] is True
    assert payload["execution"]["executor_step"] == 1
    assert payload["execution"]["status"] == "rejected"
    assert payload["execution"]["error_kind"] == "parameter_resolution_rejected"
    assert canonical_sha256(payload["execution"]) == (
        caught.value.execution_result.record_sha256
    )
    assert (payload["parameter_resolution"] is not None) is (
        provider_kind == "hybrid"
    )
    environment_rows = _read_rows(tmp_path / "environment_events.jsonl")
    assert all(
        row["event_type"] != "post_action_observation"
        for row in environment_rows
    )
    # Only the reset receipt exists; registration failure cannot fabricate the
    # post-action observation needed for another sealed verifier receipt.
    assert len(_read_rows(tmp_path / "terminal_signals.jsonl")) == 1
    assert runner.contract_validation_receipt is None
