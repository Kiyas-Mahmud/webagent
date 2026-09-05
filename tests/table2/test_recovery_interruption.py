from __future__ import annotations

import json
from pathlib import Path

import pytest

from web_agent.benchmarks.recovery_fixture import (
    RecoveryDiagnosticAdapter,
    load_registered_recovery_diagnostics,
)
from web_agent.runtime.action_parameters import DeterministicParameterProvider
from web_agent.runtime.contracts import (
    ActionType,
    PreActionDecision,
    RecoveryAssessment,
    RecoveryStrategy,
    SystemID,
    TaskSpecification,
    TerminalReason,
    TransitionAssessment,
    probability_map,
)
from web_agent.runtime.episode import EpisodeRunner
from web_agent.runtime.event_log import EpisodeEventLogs
from web_agent.runtime.executor import (
    EpisodeTimeout,
    Executor,
    ExecutorBudgetExceeded,
    ExecutorInputMutation,
    RejectedActionRegistrationError,
)
from web_agent.runtime.policy import (
    CallablePolicyAdapter,
    PolicyError,
    PolicyKind,
    SystemPolicy,
)
from web_agent.runtime.protocol import RuntimeProtocol, switches_for
from web_agent.runtime.recovery.controller import RecoveryController


SHA = "a" * 64
BBOX = (0.10, 0.10, 0.20, 0.10)


class _RecoveryExecuteInterrupt(Executor):
    def __init__(self, *args, interruption: type[Exception], **kwargs):
        super().__init__(*args, **kwargs)
        self._interruption = interruption

    def execute(self, action):
        if action.recovery_attempt_id is not None:
            raise self._interruption("injected after begin_attempt")
        return super().execute(action)


class _RecoveryRegistrationBaseFailure(BaseException):
    pass


class _RecoveryRegistrationClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _RecoveryRegistrationFaultAdapter(RecoveryDiagnosticAdapter):
    def __init__(self, scenario, *, mode: str, clock: _RecoveryRegistrationClock):
        super().__init__(scenario)
        self._registration_fault_mode = mode
        self._registration_clock = clock

    def register_rejected_action(self, action, execution) -> None:
        if self._registration_fault_mode == "failure":
            raise RuntimeError("injected recovery registration failure")
        if self._registration_fault_mode == "mutation":
            action.parameters["mutated"] = True
            return None
        if self._registration_fault_mode == "timeout":
            self._registration_clock.now += 600.0
            return None
        if self._registration_fault_mode == "base_exception":
            raise _RecoveryRegistrationBaseFailure(
                "injected recovery registration BaseException"
            )
        raise AssertionError(
            f"unknown recovery registration fault: {self._registration_fault_mode}"
        )


class _RecoveryLocalSafetyRejectExecutor(Executor):
    def _safety_error(self, action):
        if action.recovery_attempt_id is not None:
            return "injected recovery safety rejection"
        return super()._safety_error(action)


_RECOVERY_REGISTRATION_CAUSE = {
    "failure": RuntimeError,
    "mutation": ExecutorInputMutation,
    "timeout": EpisodeTimeout,
    "base_exception": _RecoveryRegistrationBaseFailure,
}


def _retry_scenario():
    scenarios, _ = load_registered_recovery_diagnostics()
    return next(
        item
        for item in scenarios
        if item.expected_strategy is RecoveryStrategy.RETRY
    )


def _task(scenario) -> TaskSpecification:
    return TaskSpecification(
        task_id=scenario.scenario_id,
        goal="exercise exception-safe recovery accounting",
        benchmark_id="recovery_fixture",
        benchmark_version="v1",
        start_state_id="failure-present",
        metadata={"task_partition": "recovery_diagnostic"},
    )


def _policy(*, fail_recovery_assessment: bool = False) -> SystemPolicy:
    policy_id = "interruption-test-policy"

    def action_predictor(task, observation, rng):
        del task, rng
        return PreActionDecision(
            decision_id=f"decision:{observation.observation_id}",
            observation_id=observation.observation_id,
            action_type=ActionType.CLICK,
            action_probabilities=probability_map(
                tuple(item.value for item in ActionType),
                ActionType.CLICK.value,
            ),
            bbox=BBOX,
            grounding_confidence=1.0,
            confidence_before=1.0,
            input_observation_ids=(observation.observation_id,),
            policy_id=policy_id,
            policy_version="v1",
        )

    def transition_predictor(task, transition, rng):
        del task, rng
        return TransitionAssessment(
            assessment_id=f"assessment:{transition.executed_action.action_id}",
            pre_observation_id=transition.pre_observation.observation_id,
            post_observation_id=transition.post_observation.observation_id,
            executed_action_id=transition.executed_action.action_id,
            predicted_failure=True,
            failure_probability=1.0,
            failure_type="REJECTED_ACTION",
            failure_type_probabilities={"REJECTED_ACTION": 1.0},
            needs_recovery=True,
            needs_recovery_probability=1.0,
            recovery_strategy=RecoveryStrategy.RETRY,
            recovery_probabilities={RecoveryStrategy.RETRY.value: 1.0},
        )

    def recovery_predictor(task, transition, rng):
        del task, rng
        if fail_recovery_assessment:
            raise PolicyError("injected recovery assessment failure")
        return RecoveryAssessment(
            assessment_id=f"recovery-assessment:{transition.attempt_id}",
            incident_id=transition.incident_id,
            attempt_id=transition.attempt_id,
            pre_recovery_observation_id=(
                transition.pre_recovery_observation.observation_id
            ),
            post_recovery_observation_id=(
                transition.post_recovery_observation.observation_id
            ),
            recovery_action_ids=tuple(
                action.action_id for action in transition.recovery_actions
            ),
            predicted_failure_resolved=True,
            predicted_resolution_probability=1.0,
            predicted_progress=True,
            predicted_progress_probability=1.0,
        )

    return SystemPolicy(
        CallablePolicyAdapter(
            policy_id=policy_id,
            policy_version="v1",
            kind=PolicyKind.TRAINED,
            checkpoint_sha256=SHA,
            action_predictor=action_predictor,
            transition_predictor=transition_predictor,
            recovery_predictor=recovery_predictor,
        ),
        switches_for(SystemID.E2),
    )


def _runner(
    root: Path,
    *,
    interruption: type[Exception] | None = None,
    fail_recovery_assessment: bool = False,
):
    scenario = _retry_scenario()
    task = _task(scenario)
    protocol = RuntimeProtocol(
        protocol_id="interruption-protocol",
        campaign_id="interruption-campaign",
        provider_id="deterministic-parameter-provider",
        provider_version="v1",
    )
    episode_id = (
        f"{protocol.campaign_id}:E2:{task.task_id}:repeat-0:seed-42"
    )
    logs = EpisodeEventLogs(
        root,
        episode_id=episode_id,
        include_memory=False,
        system_id="E2",
        task_id=task.task_id,
        repeat_id=0,
        matched_seed=42,
        timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
    )
    adapter = RecoveryDiagnosticAdapter(scenario)
    executor = (
        _RecoveryExecuteInterrupt(
            adapter,
            budgets=protocol.budgets,
            clock=lambda: 0.0,
            interruption=interruption,
        )
        if interruption is not None
        else Executor(adapter, budgets=protocol.budgets, clock=lambda: 0.0)
    )
    runner = EpisodeRunner(
        protocol=protocol,
        system_policy=_policy(
            fail_recovery_assessment=fail_recovery_assessment
        ),
        provider=DeterministicParameterProvider(),
        executor=executor,
        recovery_controller=RecoveryController(protocol.budgets),
        event_logs=logs,
    )
    return runner, task


def _attempt_rows(root: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (root / "recoveries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return [row for row in rows if row["event_type"] == "recovery_attempt"]


@pytest.mark.parametrize(
    ("interruption", "terminal_reason"),
    [
        (ExecutorBudgetExceeded, TerminalReason.ACTION_BUDGET_EXHAUSTED),
        (EpisodeTimeout, TerminalReason.TIMEOUT),
    ],
)
def test_begun_recovery_is_logged_when_executor_interrupts(
    tmp_path: Path,
    interruption: type[Exception],
    terminal_reason: TerminalReason,
) -> None:
    runtime = tmp_path / interruption.__name__
    runner, task = _runner(runtime, interruption=interruption)
    summary = runner.run(task, repeat_id=0, model_seed=42)

    attempts = _attempt_rows(runtime)
    assert summary.terminal_reason is terminal_reason
    assert summary.recovery_attempts == len(attempts) == 1
    assert attempts[0]["payload"]["interrupted"] is True
    assert attempts[0]["payload"]["interruption_kind"] == interruption.__name__
    assert attempts[0]["payload"]["attempt"]["completed"] is False
    assert attempts[0]["payload"]["attempt"]["action_ids"] == []


def test_begun_recovery_is_logged_when_post_execution_assessment_fails(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "policy-error"
    runner, task = _runner(runtime, fail_recovery_assessment=True)
    summary = runner.run(task, repeat_id=0, model_seed=42)

    attempts = _attempt_rows(runtime)
    assert summary.terminal_reason is TerminalReason.POLICY_ERROR
    assert summary.recovery_attempts == len(attempts) == 1
    assert summary.recovery_actions == 1
    assert attempts[0]["payload"]["interruption_kind"] == "PolicyError"
    assert attempts[0]["payload"]["attempt"]["completed"] is False
    assert len(attempts[0]["payload"]["attempt"]["action_ids"]) == 1


@pytest.mark.parametrize("mode", tuple(_RECOVERY_REGISTRATION_CAUSE))
def test_recovery_local_rejection_registration_fault_commits_before_propagation(
    tmp_path: Path,
    mode: str,
) -> None:
    scenario = _retry_scenario()
    task = _task(scenario)
    protocol = RuntimeProtocol(
        protocol_id="recovery-registration-protocol",
        campaign_id="recovery-registration-campaign",
        provider_id="deterministic-parameter-provider",
        provider_version="v1",
    )
    episode_id = (
        f"{protocol.campaign_id}:E2:{task.task_id}:repeat-0:seed-42"
    )
    logs = EpisodeEventLogs(
        tmp_path,
        episode_id=episode_id,
        include_memory=False,
        system_id="E2",
        task_id=task.task_id,
        repeat_id=0,
        matched_seed=42,
        timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
    )
    clock = _RecoveryRegistrationClock()
    adapter = _RecoveryRegistrationFaultAdapter(
        scenario,
        mode=mode,
        clock=clock,
    )
    runner = EpisodeRunner(
        protocol=protocol,
        system_policy=_policy(),
        provider=DeterministicParameterProvider(),
        executor=_RecoveryLocalSafetyRejectExecutor(
            adapter,
            budgets=protocol.budgets,
            clock=clock,
        ),
        recovery_controller=RecoveryController(protocol.budgets),
        event_logs=logs,
    )

    with pytest.raises(RejectedActionRegistrationError) as caught:
        runner.run(task, repeat_id=0, model_seed=42)

    assert type(caught.value.__cause__) is _RECOVERY_REGISTRATION_CAUSE[mode]
    partial = caught.value.partial_episode_summary
    assert partial["executor_steps"] == 2
    assert partial["normal_actions"] == 1
    assert partial["recovery_actions"] == 1
    assert partial["recovery_attempts"] == 1
    action_rows = [
        json.loads(line)
        for line in (tmp_path / "actions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(action_rows) == 2
    recovery_row = action_rows[-1]
    assert recovery_row["event_type"] == "recovery_action"
    assert recovery_row["payload"]["interrupted"] is True
    assert recovery_row["payload"]["execution"]["executor_step"] == 2
    assert recovery_row["payload"]["execution"]["status"] == "rejected"
    assert (
        recovery_row["payload"]["execution"]["error_kind"]
        == "safety_rejection"
    )
    attempts = _attempt_rows(tmp_path)
    assert len(attempts) == 1
    assert attempts[0]["payload"]["interrupted"] is True
    assert attempts[0]["payload"]["interruption_kind"] == (
        "RejectedActionRegistrationError"
    )
    assert attempts[0]["payload"]["attempt"]["action_ids"] == [
        recovery_row["payload"]["action"]["action_id"]
    ]
    environment_rows = [
        json.loads(line)
        for line in (tmp_path / "environment_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(
        row["event_type"] != "post_recovery_observation"
        for row in environment_rows
    )
    terminal_rows = [
        json.loads(line)
        for line in (tmp_path / "terminal_signals.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    # Reset and the failed normal action were sealed before recovery. The
    # rejected recovery registration cannot invent a third receipt.
    assert len(terminal_rows) == 2
    assert runner.contract_validation_receipt is None
