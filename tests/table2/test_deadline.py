from __future__ import annotations

import json
from pathlib import Path
import select
import time
from types import SimpleNamespace

import pytest

from web_agent.eval.table2.production_runner import (
    ProductionRunnerError,
    _finish_measurement_under_deadline,
)
from web_agent.benchmarks.fixture import DeterministicFixtureAdapter
from web_agent.benchmarks.recovery_fixture import (
    RecoveryDiagnosticAdapter,
    load_registered_recovery_diagnostics,
)
from web_agent.runtime.action_parameters import (
    CallableActionParameterProvider,
    DeterministicParameterProvider,
)
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    EpisodeSummary,
    ObservationStage,
    PreActionDecision,
    RecoveryAssessment,
    RecoveryStrategy,
    SystemID,
    TaskSpecification,
    TerminalReason,
    TransitionAssessment,
    VerifierReceiptBinding,
    canonical_sha256,
    probability_map,
)
from web_agent.runtime.episode import EpisodeRunner
from web_agent.runtime.event_log import EpisodeEventLogs
from web_agent.runtime.executor import (
    EpisodeTimeout,
    Executor,
    ExecutorCleanupTimeout,
)
from web_agent.runtime.policy import CallablePolicyAdapter, PolicyKind, SystemPolicy
from web_agent.runtime.protocol import RuntimeProtocol, switches_for
from web_agent.runtime.recovery.controller import RecoveryController
from web_agent.runtime.protocol import REGISTERED_PRE_BROWSER_SETUP_TIMEOUT_SECONDS
from web_agent.runtime.state_reset import (
    PreBrowserSetupDeadline,
    PreBrowserSetupTimeout,
)


SHA = "a" * 64
BBOX = (0.10, 0.10, 0.20, 0.10)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_pre_browser_setup_has_separate_frozen_deadline_and_evidence() -> None:
    clock = _FakeClock()
    setup = PreBrowserSetupDeadline(
        REGISTERED_PRE_BROWSER_SETUP_TIMEOUT_SECONDS,
        clock=clock,
    )
    clock.advance(119.0)
    evidence = setup.complete(
        episode_id="deadline-campaign:E0:task:repeat-0:seed-42",
        system_id=SystemID.E0,
    )
    assert evidence.elapsed_seconds == 119.0
    assert evidence.timeout_seconds == 120.0
    assert evidence.boundary == "immediately_before_webarena_environment_reset"

    expired_clock = _FakeClock()
    expired = PreBrowserSetupDeadline(
        REGISTERED_PRE_BROWSER_SETUP_TIMEOUT_SECONDS,
        clock=expired_clock,
    )
    expired_clock.advance(120.0)
    with pytest.raises(PreBrowserSetupTimeout, match="browser reset boundary"):
        expired.complete(
            episode_id="deadline-campaign:E0:task:repeat-0:seed-42",
            system_id=SystemID.E0,
        )


class _DeadlineFixtureAdapter(DeterministicFixtureAdapter):
    def __init__(self, clock: _FakeClock, *, late_operation: str) -> None:
        super().__init__()
        self._clock = clock
        self._late_operation = late_operation

    def _finish(self, operation: str) -> None:
        if operation == self._late_operation:
            self._clock.advance(600.0)

    def reset(self, task, *, episode_id, seed):
        observation = super().reset(task, episode_id=episode_id, seed=seed)
        self._finish("reset")
        return observation

    def execute(self, action):
        execution = super().execute(action)
        self._finish("execute")
        return execution

    def observe(self, *, stage, prior_action_id=None):
        observation = super().observe(
            stage=stage,
            prior_action_id=prior_action_id,
        )
        if stage is ObservationStage.POST_ACTION:
            self._finish("observe")
        return observation

    def terminal_signal(self, task, binding=None):
        signal = super().terminal_signal(task, binding)
        self._finish("terminal")
        return signal

    def close(self) -> None:
        super().close()
        self._finish("close")


def _fixture_task() -> TaskSpecification:
    return TaskSpecification(
        task_id="fixture-six-actions",
        goal="exercise the registered deadline",
        benchmark_id="fixture",
        benchmark_version="v1",
        start_state_id="s0",
    )


def _click_action(action_id: str = "action:1") -> ConcreteAction:
    return ConcreteAction(
        action_id=action_id,
        source_decision_id="decision:1",
        action_type=ActionType.CLICK,
        parameters={
            "target_x": 0.20,
            "target_y": 0.15,
            "button": "left",
            "click_count": 1,
        },
        bbox=BBOX,
    )


def _reset_executor(
    clock: _FakeClock,
    *,
    late_operation: str,
) -> tuple[Executor, object]:
    protocol = RuntimeProtocol(
        protocol_id="deadline-protocol",
        campaign_id="deadline-campaign",
        provider_id="deterministic-parameter-provider",
        provider_version="v1",
    )
    executor = Executor(
        _DeadlineFixtureAdapter(clock, late_operation=late_operation),
        budgets=protocol.budgets,
        clock=clock,
    )
    observation = executor.reset(
        _fixture_task(),
        episode_id="deadline-episode",
        seed=42,
    )
    return executor, observation


def test_reset_result_returning_at_deadline_is_rejected() -> None:
    clock = _FakeClock()
    protocol = RuntimeProtocol(
        protocol_id="deadline-protocol",
        campaign_id="deadline-campaign",
        provider_id="deterministic-parameter-provider",
        provider_version="v1",
    )
    executor = Executor(
        _DeadlineFixtureAdapter(clock, late_operation="reset"),
        budgets=protocol.budgets,
        clock=clock,
    )

    with pytest.raises(EpisodeTimeout, match="environment reset"):
        executor.reset(
            _fixture_task(),
            episode_id="deadline-episode",
            seed=42,
        )


def test_execute_result_returning_at_deadline_is_rejected_but_charged() -> None:
    clock = _FakeClock()
    executor, _ = _reset_executor(clock, late_operation="execute")

    with pytest.raises(EpisodeTimeout, match="browser execute"):
        executor.execute(_click_action())

    assert executor.steps_used == 1


def test_observation_result_returning_at_deadline_is_rejected() -> None:
    clock = _FakeClock()
    executor, _ = _reset_executor(clock, late_operation="observe")
    action = _click_action()
    executor.execute(action)

    with pytest.raises(EpisodeTimeout, match="post-action observation"):
        executor.observe_after(action)


def test_sealed_terminal_result_returning_at_deadline_is_rejected() -> None:
    clock = _FakeClock()
    executor, observation = _reset_executor(clock, late_operation="terminal")
    binding = VerifierReceiptBinding(
        receipt_kind="after_reset",
        observation_id=observation.observation_id,
        observation_sha256=observation.record_sha256,
    )

    with pytest.raises(EpisodeTimeout, match="sealed terminal evaluation"):
        executor.terminal_signal(binding)


def test_late_exception_is_classified_as_timeout() -> None:
    clock = _FakeClock()
    executor, _ = _reset_executor(clock, late_operation="none")

    def late_failure() -> None:
        clock.advance(600.0)
        raise ValueError("a late result must not become a parser failure")

    with pytest.raises(EpisodeTimeout, match="provider inference") as caught:
        executor.run_blocking("provider inference", late_failure)

    assert isinstance(caught.value.__cause__, ValueError)


def test_executor_physically_interrupts_a_nonreturning_blocking_call() -> None:
    """The wall-clock cap is an interrupt, not only a post-return check."""

    budgets = SimpleNamespace(
        max_executor_steps=30,
        episode_timeout_seconds=0.05,
    )
    executor = Executor(DeterministicFixtureAdapter(), budgets=budgets)
    executor.reset(_fixture_task(), episode_id="hard-deadline-episode", seed=42)
    started = time.monotonic()

    with pytest.raises(EpisodeTimeout, match="nonreturning canary"):
        executor.run_blocking(
            "nonreturning canary",
            lambda: select.select([], [], [], 5.0),
        )

    assert time.monotonic() - started < 1.0


class _BlockingCloseAdapter(DeterministicFixtureAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        select.select([], [], [], 5.0)


def test_executor_physically_interrupts_blocking_environment_close() -> None:
    budgets = SimpleNamespace(
        max_executor_steps=30,
        episode_timeout_seconds=0.05,
    )
    adapter = _BlockingCloseAdapter()
    executor = Executor(adapter, budgets=budgets)
    executor.reset(_fixture_task(), episode_id="close-deadline", seed=42)
    started = time.monotonic()

    with pytest.raises(EpisodeTimeout, match="environment close"):
        executor.close()

    assert time.monotonic() - started < 1.0
    assert adapter.close_calls == 1
    # A timed-out close is never retried outside the registered deadline.
    executor.close()
    assert adapter.close_calls == 1


def test_executor_physically_interrupts_blocking_pre_reset_close() -> None:
    budgets = SimpleNamespace(
        max_executor_steps=30,
        episode_timeout_seconds=0.05,
    )
    adapter = _BlockingCloseAdapter()
    executor = Executor(adapter, budgets=budgets)
    started = time.monotonic()

    with pytest.raises(EpisodeTimeout, match="environment close"):
        executor.close()

    assert time.monotonic() - started < 1.0
    assert adapter.close_calls == 1


class _CloseCanaryAdapter(DeterministicFixtureAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_environment_close_is_not_entered_after_episode_deadline() -> None:
    clock = _FakeClock()
    adapter = _CloseCanaryAdapter()
    protocol = RuntimeProtocol(
        protocol_id="deadline-protocol",
        campaign_id="deadline-campaign",
        provider_id="deterministic-parameter-provider",
        provider_version="v1",
    )
    executor = Executor(adapter, budgets=protocol.budgets, clock=clock)
    executor.reset(_fixture_task(), episode_id="expired-close", seed=42)
    clock.advance(protocol.budgets.episode_timeout_seconds)

    executor.close()

    assert executor.deadline_expired is True
    assert adapter.close_calls == 0


def _finished_summary(*, model_call_count: int = 0) -> EpisodeSummary:
    return EpisodeSummary(
        episode_id="deadline-campaign:E0:task:repeat-0:seed-42",
        protocol_id="deadline-protocol",
        system_id=SystemID.E0,
        task_id="task",
        repeat_id=0,
        model_seed=42,
        valid_for_primary=True,
        terminal_reason=TerminalReason.CLOSED,
        executor_steps=0,
        normal_actions=0,
        recovery_actions=0,
        recovery_attempts=0,
        failure_incidents=0,
        memory_queries=0,
        memory_interventions=0,
        elapsed_seconds=0.0,
        model_call_count=model_call_count,
    )


def test_finish_measurement_physically_interrupts_blocking_callback() -> None:
    budgets = SimpleNamespace(
        max_executor_steps=30,
        episode_timeout_seconds=0.05,
    )
    executor = Executor(DeterministicFixtureAdapter(), budgets=budgets)
    executor.reset(_fixture_task(), episode_id="finish-deadline", seed=42)
    started = time.monotonic()

    with pytest.raises(
        ProductionRunnerError,
        match="efficiency measurement finalization exhausted",
    ) as caught:
        _finish_measurement_under_deadline(
            executor,
            lambda measurement, summary: select.select([], [], [], 5.0),
            object(),
            _finished_summary(),
        )

    assert time.monotonic() - started < 1.0
    assert isinstance(caught.value.__cause__, EpisodeTimeout)


def test_finish_measurement_is_not_entered_after_episode_timeout() -> None:
    clock = _FakeClock()
    protocol = RuntimeProtocol(
        protocol_id="deadline-protocol",
        campaign_id="deadline-campaign",
        provider_id="deterministic-parameter-provider",
        provider_version="v1",
    )
    executor = Executor(
        DeterministicFixtureAdapter(),
        budgets=protocol.budgets,
        clock=clock,
    )
    executor.reset(_fixture_task(), episode_id="expired-finish", seed=42)
    clock.advance(protocol.budgets.episode_timeout_seconds)
    called = False

    def forbidden_finish(measurement, summary):  # pragma: no cover - invariant
        del measurement, summary
        nonlocal called
        called = True
        raise AssertionError("expired efficiency callback must not run")

    supplied = _finish_measurement_under_deadline(
        executor,
        forbidden_finish,
        object(),
        _finished_summary(model_call_count=3),
    )

    assert called is False
    assert supplied["model_call_count"] == 3
    assert set(supplied) == {
        "model_call_count",
        "input_token_count",
        "output_token_count",
        "model_parameter_count",
        "trainable_parameter_count",
        "peak_gpu_memory_mb",
        "peak_system_memory_mb",
        "training_gpu_hours",
    }
    assert all(
        value is None
        for key, value in supplied.items()
        if key != "model_call_count"
    )


def test_pre_browser_setup_physically_interrupts_a_blocking_call() -> None:
    setup = PreBrowserSetupDeadline(0.05)
    started = time.monotonic()

    with pytest.raises(PreBrowserSetupTimeout, match="setup canary"):
        setup.run_blocking(
            "setup canary",
            lambda: select.select([], [], [], 5.0),
        )

    assert time.monotonic() - started < 1.0


def _decision(observation, *, policy_id: str) -> PreActionDecision:
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


def _runner(
    clock: _FakeClock,
    *,
    system_id: SystemID,
    action_predictor,
    transition_predictor=None,
    recovery_predictor=None,
    provider=None,
    adapter=None,
    event_logs: EpisodeEventLogs | None = None,
) -> EpisodeRunner:
    protocol = RuntimeProtocol(
        protocol_id="deadline-protocol",
        campaign_id="deadline-campaign",
        provider_id=(provider or DeterministicParameterProvider()).provider_id,
        provider_version=(provider or DeterministicParameterProvider()).provider_version,
    )
    policy_id = "deadline-policy"
    policy = SystemPolicy(
        CallablePolicyAdapter(
            policy_id=policy_id,
            policy_version="v1",
            kind=PolicyKind.TRAINED,
            checkpoint_sha256=SHA,
            action_predictor=action_predictor,
            transition_predictor=transition_predictor,
            recovery_predictor=recovery_predictor,
        ),
        switches_for(system_id),
    )
    return EpisodeRunner(
        protocol=protocol,
        system_policy=policy,
        provider=provider or DeterministicParameterProvider(),
        executor=Executor(
            adapter or DeterministicFixtureAdapter(),
            budgets=protocol.budgets,
            clock=clock,
        ),
        recovery_controller=(
            RecoveryController(protocol.budgets)
            if system_id in {SystemID.E2, SystemID.E3}
            else None
        ),
        event_logs=event_logs,
    )


def test_close_deadline_overrun_fails_package_without_relabeling_agent_timeout() -> None:
    clock = _FakeClock()

    def predict(task, observation, rng):
        del task, rng
        return _decision(observation, policy_id="deadline-policy")

    with pytest.raises(ExecutorCleanupTimeout, match="environment close") as caught:
        _runner(
            clock,
            system_id=SystemID.E1,
            action_predictor=predict,
            adapter=_DeadlineFixtureAdapter(clock, late_operation="close"),
        ).run(_fixture_task(), repeat_id=0, model_seed=42)

    # Only the typed adapter-classified infrastructure contract may consume a
    # whole-block rerun.  A generic cleanup defect instead stops fail-closed.
    assert getattr(caught.value, "infrastructure_invalid", False) is False


def test_late_pre_action_policy_result_cannot_control_browser() -> None:
    clock = _FakeClock()

    def late_predict(task, observation, rng):
        del task, rng
        clock.advance(600.0)
        return _decision(observation, policy_id="deadline-policy")

    summary = _runner(
        clock,
        system_id=SystemID.E1,
        action_predictor=late_predict,
    ).run(_fixture_task(), repeat_id=0, model_seed=42)

    assert summary.terminal_reason is TerminalReason.TIMEOUT
    assert summary.executor_steps == summary.normal_actions == 0
    assert summary.elapsed_seconds == 600.0


def test_late_parameter_provider_result_cannot_control_browser() -> None:
    clock = _FakeClock()

    def predict(task, observation, rng):
        del task, rng
        return _decision(observation, policy_id="deadline-policy")

    def late_resolve(task, observation, decision, rng):
        del task, observation, decision, rng
        clock.advance(600.0)
        return {
            "target_x": 0.20,
            "target_y": 0.15,
            "button": "left",
            "click_count": 1,
        }

    provider = CallableActionParameterProvider(
        provider_id="deadline-provider",
        provider_version="v1",
        prompt_sha256=canonical_sha256("deadline-provider-prompt"),
        resolver=late_resolve,
    )
    summary = _runner(
        clock,
        system_id=SystemID.E1,
        action_predictor=predict,
        provider=provider,
    ).run(_fixture_task(), repeat_id=0, model_seed=42)

    assert summary.terminal_reason is TerminalReason.TIMEOUT
    assert summary.executor_steps == summary.normal_actions == 0


def test_late_post_action_assessment_is_not_accepted() -> None:
    clock = _FakeClock()

    def predict(task, observation, rng):
        del task, rng
        return _decision(observation, policy_id="deadline-policy")

    def late_transition(task, transition, rng):
        del task, rng
        clock.advance(600.0)
        return TransitionAssessment(
            assessment_id=f"assessment:{transition.executed_action.action_id}",
            pre_observation_id=transition.pre_observation.observation_id,
            post_observation_id=transition.post_observation.observation_id,
            executed_action_id=transition.executed_action.action_id,
            predicted_failure=False,
            failure_probability=0.0,
            failure_type="NONE",
            failure_type_probabilities={"NONE": 1.0},
            needs_recovery=False,
            needs_recovery_probability=0.0,
            recovery_strategy=RecoveryStrategy.NONE,
            recovery_probabilities={RecoveryStrategy.NONE.value: 1.0},
        )

    def unused_recovery(task, transition, rng):  # pragma: no cover - invariant
        raise AssertionError("recovery assessment must not run")

    summary = _runner(
        clock,
        system_id=SystemID.E2,
        action_predictor=predict,
        transition_predictor=late_transition,
        recovery_predictor=unused_recovery,
    ).run(_fixture_task(), repeat_id=0, model_seed=42)

    assert summary.terminal_reason is TerminalReason.TIMEOUT
    assert summary.executor_steps == summary.normal_actions == 1
    assert summary.recovery_attempts == 0


def test_late_recovery_assessment_is_logged_and_not_accepted(tmp_path: Path) -> None:
    clock = _FakeClock()
    scenarios, _ = load_registered_recovery_diagnostics()
    scenario = next(
        item for item in scenarios if item.expected_strategy is RecoveryStrategy.RETRY
    )
    task = TaskSpecification(
        task_id=scenario.scenario_id,
        goal="exercise the recovery-assessment deadline",
        benchmark_id="recovery_fixture",
        benchmark_version="v1",
        start_state_id="failure-present",
        metadata={"task_partition": "recovery_diagnostic"},
    )

    def predict(task, observation, rng):
        del task, rng
        return _decision(observation, policy_id="deadline-policy")

    def diagnose(task, transition, rng):
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

    def late_recovery_assessment(task, transition, rng):
        del task, rng
        clock.advance(600.0)
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

    episode_id = (
        f"deadline-campaign:E2:{task.task_id}:repeat-0:seed-42"
    )
    runtime = tmp_path / "runtime"
    logs = EpisodeEventLogs(
        runtime,
        episode_id=episode_id,
        include_memory=False,
        system_id="E2",
        task_id=task.task_id,
        repeat_id=0,
        matched_seed=42,
        timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
    )
    summary = _runner(
        clock,
        system_id=SystemID.E2,
        action_predictor=predict,
        transition_predictor=diagnose,
        recovery_predictor=late_recovery_assessment,
        adapter=RecoveryDiagnosticAdapter(scenario),
        event_logs=logs,
    ).run(task, repeat_id=0, model_seed=42)

    attempts = [
        json.loads(line)
        for line in (runtime / "recoveries.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if json.loads(line)["event_type"] == "recovery_attempt"
    ]
    assert summary.terminal_reason is TerminalReason.TIMEOUT
    assert summary.recovery_attempts == len(attempts) == 1
    assert summary.recovery_actions == 1
    assert attempts[0]["payload"]["interrupted"] is True
    assert attempts[0]["payload"]["interruption_kind"] == "EpisodeTimeout"
    assert attempts[0]["payload"]["attempt"]["completed"] is False
