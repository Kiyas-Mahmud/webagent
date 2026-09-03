from dataclasses import replace
import json

import pytest

from web_agent.benchmarks.fixture import (
    DeterministicFixtureAdapter,
    FixtureScenario,
    FixtureState,
    run_all_systems_fixture_smoke,
)
from web_agent.runtime.action_parameters import DeterministicParameterProvider
from web_agent.runtime.contracts import (
    ActionType,
    EpisodeSummary,
    PreActionDecision,
    RecoveryStrategy,
    SystemID,
    TaskSpecification,
    TerminalReason,
    TransitionAssessment,
    canonical_sha256,
    probability_map,
)
from web_agent.runtime.episode import EpisodeRunner
from web_agent.runtime.event_log import EpisodeEventLogs
from web_agent.runtime.executor import Executor
from web_agent.runtime.observation import ObservationBuilder
from web_agent.runtime.policy import CallablePolicyAdapter, PolicyKind, SystemPolicy
from web_agent.runtime.protocol import RuntimeProtocol, switches_for
from web_agent.runtime.recovery.controller import RecoveryController


SHA = "a" * 64
BBOX = (0.10, 0.10, 0.20, 0.10)


_ACTIONS = {
    "s0": (ActionType.CLICK, {}, BBOX),
    "s1": (ActionType.TYPE, {"text": "fixture"}, BBOX),
    "s2": (
        ActionType.SELECT,
        {
            "option": "fixture-option",
            "candidate_options": ["fixture-option"],
        },
        BBOX,
    ),
    "s3": (ActionType.SCROLL, {"direction": "down"}, None),
    "s4": (ActionType.NAVIGATE, {"url": "fixture://table2/final"}, None),
    "s5": (ActionType.PRESS_KEY, {"key": "ENTER"}, None),
}


def _task(task_id: str = "fixture-six-actions") -> TaskSpecification:
    return TaskSpecification(
        task_id=task_id,
        goal="complete the deterministic fixture",
        benchmark_id="fixture",
        benchmark_version="v1",
        start_state_id="s0",
        metadata={"duplicate_cluster_ids": ["fixture-evaluation-cluster"]},
    )


def _policy() -> SystemPolicy:
    def predict(task, observation, rng):
        del task, rng
        action_type, hints, bbox = _ACTIONS[
            str(observation.current_page_state["state_id"])
        ]
        return PreActionDecision(
            decision_id=f"decision:{observation.observation_id}",
            observation_id=observation.observation_id,
            action_type=action_type,
            action_probabilities=probability_map(
                tuple(item.value for item in ActionType), action_type.value
            ),
            bbox=bbox,
            grounding_confidence=1.0,
            confidence_before=1.0,
            input_observation_ids=(observation.observation_id,),
            policy_id="trained-fixture-policy",
            policy_version="v1",
            parameter_hints=hints,
        )

    adapter = CallablePolicyAdapter(
        policy_id="trained-fixture-policy",
        policy_version="v1",
        kind=PolicyKind.TRAINED,
        checkpoint_sha256=SHA,
        action_predictor=predict,
    )
    return SystemPolicy(adapter, switches_for(SystemID.E1))


def _runner(adapter: DeterministicFixtureAdapter) -> EpisodeRunner:
    protocol = RuntimeProtocol(
        protocol_id="table2-pilot-v1",
        campaign_id="fixture-smoke",
        provider_id="deterministic-parameter-provider",
        provider_version="v1",
        benchmark_id="table2-fixture",
        benchmark_version="v1",
    )
    return EpisodeRunner(
        protocol=protocol,
        system_policy=_policy(),
        provider=DeterministicParameterProvider(),
        executor=Executor(adapter, budgets=protocol.budgets),
    )


def test_episode_runner_completes_six_action_fixture_with_exact_accounting() -> None:
    summary = _runner(DeterministicFixtureAdapter()).run(
        _task(), repeat_id=0, model_seed=42
    )
    assert isinstance(summary, EpisodeSummary)
    assert summary.system_id is SystemID.E1
    assert summary.terminal_reason is TerminalReason.OPAQUE_VERIFIER_TERMINAL
    assert summary.valid_for_primary is True
    assert summary.executor_steps == summary.normal_actions == 6
    assert summary.recovery_actions == summary.memory_queries == 0


def test_episode_runner_threads_only_completed_history_into_each_pre_action() -> None:
    seen = []
    base_policy = _policy()
    base_adapter = base_policy.adapter

    def capture(task, observation, rng):
        seen.append(observation)
        return base_adapter.action_predictor(task, observation, rng)

    runner = _runner(DeterministicFixtureAdapter())
    runner.system_policy = SystemPolicy(
        replace(base_adapter, action_predictor=capture),
        base_policy.switches,
    )
    summary = runner.run(_task(), repeat_id=0, model_seed=42)

    assert summary.executor_steps == 6
    assert [len(item.causal_history) for item in seen] == list(range(6))
    assert seen[0].causal_history == ()
    for decision_index, observation in enumerate(seen[1:], start=1):
        history = observation.causal_history
        assert history[-1].post_observation_id == observation.observation_id
        assert tuple(item.history_index for item in history) == tuple(
            range(1, decision_index + 1)
        )
        assert tuple(item.executor_step for item in history) == tuple(
            range(1, decision_index + 1)
        )


def test_successful_reset_invalidates_episode_instead_of_granting_free_success() -> None:
    scenario = FixtureScenario(
        scenario_id="already-complete",
        task_id="already-complete",
        initial_state_id="done",
        states=(
            FixtureState(
                state_id="done",
                url="fixture://done",
                title="Done",
                observable={"state_id": "done"},
                success=True,
            ),
        ),
        transitions=(),
    )
    summary = _runner(
        DeterministicFixtureAdapter({scenario.task_id: scenario})
    ).run(_task(scenario.task_id), repeat_id=0, model_seed=42)
    assert summary.valid_for_primary is False
    assert summary.reset_already_success is True
    assert summary.terminal_reason is TerminalReason.RESET_ALREADY_SUCCESS
    assert summary.executor_steps == 0


def test_observable_error_state_is_scored_and_never_authorizes_infrastructure_rerun() -> None:
    scenario = FixtureScenario(
        scenario_id="observable-error",
        task_id="observable-error",
        initial_state_id="s0",
        states=(
            FixtureState(
                state_id="s0",
                url="fixture://observable-error",
                title="Visible browser error",
                observable={"state_id": "s0", "error_page": True},
                environment_failure=True,
            ),
        ),
        transitions=(),
    )

    summary = _runner(
        DeterministicFixtureAdapter({scenario.task_id: scenario})
    ).run(_task(scenario.task_id), repeat_id=0, model_seed=42)

    # The boolean observation/execution flag is causal agent evidence.  It
    # causes ordinary failed requests and eventually the registered loop
    # terminal; it is not the typed exception required for a paired-block rerun.
    assert summary.terminal_reason is TerminalReason.LOOP
    assert summary.valid_for_primary is True
    assert summary.environment_failure is False
    assert summary.executor_steps == 3


def test_registered_smoke_runs_complete_matched_e0_e3_packages(tmp_path) -> None:
    summaries = run_all_systems_fixture_smoke(tmp_path / "smoke")
    assert set(summaries) == set(SystemID)
    assert all(summary.executor_steps == 6 for summary in summaries.values())
    assert all(
        summary.terminal_reason is TerminalReason.OPAQUE_VERIFIER_TERMINAL
        for summary in summaries.values()
    )
    assert summaries[SystemID.E2].memory_queries == 0
    assert summaries[SystemID.E3].memory_queries == 0
    for system_id in SystemID:
        runtime = tmp_path / "smoke" / system_id.value / "runtime"
        assert (runtime / "actions.jsonl").is_file()
        assert (runtime / "terminal_signals.jsonl").is_file()


def test_episode_runner_logs_hash_bound_pre_redaction_contract_receipt(
    tmp_path,
) -> None:
    task = _task()
    episode_id = f"fixture-smoke:E1:{task.task_id}:repeat-0:seed-42"
    runtime = tmp_path / "runtime"
    logs = EpisodeEventLogs(
        runtime,
        episode_id=episode_id,
        include_memory=False,
        system_id="E1",
        task_id=task.task_id,
        repeat_id=0,
        matched_seed=42,
        timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
    )
    runner = _runner(DeterministicFixtureAdapter())
    runner.event_logs = logs

    summary = runner.run(task, repeat_id=0, model_seed=42)
    action_records = [
        json.loads(line)
        for line in (runtime / "actions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    first_action = action_records[0]["payload"]
    assert first_action["parameter_resolution"]["resolved"] is True
    assert first_action["parameter_resolution"]["latency_ms"] >= 0.0
    assert first_action["resolution_attempts"][0]["source"] == "deterministic"
    assert first_action["execution"]["evidence"]["status"] == "executed"
    assert first_action["execution"]["evidence"]["started_at_utc"]
    assert first_action["execution"]["evidence"]["ended_at_utc"]
    records = [
        json.loads(line)
        for line in (runtime / "environment_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    receipt_record = records[-1]
    assert receipt_record["event_type"] == "episode_contract_validation"
    receipt = receipt_record["payload"]
    assert receipt == runner.contract_validation_receipt
    assert receipt["summary_sha256"] == summary.record_sha256
    assert receipt["causal_log_snapshot_sha256"] == summary.event_log_sha256
    assert canonical_sha256(receipt["validation"]) == receipt["validation_sha256"]
    assert receipt["validation"]["status"] == "PASS"
    assert receipt["validation"]["observations"] == 7
    assert receipt["validation"]["actions"] == 6
    assert receipt["validation"]["executions"] == 6
    serialized = json.dumps(receipt).lower()
    assert "page_state" not in serialized
    assert "parameters" not in serialized
    assert "screenshot_path" not in serialized


def test_episode_runner_rejects_tampered_embedded_transition_before_return() -> None:
    class TamperedTransitionBuilder(ObservationBuilder):
        def transition(
            self,
            task,
            before,
            action,
            execution,
            after,
            *,
            causal_history=(),
        ):
            transition = super().transition(
                task,
                before,
                action,
                execution,
                after,
                causal_history=causal_history,
            )
            return replace(
                transition,
                post_observation=replace(
                    transition.post_observation,
                    current_page_state={"state_id": "causally-forged"},
                ),
            )

    base = _policy().adapter

    def assess(task, transition, rng):
        del task, rng
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

    def unexpected_recovery_assessment(task, transition, rng):
        del task, transition, rng
        raise AssertionError("non-failure transition must not enter recovery")

    policy = SystemPolicy(
        CallablePolicyAdapter(
            policy_id=base.policy_id,
            policy_version=base.policy_version,
            kind=PolicyKind.TRAINED,
            checkpoint_sha256=SHA,
            action_predictor=base.action_predictor,
            transition_predictor=assess,
            recovery_predictor=unexpected_recovery_assessment,
        ),
        switches_for(SystemID.E2),
    )
    protocol = RuntimeProtocol(
        protocol_id="table2-pilot-v1",
        campaign_id="fixture-smoke",
        provider_id="deterministic-parameter-provider",
        provider_version="v1",
        benchmark_id="table2-fixture",
        benchmark_version="v1",
    )
    runner = EpisodeRunner(
        protocol=protocol,
        system_policy=policy,
        provider=DeterministicParameterProvider(),
        executor=Executor(DeterministicFixtureAdapter(), budgets=protocol.budgets),
        recovery_controller=RecoveryController(protocol.budgets),
        observation_builder=TamperedTransitionBuilder(),
    )
    with pytest.raises(ValueError, match="canonical observation projection"):
        runner.run(_task(), repeat_id=0, model_seed=42)
    assert runner.contract_validation_receipt is None


def test_deferred_receipt_is_not_logged_for_outer_reset_invalidation(tmp_path) -> None:
    scenario = FixtureScenario(
        scenario_id="deferred-already-complete",
        task_id="deferred-already-complete",
        initial_state_id="done",
        states=(
            FixtureState(
                state_id="done",
                url="fixture://done",
                title="Done",
                observable={"state_id": "done"},
                success=True,
            ),
        ),
        transitions=(),
    )
    task = _task(scenario.task_id)
    episode_id = f"fixture-smoke:E1:{task.task_id}:repeat-0:seed-42"
    runtime = tmp_path / "deferred-runtime"
    logs = EpisodeEventLogs(
        runtime,
        episode_id=episode_id,
        include_memory=False,
        system_id="E1",
        task_id=task.task_id,
        repeat_id=0,
        matched_seed=42,
        timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
    )
    protocol = RuntimeProtocol(
        protocol_id="table2-pilot-v1",
        campaign_id="fixture-smoke",
        provider_id="deterministic-parameter-provider",
        provider_version="v1",
        benchmark_id="table2-fixture",
        benchmark_version="v1",
    )
    runner = EpisodeRunner(
        protocol=protocol,
        system_policy=_policy(),
        provider=DeterministicParameterProvider(),
        executor=Executor(
            DeterministicFixtureAdapter({scenario.task_id: scenario}),
            budgets=protocol.budgets,
        ),
        event_logs=logs,
        defer_contract_validation_receipt=True,
    )

    summary = runner.run(task, repeat_id=0, model_seed=42)
    assert summary.reset_already_success is True
    assert runner.contract_validation_receipt is not None
    environment = [
        json.loads(line)
        for line in (runtime / "environment_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(
        row["event_type"] != "episode_contract_validation"
        for row in environment
    )
