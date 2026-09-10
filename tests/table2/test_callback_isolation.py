from __future__ import annotations

from dataclasses import replace
import random

import pytest

from web_agent.benchmarks.fixture import DeterministicFixtureAdapter
from web_agent.runtime.action_parameters import (
    CallableActionParameterProvider,
    ParameterResolutionError,
)
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    ExecutionResult,
    ExecutionStatus,
    Observation,
    ObservationStage,
    PolicyObservation,
    PreActionDecision,
    RecoveryAssessment,
    RecoveryDecision,
    RecoveryStrategy,
    RecoveryTransitionInput,
    TaskSpecification,
    TransitionAssessment,
    probability_map,
)
from web_agent.runtime.observation import ObservationBuilder
from web_agent.runtime.executor import Executor, ExecutorInputMutation
from web_agent.runtime.policy import CallablePolicyAdapter, PolicyError, PolicyKind
from web_agent.runtime.protocol import REGISTERED_BUDGETS
from web_agent.runtime.recovery.controller import CallableRecoveryActionPlanner


SHA = "a" * 64


def _task() -> TaskSpecification:
    return TaskSpecification(
        task_id="mutation-canary",
        goal="click the visible target",
        benchmark_id="fixture",
        benchmark_version="v1",
        start_state_id="s0",
    )


def _policy_observation(observation_id: str = "obs-pre") -> PolicyObservation:
    return PolicyObservation(
        task_id="mutation-canary",
        goal="click the visible target",
        observation_id=observation_id,
        screenshot_sha256=SHA,
        screenshot_path=None,
        width=1280,
        height=720,
        url="fixture://mutation-canary",
        title="fixture",
        current_page_state={"nested": {"value": 1}},
    )


def _decision(
    observation_id: str,
    *,
    hints=None,
) -> PreActionDecision:
    return PreActionDecision(
        decision_id=f"decision:{observation_id}",
        observation_id=observation_id,
        action_type=ActionType.CLICK,
        action_probabilities=probability_map(
            tuple(item.value for item in ActionType),
            ActionType.CLICK.value,
        ),
        bbox=(0.1, 0.1, 0.2, 0.2),
        grounding_confidence=1.0,
        confidence_before=1.0,
        input_observation_ids=(observation_id,),
        policy_id="mutation-policy",
        policy_version="v1",
        parameter_hints={} if hints is None else hints,
    )


def _adapter(**overrides) -> CallablePolicyAdapter:
    values = {
        "policy_id": "mutation-policy",
        "policy_version": "v1",
        "kind": PolicyKind.TRAINED,
        "checkpoint_sha256": SHA,
        "action_predictor": lambda task, observation, rng: _decision(
            observation.observation_id
        ),
    }
    values.update(overrides)
    return CallablePolicyAdapter(**values)


def _transition():
    action = ConcreteAction(
        action_id="action-1",
        source_decision_id="decision:obs-pre",
        action_type=ActionType.CLICK,
        parameters={"target_x": 0.2, "target_y": 0.2},
        bbox=(0.1, 0.1, 0.2, 0.2),
    )
    before = Observation(
        observation_id="obs-pre",
        episode_id="episode-1",
        stage=ObservationStage.RESET,
        screenshot_sha256=SHA,
        page_state={"nested": {"value": 1}},
    )
    after = Observation(
        observation_id="obs-post",
        episode_id="episode-1",
        stage=ObservationStage.POST_ACTION,
        screenshot_sha256="b" * 64,
        page_state={"nested": {"value": 2}},
        prior_action_id=action.action_id,
    )
    execution = ExecutionResult(
        action_id=action.action_id,
        status=ExecutionStatus.EXECUTED,
        executor_step=1,
        state_changed=True,
    )
    return ObservationBuilder().transition(
        _task(),
        before,
        action,
        execution,
        after,
    )


def test_pre_action_callback_mutation_is_rejected_and_runtime_input_is_unchanged():
    observation = _policy_observation()
    original_sha256 = observation.record_sha256
    callback_input = None

    def mutate(task, received, rng):
        nonlocal callback_input
        del task, rng
        callback_input = received
        received.current_page_state["nested"]["value"] = 99
        return _decision(received.observation_id)

    policy = _adapter(action_predictor=mutate)
    with pytest.raises(PolicyError, match="mutated protected callback input"):
        policy.predict_action(_task(), observation, rng=random.Random(1))

    assert callback_input is not observation
    assert observation.record_sha256 == original_sha256
    assert observation.current_page_state == {"nested": {"value": 1}}


def test_post_action_callback_mutation_is_rejected_and_transition_is_unchanged():
    transition = _transition()
    original_sha256 = transition.record_sha256

    def mutate(task, received, rng):
        del task, rng
        received.executed_action.parameters["target_x"] = 0.9
        return TransitionAssessment(
            assessment_id="assessment-1",
            pre_observation_id=received.pre_observation.observation_id,
            post_observation_id=received.post_observation.observation_id,
            executed_action_id=received.executed_action.action_id,
            predicted_failure=False,
            failure_probability=0.0,
            failure_type="NONE",
            failure_type_probabilities={"NONE": 1.0},
            needs_recovery=False,
            needs_recovery_probability=0.0,
            recovery_strategy=RecoveryStrategy.NONE,
            recovery_probabilities={RecoveryStrategy.NONE.value: 1.0},
        )

    policy = _adapter(transition_predictor=mutate)
    with pytest.raises(PolicyError, match="mutated protected callback input"):
        policy.assess_transition(_task(), transition, rng=random.Random(2))

    assert transition.record_sha256 == original_sha256
    assert transition.executed_action.parameters["target_x"] == 0.2


def test_recovery_callback_mutation_is_rejected_and_transition_is_unchanged():
    transition = _transition()
    recovery_action = replace(
        transition.executed_action,
        action_id="recovery-action-1",
        source_decision_id="recovery-decision-1",
        recovery_attempt_id="attempt-1",
    )
    recovery_transition = RecoveryTransitionInput(
        task_id="mutation-canary",
        incident_id="incident-1",
        attempt_id="attempt-1",
        pre_recovery_observation=transition.post_observation,
        recovery_actions=(recovery_action,),
        post_recovery_observation=replace(
            transition.post_observation,
            observation_id="obs-recovered",
            causal_history=(),
        ),
    )
    original_sha256 = recovery_transition.record_sha256

    def mutate(task, received, rng):
        del task, rng
        received.recovery_actions[0].parameters["target_x"] = 0.9
        return RecoveryAssessment(
            assessment_id="recovery-assessment-1",
            incident_id=received.incident_id,
            attempt_id=received.attempt_id,
            pre_recovery_observation_id=(
                received.pre_recovery_observation.observation_id
            ),
            post_recovery_observation_id=(
                received.post_recovery_observation.observation_id
            ),
            recovery_action_ids=(received.recovery_actions[0].action_id,),
            predicted_failure_resolved=True,
            predicted_resolution_probability=1.0,
            predicted_progress=True,
            predicted_progress_probability=1.0,
        )

    policy = _adapter(recovery_predictor=mutate)
    with pytest.raises(PolicyError, match="mutated protected callback input"):
        policy.assess_recovery(
            _task(),
            recovery_transition,
            rng=random.Random(3),
        )

    assert recovery_transition.record_sha256 == original_sha256
    assert recovery_transition.recovery_actions[0].parameters["target_x"] == 0.2


def test_policy_output_is_detached_before_downstream_consumers_receive_it():
    observation = _policy_observation()
    callback_result = _decision(
        observation.observation_id,
        hints={"nested": {"value": 1}},
    )
    policy = _adapter(
        action_predictor=lambda task, received, rng: callback_result,
    )

    returned = policy.predict_action(_task(), observation, rng=random.Random(4))
    returned_sha256 = returned.record_sha256
    callback_result.parameter_hints["nested"]["value"] = 99

    assert returned is not callback_result
    assert returned.record_sha256 == returned_sha256
    assert returned.parameter_hints == {"nested": {"value": 1}}


def test_provider_and_recovery_planner_receive_only_detached_inputs():
    observation = _policy_observation()
    decision = _decision(
        observation.observation_id,
        hints={"nested": {"value": 1}},
    )
    decision_sha256 = decision.record_sha256

    def mutate_provider(task, received, selected, rng):
        del task, received, rng
        selected.parameter_hints["nested"]["value"] = 99
        return {}

    provider = CallableActionParameterProvider(
        provider_id="frozen-base",
        provider_version="v1",
        prompt_sha256=SHA,
        resolver=mutate_provider,
    )
    with pytest.raises(
        ParameterResolutionError,
        match="mutated protected callback input",
    ):
        provider.resolve(_task(), observation, decision, rng=random.Random(5))
    assert decision.record_sha256 == decision_sha256

    failed_action = ConcreteAction(
        action_id="failed-action",
        source_decision_id=decision.decision_id,
        action_type=ActionType.CLICK,
        parameters={"target_x": 0.2, "target_y": 0.2},
        bbox=decision.bbox,
    )
    failed_sha256 = failed_action.record_sha256

    def mutate_planner(task, received, selected, failed, rng):
        del task, received, selected, rng
        failed.parameters["target_x"] = 0.9
        return replace(failed, action_id="planned-action")

    planner = CallableRecoveryActionPlanner(
        planner_id="mutation-planner",
        planner_version="v1",
        callback=mutate_planner,
    )
    with pytest.raises(ValueError, match="mutated protected callback input"):
        planner.plan(
            _task(),
            observation,
            RecoveryDecision(
                decision_id="recovery-decision-1",
                incident_id="incident-1",
                strategy=RecoveryStrategy.REPLAN,
                trigger_sources=("policy",),
                diagnosis="NO_EFFECT",
            ),
            failed_action,
            rng=random.Random(6),
        )
    assert failed_action.record_sha256 == failed_sha256


def test_browser_adapter_cannot_mutate_runtime_owned_concrete_action():
    class MutatingAdapter(DeterministicFixtureAdapter):
        def execute(self, action):
            action.parameters["target_x"] = 0.9
            return super().execute(action)

    task = TaskSpecification(
        task_id="fixture-six-actions",
        goal="complete the deterministic fixture",
        benchmark_id="fixture",
        benchmark_version="v1",
        start_state_id="s0",
    )
    action = ConcreteAction(
        action_id="action-1",
        source_decision_id="decision-1",
        action_type=ActionType.CLICK,
        parameters={"target_x": 0.2, "target_y": 0.15},
        bbox=(0.1, 0.1, 0.2, 0.1),
    )
    action_sha256 = action.record_sha256
    executor = Executor(MutatingAdapter(), budgets=REGISTERED_BUDGETS)
    executor.reset(task, episode_id="mutation-episode", seed=42)

    with pytest.raises(ExecutorInputMutation, match="mutated protected"):
        executor.execute(action)

    assert executor.steps_used == 1
    assert action.record_sha256 == action_sha256
    assert action.parameters["target_x"] == 0.2
    executor.close()
