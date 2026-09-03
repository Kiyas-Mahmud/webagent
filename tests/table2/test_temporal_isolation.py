from __future__ import annotations

from dataclasses import FrozenInstanceError
import random

import pytest

from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    ExecutionResult,
    ExecutionStatus,
    Observation,
    ObservationStage,
    PreActionDecision,
    TaskSpecification,
    probability_map,
)
from web_agent.runtime.observation import ObservationBuilder
from web_agent.runtime.policy import CallablePolicyAdapter, PolicyKind, SystemPolicy
from web_agent.runtime.protocol import RuntimeProtocol, RuntimeStage, switches_for


SHA = "a" * 64


def _task() -> TaskSpecification:
    return TaskSpecification(
        task_id="task-1",
        goal="click the visible target",
        benchmark_id="fixture",
        benchmark_version="v1",
        start_state_id="s0",
    )


def _pre(*, page_settled: bool = True, environment_error: bool = False) -> Observation:
    return Observation(
        observation_id="pre-1",
        episode_id="episode-1",
        stage=ObservationStage.PRE_ACTION,
        screenshot_sha256=SHA,
        width=1280,
        height=720,
        url="fixture://state/0",
        title="fixture",
        page_state={"visible_target": "submit"},
        page_settled=page_settled,
        environment_error=environment_error,
    )


def _action_predictor(task, observation, rng: random.Random):
    return PreActionDecision(
        decision_id=f"decision-{rng.randrange(10_000)}",
        observation_id=observation.observation_id,
        action_type=ActionType.CLICK,
        action_probabilities=probability_map([item.value for item in ActionType], "CLICK"),
        bbox=(0.1, 0.1, 0.2, 0.2),
        grounding_confidence=1.0,
        confidence_before=1.0,
        input_observation_ids=(observation.observation_id,),
        policy_id="trained-policy",
        policy_version="v1",
    )


def test_pre_action_projection_is_unchanged_by_noncausal_environment_bookkeeping():
    builder = ObservationBuilder()
    first = builder.pre_action(_task(), _pre(page_settled=True, environment_error=False))
    second = builder.pre_action(_task(), _pre(page_settled=False, environment_error=True))
    assert first == second

    policy = CallablePolicyAdapter(
        policy_id="trained-policy",
        policy_version="v1",
        kind=PolicyKind.TRAINED,
        checkpoint_sha256=SHA,
        action_predictor=_action_predictor,
    )
    decision_a = policy.predict_action(_task(), first, rng=random.Random(99))
    decision_b = policy.predict_action(_task(), second, rng=random.Random(99))
    assert decision_a == decision_b


def test_stage_rng_is_system_independent_but_campaign_bound():
    first = RuntimeProtocol(protocol_id="p", campaign_id="campaign-a").rng_factory()
    second = RuntimeProtocol(protocol_id="p", campaign_id="campaign-a").rng_factory()
    different_campaign = RuntimeProtocol(protocol_id="p", campaign_id="campaign-b").rng_factory()
    key = {
        "task_id": "task-1",
        "repeat_id": 0,
        "matched_seed": 42,
        "stage": RuntimeStage.PRE_ACTION,
        "step_index": 3,
    }
    assert first.seed_for(**key) == second.seed_for(**key)
    assert first.seed_for(**key) != different_campaign.seed_for(**key)
    assert first.random(**key).random() == second.random(**key).random()


def test_e1_e2_e3_pre_action_outputs_match_with_shared_stage_rng():
    task = _task()
    observation = ObservationBuilder().pre_action(task, _pre())
    adapter = CallablePolicyAdapter(
        policy_id="trained-policy",
        policy_version="v1",
        kind=PolicyKind.TRAINED,
        checkpoint_sha256=SHA,
        action_predictor=_action_predictor,
        # These paths are required by the E2/E3 contract but are deliberately
        # not invoked during this pre-action equivalence test.
        transition_predictor=lambda *_: None,
        recovery_predictor=lambda *_: None,
    )
    factory = RuntimeProtocol(
        protocol_id="table2-pilot-v1",
        campaign_id="campaign-a",
    ).rng_factory()
    rng_key = {
        "task_id": task.task_id,
        "repeat_id": 0,
        "matched_seed": 42,
        "stage": RuntimeStage.PRE_ACTION,
        "decision_index": 3,
    }

    decisions = [
        SystemPolicy(adapter, switches_for(system_id)).predict_action(
            task,
            observation,
            rng=factory.random(**rng_key),
        )
        for system_id in ("E1", "E2", "E3")
    ]

    assert decisions[0] == decisions[1] == decisions[2]


def test_post_action_transition_requires_the_actual_executed_action_reference():
    builder = ObservationBuilder()
    action = ConcreteAction(
        action_id="action-1",
        source_decision_id="decision-1",
        action_type=ActionType.CLICK,
        parameters={"target_x": 0.2, "target_y": 0.2},
        bbox=(0.1, 0.1, 0.2, 0.2),
    )
    post = Observation(
        observation_id="post-1",
        episode_id="episode-1",
        stage=ObservationStage.POST_ACTION,
        screenshot_sha256="b" * 64,
        page_state={"visible_target": "done"},
        prior_action_id="action-1",
    )
    execution = ExecutionResult(
        action_id=action.action_id,
        status=ExecutionStatus.EXECUTED,
        executor_step=1,
        state_changed=True,
    )
    transition = builder.transition(_task(), _pre(), action, execution, post)
    assert transition.executed_action.action_id == "action-1"
    assert transition.execution_result.status is ExecutionStatus.EXECUTED
    assert transition.pre_observation.observation_id == "pre-1"
    assert transition.post_observation.observation_id == "post-1"


def test_causal_history_exposes_only_completed_hash_bound_evidence():
    builder = ObservationBuilder()
    action = ConcreteAction(
        action_id="private-action",
        source_decision_id="decision-1",
        action_type=ActionType.CLICK,
        parameters={
            "target_x": 0.2,
            "target_y": 0.2,
            "private_selector": "customer-secret-target",
        },
        bbox=(0.1, 0.1, 0.2, 0.2),
    )
    execution = ExecutionResult(
        action_id=action.action_id,
        status=ExecutionStatus.ERROR,
        executor_step=1,
        state_changed=False,
        environment_error=True,
        error_kind="observable_environment_error",
        message="private browser text must remain hidden",
    )
    post = Observation(
        observation_id="post-private",
        episode_id="episode-1",
        stage=ObservationStage.POST_ACTION,
        screenshot_sha256="b" * 64,
        page_state={"visible_target": "done"},
        prior_action_id=action.action_id,
    )

    transition = builder.transition(_task(), _pre(), action, execution, post)

    assert transition.pre_observation.causal_history == ()
    assert type(transition.post_observation.causal_history) is tuple
    assert len(transition.post_observation.causal_history) == 1
    entry = transition.post_observation.causal_history[0]
    assert entry.schema_version == "table2.causal-history.v1"
    assert entry.action_type is ActionType.CLICK
    assert entry.action_target_fingerprint == action.fingerprint
    assert entry.execution_status is ExecutionStatus.ERROR
    assert entry.executor_step == 1
    assert entry.state_changed is False
    assert entry.environment_error is True
    assert entry.execution_error_sha256 is not None
    assert entry.post_observation_sha256 == post.record_sha256
    assert entry.post_screenshot_sha256 == post.screenshot_sha256

    serialized = entry.to_json().lower()
    assert "customer-secret-target" not in serialized
    assert "private browser text" not in serialized
    assert "parameters" not in serialized
    assert "message" not in serialized
    assert "oracle" not in serialized
    assert "verifier" not in serialized
    with pytest.raises(FrozenInstanceError):
        entry.executor_step = 2  # type: ignore[misc]
