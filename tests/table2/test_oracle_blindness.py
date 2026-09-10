from __future__ import annotations

from dataclasses import fields, replace
import inspect
import random

import pytest

from web_agent.eval.table2.sealed_verifier import assert_no_verifier_evidence
from web_agent.benchmarks.fixture import DeterministicFixtureAdapter
from web_agent.runtime.action_parameters import CallableActionParameterProvider
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    ExecutionResult,
    ExecutionStatus,
    MemoryQueryResult,
    OpaqueTerminalSignal,
    PolicyObservation,
    PreActionDecision,
    RecoveryAssessment,
    RecoveryDecision,
    RecoveryStrategy,
    RecoveryTransitionInput,
    RuntimeTaskView,
    TaskSpecification,
    TransitionAssessment,
    TransitionInput,
    probability_map,
    runtime_task_view,
)
from web_agent.runtime.decision import DecisionCombiner
from web_agent.runtime.executor import Executor
from web_agent.runtime.observation import (
    CausalBoundaryError,
    assert_oracle_blind_mapping,
    assert_sealed_truth_blind_mapping,
)
from web_agent.runtime.policy import CallablePolicyAdapter, PolicyKind
from web_agent.runtime.protocol import REGISTERED_BUDGETS
from web_agent.runtime.recovery.controller import CallableRecoveryActionPlanner


SHA = "a" * 64


def test_nested_oracle_and_future_fields_are_rejected_from_runtime_inputs():
    for payload in (
        {"oracle_success": True},
        {"nested": {"reference_trajectory": ["click secret target"]}},
        {"state_after": {"future": True}},
    ):
        with pytest.raises(CausalBoundaryError):
            assert_oracle_blind_mapping(payload)


def test_eval_guard_rejects_oracle_truth_in_any_runtime_record():
    with pytest.raises(Exception):
        assert_no_verifier_evidence(
            {"event": {"task_success": True}}, context="runtime canary"
        )


@pytest.mark.parametrize(
    "alias",
    (
        "official_task_success",
        "OFFICIAL-TASK.SUCCESS",
        "sealed_oracle_result",
        "Sealed Oracle.Result",
        "posthoc_relevance_label",
        "postHoc-Relevance.Label",
        "official_verified_failure",
        "sealed_verifier_result",
        "posthoc_recovery_success",
        "official_state_after",
        "posthoc_expected_action",
        "official_verified_agent_failure",
        "official_incident_resolved",
        "official_future_state",
        "posthoc_future_screenshot",
        "tasksuccessflag",
        "reportedtasksuccess",
        "oracleanswer",
        "verifierresult",
        "groundtruthvalue",
        "futurestatevalue",
        "expectedactionvalue",
        "reward",
        "done",
        "terminated",
        "truncated",
        "evaluator_reward",
        "officialreportedtasksuccess",
        "tasksuccessresultflag",
        "officialposthocfuturestate",
        "episode_done_flag",
        "is_done",
        "terminated_flag",
        "truncated_flag",
    ),
)
def test_prefixed_suffixed_and_punctuated_oracle_aliases_are_rejected(alias: str):
    payload = {"page_state": {"visible": [{alias: True}]}}
    with pytest.raises(CausalBoundaryError):
        assert_oracle_blind_mapping(payload)
    with pytest.raises(CausalBoundaryError):
        assert_sealed_truth_blind_mapping(payload)
    with pytest.raises(Exception):
        assert_no_verifier_evidence(payload, context="runtime alias canary")


def test_alias_guard_preserves_registered_causal_prediction_fields():
    causal_payload = {
        "executor_status": "ACCEPTED",
        "transition": {
            "predicted_failure": True,
            "failure_probability": 0.8,
            "failure_type": "NO_EFFECT",
            "needs_recovery": True,
            "recovery_strategy": "REPLAN",
            "predicted_failure_resolved": False,
            "predicted_progress": True,
            "progress_bar_text": "2 of 3 complete",
        },
    }
    assert_oracle_blind_mapping(causal_payload)
    assert_sealed_truth_blind_mapping(causal_payload)
    assert_no_verifier_evidence(causal_payload, context="causal prediction fields")


def test_runtime_guard_allows_only_exact_registered_opaque_fields():
    registered_runtime_payload = {
        "verifier_event_id": "opaque-event-id",
        "verifier_token_sha256": "a" * 64,
        "transition": {"predicted_failure_resolved": False},
    }
    assert_no_verifier_evidence(
        registered_runtime_payload,
        context="registered runtime evidence",
    )
    for leaked in (
        {"verified_failure_event_count": 1},
        {"memory_candidate": {"verified_recovery_success": True}},
        {"memory_candidate": {"final_task_success": True}},
        {"official_final_task_success": True},
    ):
        with pytest.raises(Exception):
            assert_no_verifier_evidence(
                leaked,
                context="current-task truth cannot use memory provenance names",
            )


def test_indirect_metadata_envelope_and_unicode_alias_are_rejected():
    payloads = (
        {"field_name": "task_success", "value": True},
        {"field_names": ["task_success"], "values": [True]},
        {"field": {"name": "task_success", "value": True}},
        {"field": {"metadata": {"name": "task_success", "value": True}}},
        {"feature_name": "task_success", "value": True},
        {"official_field_name": "task_success", "value": True},
        {"official_reported_field_name": "task_success", "value": True},
        {"field_name": "\u043eracle_success", "value": True},
        {"\u043eracle_success": True},  # Cyrillic small-o prefix.
    )
    for payload in payloads:
        with pytest.raises(CausalBoundaryError):
            assert_oracle_blind_mapping(payload)
        with pytest.raises(CausalBoundaryError):
            assert_sealed_truth_blind_mapping(payload)
        with pytest.raises(Exception):
            assert_no_verifier_evidence(
                payload,
                context="runtime indirect/unicode canary",
            )


def test_browser_visible_text_values_are_not_mistaken_for_control_plane_keys():
    payload = {
        "visible_controls": [
            {
                "name": "task_success",
                "text": "oracle_database_status",
                "candidate_options": ["reference_answer_input"],
            }
        ]
    }
    assert_oracle_blind_mapping(payload)
    assert_sealed_truth_blind_mapping(payload)
    assert_no_verifier_evidence(payload, context="browser visible text")


def test_token_matching_avoids_substring_collisions_but_keeps_temporal_boundary():
    benign = {
        "unexpected_action_count": 0,
        "incorrect_target_count": 0,
        "email_verification_status": "pending",
        "task_successor_id": "next-task",
        "recovery_successor_state": "ready",
        "final_task_successor_id": "still-not-an-outcome",
    }
    assert_oracle_blind_mapping(benign)
    assert_sealed_truth_blind_mapping(benign)
    assert_no_verifier_evidence(benign, context="token collision canary")

    causal_post_state = {"state_after_sha256": "a" * 64}
    with pytest.raises(CausalBoundaryError):
        assert_oracle_blind_mapping(causal_post_state)
    assert_sealed_truth_blind_mapping(causal_post_state)
    assert_no_verifier_evidence(causal_post_state, context="causal post-state hash")
    final_causal_post_state = {"final_state_after_sha256": "b" * 64}
    with pytest.raises(CausalBoundaryError):
        assert_oracle_blind_mapping(final_causal_post_state)
    assert_sealed_truth_blind_mapping(final_causal_post_state)
    assert_no_verifier_evidence(
        final_causal_post_state,
        context="final causal post-state hash",
    )


def test_runtime_decision_and_memory_contracts_have_no_oracle_or_relevance_inputs():
    trigger_parameters = set(inspect.signature(DecisionCombiner.recovery_trigger).parameters)
    assert not trigger_parameters.intersection(
        {"oracle", "verifier", "task_success", "oracle_failure", "oracle_progress"}
    )
    memory_fields = set(MemoryQueryResult.__dataclass_fields__)
    assert not memory_fields.intersection({"relevant_ids", "relevance_label", "oracle_relevance"})
    terminal_fields = set(OpaqueTerminalSignal.__dataclass_fields__)
    assert terminal_fields == {"event_id", "token_sha256", "terminate"}


def _metadata_canary_task() -> TaskSpecification:
    return TaskSpecification(
        task_id="fixture-six-actions",
        goal="complete the visible fixture",
        benchmark_id="fixture",
        benchmark_version="v1",
        start_state_id="s0",
        metadata={
            "task_partition": "recovery_diagnostic",
            "nested": {
                "future": {
                    "oracle_success": True,
                    "reference_trajectory": ["hidden evaluator action"],
                }
            },
        },
    )


def _policy_observation(observation_id: str) -> PolicyObservation:
    return PolicyObservation(
        task_id="fixture-six-actions",
        goal="complete the visible fixture",
        observation_id=observation_id,
        screenshot_sha256=SHA,
        screenshot_path=None,
        width=1280,
        height=720,
        url="fixture://state",
        title="fixture",
        current_page_state={"visible": True},
    )


def _pre_action_decision(observation_id: str) -> PreActionDecision:
    return PreActionDecision(
        decision_id="decision-1",
        observation_id=observation_id,
        action_type=ActionType.CLICK,
        action_probabilities=probability_map(
            tuple(item.value for item in ActionType), ActionType.CLICK.value
        ),
        bbox=(0.1, 0.1, 0.2, 0.2),
        grounding_confidence=1.0,
        confidence_before=1.0,
        input_observation_ids=(observation_id,),
        policy_id="safe-policy",
        policy_version="v1",
    )


def test_runtime_task_view_drops_nested_metadata_at_every_decision_callback():
    full_task = _metadata_canary_task()
    clean_task = replace(full_task, metadata={"different": {"state_after": "canary"}})
    first_view = runtime_task_view(full_task)
    second_view = runtime_task_view(clean_task)
    assert first_view == second_view
    assert first_view.record_sha256 == second_view.record_sha256
    assert tuple(item.name for item in fields(RuntimeTaskView)) == ("task_id", "goal")
    assert not hasattr(first_view, "metadata")

    received: list[RuntimeTaskView] = []

    def require_narrow(task: RuntimeTaskView) -> None:
        assert type(task) is RuntimeTaskView
        assert not hasattr(task, "metadata")
        assert task.to_dict() == {
            "schema_version": "table2.runtime.v1",
            "record_type": "RuntimeTaskView",
            "task_id": full_task.task_id,
            "goal": full_task.goal,
        }
        received.append(task)

    before = _policy_observation("obs-before")
    after = _policy_observation("obs-after")
    decision = _pre_action_decision(before.observation_id)
    action = ConcreteAction(
        action_id="action-1",
        source_decision_id=decision.decision_id,
        action_type=ActionType.CLICK,
        parameters={"target_x": 0.2, "target_y": 0.2},
        bbox=decision.bbox,
    )
    transition = TransitionInput(
        task_id=full_task.task_id,
        pre_observation=before,
        executed_action=action,
        execution_result=ExecutionResult(
            action_id=action.action_id,
            status=ExecutionStatus.EXECUTED,
            executor_step=1,
            state_changed=False,
        ),
        post_observation=after,
    )
    recovery_action = replace(
        action,
        action_id="recovery-action-1",
        source_decision_id="recovery-decision-1",
        recovery_attempt_id="attempt-1",
    )
    recovery_transition = RecoveryTransitionInput(
        task_id=full_task.task_id,
        incident_id="incident-1",
        attempt_id="attempt-1",
        pre_recovery_observation=after,
        recovery_actions=(recovery_action,),
        post_recovery_observation=_policy_observation("obs-recovered"),
    )

    def action_predictor(task, observation, rng):
        del observation, rng
        require_narrow(task)
        return decision

    def transition_predictor(task, value, rng):
        del rng
        require_narrow(task)
        return TransitionAssessment(
            assessment_id="assessment-1",
            pre_observation_id=value.pre_observation.observation_id,
            post_observation_id=value.post_observation.observation_id,
            executed_action_id=value.executed_action.action_id,
            predicted_failure=True,
            failure_probability=1.0,
            failure_type="NO_EFFECT",
            failure_type_probabilities={"NO_EFFECT": 1.0},
            needs_recovery=True,
            needs_recovery_probability=1.0,
            recovery_strategy=RecoveryStrategy.REPLAN,
            recovery_probabilities={RecoveryStrategy.REPLAN.value: 1.0},
        )

    def recovery_predictor(task, value, rng):
        del rng
        require_narrow(task)
        return RecoveryAssessment(
            assessment_id="recovery-assessment-1",
            incident_id=value.incident_id,
            attempt_id=value.attempt_id,
            pre_recovery_observation_id=(
                value.pre_recovery_observation.observation_id
            ),
            post_recovery_observation_id=(
                value.post_recovery_observation.observation_id
            ),
            recovery_action_ids=tuple(
                item.action_id for item in value.recovery_actions
            ),
            predicted_failure_resolved=True,
            predicted_resolution_probability=1.0,
            predicted_progress=True,
            predicted_progress_probability=1.0,
        )

    policy = CallablePolicyAdapter(
        policy_id="safe-policy",
        policy_version="v1",
        kind=PolicyKind.TRAINED,
        checkpoint_sha256=SHA,
        action_predictor=action_predictor,
        transition_predictor=transition_predictor,
        recovery_predictor=recovery_predictor,
    )
    assert policy.predict_action(full_task, before, rng=random.Random(1)) == decision
    policy.assess_transition(full_task, transition, rng=random.Random(2))
    policy.assess_recovery(full_task, recovery_transition, rng=random.Random(3))

    provider = CallableActionParameterProvider(
        provider_id="frozen-base",
        provider_version="v1",
        prompt_sha256=SHA,
            resolver=lambda task, observation, selected, rng: (
                require_narrow(task)
                or {
                    "target_x": 0.2,
                    "target_y": 0.2,
                    "target_bbox": [0.1, 0.1, 0.2, 0.2],
                    "button": "left",
                    "click_count": 1,
                }
            ),
    )
    provider.resolve(full_task, before, decision, rng=random.Random(4))

    def planner_callback(task, observation, selected, failed, rng):
        del observation, selected, failed, rng
        require_narrow(task)
        return ConcreteAction(
            action_id="planned-action",
            source_decision_id="recovery-decision-1",
            action_type=ActionType.CLICK,
            parameters={
                "target_x": 0.8,
                "target_y": 0.8,
                "target_bbox": [0.7, 0.7, 0.2, 0.2],
                "button": "left",
                "click_count": 1,
            },
            bbox=(0.7, 0.7, 0.2, 0.2),
        )

    planner = CallableRecoveryActionPlanner(
        planner_id="safe-planner",
        planner_version="v1",
        callback=planner_callback,
    )
    planner.plan(
        full_task,
        after,
        RecoveryDecision(
            decision_id="recovery-decision-1",
            incident_id="incident-1",
            strategy=RecoveryStrategy.REPLAN,
            trigger_sources=("policy",),
            diagnosis="NO_EFFECT",
        ),
        action,
        rng=random.Random(5),
    )
    assert len(received) == 5


def test_environment_reset_keeps_the_full_task_specification():
    full_task = _metadata_canary_task()

    class CapturingAdapter(DeterministicFixtureAdapter):
        received_task = None

        def reset(self, task, *, episode_id, seed):
            self.received_task = task
            return super().reset(task, episode_id=episode_id, seed=seed)

    adapter = CapturingAdapter()
    executor = Executor(adapter, budgets=REGISTERED_BUDGETS)
    executor.reset(full_task, episode_id="metadata-boundary", seed=42)
    assert adapter.received_task is full_task
    assert adapter.received_task.metadata["nested"]["future"]["oracle_success"] is True
    executor.close()
