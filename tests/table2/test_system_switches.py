from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest

from web_agent.runtime.contracts import (
    ExecutionResult,
    ExecutionStatus,
    MemoryCandidate,
    MemoryQuery,
    RecoveryStrategy,
    TransitionAssessment,
)
from web_agent.runtime.decision import DecisionCombiner
from web_agent.runtime.memory_adapter import (
    InMemoryFrozenReader,
    MemoryAdapter,
    MemoryBoundaryError,
)
from web_agent.runtime.protocol import (
    REGISTERED_RECOVERY_TRIGGER_CONFIG,
    load_protocol_bundle,
    switches_for,
)


ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 64


def _assessment() -> TransitionAssessment:
    return TransitionAssessment(
        assessment_id="assessment-1",
        pre_observation_id="pre-1",
        post_observation_id="post-1",
        executed_action_id="action-1",
        predicted_failure=True,
        failure_probability=0.9,
        failure_type="NO_EFFECT",
        failure_type_probabilities={"NO_EFFECT": 1.0},
        needs_recovery=True,
        needs_recovery_probability=0.9,
        recovery_strategy=RecoveryStrategy.RETRY,
        recovery_probabilities={"RETRY": 1.0},
    )


@pytest.mark.parametrize("signals", tuple(product((False, True), repeat=4)))
def test_registered_recovery_policy_truth_table_uses_both_booleans_and_probabilities(
    signals,
):
    predicted_failure, failure_probability_high, needs_recovery, needs_probability_high = (
        signals
    )
    assessment = replace(
        _assessment(),
        predicted_failure=predicted_failure,
        failure_probability=0.5 if failure_probability_high else 0.499999,
        needs_recovery=needs_recovery,
        needs_recovery_probability=0.5 if needs_probability_high else 0.499999,
    )
    execution = ExecutionResult(
        action_id="action-1",
        status=ExecutionStatus.EXECUTED,
        executor_step=1,
        state_changed=False,
    )
    trigger = DecisionCombiner(
        switches_for("E2"), REGISTERED_RECOVERY_TRIGGER_CONFIG
    ).recovery_trigger(
        assessment=assessment,
        execution=execution,
        loop_detected=False,
    )
    assert trigger.triggered is any(signals)
    assert ("policy" in trigger.sources) is any(signals)


def test_registered_executor_and_loop_trigger_rules_are_independent_of_policy():
    assessment = replace(
        _assessment(),
        predicted_failure=False,
        failure_probability=0.0,
        needs_recovery=False,
        needs_recovery_probability=0.0,
    )
    combiner = DecisionCombiner(
        switches_for("E2"), REGISTERED_RECOVERY_TRIGGER_CONFIG
    )
    rejected = combiner.recovery_trigger(
        assessment=assessment,
        execution=ExecutionResult(
            action_id="action-1",
            status=ExecutionStatus.REJECTED,
            executor_step=1,
            state_changed=False,
            error_kind="INVALID_PARAMETER",
        ),
        loop_detected=False,
    )
    assert rejected.sources == ("executor",)
    loop = combiner.recovery_trigger(
        assessment=assessment,
        execution=ExecutionResult(
            action_id="action-1",
            status=ExecutionStatus.EXECUTED,
            executor_step=1,
            state_changed=False,
        ),
        loop_detected=True,
    )
    assert loop.sources == ("loop_guard",)


def test_protocol_loader_accepts_only_the_registered_e0_e3_matrix():
    bundle = load_protocol_bundle(
        ROOT / "configs" / "eval" / "table2" / "protocol.yaml",
        campaign_id="pilot-fixture",
    )
    assert [item.system_id.value for item in bundle.systems] == ["E0", "E1", "E2", "E3"]
    assert bundle.system("E2").switches.recovery_controller is True
    assert bundle.system("E2").switches.memory_query is False
    assert bundle.system("E3").switches.memory_query is True
    assert bundle.system("E3").switches.evaluation_memory_write is False


def test_e2_and_e3_produce_the_same_no_memory_shadow_decision():
    execution = ExecutionResult(
        action_id="action-1",
        status=ExecutionStatus.EXECUTED,
        executor_step=1,
        state_changed=False,
    )
    shadows = []
    for system in ("E2", "E3"):
        combiner = DecisionCombiner(
            switches_for(system), REGISTERED_RECOVERY_TRIGGER_CONFIG
        )
        trigger = combiner.recovery_trigger(
            assessment=_assessment(), execution=execution, loop_detected=False
        )
        shadows.append(
            combiner.no_memory_shadow(
                trigger=trigger, assessment=_assessment(), incident_id="incident-1"
            )
        )
    assert shadows[0].strategy == shadows[1].strategy
    assert shadows[0].diagnosis == shadows[1].diagnosis


def test_e2_cannot_retrieve_and_irrelevant_e3_memory_preserves_shadow():
    reader = InMemoryFrozenReader(
        reader_id="fixture-reader",
        index_sha256=SHA,
        candidates=(
            MemoryCandidate(
                memory_id="memory-1",
                source_split="train",
                source_task_id="other-task",
                source_episode_id="other-episode",
                duplicate_cluster_id="train-cluster",
                strategy=RecoveryStrategy.BACKTRACK,
                similarity=0.1,
                memory_update_flag=True,
                verified_recovery_success=True,
                final_task_success=True,
            ),
        ),
    )
    with pytest.raises(MemoryBoundaryError):
        MemoryAdapter(switches_for("E2"), reader=reader, admission_threshold=0.5)

    combiner = DecisionCombiner(
        switches_for("E3"), REGISTERED_RECOVERY_TRIGGER_CONFIG
    )
    execution = ExecutionResult(
        action_id="action-1",
        status=ExecutionStatus.EXECUTED,
        executor_step=1,
        state_changed=False,
    )
    trigger = combiner.recovery_trigger(
        assessment=_assessment(), execution=execution, loop_detected=False
    )
    shadow = combiner.no_memory_shadow(
        trigger=trigger, assessment=_assessment(), incident_id="incident-1"
    )
    decision = MemoryAdapter(
        switches_for("E3"), reader=reader, admission_threshold=0.5
    ).apply_post_failure(
        query=MemoryQuery(
            query_id="query-1",
            task_id="task-1",
            episode_id="episode-1",
            incident_id="incident-1",
            post_failure_observation_id="post-1",
            failed_action_id="action-1",
            diagnosis="NO_EFFECT",
            duplicate_cluster_ids=("eval-cluster",),
        ),
        shadow_decision=shadow,
        rng=__import__("random").Random(5),
    )
    assert decision.query_result.admitted is False
    assert decision.final_decision == decision.shadow_decision
    assert decision.query_result.shadow_decision_sha256 == shadow.sha256
