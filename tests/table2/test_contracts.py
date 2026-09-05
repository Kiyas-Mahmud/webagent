from dataclasses import replace

import pytest

from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    EpisodeContractBundle,
    EpisodeSummary,
    ExecutionEvidence,
    ExecutionResult,
    ExecutionStatus,
    MAX_EXECUTION_MESSAGE_CHARS,
    MemoryQuery,
    MemoryQueryResult,
    Observation,
    ObservationStage,
    OpaqueTerminalSignal,
    ParameterResolutionAttempt,
    ParameterResolutionTrace,
    PolicyObservation,
    PreActionDecision,
    PreActionParseRejection,
    RecoveryAssessment,
    RecoveryAttempt,
    RecoveryDecision,
    RecoveryStrategy,
    RecoveryTransitionInput,
    SystemID,
    TerminalReason,
    TransitionInput,
    TransitionAssessment,
    VerifierReceiptBinding,
    VersionedRecord,
    completed_causal_history_entry,
    probability_map,
    validate_episode_foreign_keys,
)


SHA = "a" * 64


def test_provider_and_execution_evidence_fail_closed_on_internal_contradictions() -> None:
    rejected = ParameterResolutionAttempt(
        source="deterministic",
        status="REJECTED",
        latency_ms=1.0,
        reason="missing text",
    )
    with pytest.raises(ValueError, match="below measured"):
        ParameterResolutionTrace(
            provider_id="hybrid",
            provider_version="v1",
            attempts=(rejected,),
            resolved=False,
            resolution_source=None,
            latency_ms=0.5,
        )
    evidence = ExecutionEvidence(
        action_id="action-1",
        started_at_utc="2025-01-01T00:00:00+00:00",
        ended_at_utc="2025-01-01T00:00:01+00:00",
        status=ExecutionStatus.EXECUTED,
    )
    with pytest.raises(ValueError, match="contradicts"):
        ExecutionResult(
            action_id="action-1",
            status=ExecutionStatus.ERROR,
            executor_step=1,
            state_changed=False,
            evidence=evidence,
        )
    with pytest.raises(ValueError, match="registered bound"):
        ExecutionResult(
            action_id="action-1",
            status=ExecutionStatus.REJECTED,
            executor_step=1,
            state_changed=False,
            message="x" * (MAX_EXECUTION_MESSAGE_CHARS + 1),
        )


def _bundle() -> EpisodeContractBundle:
    episode_id = "campaign:E1:fixture:repeat-0:seed-42"
    before = Observation(
        observation_id="obs-before",
        episode_id=episode_id,
        stage=ObservationStage.RESET,
        screenshot_sha256=SHA,
        page_state={"state_id": "before"},
    )
    decision = PreActionDecision(
        decision_id="decision-1",
        observation_id=before.observation_id,
        action_type=ActionType.CLICK,
        action_probabilities=probability_map(
            tuple(item.value for item in ActionType), ActionType.CLICK.value
        ),
        bbox=(0.1, 0.1, 0.2, 0.2),
        grounding_confidence=1.0,
        confidence_before=1.0,
        input_observation_ids=(before.observation_id,),
        policy_id="fixture-policy",
        policy_version="v1",
    )
    action = ConcreteAction(
        action_id="action-1",
        source_decision_id=decision.decision_id,
        action_type=decision.action_type,
        parameters={"target_x": 0.2, "target_y": 0.2},
        bbox=decision.bbox,
    )
    after = Observation(
        observation_id="obs-after",
        episode_id=episode_id,
        stage=ObservationStage.POST_ACTION,
        screenshot_sha256="b" * 64,
        page_state={"state_id": "after"},
        prior_action_id=action.action_id,
    )
    execution = ExecutionResult(
        action_id=action.action_id,
        status=ExecutionStatus.EXECUTED,
        executor_step=1,
        state_changed=True,
    )
    history_entry = completed_causal_history_entry(
        history_index=1,
        action=action,
        execution=execution,
        post_observation=after,
    )
    before_policy = PolicyObservation(
        task_id="fixture",
        goal="complete fixture",
        observation_id=before.observation_id,
        screenshot_sha256=before.screenshot_sha256,
        screenshot_path=None,
        width=before.width,
        height=before.height,
        url=before.url,
        title=before.title,
        current_page_state=before.page_state,
    )
    after_policy = replace(
        before_policy,
        observation_id=after.observation_id,
        screenshot_sha256=after.screenshot_sha256,
        current_page_state=after.page_state,
        causal_history=(history_entry,),
    )
    return EpisodeContractBundle(
        episode_id=episode_id,
        observations=(before, after),
        decisions=(decision,),
        actions=(action,),
        executions=(execution,),
        transition_inputs=(
            TransitionInput(
                task_id="fixture",
                pre_observation=before_policy,
                executed_action=action,
                execution_result=execution,
                post_observation=after_policy,
            ),
        ),
        transition_assessments=(
            TransitionAssessment(
                assessment_id="assessment-1",
                pre_observation_id=before.observation_id,
                post_observation_id=after.observation_id,
                executed_action_id=action.action_id,
                predicted_failure=False,
                failure_probability=0.0,
                failure_type="NONE",
                failure_type_probabilities={"NONE": 1.0},
                needs_recovery=False,
                needs_recovery_probability=0.0,
                recovery_strategy=RecoveryStrategy.NONE,
                recovery_probabilities={RecoveryStrategy.NONE.value: 1.0},
            ),
        ),
        recovery_decisions=(),
        recovery_attempts=(),
        recovery_transitions=(),
        recovery_assessments=(),
        memory_queries=(),
        memory_results=(),
        parse_rejections=(),
        summary=EpisodeSummary(
            episode_id=episode_id,
            protocol_id="table2-pilot-v1",
            system_id=SystemID.E2,
            task_id="fixture",
            repeat_id=0,
            model_seed=42,
            valid_for_primary=True,
            terminal_reason=TerminalReason.CLOSED,
            executor_steps=1,
            normal_actions=1,
            recovery_actions=0,
            recovery_attempts=0,
            failure_incidents=0,
            memory_queries=0,
            memory_interventions=0,
            elapsed_seconds=0.1,
        ),
    )


def test_probability_contracts_reject_missing_nonfinite_and_unnormalized_evidence():
    bundle = _bundle()
    decision = bundle.decisions[0]
    assessment = bundle.transition_assessments[0]

    with pytest.raises(ValueError, match="class set differs"):
        replace(decision, action_probabilities={ActionType.CLICK.value: 1.0})
    with pytest.raises(ValueError, match="must be in "):
        replace(
            decision,
            action_probabilities={
                **decision.action_probabilities,
                ActionType.CLICK.value: float("nan"),
            },
        )
    with pytest.raises(ValueError, match="must be a non-empty"):
        replace(assessment, failure_type_probabilities={})
    with pytest.raises(ValueError, match="must be in "):
        replace(assessment, failure_type_probabilities={"NONE": 2.0})
    with pytest.raises(ValueError, match="unknown labels"):
        replace(assessment, recovery_probabilities={"ORACLE_RESCUE": 1.0})
    with pytest.raises(ValueError, match="must sum to 1"):
        replace(
            assessment,
            recovery_probabilities={
                RecoveryStrategy.NONE.value: 0.4,
                RecoveryStrategy.RETRY.value: 0.4,
            },
        )


def _two_action_bundle() -> EpisodeContractBundle:
    bundle = _bundle()
    previous = bundle.observations[-1]
    decision = PreActionDecision(
        decision_id="decision-2",
        observation_id=previous.observation_id,
        action_type=ActionType.CLICK,
        action_probabilities=probability_map(
            tuple(item.value for item in ActionType), ActionType.CLICK.value
        ),
        bbox=(0.3, 0.3, 0.2, 0.2),
        grounding_confidence=1.0,
        confidence_before=1.0,
        input_observation_ids=(previous.observation_id,),
        policy_id="fixture-policy",
        policy_version="v1",
    )
    action = ConcreteAction(
        action_id="action-2",
        source_decision_id=decision.decision_id,
        action_type=decision.action_type,
        parameters={"target_x": 0.4, "target_y": 0.4},
        bbox=decision.bbox,
    )
    after = Observation(
        observation_id="obs-final",
        episode_id=bundle.episode_id,
        stage=ObservationStage.POST_ACTION,
        screenshot_sha256="c" * 64,
        page_state={"state_id": "final"},
        prior_action_id=action.action_id,
    )
    pre_policy = bundle.transition_inputs[-1].post_observation
    execution = ExecutionResult(
        action_id=action.action_id,
        status=ExecutionStatus.EXECUTED,
        executor_step=2,
        state_changed=True,
    )
    history_entry = completed_causal_history_entry(
        history_index=2,
        action=action,
        execution=execution,
        post_observation=after,
    )
    post_policy = replace(
        pre_policy,
        observation_id=after.observation_id,
        screenshot_sha256=after.screenshot_sha256,
        current_page_state=after.page_state,
        causal_history=(*pre_policy.causal_history, history_entry),
    )
    transition = TransitionInput(
        task_id=bundle.summary.task_id,
        pre_observation=pre_policy,
        executed_action=action,
        execution_result=execution,
        post_observation=post_policy,
    )
    assessment = TransitionAssessment(
        assessment_id="assessment-2",
        pre_observation_id=pre_policy.observation_id,
        post_observation_id=post_policy.observation_id,
        executed_action_id=action.action_id,
        predicted_failure=False,
        failure_probability=0.0,
        failure_type="NONE",
        failure_type_probabilities={"NONE": 1.0},
        needs_recovery=False,
        needs_recovery_probability=0.0,
        recovery_strategy=RecoveryStrategy.NONE,
        recovery_probabilities={RecoveryStrategy.NONE.value: 1.0},
    )
    return replace(
        bundle,
        observations=(*bundle.observations, after),
        decisions=(*bundle.decisions, decision),
        actions=(*bundle.actions, action),
        executions=(
            *bundle.executions,
            execution,
        ),
        transition_inputs=(*bundle.transition_inputs, transition),
        transition_assessments=(*bundle.transition_assessments, assessment),
        summary=replace(
            bundle.summary,
            executor_steps=2,
            normal_actions=2,
        ),
    )


def _recovery_bundle() -> EpisodeContractBundle:
    bundle = _bundle()
    recovery_decision = RecoveryDecision(
        decision_id="recovery-decision-1",
        incident_id="incident-1",
        strategy=RecoveryStrategy.RETRY,
        trigger_sources=("policy",),
        diagnosis="NO_EFFECT",
    )
    recovery_action = ConcreteAction(
        action_id="recovery-action-1",
        source_decision_id=recovery_decision.decision_id,
        action_type=ActionType.CLICK,
        parameters={"target_x": 0.2, "target_y": 0.2},
        bbox=(0.1, 0.1, 0.2, 0.2),
        recovery_attempt_id="attempt-1",
    )
    after = Observation(
        observation_id="obs-recovered",
        episode_id=bundle.episode_id,
        stage=ObservationStage.POST_RECOVERY,
        screenshot_sha256="c" * 64,
        page_state={"state_id": "recovered"},
        prior_action_id=recovery_action.action_id,
    )
    pre_policy = bundle.transition_inputs[-1].post_observation
    recovery_execution = ExecutionResult(
        action_id=recovery_action.action_id,
        status=ExecutionStatus.EXECUTED,
        executor_step=2,
        state_changed=True,
    )
    history_entry = completed_causal_history_entry(
        history_index=2,
        action=recovery_action,
        execution=recovery_execution,
        post_observation=after,
    )
    post_policy = replace(
        pre_policy,
        observation_id=after.observation_id,
        screenshot_sha256=after.screenshot_sha256,
        current_page_state=after.page_state,
        causal_history=(*pre_policy.causal_history, history_entry),
    )
    recovery_transition = RecoveryTransitionInput(
        task_id=bundle.summary.task_id,
        incident_id=recovery_decision.incident_id,
        attempt_id="attempt-1",
        pre_recovery_observation=pre_policy,
        recovery_actions=(recovery_action,),
        post_recovery_observation=post_policy,
    )
    recovery_assessment = RecoveryAssessment(
        assessment_id="recovery-assessment-1",
        incident_id=recovery_decision.incident_id,
        attempt_id="attempt-1",
        pre_recovery_observation_id=pre_policy.observation_id,
        post_recovery_observation_id=post_policy.observation_id,
        recovery_action_ids=(recovery_action.action_id,),
        predicted_failure_resolved=True,
        predicted_resolution_probability=1.0,
        predicted_progress=True,
        predicted_progress_probability=1.0,
    )
    attempt = RecoveryAttempt(
        attempt_id="attempt-1",
        incident_id=recovery_decision.incident_id,
        strategy=recovery_decision.strategy,
        incident_attempt_index=1,
        episode_attempt_index=1,
        action_ids=(recovery_action.action_id,),
        completed=True,
        predicted_assessment_id=recovery_assessment.assessment_id,
    )
    return replace(
        bundle,
        observations=(*bundle.observations, after),
        recovery_decisions=(recovery_decision,),
        actions=(*bundle.actions, recovery_action),
        executions=(
            *bundle.executions,
            recovery_execution,
        ),
        recovery_attempts=(attempt,),
        recovery_transitions=(recovery_transition,),
        recovery_assessments=(recovery_assessment,),
        summary=replace(
            bundle.summary,
            executor_steps=2,
            recovery_actions=1,
            recovery_attempts=1,
            failure_incidents=1,
        ),
    )


def test_versioned_contract_round_trip_is_typed_and_lossless() -> None:
    original = _bundle()
    restored = VersionedRecord.from_dict(original.to_dict())
    assert restored == original
    assert isinstance(restored, EpisodeContractBundle)
    assert restored.decisions[0].action_type is ActionType.CLICK


def test_contract_decoder_rejects_unknown_fields() -> None:
    payload = _bundle().to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        EpisodeContractBundle.from_dict(payload)


def test_episode_foreign_keys_reconcile_all_runtime_counts() -> None:
    result = validate_episode_foreign_keys(_bundle())
    assert result.status == "PASS"
    assert result.actions == result.executions == 1


def test_episode_foreign_keys_reject_missing_decision() -> None:
    malformed = replace(_bundle(), decisions=())
    with pytest.raises(ValueError, match="missing decision"):
        validate_episode_foreign_keys(malformed)


def test_episode_foreign_keys_reject_altered_embedded_action_with_same_id() -> None:
    bundle = _bundle()
    transition = bundle.transition_inputs[0]
    altered = replace(
        transition.executed_action,
        parameters={"target_x": 0.9, "target_y": 0.9},
    )
    malformed = replace(
        bundle,
        transition_inputs=(replace(transition, executed_action=altered),),
    )
    with pytest.raises(ValueError, match="embedded action differs"):
        validate_episode_foreign_keys(malformed)


def test_episode_foreign_keys_reject_altered_policy_observation_with_same_id() -> None:
    bundle = _bundle()
    transition = bundle.transition_inputs[0]
    altered = replace(
        transition.post_observation,
        current_page_state={"state_id": "forged"},
    )
    malformed = replace(
        bundle,
        transition_inputs=(replace(transition, post_observation=altered),),
    )
    with pytest.raises(ValueError, match="canonical observation projection"):
        validate_episode_foreign_keys(malformed)


def test_episode_foreign_keys_reject_action_execution_cardinality_and_order() -> None:
    bundle = _bundle()
    extra = replace(
        bundle.actions[0],
        action_id="unexecuted-action",
    )
    with pytest.raises(ValueError, match="cardinality or order"):
        validate_episode_foreign_keys(replace(bundle, actions=(*bundle.actions, extra)))

    two_actions = _two_action_bundle()
    with pytest.raises(ValueError, match="cardinality or order"):
        validate_episode_foreign_keys(
            replace(two_actions, executions=tuple(reversed(two_actions.executions)))
        )


def test_transition_input_binds_the_exact_executor_result() -> None:
    transition = _bundle().transition_inputs[0]
    assert transition.execution_result.action_id == transition.executed_action.action_id
    assert transition.execution_result.status is ExecutionStatus.EXECUTED

    with pytest.raises(ValueError, match="execution result belongs to another action"):
        replace(
            transition,
            execution_result=replace(
                transition.execution_result,
                action_id="different-action",
            ),
        )

    tampered_bundle = replace(
        _bundle(),
        transition_inputs=(
            replace(
                transition,
                execution_result=replace(
                    transition.execution_result,
                    state_changed=False,
                ),
            ),
        ),
    )
    with pytest.raises(
        ValueError,
        match="embedded execution differs from canonical execution",
    ):
        validate_episode_foreign_keys(tampered_bundle)


def test_episode_foreign_keys_reject_tampered_or_future_causal_history() -> None:
    bundle = _bundle()
    transition = bundle.transition_inputs[0]
    canonical_entry = transition.post_observation.causal_history[0]
    tampered_post = replace(
        transition.post_observation,
        causal_history=(replace(canonical_entry, state_changed=False),),
    )
    with pytest.raises(ValueError, match="causal history differs"):
        validate_episode_foreign_keys(
            replace(
                bundle,
                transition_inputs=(
                    replace(transition, post_observation=tampered_post),
                ),
            )
        )

    forged_pre_entry = replace(
        canonical_entry,
        post_observation_id=transition.pre_observation.observation_id,
        post_observation_sha256=bundle.observations[0].record_sha256,
        post_screenshot_sha256=bundle.observations[0].screenshot_sha256,
    )
    forged_pre = replace(
        transition.pre_observation,
        causal_history=(forged_pre_entry,),
    )
    with pytest.raises(ValueError, match="causal history differs"):
        validate_episode_foreign_keys(
            replace(
                bundle,
                transition_inputs=(
                    replace(transition, pre_observation=forged_pre),
                ),
            )
        )


def test_episode_foreign_keys_reject_altered_recovery_action_with_same_id() -> None:
    bundle = _recovery_bundle()
    transition = bundle.recovery_transitions[0]
    altered = replace(
        transition.recovery_actions[0],
        parameters={"target_x": 0.7, "target_y": 0.7},
    )
    malformed = replace(
        bundle,
        recovery_transitions=(
            replace(transition, recovery_actions=(altered,)),
        ),
    )
    with pytest.raises(ValueError, match="embedded action differs"):
        validate_episode_foreign_keys(malformed)


def test_episode_foreign_keys_replay_memory_hash_and_result_cardinality() -> None:
    bundle = _bundle()
    query = MemoryQuery(
        query_id="query-1",
        task_id=bundle.summary.task_id,
        episode_id=bundle.episode_id,
        incident_id="incident-1",
        post_failure_observation_id=bundle.observations[-1].observation_id,
        failed_action_id=bundle.actions[-1].action_id,
        diagnosis="NO_EFFECT",
        duplicate_cluster_ids=("cluster-1",),
        post_failure_observation_sha256=bundle.observations[-1].record_sha256,
        post_action_input_sha256=bundle.transition_inputs[-1].record_sha256,
    )
    result = MemoryQueryResult(
        query_id=query.query_id,
        shadow_decision_sha256=SHA,
        candidate_ids=(),
        scores=(),
        exclusion_reasons={},
        admitted_candidate_id=None,
        admitted=False,
        changed_strategy=False,
        changed_target_or_parameters=False,
        final_strategy=RecoveryStrategy.RETRY,
    )
    complete = replace(
        bundle,
        memory_queries=(query,),
        memory_results=(result,),
        summary=replace(bundle.summary, memory_queries=1),
    )
    assert validate_episode_foreign_keys(complete).status == "PASS"

    with pytest.raises(ValueError, match="query/result cardinality"):
        validate_episode_foreign_keys(replace(complete, memory_results=()))
    with pytest.raises(ValueError, match="observation hash differs"):
        validate_episode_foreign_keys(
            replace(complete, memory_queries=(replace(query, post_failure_observation_sha256="b" * 64),))
        )


def test_episode_foreign_keys_reconcile_parser_rejection_without_fake_action() -> None:
    bundle = _bundle()
    rejection = PreActionParseRejection(
        request_id="parse-request-1",
        observation_id=bundle.observations[0].observation_id,
        policy_id="base-parser",
        policy_version="v1",
        decision_index=1,
        error_kind="invalid_pre_action_parser_output",
        error_sha256=SHA,
    )
    execution = ExecutionResult(
        action_id=rejection.request_id,
        status=ExecutionStatus.REJECTED,
        executor_step=1,
        state_changed=False,
        error_kind="pre_action_parse_rejected",
    )
    parsed = replace(
        bundle,
        observations=(bundle.observations[0],),
        decisions=(),
        actions=(),
        executions=(execution,),
        transition_inputs=(),
        transition_assessments=(),
        parse_rejections=(rejection,),
        summary=replace(bundle.summary, system_id=SystemID.E0),
    )
    result = validate_episode_foreign_keys(parsed)
    assert result.actions == 0
    assert result.executions == 1

    with pytest.raises(ValueError, match="parse rejection execution receipt"):
        validate_episode_foreign_keys(
            replace(
                parsed,
                executions=(replace(execution, status=ExecutionStatus.EXECUTED),),
            )
        )


@pytest.mark.parametrize("terminate", [1, 0, "false", None])
def test_opaque_terminal_signal_requires_exact_boolean(terminate) -> None:
    with pytest.raises(ValueError, match="exact boolean"):
        OpaqueTerminalSignal(
            event_id="sealed-event",
            token_sha256=SHA,
            terminate=terminate,
        )


@pytest.mark.parametrize("bad_hash", ["A" * 64, "g" * 64, "a" * 63])
def test_opaque_signal_and_receipt_hashes_require_lowercase_hex(bad_hash: str) -> None:
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        OpaqueTerminalSignal(
            event_id="sealed-event",
            token_sha256=bad_hash,
            terminate=False,
        )
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        VerifierReceiptBinding(
            receipt_kind="after_reset",
            observation_id="obs-reset",
            observation_sha256=bad_hash,
        )
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        VerifierReceiptBinding(
            receipt_kind="after_normal_action",
            observation_id="obs-post",
            observation_sha256=SHA,
            action_id="action-1",
            action_sha256=bad_hash,
        )
