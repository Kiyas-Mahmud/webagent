"""Synthetic contract tests for label-backed memory, never live outcome evidence."""
from types import SimpleNamespace as NS
from threading import Lock
import numpy as np
import pytest
import torch
from web_agent.memory.label_runtime import LabelBackedMemoryAdapter, strategy_exclusion
from web_agent.memory.label_backed_store import LabelMemoryHit
from web_agent.runtime.contracts import ActionType, ExecutionStatus, RecoveryDecision, RecoveryStrategy
from web_agent.runtime.memory_adapter import MemoryBoundaryError


def setup_adapter():
    strategies=iter(['RETRY','ALTERNATIVE_TARGET'])
    store=NS(manifest_sha256='a'*64,checkpoint_sha256='b'*64,
        query=lambda vector,**kwargs:(LabelMemoryHit('train-item','train-task','ERROR',next(strategies),.9,True),))
    runtime=NS(lock=Lock(),torch=torch,batch=NS(transition=lambda task,transition:{}),
        model=NS(memory_embedding=lambda batch:torch.ones((1,768))))
    adapter=LabelBackedMemoryAdapter(store=store,runtime=runtime,transitions={})
    shadow=RecoveryDecision(decision_id='shadow',incident_id='incident',strategy=RecoveryStrategy.REPLAN,diagnosis='ERROR',trigger_sources=('policy',))
    return adapter,shadow


def request(number):
    query=NS(incident_id='incident',query_id=f'query-{number}',post_failure_observation_id=f'post-{number}',failed_action_id=f'action-{number}',post_action_input_sha256=str(number)*64,task_id='eval-task',record_sha256='d'*64)
    transition=NS(task_id='eval-task',post_observation=NS(goal='goal'),executed_action=NS(action_id=query.failed_action_id),record_sha256=query.post_action_input_sha256)
    embedding=NS(query_id=query.query_id,post_failure_observation_id=query.post_failure_observation_id,failed_action_id=query.failed_action_id,checkpoint_sha256='b'*64,post_action_input_sha256=query.post_action_input_sha256,post_action_input=transition)
    return query,embedding


def test_distinct_queries_get_distinct_recovery_decision_ids():
    adapter,shadow=setup_adapter();decisions=[]
    for number in [1,2]:
        query,embedding=request(number)
        decisions.append(adapter.apply_post_failure(query=query,shadow_decision=shadow,rng=None,embedding_request=embedding))
    assert decisions[0].final_decision.decision_id!=decisions[1].final_decision.decision_id
    assert decisions[0].final_decision.strategy==RecoveryStrategy.RETRY
    assert decisions[1].final_decision.strategy==RecoveryStrategy.ALTERNATIVE_TARGET
    assert shadow.strategy==RecoveryStrategy.REPLAN


def test_stale_action_binding_fails_before_embedding():
    adapter,shadow=setup_adapter();query,embedding=request(2)
    embedding.failed_action_id='old-action'
    with pytest.raises(MemoryBoundaryError,match='binding mismatch'):
        adapter.apply_post_failure(query=query,shadow_decision=shadow,rng=None,embedding_request=embedding)
    assert adapter.calls==0



def transition(status=ExecutionStatus.REJECTED, history=()):
    return NS(execution_result=NS(status=status, error_kind='parameter_resolution_rejected'
                                 if status is ExecutionStatus.REJECTED else None),
              post_observation=NS(causal_history=history))


def navigation(status, changed=True):
    return NS(action_type=ActionType.NAVIGATE, execution_status=status,
              state_changed=changed, environment_error=False)


def test_rejected_navigation_is_not_evidence_for_backtracking():
    assert strategy_exclusion('BACKTRACK', transition(history=(navigation(ExecutionStatus.REJECTED),)))
    assert strategy_exclusion('BACKTRACK', transition())


def test_executed_state_changing_navigation_passes_necessary_condition():
    assert strategy_exclusion('BACKTRACK', transition(history=(navigation(ExecutionStatus.EXECUTED),))) is None
    assert strategy_exclusion('BACKTRACK', transition(history=(navigation(ExecutionStatus.EXECUTED, False),)))


@pytest.mark.parametrize('strategy', ['RETRY', 'ALTERNATIVE_TARGET'])
def test_unresolved_parameters_cannot_be_repaired_by_repeating_or_retargeting(strategy):
    assert strategy_exclusion(strategy, transition()) == 'FAILED_ACTION_PARAMETERS_UNRESOLVED'
    assert strategy_exclusion(strategy, transition(ExecutionStatus.EXECUTED)) is None


def test_replan_is_still_available_after_parameter_failure():
    assert strategy_exclusion('REPLAN', transition()) is None


def test_abstention_preserves_shadow_and_does_not_pick_lower_ranked_advice():
    adapter, shadow = setup_adapter()
    adapter.require_strategy_applicability = True
    adapter.store.query = lambda *args, **kwargs: (
        LabelMemoryHit('bad-backtrack', 'train-1', 'ERROR', 'BACKTRACK', .95, True),
        LabelMemoryHit('lower-replan', 'train-2', 'ERROR', 'REPLAN', .9, True))
    query, embedding = request(1)
    embedding.post_action_input.post_observation.causal_history = ()
    result = adapter.apply_post_failure(query=query, shadow_decision=shadow, rng=None, embedding_request=embedding)
    assert result.final_decision is shadow
    assert not result.query_result.admitted
    assert not result.query_result.changed_strategy
    assert result.query_result.candidate_ids == ('bad-backtrack', 'lower-replan')
    assert result.query_result.exclusion_reasons == {'bad-backtrack': 'BACKTRACK_NO_EXECUTED_IN_EPISODE_NAVIGATION'}


def test_applicable_candidate_is_still_allowed_to_intervene():
    adapter, shadow = setup_adapter()
    adapter.require_strategy_applicability = True
    query, embedding = request(1)
    embedding.post_action_input.execution_result = transition(ExecutionStatus.EXECUTED).execution_result
    result = adapter.apply_post_failure(query=query, shadow_decision=shadow, rng=None, embedding_request=embedding)
    assert result.query_result.admitted and result.query_result.changed_strategy
    assert result.final_decision.strategy.value == 'RETRY'
