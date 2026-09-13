"""H-memory engineering fixtures; no trained model inference or live outcomes."""
from dataclasses import replace
import json
from threading import Lock
from types import SimpleNamespace as NS

import pytest
import torch

from web_agent.memory.hybrid_runtime import HybridMemoryAdapter, build_hybrid_memory, FROZEN_THRESHOLD
from web_agent.memory.label_experience import LabelExperienceMaterial
from web_agent.runtime.contracts import (
    ActionType, ConcreteAction, CausalHistoryEntry, ExecutionResult, ExecutionStatus,
    MemoryQuery, RecoveryDecision, RecoveryStrategy, TransitionInput,
)
from web_agent.runtime.memory_adapter import MemoryBoundaryError, PostFailureEmbeddingRequest
from web_agent.runtime.model_calls import ModelCallLedger, activate_model_call_ledger, deactivate_model_call_ledger, ModelCallBudgetExceeded
from tests.table2.test_development_v3 import observation
from tests.table2.test_label_experience_context import setup


def inputs(number=1):
    pre=observation(oid=f'episode:obs:{number}')
    action=ConcreteAction(action_id=f'failed-{number}',source_decision_id='policy',action_type=ActionType.NAVIGATE,parameters={})
    execution=ExecutionResult(action_id=action.action_id,status=ExecutionStatus.REJECTED,executor_step=1,
        state_changed=False,error_kind='parameter_resolution_rejected')
    post=observation(oid=f'episode:obs:{number+1}')
    history=CausalHistoryEntry(history_index=1,action_id=action.action_id,action_type=action.action_type,
        action_target_fingerprint=action.fingerprint,execution_status=execution.status,executor_step=1,
        state_changed=False,environment_error=False,execution_error_sha256='e'*64,
        post_observation_id=post.observation_id,post_observation_sha256=post.record_sha256,post_screenshot_sha256=post.screenshot_sha256)
    post=replace(post,causal_history=(history,))
    transition=TransitionInput(task_id='task',pre_observation=pre,executed_action=action,execution_result=execution,post_observation=post)
    query=MemoryQuery(query_id=f'query-{number}',task_id='task',episode_id='episode',incident_id='incident',
        post_failure_observation_id=post.observation_id,failed_action_id=action.action_id,diagnosis='parameter_resolution_rejected',
        duplicate_cluster_ids=('fixture-development',),post_failure_observation_sha256='a'*64,
        post_action_input_sha256=transition.record_sha256,processor_contract_sha256='c'*64,checkpoint_sha256='b'*64)
    request=PostFailureEmbeddingRequest(query_id=query.query_id,post_failure_observation_id=post.observation_id,
        failed_action_id=action.action_id,post_action_input=transition,post_failure_observation_sha256='a'*64,
        post_action_input_sha256=transition.record_sha256,processor_contract_sha256='c'*64,checkpoint_sha256='b'*64)
    shadow=RecoveryDecision(decision_id='shadow',incident_id='incident',strategy=RecoveryStrategy.REPLAN,
        diagnosis='parameter_resolution_rejected',trigger_sources=('executor',))
    return query,shadow,request


def adapter_fixture(tmp_path):
    store,path,hits=setup(tmp_path)
    store.frozen=True;store.write_enabled=False;store.threshold=FROZEN_THRESHOLD
    material=LabelExperienceMaterial(path,store=store,include_source_context=True)
    calls=[]
    def embedding(batch):
        calls.append('embedding')
        # The durable H2 shadow must predate even the scripted embedding call.
        assert list((tmp_path/'receipts').glob('*-shadow.json'))
        return torch.ones((1,768))
    runtime=NS(checkpoint_sha256='b'*64,lock=Lock(),torch=torch,
        batch=NS(transition=lambda *args:{}),model=NS(memory_embedding=embedding))
    adapter=HybridMemoryAdapter(episode_id='episode',evidence_dir=tmp_path/'receipts',store=store,runtime=runtime,
        transitions={},recovery_context_material=material)
    return adapter,calls,hits


@pytest.mark.parametrize('system',['H0','H1','H2'])
def test_nonmemory_systems_do_not_touch_dependencies(system):
    class Never:
        def __getattribute__(self,name):raise AssertionError('dependency accessed')
        def __fspath__(self):raise AssertionError('file accessed')
    assert build_hybrid_memory(system=system,episode_id='episode',runtime=Never(),store_directory=Never(),material_path=Never()) is None


def test_admission_changes_only_advisory_context_and_is_charged_once(tmp_path):
    adapter,calls,hits=adapter_fixture(tmp_path);query,shadow,request=inputs()
    before=(query.record_sha256,shadow.sha256,request.record_sha256)
    ledger=ModelCallLedger('episode');token=activate_model_call_ledger(ledger)
    try:result=adapter.apply_post_failure(query=query,shadow_decision=shadow,rng=None,embedding_request=request)
    finally:deactivate_model_call_ledger(token)
    assert ledger.count==1 and calls==['embedding']
    assert result.shadow_decision is shadow
    assert result.query_result.candidate_ids==tuple(h.memory_id for h in hits)
    assert result.query_result.scores==tuple(h.similarity for h in hits)
    assert result.query_result.context_candidate_ids==('memory-1',)
    assert result.final_decision.strategy==shadow.strategy
    item=result.final_decision.memory_experiences[0]
    assert item.recovery_action_value is None and item.reflection is None
    assert item.availability['recovery_action_value'] is False
    assert before==(query.record_sha256,shadow.sha256,request.record_sha256)
    receipt=json.loads((adapter.evidence_dir/'query-0001-result.json').read_text())
    assert not receipt['generation_exposure_measured'] and not receipt['completion_benefit_measured']


@pytest.mark.parametrize('empty_reason',['threshold','applicability'])
def test_empty_admission_preserves_complete_h2_object(tmp_path,empty_reason):
    adapter,_,hits=adapter_fixture(tmp_path);query,shadow,request=inputs()
    planned=ConcreteAction(action_id='planned',source_decision_id=shadow.decision_id,
        action_type=ActionType.TYPE,parameters={'text':' Exact H2 value '},bbox=(.1,.2,.3,.1))
    shadow=replace(shadow,planned_action=planned)
    adapter.store.query=lambda *args,**kwargs:tuple(replace(h,admitted=False) for h in hits) if empty_reason=='threshold' else (hits[0],)
    result=adapter.apply_post_failure(query=query,shadow_decision=shadow,rng=None,embedding_request=request)
    assert result.final_decision is shadow and result.shadow_decision is shadow
    assert result.final_decision.planned_action.parameters['text']==' Exact H2 value '
    assert not result.query_result.intervened


def test_repeat_candidates_logged_without_changing_selection(tmp_path):
    adapter,_,_=adapter_fixture(tmp_path)
    for n in (1,2):
        q,s,r=inputs(n);adapter.apply_post_failure(query=q,shadow_decision=s,rng=None,embedding_request=r)
    receipt=json.loads((adapter.evidence_dir/'query-0002-result.json').read_text())
    assert receipt['candidate_occurrences']=={'memory-0':2,'memory-1':2,'memory-2':2}
    assert receipt['total_candidate_occurrences']==6


@pytest.mark.parametrize('field,value',[('episode_id','other'),('task_id','other'),('incident_id','other'),
    ('checkpoint_sha256','f'*64),('processor_contract_sha256','f'*64),('post_failure_observation_sha256','f'*64)])
def test_mismatched_binding_rejected_before_dispatch(tmp_path,field,value):
    adapter,calls,_=adapter_fixture(tmp_path);q,s,r=inputs()
    with pytest.raises(MemoryBoundaryError):adapter.apply_post_failure(query=replace(q,**{field:value}),shadow_decision=s,rng=None,embedding_request=r)
    assert not calls


def test_exhausted_model_budget_keeps_shadow_but_prevents_embedding(tmp_path):
    adapter,calls,_=adapter_fixture(tmp_path);q,s,r=inputs()
    ledger=ModelCallLedger('episode',maximum=1,count=1);token=activate_model_call_ledger(ledger)
    try:
        with pytest.raises(ModelCallBudgetExceeded):adapter.apply_post_failure(query=q,shadow_decision=s,rng=None,embedding_request=r)
    finally:deactivate_model_call_ledger(token)
    assert not calls and (adapter.evidence_dir/'query-0001-shadow.json').exists()


def test_changed_configuration_fails_closed(tmp_path):
    adapter,calls,_=adapter_fixture(tmp_path);q,s,r=inputs();adapter.store.threshold=.5
    with pytest.raises(MemoryBoundaryError,match='configuration'):adapter.apply_post_failure(query=q,shadow_decision=s,rng=None,embedding_request=r)
    assert not calls


def test_factory_requires_known_system_and_pinned_manifest(tmp_path):
    with pytest.raises(MemoryBoundaryError,match='identity'):
        build_hybrid_memory(system='E3',episode_id='episode')
    with pytest.raises(MemoryBoundaryError,match='bindings'):
        build_hybrid_memory(system='H3',episode_id='episode')
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(MemoryBoundaryError,match='manifest mismatch'):
        build_hybrid_memory(system='H3',episode_id='episode',store_directory=tmp_path,
            material_path=tmp_path/'absent',evidence_dir=tmp_path/'output',
            expected_manifest_sha256='a'*64,expected_material_sha256='b'*64)
    assert not (tmp_path/'output').exists()


def test_strategy_override_is_rejected_even_if_shared_adapter_changes(tmp_path,monkeypatch):
    from web_agent.memory.label_runtime import LabelBackedMemoryAdapter
    original=LabelBackedMemoryAdapter.apply_post_failure
    def override(self,**kwargs):
        result=original(self,**kwargs)
        return replace(result,final_decision=replace(result.final_decision,strategy=RecoveryStrategy.RETRY))
    monkeypatch.setattr(LabelBackedMemoryAdapter,'apply_post_failure',override)
    adapter,_,_=adapter_fixture(tmp_path);q,s,r=inputs();before=s.sha256
    with pytest.raises(MemoryBoundaryError,match='altered'):
        adapter.apply_post_failure(query=q,shadow_decision=s,rng=None,embedding_request=r)
    assert s.sha256==before


def test_shadow_mutation_is_detected_and_original_is_protected(tmp_path,monkeypatch):
    from web_agent.memory.label_runtime import LabelBackedMemoryAdapter
    def mutate(self,**kwargs):
        kwargs['shadow_decision'].planned_action.parameters['text']='mutated'
        raise ValueError('callback failed')
    monkeypatch.setattr(LabelBackedMemoryAdapter,'apply_post_failure',mutate)
    adapter,_,_=adapter_fixture(tmp_path);q,s,r=inputs()
    s=replace(s,planned_action=ConcreteAction(action_id='plan',source_decision_id='shadow',
        action_type=ActionType.TYPE,parameters={'text':'Original'}))
    with pytest.raises(MemoryBoundaryError,match='mutated'):
        adapter.apply_post_failure(query=q,shadow_decision=s,rng=None,embedding_request=r)
    assert s.planned_action.parameters['text']=='Original'


def test_memory_reaches_generation_context_without_overriding_model_choice(tmp_path,monkeypatch):
    from web_agent.runtime.causal_recovery_planner import CausalRecoveryRuntime
    from web_agent.runtime.named_target_policy import NamedTargetRuntime
    from web_agent.runtime.contracts import RuntimeTaskView, detached_record_copy
    adapter,_,_=adapter_fixture(tmp_path);q,s,r=inputs()
    result=adapter.apply_post_failure(query=q,shadow_decision=s,rng=None,embedding_request=r)
    monkeypatch.setattr(NamedTargetRuntime,'__init__',lambda *a,**k:None)
    seen=[]
    def generate(self,**kwargs):
        seen.append(kwargs['suffix'])
        # Scripted planner deliberately chooses TYPE despite retrieved CLICK.
        return json.dumps({'action_type':'TYPE','target':'Field','bbox':[.1,.2,.3,.1],'value':'Chosen text'})
    monkeypatch.setattr(NamedTargetRuntime,'_generate',generate)
    planner=CausalRecoveryRuntime()
    transition=r.post_action_input
    decision=planner.predict_recovery(RuntimeTaskView(task_id='task',goal='visible task'),detached_record_copy(transition.post_observation),
        result.final_decision,transition.executed_action,rng=None)
    assert decision.action_type is ActionType.TYPE and decision.parameter_hints['text']=='Chosen text'
    context=json.loads(seen[0].split('\ncausal_recovery_context: ')[1])
    assert [x['memory_id'] for x in context['retrieved_training_examples']]==['memory-1']
    assert context['retrieved_training_examples'][0]['reflection'] is None
    assert planner._context_suffix is None
