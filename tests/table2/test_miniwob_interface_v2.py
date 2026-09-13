"""Engineering and statistical regression cases; no claimed model efficacy."""
from dataclasses import replace
from types import SimpleNamespace as NS
import hashlib
import json
import random

import pytest

from web_agent.benchmarks.miniwob_controls import CONTROL_INTERFACE,aliases,bind_controls,supported_actions,validate_control_action
from web_agent.runtime.named_target_policy import resolve_named_target
from web_agent.runtime.policy import ActionParseError
from web_agent.runtime.observation import assert_oracle_blind_mapping
from web_agent.runtime.contracts import RecoveryPlannerContext,RecoveryDecision,RecoveryStrategy,RuntimeTaskView,ConcreteAction,ActionType,PolicyObservation
from web_agent.runtime.recovery.controller import CallableRecoveryActionPlanner,RecoveryController
from web_agent.runtime.protocol import RuntimeBudgets
from web_agent.runtime.action_parameters import ParameterResolutionError
from web_agent.eval.table2.miniwob_reporting import analyze,log_matches,paired_interval


def raw_control(**kw):
    return dict(tag='input',input_type='text',role='',source_id='field',name='',text='',
        accessible_names=[{'value':'Person','source':'associated_label'}],value='',value_present=False,
        target_bbox=[.1,.2,.3,.1],candidate_options=[],destination='',clickable=True,link_like=False,
        contenteditable=False,disabled=False,readonly=False,checked=None,selected=[],focused=False,**kw)


def view(raw=None):
    controls=bind_controls({'controls':raw or [raw_control()]},'episode:obs:1')
    return PolicyObservation(task_id='task',goal='visible task',observation_id='episode:obs:1',
        screenshot_sha256='a'*64,screenshot_path=None,width=1000,height=600,url='fixture://test',title='',
        current_page_state={'control_interface':CONTROL_INTERFACE,'visible_controls':controls})


def test_control_projection_preserves_causal_boundary_and_exact_value():
    obs=view(); assert_oracle_blind_mapping(obs.current_page_state)
    raw=json.dumps(dict(action_type='TYPE',target='Person',bbox=None,value='Exact Case 123'))
    parsed,_=resolve_named_target(raw,obs,allow_role_target=True)
    assert parsed['action_type']=='TYPE' and parsed['value']=='Exact Case 123'
    assert parsed['bbox']==tuple(obs.current_page_state['visible_controls'][0]['target_bbox']) or list(parsed['bbox'])==[.1,.2,.3,.1]
    raw=json.dumps(dict(action_type='TYPE',target='o1:c0',bbox=None,value='value'))
    assert resolve_named_target(raw,obs)[0]['value']=='value'


def test_duplicates_stale_ids_and_wrong_action_never_choose_a_target():
    obs=view([raw_control(),{**raw_control(),'source_id':'second','target_bbox':[.6,.2,.3,.1]}])
    with pytest.raises(ActionParseError):resolve_named_target(json.dumps(dict(action_type='TYPE',target='Person',bbox=None,value='x')),obs)
    with pytest.raises(ActionParseError):resolve_named_target(json.dumps(dict(action_type='TYPE',target='o0:c0',bbox=None,value='x')),obs)
    checkbox={**raw_control(),'input_type':'checkbox','checked':False}
    assert supported_actions(checkbox)==('CLICK',)
    with pytest.raises(ActionParseError):resolve_named_target(json.dumps(dict(action_type='TYPE',target='Person',bbox=None,value='x')),view([checkbox]))


@pytest.mark.parametrize('flag',['disabled','readonly'])
def test_noneditable_controls_reject_type(flag):
    c={**raw_control(),flag:True};obs=view([c])
    with pytest.raises(ValueError,match='INCOMPATIBLE'):validate_control_action('TYPE',{},(.1,.2,.3,.1),obs)


def test_coordinate_action_preserves_point_and_stale_geometry_rejects():
    obs=view();box=(.12,.21,.1,.05)
    values={'target_x':.17,'target_y':.235}
    assert validate_control_action('CLICK',values,box,obs)['source_id']=='field'
    assert values=={'target_x':.17,'target_y':.235} and box==(.12,.21,.1,.05)
    obs.current_page_state['visible_controls'][0]['observation_id']='old'
    with pytest.raises(ValueError,match='STALE'):validate_control_action('CLICK',values,box,obs)


@pytest.mark.parametrize('error,stage',[(ActionParseError('bad JSON'),'model_parse'),
    (ParameterResolutionError('no text'),'parameter_resolution'),(ValueError('invalid strategy'),'strategy_validation')])
def test_completed_rejections_keep_stage_and_consume_attempt(error,stage):
    obs=view();task=RuntimeTaskView(task_id='task',goal='visible task')
    context=RecoveryPlannerContext(task_id='task',episode_id='episode',incident_id='incident',observation_id=obs.observation_id,completed_actions=())
    captured=[]
    def callback(*args,planner_context):
        captured.append(planner_context);raise error
    planner=CallableRecoveryActionPlanner(planner_id='test',planner_version='v2',callback=callback,accepts_context=True)
    controller=RecoveryController(RuntimeBudgets(),action_planner=planner)
    decision=RecoveryDecision(decision_id='d',incident_id='incident',strategy=RecoveryStrategy.REPLAN,trigger_sources=('executor',),diagnosis='execution rejected')
    failed=ConcreteAction(action_id='a',source_decision_id='d0',action_type=ActionType.NAVIGATE,parameters={})
    plan=controller.begin_attempt(decision,failed,task=task,post_failure_observation=obs,rng=random.Random(42),planner_context=context)
    assert plan.resolution_status=='REJECTED' and plan.rejection_stage==stage
    assert controller.episode_attempts==1 and captured[0] is not context
    assert captured[0].to_dict()==context.to_dict()


def test_context_rejects_oracle_fields():
    with pytest.raises(ValueError):
        RecoveryPlannerContext(task_id='t',episode_id='e',incident_id='i',observation_id='o',completed_actions=(),
            previous_attempt={'attempt_id':'a','stage':'oracle_success','code':'x','detail':'x','reward':1})


def test_runtime_builds_full_history_from_owned_actions(tmp_path,monkeypatch):
    import web_agent.runtime.recovery.controller as module
    from web_agent.benchmarks.recovery_fixture import run_failure_memory_intervention_smoke
    original=module.CallableRecoveryActionPlanner;contexts=[]
    def factory(**kw):
        callback=kw.pop('callback')
        def capture(*args,planner_context):
            contexts.append(planner_context)
            assert planner_context.completed_actions
            assert all(set(a)=={'action_id','action_type','bbox','parameters','execution_status','state_changed','environment_error'} for a in planner_context.completed_actions)
            return callback(*args)
        return original(**kw,callback=capture,accepts_context=True)
    monkeypatch.setattr(module,'CallableRecoveryActionPlanner',factory)
    run_failure_memory_intervention_smoke(tmp_path/'synthetic')
    assert contexts and all(c.completed_actions[-1]['action_id'] for c in contexts)


def toy_rows():
    blocks=[{'task':t,'repeat_id':r} for t in ('a','b') for r in range(5)]
    plan={'blocks':blocks,'systems':['E0','E1','E2','E3'],'scope':'synthetic'}
    summary=dict(valid_for_primary=True,environment_failure=False,executor_steps=1,recovery_actions=0,recovery_attempts=0,
        memory_queries=0,memory_interventions=0,model_call_count=1,elapsed_seconds=.1)
    rows=[dict(**b,system=s,success=s in {'E2','E3'},summary=summary) for b in blocks for s in plan['systems']]
    return plan,rows


def test_matched_denominators_holm_and_degenerate_intervals():
    plan,rows=toy_rows();result=analyze(plan,rows)
    assert result['all_included_episodes_present']
    recovery,memory=result['paired_contrasts']
    assert recovery['paired_instances']==10 and recovery['improved']==10
    assert recovery['holm_p']==pytest.approx(4/1024) and recovery['conclusion']=='POSITIVE_INCREMENT_SUPPORTED'
    assert memory['holm_p']==1 and memory['ci95']==[0.,0.]
    missing=analyze(plan,rows[:-1])
    assert not missing['all_included_episodes_present'] and all(s['episodes']==9 for s in missing['systems'])
    with pytest.raises(ValueError,match='duplicate'):analyze(plan,rows+[rows[0]])
    assert paired_interval([('a',1),('b',-1)])==[0.,0.]


def test_redacted_numeric_and_text_evidence_is_verified():
    for raw in ('Exact Text',.243976):
        logged={'redacted':True,'sha256':hashlib.sha256(str(raw).encode()).hexdigest()}
        assert log_matches(raw,logged) and not log_matches('different',logged)


def test_task_binding_preserves_goal_and_block_identity():
    from web_agent.eval.table2.miniwob_study import task_binding
    task=NS(to_dict=lambda:dict(task_id='miniwob.enter-text',goal='Type Exact Text.'))
    block=dict(task='enter-text',repeat_id=0,stage_reset_seed=123,browser_reset_seed=123)
    record=task_binding(task,block)
    assert all(record[k]==v for k,v in block.items())
    assert record['task_specification']==task.to_dict()
    assert json.loads(json.dumps(record))['task_specification']['goal']=='Type Exact Text.'


def test_memory_v2_uses_only_existing_source_context(tmp_path):
    from web_agent.memory.label_backed_store import LabelMemoryHit
    from web_agent.memory.label_experience import LabelExperienceMaterial
    from web_agent.runtime.contracts import RetrievedRecoveryExperience
    source=dict(memory_id='m',source_split='train',evidence_basis='training_dataset_labels',material=dict(
        canonical_task_id='train',failure_type='ERROR',failed_action='UNAVAILABLE',failed_action_available=False,
        executed_recovery_action='TYPE',recovery_strategy='REPLAN',recovery_action_value='',reflection_text='',
        task_description='Original source task',website_domain='example.org'))
    raw=(json.dumps(source)+'\n').encode();path=tmp_path/'material.jsonl';path.write_bytes(raw)
    store=NS(memory_input_sha256=hashlib.sha256(raw).hexdigest())
    reader=LabelExperienceMaterial(path,store=store,include_source_context=True)
    item=reader.for_hit(LabelMemoryHit('m','train','ERROR','REPLAN',.8,True))
    assert item.source_task_description=='Original source task' and item.source_domain=='example.org'
    assert item.failed_action is None and item.reflection is None and item.recovery_action_value is None
    assert item.availability['source_domain'] and not item.availability['reflection']
    assert RetrievedRecoveryExperience.from_dict(item.to_dict()).record_sha256==item.record_sha256
    with pytest.raises(ValueError,match='availability'):replace(item,availability={**item.availability,'reflection':True})


def test_e0_and_parameter_generation_receive_same_controls(tmp_path,monkeypatch):
    from threading import Lock
    from web_agent.runtime.qwen2vl_pc01 import _UnadaptedBaseRuntime
    from web_agent.runtime.named_target_policy import NamedTargetRuntime
    from web_agent.benchmarks.miniwob_controls import prompt_controls
    seen=[]
    def generate(self,**kw):
        seen.append(kw['suffix'])
        return json.dumps(dict(action_type='CLICK',target='o1:c0',bbox=None,value=None))
    monkeypatch.setattr(_UnadaptedBaseRuntime,'_generate',generate)
    base=NS(model=None,processor=None,torch=None,lock=Lock())
    runtime=NamedTargetRuntime(base=base,evidence_dir=tmp_path,include_control_context=True)
    obs=view();task=RuntimeTaskView(task_id='task',goal='visible task')
    runtime._generate(prompt='prompt',task=task,observation=obs,suffix='')
    expected=prompt_controls(obs)
    assert json.loads(seen[0].split('\ncurrent_controls: ')[1])['controls']==expected
    parameter_base=NS(_generate=lambda **kw:(seen.append(kw['suffix']) or '{"status":"REJECTED"}'))
    with pytest.raises(ParameterResolutionError):
        _UnadaptedBaseRuntime.resolve_parameters(parameter_base,task,obs,
            NS(action_type=ActionType.NAVIGATE,bbox=None,parameter_hints={}),None)
    assert json.loads(seen[1].split('\ncurrent_controls: ')[1])['controls']==expected
