"""Engineering checks; no claimed model or browser efficacy."""
from dataclasses import replace
from types import SimpleNamespace as NS
import json
import pytest
from web_agent.runtime.contracts import ActionType, ConcreteAction, PolicyObservation
from web_agent.benchmarks.miniwob_controls import bind_controls, CONTROL_INTERFACE
from web_agent.runtime.observable_progress import observable_step_effect, register_novel_effect
from web_agent.memory.context_applicability import context_exclusion


def observation(value='', kind='text', checked=None, selected=(), focused=False, oid='episode:obs:1'):
    c=dict(tag='select' if kind=='select' else 'input',input_type=kind,role='',source_id='field',name='',text='',
        accessible_names=[{'value':'Field','source':'associated_label'}],value=value,value_present=bool(value),
        target_bbox=[.1,.2,.3,.1],candidate_options=['one','two'] if kind=='select' else [],destination='',
        clickable=True,link_like=False,contenteditable=False,disabled=False,readonly=False,
        checked=checked,selected=list(selected),focused=focused)
    return PolicyObservation(task_id='task',goal='visible task',observation_id=oid,
        screenshot_sha256='a'*64,screenshot_path=None,width=1000,height=600,url='http://fixture/',title='',
        current_page_state={'control_interface':CONTROL_INTERFACE,'visible_controls':bind_controls({'controls':[c]},oid)})


def action(kind='TYPE', **params):
    return ConcreteAction(action_id='action',source_decision_id='decision',action_type=ActionType(kind),
        bbox=(.1,.2,.3,.1),parameters=params)


def test_exact_text_effect_is_distinct_from_focus_or_partial_entry():
    a=action(text='Exact Case');before=observation()
    assert observable_step_effect(a,before,observation('Exact Case',oid='episode:obs:2'))['observed_effect']
    for after in [observation(focused=True),observation('Exact'),observation('exact case')]:
        assert not observable_step_effect(a,before,after)['observed_effect']
    assert not observable_step_effect(a,observation('Exact Case'),observation('Exact Case'))['observed_effect']
    assert not observable_step_effect(a,observation(kind='password'),observation(value=None,kind='password',focused=True))['observed_effect']


def test_checkbox_toggle_and_selection_are_observed_but_cycles_are_not_novel():
    off=observation(kind='checkbox',checked=False);on=observation(kind='checkbox',checked=True,oid='episode:obs:2')
    seen=set();a=action('CLICK')
    assert register_novel_effect(observable_step_effect(a,off,on),seen)['novel_effect']
    assert not register_novel_effect(observable_step_effect(a,on,off),seen)['novel_effect']
    assert not register_novel_effect(observable_step_effect(a,off,on),seen)['novel_effect']
    assert register_novel_effect(observable_step_effect(a,off,on),set())['novel_effect']
    assert observable_step_effect(action('SELECT',option='two'),observation(kind='select',selected=['one']),
        observation(kind='select',selected=['two']))['observed_effect']
    assert not observable_step_effect(action('SELECT',option='one'),observation(kind='select',selected=['one']),
        observation(kind='select',selected=['two']))['observed_effect']


def test_stale_or_different_control_does_not_earn_progress():
    pre=observation();post=observation('Exact Case');post.current_page_state['visible_controls'][0]['source_id']='different'
    assert not observable_step_effect(action(text='Exact Case'),pre,post)['observed_effect']
    with pytest.raises(ValueError,match='task mismatch'):
        observable_step_effect(action(text='Exact Case'),pre,replace(post,task_id='other'))


def transition():
    return {'executed_action':{'action_type':'NAVIGATE'},
        'execution_result':{'status':'rejected','error_kind':'parameter_resolution_rejected'},
        'post_observation':{'causal_history':[], 'current_page_state':observation().current_page_state}}


def example(kind='CLICK',strategy='ALTERNATIVE_TARGET',value=None):
    return dict(recorded_strategy=strategy,recovery_action=kind,recovery_action_value=value)


def test_memory_applicability_uses_no_task_answers_and_keeps_missing_values_missing():
    t=transition()
    assert context_exclusion(example(strategy='BACKTRACK'),t)=='BACKTRACK_NO_EXECUTED_IN_EPISODE_NAVIGATION'
    for kind in ['TYPE','SELECT','PRESS_KEY','SCROLL']:
        assert context_exclusion(example(kind),t)=='CORRECTIVE_ARGUMENT_UNAVAILABLE'
    assert context_exclusion(example('NAVIGATE'),t)=='SAME_UNRESOLVED_ACTION'
    assert context_exclusion(example(),t) is None
    assert context_exclusion(example('SELECT',value='two'),t)=='NO_CURRENT_COMPATIBLE_CONTROL'


def test_selective_adapter_preserves_top_three_and_abstains_without_replacement():
    from tests.table2.test_label_memory_runtime import setup_adapter,request
    from web_agent.memory.label_backed_store import LabelMemoryHit
    from web_agent.runtime.contracts import RetrievedRecoveryExperience,RecoveryStrategy
    adapter,shadow=setup_adapter();query,embedding=request(1)
    adapter.selective_context=True
    adapter.store.query=lambda *a,**kw:tuple(LabelMemoryHit(str(i),'train','ERROR','BACKTRACK',.9-i*.01,True) for i in range(3))
    adapter.recovery_context_material=NS(sha256='f'*64,for_hit=lambda h:RetrievedRecoveryExperience(
        memory_id=h.memory_id,source_task_id='train',failure_type='ERROR',failed_action=ActionType.NAVIGATE,
        recovery_action=ActionType.NAVIGATE,recorded_strategy=RecoveryStrategy.BACKTRACK))
    embedding.post_action_input.to_dict=lambda:transition()
    result=adapter.apply_post_failure(query=query,shadow_decision=shadow,rng=None,embedding_request=embedding)
    assert result.final_decision is shadow
    assert result.query_result.candidate_ids==('0','1','2')
    assert not result.query_result.admitted and not result.query_result.changed_recovery_context
    assert len(result.query_result.exclusion_reasons)==3


def test_profile_cannot_prepare_final_evaluation(tmp_path,monkeypatch):
    from web_agent.eval.table2 import miniwob_study as study
    monkeypatch.setattr(study,'CONFIG',study.ROOT/'configs/eval/table2/miniwob_development_v3.json')
    with pytest.raises(ValueError,match='development only'):
        study.prepare(tmp_path/'forbidden','evaluation')


def test_full_episode_continues_after_text_effect_with_negative_learned_assessment(tmp_path):
    from web_agent.benchmarks.base import BenchmarkAdapter,AdapterExecution
    from web_agent.runtime.contracts import (Observation,ObservationStage,OpaqueTerminalSignal,ExecutionStatus,
        TaskSpecification,PreActionDecision,TransitionAssessment,RecoveryAssessment,RecoveryStrategy,SystemID,
        probability_map,canonical_sha256)
    from web_agent.runtime.episode import EpisodeRunner
    from web_agent.runtime.executor import Executor
    from web_agent.runtime.event_log import EpisodeEventLogs
    from web_agent.runtime.protocol import RuntimeProtocol,switches_for
    from web_agent.runtime.policy import CallablePolicyAdapter,PolicyKind,SystemPolicy
    from web_agent.runtime.action_parameters import DeterministicParameterProvider
    from web_agent.runtime.recovery.controller import RecoveryController,CallableRecoveryActionPlanner

    class Form(BenchmarkAdapter):
        benchmark_id='fixture';benchmark_version='v1'
        def reset(self,task,*,episode_id,seed):
            self.task=task;self.eid=episode_id;self.i=0;self.value='';self.done=False
            return self.observe(stage=ObservationStage.RESET)
        def observe(self,*,stage,prior_action_id=None):
            self.i+=1;oid=f'{self.eid}:obs:{self.i}'
            state=observation(self.value,oid=oid).current_page_state
            from web_agent.benchmarks.miniwob_controls import compatible_actions
            from web_agent.runtime.recovery.strategies import build_recovery_target_evidence
            state['recovery_target_evidence']=build_recovery_target_evidence(task=self.task,observation_id=oid,
                compatible_actions=compatible_actions(state['visible_controls']))
            return Observation(observation_id=oid,episode_id=self.eid,stage=stage,prior_action_id=prior_action_id,
                screenshot_sha256=canonical_sha256({'value':self.value,'done':self.done}),
                screenshot_path=None,width=1000,height=600,url='http://fixture/',title='',page_state=state)
        def execute(self,a):
            if a.action_type is ActionType.TYPE:self.value=a.parameters['text']
            else:self.done=True
            return AdapterExecution(status=ExecutionStatus.EXECUTED,state_changed=True)
        def terminal_signal(self,task,binding=None):
            return OpaqueTerminalSignal(event_id=f'term:{self.i}',token_sha256='f'*64,terminate=self.done)
        def close(self):
            pass
    def predict(task,o,rng):
        return PreActionDecision(decision_id='d:'+o.observation_id,observation_id=o.observation_id,
            action_type=ActionType.NAVIGATE,action_probabilities=probability_map(tuple(x.value for x in ActionType),'NAVIGATE'),
            bbox=None,grounding_confidence=1.,confidence_before=1.,input_observation_ids=(o.observation_id,),
            policy_id='fixture',policy_version='v1',parameter_hints={})
    def diagnose(task,t,rng):
        return TransitionAssessment(assessment_id='assessment:'+t.executed_action.action_id,
            pre_observation_id=t.pre_observation.observation_id,post_observation_id=t.post_observation.observation_id,
            executed_action_id=t.executed_action.action_id,predicted_failure=True,failure_probability=1.,
            failure_type='LOOP_DETECTED',failure_type_probabilities={'LOOP_DETECTED':1.},needs_recovery=True,
            needs_recovery_probability=1.,recovery_strategy=RecoveryStrategy.REPLAN,recovery_probabilities={'REPLAN':1.})
    def assess(task,t,rng):
        return RecoveryAssessment(assessment_id='negative:'+t.attempt_id,incident_id=t.incident_id,attempt_id=t.attempt_id,
            pre_recovery_observation_id=t.pre_recovery_observation.observation_id,
            post_recovery_observation_id=t.post_recovery_observation.observation_id,
            recovery_action_ids=tuple(a.action_id for a in t.recovery_actions),
            predicted_failure_resolved=False,predicted_resolution_probability=0.,predicted_progress=False,predicted_progress_probability=0.)
    def propose(task,o,d,failed,rng):
        params={'target_x':.25,'target_y':.25,'target_bbox':[.1,.2,.3,.1]}
        proposed=(action('CLICK',button='left',click_count=1,**params)
                  if o.current_page_state['visible_controls'][0]['value'] else action(text='Exact Case',**params))
        return replace(proposed,
            action_id=d.decision_id+':proposed',source_decision_id=d.decision_id)
    protocol=RuntimeProtocol(protocol_id='fixture-progress',campaign_id='fixture-progress',benchmark_id='fixture',benchmark_version='v1',
        provider_id='deterministic-parameter-provider',provider_version='v1')
    task=TaskSpecification(task_id='task',goal='Enter text then complete the form',benchmark_id='fixture',benchmark_version='v1',start_state_id='empty')
    adapter=CallablePolicyAdapter(policy_id='fixture',policy_version='v1',kind=PolicyKind.TRAINED,checkpoint_sha256='a'*64,
        action_predictor=predict,transition_predictor=diagnose,recovery_predictor=assess)
    eid='fixture-progress:E2:task:repeat-0:seed-42'
    runner=EpisodeRunner(protocol=protocol,system_policy=SystemPolicy(adapter,switches_for(SystemID.E2)),
        provider=DeterministicParameterProvider(),executor=Executor(Form(),budgets=protocol.budgets),
        recovery_controller=RecoveryController(protocol.budgets,action_planner=CallableRecoveryActionPlanner(
            planner_id='fixture',planner_version='v1',callback=propose)),observable_progress_continuation=True,
        event_logs=EpisodeEventLogs(tmp_path,episode_id=eid,include_memory=False))
    result=runner.run(task,repeat_id=0,model_seed=42)
    assert result.terminal_reason.value=='opaque_verifier_terminal'
    assert result.normal_actions==2 and result.recovery_actions==2 and result.failure_incidents==2
    assert result.executor_steps==4 and result.recovery_attempts==2
    records=[json.loads(l) for l in (tmp_path/'recoveries.jsonl').read_text().splitlines()]
    effects=[r['payload'] for r in records if r['event_type']=='observable_step_progress']
    assert [e['continuation_allowed'] for e in effects]==[True,False]
    assert all(not r['payload']['predicted_assessment']['predicted_failure_resolved']
               for r in records if r['event_type']=='recovery_attempt' and 'predicted_assessment' in r['payload'])
