"""Full-controller scripted fixtures, not live H-system performance results.

The existing E2 fixture switch contract exercises the default-off runner feature.
H identity/runner registration remains Task 5; no E experiment is relabeled H.
"""
from dataclasses import replace
import json
import pytest

from web_agent.benchmarks.base import BenchmarkAdapter, AdapterExecution
from web_agent.benchmarks.miniwob_controls import compatible_actions
from web_agent.runtime.action_parameters import DeterministicParameterProvider
from web_agent.runtime.contracts import (
    ActionType, ConcreteAction, Observation, ObservationStage, OpaqueTerminalSignal,
    ExecutionStatus, TaskSpecification, PreActionDecision, TransitionAssessment,
    RecoveryAssessment, RecoveryStrategy, SystemID, TerminalReason,
    probability_map, canonical_sha256,
)
from web_agent.runtime.episode import EpisodeRunner
from web_agent.runtime.event_log import EpisodeEventLogs
from web_agent.runtime.executor import Executor, EpisodeTimeout
from web_agent.runtime.model_calls import record_model_call
from web_agent.runtime.policy import CallablePolicyAdapter, PolicyKind, SystemPolicy, ActionParseError
from web_agent.runtime.protocol import RuntimeProtocol, switches_for
from web_agent.runtime.recovery.controller import RecoveryController, CallableRecoveryActionPlanner
from web_agent.runtime.recovery.strategies import build_recovery_target_evidence
from tests.table2.test_development_v3 import observation


def run_fixture(root, *, hybrid=True, text=False, terminal_at=4, still=False,
                rejected=False, browser_rejected=False, strategy=RecoveryStrategy.REPLAN,
                positive=False, progress=False, different_normal=False, extra_calls=0,
                timeout=False, observable=False, rerun=False, abort_after=None):
    seen=[]
    class Page(BenchmarkAdapter):
        benchmark_id='fixture';benchmark_version='v1'
        def reset(self,task,*,episode_id,seed):
            self.task=task;self.eid=episode_id;self.i=0;self.n=0;self.value=''
            return self.observe(stage=ObservationStage.RESET)
        def observe(self,*,stage,prior_action_id=None):
            self.i+=1;oid=f'{self.eid}:obs:{self.i}'
            state=observation(self.value,oid=oid).current_page_state
            state['recovery_target_evidence']=build_recovery_target_evidence(task=self.task,observation_id=oid,
                compatible_actions=compatible_actions(state['visible_controls']))
            return Observation(observation_id=oid,episode_id=self.eid,stage=stage,prior_action_id=prior_action_id,
                screenshot_sha256=canonical_sha256({'visible_counter':0 if still else self.n,'text':self.value}),
                screenshot_path=None,width=1000,height=600,url='http://fixture/',title='',page_state=state)
        def execute(self,a):
            self.n+=1
            if a.action_type is ActionType.TYPE:self.value=a.parameters['text']
            if browser_rejected and a.recovery_attempt_id:
                return AdapterExecution(status=ExecutionStatus.REJECTED,state_changed=False)
            return AdapterExecution(status=ExecutionStatus.EXECUTED,state_changed=not still)
        def terminal_signal(self,task,binding=None):
            return OpaqueTerminalSignal(event_id=f'term:{self.i}',token_sha256='f'*64,
                terminate=terminal_at is not None and self.n>=terminal_at)
        def close(self):pass
    def predict(t,o,rng):
        for _ in range(extra_calls):record_model_call(stage='pre_action_policy',component_id='scripted-extra')
        return PreActionDecision(decision_id='d:'+o.observation_id,observation_id=o.observation_id,
            action_type=ActionType.TYPE if different_normal else ActionType.CLICK,
            action_probabilities=probability_map(tuple(a.value for a in ActionType),'TYPE' if different_normal else 'CLICK'),
            bbox=(.1,.2,.3,.1),grounding_confidence=1.,confidence_before=1.,input_observation_ids=(o.observation_id,),
            policy_id='fixture',policy_version='v1',parameter_hints={'text':o.observation_id} if different_normal else {})
    def diagnose(t,x,rng):
        selected_strategy = RecoveryStrategy.ABORT if abort_after is not None and len(seen)>=abort_after else strategy
        return TransitionAssessment(assessment_id='diag:'+x.executed_action.action_id,
            pre_observation_id=x.pre_observation.observation_id,post_observation_id=x.post_observation.observation_id,
            executed_action_id=x.executed_action.action_id,predicted_failure=True,failure_probability=1.,
            failure_type='LOOP_DETECTED',failure_type_probabilities={'LOOP_DETECTED':1.},needs_recovery=True,
            needs_recovery_probability=1.,recovery_strategy=selected_strategy,recovery_probabilities={selected_strategy.value:1.})
    def assess(t,x,rng):
        seen.append(x)
        if timeout:raise EpisodeTimeout('scripted timeout during assessment')
        return RecoveryAssessment(assessment_id='assess:'+x.attempt_id,incident_id=x.incident_id,attempt_id=x.attempt_id,
            pre_recovery_observation_id=x.pre_recovery_observation.observation_id,
            post_recovery_observation_id=x.post_recovery_observation.observation_id,
            recovery_action_ids=tuple(a.action_id for a in x.recovery_actions),predicted_failure_resolved=positive,
            predicted_resolution_probability=float(positive),predicted_progress=progress,predicted_progress_probability=float(progress))
    def propose(t,o,d,failed,rng):
        if rejected:raise ActionParseError('scripted malformed proposal')
        kind=ActionType.TYPE if text and not o.current_page_state['visible_controls'][0]['value'] else ActionType.CLICK
        return ConcreteAction(action_id=d.decision_id+':proposed',source_decision_id=d.decision_id,action_type=kind,
            bbox=(.1,.2,.3,.1),parameters={'target_x':.25,'target_y':.25,'target_bbox':[.1,.2,.3,.1],
                **({'text':'Exact Case'} if kind is ActionType.TYPE else {'button':'left','click_count':1})})
    protocol=RuntimeProtocol(protocol_id='hybrid-controller-fixture',campaign_id='hybrid-controller-fixture',
        benchmark_id='fixture',benchmark_version='v1',provider_id='deterministic-parameter-provider',provider_version='v1')
    policy=CallablePolicyAdapter(policy_id='fixture',policy_version='v1',kind=PolicyKind.TRAINED,checkpoint_sha256='a'*64,
        action_predictor=predict,transition_predictor=diagnose,recovery_predictor=assess)
    controller=RecoveryController(protocol.budgets,action_planner=CallableRecoveryActionPlanner(
        planner_id='fixture',planner_version='v1',callback=propose))
    eid='hybrid-controller-fixture:E2:task:repeat-0:seed-42'
    runner=EpisodeRunner(protocol=protocol,system_policy=SystemPolicy(policy,switches_for(SystemID.E2)),
        provider=DeterministicParameterProvider(),executor=Executor(Page(),budgets=protocol.budgets),
        recovery_controller=controller,hybrid_continuation=hybrid,observable_progress_continuation=observable,
        event_logs=EpisodeEventLogs(root,episode_id=eid,include_memory=False))
    task=TaskSpecification(task_id='task',goal='Scripted controller engineering fixture',benchmark_id='fixture',benchmark_version='v1',start_state_id='initial')
    result=runner.run(task,repeat_id=0,model_seed=42)
    if rerun:
        first=result
        runner.event_logs=None
        # Episode completion closes its executor. Reuse controller/runner state
        # with a fresh executor, rather than reopening the closed browser.
        runner.executor=Executor(Page(),budgets=protocol.budgets)
        result=runner.run(task,repeat_id=0,model_seed=42)
        assert result.failure_incidents==first.failure_incidents
        assert result.recovery_attempts==first.recovery_attempts
    rows=[json.loads(line) for p in root.glob('*.jsonl') for line in p.read_text().splitlines()]
    return result,controller,seen,rows


@pytest.mark.parametrize('text',[False,True])
def test_negative_assessment_exhausts_incident_then_normal_action_completes(tmp_path,text):
    result,controller,seen,rows=run_fixture(tmp_path,text=text,observable=True)
    assert result.terminal_reason is TerminalReason.OPAQUE_VERIFIER_TERMINAL
    assert result.normal_actions==2 and result.recovery_actions==2 and result.executor_steps==4
    assert controller.episode_attempts==len(seen)==2
    assert len({x.incident_id for x in seen})==1
    events=[r['payload'] for r in rows if r['event_type']=='hybrid_continuation']
    assert events[-1]['incident_status']=='exhausted_unresolved' and events[-1]['next']=='normal'
    effects=[r['payload'] for r in rows if r['event_type']=='observable_step_progress']
    assert all(not x['continuation_allowed'] for x in effects)
    if text:assert effects[0]['observed_effect']


def test_default_off_preserves_legacy_stop(tmp_path):
    result,controller,seen,rows=run_fixture(tmp_path,hybrid=False)
    assert result.terminal_reason is TerminalReason.RECOVERY_BUDGET_EXHAUSTED
    assert result.executor_steps==3 and result.normal_actions==1 and len(seen)==2
    assert not any(r['event_type'].startswith('hybrid_') for r in rows)


def test_repeated_failure_keeps_spent_incident_across_new_observations(tmp_path):
    result,controller,seen,rows=run_fixture(tmp_path,terminal_at=None)
    assert result.terminal_reason is TerminalReason.ACTION_BUDGET_EXHAUSTED
    assert result.executor_steps==30 and result.normal_actions==28 and result.recovery_attempts==2
    assert result.failure_incidents==1 and controller.episode_attempts==2
    bindings=[r['payload'] for r in rows if r['event_type']=='hybrid_incident_binding']
    assert len({x['incident_id'] for x in bindings})==1
    assert len({x['post_observation_id'] for x in bindings})>2


def test_different_failed_requests_still_cannot_exceed_episode_recovery_cap(tmp_path):
    result,controller,seen,rows=run_fixture(tmp_path,terminal_at=None,different_normal=True)
    assert result.terminal_reason is TerminalReason.ACTION_BUDGET_EXHAUSTED
    assert result.executor_steps==30 and result.recovery_attempts==4 and controller.episode_attempts==4
    assert max(controller._incident_attempts.values())==2
    assert any(r['event_type']=='hybrid_continuation' and r['payload']['reason']=='episode_recovery_allowance_exhausted' for r in rows)


@pytest.mark.parametrize('setting',[{'rejected':True},{'browser_rejected':True}])
def test_failed_plans_or_execution_exhaust_without_silent_success(tmp_path,setting):
    result,controller,seen,rows=run_fixture(tmp_path,terminal_at=6,**setting)
    assert result.terminal_reason is TerminalReason.OPAQUE_VERIFIER_TERMINAL
    assert result.recovery_attempts==2 and not seen
    assert any(r['event_type']=='hybrid_continuation' and r['payload']['incident_status']=='exhausted_unresolved' for r in rows)


def test_abort_remains_terminal(tmp_path):
    result,controller,seen,rows=run_fixture(tmp_path,strategy=RecoveryStrategy.ABORT)
    assert result.terminal_reason is TerminalReason.ABORT and result.normal_actions==1


@pytest.mark.parametrize('different,cap',[(False,2),(True,4)])
def test_abort_remains_terminal_after_recovery_allowance_exhaustion(tmp_path,different,cap):
    result,controller,seen,rows=run_fixture(tmp_path,terminal_at=None,different_normal=different,abort_after=cap)
    assert result.terminal_reason is TerminalReason.ABORT
    assert result.recovery_attempts==cap and len(seen)==cap
    assert any(r['event_type']=='hybrid_continuation' and r['payload']['next']=='stop_abort' for r in rows)


def test_loop_stops_even_inside_recovery(tmp_path):
    result,controller,seen,rows=run_fixture(tmp_path,still=True,terminal_at=None)
    assert result.terminal_reason is TerminalReason.LOOP and result.executor_steps==3
    assert len(seen)==2  # learned assessment is still recorded after executed actions


def test_browser_terminal_takes_priority_over_loop(tmp_path):
    result,_,seen,_=run_fixture(tmp_path,still=True,terminal_at=3)
    assert result.terminal_reason is TerminalReason.OPAQUE_VERIFIER_TERMINAL and len(seen)==2


@pytest.mark.parametrize('setting',[{'positive':True},{'progress':True}])
def test_learned_positive_path_returns_to_normal_before_two_attempts(tmp_path,setting):
    result,controller,seen,rows=run_fixture(tmp_path,terminal_at=3,**setting)
    assert result.terminal_reason is TerminalReason.OPAQUE_VERIFIER_TERMINAL
    assert result.normal_actions==2 and result.recovery_attempts==1


def test_timeout_cannot_be_bypassed_by_continuation(tmp_path):
    result,_,_,_=run_fixture(tmp_path,timeout=True)
    assert result.terminal_reason is TerminalReason.TIMEOUT and result.normal_actions==1


def test_model_call_limit_stops_before_additional_browser_action(tmp_path):
    result,_,_,rows=run_fixture(tmp_path,terminal_at=None,extra_calls=40)
    dispatches=[r for r in rows if r['event_type']=='model_call_dispatch']
    assert result.terminal_reason is TerminalReason.POLICY_ERROR
    assert len(dispatches)==102 and result.executor_steps<30


def test_refreshed_control_identity_does_not_reset_failure_key():
    def obs(n):
        policy=observation(oid=f'e:obs:{n}')
        return Observation(observation_id=policy.observation_id,episode_id='e',stage=ObservationStage.RESET,
            screenshot_sha256='a'*64,screenshot_path=None,width=1000,height=600,url='http://fixture/',title='',page_state=policy.current_page_state)
    def action(n,value):
        return ConcreteAction(action_id=f'a{n}',source_decision_id=f'd{n}',action_type=ActionType.TYPE,
            bbox=(.1,.2,.3,.1),parameters={'text':value,'target_control_id':f'o{n}:c0'})
    assert EpisodeRunner._hybrid_failure_key(action(1,'x'),obs(1))==EpisodeRunner._hybrid_failure_key(action(2,'x'),obs(2))
    assert EpisodeRunner._hybrid_failure_key(action(1,'x'),obs(1))!=EpisodeRunner._hybrid_failure_key(action(2,'y'),obs(2))


def test_learned_progress_does_not_refresh_spent_attempts(tmp_path):
    result,controller,seen,rows=run_fixture(tmp_path,progress=True,terminal_at=7)
    assert result.terminal_reason is TerminalReason.OPAQUE_VERIFIER_TERMINAL
    assert result.failure_incidents==1 and result.recovery_attempts==2
    assert [x.attempt_id.rsplit(':',1)[1] for x in seen]==['1','2']


def test_learned_resolution_closes_binding_but_keeps_episode_cap(tmp_path):
    result,controller,seen,rows=run_fixture(tmp_path,positive=True,terminal_at=None)
    assert result.terminal_reason is TerminalReason.ACTION_BUDGET_EXHAUSTED
    assert result.recovery_attempts==4 and controller.episode_attempts==4
    assert len({x.incident_id for x in seen})==4


def test_episode_reset_clears_incident_registry_only_at_reset(tmp_path):
    result,_,_,_=run_fixture(tmp_path,terminal_at=None,rerun=True)
    assert result.recovery_attempts==2 and result.failure_incidents==1
