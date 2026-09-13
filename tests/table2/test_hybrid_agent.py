"""Hybrid adapter engineering checks. Scripted callbacks are not live results."""
from dataclasses import replace
import json
import random
from types import SimpleNamespace

import pytest

from web_agent.runtime.hybrid_action_policy import HybridActionPolicy, HybridActionContext, trained_advice
from web_agent.runtime.contracts import (
    ActionType, CausalHistoryEntry, ConcreteAction, ExecutionStatus,
    PreActionDecision, RuntimeTaskView,
)
from web_agent.runtime.policy import ActionParseError, CallablePolicyAdapter, PolicyError, PolicyKind
from web_agent.runtime.model_calls import (
    ModelCallLedger, ModelCallBudgetExceeded, activate_model_call_ledger, deactivate_model_call_ledger,
)
from web_agent.runtime.action_parameters import DeterministicParameterProvider
from tests.table2.test_development_v3 import observation
from tests.table2.test_development_v4 import view, ONE_HIT, TWO_HIT


def task(obs):
    return RuntimeTaskView(task_id=obs.task_id, goal=obs.goal)


def advice(obs):
    probabilities = dict(zip((a.value for a in ActionType), [.01, .02, .03, .04, .85, .05]))
    selected = max(probabilities, key=probabilities.get)
    return PreActionDecision(
        decision_id='trained', observation_id=obs.observation_id,
        action_type=ActionType(selected), action_probabilities=probabilities,
        bbox=(.01, .02, .1, .2), grounding_confidence=.85, confidence_before=.135,
        input_observation_ids=(obs.observation_id,), policy_id='trained', policy_version='v1')


def build(tmp_path, system='H1', raw=None, callback=None):
    calls = []
    def predict(t, o, rng):
        calls.append('advice')
        return callback(t, o, rng) if callback else advice(o)
    trained = CallablePolicyAdapter(policy_id='trained', policy_version='v1', kind=PolicyKind.TRAINED,
                                    action_predictor=predict, checkpoint_sha256='b'*64)
    prompts = []
    def generate(**kw):
        calls.append('generation'); prompts.append(kw)
        return raw if raw is not None else json.dumps(dict(action_type='TYPE', target='Field', bbox=None, value=' Exact  Case! '))
    actor = HybridActionPolicy(system=system, episode_id='episode', base=SimpleNamespace(_generate=generate),
                              evidence_dir=tmp_path/system, trained_policy=None if system=='H0' else trained)
    return actor, calls, prompts


def predict(actor, obs=None, **kwargs):
    obs = obs or observation()
    return actor.predict_action(task(obs), obs, rng=random.Random(42), **kwargs)


@pytest.mark.parametrize('system,count', [('H0',1),('H1',2),('H2',2),('H3',2)])
def test_advice_isolation_and_exact_dispatch_count(tmp_path, system, count):
    actor, calls, prompts = build(tmp_path, system)
    events=[]; ledger=ModelCallLedger('episode', sink=events.append); token=activate_model_call_ledger(ledger)
    try:
        decision=predict(actor)
    finally:
        deactivate_model_call_ledger(token)
    assert ledger.count==count==len(calls)
    context=json.loads(prompts[0]['suffix'].split(': ',1)[1])
    assert ('trained_advice' in context)==(system!='H0')
    assert all(e['stage']=='pre_action_policy' for e in events)
    assert decision.action_type is ActionType.TYPE
    assert decision.parameter_hints['text']==' Exact  Case! '
    assert decision.policy_id==f'miniwob-{system.lower()}-normal'
    receipt=json.loads(next(actor.evidence_dir.glob('*.json')).read_text())
    assert ('trained_advice' in receipt)==(system!='H0')
    assert json.loads(receipt['raw_response'])['value']==' Exact  Case! '
    assert receipt['resolved_proposal']['value']==' Exact  Case! '
    assert 'execution' not in receipt  # generation is not execution


def test_advice_projection_preserves_all_outputs_and_is_detached():
    original=advice(observation()); before=original.record_sha256
    projection=trained_advice(original,checkpoint_sha256='b'*64)
    assert projection['action_probabilities']==original.action_probabilities
    assert len(projection['action_probabilities'])==6
    assert projection['argmax']==original.action_type.value
    assert projection['bbox']==list(original.bbox)
    assert projection['grounding_confidence']==original.grounding_confidence
    assert projection['confidence_before']==original.confidence_before
    projection['action_probabilities']['CLICK']=0.777
    projection['bbox'][0]=.6
    assert original.record_sha256==before


def test_all_systems_share_prompt_and_controls(tmp_path):
    inputs=[]
    for system in ['H0','H1','H2','H3']:
        actor,_,prompts=build(tmp_path,system);predict(actor)
        context=json.loads(prompts[0]['suffix'].split(': ',1)[1]);context.pop('trained_advice',None)
        inputs.append((prompts[0]['prompt'],context))
    assert all(item==inputs[0] for item in inputs)


def test_budget_stops_before_second_dispatch_and_preserves_advice_receipt(tmp_path):
    actor,calls,_=build(tmp_path);ledger=ModelCallLedger('episode',maximum=1)
    token=activate_model_call_ledger(ledger)
    try:
        with pytest.raises(ModelCallBudgetExceeded):predict(actor)
    finally:deactivate_model_call_ledger(token)
    assert calls==['advice'] and ledger.count==1
    receipt=json.loads(next(actor.evidence_dir.glob('*.json')).read_text())
    assert 'trained_advice' in receipt and 'raw_response' not in receipt
    assert receipt['error_type']=='ModelCallBudgetExceeded'


@pytest.mark.parametrize('raw', [
    '{}', '{"action_type":"CLICK"}',
    '{"action_type":"CLICK","target":null,"bbox":null,"value":null} {}',
    '{"action_type":"CLICK","action_type":"TYPE","target":null,"bbox":null,"value":null}',
    '```json\n{"action_type":"CLICK","target":null,"bbox":null,"value":null}\n```',
])
def test_one_exact_object_required_and_rejection_charged(tmp_path,raw):
    actor,calls,_=build(tmp_path,raw=raw);ledger=ModelCallLedger('episode');token=activate_model_call_ledger(ledger)
    try:
        with pytest.raises(ActionParseError):predict(actor)
    finally:deactivate_model_call_ledger(token)
    assert ledger.count==2 and calls==['advice','generation']
    receipt=json.loads(next(actor.evidence_dir.glob('*.json')).read_text())
    assert receipt['raw_response']==raw and receipt['status']=='REJECTED'


@pytest.mark.parametrize('target,point', [('o2:c0',ONE_HIT),('o2:c1',TWO_HIT)])
def test_same_selected_overlapping_control_reaches_existing_provider(tmp_path,target,point):
    obs=view(); raw=json.dumps(dict(action_type='CLICK',target=target,bbox=None,value=None))
    actor,_,_=build(tmp_path,raw=raw);dec=predict(actor,obs)
    params=DeterministicParameterProvider().resolve(task(obs),obs,dec,rng=random.Random(42))
    assert params.values['target_control_id']==target
    assert [params.values['target_x'],params.values['target_y']]==point


@pytest.mark.parametrize('target', ['o0:c0','missing'])
def test_stale_or_missing_id_is_not_repaired(tmp_path,target):
    actor,_,_=build(tmp_path,raw=json.dumps(dict(action_type='TYPE',target=target,bbox=None,value='x')))
    with pytest.raises(ActionParseError):predict(actor)


@pytest.mark.parametrize('flag', ['disabled','readonly'])
def test_incompatible_controls_rejected(tmp_path,flag):
    obs=observation();obs.current_page_state['visible_controls'][0][flag]=True
    actor,_,_=build(tmp_path)
    with pytest.raises(ActionParseError):predict(actor,obs)


def test_coordinate_only_prediction_is_not_moved(tmp_path):
    box=[.5,.5,.1,.1]
    actor,_,_=build(tmp_path,raw=json.dumps(dict(action_type='CLICK',target=None,bbox=box,value=None)))
    assert predict(actor).bbox==tuple(box)


def with_history():
    actions=tuple(ConcreteAction(action_id=f'a{i}',source_decision_id=f'd{i}',action_type=ActionType.TYPE,
                                parameters={'text':f' Exact {i} ','target_control_id':f'o{i}:c0'},
                                bbox=(.1,.2,.3,.1)) for i in [1,2])
    histories=tuple(CausalHistoryEntry(history_index=i,action_id=a.action_id,action_type=a.action_type,
        action_target_fingerprint=a.fingerprint,execution_status=ExecutionStatus.EXECUTED,executor_step=i,
        state_changed=True,environment_error=False,execution_error_sha256=None,
        post_observation_id=f'episode:obs:{i+1}',post_observation_sha256='c'*64,post_screenshot_sha256='d'*64)
        for i,a in enumerate(actions,1))
    obs=replace(observation(oid='episode:obs:3'),causal_history=histories)
    return obs,actions


def test_completed_exact_actions_and_rejection_are_detached_and_causal(tmp_path):
    obs,actions=with_history()
    rejection=dict(observation_id=obs.observation_id,stage='parameter_resolution',code='MISSING_VALUE',detail='TYPE needs text')
    ctx=HybridActionContext('episode',obs.observation_id,actions,rejection)
    actor,_,prompts=build(tmp_path);predict(actor,obs,context=ctx)
    data=json.loads(prompts[0]['suffix'].split(': ',1)[1])
    assert [a['parameters']['text'] for a in data['completed_actions']]==[' Exact 1 ',' Exact 2 ']
    assert data['last_rejection']==rejection
    data['completed_actions'][0]['parameters']['text']='changed'
    assert actions[0].parameters['text']==' Exact 1 '
    with pytest.raises(PolicyError,match='complete causal'):predict(actor,obs)
    with pytest.raises(PolicyError,match='fingerprint'):
        predict(actor,obs,context=replace(ctx,completed_actions=(replace(actions[0],parameters={'text':'future'}),actions[1])))
    with pytest.raises(PolicyError,match='episode'):predict(actor,obs,context=replace(ctx,episode_id='other'))
    with pytest.raises(PolicyError,match='observation'):predict(actor,obs,context=replace(ctx,observation_id='future'))
    with pytest.raises(PolicyError,match='rejection'):
        predict(actor,obs,context=replace(ctx,last_rejection={**rejection,'observation_id':'future'}))


def test_mutating_trained_callback_is_detected_before_generation(tmp_path):
    def corrupt(t,o,rng):
        o.current_page_state['extra']='mutation'
        return advice(o)
    actor,calls,_=build(tmp_path,callback=corrupt);obs=observation();before=obs.record_sha256
    with pytest.raises(PolicyError,match='mutated'):predict(actor,obs)
    assert calls==['advice'] and obs.record_sha256==before


@pytest.mark.parametrize('system',['H0','H1'])
def test_no_learned_recovery_in_h0_h1(tmp_path,system):
    actor,calls,_=build(tmp_path,system)
    with pytest.raises(PolicyError):actor.assess_transition(None,None,rng=random.Random(42))
    with pytest.raises(PolicyError):actor.assess_recovery(None,None,rng=random.Random(42))
    assert not calls


def test_receipts_cannot_overwrite_existing_evidence(tmp_path):
    actor,_,_=build(tmp_path);predict(actor)
    before=next(actor.evidence_dir.glob('*.json')).read_bytes()
    with pytest.raises(FileExistsError):build(tmp_path)
    assert next(actor.evidence_dir.glob('*.json')).read_bytes()==before


@pytest.mark.parametrize('kind,value', [('CLICK','Field'),('TYPE',' Exact  Case! '),
    ('SELECT','two'),('SCROLL','down'),('NAVIGATE','https://example.org/'),('PRESS_KEY','TAB')])
def test_six_action_classes_and_issued_values_survive(tmp_path,kind,value):
    obs=observation(kind='select' if kind=='SELECT' else 'text')
    raw=json.dumps(dict(action_type=kind,target='Field' if kind in {'CLICK','TYPE','SELECT'} else None,
                        bbox=None,value=value))
    actor,_,_=build(tmp_path,raw=raw);dec=predict(actor,obs)
    assert dec.action_type.value==kind and dec.parameter_hints['value']==value


def test_non_argmax_advice_is_rejected():
    original=advice(observation())
    wrong=next(a for a in ActionType if a!=original.action_type)
    with pytest.raises(PolicyError,match='unmasked'):
        trained_advice(replace(original,action_type=wrong),checkpoint_sha256='b'*64)


def test_h0_rejects_advice_dependency_and_masked_backend_rejected(tmp_path):
    actor,_,_=build(tmp_path)
    with pytest.raises(ValueError,match='H0'):
        HybridActionPolicy(system='H0',episode_id='episode',base=actor.base,evidence_dir=tmp_path/'h0',
                           trained_policy=actor.trained_policy)
    class Masked:
        restrict_to_executable_actions=True
        def predict(self,*args):raise AssertionError('must not run')
    with pytest.raises(ValueError,match='masking'):
        HybridActionPolicy(system='H1',episode_id='episode',base=actor.base,evidence_dir=tmp_path/'masked',
                           trained_policy=replace(actor.trained_policy,action_predictor=Masked().predict))


def test_trained_callback_receives_unchanged_pre_action_observation(tmp_path):
    obs=observation(); original=obs.record_sha256
    def unchanged(t,o,rng):
        assert o.record_sha256==original
        assert t==task(obs)
        return advice(o)
    actor,_,_=build(tmp_path,callback=unchanged);predict(actor,obs)
    assert obs.record_sha256==original
