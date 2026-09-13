"""No model inference: semantic parity, exact parsing and production-loop replay."""
from dataclasses import replace
import json
import random
from types import SimpleNamespace

import pytest

from web_agent.runtime.hybrid_action_policy import HybridActionPolicy, HybridActionContext
from web_agent.runtime.hybrid_interface import semantic_context, parse_action
from web_agent.runtime.policy import ActionParseError, CallablePolicyAdapter, PolicyKind
from tests.table2.test_hybrid_agent import advice, task, with_history
from tests.table2.test_development_v3 import observation
from tests.table2.test_hybrid_runner import engineering_run


def actor(tmp_path, system, *, raw=None, elapsed=1.):
    seen=[]
    def generate(**kw):
        seen.append(kw)
        return raw or '```json\n{"action_type":"TYPE","target":"Field","bbox":null,"value":" Exact  Case! "}\n```'
    trained=CallablePolicyAdapter(policy_id='trained',policy_version='v1',kind=PolicyKind.TRAINED,
        checkpoint_sha256='b'*64,action_predictor=lambda t,o,rng:replace(advice(o),latency_ms=elapsed,
            decision_id=f'{system}:timed-decision'))
    result=HybridActionPolicy(system=system,episode_id=f'campaign:{system}',
        base=SimpleNamespace(_generate=generate),evidence_dir=tmp_path/system,
        trained_policy=None if system=='H0' else trained,interface_version=2)
    return result,seen


def test_realistic_system_ids_and_latency_do_not_enter_generation(tmp_path):
    inputs=[];receipts=[]
    for i,s in enumerate(('H0','H1','H2','H3')):
        a,seen=actor(tmp_path,s,elapsed=240.+i)
        o=observation(oid=f'campaign:{s}:obs:1')
        d=a.predict_action(task(o),o,rng=random.Random(42))
        assert d.parameter_hints['text']==' Exact  Case! '
        inputs.append((seen[0]['prompt'],json.loads(seen[0]['suffix'].split(': ',1)[1])))
        receipts.append(json.loads(next(a.evidence_dir.glob('*.json')).read_text()))
    assert inputs[1]==inputs[2]==inputs[3]
    assert {k:v for k,v in inputs[1][1].items() if k!='trained_advice'}==inputs[0][1]
    assert 'trained_advice' not in inputs[0][1]
    assert receipts[2]['audit_context']!=receipts[3]['audit_context']
    assert receipts[2]['trained_advice']['source_decision_sha256']!=receipts[3]['trained_advice']['source_decision_sha256']
    assert 'source_decision_sha256' not in json.dumps(inputs)
    assert 'campaign:H' not in json.dumps(inputs)


def test_repeated_same_observation_ignores_only_provenance(tmp_path):
    a,seen=actor(tmp_path,'H2');o=observation(oid='campaign:H2:obs:1')
    counter=iter((1., 2.))
    a.trained_policy=replace(a.trained_policy,action_predictor=lambda t,o,rng:
        replace(advice(o),latency_ms=next(counter)))
    for _ in range(2):a.predict_action(task(o),o,rng=random.Random(42))
    assert seen[0]['suffix']==seen[1]['suffix']
    receipts=[json.loads(p.read_text()) for p in sorted(a.evidence_dir.glob('*.json'))]
    assert receipts[0]['trained_advice']['source_decision_sha256']!=receipts[1]['trained_advice']['source_decision_sha256']


def test_history_targets_values_and_diagnostics_survive_projection():
    o,actions=with_history()
    raw=HybridActionContext('episode',o.observation_id,actions,
        dict(observation_id=o.observation_id,stage='parameter_resolution',code='MISSING_VALUE',
             detail='rejected episode:obs:3 target o3:c0')).project(o)
    clean=semantic_context(raw,episode_id='episode')
    assert [a['parameters'] for a in clean['completed_actions']]==[dict(a.parameters) for a in actions]
    assert [a['bbox'] for a in clean['completed_actions']]==[list(a.bbox) for a in actions]
    assert clean['last_rejection']==dict(stage='parameter_resolution',code='MISSING_VALUE',detail='rejected <episode>:obs:3 target o3:c0')
    assert all('action_id' not in a and 'post_observation_id' not in a for a in clean['completed_actions'])
    clean['completed_actions'][0]['parameters']['text']='mutated'
    assert raw['completed_actions'][0]['parameters']['text']==' Exact 1 '


@pytest.mark.parametrize('kind,value',[('CLICK',None),('TYPE',' Exact Case! '),('SELECT','two'),
    ('SCROLL','down'),('NAVIGATE','https://example.org/'),('PRESS_KEY','TAB')])
def test_single_fence_preserves_all_six_actions(kind,value):
    o=observation(kind='select' if kind=='SELECT' else 'text')
    fields=dict(action_type=kind,target='Field' if kind in {'CLICK','TYPE','SELECT'} else None,bbox=None,value=value)
    bare=json.dumps(fields); parsed,resolution=parse_action('```json\n'+bare+'\n```',o)
    assert parsed==parse_action(bare,o)[0]
    assert parsed['action_type']==kind and parsed['value']==value
    assert resolution['output_envelope']=='single_json_fence'


@pytest.mark.parametrize('raw',[
    '```json\n{}\n```',
    '```json\n{"action_type":"CLICK","target":"Field","bbox":null,"value":null} {}\n```',
    '```json\n{"action_type":"CLICK","action_type":"TYPE","target":"Field","bbox":null,"value":null}\n```',
    'Explanation\n```json\n{"action_type":"CLICK","target":"Field","bbox":null,"value":null}\n```',
    '{"action_type":"CLICK|TYPE|SELECT|SCROLL|NAVIGATE|PRESS_KEY","target":null,"bbox":null,"value":null}',
    '```json\n[{"action_type":"CLICK","target":"Field","bbox":null,"value":null}]\n```',
    '{"action_type":"TYPE","target":"Field","bbox":null,"value":["x"]}',
    '{"action_type":["CLICK","TYPE"],"target":"Field","bbox":null,"value":null}',
])
def test_no_action_invention_or_ambiguous_envelope(raw):
    with pytest.raises(ActionParseError):parse_action(raw,observation())


@pytest.mark.parametrize('target',['o0:c0','missing'])
def test_stale_targets_still_reject(target):
    with pytest.raises(ActionParseError):
        parse_action(json.dumps(dict(action_type='TYPE',target=target,bbox=None,value='x')),observation())


def test_disabled_and_coordinates_keep_executor_contract():
    o=observation();o.current_page_state['visible_controls'][0]['disabled']=True
    with pytest.raises(ActionParseError):
        parse_action(json.dumps(dict(action_type='TYPE',target='Field',bbox=None,value='x')),o)
    box=[.5,.5,.1,.1]
    assert parse_action(json.dumps(dict(action_type='CLICK',target=None,bbox=box,value=None)),o)[0]['bbox']==box


@pytest.mark.parametrize('invalid_recovery',[False,True])
def test_v2_complete_controller_memory_feedback_and_audit(tmp_path,monkeypatch,invalid_recovery):
    root=tmp_path/'run'
    _,rows,audit=engineering_run(root,monkeypatch,interface_version=2,invalid_recovery=invalid_recovery)
    assert audit['status']=='PASS',audit['errors']
    assert audit['live_path_verified'] is False
    assert all('error' not in r for r in rows)
    assert sum(r['summary']['memory_queries'] for r in rows)>0
    for path in root.glob('*/repeat-0/H*/named-recovery-outputs/*.json'):
        r=json.loads(path.read_text());suffix=r['input_suffix']
        assert 'planner_context_sha256' not in suffix and 'attempt_id' not in suffix
        assert 'action_id' not in suffix and 'miniwob-label-memory-development' not in suffix
        assert r['audit_context']['context']['planner_context_sha256']


def test_v2_auditor_rejects_model_visible_provenance(tmp_path,monkeypatch):
    import hashlib
    from web_agent.eval.table2.miniwob_reporting import audit_and_report
    root=tmp_path/'run';_,_,audit=engineering_run(root,monkeypatch,interface_version=2)
    assert audit['status']=='PASS'
    path=next(root.glob('*/repeat-0/H3/hybrid-normal-outputs/proposal-0001.json'))
    receipt=json.loads(path.read_text())
    context=json.loads(receipt['input_suffix'].split(': ',1)[1]);context['episode_id']=receipt['episode_id']
    receipt['input_suffix']='\nhybrid_action_context: '+json.dumps(context,sort_keys=True)
    receipt['input_suffix_sha256']=hashlib.sha256(receipt['input_suffix'].encode()).hexdigest()
    path.write_text(json.dumps(receipt))
    assert audit_and_report(root,report_dir=tmp_path/'tamper')['status']=='FAIL'


def test_recovery_semantics_match_before_memory_and_keep_missing_content(tmp_path,monkeypatch):
    root=tmp_path/'run';_,_,audit=engineering_run(root,monkeypatch,interface_version=2)
    assert audit['status']=='PASS'
    for h2 in root.glob('*/repeat-0/H2/named-recovery-outputs/*.json'):
        h3=h2.parent.parent.parent/'H3/named-recovery-outputs'/h2.name
        contexts=[]
        for p in (h2,h3):
            r=json.loads(p.read_text())
            context=json.loads(r['input_suffix'].split('\nnext_action_request: ')[0].split(': ',1)[1])
            for e in context.get('retrieved_training_examples',[]):
                assert e['recovery_action_value'] is None and e['reflection'] is None
            context.pop('retrieved_training_examples',None);context.pop('memory_usage',None)
            contexts.append(context)
        assert contexts[0]==contexts[1]
