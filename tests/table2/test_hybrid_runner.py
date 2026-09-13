"""Production H runner with scripted backends: engineering, never performance evidence."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from threading import Lock
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from web_agent.benchmarks.base import BenchmarkAdapter, AdapterExecution
from web_agent.benchmarks.miniwob_controls import bind_controls, compatible_actions
from web_agent.runtime.contracts import (ActionType, HybridSystemID, SystemID, EpisodeSummary,
    Observation, ObservationStage, OpaqueTerminalSignal, ExecutionStatus,
    TransitionAssessment, RecoveryAssessment, RecoveryStrategy, TaskSpecification)
from web_agent.runtime.observation import ProcessorParityContract
from web_agent.runtime.protocol import switches_for, SystemSwitches
from web_agent.runtime.action_parameters import build_registered_hybrid_parameter_provider
from web_agent.runtime.recovery.strategies import build_recovery_target_evidence
from tests.table2.test_development_v3 import observation
from tests.table2.test_hybrid_agent import advice

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/eval/table2/miniwob_hybrid_dev_v1.json'


class ScriptedPage(BenchmarkAdapter):
    benchmark_id='miniwob';benchmark_version='0.14.3'
    def __init__(self,folder,**kwargs):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.n=0;self.i=0;self.value='';self.outcomes=[]
        self.hybrid_interface_version=kwargs.get('hybrid_interface_version',1)
    def reset(self,task,*,episode_id,seed):
        self.task=task;self.eid=episode_id
        return self.observe(stage=ObservationStage.RESET)
    def observe(self,*,stage,prior_action_id=None):
        self.i+=1;oid=f'{self.eid}:obs:{self.i}'
        state=observation(self.value,oid=oid).current_page_state
        if self.hybrid_interface_version==3:state['hybrid_interface_version']=3
        field=dict(state['visible_controls'][0])
        choice={**field,'source_id':'choice','input_type':'checkbox','checked':self.n>=2,'target_bbox':[.5,.2,.1,.1],
            'accessible_names':[{'value':'Choice','source':'associated_label'}]}
        finish={**field,'source_id':'finish','tag':'a','input_type':'','link_like':True,'target_bbox':[.7,.2,.1,.1],
            'accessible_names':[{'value':'Finish','source':'text'}],'text':'Finish','destination':'http://fixture/finish'}
        if self.hybrid_interface_version==3:
            # Scripted geometry explicitly states its hit evidence; real
            # browser fixtures obtain this from elementFromPoint instead.
            for c in (field,choice,finish):
                x,y,w,h=c['target_bbox'];c['hit_point']=[x+w/2,y+h/2]
        state['visible_controls']=bind_controls({'controls':[field,choice,finish]},oid)
        state['recovery_target_evidence']=build_recovery_target_evidence(task=self.task,observation_id=oid,
            compatible_actions=compatible_actions(state['visible_controls']))
        path=self.folder/f'observation-{self.i}.png';path.write_bytes(f'engineering image {self.n} {self.value}'.encode());path.chmod(0o444)
        obs=Observation(observation_id=oid,episode_id=self.eid,stage=stage,prior_action_id=prior_action_id,
            screenshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),screenshot_path=str(path),width=1000,height=600,
            url='http://fixture/',title='',page_state=state)
        (self.folder/f'observation-{self.i}.json').write_text(json.dumps(obs.to_dict()))
        return obs
    def execute(self,action):
        self.n+=1
        if action.action_type is ActionType.TYPE:self.value=action.parameters['text']
        self.outcomes.append({'terminated':self.n>=4,'truncated':False,'raw_reward':1. if self.n>=4 else 0.})
        (self.folder/'browser-outcomes.json').write_text(json.dumps(self.outcomes))
        return AdapterExecution(status=ExecutionStatus.EXECUTED,state_changed=True)
    def terminal_signal(self,task,binding=None):
        return OpaqueTerminalSignal(event_id=f'stop:{self.i}',token_sha256='f'*64,terminate=self.n>=4)
    def close(self):pass


class ScriptedBase:
    def __init__(self,*,invalid_first=False,invalid_recovery=False):
        self.invalid_first=invalid_first;self.invalid_recovery=invalid_recovery;self.calls={};self.model=None;self.processor=None;self.torch=torch;self.lock=Lock()
    def predict_action(self,*args,**kw):raise AssertionError('H must use the shared hybrid actor')
    def _generate(self,*,prompt,task,observation,suffix):
        if self.invalid_recovery and suffix.startswith('\ncausal_recovery_context:'):return '{}'
        oid=observation.observation_id;self.calls[oid]=self.calls.get(oid,0)+1
        if self.invalid_first and oid.endswith(':1') and self.calls[oid]==1:return '{}'
        n=len(observation.causal_history)
        kind,source,value=('CLICK','field',None)
        if n==1:kind,source,value=('TYPE','field',' Exact  Case! ') if task.task_id.endswith('text') else ('CLICK','choice',None)
        if n==2:kind,source,value=('TYPE','field','Completed text') if task.task_id.endswith('text') else ('CLICK','field',None)
        if n>=3:kind,source,value=('CLICK','finish',None)
        control=next(c for c in observation.current_page_state['visible_controls'] if c['source_id']==source)
        return json.dumps(dict(action_type=kind,target=control['control_id'],bbox=None,value=value))


class ScriptedTrained:
    def __init__(self,config):
        self.checkpoint_sha256=config['checkpoint_and_base_bindings']['checkpoint_sha256']
        self.lock=Lock();self.torch=torch;self.batch=NS(transition=lambda *args:{})
        v=np.load(Path(config['asset_paths']['memory'])/'embeddings.npy',mmap_mode='r')[987]
        self.model=NS(memory_embedding=lambda batch:torch.from_numpy(np.array(v,copy=True)).unsqueeze(0))
    def action(self,t,o,rng):
        return replace(advice(o),decision_id='trained:'+o.observation_id,policy_id='validation-selected-web-agent')
    def diagnose(self,t,x,rng):
        return TransitionAssessment(assessment_id='diag:'+x.executed_action.action_id,
            pre_observation_id=x.pre_observation.observation_id,post_observation_id=x.post_observation.observation_id,
            executed_action_id=x.executed_action.action_id,predicted_failure=True,failure_probability=1.,
            failure_type='LOOP_DETECTED',failure_type_probabilities={'LOOP_DETECTED':1.},needs_recovery=True,
            needs_recovery_probability=1.,recovery_strategy=RecoveryStrategy.REPLAN,recovery_probabilities={'REPLAN':1.})
    def assess(self,t,x,rng):
        return RecoveryAssessment(assessment_id='assess:'+x.attempt_id,incident_id=x.incident_id,attempt_id=x.attempt_id,
            pre_recovery_observation_id=x.pre_recovery_observation.observation_id,
            post_recovery_observation_id=x.post_recovery_observation.observation_id,
            recovery_action_ids=tuple(a.action_id for a in x.recovery_actions),predicted_failure_resolved=False,
            predicted_resolution_probability=0.,predicted_progress=False,predicted_progress_probability=0.)


def engineering_run(root,monkeypatch,*,invalid_first=False,invalid_recovery=False,page_class=ScriptedPage,
                    interface_version=1):
    from web_agent.eval.table2 import miniwob_study as study, miniwob_model
    from web_agent.benchmarks import browsergym_miniwob
    config=json.loads(CONFIG.read_text())
    if interface_version != 1:
        config['hybrid_interface_version'] = interface_version
        config['profile'] = f'miniwob-hybrid-dev-v{interface_version}-engineering'
    if interface_version==3:
        config['development_revision'].update(hybrid_proposal_repair=True,stable_retry_targets=True)
    protocol=study.runtime_protocol(config['development_campaign_id'],hybrid=True)
    blocks=[]
    for name in ('fixture-text','fixture-click'):
        task=TaskSpecification(task_id='miniwob.'+name,goal='Scripted engineering sequence',benchmark_id='miniwob',benchmark_version='0.14.3',start_state_id='seeded-reset',development_partition=True)
        blocks.append({'task':name,'repeat_id':0,'stage_reset_seed':42,'browser_reset_seed':42,'task_specification':task.to_dict()})
    plan={'evidence_type':'SCRIPTED_ENGINEERING','actual_model_inferences':0,'profile':config['profile'],'phase':'development','config':config,'blocks':blocks,'systems':config['systems'],
        'episodes_planned':8,'runtime_protocol':protocol.to_dict(),'scope':'SCRIPTED ENGINEERING ONLY; zero model inference',
        'recovery_prompt_path':str(ROOT/'configs/eval/table2'/('miniwob_recovery_prompt_v4.txt' if interface_version==3 else 'miniwob_recovery_prompt_v3.txt')),
        'analysis':{'contrasts':config['primary_contrasts'],'secondary_contrasts':config['secondary_contrasts']}}
    trained=ScriptedTrained(config);base=ScriptedBase(invalid_first=invalid_first,invalid_recovery=invalid_recovery)
    processor=ProcessorParityContract.from_dict(json.loads((Path(config['asset_paths']['export'])/'processor_contract.json').read_text()))
    bundle=NS(selected=NS(action_predictor=trained.action,transition_predictor=trained.diagnose,recovery_predictor=trained.assess,processor_contract=processor),
        e0=NS(action_predictor=base.predict_action,processor_contract=processor))
    def no_fallback(*args,**kwargs):raise AssertionError('scripted proposals have complete parameters')
    provider=build_registered_hybrid_parameter_provider(fallback_resolver=no_fallback,fallback_policy_id='engineering-unused',fallback_policy_version='v1',prompt_text='engineering',decoding_parameters={})
    # These substitutions are confined to this engineering harness. Production
    # prepare/run always verify source/assets and load the existing checkpoint.
    monkeypatch.setattr(study,'verify_plan',lambda *a,**kw:plan)
    from web_agent.runtime.qwen2vl_pc01 import _UnadaptedBaseRuntime
    monkeypatch.setattr(_UnadaptedBaseRuntime,'_generate',lambda self,**kw:base._generate(**kw))
    if interface_version==3:
        monkeypatch.setattr(_UnadaptedBaseRuntime,'_generate_with_metadata',lambda self,**kw:
            (base._generate(**kw),{'available':False,'reason':'scripted_engineering_zero_model_inference'}))
    # No GPU/model use in this explicitly scripted fixture. The production
    # runner's GPU inventory and model loader remain enabled for real runs.
    check_output=study.subprocess.check_output
    monkeypatch.setattr(study.subprocess,'check_output',lambda args,**kw:
        '' if args[0]=='nvidia-smi' else check_output(args,**kw))
    monkeypatch.setattr(miniwob_model,'load_models',lambda c:(bundle,provider,{'checkpoint_sha256':trained.checkpoint_sha256}))
    monkeypatch.setattr(browsergym_miniwob,'MiniWoBAdapter',page_class)
    root.mkdir(parents=True,exist_ok=False)
    (root/'plan.json').write_text(json.dumps(plan,indent=2))
    study.run(root)
    return plan,json.loads((root/'results.json').read_text()),json.loads((root/'audit.json').read_text())


def test_hybrid_switch_roundtrip_and_historical_enum_unchanged():
    assert [s.value for s in SystemID]==['E0','E1','E2','E3']
    for s in list(SystemID)+list(HybridSystemID):
        flags=switches_for(s)
        assert SystemSwitches.from_dict(flags.to_dict())==flags
        assert flags.evaluation_memory_write is False


def test_frozen_h_scope_and_development_only(tmp_path,monkeypatch):
    from web_agent.eval.table2.hybrid_study import frozen_scope
    from web_agent.eval.table2 import miniwob_study as study
    c=json.loads(CONFIG.read_text());scope=frozen_scope(c)
    assert len(scope['blocks'])==6 and scope['eligible_episodes']==24
    monkeypatch.setattr(study,'CONFIG',CONFIG)
    with pytest.raises(ValueError,match='development only'):study.prepare(tmp_path/'bad','evaluation')
    c['development_campaign_id']='wrong'
    with pytest.raises(AssertionError):frozen_scope(c)


@pytest.mark.parametrize('invalid_first',[False,True])
def test_complete_production_h_runner_and_audit(tmp_path,monkeypatch,invalid_first):
    plan,rows,audit=engineering_run(tmp_path/'run',monkeypatch,invalid_first=invalid_first)
    assert audit['status']=='PASS',audit['errors']
    for row in rows:
        assert 'error' not in row,row
        s=EpisodeSummary.from_dict(row['summary'])
        assert s.system_id is HybridSystemID(row['system']) and s.valid_for_primary
        assert s.executor_steps==4+invalid_first
        assert s.recovery_attempts==(2 if row['system'] in {'H2','H3'} else 0)
        assert s.memory_queries==(2 if row['system']=='H3' else 0)
    analysis=json.loads((tmp_path/'run/analysis.json').read_text())
    assert [c['contrast'] for c in analysis['paired_contrasts']]==['H2 minus H1','H3 minus H2']
    assert analysis['secondary_contrasts'][0]['contrast']=='H1 minus H0'


def test_recovery_parse_feedback_survives_exhaustion_into_next_normal_request(tmp_path,monkeypatch):
    root=tmp_path/'run'
    _,rows,audit=engineering_run(root,monkeypatch,invalid_recovery=True)
    assert audit['status']=='PASS',audit['errors']
    for row in rows:
        if row['system'] not in {'H2','H3'}:continue
        assert row['summary']['recovery_attempts']==4
        assert row['summary']['recovery_actions']==0
        receipts=sorted((root/row['task']/'repeat-0'/row['system']/'hybrid-normal-outputs').glob('proposal-*.json'))
        second=json.loads(receipts[1].read_text())
        context=json.loads(second['input_suffix'].split(': ',1)[1])
        assert context['last_rejection']['stage']=='parsing'
        assert context['last_rejection']['observation_id']==context['observation_id']
        assert len(context['completed_actions'])==1


def test_missing_pairs_and_policy_failures_keep_h_denominators():
    from web_agent.eval.table2.miniwob_reporting import analyze
    fields=dict(executor_steps=1,recovery_actions=0,recovery_attempts=0,memory_queries=0,
        memory_interventions=0,model_call_count=1,elapsed_seconds=1,valid_for_primary=True,environment_failure=False)
    plan={'systems':['H0','H1','H2','H3'],'blocks':[{'task':'a','repeat_id':0},{'task':'b','repeat_id':0}],
        'scope':'engineering','analysis':{'contrasts':[['H1','H2'],['H2','H3']],'secondary_contrasts':[['H0','H1']]}}
    rows=[{'task':t,'repeat_id':0,'system':s,'success':s in {'H2','H3'},'summary':fields} for t in ('a','b') for s in plan['systems']]
    result=analyze(plan,rows)
    assert result['all_included_episodes_present'] and result['paired_contrasts'][0]['paired_instances']==2
    assert result['systems'][1]['episodes']==2 and result['systems'][1]['completed']==0
    missing=analyze(plan,rows[:-1]);assert not missing['all_included_episodes_present']
    assert missing['paired_contrasts'][0]['paired_instances']==1
    with pytest.raises(ValueError,match='duplicate'):analyze(plan,rows+[rows[0]])
    excluded=analyze(plan,rows[:4]+[{'task':'b','repeat_id':0,'status':'EXCLUDED_TRAIN_OVERLAP'}])
    assert excluded['all_included_episodes_present'] and excluded['paired_contrasts'][0]['paired_instances']==1


def test_independent_audit_rejects_changed_advice_and_action_receipt(tmp_path,monkeypatch):
    from web_agent.eval.table2.hybrid_reporting import audit_hybrid_episode
    from web_agent.eval.table2.miniwob_reporting import audit_and_report
    root=tmp_path/'run';plan,rows,audit=engineering_run(root,monkeypatch)
    assert audit['status']=='PASS'
    row=next(r for r in rows if r['system']=='H1');folder=root/row['task']/'repeat-0/H1'
    path=folder/'hybrid-normal-outputs/proposal-0001.json';original=path.read_bytes()
    receipt=json.loads(original);receipt['trained_advice']['argmax']='CLICK'
    path.write_text(json.dumps(receipt))
    binding=json.loads((folder.parent/'task-binding.json').read_text())
    with pytest.raises(AssertionError):audit_hybrid_episode(plan,row,folder,binding)
    path.write_bytes(original)
    path=folder/'hybrid-action-records/action-0001.json'
    path.chmod(0o644);path.write_text('{}')
    revised=audit_and_report(root,report_dir=tmp_path/'tamper-audit')
    assert revised['status']=='FAIL' and any('raw action receipt changed' in e['error'] for e in revised['errors'])
