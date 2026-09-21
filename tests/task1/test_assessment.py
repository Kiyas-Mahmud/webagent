"""Engineering fixtures only. No image assets, trained models or live results."""
import copy
import json
from pathlib import Path
import pytest
from web_agent.eval.task1.core import (make_views,validate_request,parse_response,execute,
                                      request_key,read,SYSTEMS,PHASES,write_new)
from web_agent.eval.task1.backends import prompt,method_instruction,require_local_images
from web_agent.eval.task1.reporting import metrics,score,collect_records,paired_interval

def test_selected_pc_matches_notebook_candidate_and_report():
    from web_agent.eval.task1.core import ROOT, CHECKPOINT_HASH
    contract = read(ROOT / 'configs/eval/task1/pc03_run_contract.json')
    reference = read(ROOT / 'configs/eval/task1/pc03_reference.json')
    notebook = read(ROOT / 'notebooks/dgx_three_model_comparison.ipynb')
    launch = [c for c in notebook['cells']
              if '# 9. PC 3 only' in ''.join(c.get('source', []))]
    assert len(launch) == 1
    candidate = contract['candidate']['model_id']
    assert candidate == 'internvl35_8b_gold_v2_8_dgx'
    assert f"run_candidate('{candidate}', phase='auto')" in ''.join(launch[0]['source'])
    assert f'/runs/{candidate}/' in reference['source_report_path']
    assert reference['checkpoint_sha256'] == CHECKPOINT_HASH
    assert 'pc03' in SYSTEMS and 'pc02' not in SYSTEMS

@pytest.fixture
def rows():
    def row(step,action,outcome,recovery=None):
        return {'meta':{'sample_id':f't:{step}','task_id':'t','step_index':step,'secret':'SENTINEL'},
                'inputs':{'state_before':f'images/b{step}.png','state_after':f'images/a{step}.png',
                          'task_description':'Fill the form','website_domain':'example.org','secret':'SENTINEL'},
                'labels':{'action_type':action,'outcome_label':outcome,'failure_type_4':'ACTION_MISMATCH',
                          'recovery_strategy':'REPLAN' if recovery is not None else 'NONE',
                          'recovery_success':recovery,'memory_update_flag':True,'action_target_bbox':{'x':1},
                          'secret':'SENTINEL'}}
    return [row(0,'CLICK','FAILURE',False),row(1,'TYPE','FAILURE')]

def test_correct_recovery_boundary_and_no_labels(rows):
    views,targets,_=make_views(rows,['t:0'])
    assert len(views)==2
    assert views[1]['before_image']=='images/a0.png'
    assert views[1]['after_image']=='images/a1.png'
    assert views[1]['executed_action']=={'type':'TYPE','value':None}
    assert 'SENTINEL' not in json.dumps(views)
    assert targets[request_key(views[1])]['outcome_label']=='FAILURE'
    assert 'FAILURE' not in json.dumps(views)

def test_nonconsecutive_dependency_rejected(rows):
    rows[1]['meta']['step_index']=3
    with pytest.raises(ValueError,match='linkage'): make_views(rows,['t:0'])

def test_missing_dependency_rejected(rows):
    with pytest.raises(ValueError,match='linkage'):make_views(rows[:1],['t:0'])

@pytest.mark.parametrize('extra',['outcome_label','failure_type_4','memory_update_flag','stratum','bbox'])
def test_request_allowlist(rows,extra):
    r=make_views(rows,['t:0'])[0][0];r[extra]='answer'
    with pytest.raises(ValueError):validate_request(r)

@pytest.mark.parametrize('path',['../answer.png','/absolute.png','images/../private.png','labels.json'])
def test_no_image_path_escape(rows,path):
    r=make_views(rows,['t:0'])[0][0];r['before_image']=path
    with pytest.raises(ValueError):validate_request(r)

@pytest.mark.parametrize('system',SYSTEMS[:3])
def test_adapted_prompt_has_no_labels_or_future(rows,system):
    r=make_views(rows,['t:0'])[0][0]
    text=prompt(system,r)
    assert 'SENTINEL' not in text and 'images/' not in text and 't:0' not in text
    assert 'TYPE' not in text  # Future recorded recovery action is absent.
    assert 'OFFLINE ADAPTATION' in text
    assert method_instruction(system) in text

def test_methods_are_distinct():
    assert len({method_instruction(s) for s in SYSTEMS[:3]})==3

@pytest.mark.parametrize('raw',[
    '{}','[]','{"outcome_label":"SUCCESS"} text',
    '{"outcome_label":"SUCCESS","outcome_label":"FAILURE","reason":"x"}',
    '{"outcome_label":"SUCCESS","reason":"x"} {"outcome_label":"FAILURE","reason":"x"}',
    '```json\n{"outcome_label":"SUCCESS","reason":"x"}\n```',
    '{"outcome_label":true,"reason":"x"}'])
def test_invalid_output_never_repaired(raw):
    assert parse_response(raw,PHASES[1])['status']=='parse_error'

def test_abstention_distinct():
    assert parse_response('{"outcome_label":"ABSTAIN","reason":"not visible"}',PHASES[1])['status']=='abstention'

def test_valid_recovery():
    p=parse_response('{"outcome_label":"FAILURE","reason":"unchanged"}',PHASES[1])
    assert p['status']=='valid' and p['predictions']=={'outcome_label':'FAILURE'}

class Fake:
    def __init__(self,fail=False):self.calls=0;self.fail=fail
    def predict(self,r):
        self.calls+=1
        if self.fail: raise RuntimeError('fixture infrastructure failure')
        pred={'outcome_label':'FAILURE'}
        if r['phase']==PHASES[0]:pred['failure_type_4']='ACTION_MISMATCH'
        return {'status':'valid','predictions':pred,'model_calls':1}

def test_resume_unstarted_only_and_identity(rows,tmp_path):
    requests=make_views(rows,['t:0'])[0]; backend=Fake()
    execute(requests,'pc03',backend,tmp_path,{'profile':1})
    execute(requests,'pc03',backend,tmp_path,{'profile':1})
    assert backend.calls==2
    with pytest.raises(ValueError,match='identity'):execute(requests,'pc03',backend,tmp_path,{'profile':2})

def test_interrupted_attempt_preserved(rows,tmp_path):
    requests=make_views(rows,['t:0'])[0];key=request_key(requests[0])
    write_new(tmp_path/(key+'.started.json'),{'request_key':key})
    b=Fake()
    with pytest.raises(RuntimeError,match='Review required'):execute(requests,'pc03',b,tmp_path,{})
    assert b.calls==0 and read(tmp_path/(key+'.json'))['status']=='interrupted'

def test_infrastructure_error_stops_and_is_saved(rows,tmp_path):
    requests=make_views(rows,['t:0'])[0];b=Fake(True)
    with pytest.raises(RuntimeError):execute(requests,'pc03',b,tmp_path,{})
    assert b.calls==1
    assert read(tmp_path/(request_key(requests[0])+'.json'))['status']=='infrastructure_error'

def test_scorer_rejects_wrong_record_binding(rows,tmp_path):
    requests=make_views(rows,['t:0'])[0]
    execute(requests,'pc03',Fake(),tmp_path/'pc03',{'p':1})
    with pytest.raises(ValueError):collect_records(tmp_path,'pc03',requests,{'p':2})

def test_metrics_match_sklearn():
    from sklearn.metrics import matthews_corrcoef,f1_score,balanced_accuracy_score
    t=['a','b','b','a','c'];p=['a','a','b','c','c'];labels=['a','b','c']
    m=metrics(t,p,labels)
    assert m['mcc']==pytest.approx(matthews_corrcoef(t,p))
    assert m['macro_f1']==pytest.approx(f1_score(t,p,labels=labels,average='macro'))
    assert m['balanced_accuracy']==pytest.approx(balanced_accuracy_score(t,p))

def test_missing_and_invalid_denominators(rows):
    requests,targets,_=make_views(rows,['t:0'])
    r=requests[0];key=request_key(r)
    records={'pc03':{key:{'status':'parse_error','predictions':{}}}}
    result=score(requests,targets,records)
    assert result['status']=='incomplete'
    m=result['systems'][PHASES[0]]['pc03']['endpoints']['outcome_label']
    assert m['mcc'] is None and m['all_case_accuracy']==0 and m['valid_denominator']==0
    assert result['systems'][PHASES[1]]['pc03']['statuses']['missing']==1

def test_pairing_groups_and_repeatability():
    rows=[('a','S','S','F'),('a','F','S','F'),('b','S','S','S')]
    a=paired_interval(rows,['S','F'],resamples=100)
    assert a==paired_interval(rows,['S','F'],resamples=100)
    assert a['pairs']==3 and a['task_groups']==2

def test_local_image_mount_must_exist(tmp_path):
    with pytest.raises(RuntimeError,match='local dataset root'):require_local_images(tmp_path)
    (tmp_path/'images').mkdir()
    require_local_images(tmp_path)

def test_relative_image_mount_rejected():
    with pytest.raises(RuntimeError,match='absolute local'):require_local_images('.')

@pytest.mark.parametrize('logits,expected',[([3.,1.],'SUCCESS'),([1.,3.],'FAILURE')])
def test_pc03_trained_outcome_mapping_unchanged(rows,logits,expected):
    import torch
    from web_agent.eval.task1.backends import PC03Assessor
    backend=PC03Assessor.__new__(PC03Assessor);backend.torch=torch;backend.batch=lambda r:{}
    backend.model=lambda b:{'outcome':torch.tensor([logits]),'failure_type':torch.tensor([[0.,0.,5.,0.]])}
    r=make_views(rows,['t:0'])[0][0]
    result=backend.predict(r)
    assert result['predictions']=={'outcome_label':expected,'failure_type_4':'ACTION_MISMATCH'}

@pytest.mark.parametrize('logit,expected',[(0.,'FAILURE'),(-1.,'FAILURE'),(0.01,'SUCCESS')])
def test_pc03_recovery_threshold_unchanged(rows,logit,expected):
    import torch
    from web_agent.eval.task1.backends import PC03Assessor
    backend=PC03Assessor.__new__(PC03Assessor);backend.torch=torch;backend.batch=lambda r:{}
    backend.model=lambda b:{'recovery_outcome':torch.tensor([[logit]])}
    assert backend.predict(make_views(rows,['t:0'])[0][1])['predictions']['outcome_label']==expected

def test_complete_fixture_run_keeps_failed_predictions(rows,tmp_path):
    requests,targets,_=make_views(rows,['t:0']);records={}
    for system in SYSTEMS:
        execute(requests,system,Fake(),tmp_path/system,{})
        records[system]=collect_records(tmp_path,system,requests,{})
    # One wrong reference ensures the scorer cannot equate execution with correctness.
    targets[request_key(requests[0])]['outcome_label']='SUCCESS'
    result=score(requests,targets,records)
    assert result['status']=='complete'
    assert result['systems'][PHASES[0]]['pc03']['endpoints']['outcome_label']['all_case_accuracy']==0
    assert result['systems'][PHASES[1]]['pc03']['endpoints']['outcome_label']['all_case_accuracy']==1

def test_prepared_real_manifest_is_disjoint_and_unchanged():
    from web_agent.eval.task1.core import ROOT,load_prepared,file_hash,SELECTION_HASH
    folder=ROOT/'docs/evidence/task1-assessment-prepared-v1'
    pilot=load_prepared(folder);dev=load_prepared(folder,True)
    assert len(pilot)==360 and sum(r['phase']==PHASES[1] for r in pilot)==120
    assert len({r['case_id'] for r in dev})==12
    assert not {r['task_id'] for r in dev}&{r['task_id'] for r in pilot}
    assert file_hash(ROOT/'docs/evidence/task1-mini-validation-240-v1/manifest.json')==SELECTION_HASH
