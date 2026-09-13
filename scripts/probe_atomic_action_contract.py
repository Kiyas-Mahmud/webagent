"""One compact atomic-action candidate on the six frozen diagnostic states.

Six generations only. Translation preserves the model's operation and argument;
it never reads a task goal to supply an action, target or missing value.
"""
import argparse
import json
import os
from pathlib import Path
import random
import re
import subprocess
from types import SimpleNamespace

from scripts.probe_hybrid_planning_v2 import sha, write
from web_agent.runtime.contracts import PolicyObservation, RuntimeTaskView
from web_agent.runtime.hybrid_action_policy import _HybridGenerator
from web_agent.runtime.action_parameters import DeterministicParameterProvider

ROOT=Path(__file__).resolve().parents[1]
PROMPT=('Choose the next operation that advances the current task. Return only one JSON object '
        'with three keys: "operation", "control", "argument". '
        'operation must be CLICK, TYPE, SELECT, SCROLL, NAVIGATE, or PRESS_KEY. '
        'For CLICK, TYPE, or SELECT, control is a current control ID; otherwise control is null. '
        'argument is the exact text to enter for TYPE, the option for SELECT, up or down for SCROLL, '
        'the destination URL for NAVIGATE, or the key for PRESS_KEY. For CLICK it is null. '
        'Choose only the immediate next operation; do not report later steps as already completed.\n')


def translate(raw):
    # Duplicate fields and non-JSON output fail; do not silently repair a proposal.
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate field: '+key)
            result[key]=value
        return result
    match=re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```',raw.strip(),re.DOTALL)
    obj=json.loads(match.group(1) if match else raw,object_pairs_hook=pairs)
    if not isinstance(obj,dict) or set(obj)!={'operation','control','argument'}:
        raise ValueError('one atomic operation object with exactly three fields required')
    if obj['operation'] not in ('CLICK','TYPE','SELECT','SCROLL','NAVIGATE','PRESS_KEY'):
        raise ValueError('unknown operation')
    if any(obj[k] is not None and type(obj[k]) is not str for k in ('control','argument')):
        raise ValueError('control and argument must be strings or null')
    return dict(action_type=obj['operation'],target=obj['control'],value=obj['argument'],bbox=None)


def prepare(out, source):
    p=json.loads((source/'plan.json').read_text())
    assert p['profile']=='six-action-context-diagnostic-v1'
    assert json.loads((source/'results.json').read_text())['status']=='COMPLETE'
    out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',dict(profile='atomic-action-contract-candidate-v1',source=str(source),
          source_plan_sha256=sha(source/'plan.json'),script_sha256=sha(__file__),
          helper_sha256=sha(ROOT/'scripts/probe_hybrid_planning_v2.py'),prompt=PROMPT,
          config=p['config'],generation_kwargs=p['generation_kwargs'],cases=p['cases'],
          generations=6,browser_actions=0,memory_queries=0,
          changes='Compact operation/control/argument contract and current-control-only context; no advice/history. This is a combined interface candidate, not an isolated causal attribution.',
          stopping_rule='Exactly one generation per fixed case; no adaptive prompts, repairs or extra trials.',
          claim_limit='A valid instruction-following proposal is not browser execution or task completion.'))
    (out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    print('FROZEN: six atomic-contract generations',flush=True)


def run(out):
    from web_agent.eval.table2.miniwob_study import verify_plan
    from web_agent.eval.table2.miniwob_model import load_models
    from web_agent.runtime.qwen2vl_pc01 import REGISTERED_GENERATION_KWARGS
    p=json.loads((out/'plan.json').read_text())
    assert sha(out/'plan.json')==(out/'plan.sha256').read_text().strip()
    assert sha(__file__)==p['script_sha256'] and sha(ROOT/'scripts/probe_hybrid_planning_v2.py')==p['helper_sha256']
    assert dict(REGISTERED_GENERATION_KWARGS)==p['generation_kwargs']
    source=Path(p['source']);assert sha(source/'plan.json')==p['source_plan_sha256']
    parent=json.loads((source/'plan.json').read_text());run_root=Path(parent['source_run'])
    assert sha(run_root/'plan.json')==parent['source_plan_sha256'];verify_plan(run_root)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).splitlines()
    active=[x for x in active if x.strip() and x.partition(',')[2].strip()!='/usr/libexec/gnome-remote-desktop-daemon']
    if active:raise RuntimeError('Other GPU work preserved: '+repr(active))
    write(out/'started.json',{'planned_generations':6,'resume_allowed':False})
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    bundle,_,_=load_models(p['config']);base=bundle.e0.action_predictor.__self__;rows=[]
    for case in p['cases']:
        assert sha(case['source'])==case['source_sha256']
        obs=PolicyObservation.from_dict(case['observation']);task=RuntimeTaskView.from_dict(case['task'])
        assert sha(obs.screenshot_path)==obs.screenshot_sha256
        controls=case['context']['current_controls']
        suffix='\nCurrent observed controls: '+json.dumps(controls,sort_keys=True)
        write(out/f"{case['case']}-started.json",{'prompt':p['prompt'],'suffix':suffix,'task':task.to_dict()})
        raw,metadata=base._generate_with_metadata(prompt=p['prompt'],task=task,observation=obs,suffix=suffix)
        row=dict(case=case['case'],raw_response=raw,generation_metadata=metadata)
        try:
            translated=translate(raw);row['translated']=translated
            frozen=SimpleNamespace(_generate_with_metadata=lambda **kw:(json.dumps(translated),metadata))
            receipt={};backend=_HybridGenerator(base=frozen,prompt='',context={},receipt=receipt,
                                                 policy_id='atomic-diagnostic',interface_version=3)
            decision=backend.predict_action(task,obs,random.Random(42))
            row['proposal']=receipt['resolved_proposal']
            params=DeterministicParameterProvider().resolve(task,obs,decision,rng=random.Random(42))
            row.update(status='PARAMETERS_VALID',parameters=dict(params.values))
        except Exception as exc:row.update(status='REJECTED',error_type=type(exc).__name__,error=str(exc))
        rows.append(row);write(out/f"{case['case']}.json",row);print(json.dumps(row),flush=True)
    verify_plan(run_root)
    write(out/'results.json',dict(status='COMPLETE',generations=6,browser_actions=0,memory_queries=0,rows=rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    if args.operation=='prepare':prepare(args.output,args.source)
    else:run(args.output)
