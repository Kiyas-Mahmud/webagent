"""Replay the existing one-fallback provider on four rejected atomic proposals.

This does not replace the selected operation, invent arguments or execute a
browser command. Each fallback generation is separately preserved and counted.
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

from scripts.probe_hybrid_planning_v2 import sha, write
from web_agent.runtime.contracts import PolicyObservation, RuntimeTaskView
from web_agent.runtime.hybrid_action_policy import _HybridGenerator
from web_agent.runtime.model_calls import ModelCallLedger, activate_model_call_ledger, deactivate_model_call_ledger

ROOT=Path(__file__).resolve().parents[1]


def prepare(out,source):
    p=json.loads((source/'plan.json').read_text());r=json.loads((source/'results.json').read_text())
    assert p['profile']=='atomic-action-contract-candidate-v1' and r['status']=='COMPLETE'
    rows=[x for x in r['rows'] if x['status']=='REJECTED']
    assert [x['case'] for x in rows]==['SELECT','SCROLL','NAVIGATE','PRESS_KEY']
    out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',dict(profile='atomic-parameter-diagnostic-v1',source=str(source),
          source_plan_sha256=sha(source/'plan.json'),source_results_sha256=sha(source/'results.json'),
          script_sha256=sha(__file__),helper_sha256=sha(ROOT/'scripts/probe_hybrid_planning_v2.py'),
          config=p['config'],generation_kwargs=p['generation_kwargs'],cases=p['cases'],rejected=rows,
          planned_generations=4,browser_actions=0,memory_queries=0,
          stopping_rule='Existing deterministic-then-one-fallback provider once for each rejected proposal; no retries or prompt changes.'))
    (out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    print('FROZEN: four existing parameter-provider fallbacks',flush=True)


def run(out):
    from web_agent.eval.table2.miniwob_study import verify_plan
    from web_agent.eval.table2.miniwob_model import load_models
    from web_agent.runtime.qwen2vl_pc01 import REGISTERED_GENERATION_KWARGS
    p=json.loads((out/'plan.json').read_text())
    assert sha(out/'plan.json')==(out/'plan.sha256').read_text().strip()
    assert sha(__file__)==p['script_sha256'] and sha(ROOT/'scripts/probe_hybrid_planning_v2.py')==p['helper_sha256']
    assert dict(REGISTERED_GENERATION_KWARGS)==p['generation_kwargs']
    source=Path(p['source']);assert sha(source/'plan.json')==p['source_plan_sha256'] and sha(source/'results.json')==p['source_results_sha256']
    atomic=json.loads((source/'plan.json').read_text())
    ctx=json.loads((Path(atomic['source'])/'plan.json').read_text()); original=Path(ctx['source_run'])
    assert sha(original/'plan.json')==ctx['source_plan_sha256'];verify_plan(original)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).splitlines()
    active=[x for x in active if x.strip() and x.partition(',')[2].strip()!='/usr/libexec/gnome-remote-desktop-daemon']
    if active:raise RuntimeError('Other GPU work preserved: '+repr(active))
    write(out/'started.json',dict(planned_generations=4,resume_allowed=False))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    bundle,provider,_=load_models(p['config']);base=bundle.e0.action_predictor.__self__
    measured=[]
    def generate(**kwargs):
        if len(measured)>=4:raise RuntimeError('Diagnostic generation budget exhausted')
        i=len(measured)+1
        write(out/f'call-{i}-started.json',dict(prompt=kwargs['prompt'],suffix=kwargs['suffix'],
                                              task=kwargs['task'].to_dict(),observation=kwargs['observation'].to_dict()))
        raw,meta=base._generate_with_metadata(**kwargs)
        measured.append(dict(raw_response=raw,generation_metadata=meta))
        write(out/f'call-{i}.json',measured[-1]);return raw
    base._generate=generate
    events=[];ledger=ModelCallLedger('atomic-parameter-diagnostic',sink=events.append)
    token=activate_model_call_ledger(ledger);results=[]
    try:
        for row in p['rejected']:
            case=next(c for c in p['cases'] if c['case']==row['case'])
            assert sha(case['source'])==case['source_sha256']
            obs=PolicyObservation.from_dict(case['observation']);task=RuntimeTaskView.from_dict(case['task'])
            assert sha(obs.screenshot_path)==obs.screenshot_sha256
            frozen=SimpleNamespace(_generate_with_metadata=lambda **kw:(json.dumps(row['translated']),{}))
            backend=_HybridGenerator(base=frozen,prompt='',context={},receipt={},policy_id='atomic-diagnostic',interface_version=3)
            decision=backend.predict_action(task,obs,random.Random(42))
            result=dict(case=row['case'],decision=decision.to_dict(),original_rejection=row['error'])
            before=len(measured)
            try:
                params=provider.resolve(task,obs,decision,rng=random.Random(42))
                result.update(status='RESOLVED',parameters=params.to_dict())
                assert params.action_type==decision.action_type
            except Exception as exc:
                result.update(status='REJECTED',error_type=type(exc).__name__,error=str(exc))
            assert len(measured)-before==1
            result['fallback_generation']=measured[-1];results.append(result)
            write(out/f"{row['case']}.json",result);print(json.dumps(result),flush=True)
    finally:deactivate_model_call_ledger(token)
    assert ledger.count==len(measured)==4
    verify_plan(original)
    write(out/'results.json',dict(status='COMPLETE',generations=4,ledger_count=ledger.count,
                                  browser_actions=0,memory_queries=0,rows=results,model_call_events=events))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    if args.operation=='prepare':prepare(args.output,args.source)
    else:run(args.output)
