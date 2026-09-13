"""Nine declared generations on saved development observations; no browser actions.

Compare the existing single-action input with a model-generated plan followed by
the same action contract. Never turn plan text into executor commands or count
this diagnostic as task completion. Old profiles and source evidence stay frozen.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from web_agent.runtime.contracts import PolicyObservation, RuntimeTaskView
from web_agent.runtime.hybrid_interface import parse_action
from web_agent.runtime.policy import ActionParseError

ROOT=Path(__file__).resolve().parents[1]
CASES=('enter-text','enter-text-2','click-button-sequence')
PLAN_PROMPT=('Describe the remaining browser operations needed to complete the current task, '
    'in their required order, using the screenshot and current controls. State what is already '
    'observed and what still needs to change. Use only values provided by the task or current '
    'observation. This is a tentative plan, not an executed action or a claim of success. '
    'Keep it concise.\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def action_suffix(original, generated_plan):
    # Exact model output is advisory data. Never extract/execute steps from it.
    context=json.loads(original.removeprefix('\nhybrid_action_context: '))
    context['tentative_model_plan']={'text':generated_plan,'already_executed':False,
        'instruction':'Check this fallible plan against the current observation and choose only one immediate action.'}
    return '\nhybrid_action_context: '+json.dumps(context,sort_keys=True)


def prepare(out, source):
    from web_agent.eval.table2.miniwob_study import verify_plan
    from web_agent.runtime.qwen2vl_pc01 import REGISTERED_GENERATION_KWARGS
    plan=verify_plan(source)
    assert plan['profile']=='miniwob-hybrid-dev-v3' and plan['phase']=='development'
    assert json.loads((source/'completion.json').read_text())['status']=='COMPLETE'
    cases=[]
    for name in CASES:
        path=source/name/'repeat-0/H2/hybrid-normal-outputs/proposal-0001.json'
        saved=json.loads(path.read_text());obs=PolicyObservation.from_dict(saved['observation'])
        assert obs.record_sha256==saved['observation_sha256']
        assert sha(obs.screenshot_path)==obs.screenshot_sha256
        cases.append({'task':name,'source':str(path),'source_sha256':sha(path),
            'observation':obs.to_dict(),'runtime_task':saved['task'],
            'prompt':saved['prompt'],'suffix':saved['input_suffix'],
            'historical_raw_response':saved['raw_response']})
    out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',{'profile':'hybrid-two-stage-planning-diagnostic-v2',
        'source_run':str(source),'source_plan_sha256':sha(source/'plan.json'),
        'script_sha256':sha(__file__),'config':plan['config'],'cases':cases,
        'planning_prompt':PLAN_PROMPT,'generation_kwargs':dict(REGISTERED_GENERATION_KWARGS),
        'planned_generations':9,'sequence_per_case':['baseline_action','tentative_plan','action_with_plan'],
        'browser_actions':0,'memory_queries':0,'live_episodes':0,
        'actor_identity':'frozen unadapted base; trained checkpoint advice is preserved in saved context',
        'hypothesis':'An explicit tentative plan can change the next-action decision without inserted task solutions.',
        'limits':'One declared candidate, three saved development observations; no adaptive prompts, extra trials or execution.'})
    (out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    print(json.dumps({'status':'FROZEN','generations':9,'output':str(out)}),flush=True)


def run(out):
    from web_agent.eval.table2.miniwob_study import verify_plan
    from web_agent.eval.table2.miniwob_model import load_models
    from web_agent.runtime.qwen2vl_pc01 import REGISTERED_GENERATION_KWARGS
    p=json.loads((out/'plan.json').read_text())
    assert sha(out/'plan.json')==(out/'plan.sha256').read_text().strip()
    assert sha(__file__)==p['script_sha256'] and dict(REGISTERED_GENERATION_KWARGS)==p['generation_kwargs']
    source=Path(p['source_run']);assert sha(source/'plan.json')==p['source_plan_sha256'];verify_plan(source)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).splitlines()
    active=[x for x in active if x.strip() and x.partition(',')[2].strip()!='/usr/libexec/gnome-remote-desktop-daemon']
    if active:raise RuntimeError('Other GPU work preserved: '+repr(active))
    write(out/'started.json',{'planned_generations':9,'resume_allowed':False})
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    bundle,_,_=load_models(p['config']);base=bundle.e0.action_predictor.__self__
    rows=[]
    for case in p['cases']:
        assert sha(case['source'])==case['source_sha256']
        obs=PolicyObservation.from_dict(case['observation']);task=RuntimeTaskView.from_dict(case['runtime_task'])
        assert sha(obs.screenshot_path)==obs.screenshot_sha256
        generated_plan=None
        for stage in p['sequence_per_case']:
            prompt=p['planning_prompt'] if stage=='tentative_plan' else case['prompt']
            suffix=action_suffix(case['suffix'],generated_plan) if stage=='action_with_plan' else case['suffix']
            number=len(rows)+1
            write(out/f'call-{number:02d}-started.json',{'task':case['task'],'stage':stage,'prompt':prompt,'suffix':suffix})
            raw,metadata=base._generate_with_metadata(prompt=prompt,task=task,observation=obs,suffix=suffix)
            row={'task':case['task'],'stage':stage,'raw_response':raw,'generation_metadata':metadata}
            if stage=='tentative_plan':
                generated_plan=raw;row['status']='UNEXECUTED_PLAN_TEXT'
            else:
                try:
                    proposal,resolution=parse_action(raw,obs,interface_version=3)
                    row.update(status='RESOLVED',proposal=proposal,resolution=resolution)
                except ActionParseError as exc:
                    row.update(status='REJECTED',diagnostic=str(exc))
                if stage=='baseline_action':row['matches_historical_raw']=raw==case['historical_raw_response']
            write(out/f'call-{number:02d}.json',row);rows.append(row)
            print(json.dumps(row),flush=True)
    assert len(rows)==p['planned_generations']==9
    verify_plan(source)
    write(out/'results.json',{'status':'COMPLETE','actual_generations':9,'browser_actions':0,
        'live_episodes':0,'memory_queries':0,'rows':rows,
        'claim':'Saved-state planning diagnostic only; valid proposals and plans do not establish execution or completion.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    if args.operation=='prepare':prepare(args.output,args.source)
    else:run(args.output)
