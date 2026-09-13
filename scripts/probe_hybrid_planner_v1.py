"""Eight frozen saved-state generations; no browser actions or task solutions.

Each intervention changes one input factor and is compared to a fresh baseline.
No prompt selection, additional calls or live evaluation is performed here.
"""
import argparse
import hashlib
import json
from pathlib import Path

from web_agent.runtime.contracts import PolicyObservation, RuntimeTaskView
from web_agent.runtime.hybrid_interface import parse_action
from web_agent.runtime.policy import ActionParseError

ROOT=Path(__file__).resolve().parents[1]
REMINDER=ROOT/'configs/eval/table2/hybrid_planner_contract_probe_v1.txt'
CASES=(('enter-text',1),('click-button-sequence',2),('click-tab-2',3))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def suffix_for(saved, variant, reminder):
    original=saved['input_suffix']
    if variant=='baseline':return original
    if variant=='contract_last':return original+'\nnext_response_contract: '+reminder
    if variant!='history_text':raise ValueError('unknown probe variant')
    context=json.loads(original.removeprefix('\nhybrid_action_context: '))
    history=context['completed_actions']
    if not history:raise ValueError('history intervention requires completed actions')
    for action in history:
        if set(action)-{'action_type','bbox','parameters','execution_status','state_changed','environment_error'}:
            raise ValueError('unrecognized history fields; do not silently drop them')
    # Exact issued strings, targets and numbers survive json scalar encoding.
    # The past-execution field names are not presented as an action object.
    context['completed_actions']=[
        'Past execution '+str(i)+': '+ '; '.join(
            [f'kind={a["action_type"]}',f'box={json.dumps(a.get("bbox"))}']+
            [f'issued {k}={json.dumps(v,ensure_ascii=True)}' for k,v in sorted(a.get('parameters',{}).items())]+
            [f'observed {k}={json.dumps(a[k])}' for k in ('execution_status','state_changed','environment_error') if k in a])
        for i,a in enumerate(history,1)]
    return '\nhybrid_action_context: '+json.dumps(context,sort_keys=True)


def prepare(out, source):
    from web_agent.eval.table2.miniwob_study import verify_plan
    from web_agent.runtime.qwen2vl_pc01 import REGISTERED_GENERATION_KWARGS
    plan=verify_plan(source)
    assert plan['profile']=='miniwob-hybrid-dev-v2'
    assert json.loads((source/'completion.json').read_text())['status']=='COMPLETE'
    reminder=REMINDER.read_text();calls=[]
    for task,n in CASES:
        path=source/task/'repeat-0/H2/hybrid-normal-outputs'/f'proposal-{n:04d}.json'
        saved=json.loads(path.read_text());obs=PolicyObservation.from_dict(saved['observation'])
        assert obs.record_sha256==saved['observation_sha256']
        assert sha(obs.screenshot_path)==obs.screenshot_sha256
        variants=('baseline','contract_last') if task=='enter-text' else ('baseline','contract_last','history_text')
        for variant in variants:
            calls.append({'task':task,'variant':variant,'source':str(path),'source_sha256':sha(path),
                          'observation':obs.to_dict(),'runtime_task':saved['task'],
                          'prompt':saved['prompt'],'suffix':suffix_for(saved,variant,reminder),
                          'original_raw_response':saved['raw_response'],'original_status':saved['status']})
    assert len(calls)==8
    write(out/'plan.json',{'profile':'hybrid-planner-probe-v1','evidence_type':'SAVED_STATE_GENERATION_DIAGNOSTIC',
        'source_run':str(source),'source_plan_sha256':sha(source/'plan.json'),'config':plan['config'],
        'script_sha256':sha(__file__),'reminder_sha256':sha(REMINDER),'calls':calls,
        'generation_kwargs':dict(REGISTERED_GENERATION_KWARGS),'model_seed':42,
        'live_episodes':0,'browser_actions':0,'memory_queries':0,
        'hypotheses':['final response contract reduces schema/stale-target/preparation mistakes',
                      'history represented as past-event text reduces copying its object schema'],
        'limits':'One call per frozen case/variant; no combinations, prompt selection, execution or extra trials.'})
    (out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')


def run(out):
    from web_agent.eval.table2.miniwob_study import verify_plan
    from web_agent.eval.table2.miniwob_model import load_models
    from web_agent.runtime.qwen2vl_pc01 import REGISTERED_GENERATION_KWARGS
    plan=json.loads((out/'plan.json').read_text());assert sha(out/'plan.json')==(out/'plan.sha256').read_text().strip()
    assert sha(__file__)==plan['script_sha256'] and sha(REMINDER)==plan['reminder_sha256']
    assert dict(REGISTERED_GENERATION_KWARGS)==plan['generation_kwargs']
    assert sha(Path(plan['source_run'])/'plan.json')==plan['source_plan_sha256']
    verify_plan(Path(plan['source_run']))
    # This diagnostic never resumes started inference; partial attempts remain.
    write(out/'started.json',{'planned_generations':len(plan['calls'])})
    bundle,_,_=load_models(plan['config']);runtime=bundle.e0.action_predictor.__self__
    original_generate=runtime.model.generate;measurements=[]
    def measured_generate(*args,**kwargs):
        output=original_generate(*args,**kwargs)
        prompt_tokens=int(kwargs['input_ids'].shape[-1]);generated=int(output.shape[-1])-prompt_tokens
        eos=runtime.model.generation_config.eos_token_id
        eos=[] if eos is None else ([eos] if isinstance(eos,int) else list(eos))
        measurements.append({'generated_tokens':generated,'last_token_is_eos':int(output[0,-1]) in eos,
                             'reached_cap':generated==kwargs['max_new_tokens']})
        return output
    runtime.model.generate=measured_generate
    rows=[]
    try:
        for i,case in enumerate(plan['calls'],1):
            assert sha(case['source'])==case['source_sha256']
            obs=PolicyObservation.from_dict(case['observation']);task=RuntimeTaskView.from_dict(case['runtime_task'])
            assert sha(obs.screenshot_path)==obs.screenshot_sha256
            write(out/f'call-{i:02d}-started.json',{'task':case['task'],'variant':case['variant']})
            raw=runtime._generate(prompt=case['prompt'],task=task,observation=obs,suffix=case['suffix'])
            row={'case':i,'task':case['task'],'variant':case['variant'],'raw_response':raw,
                 'matches_historical_raw':raw==case['original_raw_response'],**measurements[-1]}
            try:
                parsed,resolution=parse_action(raw,obs)
                row.update(status='RESOLVED',proposal=parsed,resolution=resolution)
                history=json.loads(json.loads(Path(case['source']).read_text())['input_suffix'].split(': ',1)[1])['completed_actions']
                row['repeats_last_action_type_and_box']=bool(history and parsed['action_type']==history[-1]['action_type'] and parsed['bbox']==history[-1]['bbox'])
            except ActionParseError as exc:
                row.update(status='REJECTED',error=str(exc),failure_stage=getattr(exc,'failure_stage','parsing'))
            write(out/f'call-{i:02d}.json',row);rows.append(row)
            print(json.dumps(row),flush=True)
    finally:
        runtime.model.generate=original_generate
    assert len(measurements)==len(rows)==8
    write(out/'results.json',{'status':'COMPLETE','actual_generations':8,'browser_actions':0,
         'live_episodes':0,'memory_queries':0,'results':rows,
         'claim':'Counterfactual proposals on saved observations; not task completions or an E/H system comparison.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    if args.operation=='prepare':prepare(args.output,args.source)
    else:run(args.output)
