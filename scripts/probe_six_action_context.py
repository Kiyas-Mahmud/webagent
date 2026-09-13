"""Frozen six-state context ablation: 6 trained forwards + 18 generations.

Saved development/engineering observations only. No browser execution, memory
query, training, prompt search or final-evaluation data. Cases never supply
expected action labels to the generator except through the actual user goal.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

from scripts.probe_hybrid_planning_v2 import sha, write
from web_agent.runtime.contracts import PolicyObservation, RuntimeTaskView, ConcreteAction
from web_agent.runtime.hybrid_action_policy import HybridActionContext, trained_advice, _HybridGenerator
from web_agent.runtime.hybrid_interface import semantic_context
from web_agent.runtime.action_parameters import DeterministicParameterProvider

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('controls', 'controls_advice', 'controls_advice_history')
FIXTURE_GOALS = {
    'SELECT': 'Choose the option one from the dropdown.',
    'SCROLL': 'Scroll down the page to inspect the content below the current viewport.',
    'NAVIGATE': 'Navigate to http://hybrid.fixture.local/finish.',
    'PRESS_KEY': 'Press the Tab key to move keyboard focus to the next control.',
}


def variant_context(full, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    result = deepcopy(full)
    if variant != 'controls_advice_history':
        # Omit history rather than falsely reporting that no prior actions exist.
        result.pop('causal_history', None)
    if variant == 'controls':
        result.pop('trained_advice', None)
    return result


def prepare(out, source, fixture):
    from web_agent.eval.table2.miniwob_study import verify_plan
    from web_agent.runtime.qwen2vl_pc01 import REGISTERED_GENERATION_KWARGS
    plan = verify_plan(source)
    assert plan['profile'] == 'miniwob-hybrid-dev-v3'
    assert json.loads((source/'completion.json').read_text())['status'] == 'COMPLETE'
    cases = []
    for kind, name in [('CLICK', 'click-button-sequence'), ('TYPE', 'enter-text-2')]:
        p = source/name/'repeat-0/H1/hybrid-normal-outputs/proposal-0002.json'
        r = json.loads(p.read_text())
        cases.append(dict(case=kind, origin='SAVED_LIVE_DEVELOPMENT', source=str(p),
                          source_sha256=sha(p), observation=r['observation'], task=r['task'],
                          context=r['audit_context'], historical_raw=r['raw_response']))
    op = fixture/'observation-4.json'
    ep = fixture/'executions.json'
    original = PolicyObservation.from_dict(json.loads(op.read_text()))
    actions = tuple(ConcreteAction.from_dict(x['action']) for x in json.loads(ep.read_text())[:3])
    for kind, goal in FIXTURE_GOALS.items():
        obs = replace(original, goal=goal)
        task = RuntimeTaskView(task_id=obs.task_id, goal=goal)
        context = HybridActionContext('fixture', obs.observation_id, actions).project(obs)
        cases.append(dict(case=kind, origin='SAVED_ENGINEERING_STATE_WITH_NEW_DIAGNOSTIC_GOAL',
                          source=str(op), source_sha256=sha(op), setup_actions_source=str(ep),
                          setup_actions_sha256=sha(ep), observation=obs.to_dict(),
                          task=task.to_dict(), context=context))
    for case in cases:
        obs = PolicyObservation.from_dict(case['observation'])
        assert sha(obs.screenshot_path) == obs.screenshot_sha256
        assert len(obs.causal_history) == len(case['context']['completed_actions']) > 0
    prompt = ROOT/'configs/eval/table2/miniwob_hybrid_action_prompt_v3.txt'
    out.mkdir(parents=True, exist_ok=False)
    write(out/'plan.json', dict(profile='six-action-context-diagnostic-v1',
          cases=cases, variants=VARIANTS, source_run=str(source), source_plan_sha256=sha(source/'plan.json'),
          script_sha256=sha(__file__), helper_sha256=sha(ROOT/'scripts/probe_hybrid_planning_v2.py'),
          config=plan['config'], prompt=prompt.read_text(), prompt_sha256=sha(prompt),
          generation_kwargs=dict(REGISTERED_GENERATION_KWARGS), model_seed=42,
          trained_forwards=6, generations=18, browser_actions=0, memory_queries=0,
          hypothesis='Adding trained advice or prior action history changes action selection or parameter validity.',
          stopping_rule='Exactly six states and three input variants; preserve every output; no adaptive extra calls.',
          limits='One state per intended operation, mixed engineering/development origins; instruction-following diagnostics, not completion or generalization.'))
    (out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    print('FROZEN: 6 trained forwards, 18 generations, zero browser actions', flush=True)


def run(out):
    from web_agent.eval.table2.miniwob_study import verify_plan
    from web_agent.eval.table2.miniwob_model import load_models
    from web_agent.runtime.qwen2vl_pc01 import REGISTERED_GENERATION_KWARGS
    p = json.loads((out/'plan.json').read_text())
    assert sha(out/'plan.json') == (out/'plan.sha256').read_text().strip()
    assert sha(__file__) == p['script_sha256']
    assert sha(ROOT/'scripts/probe_hybrid_planning_v2.py') == p['helper_sha256']
    assert dict(REGISTERED_GENERATION_KWARGS) == p['generation_kwargs']
    source = Path(p['source_run'])
    assert sha(source/'plan.json') == p['source_plan_sha256']
    verify_plan(source)
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'], text=True).splitlines()
    active = [x for x in active if x.strip() and x.partition(',')[2].strip() != '/usr/libexec/gnome-remote-desktop-daemon']
    if active:
        raise RuntimeError('Other GPU work preserved: '+repr(active))
    write(out/'started.json', {'planned_model_calls': 24, 'resume_allowed': False})
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    bundle, _, manifest = load_models(p['config'])
    base = bundle.e0.action_predictor.__self__
    rows = []
    for case in p['cases']:
        assert sha(case['source']) == case['source_sha256']
        if 'setup_actions_source' in case:
            assert sha(case['setup_actions_source']) == case['setup_actions_sha256']
        obs = PolicyObservation.from_dict(case['observation'])
        task = RuntimeTaskView.from_dict(case['task'])
        assert sha(obs.screenshot_path) == obs.screenshot_sha256
        write(out/f"{case['case']}-advice-started.json", {'observation_sha256':obs.record_sha256})
        decision = bundle.selected.action_predictor(task, obs, random.Random(42))
        advice = trained_advice(decision, checkpoint_sha256=manifest['checkpoint_sha256'])
        write(out/f"{case['case']}-advice.json", {'decision':decision.to_dict(), 'advice':advice})
        context = deepcopy(case['context']); context['trained_advice'] = advice
        full = semantic_context(context, episode_id=context['episode_id'], interface_version=3)
        for variant in VARIANTS:
            inp = variant_context(full, variant)
            suffix = '\nhybrid_action_context: '+json.dumps(inp, sort_keys=True)
            prefix = out/f"{case['case']}-{variant}"
            write(str(prefix)+'-started.json', {'prompt':p['prompt'], 'suffix':suffix})
            raw, metadata = base._generate_with_metadata(prompt=p['prompt'], task=task, observation=obs, suffix=suffix)
            row = dict(case=case['case'], origin=case['origin'], variant=variant,
                       raw_response=raw, generation_metadata=metadata)
            if variant == VARIANTS[-1] and 'historical_raw' in case:
                row['matches_historical_raw'] = raw == case['historical_raw']
            receipt = {}
            try:
                # Reuse the real parser and decision conversion, with no extra model call.
                frozen = SimpleNamespace(_generate_with_metadata=lambda **kwargs: (raw,metadata))
                backend = _HybridGenerator(base=frozen,prompt=p['prompt'],context=inp,receipt=receipt,
                                            policy_id='diagnostic',interface_version=3)
                parsed = backend.predict_action(task,obs,random.Random(42))
                row.update(proposal=receipt['resolved_proposal'],decision=parsed.to_dict(),status='RESOLVED')
                values = DeterministicParameterProvider().resolve(task,obs,parsed,rng=random.Random(42))
                row.update(parameters=dict(values.values),status='PARAMETERS_VALID')
            except Exception as exc:
                row.update(status='REJECTED',error_type=type(exc).__name__,error=str(exc),receipt=receipt)
            write(str(prefix)+'.json',row);rows.append(row)
            print(json.dumps({k:row[k] for k in ['case','variant','status','raw_response']}),flush=True)
    verify_plan(source)
    write(out/'results.json',dict(status='COMPLETE',trained_forwards=6,generations=len(rows),
          browser_actions=0,memory_queries=0,live_episodes=0,rows=rows))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--fixture',type=Path)
    args=parser.parse_args()
    if args.operation=='prepare':prepare(args.output,args.source,args.fixture)
    else:run(args.output)
