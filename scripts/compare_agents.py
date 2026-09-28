"""Baseline vs ours on the real-website suite: freeze a plan, run it resumably, report.

  freeze  --out DIR [--repeats 2]   fixed plan: tasks, order, settings, source hashes
  run     DIR                       runs missing episodes; safe to stop and rerun
  report  DIR                       paired completion analysis

Baseline = native Browser Use + frozen Qwen2.5-VL-7B base (system A).
Ours     = the same agent + P1 trained assessment/recovery + P4 experience memory (C).
Both systems alternate per task (A,C then C,A on the next repeat) so site drift
hits them equally. Memory starts empty in DIR/memory and learns during the run.
Completion comes only from the task's URL rule; it never reaches the agent.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from web_agent.eval.task2.campaign import BROWSER_PYTHON, ROOT
from web_agent.eval.task2.ipc import Worker

SYSTEMS = {'A': 'baseline', 'C': 'ours'}
SETTINGS = ROOT/'configs/eval/task2/qwen25_v1.json'
WEB_TASKS = ROOT/'configs/eval/task2/web_tasks_v1.json'
CALIBRATION = Path('/home/aiub/kiyas/table2-evidence/p4-qwen25-embeddings-v1/manifest.json')
SOURCES = ['src/web_agent/eval/task2/live.py', 'src/web_agent/eval/task2/web_worker.py',
           'src/web_agent/eval/task2/native_model.py', 'src/web_agent/eval/task2/model_worker.py',
           'src/web_agent/eval/task2/assessment.py', 'src/web_agent/eval/task2/elements.py',
           'src/web_agent/memory/experience.py', 'scripts/compare_agents.py',
           'configs/eval/task2/web_tasks_v1.json', 'configs/eval/task2/qwen25_v1.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(out, repeats, only=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    suite = json.loads(WEB_TASKS.read_text())
    if only:
        suite['tasks'] = [t for t in suite['tasks'] if t['id'] in only]
    base = json.loads(SETTINGS.read_text())
    calibration = json.loads(CALIBRATION.read_text())
    settings = dict(base['settings'], actor_label=base['actor_label'], max_agent_steps=15, memory_mode='experience',
                    experience_memory={'directory': str(out/'memory'), 'threshold': calibration['admission_threshold'], 'write': True})
    settings['budget'] = dict(settings['budget'], max_executor_requests=15)
    order = []
    for repeat in range(repeats):
        for task in suite['tasks']:
            systems = ('A', 'C') if repeat % 2 == 0 else ('C', 'A')
            order += [{'repeat': repeat, 'task': task['id'], 'goal': task['goal'], 'system': s} for s in systems]
    plan = {'schema': 'web-comparison-v1', 'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'repeats': repeats,
            'systems': SYSTEMS, 'settings': settings, 'suite': suite['suite'], 'episodes': order,
            'sources': {p: sha(ROOT/p) for p in SOURCES},
            'analysis': 'paired per (task, repeat): exact two-sided sign test on discordant pairs; completion = URL rule'}
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(f'frozen {len(order)} episodes ({len(suite["tasks"])} tasks x {repeats} repeats x 2 systems) -> {out}')


def folder_for(out, e):
    return out/'episodes'/f"{e['repeat']}-{e['task']}-{e['system']}"


async def run(out):
    from web_agent.eval.task2.live import episode
    out = Path(out).resolve(); plan = json.loads((out/'plan.json').read_text())
    changed = [p for p, digest in plan['sources'].items() if sha(ROOT/p) != digest]
    if changed:
        raise SystemExit('Frozen source changed since freeze: '+', '.join(changed))
    todo = []
    for e in plan['episodes']:
        folder = folder_for(out, e)
        if (folder/'result.json').exists():
            continue
        if folder.exists():
            # Interrupted (shutdown or infrastructure error): keep it, never count it, rerun.
            keep = out/'interrupted'/f"{folder.name}-{time.strftime('%Y%m%dT%H%M%S')}"
            keep.parent.mkdir(exist_ok=True); shutil.move(str(folder), keep)
            with (out/'interrupted.jsonl').open('a') as log:
                log.write(json.dumps({'episode': folder.name, 'moved_to': str(keep), 'at': time.time()})+'\n')
        todo.append(e)
    total = len(plan['episodes']); done = total-len(todo)
    print(f'{done}/{total} already done; running {len(todo)}', flush=True)
    if not todo:
        return
    stamp = time.strftime('%Y%m%dT%H%M%S')
    model = Worker(ROOT/'.venv/bin/python', 'web_agent.eval.task2.model_worker', ['qwen25'], out/f'model-{stamp}.log')
    browser = Worker(BROWSER_PYTHON, 'web_agent.eval.task2.web_worker', [out/'browser'/stamp], out/f'browser-{stamp}.log',
                     env={'PLAYWRIGHT_BROWSERS_PATH': '/home/aiub/kiyas/table2-inputs/miniwob-browsers'})
    try:
        print('loading models (about 7 minutes)...', flush=True)
        await asyncio.to_thread(model.call, 'initialize', timeout=1200)
        for e in todo:
            done += 1
            block = {'task': e['task'], 'repeat_id': e['repeat'], 'browser_reset_seed': 0, 'goal': e['goal']}
            print(f"\n[{done}/{total}] {SYSTEMS[e['system']]:8s} {e['task']}  (repeat {e['repeat']})", flush=True)
            try:
                r = await episode(system=e['system'], block=block, root=folder_for(out, e),
                                  browser_worker=browser, model_worker=model, settings=plan['settings'])
                print(f"   completed={r['completion']}  steps={r['agent_steps']}  recoveries={r['counters']['executed_recoveries']}  "
                      f"memory shown={r['counters']['memory_exposures']}  {r['elapsed_seconds']:.0f}s", flush=True)
            except Exception:
                print('   INFRASTRUCTURE ERROR (not a result; rerun on next `run`):\n'+traceback.format_exc(), flush=True)
    finally:
        browser.close(); model.close()
        # Browser Use leaves background tasks that block asyncio shutdown; every
        # episode is saved, so leave from inside the loop instead of hanging.
        print('run finished; episodes saved', flush=True)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


def exact_sign_p(improved, worsened):
    n = improved+worsened
    if not n:
        return 1.0
    return min(1.0, 2*sum(math.comb(n, k) for k in range(min(improved, worsened)+1))/2**n)


def report(out):
    out = Path(out).resolve(); plan = json.loads((out/'plan.json').read_text())
    results = {}
    for e in plan['episodes']:
        path = folder_for(out, e)/'result.json'
        if path.exists():
            results[(e['repeat'], e['task'], e['system'])] = json.loads(path.read_text())
    pairs = sorted({(r, t) for r, t, _ in results if (r, t, 'A') in results and (r, t, 'C') in results})
    improved = [p for p in pairs if results[p+('C',)]['completion'] and not results[p+('A',)]['completion']]
    worsened = [p for p in pairs if results[p+('A',)]['completion'] and not results[p+('C',)]['completion']]
    summary = {}
    for s, name in SYSTEMS.items():
        rs = [v for k, v in results.items() if k[2] == s]
        summary[name] = {'episodes': len(rs), 'completed': sum(r['completion'] for r in rs),
                         'recoveries': sum(r['counters']['executed_recoveries'] for r in rs),
                         'memory_shown': sum(r['counters']['memory_exposures'] for r in rs),
                         'memory_written': sum(r['counters'].get('memory_writes', 0) for r in rs),
                         'mean_seconds': round(sum(r['elapsed_seconds'] for r in rs)/max(len(rs), 1))}
    result = {'planned_episodes': len(plan['episodes']), 'finished_episodes': len(results), 'pairs': len(pairs),
              'summary': summary, 'ours_minus_baseline': (len(improved)-len(worsened))/len(pairs) if pairs else None,
              'improved_pairs': [list(p) for p in improved], 'worsened_pairs': [list(p) for p in worsened],
              'exact_sign_test_p': exact_sign_p(len(improved), len(worsened))}
    (out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f"episodes finished {len(results)}/{len(plan['episodes'])}, complete pairs {len(pairs)}")
    for name, s in summary.items():
        print(f"  {name:8s} completed {s['completed']}/{s['episodes']}  recoveries {s['recoveries']}  "
              f"memory shown {s['memory_shown']} written {s['memory_written']}  mean {s['mean_seconds']}s/episode")
    print(f"  pairs: ours better {len(improved)}, baseline better {len(worsened)}, same {len(pairs)-len(improved)-len(worsened)}  "
          f"exact sign test p = {result['exact_sign_test_p']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze'); f.add_argument('--out', required=True); f.add_argument('--repeats', type=int, default=2)
    f.add_argument('--tasks', nargs='*', help='subset of task ids (smoke tests only)')
    sub.add_parser('run').add_argument('dir')
    sub.add_parser('report').add_argument('dir')
    args = parser.parse_args()
    os.environ['ANONYMIZED_TELEMETRY'] = 'false'
    os.environ['BROWSER_USE_CONFIG_DIR'] = str(ROOT/'.task2-assets/browser-use-config')
    if args.command == 'freeze':
        freeze(args.out, args.repeats, args.tasks)
    elif args.command == 'report':
        report(args.dir)
    else:
        try:
            asyncio.run(run(args.dir))
        except BaseException:
            traceback.print_exc()
        # Browser Use leaves background tasks that block asyncio shutdown; results are saved.
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
