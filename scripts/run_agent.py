"""Run one MiniWoB task with Browser Use alone (baseline) or with our pillars (ours).

Development/demo runner: same episode loop, models and budgets as the Task 2
campaigns, but no freeze or audit. Results are never evaluation evidence.

  PYTHONPATH=src .task2-assets/browser-use-env/bin/python scripts/run_agent.py \
      --task click-checkboxes-soft --mode ours --seed 7
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
import traceback

from web_agent.eval.task2.campaign import BROWSER_PYTHON, ROOT
from web_agent.eval.task2.ipc import Worker

MODES = {'baseline': 'A', 'ours': 'C'}
SETTINGS = json.loads((ROOT/'configs/eval/task2/qwen25_v1.json').read_text())
CALIBRATION = Path('/home/aiub/kiyas/table2-evidence/p4-qwen25-embeddings-v1/manifest.json')
WEB_TASKS = ROOT/'configs/eval/task2/web_tasks_v1.json'


def trace(folder):
    """Print one line per step: action, what the heads said, recovery and memory."""
    for event_path in sorted(folder.glob('action-*.json')):
        step = event_path.stem.split('-')[1]
        event = json.loads(event_path.read_text())
        line = f"step {int(step):2d}  {event.get('action_type') or '-':9s} target={event.get('target')} value={event.get('value')}"
        if event.get('recovery_proposal'):
            line += '  [recovery attempt]'
        if event.get('native_results') and event['native_results'][0].get('error'):
            line += '  REJECTED: '+event['native_results'][0]['error'][:60]
        print(line)
        assessment = folder/f'assessment-{step}.json'
        if assessment.exists():
            a = json.loads(assessment.read_text())['assessment']
            probability = a.get('outcome_probabilities', {}).get('FAILURE')
            print(f"          P1 heads: {a['signals']}" + (f'  P(failure)={probability:.2f}' if probability is not None else ''))
        memory = folder/f'memory-{step}.json'
        if memory.exists():
            m = json.loads(memory.read_text())
            reasons = [c['exclusion_reason'] or 'ADMITTED' for c in m['candidates']]
            store = f" store={m['store_size']} P(store)={m['store_probability']:.2f}" if 'store_size' in m else ''
            print(f"          P4 memory: {len(m['examples'])} admitted of {len(m['candidates'])} ({', '.join(reasons) or 'empty store'}){store}")
    for written in sorted(folder.glob('experience-incident-*.json')):
        w = json.loads(written.read_text())
        print(f"P4 write {w['incident_id']}: resolved={w['resolved']} P(store)={w['store_probability']:.2f} stored={w['stored']}"
              + (f" -> {w['memory_id']} (store size {w['store_size']})" if w['stored'] else ''))


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True, nargs='+',
                        help='MiniWoB family (e.g. click-checkboxes-soft) or, with --suite web, task ids from configs/eval/task2/web_tasks_v1.json (or "all")')
    parser.add_argument('--suite', choices=['miniwob', 'web'], default='miniwob',
                        help='web = live real-website tasks on domains from the training dataset (1280x720)')
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--seed', type=int, nargs='+', default=[0],
                        help='MiniWoB reset seed(s); several run in order with one model load and a shared memory store')
    parser.add_argument('--max-steps', type=int, default=15)
    parser.add_argument('--memory', choices=['experience', 'label'], default='experience',
                        help='P4 for --mode ours: the agent\'s own experience memory (default) or the historical label store')
    parser.add_argument('--memory-dir', default=str(ROOT/'.task2-assets/experience-memory/development'))
    parser.add_argument('--memory-readonly', action='store_true', help='retrieve but never write')
    args = parser.parse_args()
    os.environ['ANONYMIZED_TELEMETRY'] = 'false'
    os.environ['BROWSER_USE_CONFIG_DIR'] = str(ROOT/'.task2-assets/browser-use-config')
    from web_agent.eval.task2.live import episode
    web = {t['id']: t for t in json.loads(WEB_TASKS.read_text())['tasks']}
    tasks = list(web) if args.suite == 'web' and args.task == ['all'] else args.task
    if args.suite == 'web' and (unknown := [t for t in tasks if t not in web]):
        parser.error('unknown web task(s): '+', '.join(unknown))
    label = tasks[0] if len(tasks) == 1 else f'{len(tasks)}tasks'
    out = ROOT/'.task2-assets/demo'/f"{time.strftime('%Y%m%dT%H%M%S')}-{args.suite}-{label}-{args.mode}-s{'-'.join(map(str, args.seed))}"
    out.mkdir(parents=True)
    settings = dict(SETTINGS['settings'], actor_label=SETTINGS['actor_label'], max_agent_steps=args.max_steps)
    settings['budget'] = dict(settings['budget'], max_executor_requests=args.max_steps)
    if args.memory == 'experience':
        calibration = json.loads(CALIBRATION.read_text())
        settings.update(memory_mode='experience', experience_memory={
            'directory': args.memory_dir, 'threshold': calibration['admission_threshold'], 'write': not args.memory_readonly})
    model = Worker(ROOT/'.venv/bin/python', 'web_agent.eval.task2.model_worker', [SETTINGS['backend']], out/'model.log')
    worker = 'web_agent.eval.task2.web_worker' if args.suite == 'web' else 'web_agent.eval.task2.browser_worker'
    browser = Worker(BROWSER_PYTHON, worker, [out/'browser'], out/'browser.log',
                     env={'PLAYWRIGHT_BROWSERS_PATH': '/home/aiub/kiyas/table2-inputs/miniwob-browsers'})
    results = []
    try:
        print('loading models (about 7 minutes)...', flush=True)
        await asyncio.to_thread(model.call, 'initialize', timeout=1200)
        runs = [(task, seed) for task in tasks for seed in args.seed]
        for index, (task, seed) in enumerate(runs):
            if args.suite == 'web':
                goal = web[task]['goal']
            else:
                goal = (await asyncio.to_thread(browser.call, 'reset', episode_id=f'goal-probe-{index}', task=task, seed=seed))['goal']
            block = {'task': task, 'repeat_id': index, 'browser_reset_seed': seed, 'goal': goal}
            print(f"\n=== episode {index+1}/{len(runs)}  {task}  seed {seed}  mode {args.mode}\ntask: {goal}", flush=True)
            folder = out/f'episode-{index:02d}-{task}-s{seed}'
            try:
                result = await episode(system=MODES[args.mode], block=block, root=folder,
                                       browser_worker=browser, model_worker=model, settings=settings)
            except Exception:
                # Report and continue; the episode folder keeps infrastructure-error.json.
                print('EPISODE ERROR (infrastructure, not a task result):\n'+traceback.format_exc(), flush=True)
                results.append({'completion': False, 'infrastructure_error': True})
                continue
            trace(folder)
            print(f"COMPLETED: {result['completion']}  steps={result['agent_steps']}  stop={result['stop_reason']}  "
                  f"recoveries={result['counters']['executed_recoveries']}  memory shown={result['counters']['memory_exposures']}  "
                  f"memory written={result['counters'].get('memory_writes', 0)}  {result['elapsed_seconds']:.0f}s", flush=True)
            results.append(result)
    finally:
        browser.close(); model.close()
    errors = sum(r.get('infrastructure_error', False) for r in results)
    print(f"\n{sum(r['completion'] for r in results)}/{len(results)} completed"+(f"  ({errors} infrastructure errors)" if errors else '')+f"  files: {out}")
    # Browser Use leaves background tasks alive after its session stops, which
    # blocks asyncio shutdown; workers are closed and results saved, so exit now.
    sys.stdout.flush(); os._exit(0)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except BaseException:
        # Browser Use's leftover tasks can block asyncio shutdown; never hang silently.
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
