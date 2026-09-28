"""Live terminal progress bar for a run_agent.py run (read-only; never touches the run).

  python3 scripts/progress.py                 # latest run under .task2-assets/demo
  python3 scripts/progress.py RUN_DIR --once  # print once and exit
"""
import json
from pathlib import Path
import re
import sys
import time

DEMO = Path(__file__).resolve().parents[1]/'.task2-assets/demo'


def fmt(seconds):
    seconds = int(max(seconds, 0))
    return f'{seconds//3600}h{seconds%3600//60:02d}m' if seconds >= 3600 else f'{seconds//60}m{seconds%60:02d}s'


def snapshot(run):
    if (run/'plan.json').exists():  # compare_agents.py run
        total = len(json.loads((run/'plan.json').read_text())['episodes'])
        episodes = sorted((run/'episodes').glob('*'), key=lambda p: p.stat().st_ctime)
    else:
        match = re.search(r'-(\d+)tasks-', run.name)
        episodes = sorted(run.glob('episode-*'))
        total = int(match.group(1)) if match else max(len(episodes), 1)
    finished = lambda e: (e/'result.json').exists() or (e/'infrastructure-error.json').exists()
    done = [json.loads((e/'result.json').read_text()) for e in episodes if (e/'result.json').exists()]
    errors = sum((e/'infrastructure-error.json').exists() and not (e/'result.json').exists() for e in episodes)
    current = next((e for e in episodes if not finished(e)), None)
    started = (run/'model.log').stat().st_mtime if (run/'model.log').exists() else run.stat().st_mtime
    elapsed = time.time()-started
    per_episode = sum(r['elapsed_seconds'] for r in done)/len(done) if done else None
    lines = []
    count = len(done)+errors
    width = 40; filled = int(width*count/total)
    lines.append(f"[{'█'*filled}{'░'*(width-filled)}] {count}/{total} episodes  ({100*count//total}%)"
                 + (f"   {errors} infrastructure error(s)" if errors else ''))
    lines.append(f"completed OK: {sum(r['completion'] for r in done)}/{len(done)}   "
                 f"recoveries: {sum(r['counters']['executed_recoveries'] for r in done)}   "
                 f"memory written: {sum(r['counters'].get('memory_writes', 0) for r in done)}")
    if current is not None:
        steps = len(list(current.glob('action-*.json')))
        name = re.sub(r'^episode-\d+-|-s\d+$', '', current.name).replace('-A', '  [baseline]').replace('-C', '  [ours]')
        lines.append(f"now: {name}  step {steps}/15")
    elif not episodes:
        lines.append('now: loading models (about 7 minutes)...')
    remaining = total-count
    eta = f"~{fmt(per_episode*remaining)} left" if per_episode and remaining else ('finished' if not remaining else 'estimating...')
    lines.append(f"elapsed {fmt(elapsed)}   {eta}")
    return lines, remaining == 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    while True:
        # Without a path, always follow the newest run (a new run appears as a new folder).
        runs = [p for p in (*DEMO.glob('*'), *(DEMO.parent/'comparison').glob('*')) if p.is_dir()]
        run = Path(args[0]) if args else max(runs, key=lambda p: p.stat().st_ctime)
        lines, finished = snapshot(run)
        if '--once' in sys.argv:
            print(run.name); print('\n'.join(lines)); return
        sys.stdout.write('\033[2J\033[H'+run.name+'\n'+'\n'.join(lines)+'\n\n(Ctrl+C to exit the view; the run keeps going)\n')
        sys.stdout.flush()
        if finished and args:return
        time.sleep(5)


if __name__ == '__main__':
    try:main()
    except KeyboardInterrupt:print()
