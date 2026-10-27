"""Live progress bar for scripts/evaluate_locked_test.py (read-only; never touches the run).

    python3 scripts/locked_test_progress.py            # live, refreshes every 5 s, Ctrl+C to quit
    python3 scripts/locked_test_progress.py --once     # print once and exit

Progress is read from the finished chunk files in results/locked_test/<model>_parts/
and the timestamps in the log, so it works for a run started from the notebook or the shell.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
CHUNK_ROWS = 400
ROWS = {'val': 7_861, 'test': 7_855}
CHUNKS = {k: math.ceil(v / CHUNK_ROWS) for k, v in ROWS.items()}   # 20 + 20


def running() -> bool:
    for pid in os.listdir('/proc'):
        if pid.isdigit():
            try:
                cmd = (Path('/proc') / pid / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            except OSError:
                continue
            if 'python' in cmd and 'evaluate_locked_test.py' in cmd and 'locked_test_progress' not in cmd:
                return True
    return False


def bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total)
    return '[' + '█' * filled + '░' * (width - filled) + f'] {done:2d}/{total}'


def snapshot(model: str) -> tuple[str, bool]:
    out_dir = ROOT / 'results' / 'locked_test'
    parts, log, out = out_dir / f'{model}_parts', out_dir / f'{model}.log', out_dir / f'{model}.json'
    done = {k: len(list(parts.glob(f'{k}_[0-9][0-9][0-9].npz'))) for k in CHUNKS}
    text = log.read_text(errors='replace').replace('\r', '\n') if log.is_file() else ''
    stamps = [dt.datetime.combine(dt.date.today(), dt.time.fromisoformat(m))
              for m in re.findall(r'^(\d\d:\d\d:\d\d) (?:val|test) chunk \d+/\d+ scored$', text, re.M)]
    recent = [b - a for a, b in zip(stamps, stamps[1:]) if dt.timedelta(0) < b - a < dt.timedelta(minutes=30)][-6:]
    per_chunk = sum(recent, dt.timedelta()) / len(recent) if recent else None
    total_done, total = done['val'] + done['test'], sum(CHUNKS.values())
    finished, alive = out.is_file(), running()
    reproduced = re.search(r'validation reproduced: outcome MCC ([0-9.]+)', text)
    failed = re.search(r'does not reproduce|Traceback|Error:', text)

    lines = [f'Locked-test evaluation  {model}   {dt.datetime.now():%H:%M:%S}', '',
             f'  Validation {bar(done["val"], CHUNKS["val"])}   ({ROWS["val"]:,} rows)',
             f'  Test       {bar(done["test"], CHUNKS["test"])}   ({ROWS["test"]:,} rows)',
             f'  Overall    {100 * total_done / total:5.1f} %', '']
    if reproduced:
        lines.append(f'  Validation check PASSED: outcome MCC {reproduced.group(1)} (expected 0.678); test is open')
    elif done['val'] == CHUNKS['val'] and alive:
        lines.append('  Validation done: computing the check, then the test opens')
    else:
        lines.append('  Validation check: after the last validation chunk (test stays locked until it passes)')
    if stamps:
        lines.append(f'  Last chunk finished: {stamps[-1]:%H:%M:%S}')
    if per_chunk and not finished:
        eta = dt.datetime.now() + per_chunk * (total - total_done)
        lines.append(f'  Speed: {per_chunk.total_seconds() / 60:.1f} min per chunk  |  estimated finish: {eta:%H:%M}')
    if finished:
        lines.append(f'  FINISHED: result written to {out.relative_to(ROOT)}')
    elif failed and not alive:
        lines.append('  STOPPED WITH AN ERROR: see the end of the log')
    elif not alive:
        lines.append('  NOT RUNNING: run notebook cells 1-4 again to resume (finished chunks are kept)')
    else:
        lines.append('  Status: RUNNING')
    return '\n'.join(lines), finished or not alive


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='qwen25vl_7b_gold_v2_8_dgx')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--every', type=float, default=5.0, help='refresh interval in seconds')
    args = ap.parse_args()
    if args.once:
        print(snapshot(args.model)[0])
        return
    try:
        while True:
            text, stop = snapshot(args.model)
            print('\033[2J\033[H' + text, flush=True)    # clear screen, redraw
            if stop:
                break
            time.sleep(args.every)
    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    main()
