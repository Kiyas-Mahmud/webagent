# Baseline vs ours on real websites — runbook

Plan: `.task2-assets/comparison/web-v1/` (frozen with `compare_agents.py freeze`).
20 tasks (`configs/eval/task2/web_tasks_v1.json`) × 2 repeats × 2 systems = 80 episodes.

- **Baseline** = native Browser Use + frozen Qwen2.5-VL-7B base.
- **Ours** = the same agent + P1 (trained failure/recovery assessment) + P4
  (experience memory, empty at start, learning during the run).
- Order alternates per task (baseline→ours, then ours→baseline on repeat 2).
- Completion = the task's URL rule, checked by the environment, never shown to the agent.
- 15 steps per episode, identical budgets for both systems.

## Start (morning)

```bash
cd /home/aiub/kiyas/webagent
PYTHONPATH=src:scripts systemd-inhibit --what=shutdown:sleep:idle --who=web-comparison \
  --why="agent comparison running" --mode=block \
  .task2-assets/browser-use-env/bin/python -u scripts/compare_agents.py run .task2-assets/comparison/web-v1 \
  2>&1 | grep --line-buffered -vE "^(INFO|DEBUG|WARNING)|^\s*$|Loading checkpoint|^=+$" \
  | tee -a .task2-assets/comparison/web-v1.log
```

Watch progress in a second terminal: `python3 scripts/progress.py`

## Stopping and resuming

Safe to stop (Ctrl+C, or at the end of lab time). Rerun the same `run` command:
finished episodes are skipped; an interrupted episode is moved to
`interrupted/` (logged in `interrupted.jsonl`, never counted) and run again.
The memory store persists in `web-v1/memory/`.

## Result

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_agents.py report .task2-assets/comparison/web-v1
```

Prints completions per system, pairs where ours was better / baseline better,
and the exact two-sided sign test; writes `report.json`.

## Timing

About 7 minutes to load models, then ~4–5 minutes per episode:
80 episodes ≈ 6 hours.
