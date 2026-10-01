# Baseline vs ours on real websites — runbook

> **Status 2026-10-01.** `web-v2` is finished (76/80 episodes; Stanford lost to
> load timeouts): A 27/38, C 20/38 after correcting one false completion. It
> used the old synchronous advice and is superseded by recovery v3.3 (see
> [TASK2_FAILURE_AWARE_AGENT_METHOD.md](TASK2_FAILURE_AWARE_AGENT_METHOD.md)).
> The next plan, `web-v3`, is **not frozen yet**: it waits for the task-suite
> fixes (exact-page rules, arXiv/Debian goals, held-out tasks). The commands
> below are unchanged; replace `web-v2` with the new plan directory.

Plan: `.task2-assets/comparison/web-v2/` (frozen with `compare_agents.py freeze`).
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
  .task2-assets/browser-use-env/bin/python -u scripts/compare_agents.py run .task2-assets/comparison/web-v2 \
  2>&1 | grep --line-buffered -vE "^(INFO|DEBUG|WARNING)|^\s*$|Loading checkpoint|^=+$" \
  | tee -a .task2-assets/comparison/web-v2.log
```

Watch progress in a second terminal: `python3 scripts/progress.py`

## Stopping and resuming

Safe to stop (Ctrl+C, or at the end of lab time). Rerun the same `run` command:
finished episodes are skipped; an interrupted episode is moved to
`interrupted/` (logged in `interrupted.jsonl`, never counted) and run again.
The memory store persists in `web-v2/memory/`.

## Result

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_agents.py report .task2-assets/comparison/web-v2
```

Prints completions per system, pairs where ours was better / baseline better,
and the exact two-sided sign test; writes `report.json`.

## Timing

About 7 minutes to load models, then ~4–5 minutes per episode:
80 episodes ≈ 6 hours.
