"""Development-only overlapping-target study; original runner and prompt reused.

Only the target-resolution rule changes from development-v3. The recovery
prompt, planner context, budgets, tasks, resets and model seed are unchanged, so
the model-facing interface is identical and the difference is attributable to
execution, not to re-prompting.
"""
from web_agent.eval.table2 import miniwob_study as study

study.CONFIG = study.ROOT / 'configs/eval/table2/miniwob_development_v4.json'
study.PROMPT = study.ROOT / 'configs/eval/table2/miniwob_recovery_prompt_v3.txt'
study.ENGINEERING = study.ROOT.parent / 'table2-evidence/miniwob-development-v4-engineering'
study.EXTRA_SOURCES = [study.ROOT / 'scripts/run_table2_development_v4.py',
                       study.ROOT / 'scripts/check_miniwob_progress_v3.py',
                       study.ROOT / 'scripts/check_miniwob_overlap_v4.py']

if __name__ == '__main__':
    study.main()
