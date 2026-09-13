"""Separate, development-only planner/continuation study; original runner reused."""
from web_agent.eval.table2 import miniwob_study as study

study.CONFIG = study.ROOT / 'configs/eval/table2/miniwob_development_v3.json'
study.PROMPT = study.ROOT / 'configs/eval/table2/miniwob_recovery_prompt_v3.txt'
study.ENGINEERING = study.ROOT.parent / 'table2-evidence/miniwob-development-v3-engineering'
study.EXTRA_SOURCES = [study.ROOT / 'scripts/run_table2_development_v3.py', study.ROOT / 'scripts/check_miniwob_progress_v3.py']

if __name__ == '__main__':
    study.main()
