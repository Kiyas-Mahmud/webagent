"""Development-only overlapping-target study; original runner and prompt reused.

Every model input is byte-identical to development-v3: same prompt, same planner
context, same control projection and the same advertised schema label. Only
target execution differs, in the two places that decided target identity by
counting box intersections instead of asking the browser what a click reaches.
"""
from web_agent.eval.table2 import miniwob_study as study

study.CONFIG = study.ROOT / 'configs/eval/table2/miniwob_development_v6.json'
study.PROMPT = study.ROOT / 'configs/eval/table2/miniwob_recovery_prompt_v3.txt'
study.ENGINEERING = study.ROOT.parent / 'table2-evidence/miniwob-development-v6-engineering'
study.EXTRA_SOURCES = [study.ROOT / 'scripts/run_table2_development_v6.py',
                       study.ROOT / 'scripts/check_miniwob_progress_v3.py',
                       study.ROOT / 'scripts/check_miniwob_overlap_v4.py']

if __name__ == '__main__':
    study.main()
