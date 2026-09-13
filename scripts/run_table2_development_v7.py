"""Development-only executable-action-selection study.

Adds one declared change to development-v6: the trained policy selects among the
action classes the current page can execute, using the observation's own
registered visible-target evidence — the same oracle-blind evidence the
generated interfaces already receive as per-control `supported_actions`.

The learned distribution is not modified and is still logged in full; only the
set the argmax ranges over narrows. Prompts, planner context, control
projection, decoding, checkpoint, memory and seeds are unchanged.
"""
from web_agent.eval.table2 import miniwob_study as study

study.CONFIG = study.ROOT / 'configs/eval/table2/miniwob_development_v7.json'
study.PROMPT = study.ROOT / 'configs/eval/table2/miniwob_recovery_prompt_v3.txt'
study.ENGINEERING = study.ROOT.parent / 'table2-evidence/miniwob-development-v7-engineering'
study.EXTRA_SOURCES = [study.ROOT / 'scripts/run_table2_development_v7.py',
                       study.ROOT / 'scripts/check_miniwob_progress_v3.py',
                       study.ROOT / 'scripts/check_miniwob_overlap_v4.py']

if __name__ == '__main__':
    study.main()
