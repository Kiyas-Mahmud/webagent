"""Development-only overlapping-target study with a frozen model-facing interface.

Identical to the development-v4 attempt except that the control projection's
advertised schema label no longer follows the observation's interface version.
That one character had reached the E0 prompt and moved its generation, which
confounded execution with re-prompting. Every model input is now byte-identical
to development-v3, so only execution differs.
"""
from web_agent.eval.table2 import miniwob_study as study

study.CONFIG = study.ROOT / 'configs/eval/table2/miniwob_development_v5.json'
study.PROMPT = study.ROOT / 'configs/eval/table2/miniwob_recovery_prompt_v3.txt'
study.ENGINEERING = study.ROOT.parent / 'table2-evidence/miniwob-development-v5-engineering'
study.EXTRA_SOURCES = [study.ROOT / 'scripts/run_table2_development_v5.py',
                       study.ROOT / 'scripts/check_miniwob_progress_v3.py',
                       study.ROOT / 'scripts/check_miniwob_overlap_v4.py']

if __name__ == '__main__':
    study.main()
