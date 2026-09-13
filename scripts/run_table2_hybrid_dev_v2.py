"""Separate development-only v2 candidate. Completed v1 outputs cannot resume."""
from web_agent.eval.table2 import miniwob_study as study

study.CONFIG = study.ROOT / 'configs/eval/table2/miniwob_hybrid_dev_v2.json'
study.PROMPT = study.ROOT / 'configs/eval/table2/miniwob_recovery_prompt_v3.txt'
study.ENGINEERING = study.ROOT.parent / 'table2-evidence/hybrid-interface-v2-engineering'
study.EXTRA_SOURCES = [study.ROOT / 'scripts/run_table2_hybrid_dev_v2.py',
    study.ROOT / 'configs/eval/table2/miniwob_hybrid_action_prompt_v2.txt',
    study.ROOT / 'scripts/replay_hybrid_interface_v2.py',
    study.ROOT / 'scripts/check_hybrid_runner.py']

if __name__ == '__main__':
    study.main()
