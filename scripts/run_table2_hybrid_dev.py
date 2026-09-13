"""Development-only H0–H3 launcher. Task 6 performs prepare/run; never final evaluation."""
from web_agent.eval.table2 import miniwob_study as study

study.CONFIG = study.ROOT / 'configs/eval/table2/miniwob_hybrid_dev_v1.json'
study.PROMPT = study.ROOT / 'configs/eval/table2/miniwob_recovery_prompt_v3.txt'
study.ENGINEERING = study.ROOT.parent / 'table2-evidence/hybrid-dev-v1-task5-engineering'
study.EXTRA_SOURCES = [study.ROOT / 'scripts/run_table2_hybrid_dev.py',
    study.ROOT / 'configs/eval/table2/miniwob_hybrid_action_prompt_v1.txt',
    study.ROOT / 'scripts/check_hybrid_action_adapter.py',
    study.ROOT / 'scripts/check_hybrid_memory.py',
    study.ROOT / 'scripts/check_hybrid_runner.py',
    study.ROOT / 'scripts/check_miniwob_overlap_v4.py']

if __name__ == '__main__':
    study.main()
