"""Bounded proposal repair and stable retry targets; development only."""
from web_agent.eval.table2 import miniwob_study as study

study.CONFIG = study.ROOT / 'configs/eval/table2/miniwob_hybrid_dev_v3.json'
study.PROMPT = study.ROOT / 'configs/eval/table2/miniwob_recovery_prompt_v4.txt'
study.ENGINEERING = study.ROOT.parent / 'table2-evidence/hybrid-interface-v3-engineering'
study.EXTRA_SOURCES = [study.ROOT / name for name in (
    'scripts/run_table2_hybrid_dev_v3.py',
    'configs/eval/table2/miniwob_hybrid_action_prompt_v3.txt',
    'configs/eval/table2/miniwob_hybrid_dev_v3_scope.json',
    'scripts/check_hybrid_runner.py',
    'scripts/check_hybrid_action_adapter.py',
    'scripts/check_hybrid_stable_retry_v3.py',
)]

if __name__ == '__main__':
    study.main()
