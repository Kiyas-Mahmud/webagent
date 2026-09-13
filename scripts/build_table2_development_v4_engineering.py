"""Build the development-v4 engineering receipt from checks that already ran.

The receipt binds the exact source bytes the frozen study will hash, plus the
regression log and the real-browser fixtures. It records engineering evidence
only: nothing here involves a model call or establishes task completion.
"""
from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from web_agent.eval.table2 import miniwob_study as study

study.CONFIG = study.ROOT / 'configs/eval/table2/miniwob_development_v4.json'
study.PROMPT = study.ROOT / 'configs/eval/table2/miniwob_recovery_prompt_v3.txt'
study.EXTRA_SOURCES = [study.ROOT / 'scripts/run_table2_development_v4.py',
                       study.ROOT / 'scripts/check_miniwob_progress_v3.py',
                       study.ROOT / 'scripts/check_miniwob_overlap_v4.py']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1])
    pytest_log = Path(sys.argv[2])
    passed = int(sys.argv[3])
    fixtures = {name: Path(path) for name, path in (item.split('=', 1) for item in sys.argv[4:])}
    out.mkdir(parents=True, exist_ok=True)
    (out/'regression-pytest.log').write_bytes(pytest_log.read_bytes())
    receipt = {
        'status': 'PASS',
        'evidence_type': 'ENGINEERING_NOT_LIVE',
        'profile': 'miniwob-development-v4',
        'declared_change': 'browser_hit_target_resolution',
        'tested_sources': study.sources(),
        'regression_tests': {'passed': passed, 'log': str(out/'regression-pytest.log'),
                             'sha256': sha(out/'regression-pytest.log')},
        'browser_fixtures': {},
        'limits': ('Scripted checks and a scripted browser sequence do not establish '
                   'frozen-model task completion, recovery benefit or memory benefit.'),
    }
    for name, path in fixtures.items():
        target = out/f'{name}'
        target.mkdir(parents=True, exist_ok=True)
        (target/'result.json').write_bytes(path.read_bytes())
        record = json.loads(path.read_text())
        assert record['status'] == 'PASS' and record['model_calls'] == 0, name
        receipt['browser_fixtures'][name] = {'path': str(target/'result.json'),
                                             'sha256': sha(target/'result.json'),
                                             'model_calls': record['model_calls']}
    study.write_json(out/'engineering-checks.json', receipt)
    print(json.dumps({'status': 'PASS', 'tested_sources': len(receipt['tested_sources']),
                      'regression_tests_passed': passed,
                      'browser_fixtures': sorted(receipt['browser_fixtures'])}))


if __name__ == '__main__':
    main()
