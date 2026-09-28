"""Run the Task 2 regression suite and bind its PASS to exact producing-source hashes."""
import subprocess
import sys
from pathlib import Path

from web_agent.eval.task1.core import file_hash, write_new
from web_agent.eval.task2.campaign import source_paths

receipt = Path(sys.argv[1]); log = receipt.with_suffix('.log')
before = {str(p): file_hash(p) for p in source_paths()}
with log.open('x') as stream:
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/task2'], stdout=stream, stderr=subprocess.STDOUT)
after = {str(p): file_hash(p) for p in source_paths()}
if before != after:
    raise SystemExit('Sources changed while testing')
summary = log.read_text().strip().splitlines()[-1]
write_new(receipt, {'status': 'PASS' if result.returncode == 0 else 'FAIL', 'summary': summary,
                    'tests': int(summary.split()[0]) if result.returncode == 0 else None,
                    'log': str(log), 'log_sha256': file_hash(log), 'sources': before})
print(summary)
