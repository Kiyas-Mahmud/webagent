"""Serial local worker transport with preserved errors and bounded waits."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import time


class Worker:
    def __init__(self, python, module, args, log, *, env=None):
        self.log = Path(log).open('x')
        self.process = subprocess.Popen([str(python), '-u', '-m', module, *map(str,args)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            bufsize=0, env=dict(os.environ, PYTHONPATH=str(Path('src').resolve()), **(env or {})))
        self.buffer = b''
        self.closed = False

    def call(self, op, timeout=600, **kwargs):
        if self.closed:raise RuntimeError('Worker is closed')
        self.process.stdin.write((json.dumps(dict(op=op, **kwargs), allow_nan=False)+'\n').encode())
        self.process.stdin.flush()
        deadline = time.monotonic()+timeout
        with selectors.DefaultSelector() as sel:
            sel.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                if self.closed:raise RuntimeError('Worker closed during '+op)
                remaining = deadline-time.monotonic()
                if b'\n' not in self.buffer:
                    if remaining <= 0:
                        raise TimeoutError('Worker request timed out: '+op)
                    if not sel.select(min(remaining,1.0)):continue
                    data = os.read(self.process.stdout.fileno(), 65536)
                    if not data:raise RuntimeError('Worker exited during '+op)
                    self.buffer += data
                    continue
                line, self.buffer = self.buffer.split(b'\n', 1)
                line = line.decode('utf-8', errors='replace')
                if not line.startswith('TASK2:'):
                    self.log.write(line); self.log.flush()
                    continue
                result = json.loads(line[6:])
                if not result['ok']:
                    raise RuntimeError(result.get('traceback',result['error']))
                return result['value']

    def close(self):
        self.closed = True
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait()
        self.log.close()
