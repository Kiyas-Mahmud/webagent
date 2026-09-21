"""Supervise lifecycle only; never changes a model request, action or episode result."""
import argparse,json,os,subprocess,time
from pathlib import Path
import psutil
from web_agent.eval.task1.core import write_new,file_hash

p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();root=Path(args.run).resolve()
log=root/'execution.log'
write_new(root/'launch-supervision.json',{'script_sha256':file_hash(__file__),'script':str(Path(__file__).resolve()),
 'rule':'After all planned episodes are saved and all child workers exit, allow 30 seconds for coordinator shutdown, then terminate only that idle coordinator. Infrastructure failures stop without automatic retries.',
 'development_readiness_sha256':file_hash('.task2-assets/task2-development-v4/development-readiness.json')})
with log.open('x') as stream:
 proc=subprocess.Popen(['.task2-assets/browser-use-env/bin/python','-u','-m','web_agent.eval.task2.campaign','run','--out',str(root)],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(Path('src').resolve())))
 idle_since=None;cleanup=None
 while proc.poll() is None:
  time.sleep(2)
  state=json.loads((root/'status.json').read_text())
  try:children=psutil.Process(proc.pid).children(recursive=True)
  except psutil.NoSuchProcess:break
  finished=state['stage']=='execution_complete_audit_pending' and state['saved']==state['expected']
  failed=state['stage']=='review_required'
  if (finished or failed) and not children:
   if idle_since is None:idle_since=time.monotonic()
   if time.monotonic()-idle_since>=30:
    cleanup={'reason':'idle coordinator after worker exit','episodes_complete':finished,'pid':proc.pid,'time':time.time()}
    proc.terminate()
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
  else:idle_since=None
 exit_code=proc.wait()
write_new(root/'supervision-result.json',{'coordinator_exit_code':exit_code,'cleanup':cleanup})
state=json.loads((root/'status.json').read_text())
if state['stage']!='execution_complete_audit_pending' or state['saved']!=state['expected']:
 raise SystemExit('Stopped: incomplete run requires review; no automatic retry')
with (root.parent/(root.name+'-audit.log')).open('x') as stream:
 result=subprocess.run(['.venv/bin/python','-m','web_agent.eval.task2.reporting',str(root)],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(Path('src').resolve())))
if result.returncode:raise SystemExit('Completed execution but independent audit requires review')
state.update(stage='audited_complete',audit='PASS')
(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
print('All episodes saved; independent audit PASS',flush=True)
