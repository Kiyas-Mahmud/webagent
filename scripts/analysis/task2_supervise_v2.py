"""Supervise a campaign_v2 run; never changes a model request, action or episode result.

The coordinator runs under a systemd shutdown/sleep inhibitor, so a desktop
power-off shows a warning instead of silently killing the run (the 2026-09-16
native-evaluation-v1 stop). Infrastructure failures stop without retries.
"""
import argparse,json,os,subprocess,time
from pathlib import Path
import psutil
from web_agent.eval.task1.core import write_new,file_hash

p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();root=Path(args.run).resolve()
plan=json.loads((root/'plan.json').read_text())
write_new(root/f'launch-supervision-{time.strftime("%Y%m%dT%H%M%S")}.json',{'script_sha256':file_hash(__file__),'script':str(Path(__file__).resolve()),
 'rule':'Run under a shutdown/sleep inhibitor. After all planned episodes are saved and all child workers exit, allow 30 seconds for coordinator shutdown, then terminate only that idle coordinator. Infrastructure failures stop without automatic retries.',
 'phase':plan['phase'],'plan_sha256':file_hash(root/'plan.json')})
command=['systemd-inhibit','--what=shutdown:sleep:idle','--who=task2-'+plan['phase'],
         '--why=Task 2 browser campaign running: '+root.name,'--mode=block',
         '.task2-assets/browser-use-env/bin/python','-u','-m','web_agent.eval.task2.campaign_v2','run','--out',str(root)]
with (root/'execution.log').open('a') as stream:
 proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(Path('src').resolve())))
 idle_since=None;cleanup=None
 memory_stop=None
 while proc.poll() is None:
  time.sleep(2)
  available=psutil.virtual_memory().available/1e9
  if available<12:
   # Host protection: GPU memory is host RAM on GB10; never let a run freeze the machine.
   memory_stop={'available_gb':available,'time':time.time()}
   for child in psutil.Process(proc.pid).children(recursive=True):
    try:child.kill()
    except psutil.NoSuchProcess:pass
   proc.kill();proc.wait();break
  state=json.loads((root/'status.json').read_text())
  try:children=[c for c in psutil.Process(proc.pid).children(recursive=True) if c.pid!=state.get('pid')]
  except psutil.NoSuchProcess:break
  finished=state['stage']=='execution_complete_audit_pending' and state['saved']==state['expected']
  failed=state['stage']=='review_required'
  if (finished or failed) and not children:
   if idle_since is None:idle_since=time.monotonic()
   if time.monotonic()-idle_since>=30:
    cleanup={'reason':'idle coordinator after worker exit','episodes_complete':finished,'pid':state.get('pid'),'time':time.time()}
    # Terminate the idle coordinator itself; its inhibitor wrapper then exits.
    try:psutil.Process(state['pid']).terminate()
    except psutil.NoSuchProcess:pass
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
  else:idle_since=None
 exit_code=proc.wait()
write_new(root/f'supervision-result-{time.strftime("%Y%m%dT%H%M%S")}.json',{'coordinator_exit_code':exit_code,'cleanup':cleanup,'host_memory_stop':memory_stop})
if memory_stop:raise SystemExit('Stopped: host memory below 12 GB; run killed to protect the machine; review required')
state=json.loads((root/'status.json').read_text())
if state['stage']!='execution_complete_audit_pending' or state['saved']!=state['expected']:
 raise SystemExit('Stopped: incomplete run requires review; no automatic retry')
with (root.parent/(root.name+'-audit.log')).open('x') as stream:
 result=subprocess.run(['.venv/bin/python','-m','web_agent.eval.task2.reporting',str(root)],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(Path('src').resolve())))
if result.returncode:raise SystemExit('Completed execution but independent audit requires review')
state.update(stage='audited_complete',audit='PASS')
(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
print('All episodes saved; independent audit PASS',flush=True)
