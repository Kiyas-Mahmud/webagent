"""Full H0–H3 runner on a scripted real-browser fixture; zero model inference.

Fake backend callbacks and a local HTML page exercise production orchestration,
parsing, parameters, recovery, assessment, continuation, memory and reporting.
The four-request fixture stop/reward is not a MiniWoB task completion result.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.table2.test_hybrid_runner import engineering_run, ScriptedPage
from web_agent.benchmarks.browsergym_webarena import _action_code
from web_agent.benchmarks.miniwob_controls import CONTROL_INTERFACE, bind_controls, compatible_actions, validate_control_action
from web_agent.benchmarks.base import AdapterExecution
from web_agent.runtime.contracts import Observation, ExecutionStatus
from web_agent.runtime.recovery.strategies import build_recovery_target_evidence


class BrowserPage(ScriptedPage):
    def __init__(self,folder,**kwargs):
        super().__init__(folder,**kwargs)
        self.worker=subprocess.Popen(['/home/aiub/kiyas/table2-envs/miniwob-feasibility/bin/python',
            str(Path(__file__).with_name('check_hybrid_action_adapter.py')),'--browser-worker'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        self.closed=False
    def rpc(self,**payload):
        self.worker.stdin.write(json.dumps(payload)+'\n');self.worker.stdin.flush()
        reply=json.loads(self.worker.stdout.readline())
        if not reply['ok']:raise RuntimeError(reply['error'])
        return reply.get('result')
    def observe(self,*,stage,prior_action_id=None):
        self.i+=1;oid=f'{self.eid}:obs:{self.i}';path=self.folder/f'observation-{self.i}.png'
        snapshot=self.rpc(op='observe',path=str(path))
        controls=bind_controls(snapshot['controls'],oid)
        state={'control_interface':CONTROL_INTERFACE,'visible_controls':controls,
            'recovery_target_evidence':build_recovery_target_evidence(task=self.task,observation_id=oid,
                compatible_actions=compatible_actions(controls))}
        if self.hybrid_interface_version==3:state['hybrid_interface_version']=3
        obs=Observation(observation_id=oid,episode_id=self.eid,stage=stage,prior_action_id=prior_action_id,
            screenshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),screenshot_path=str(path),width=800,height=400,
            url=snapshot['url'],title='',page_state=state)
        (self.folder/f'observation-{self.i}.json').write_text(json.dumps(obs.to_dict()))
        self.current=obs
        return obs
    def execute(self,action):
        obs=type('Current',(),{'observation_id':self.current.observation_id,'current_page_state':self.current.page_state})()
        expected=validate_control_action(action.action_type.value,action.parameters,action.bbox,obs,
            control_id=action.parameters.get('target_control_id'))
        result=self.rpc(op='execute',kind=action.action_type.value,parameters=dict(action.parameters),bbox=action.bbox,
            observation_id=obs.observation_id,expected_control=expected,
            hybrid_interface_version=self.hybrid_interface_version,
            code=_action_code(action,stable_target_identity=self.hybrid_interface_version==3))
        if action.action_type.value=='TYPE':assert result['text']==action.parameters['text']
        self.n+=1
        self.outcomes.append({'terminated':self.n>=4,'truncated':False,'raw_reward':1. if self.n>=4 else 0.,
            'evidence_type':'SCRIPTED_FOUR_REQUEST_FIXTURE','observed_state':result})
        (self.folder/'browser-outcomes.json').write_text(json.dumps(self.outcomes))
        return AdapterExecution(status=ExecutionStatus.EXECUTED,state_changed=True)
    def close(self):
        if not self.closed:
            try:self.rpc(op='close')
            finally:
                self.worker.wait(timeout=15);self.closed=True


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--interface-version',type=int,choices=(1,2,3),default=1)
    args=parser.parse_args()
    root=args.output.resolve()
    with pytest.MonkeyPatch.context() as patch:
        plan,rows,audit=engineering_run(root,patch,page_class=BrowserPage,interface_version=args.interface_version)
    assert audit['status']=='PASS',audit['errors']
    assert audit['live_path_verified'] is False
    for row in rows:
        assert 'error' not in row and row['summary']['executor_steps']==4,row
    assessments=[e['payload']['predicted_assessment'] for path in root.rglob('recoveries.jsonl')
        for line in path.read_text().splitlines() if (e:=json.loads(line))['event_type']=='recovery_attempt' and 'predicted_assessment' in e['payload']]
    assert len(assessments)==8 and all(not a['predicted_failure_resolved'] and not a['predicted_progress'] for a in assessments)
    receipt={'status':'PASS','evidence_type':'SCRIPTED_REAL_BROWSER_ENGINEERING','actual_model_inferences':0,
        'hybrid_interface_version':args.interface_version,
        'new_embeddings':0,'browser_requests':sum(r['summary']['executor_steps'] for r in rows),
        'logical_dispatches':sum(r['summary']['model_call_count'] for r in rows),
        'negative_assessments':len(assessments),'systems':['H0','H1','H2','H3'],'sequences':['typing','click-only'],
        'live_evaluation_episodes':0,'claim':'Verifies wiring and continuation; does not measure model performance.'}
    (root/'engineering-result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
