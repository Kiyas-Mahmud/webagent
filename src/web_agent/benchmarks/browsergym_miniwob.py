"""Thin MiniWoB adapter for the existing episode runtime."""
from pathlib import Path
import json
import subprocess
import threading

from web_agent.benchmarks.base import BenchmarkAdapter, AdapterExecution
from web_agent.benchmarks.browsergym_webarena import _action_code
from web_agent.benchmarks.miniwob_controls import CONTROL_INTERFACE, bind_controls, compatible_actions, validate_control_action
from web_agent.runtime.contracts import Observation, ObservationStage, ExecutionStatus, OpaqueTerminalSignal, canonical_sha256
from web_agent.runtime.recovery.strategies import build_recovery_target_evidence, build_point_target_evidence


class MiniWoBAdapter(BenchmarkAdapter):
    benchmark_id='miniwob'
    benchmark_version='0.14.3'

    def __init__(self, folder, *, browser_python, hybrid_interface_version=1):
        self.hybrid_interface_version=hybrid_interface_version
        self.folder=Path(folder); self.folder.mkdir(exist_ok=False,parents=True)
        self.lock=threading.Lock(); self.i=0; self.last=None; self.current=None
        self.log=(self.folder/'browser.log').open('x')
        self.proc=subprocess.Popen([browser_python,'-m','web_agent.benchmarks.miniwob_worker',str(self.folder)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True)

    def rpc(self,op,**kw):
        with self.lock:
            self.proc.stdin.write(json.dumps(dict(op=op,**kw))+'\n'); self.proc.stdin.flush()
            line=self.proc.stdout.readline()
            if not line: raise RuntimeError('browser worker exited')
            result=json.loads(line)
            if not result['ok']: raise RuntimeError(result['error'])
            return result['value']

    def reset(self,task,*,episode_id,seed):
        self.task=task; self.eid=episode_id
        goal=self.rpc('reset',seed=seed,task=task.task_id.removeprefix('miniwob.'))['goal']
        if goal!=task.goal: raise ValueError('reset goal differs from frozen task binding')
        return self.observe(stage=ObservationStage.RESET)

    def observe(self,*,stage,prior_action_id=None):
        self.i+=1
        result=self.rpc('observe',name=f'observation-{self.i}.png')
        self.last=result['screenshot_sha256']; oid=f'{self.eid}:obs:{self.i}'
        controls=bind_controls(result.pop('control_snapshot'),oid)
        evidence=compatible_actions(controls)
        state={'control_interface':CONTROL_INTERFACE,'visible_controls':controls,
            'observable_select_controls':[{'target_bbox':c['target_bbox'],'candidate_options':c['candidate_options']} for c in controls if c['tag']=='select'],
            'recovery_target_evidence':build_recovery_target_evidence(task=self.task,observation_id=oid,compatible_actions=evidence),
            'visible_point_targets':build_point_target_evidence(task=self.task,observation_id=oid,compatible_actions=evidence)}
        if self.hybrid_interface_version == 3:
            state['hybrid_interface_version']=3
        observation=Observation(observation_id=oid,episode_id=self.eid,stage=stage,prior_action_id=prior_action_id,page_state=state,**result)
        self.current=observation
        (self.folder/f'observation-{self.i}.json').write_text(observation.to_json()+'\n')
        return observation

    def execute(self,action):
        from web_agent.runtime.observation import ObservationBuilder
        view=ObservationBuilder().pre_action(self.task,self.current)
        try: expected=validate_control_action(action.action_type.value,action.parameters,action.bbox,view,
            control_id=action.parameters.get('target_control_id'))
        except ValueError as exc:
            return AdapterExecution(status=ExecutionStatus.REJECTED,state_changed=False,error_kind='target_resolution_rejected',message=str(exc))
        before=self.last
        result=self.rpc('execute',code=_action_code(action,stable_target_identity=self.hybrid_interface_version==3),expected_control=expected,
            observation_id=self.current.observation_id,action_type=action.action_type.value,
            parameters=dict(action.parameters),bbox=action.bbox,
            hybrid_interface_version=self.hybrid_interface_version)
        self.i+=1
        after=self.rpc('observe',name=f'execution-{self.i}.png')
        error=result['action_error']
        return AdapterExecution(status=ExecutionStatus.REJECTED if result['rejected'] else ExecutionStatus.ERROR if error else ExecutionStatus.EXECUTED,
            state_changed=before!=after['screenshot_sha256'],error_kind='target_resolution_rejected' if result['rejected'] else 'BROWSER_ACTION_ERROR' if error else None,message=error)

    def terminal_signal(self,task,binding=None):
        return OpaqueTerminalSignal(event_id=f'{self.eid}:terminal:{self.i}',
            token_sha256=canonical_sha256({'episode':self.eid,'index':self.i,'binding':binding.record_sha256 if binding else None}),
            terminate=self.rpc('terminal')['terminate'])

    def close(self):
        if self.proc.poll() is None:
            try: self.rpc('close')
            finally:
                self.proc.stdin.close()
                self.proc.wait(timeout=20)
        self.log.close()
