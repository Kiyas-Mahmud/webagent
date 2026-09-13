"""CPU-only reproduction of audited hybrid failure routes; no model/browser run.

Run from the repository with .venv/bin/python scripts/analyze_hybrid_failure_routes.py.
Uses scripted test callbacks, creates a fresh /tmp evidence folder, and does not
read evaluation outcomes, checkpoint weights, source datasets or image files.
The emitted model-call counts are dispatch accounting for scripted callbacks.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from web_agent.runtime.contracts import HybridSystemID, TaskSpecification
from web_agent.runtime.hybrid_action_policy import HybridActionPolicy
from web_agent.runtime.policy import CallablePolicyAdapter, PolicyKind, SystemPolicy
from web_agent.runtime.protocol import RuntimeProtocol, switches_for
from web_agent.runtime.episode import EpisodeRunner
from web_agent.runtime.executor import Executor
from web_agent.runtime.action_parameters import DeterministicParameterProvider
from web_agent.runtime.recovery.controller import RecoveryController
from tests.table2.test_hybrid_runner import ScriptedPage
from tests.table2.test_hybrid_agent import advice

root=Path(tempfile.mkdtemp(prefix='recovery-loop-audit-'))
for name,raw in [('parse','{}'),('target',json.dumps(dict(action_type='CLICK',target='missing',bbox=None,value=None)))]:
 counts=dict(generation=0,trained_advice=0,diagnosis=0,loop_record=0)
 def generate(**kw):
  counts['generation']+=1
  return raw
 def action(t,o,rng):
  counts['trained_advice']+=1
  return advice(o)
 def diagnose(*args,**kw):
  counts['diagnosis']+=1
  raise AssertionError('unexpected diagnosis')
 trained=CallablePolicyAdapter(policy_id='trained',policy_version='v1',kind=PolicyKind.TRAINED,
  checkpoint_sha256='a'*64,action_predictor=action,transition_predictor=diagnose,recovery_predictor=diagnose)
 protocol=RuntimeProtocol(protocol_id='audit',campaign_id='audit',benchmark_id='miniwob',benchmark_version='0.14.3',
  provider_id='deterministic-parameter-provider',provider_version='v1')
 eid='audit:H2:task:repeat-0:seed-42'
 actor=HybridActionPolicy(system='H2',episode_id=eid,base=SimpleNamespace(_generate=generate),trained_policy=trained,
  evidence_dir=root/name/'proposals',interface_version=2)
 page=ScriptedPage(root/name/'page')
 controller=RecoveryController(protocol.budgets)
 runner=EpisodeRunner(protocol=protocol,system_policy=SystemPolicy(actor,switches_for(HybridSystemID.H2)),
  provider=DeterministicParameterProvider(),executor=Executor(page,budgets=protocol.budgets),recovery_controller=controller,
  hybrid_continuation=True)
 original=runner.loop_guard.record
 def record(*args,**kw):
  counts['loop_record']+=1
  return original(*args,**kw)
 runner.loop_guard.record=record
 task=TaskSpecification(task_id='task',goal='Scripted reject audit',benchmark_id='miniwob',benchmark_version='0.14.3',start_state_id='initial',development_partition=True)
 summary=runner.run(task,repeat_id=0,model_seed=42)
 print(json.dumps(dict(case=name,counts=counts,terminal=summary.terminal_reason.value,executor_steps=summary.executor_steps,
  recovery_attempts=summary.recovery_attempts,browser_actions=page.n,model_calls=summary.model_call_count)))
print('evidence',root)

# Same visible controls, new capture IDs: distinguish stable state and action hashing.
from web_agent.runtime.contracts import Observation, ObservationStage, ConcreteAction, ActionType, RuntimeTaskView, RecoveryDecision, RecoveryStrategy
from web_agent.runtime.decision import LoopGuard
from web_agent.runtime.protocol import LoopRule, RuntimeBudgets
from web_agent.runtime.recovery.strategies import build_recovery_target_evidence
from tests.table2.test_development_v3 import observation

def loop_probe(states, named):
 guard=LoopGuard(LoopRule());out=[]
 for n,s in enumerate(states,1):
  p=observation(oid=f'e:obs:{n}')
  o=Observation(observation_id=p.observation_id,episode_id='e',stage=ObservationStage.RESET,
   screenshot_sha256=s*64,screenshot_path=None,width=p.width,height=p.height,url=p.url,title=p.title,page_state=p.current_page_state)
  a=ConcreteAction(action_id=f'a{n}',source_decision_id=f'd{n}',action_type=ActionType.CLICK,bbox=(.1,.2,.3,.1),
   parameters={'target_x':.25,'target_y':.25,**({'target_control_id':f'o{n}:c0'} if named else {})})
  out.append(guard.record(o,a))
 return out
for sequence in ['aaa','ababa']:
 print(json.dumps(dict(case='loop_fingerprint',states=sequence,refreshing_id=loop_probe(sequence,True),without_id=loop_probe(sequence,False))))

a=ConcreteAction(action_id='a',source_decision_id='d',action_type=ActionType.CLICK,bbox=(.1,.2,.3,.1),
 parameters={'target_x':.25,'target_y':.25,'target_bbox':[.1,.2,.3,.1],'button':'left','click_count':1,'target_control_id':'o1:c0'})
t=RuntimeTaskView(task_id='task',goal='visible task')
d=RecoveryDecision(decision_id='rd',incident_id='incident',strategy=RecoveryStrategy.RETRY,trigger_sources=('policy',),diagnosis='NO_EFFECT')
for n in [1,2]:
 o=observation(oid=f'episode:obs:{n}')
 o.current_page_state['visible_controls'][0]['hit_point']=[.25,.25]
 o.current_page_state['recovery_target_evidence']=build_recovery_target_evidence(task=t,observation_id=o.observation_id,compatible_actions=[a])
 p=RecoveryController(RuntimeBudgets()).begin_attempt(d,a,task=t,post_failure_observation=o)
 print(json.dumps(dict(case='retry_identity',observation=o.observation_id,current_control=o.current_page_state['visible_controls'][0]['control_id'],status=p.resolution_status,reason=p.rejection_reason)))

