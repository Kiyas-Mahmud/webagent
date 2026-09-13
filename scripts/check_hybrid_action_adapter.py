"""Real-browser Task 2 fixture with scripted advice/generation, no model weights.

Run using the existing model Python and PLAYWRIGHT_BROWSERS_PATH. This checks
the new adapter -> existing parameter provider -> v6 target validation -> shared
Playwright action code. It is not a MiniWoB episode or performance measurement.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import subprocess


def browser_worker():
    # This branch runs exclusively in the existing browser environment.
    from playwright.sync_api import sync_playwright
    from web_agent.benchmarks.miniwob_controls import CONTROL_INTERFACE, CONTROL_JAVASCRIPT, bind_controls, validate_control_action
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True)
        try:
            page=browser.new_page(viewport={'width':800,'height':400})
            html='<label for="field">Field</label><input id="field"><select id="select"><option>one</option><option>two</option></select><label><input id="choice" type="checkbox">Choice</label><a id="finish" href="http://hybrid.fixture.local/finish">Finish</a><div style="height:1200px"></div>'
            page.route('**/*',lambda route:route.fulfill(body=html,content_type='text/html'))
            page.goto('http://hybrid.fixture.local/start')
            for line in sys.stdin:
                q=json.loads(line)
                try:
                    if q['op']=='close':
                        print(json.dumps({'ok':True}),flush=True);break
                    if q['op']=='observe':
                        page.screenshot(path=q['path']);Path(q['path']).chmod(0o444)
                        result={'url':page.url,'controls':page.evaluate(CONTROL_JAVASCRIPT)}
                    else:
                        controls=bind_controls(page.evaluate(CONTROL_JAVASCRIPT),q['observation_id'])
                        obs=SimpleNamespace(observation_id=q['observation_id'],current_page_state={
                            'control_interface':CONTROL_INTERFACE,'visible_controls':controls})
                        if q.get('hybrid_interface_version')==3:
                            obs.current_page_state['hybrid_interface_version']=3
                        actual=validate_control_action(q['kind'],q['parameters'],q['bbox'],obs,
                            control_id=q['parameters'].get('target_control_id'))
                        expected=q['expected_control']
                        if expected is not None:
                            fields=('source_id','tag','input_type','role','target_bbox','name','text',
                                    'accessible_names','candidate_options','disabled','readonly','contenteditable','hit_point')
                            if any(actual[k]!=expected[k] for k in fields):
                                raise ValueError('STALE_TARGET_BEFORE_EXECUTION')
                        exec(q['code'],{'page':page})
                        result={'text':page.locator('#field').input_value(),
                            'selected':page.locator('#select').input_value(),
                            'checked':page.locator('#choice').is_checked(),
                            'link_focused':page.locator('a').evaluate('(e)=>e===document.activeElement'),
                            'scroll_y':page.evaluate('window.scrollY'),'url':page.url}
                    print(json.dumps({'ok':True,'result':result}),flush=True)
                except Exception as exc:
                    print(json.dumps({'ok':False,'error':str(exc)}),flush=True)
        finally:browser.close()


if len(sys.argv)>1 and sys.argv[1]=='--browser-worker':
    browser_worker()
    raise SystemExit(0)

from web_agent.benchmarks.browsergym_webarena import _action_code
from web_agent.benchmarks.miniwob_controls import CONTROL_INTERFACE, CONTROL_JAVASCRIPT, bind_controls, validate_control_action
from web_agent.runtime.action_parameters import DeterministicParameterProvider
from web_agent.runtime.contracts import (
    ActionType, CausalHistoryEntry, ConcreteAction, ExecutionResult, ExecutionStatus,
    PolicyObservation, PreActionDecision, RuntimeTaskView,
)
from web_agent.runtime.hybrid_action_policy import HybridActionPolicy, HybridActionContext
from web_agent.runtime.model_calls import ModelCallLedger, activate_model_call_ledger, deactivate_model_call_ledger
from web_agent.runtime.policy import CallablePolicyAdapter, PolicyKind


def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    goal='Engineering fixture: scripted actions, not an autonomous task.'
    task=RuntimeTaskView(task_id='fixture',goal=goal)
    sequence=[('TYPE','field',' Exact  Case! '),('SELECT','select','two'),('CLICK','choice',None),
              ('PRESS_KEY',None,'TAB'),('SCROLL',None,'down'),('NAVIGATE',None,'http://hybrid.fixture.local/finish')]
    index=0
    def generate(**kw):
        kind,source,value=sequence[index]
        controls=kw['observation'].current_page_state['visible_controls']
        target=next(c['control_id'] for c in controls if c['source_id']==source) if source else None
        return json.dumps(dict(action_type=kind,target=target,bbox=None,value=value))
    def trained(t,o,rng):
        return PreActionDecision(decision_id=f'fixture-advice-{index}',observation_id=o.observation_id,
            action_type=ActionType.NAVIGATE,action_probabilities={a.value:float(a==ActionType.NAVIGATE) for a in ActionType},
            bbox=(.1,.1,.2,.2),grounding_confidence=.8,confidence_before=.2,
            input_observation_ids=(o.observation_id,),policy_id='fixture-trained',policy_version='v1')
    trained_policy=CallablePolicyAdapter(policy_id='fixture-trained',policy_version='v1',kind=PolicyKind.TRAINED,
        action_predictor=trained,checkpoint_sha256='b'*64)
    actor=HybridActionPolicy(system='H1',episode_id='fixture',base=SimpleNamespace(_generate=generate),
                            evidence_dir=root/'proposals',trained_policy=trained_policy)
    ledger_events=[];ledger=ModelCallLedger('fixture',sink=ledger_events.append)
    token=activate_model_call_ledger(ledger)
    actions=[];history=[];executions=[]
    worker=subprocess.Popen(['/home/aiub/kiyas/table2-envs/miniwob-feasibility/bin/python',
        str(Path(__file__).resolve()),'--browser-worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    def request(**payload):
        worker.stdin.write(json.dumps(payload)+'\n');worker.stdin.flush()
        line=worker.stdout.readline()
        if not line:raise RuntimeError('browser fixture worker exited')
        reply=json.loads(line)
        if not reply['ok']:raise RuntimeError(reply['error'])
        return reply.get('result')
    try:
        def observe(number):
            oid=f'fixture:obs:{number}';path=root/f'observation-{number}.png'
            snapshot=request(op='observe',path=str(path))
            controls=bind_controls(snapshot['controls'],oid)
            return PolicyObservation(task_id='fixture',goal=goal,observation_id=oid,
                screenshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),screenshot_path=str(path),
                width=800,height=400,url=snapshot['url'],title='',current_page_state={
                    'control_interface':CONTROL_INTERFACE,'visible_controls':controls,
                    'observable_select_controls':[{'target_bbox':c['target_bbox'],'candidate_options':c['candidate_options']} for c in controls if c['tag']=='select'],
                    'allowed_navigation_urls':['http://hybrid.fixture.local/finish']})
        obs=observe(1)
        for index,(kind,_,value) in enumerate(sequence):
            obs=replace(obs,causal_history=tuple(history))
            (root/f'observation-{index+1}.json').write_text(json.dumps(obs.to_dict(),indent=2)+'\n')
            decision=actor.predict_action(task,obs,rng=random.Random(42),
                context=HybridActionContext('fixture',obs.observation_id,tuple(actions)))
            parameters=DeterministicParameterProvider().resolve(task,obs,decision,rng=random.Random(42))
            action=ConcreteAction(action_id=f'fixture:a{index+1}',source_decision_id=decision.decision_id,
                action_type=decision.action_type,parameters=parameters.values,bbox=decision.bbox)
            assert action.action_type.value==kind
            expected=validate_control_action(kind,action.parameters,action.bbox,obs,
                                              control_id=action.parameters.get('target_control_id'))
            code=_action_code(action)
            state=request(op='execute',kind=kind,parameters=dict(action.parameters),bbox=action.bbox,
                          observation_id=obs.observation_id,expected_control=expected,code=code)
            if kind=='TYPE':assert state['text']==value
            if kind=='SELECT':assert state['selected']==value
            if kind=='CLICK':assert state['checked']
            if kind=='PRESS_KEY':assert state['link_focused']
            if kind=='SCROLL':assert state['scroll_y']>0
            if kind=='NAVIGATE':assert state['url']==value
            after=observe(index+2)
            result=ExecutionResult(action_id=action.action_id,status=ExecutionStatus.EXECUTED,
                executor_step=index+1,state_changed=after.screenshot_sha256!=obs.screenshot_sha256)
            executions.append({'action':action.to_dict(),'execution':result.to_dict(),'code':code,'observed_state':state})
            (root/'executions.json').write_text(json.dumps(executions,indent=2)+'\n')
            history.append(CausalHistoryEntry(history_index=index+1,action_id=action.action_id,
                action_type=action.action_type,action_target_fingerprint=action.fingerprint,
                execution_status=result.status,executor_step=index+1,state_changed=result.state_changed,
                environment_error=False,execution_error_sha256=None,post_observation_id=after.observation_id,
                post_observation_sha256=after.record_sha256,post_screenshot_sha256=after.screenshot_sha256))
            actions.append(action);obs=after
        assert ledger.count==12
    finally:
        deactivate_model_call_ledger(token)
        worker.stdin.close()
        try:worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.terminate();worker.wait(timeout=10)
    (root/'model-dispatches.json').write_text(json.dumps(ledger_events,indent=2)+'\n')
    (root/'result.json').write_text(json.dumps({'status':'PASS','evidence_type':'SCRIPTED_BROWSER_FIXTURE',
        'actual_model_inferences':0,'scripted_model_dispatches':ledger.count,'executed_actions':len(actions),
        'action_classes':[a.action_type.value for a in actions],'live_evaluation_episodes':0},indent=2)+'\n')
    print('PASS: all six scripted adapter actions executed; zero actual model inference')


if __name__=='__main__':main()
