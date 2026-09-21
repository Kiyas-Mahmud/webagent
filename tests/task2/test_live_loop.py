import asyncio
import json
import sys
from types import SimpleNamespace, ModuleType
import pytest
from web_agent.eval.task2 import live


@pytest.mark.parametrize('system',['A','B','C'])
def test_native_loop_recovery_parse_failures_are_bounded_and_isolated(tmp_path, monkeypatch,system):
    views=ModuleType('browser_use.agent.views');views.AgentStepInfo=lambda **kw:kw
    monkeypatch.setitem(sys.modules,'browser_use.agent.views',views)
    issued=[]; seq=0
    class Output:
        action=[SimpleNamespace(model_dump=lambda **kw:{'input':{'index':1,'text':'exact'}})]
        def model_dump(self,**kw):return {'action':[{'input':{'index':1,'text':'exact'}}]}
    result=SimpleNamespace(error=None,model_dump=lambda **kw:{'error':None})
    class Agent:
        def __init__(self,llm):self.llm=llm;self.state=SimpleNamespace(consecutive_failures=0,last_model_output=None,last_result=[])
        async def step(self,info):
            issued.append(self.llm.advice)
            self.llm.budget.charge('actor');self.llm.calls+=1
            self.state.last_model_output=Output() if info['step_number']==0 else None
            self.state.last_result=[result] if info['step_number']==0 else []
            if info['step_number']==0:
                (tmp_path/'episode'/f'actor-{self.llm.calls:04d}-parsed.json').write_text(json.dumps({'native_proposal':Output().model_dump()}))
    class Browser:
        async def stop(self):pass
    async def create(reset,llm,root):return Agent(llm),Browser()
    monkeypatch.setattr(live,'create_agent',create)
    class Worker:
        def call(self,op,**kw):
            nonlocal seq
            if op=='reset':return {'goal':'goal','image_root':str(tmp_path)}
            if op=='observe':
                seq+=1
                return {'episode_id':kw['episode_id'],'observation_id':f'o{seq}','sequence':seq,'image':'images/x.png','sha256':'a'*64,'controls':[]}
            if op=='score':return {'terminated':False,'invalid_url':False,'raw_reward':0}
            if op=='assess':
                t=kw['transition']
                return {k:t[k] for k in ('episode_id','action_id','phase','incident_id')}|{'signals':{'outcome_label':'FAILURE'}}
            if op=='memory':return {'examples':[],'candidates':[]}
    settings={'budget':{},'max_agent_steps':4,'recovery_attempts_per_episode':4,'recovery_attempts_per_incident':2,'excluded_memory_tasks':[]}
    block={'task':'test','repeat_id':0,'browser_reset_seed':42,'goal':'goal'}
    r=asyncio.run(live.episode(system=system,block=block,root=tmp_path/'episode',browser_worker=Worker(),model_worker=Worker(),settings=settings))
    assert not r['completion'] and r['executor_requests']==1 and r['counters']['executed']==1
    assert r['recovery_attempts']==(0 if system=='A' else 2)
    if system=='A':assert issued==[None]*4 and r['model_calls']=={'actor':4}
    else:
        assert issued[0] is None and issued[1] and issued[2] and issued[3] is None
        assert r['model_calls']['assessment']==1
    assert ('memory_query' in r['model_calls'])==(system=='C')
