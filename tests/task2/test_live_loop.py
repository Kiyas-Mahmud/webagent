import asyncio
import json
import sys
from types import SimpleNamespace, ModuleType
import pytest
from web_agent.eval.task2 import live


def run_loop(tmp_path, monkeypatch, system, p_fail=None, **extra):
    views=ModuleType('browser_use.agent.views');views.AgentStepInfo=lambda **kw:kw
    monkeypatch.setitem(sys.modules,'browser_use.agent.views',views)
    issued=[]; seq=0; order=[]
    class Output:
        action=[SimpleNamespace(model_dump=lambda **kw:{'input':{'index':1,'text':'exact'}})]
        def model_dump(self,**kw):return {'action':[{'input':{'index':1,'text':'exact'}}]}
    result=SimpleNamespace(error=None,model_dump=lambda **kw:{'error':None})
    class Agent:
        def __init__(self,llm):self.llm=llm;self.state=SimpleNamespace(consecutive_failures=0,last_model_output=None,last_result=[])
        async def step(self,info):
            order.append(('capture',info['step_number']))
            await self.llm.ready()  # the real Browser Use step awaits this inside ainvoke
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
            if op=='facts':return {'url':'u','scroll':{'at_bottom':True},'goal_links':[],'search_boxes':[],'field':None}
            if op=='assess':
                t=kw['transition']; order.append(('assess',t['action_id']))
                out={k:t[k] for k in ('episode_id','action_id','phase','incident_id')}|{'signals':{'outcome_label':'FAILURE'}}
                if p_fail is not None and t['phase']=='interaction_assessment':
                    out['outcome_probabilities']={'SUCCESS':1-p_fail,'FAILURE':p_fail}
                return out
            if op=='memory':return {'examples':[],'candidates':[]}
    settings={'budget':{},'max_agent_steps':4,'recovery_attempts_per_episode':4,'recovery_attempts_per_incident':2,'excluded_memory_tasks':[],**extra}
    block={'task':'test','repeat_id':0,'browser_reset_seed':42,'goal':'goal'}
    r=asyncio.run(live.episode(system=system,block=block,root=tmp_path/'episode',browser_worker=Worker(),model_worker=Worker(),settings=settings))
    return r,issued,order


@pytest.mark.parametrize('system',['A','B','C'])
def test_native_loop_recovery_parse_failures_are_bounded_and_isolated(tmp_path, monkeypatch,system):
    r,issued,_=run_loop(tmp_path,monkeypatch,system)
    assert not r['completion'] and r['executor_requests']==1 and r['counters']['executed']==1
    assert r['recovery_attempts']==(0 if system=='A' else 2)
    if system=='A':assert issued==[None]*4 and r['model_calls']=={'actor':4}
    else:
        assert issued[0] is None and issued[1] and issued[2] and issued[3] is None
        assert r['model_calls']['assessment']==1
    assert ('memory_query' in r['model_calls'])==(system=='C')


@pytest.mark.parametrize('p_fail,triggered',[(.59,False),(.899,False),(.9,True),(.99,True)])
def test_only_confident_failure_opens_recovery(tmp_path, monkeypatch, p_fail, triggered):
    # Development-run regression: a correct TYPE scored P(FAILURE)=0.59 and the advice made the actor retype.
    r,issued,_=run_loop(tmp_path,monkeypatch,'B',p_fail,recovery_trigger_probability=.9)
    assert bool(issued[1])==triggered and r['recovery_attempts']==(2 if triggered else 0)
    assert r['counters'].get('uncertain_failures',0)==(0 if triggered else 1)
    assert (tmp_path/'episode'/'gate-0000.json').exists()!=triggered


def test_v3_note_after_next_page_capture(tmp_path, monkeypatch):
    # web-v2 regression: a synchronous assessment delayed the actor's next page capture.
    r,issued,order=run_loop(tmp_path,monkeypatch,'B',.99,recovery_trigger_probability=.9,recovery_mode='v3')
    assert order.index(('capture',1)) < order.index(('assess','a0'))
    note=issued[1]
    assert note['schema']=='task2.recovery_note.v3' and 'did not make progress' in note['text']
    assert 'bottom of the page' not in note['text']  # a TYPE, not a scroll
    for label in ('ACTION_MISMATCH','REPLAN','PERCEPTION','strategy'):
        assert label not in note['text']
    assert r['counters']['blocked_actions']==1
    assert (tmp_path/'episode'/'note-0000.json').exists()


def test_v3_off_keeps_legacy_advice(tmp_path, monkeypatch):
    r,issued,_=run_loop(tmp_path,monkeypatch,'B',.99,recovery_trigger_probability=.9)
    assert issued[1]['schema']=='task2.advisory.v2'


@pytest.mark.parametrize('proposed,xpath,same',[
    ({'action_type':'CLICK','target':'9','value':None},'html/body/input',True),
    ({'action_type':'CLICK','target':'9','value':None},'html/body/a',False),
    ({'action_type':'TYPE','target':'9','value':'abc '},'html/body/input',True),
    ({'action_type':'TYPE','target':'9','value':'other'},'html/body/input',False),
    ({'action_type':'SCROLL','target':None,'value':'{"down": true, "pages": 0.5}'},None,True),
    ({'action_type':'SCROLL','target':None,'value':'{"down": false, "pages": 1.0}'},None,False),
])
def test_same_action(proposed,xpath,same):
    blocked={'CLICK':{'action_type':'CLICK','target':'4','value':None,'xpath':'html/body/input'},
             'TYPE':{'action_type':'TYPE','target':'4','value':'abc','xpath':'html/body/input'},
             'SCROLL':{'action_type':'SCROLL','target':None,'value':'{"down": true, "pages": 1.0}','xpath':None}}[proposed['action_type']]
    assert live.same_action(blocked,proposed,xpath)==same


def test_search_query_comes_from_goal_text():
    assert live.search_query('Search arXiv for "Attention Is All You Need" and open it.','https://arxiv.org/')=='Attention Is All You Need'
    assert live.search_query('Open the EPA page about climate change.','https://www.epa.gov/')=='climate change'
    assert live.search_query('Open the IETF page about RFCs.','https://www.ietf.org/')=='RFCs'


def test_block_scope_ignores_query_string():
    assert live.page_path('https://arxiv.org/search/?query=x')==live.page_path('https://arxiv.org/search')
    assert live.page_path('https://arxiv.org/search')!=live.page_path('https://arxiv.org/')


def test_search_box_found_in_browser_use_elements():
    node=lambda tag,**a:SimpleNamespace(node_name=tag,attributes=a)
    elements={3:node('A',href='/x'),7:node('INPUT',type='text',placeholder='Search EPA.gov'),9:node('INPUT',type='search')}
    assert live.search_box(elements)==7
    assert live.search_box({1:node('INPUT',type='text',name='email')}) is None
