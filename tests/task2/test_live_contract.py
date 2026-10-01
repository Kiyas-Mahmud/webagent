import base64
import io
from pathlib import Path

from PIL import Image
import pytest

from web_agent.eval.task2.live import Budget, BudgetExhausted, recorded_action
from web_agent.eval.task2.native_model import strict_object
from web_agent.eval.task2.model_worker import native_messages


def test_total_budget_includes_memory_and_shadow_calls():
    b=Budget(max_model_calls=4)
    for kind in ('actor','assessment','memory_query','memory_actor'):b.charge(kind)
    with pytest.raises(BudgetExhausted):b.charge('actor')
    assert sum(b.calls.values())==4


def test_wall_clock_budget():
    b=Budget(seconds=0)
    assert not b.available()


@pytest.mark.parametrize('raw', ['```json\n{}\n```','{} {}','[]','{"x":1,"x":2}','{"x":NaN}'])
def test_no_response_repair(raw):
    with pytest.raises(ValueError):strict_object(raw)


def test_native_values_exact_and_no_action_replacement():
    assert recorded_action({'input':{'index':5,'text':' A$B\nC! '}})=={'action_type':'TYPE','target':'5','value':' A$B\nC! '}
    assert recorded_action({'select_dropdown':{'index':2,'text':'Two'}})['action_type']=='SELECT'
    assert recorded_action({'done':{'success':True,'text':'finished'}}) is None
    with pytest.raises(ValueError):recorded_action({'evaluate':{'code':'secret'}})
    with pytest.raises(ValueError):recorded_action({'click':{'index':1},'input':{'index':2,'text':'x'}})


def test_local_image_order_and_native_prompt_preservation():
    images=[]
    for color in ('red','blue'):
        buf=io.BytesIO();Image.new('RGB',(3,3),color).save(buf,format='PNG')
        images.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}})
    r={'messages':[{'role':'system','content':'Original system'},{'role':'user','content':[*images,{'type':'text','text':'Actual goal'}]}],
       'output_schema':{'type':'object'},'advice':None}
    chat,ims=native_messages(r)
    assert chat[0]['content'][0]['text']=='Original system'
    assert chat[1]['content'][-1]['text']=='Actual goal'
    assert [im.getpixel((0,0)) for im in ims]==[(255,0,0),(0,0,255)]
    assert len(chat)==3  # No empty advisory message silently changing B/C input.
    r['messages'][1]['content'][0]['image_url']['url']='https://example.com/image.png'
    with pytest.raises(ValueError,match='inline local'):native_messages(r)


def test_v3_note_and_guard_text_reach_actor_plainly():
    r={'messages':[{'role':'user','content':'goal'}],'output_schema':{'type':'object'},
       'advice':{'schema':'task2.recovery_note.v3','text':'FAILURE MONITOR: note'},
       'previous_proposal_rejection':{'stage':'recovery_guard','reason':"'click' was already tried"}}
    chat,_=native_messages(r)
    texts=[c['content'][0]['text'] for c in chat[-2:]]
    assert texts==["PROPOSAL BLOCKED, NOT EXECUTED: 'click' was already tried",'FAILURE MONITOR: note']


@pytest.mark.parametrize('replies,blocked_index,expected,overrides', [
    (['{"action":[{"click":{"index":4}}]}','{"action":[{"click":{"index":7}}]}'],4,7,0),
    (['{"action":[{"click":{"index":4}}]}']*3,4,4,1),
    (['{"action":[{"click":{"index":7}}]}'],4,7,0),
])
def test_guard_regenerates_but_never_substitutes(tmp_path, monkeypatch, replies, blocked_index, expected, overrides):
    import asyncio, json, sys
    from types import ModuleType, SimpleNamespace
    from web_agent.eval.task2.native_model import LocalInternVL
    views=ModuleType('browser_use.llm.views');views.ChatInvokeCompletion=lambda **kw:kw
    monkeypatch.setitem(sys.modules,'browser_use.llm.views',views)
    class Output:
        def __init__(self,value):self.value=value;self.action=[SimpleNamespace(model_dump=lambda **kw:value['action'][0])]
        def model_dump(self,**kw):return self.value
    class Format:
        model_json_schema=staticmethod(lambda:{'type':'object'})
        model_validate=staticmethod(lambda value,strict:Output(value))
    sent=[]
    class Worker:
        def call(self,op,request):
            sent.append(request);return {'raw_response':replies[len(sent)-1],'limit_hit':False}
    llm=LocalInternVL(Worker(),tmp_path,Budget(max_model_calls=10))
    async def guard(completion):
        return 'blocked' if completion.action[0].model_dump()['click']['index']==blocked_index else None
    llm.guard=guard
    out=asyncio.run(llm.ainvoke([],Format))['completion']
    assert out.value['action'][0]['click']['index']==expected
    assert llm.guard_overrides==overrides and len(sent)==len(replies)
    assert all(s['previous_proposal_rejection']['stage']=='recovery_guard' for s in sent[1:])


@pytest.mark.parametrize('reply,expected_index,generations', [('2',7,1), ('0',None,2), ('I pick option 9',None,2)])
def test_option_choice_runs_through_browser_use_output(tmp_path, monkeypatch, reply, expected_index, generations):
    import asyncio, json, sys
    from types import ModuleType, SimpleNamespace
    from web_agent.eval.task2.native_model import LocalInternVL
    views=ModuleType('browser_use.llm.views');views.ChatInvokeCompletion=lambda **kw:kw
    monkeypatch.setitem(sys.modules,'browser_use.llm.views',views)
    class Output:
        def __init__(self,value):self.value=value;self.action=[SimpleNamespace(model_dump=lambda **kw:value['action'][0])]
        def model_dump(self,**kw):return self.value
    class Format:
        model_json_schema=staticmethod(lambda:{'type':'object'})
        model_validate=staticmethod(lambda value,strict:Output(value))
    sent=[]
    class Worker:
        def call(self,op,request):
            sent.append(request)
            return {'raw_response':reply if request.get('plain') else '{"action":[{"scroll":{"down":true}}]}','limit_hit':False}
    llm=LocalInternVL(Worker(),tmp_path,Budget(max_model_calls=10))
    options=[{'label':'Open the link','action':{'navigate':{'url':'https://x.org/a','new_tab':False}}},
             {'label':'Search','action':{'click':{'index':7}},'then':{'label':'Enter','action':{'send_keys':{'keys':'Enter'}}}}]
    async def provider(advice,followup):return ('Task: t\nFAILURE MONITOR: note',options)
    llm.options_provider=provider
    out=asyncio.run(llm.ainvoke([],Format))['completion']
    assert len(sent)==generations and sent[0]['plain'] and 'Reply with only the number' in sent[0]['messages'][0]['content'][-1]['text']
    if expected_index:
        assert out.value['action']==[{'click':{'index':7}}] and llm.followup['label']=='Enter' and llm.option_choices==1
    else:
        assert out.value['action']==[{'scroll':{'down':True}}] and llm.followup is None
