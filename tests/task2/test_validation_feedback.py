import asyncio
from pathlib import Path
import json
import sys
from types import ModuleType, SimpleNamespace
import pytest
from web_agent.eval.task2.native_model import LocalInternVL
from web_agent.eval.task2.model_worker import native_messages
from web_agent.eval.task2.live import Budget


def test_precise_rejection_delivered_once_and_cleared_after_valid_output(tmp_path,monkeypatch):
    views=ModuleType('browser_use.llm.views');views.ChatInvokeCompletion=lambda **kw:SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules,'browser_use.llm.views',views)
    class Output:
        @staticmethod
        def model_json_schema():return {'type':'object'}
        @staticmethod
        def model_validate(value,strict=True):return value
    class Worker:
        def __init__(self):self.requests=[]
        def call(self,op,request):
            self.requests.append(request)
            actions=[{'input':{'index':1,'text':'exact'}}]
            if len(self.requests)==1:actions=[]
            return {'raw_response':json.dumps({'action':actions}),'limit_hit':False}
    worker=Worker();actor=LocalInternVL(worker,tmp_path,Budget())
    async def scenario():
        with pytest.raises(ValueError,match='nonempty'):await actor.ainvoke([],Output)
        await actor.ainvoke([],Output)
        await actor.ainvoke([],Output)
    asyncio.run(scenario())
    first,second,third=worker.requests
    assert first['previous_proposal_rejection'] is None
    feedback=second['previous_proposal_rejection']
    assert feedback['stage']=='structured_output_validation'
    assert feedback['reason']=='A nonempty native action list is required'
    assert len(json.loads(feedback['rejected_response'])['action'])==0
    chat,_=native_messages(second)
    assert feedback['reason'] in chat[-1]['content'][0]['text']
    assert third['previous_proposal_rejection'] is None
    assert actor.budget.calls=={'actor':3}
