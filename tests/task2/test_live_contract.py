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
