"""Scripted browser fixture for observable effects; no model inference."""
from pathlib import Path
import hashlib
import json
import sys
from playwright.sync_api import sync_playwright
from web_agent.benchmarks.miniwob_controls import CONTROL_JAVASCRIPT,CONTROL_INTERFACE,bind_controls
from web_agent.benchmarks.browsergym_webarena import _action_code
from web_agent.runtime.contracts import PolicyObservation,ConcreteAction,ActionType
from web_agent.runtime.observable_progress import observable_step_effect,register_novel_effect

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':800,'height':400})
    page.set_content('<label for="field">Field</label><input id="field"><label><input type="checkbox" id="choice">Choice</label><select id="select"><option>one</option><option>two</option></select>')
    index=0
    def observe():
        global index
        index+=1;oid=f'fixture:obs:{index}';data=page.screenshot();path=out/f'observation-{index}.png';path.write_bytes(data)
        return PolicyObservation(task_id='fixture',goal='Scripted engineering check',observation_id=oid,
            screenshot_sha256=hashlib.sha256(data).hexdigest(),screenshot_path=str(path),width=800,height=400,
            url=page.url,title='',current_page_state={'control_interface':CONTROL_INTERFACE,
                'visible_controls':bind_controls(page.evaluate(CONTROL_JAVASCRIPT),oid)})
    seen=set();records=[]
    for kind,key,extra,expected in [('CLICK','field',{'button':'left','click_count':1},False),
        ('TYPE','field',{'text':'Exact fixture text'},True),('TYPE','field',{'text':'Exact fixture text'},False),
        ('SELECT','select',{'option':'two','candidate_options':['one','two']},True),
        ('CLICK','choice',{'button':'left','click_count':1},True),('CLICK','choice',{'button':'left','click_count':1},False)]:
        before=observe();c=next(c for c in before.current_page_state['visible_controls'] if c['source_id']==key);x,y,w,h=c['target_bbox']
        a=ConcreteAction(action_id=f'fixture-action:{index}',source_decision_id='scripted',action_type=ActionType(kind),
            bbox=tuple(c['target_bbox']),parameters={'target_x':x+w/2,'target_y':y+h/2,'target_bbox':c['target_bbox'],**extra})
        exec(_action_code(a),{'page':page})
        record=register_novel_effect(observable_step_effect(a,before,observe()),seen)
        assert record['novel_effect']==expected,(kind,record)
        records.append(record)
    browser.close()
(out/'result.json').write_text(json.dumps({'status':'PASS','evidence_type':'SCRIPTED_BROWSER_FIXTURE','model_calls':0,'checks':records},indent=2)+'\n')
print('PASS: real browser text, dropdown, focus and checkbox-cycle effects')
