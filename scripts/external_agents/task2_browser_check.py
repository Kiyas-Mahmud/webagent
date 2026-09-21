"""Native Browser Use / BrowserGym connection check; no model calls."""
import asyncio
import json
import os
from pathlib import Path
import sys
import traceback

os.environ['ANONYMIZED_TELEMETRY']='false'
os.environ['BROWSER_USE_CONFIG_DIR']=str(Path('.task2-assets/browser-use-config').resolve())
os.environ['BROWSER_USE_LOGGING_LEVEL']='error'

from web_agent.eval.task1.core import write_new
from web_agent.eval.task2.ipc import Worker
from web_agent.eval.task2.live import create_agent, Budget
from web_agent.eval.task2.native_model import LocalInternVL


async def main():
    root=Path(sys.argv[1]).resolve();root.mkdir(parents=True,exist_ok=False)
    worker=Worker('/home/aiub/kiyas/table2-envs/miniwob-feasibility/bin/python',
        'web_agent.eval.task2.browser_worker',[root/'browser'],root/'browser.log',
        env={'MINIWOB_URL':Path('/home/aiub/kiyas/table2-inputs/miniwob-plusplus/miniwob/html/miniwob').as_uri()+'/',
             'PLAYWRIGHT_BROWSERS_PATH':'/home/aiub/kiyas/table2-inputs/miniwob-browsers'})
    browser=None
    try:
        reset=await asyncio.to_thread(worker.call,'reset',episode_id='fixture',task='click-button',seed=42)
        actor=LocalInternVL(None,root,Budget())
        agent,browser=await create_agent(reset,actor,root)
        state=await browser.get_browser_state_summary(include_screenshot=True)
        write_new(root/'native-state.json',{'url':state.url,'screenshot_present':bool(state.screenshot),
            'controls':{str(k):{'tag':v.tag_name,'attributes':v.attributes} for k,v in state.dom_state.selector_map.items()},
            'actions':list(agent.tools.registry.registry.actions)})
        assert state.url==reset['url'] and state.screenshot and state.dom_state.selector_map
        html='''<html><body onkeydown="this.dataset.key=event.key"><button id="button" onclick="document.body.dataset.clicked='yes'">Check</button>
        <input id="field" aria-label="Field"><select id="choice" aria-label="Choice"><option value="one">One</option><option value="two">Two</option></select>
        <div style="height:2000px">Scroll fixture</div></body></html>'''
        await asyncio.to_thread(worker.call,'fixture',html=html)
        state=await browser.get_browser_state_summary(include_screenshot=True,cached=False)
        ids={v.attributes.get('id'):k for k,v in state.dom_state.selector_map.items() if v.attributes.get('id')}
        checks=[('CLICK',{'click':{'index':ids['button']}}),
                ('TYPE',{'input':{'index':ids['field'],'text':'Exact Value 42!','clear':True}}),
                ('SELECT',{'select_dropdown':{'index':ids['choice'],'text':'Two'}}),
                ('PRESS_KEY',{'send_keys':{'keys':'Tab'}}),
                ('SCROLL',{'scroll':{'down':True,'pages':1.0}}),
                ('NAVIGATE',{'navigate':{'url':reset['url'],'new_tab':False}})]
        outcomes=[]
        for action_type,raw in checks:
            before=await asyncio.to_thread(worker.call,'observe',episode_id='fixture')
            if action_type in ('CLICK','TYPE','SELECT'):
                state=await browser.get_browser_state_summary(include_screenshot=True,cached=False)
                ids={v.attributes.get('id'):k for k,v in state.dom_state.selector_map.items() if v.attributes.get('id')}
                name=next(iter(raw));raw[name]['index']=ids[{'CLICK':'button','TYPE':'field','SELECT':'choice'}[action_type]]
            actions=[agent.ActionModel.model_validate(raw)]
            results=await agent.multi_act(actions)
            assert results and not any(r.error for r in results),[r.model_dump() for r in results]
            observed=await asyncio.to_thread(worker.call,'fixture_state')
            after=await asyncio.to_thread(worker.call,'observe',episode_id='fixture')
            expected={'CLICK':observed['clicked']=='yes','TYPE':observed['value']=='Exact Value 42!',
                      'SELECT':observed['selected']=='two','PRESS_KEY':observed['key']=='Tab',
                      'SCROLL':observed['scrollY']>0,'NAVIGATE':observed['url']==reset['url']}[action_type]
            assert expected,(action_type,observed)
            outcomes.append({'action_type':action_type,'native_action':raw,'observed':observed,
                             'before':before,'after':after,'image_root':reset['image_root']})
        write_new(root/'result.json',{'status':'PASS','scope':'scripted native six-action engineering fixture',
                   'actions':outcomes,'model_calls':0,'live_model_episodes':0})
        print('PASS native Browser Use / BrowserGym connection and six actions',flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if browser is not None:await browser.stop()
        worker.close()


asyncio.run(main())
