"""Browser engineering fixtures, never model/pilot completion evidence."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import sys
import threading

from playwright.sync_api import sync_playwright
from web_agent.benchmarks.miniwob_controls import CONTROL_JAVASCRIPT, bind_controls, supported_actions
from web_agent.benchmarks.browsergym_webarena import _action_code
from web_agent.runtime.contracts import ConcreteAction, ActionType


def main():
    root=Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=False)
    (root/'next.html').write_text('<p>Navigation fixture</p>')
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(root)))
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1000,'height':600})
        page.set_content('''<button id="button" onclick="this.textContent='Clicked'">Press</button>
          <label for="name">Person</label><input id="name">
          <p><label>Verification</label><input id="verify" type="password" value="hidden-secret"></p>
          <label><input type="checkbox" id="check">Choice A</label>
          <label><input type="radio" name="radio" id="radio">Choice B</label>
          <input aria-label="Read only" readonly><input placeholder="Hint" disabled>
          <input id="aria" aria-labelledby="caption"><span id="caption">Accessible caption</span>
          <label for="d1">Duplicate</label><input id="d1"><label for="d2">Duplicate</label><input id="d2">
          <span role="link" style="cursor:pointer;text-decoration:underline" id="custom">Custom</span>
          <select id="select"><option value="alpha">Alpha</option><option value="beta">Beta</option></select>
          <a id="nav">Next</a><div style="height:2000px"></div>''')
        page.locator('#nav').evaluate('(e,url)=>e.href=url',f'http://127.0.0.1:{server.server_port}/next.html')
        controls=bind_controls(page.evaluate(CONTROL_JAVASCRIPT),'fixture:obs:1')
        by={c['source_id']:c for c in controls}
        names=lambda c:{x['value'] for x in c['accessible_names']}
        assert 'Person' in names(by['name'])
        assert 'Verification' in names(by['verify'])
        assert 'Choice A' in names(by['check']) and 'Choice B' in names(by['radio'])
        assert 'Accessible caption' in names(by['aria'])
        assert by['verify']['value'] is None and 'hidden-secret' not in json.dumps(controls)
        assert by['check']['value'] is None and by['check']['text']==''
        assert 'CLICK' in supported_actions(by['custom'])
        assert all('TYPE' not in supported_actions(c) for c in controls if c['readonly'] or c['disabled'])
        (root/'controls.json').write_text(json.dumps(controls,indent=2)+'\n')
        executed=[]
        def run(kind,params,bbox=None):
            action=ConcreteAction(action_id='fixture-'+kind,source_decision_id='scripted-fixture',action_type=ActionType(kind),parameters=params,bbox=bbox)
            exec(_action_code(action),{'page':page})
            executed.append(kind)
        def target(key):
            b=by[key]['target_bbox'];x,y,w,h=b
            return {'target_x':x+w/2,'target_y':y+h/2,'target_bbox':b},tuple(b)
        values,box=target('button');run('CLICK',{**values,'button':'left','click_count':1},box)
        assert page.locator('#button').inner_text()=='Clicked'
        values,box=target('name');run('TYPE',{**values,'text':'Exact fixture text'},box)
        assert page.locator('#name').input_value()=='Exact fixture text'
        values,box=target('select');run('SELECT',{**values,'option':'beta','candidate_options':['alpha','beta']},box)
        assert page.locator('#select').input_value()=='beta'
        before=page.evaluate('document.activeElement.id');run('PRESS_KEY',{'key':'TAB'})
        assert page.evaluate('document.activeElement.id')!=before
        run('SCROLL',{'direction':'down','amount':.5,'container':'viewport'});page.wait_for_timeout(100)
        assert page.evaluate('scrollY')>0
        run('NAVIGATE',{'url':by['nav']['destination']})
        assert page.url.endswith('/next.html')
        browser.close()
    server.shutdown();server.server_close()
    (root/'result.json').write_text(json.dumps({'status':'PASS','evidence_type':'BROWSER_ENGINEERING_FIXTURES',
        'model_episodes':0,'executed_action_types':executed,'controls':len(controls)},indent=2)+'\n')
    print('PASS: six scripted executor actions and observable label/state fixtures')


if __name__=='__main__': main()
