"""Serial BrowserGym worker; rewards stay outside policy observations."""
from pathlib import Path
import hashlib
import json
import sys
import traceback

from web_agent.benchmarks.miniwob_controls import CONTROL_JAVASCRIPT, bind_controls, validate_control_action
from web_agent.benchmarks.miniwob_support import raw_reward_outcome


def main():
    import gymnasium as gym
    import browsergym.miniwob  # noqa: F401
    from PIL import Image
    from types import SimpleNamespace
    root = Path(sys.argv[1]); env = None; terminal = False; events = []
    for line in sys.stdin:
        try:
            q=json.loads(line); op=q['op']
            if op == 'reset':
                if env is not None: env.close()
                env=gym.make('browsergym/miniwob.'+q['task'],headless=True,action_mapping=lambda a:a)
                obs,_=env.reset(seed=q['seed']%(2**32)); terminal=False
                r={'goal':obs['goal']}
            elif op == 'observe':
                obs=env.unwrapped._get_obs()
                path=root/q['name']; Image.fromarray(obs['screenshot']).save(path); path.chmod(0o444)
                h,w=obs['screenshot'].shape[:2]
                r=dict(screenshot_path=str(path),screenshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       width=w,height=h,url=env.unwrapped.page.url,title=env.unwrapped.page.title(),
                       control_snapshot=env.unwrapped.page.evaluate(CONTROL_JAVASCRIPT))
            elif op == 'execute':
                expected=q.get('expected_control'); rejection=None
                if expected is not None:
                    from web_agent.benchmarks.miniwob_controls import CONTROL_INTERFACE
                    controls=bind_controls(env.unwrapped.page.evaluate(CONTROL_JAVASCRIPT),q['observation_id'])
                    view=SimpleNamespace(observation_id=q['observation_id'],current_page_state={
                        'control_interface':CONTROL_INTERFACE,'visible_controls':controls})
                    if q.get('hybrid_interface_version') == 3:
                        view.current_page_state['hybrid_interface_version']=3
                    try:
                        actual=validate_control_action(q['action_type'],q['parameters'],q['bbox'],view,
                            control_id=q['parameters'].get('target_control_id'))
                        fields=('source_id','tag','input_type','role','target_bbox','name','text',
                                'accessible_names','candidate_options','disabled','readonly','contenteditable',
                                'hit_point')
                        if any(actual[k]!=expected[k] for k in fields):
                            raise ValueError('STALE_TARGET_BEFORE_EXECUTION')
                    except ValueError as exc: rejection=str(exc)
                if rejection:
                    r={'action_error':rejection,'rejected':True}
                else:
                    obs,reward,term,trunc,info=env.step(q['code'])
                    terminal=bool(term or trunc)
                    events.append(raw_reward_outcome(reward=reward,terminated=term,truncated=trunc,info=info))
                    # Persist immediately even if the coordinator later fails.
                    (root/'browser-outcomes.json').write_text(json.dumps(events,indent=2)+'\n')
                    r={'action_error':obs.get('last_action_error',''),'rejected':False}
            elif op == 'terminal': r={'terminate':terminal}
            elif op == 'close':
                if env is not None: env.close()
                env=None
                (root/'browser-outcomes.json').write_text(json.dumps(events,indent=2)+'\n')
                r={}
            else: raise ValueError('unknown browser operation')
            print(json.dumps({'ok':True,'value':r}),flush=True)
        except Exception as exc:
            print(json.dumps({'ok':False,'error':str(exc),'traceback':traceback.format_exc()}),flush=True)
    if env is not None: env.close()


if __name__ == '__main__': main()
