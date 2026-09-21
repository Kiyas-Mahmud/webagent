"""BrowserGym owns reset/scoring; native Browser Use executes through local CDP."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback
import functools
import http.server
import threading


def main():
    import gymnasium as gym
    import browsergym.miniwob
    from web_agent.benchmarks.miniwob_controls import CONTROL_JAVASCRIPT, bind_controls
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Serve unchanged benchmark files; Browser Use's domain policy rejects file URLs.
    directory='/home/aiub/kiyas/table2-inputs/miniwob-plusplus/miniwob/html'
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(http.server.SimpleHTTPRequestHandler,directory=directory))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    os.environ['MINIWOB_URL']=f'http://127.0.0.1:{server.server_address[1]}/miniwob/'
    env = None
    sequence = 0
    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request['op']
            if op == 'reset':
                if env is not None:
                    env.close()
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                wrapper = root / 'chromium-with-cdp'
                chrome = '/home/aiub/kiyas/table2-inputs/miniwob-browsers/chromium-1117/chrome-linux/chrome'
                wrapper.write_text('#!' + sys.executable + '\nimport os,sys\nos.execv(' + repr(chrome) + ', [' + repr(chrome) + ', ' + repr('--remote-debugging-port='+str(port)) + ', "--remote-debugging-address=127.0.0.1"] + sys.argv[1:])\n')
                wrapper.chmod(0o700)
                episode_root = root / request['episode_id']
                (episode_root / 'images').mkdir(parents=True, exist_ok=False)
                env = gym.make('browsergym/miniwob.'+request['task'], headless=True,
                               action_mapping=lambda a:a, pw_chromium_kwargs={'executable_path':str(wrapper)})
                obs, _ = env.reset(seed=request['seed'])
                sequence = 0
                value = {'goal':obs['goal'], 'url':env.unwrapped.page.url, 'cdp_url':f'http://127.0.0.1:{port}',
                         'image_root':str(episode_root),'viewport':env.unwrapped.page.viewport_size}
            elif op == 'observe':
                p = episode_root / 'images' / f'{sequence:04d}.png'
                env.unwrapped.page.screenshot(path=str(p))
                p.chmod(0o444)
                observation_id = f'o{sequence}'
                value = {'episode_id':request['episode_id'], 'observation_id':observation_id,
                         'sequence':sequence, 'image':str(p.relative_to(episode_root)),
                         'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                         'controls':bind_controls(env.unwrapped.page.evaluate(CONTROL_JAVASCRIPT), observation_id)}
                sequence += 1
            elif op == 'score':
                # Independent scorer only: never expose raw reward/reason to actor/model IPC.
                reward, done, _, info = env.unwrapped.task.validate(env.unwrapped.page, [])
                value = {'terminated':bool(done), 'raw_reward':float(info.get('RAW_REWARD_GLOBAL', 0)),
                         'binary_reward':float(reward), 'invalid_url':bool(info.get('error'))}
            elif op == 'fixture':
                # Engineering only; prohibited in the model-run coordinator.
                env.unwrapped.page.set_content(request['html'])
                value = {}
            elif op == 'fixture_state':
                value=env.unwrapped.page.evaluate('''() => ({url:location.href, clicked:document.body.dataset.clicked || null,
                    value:document.querySelector('#field')?.value || null, selected:document.querySelector('#choice')?.value || null,
                    key:document.body.dataset.key || null, scrollY:window.scrollY})''')
            elif op == 'close':
                if env is not None:
                    env.close()
                    env = None
                value = {}
            else:
                raise ValueError('Unknown browser worker operation')
            print('TASK2:' + json.dumps({'ok':True, 'value':value}), flush=True)
        except Exception as exc:
            print('TASK2:' + json.dumps({'ok':False, 'error':str(exc), 'traceback':traceback.format_exc()}), flush=True)
    if env is not None:
        env.close()
    server.shutdown();server.server_close()


if __name__ == '__main__':
    main()
