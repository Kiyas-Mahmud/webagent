"""Live real-website environment for the agent (same IPC contract as browser_worker).

Tasks come from configs/eval/task2/web_tasks_v1.json: sites taken from the
user's own dataset, rendered at the training screenshot size (1280x720).
Native Browser Use acts through CDP on this Chromium; this worker only resets,
observes and scores. The completion rule is never exposed to the agent.
"""
import hashlib
import json
from pathlib import Path
import socket
import sys
import traceback

CHROME = '/home/aiub/kiyas/table2-inputs/miniwob-browsers/chromium-1117/chrome-linux/chrome'
TASKS = Path(__file__).resolve().parents[4]/'configs/eval/task2/web_tasks_v1.json'
# Headless Chromium announces itself as "HeadlessChrome"; several dataset sites refuse that.
USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'


def clamp_rounding(snapshot, tolerance=1e-5):
    """The extractor rounds left/top and width/height separately to 6 decimals, so a
    control touching the page edge can end at 1.000001. Trim only that rounding
    overshoot; anything larger is left for bind_controls to reject."""
    for row in snapshot['controls']:
        x, y, w, h = row['target_bbox']
        if 1 < x+w <= 1+tolerance:w = round(1-x, 6)
        if 1 < y+h <= 1+tolerance:h = round(1-y, 6)
        row['target_bbox'] = [x, y, w, h]
    return snapshot


def screenshot(page, path, timeout=30000):
    """Viewport screenshot. Playwright waits for web fonts first; some sites (TED)
    never finish, so fall back to Chrome's own capture of the same viewport."""
    import base64
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    try:
        page.screenshot(path=str(path), timeout=timeout)
    except PlaywrightTimeout:
        session = page.context.new_cdp_session(page)
        try:
            data = session.send('Page.captureScreenshot', {'format': 'png'})['data']
        finally:
            session.detach()
        Path(path).write_bytes(base64.b64decode(data))


def main():
    from playwright.sync_api import sync_playwright
    from web_agent.benchmarks.miniwob_controls import CONTROL_JAVASCRIPT, bind_controls
    root = Path(sys.argv[1]).resolve(); root.mkdir(parents=True, exist_ok=True)
    suite = json.loads(TASKS.read_text())
    tasks = {t['id']: t for t in suite['tasks']}
    viewport = suite['viewport']
    playwright = sync_playwright().start()
    browser = context = task = episode_root = None
    sequence = 0

    def page():
        # Browser Use may open a new tab; the newest open page is the one it works on.
        return [p for p in context.pages if not p.is_closed()][-1]

    def close():
        nonlocal browser, context
        if browser is not None:
            browser.close(); browser = context = None

    for line in sys.stdin:
        try:
            request = json.loads(line); op = request['op']
            if op == 'reset':
                close()
                task = tasks[request['task']]
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
                browser = playwright.chromium.launch(executable_path=CHROME, headless=True,
                    args=[f'--remote-debugging-port={port}', '--remote-debugging-address=127.0.0.1'])
                context = browser.new_context(viewport=viewport, user_agent=USER_AGENT, locale='en-US')
                start = context.new_page()
                start.goto(task['start_url'], wait_until='load', timeout=60000)
                start.wait_for_timeout(1500)
                episode_root = root/request['episode_id']
                (episode_root/'images').mkdir(parents=True, exist_ok=False)
                sequence = 0
                value = {'goal': task['goal'], 'url': start.url, 'cdp_url': f'http://127.0.0.1:{port}',
                         'image_root': str(episode_root), 'viewport': viewport,
                         'allowed_domains': task['allowed_domains'], 'website_domain': task['dataset_domain'],
                         'task_id': 'web.'+task['id']}
            elif op == 'observe':
                current = page()
                current.wait_for_load_state('load', timeout=30000)
                p = episode_root/'images'/f'{sequence:04d}.png'
                screenshot(current, p); p.chmod(0o444)
                observation_id = f'o{sequence}'
                value = {'episode_id': request['episode_id'], 'observation_id': observation_id, 'sequence': sequence,
                         'image': str(p.relative_to(episode_root)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                         'controls': bind_controls(clamp_rounding(current.evaluate(CONTROL_JAVASCRIPT)), observation_id),
                         'url': current.url}
                sequence += 1
            elif op == 'score':
                # Independent scorer only: never exposed to actor or model inputs.
                url = page().url
                done = task['success']['url_contains'] in url.lower()
                value = {'terminated': done, 'raw_reward': 1.0 if done else 0.0, 'binary_reward': 1.0 if done else 0.0,
                         'invalid_url': False, 'url': url}
            elif op == 'close':
                close(); value = {}
            else:
                raise ValueError('Unknown web worker operation')
            print('TASK2:'+json.dumps({'ok': True, 'value': value}), flush=True)
        except Exception as exc:
            print('TASK2:'+json.dumps({'ok': False, 'error': str(exc), 'traceback': traceback.format_exc()}), flush=True)
    close(); playwright.stop()


if __name__ == '__main__':
    main()
