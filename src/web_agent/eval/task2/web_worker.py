"""Live real-website environment for the agent (same IPC contract as browser_worker).

Tasks come from configs/eval/task2/web_tasks_v2.json: sites taken from the
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
TASKS = Path(__file__).resolve().parents[4]/'configs/eval/task2/web_tasks_v2.json'
# Headless Chromium announces itself as "HeadlessChrome"; several dataset sites refuse that.
USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'


def page_identity(url):
    """Lower-cased host without 'www.' plus path without trailing slash; query and fragment ignored."""
    from urllib.parse import urlsplit
    parts = urlsplit((url or '').lower())
    host = (parts.hostname or '')
    host = host[4:] if host.startswith('www.') else host
    return host + parts.path.rstrip('/')


def task_complete(success, url):
    """web-tasks-v2 rules. 'page': exact page; 'page_prefix': that page or a sub-path (list pages);
    'site': any page on that host. The v1 'url_contains' rule is kept for reproducing v1 runs."""
    page = page_identity(url)
    if 'page' in success:
        return page == success['page']
    if 'page_prefix' in success:
        return page == success['page_prefix'] or page.startswith(success['page_prefix'] + '/')
    if 'site' in success:
        return page.split('/')[0] == success['site']
    if 'url_contains' in success:
        return success['url_contains'] in (url or '').lower()
    raise ValueError('Unknown success rule')


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


PAGE_FACTS_JAVASCRIPT = r"""
() => {
  const clean = s => (s || '').replace(/\s+/g, ' ').trim();
  const shown = e => { const r = e.getBoundingClientRect(), s = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  const root = document.scrollingElement || document.documentElement;
  const links = [...document.querySelectorAll('a[href]')].map(a => ({
    text: clean(a.innerText || a.textContent || a.getAttribute('aria-label') || a.getAttribute('title')).slice(0, 80),
    href: a.href, visible: shown(a)})).filter(l => l.href.startsWith('http'));
  const searches = [...document.querySelectorAll('input, textarea')].filter(e =>
      /search|query|^q$/i.test([e.type, e.name, e.id, e.placeholder, e.getAttribute('aria-label'), e.getAttribute('role')].join(' ')))
    .map(e => ({label: clean(e.placeholder || e.getAttribute('aria-label') || e.title || e.name).slice(0, 60), visible: shown(e)}));
  let active = document.activeElement;
  while (active && active.shadowRoot && active.shadowRoot.activeElement) active = active.shadowRoot.activeElement;
  let field = null;
  if (active && /^(INPUT|TEXTAREA)$/.test(active.tagName)) {
    const form = active.form || active.closest('form');
    const submit = form ? form.querySelector('button[type=submit], input[type=submit], button:not([type])') : null;
    field = {label: clean(active.placeholder || active.getAttribute('aria-label') || active.title || active.name).slice(0, 60),
             value: (active.value || '').slice(0, 120), in_form: !!form,
             submit: submit ? clean(submit.innerText || submit.value || submit.getAttribute('aria-label')).slice(0, 40) || 'submit' : null};
  }
  return {scroll_y: Math.round(window.scrollY), viewport_h: window.innerHeight, page_h: root.scrollHeight,
          at_top: window.scrollY <= 2, at_bottom: window.scrollY + window.innerHeight >= root.scrollHeight - 2,
          links, searches, field};
}
"""
STOPWORDS = {'the', 'a', 'an', 'of', 'and', 'or', 'to', 'on', 'in', 'for', 'that', 'which', 'with', 'from', 'by',
             'you', 'your', 'is', 'are', 'all', 'need', 'it', 'its', 'at', 'as', 'be',
             'open', 'page', 'pages', 'about', 'lists', 'list', 'its', 'his', 'her', 'their', 'this', 'site', 'website',
             'ways', 'way', 'find', 'look', 'up', 'go', 'use', 'using', 'checking', 'check', 'navigate', 'view'}


def stem(word):
    if len(word) > 5 and word.endswith('ing'):
        word = word[:-3]
        return word[:-1] if len(word) > 2 and word[-1] == word[-2] else word
    if len(word) > 4 and word.endswith('ed'):
        return word[:-2]
    if len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


def tokens(text):
    import re
    return {stem(w) for w in re.findall(r'[a-z0-9]+', str(text).lower())}


def goal_links(goal, links, current_url, limit=3):
    """Links whose text or URL path share goal words, weighted by rarity on this page.

    Uses only the goal text the agent already sees and the page's own links;
    never the task's completion rule.
    """
    import math
    from urllib.parse import urlsplit
    # The site's own name (epa, gov, debian...) is in almost every link: no signal.
    site = tokens(urlsplit(current_url).hostname or '')
    wanted = {w for w in tokens(goal) if len(w) > 2 and w not in STOPWORDS and w not in site}
    here = current_url.split('#')[0].rstrip('/')
    seen, rows = set(), []
    for link in links:
        href = link['href'].split('#')[0]
        if href.rstrip('/') == here or href in seen:
            continue
        seen.add(href)
        parts = urlsplit(href)
        rows.append(dict(link, href=href, words=tokens(link['text']) | tokens(parts.path)))
    if not rows or not wanted:
        return []
    frequency = {w: sum(w in r['words'] for r in rows) for w in wanted}
    weight = {w: math.log((1 + len(rows)) / (1 + n)) for w, n in frequency.items()}
    scored = []
    for r in rows:
        score = sum(weight[w] for w in wanted & r['words'])
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    return [{'text': r['text'], 'href': r['href'], 'visible': r['visible'], 'score': round(s, 3)} for s, r in scored[:limit]]


def page_facts(page, goal):
    raw = page.evaluate(PAGE_FACTS_JAVASCRIPT)
    return {'url': page.url, 'scroll': {k: raw[k] for k in ('scroll_y', 'viewport_h', 'page_h', 'at_top', 'at_bottom')},
            'goal_links': goal_links(goal, raw['links'], page.url),
            'search_boxes': [s for s in raw['searches'] if s['visible']][:2], 'field': raw['field']}


def main():
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
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
                notes = {}
                try:
                    current.wait_for_load_state('load', timeout=30000)
                except PlaywrightTimeout:
                    # Some pages never fire 'load' (Stanford, web-v2: 4/4 episodes lost); observe what rendered.
                    notes['load_timeout'] = True
                p = episode_root/'images'/f'{sequence:04d}.png'
                screenshot(current, p); p.chmod(0o444)
                observation_id = f'o{sequence}'
                try:
                    controls = bind_controls(clamp_rounding(current.evaluate(CONTROL_JAVASCRIPT)), observation_id)
                except ValueError as exc:
                    # The MiniWoB control contract is stricter than some real pages (EPA search, v3.3);
                    # controls only feed memory applicability, so record the problem and continue.
                    controls, notes['controls_error'] = [], str(exc)
                value = {'episode_id': request['episode_id'], 'observation_id': observation_id, 'sequence': sequence,
                         'image': str(p.relative_to(episode_root)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                         'controls': controls, 'url': current.url, **notes}
                sequence += 1
            elif op == 'score':
                # Independent scorer only: never exposed to actor or model inputs.
                url = page().url
                done = task_complete(task['success'], url)
                value = {'terminated': done, 'raw_reward': 1.0 if done else 0.0, 'binary_reward': 1.0 if done else 0.0,
                         'invalid_url': False, 'url': url}
            elif op == 'facts':
                # Recovery evidence from the live page; the goal text is the only task input.
                value = page_facts(page(), request['goal'])
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
