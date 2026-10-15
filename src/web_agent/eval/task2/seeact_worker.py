"""SeeAct (OSU NLP, ICML 2024) as a raw baseline agent on the live-website suite.

Runs in `.task2-assets/seeact-env` (package `seeact` 0.2.9.0 from the pinned
clone in `.task2-assets/external/SeeAct`). Same line-based IPC as web_worker.

SeeAct's decision logic is untouched: its prompts, viewport element listing,
set-of-mark boxes, answer parsing and Playwright execution are the package's own
code (`SeeActAgent.predict` / `execute`). Three integration changes only:
  1. the two model calls per step are moved out to the orchestrator, so the same
     local Qwen2.5-VL-7B serves this agent (`prepare` -> model -> model -> `finish`);
  2. the browser is attached over CDP to the headless Chromium that web_worker
     opened (the package hard-codes headless=False), so start page, viewport, user
     agent and scoring are identical to the Browser Use runs;
  3. robustness guards: Locator objects are stripped before `drawBoxes` (not
     JSON-serialisable); the package's execute() error path, which reads
     `pred_element['description']` when the element is None, is caught and logged
     to SeeAct's own history as it intends; after each action the worker lets a
     navigation start and finish (1 s + load, max 15 s) and retries the element
     listing when the page is still navigating, like Browser Use's post-action wait.
Leaving the task's allowed domains is handled as Browser Use does: the navigation
is undone (go back) and the agent is told in its history.
"""
import asyncio
import json
import logging
import os
from os.path import dirname
from pathlib import Path
import sys
import traceback

from web_agent.eval.task2.seeact_common import host_allowed


async def settle(page, pause_ms=1000, load_timeout_ms=15000):
    """Let a navigation triggered by the last action start and finish."""
    try:
        await page.wait_for_timeout(pause_ms)
        await page.wait_for_load_state('load', timeout=load_timeout_ms)
    except Exception:
        pass


async def screenshot(page, path, timeout=30000):
    """Same fallback as web_worker.screenshot: Playwright waits for web fonts first and
    some sites (TED) never finish, so capture the same viewport through Chrome directly."""
    import base64
    from playwright.async_api import TimeoutError as PlaywrightTimeout
    try:
        await page.screenshot(path=str(path), timeout=timeout)
        return False
    except PlaywrightTimeout:
        session = await page.context.new_cdp_session(page)
        try:
            data = (await session.send('Page.captureScreenshot', {'format': 'png'}))['data']
        finally:
            await session.detach()
        Path(path).write_bytes(base64.b64decode(data))
        return True


class LocalSeeAct:
    """Wraps an unmodified SeeActAgent; see module docstring for the three changes."""

    def __init__(self, *, goal, start_url, viewport, allowed_domains, root):
        import seeact
        from seeact.agent import SeeActAgent
        os.environ.setdefault('OPENAI_API_KEY', 'unused-local-model')  # engine_factory asserts a key; the engine is never called
        logging.getLogger('SeeActAgent').handlers.clear()  # one log file per episode
        self.agent = SeeActAgent(save_file_dir=str(root), default_task=goal, default_website=start_url,
                                 headless=True, viewport=viewport, model='gpt-4o', temperature=0)
        self.agent.engine = None  # any accidental in-package model call fails loudly
        self.mark_page_js = Path(dirname(seeact.__file__), 'mark_page.js').read_text()
        self.allowed_domains = allowed_domains
        self.blocked_navigations = 0
        self.pending = None
        self.last_elements = []
        self.playwright = self.browser = None

    async def attach(self, cdp_url):
        from playwright.async_api import async_playwright
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.connect_over_cdp(cdp_url)
        context = self.browser.contexts[0]
        page = [p for p in context.pages if not p.is_closed()][-1]
        agent = self.agent
        agent.session_control.update(browser=self.browser, context=context, active_page=page)
        context.on('page', agent.page_on_open_handler)
        await agent.page_on_open_handler(page)
        return {'url': page.url, 'pages': len(context.pages)}

    async def prepare(self):
        """SeeActAgent.predict() up to the first model call (text_choice_som strategy)."""
        from seeact.data_utils.format_prompt_utils import format_options, generate_option_name
        from seeact.demo_utils.browser_helper import get_interactive_elements_with_playwright
        from seeact.demo_utils.format_prompt import format_choices
        agent = self.agent
        agent.time_step += 1
        try:
            await agent.session_control['active_page'].wait_for_load_state('load')
        except Exception:
            pass
        for attempt in range(3):
            try:
                elements = await get_interactive_elements_with_playwright(agent.page, agent.config['browser']['viewport'])
                break
            except Exception as exc:
                if attempt == 2 or 'Execution context was destroyed' not in str(exc):
                    raise
                await settle(agent.page)
        elements = sorted(elements, key=lambda el: (el['center_point'][1], el['center_point'][0]))
        elements = [{**x, 'idx': i, 'option': generate_option_name(i)} for i, x in enumerate(elements)]
        mark_error = None
        try:
            await agent.page.evaluate(self.mark_page_js)
            await agent.page.evaluate('unmarkPage()')
            await agent.page.evaluate('elements => { return window.som.drawBoxes(elements); }',
                                      [{k: v for k, v in e.items() if k != 'selector'} for e in elements])
        except Exception as exc:
            mark_error = str(exc)
        choices = format_choices(elements)
        prompt = agent.generate_prompt(task=agent.tasks[-1], previous=agent.taken_actions, choices=choices)
        Path(agent.screenshot_path).parent.mkdir(parents=True, exist_ok=True)
        font_fallback = await screenshot(agent.page, screenshot_path := agent.screenshot_path)
        self.pending = {'elements': elements, 'choices': choices, 'prompt': prompt}
        self.last_elements = elements
        return {'step': agent.time_step, 'url': agent.page.url, 'screenshot': screenshot_path, 'screenshot_font_fallback': font_fallback,
                'elements': len(elements),
                'element_list': [{'option': e['option'], 'tag_with_role': e['tag_with_role'], 'tag': e['tag'],
                                  'description': e['description'], 'center_point': list(e['center_point'])} for e in elements],
                'question_description': agent.prompts['question_description'],
                'choices': choices, 'options_text': format_options(choices), 'prompt': prompt,
                'previous_actions': list(agent.taken_actions), 'mark_error': mark_error}

    def parse(self, output1):
        """SeeAct's grounding of a turn-1 answer, without executing (used by the repeat guard)."""
        from seeact.data_utils.format_prompt_utils import get_index_from_option_name
        from seeact.demo_utils.format_prompt import postprocess_action_lmm
        elements = (self.pending or {}).get('elements') or self.last_elements or []
        pred_element_label, pred_action, pred_value = postprocess_action_lmm(output1)
        element_id = get_index_from_option_name(pred_element_label) if len(pred_element_label) in (1, 2) else None
        pred_element = elements[element_id] if element_id is not None and element_id < len(elements) else None
        return pred_element_label, pred_action, pred_value, pred_element

    async def finish(self, output0, output1):
        """Rest of SeeActAgent.predict() (parse, ground) followed by execute()."""
        agent = self.agent
        if self.pending is None:
            raise RuntimeError('finish before prepare')
        pred_element_label, pred_action, pred_value, pred_element = self.parse(output1)
        self.pending = None
        prediction = {'action_generation': output0, 'action_grounding': output1, 'element': pred_element,
                      'action': pred_action, 'value': pred_value}
        agent.predictions.append(prediction)
        return await self.execute(pred_element, pred_action, pred_value, pred_element_label)

    async def act(self, action, value=None, element_option=None, label=None):
        """Execute a framework recovery option as a native SeeAct action on the last prepared page."""
        agent = self.agent
        self.pending = None
        element = next((e for e in (self.last_elements or []) if e['option'] == element_option), None) if element_option else None
        if element_option and element is None:
            raise RuntimeError('Recovery option element is no longer on the page')
        agent.predictions.append({'action_generation': None, 'action_grounding': 'recovery option: '+str(label),
                                  'element': element, 'action': action, 'value': value or 'None'})
        return await self.execute(element, action, value or 'None', element_option or 'Invalid')

    async def execute(self, pred_element, pred_action, pred_value, pred_element_label):
        """SeeActAgent.execute() with the three guards from the module docstring."""
        agent = self.agent
        try:
            await agent.page.evaluate('unmarkPage()')
        except Exception:
            pass
        url_before = agent.page.url
        executed = False; error = None; action = pred_action
        try:
            if (pred_action not in agent.no_element_op) and pred_element is None:
                action = 'NONE'
            new_action = await agent.perform_action(pred_element, action, pred_value, None, None)
            agent.taken_actions.append(new_action)
            if action != 'NONE':
                agent.valid_op += 1; agent.continuous_no_op = 0
            else:
                agent.continuous_no_op += 1
            executed = action != 'NONE'
        except Exception as exc:
            description = pred_element['description'] if pred_element else 'no element'  # guard 3
            new_action = f"Failed to perform {pred_action} on {description} with value '{pred_value}': {exc}"
            error = str(exc)
            agent.taken_actions.append(new_action)
            agent.continuous_no_op += 1
        blocked = None
        if action != 'NONE':
            await settle(agent.page)
        try:
            if not host_allowed(agent.page.url, self.allowed_domains):
                blocked = agent.page.url
                self.blocked_navigations += 1
                try:
                    await agent.page.go_back(timeout=15000)
                except Exception:
                    await agent.page.goto(url_before, timeout=30000)
                agent.taken_actions.append(f'Navigation to {blocked} was blocked: outside the allowed websites for this task')
        except Exception as exc:
            error = (error or '') + f' [domain check: {exc}]'
        return {'parsed': {'element': pred_element_label, 'action': pred_action, 'value': pred_value},
                'action': action, 'element_option': pred_element['option'] if pred_element else None,
                'element_description': pred_element['description'] if pred_element else None,
                'element_tag': pred_element['tag_with_role'] if pred_element else None,
                'value': pred_value, 'executed': executed, 'error': error, 'history_line': new_action,
                'blocked_navigation': blocked, 'url': agent.page.url, 'terminate': bool(agent.complete_flag),
                'continuous_no_op': agent.continuous_no_op, 'valid_op': agent.valid_op}

    async def close(self):
        summary = {'num_step': len(self.agent.taken_actions), 'action_history': list(self.agent.taken_actions),
                   'blocked_navigations': self.blocked_navigations}
        try:
            main = Path(self.agent.main_path)
            (main/'result.json').write_text(json.dumps({'task': self.agent.tasks, 'website': self.agent.config['basic']['default_website'],
                                                         **summary}, indent=2))
            (main/'all_predictions.json').write_text(json.dumps(self.agent.predictions, default=str, indent=2))
        except Exception:
            pass
        for handler in logging.getLogger('SeeActAgent').handlers:
            handler.close()
        try:
            if self.browser is not None:
                await self.browser.close()  # detaches from CDP; web_worker owns the Chromium process
        except Exception:
            pass
        if self.playwright is not None:
            await self.playwright.stop()
        return summary


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    agent = None
    for line in sys.stdin:
        try:
            request = json.loads(line); op = request['op']
            if op == 'attach':
                if agent is not None:
                    loop.run_until_complete(agent.close())
                agent = LocalSeeAct(goal=request['goal'], start_url=request['start_url'], viewport=request['viewport'],
                                    allowed_domains=request['allowed_domains'], root=request['root'])
                value = loop.run_until_complete(agent.attach(request['cdp_url']))
                value['seeact_dir'] = agent.agent.main_path
            elif op == 'prepare':
                value = loop.run_until_complete(agent.prepare())
            elif op == 'finish':
                value = loop.run_until_complete(agent.finish(request['output0'], request['output1']))
            elif op == 'parse':
                label, action, val, element = agent.parse(request['output1'])
                value = {'parsed': {'element': label, 'action': action, 'value': val},
                         'element_option': element['option'] if element else None,
                         'element_description': element['description'] if element else None,
                         'element_tag': element['tag_with_role'] if element else None}
            elif op == 'act':
                value = loop.run_until_complete(agent.act(request['action'], request.get('value'), request.get('element_option'), request.get('label')))
            elif op == 'close':
                value = loop.run_until_complete(agent.close()) if agent is not None else {}
                agent = None
            elif op == 'ping':
                from importlib.metadata import version
                value = {'seeact': version('seeact'), 'playwright': version('playwright')}
            else:
                raise ValueError('Unknown SeeAct worker operation')
            print('TASK2:'+json.dumps({'ok': True, 'value': value}, default=str), flush=True)
        except Exception as exc:
            print('TASK2:'+json.dumps({'ok': False, 'error': str(exc), 'traceback': traceback.format_exc()}), flush=True)
    if agent is not None:
        loop.run_until_complete(agent.close())
    loop.close()


if __name__ == '__main__':
    main()
