"""One episode of raw SeeAct (system S) on the live-website suite.

Same browser worker, scorer, model worker, budget and record layout as the
Browser Use episodes in live.py. No P1 assessment, recovery or memory: this is
the unmodified SeeAct agent driven by the same frozen Qwen2.5-VL-7B actor.
One agent step = SeeAct's two model calls (reasoning, then element choice).
"""
import asyncio
import base64
import json
from pathlib import Path
import time

from web_agent.eval.task1.core import write_new
from web_agent.eval.task2.live import Budget, BudgetExhausted
from web_agent.eval.task2.seeact_common import action_type

SYSTEM = 'S'
MAX_CONTINUOUS_NO_OP = 5  # SeeActAgent default max_continuous_no_op


def image_part(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+data}}


def turn_messages(prompt, screenshot, output0=None):
    """SeeAct's OpenAIEngine message layout: system prompt, user text + marked screenshot, then (turn 1) its own answer and the referring prompt."""
    prompt0, prompt1, prompt2 = prompt
    messages = [{'role': 'system', 'content': prompt0},
                {'role': 'user', 'content': [{'type': 'text', 'text': prompt1}, image_part(screenshot)]}]
    if output0 is not None:
        messages += [{'role': 'assistant', 'content': '\n\n'+output0},
                     {'role': 'user', 'content': prompt2}]
    return messages


async def seeact_episode(*, block, root, browser_worker, model_worker, seeact_worker, settings):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    episode_id = f"{block['task']}-r{block['repeat_id']}-{SYSTEM}"
    write_new(root/'started.json', {'episode_id': episode_id, 'system': SYSTEM, 'block': block, 'settings': settings})
    reset = await asyncio.to_thread(browser_worker.call, 'reset', episode_id=episode_id, task=block['task'], seed=block['browser_reset_seed'])
    if reset['goal'] != block['goal']:
        raise RuntimeError('Frozen reset goal mismatch')
    await asyncio.to_thread(model_worker.call, 'reset')
    budget = Budget(**settings['budget'])
    counters = {'executed': 0, 'rejected': 0, 'no_action': 0, 'invalid_outputs': 0, 'limit_hits': 0,
                'blocked_navigations': 0, 'executed_recoveries': 0, 'memory_exposures': 0}
    stop = 'budget'; score = {'terminated': False, 'raw_reward': 0.0, 'invalid_url': False}; step = 0; steps_completed = 0
    attached = False
    try:
        attach = await asyncio.to_thread(seeact_worker.call, 'attach', cdp_url=reset['cdp_url'], goal=reset['goal'],
                                         start_url=reset['url'], viewport=reset['viewport'],
                                         allowed_domains=reset['allowed_domains'], root=str(root/'seeact'))
        attached = True
        write_new(root/'seeact-attach.json', attach)
        before = await asyncio.to_thread(browser_worker.call, 'observe', episode_id=episode_id)
        write_new(root/'observation-0000.json', before)
        while step < settings['max_agent_steps'] and budget.available(2) and budget.executor_requests < budget.max_executor_requests:
            prep = await asyncio.to_thread(seeact_worker.call, 'prepare')
            write_new(root/f'seeact-{step:04d}-prepare.json', prep)
            request = {'messages': turn_messages(prep['prompt'], prep['screenshot']), 'output_schema': None, 'advice': None, 'plain': True}
            try:
                budget.charge('seeact_turn0')
            except BudgetExhausted:
                # The 2700 s wall clock ran out during SeeAct's own page processing (e.g. its
                # element listing takes minutes on very large pages): a failed episode, not an error.
                stop = 'wall_clock'; break
            out0 = await asyncio.to_thread(model_worker.call, 'generate', request=request)
            write_new(root/f'actor-{step:04d}-turn0.json', out0)
            request = {'messages': turn_messages(prep['prompt'], prep['screenshot'], out0['raw_response']), 'output_schema': None, 'advice': None, 'plain': True}
            try:
                budget.charge('seeact_turn1')
            except BudgetExhausted:
                stop = 'wall_clock'; break
            out1 = await asyncio.to_thread(model_worker.call, 'generate', request=request)
            write_new(root/f'actor-{step:04d}-turn1.json', out1)
            counters['limit_hits'] += int(bool(out0.get('limit_hit'))) + int(bool(out1.get('limit_hit')))
            fin = await asyncio.to_thread(seeact_worker.call, 'finish', output0=out0['raw_response'], output1=out1['raw_response'])
            budget.executor_requests += 1
            score = await asyncio.to_thread(browser_worker.call, 'score')
            write_new(root/f'score-{step:04d}.json', score)
            after = await asyncio.to_thread(browser_worker.call, 'observe', episode_id=episode_id)
            write_new(root/f'observation-{step+1:04d}.json', after)
            kind = action_type(fin['action'])
            status = 'executed' if fin['executed'] else ('rejected' if fin['error'] else 'none')
            if fin['parsed']['action'] == 'None':
                counters['invalid_outputs'] += 1
            if fin['blocked_navigation']:
                counters['blocked_navigations'] += 1
            counters['executed' if status == 'executed' else 'rejected' if status == 'rejected' else 'no_action'] += 1
            event = {'step': step, 'actor_calls': 2, 'action_type': kind, 'seeact_action': fin['action'],
                     'target': fin['element_option'], 'element': {'tag': fin['element_tag'], 'description': fin['element_description']} if fin['element_option'] else None,
                     'value': fin['value'] or None, 'execution_status': status, 'error': fin['error'],
                     'history_line': fin['history_line'], 'blocked_navigation': fin['blocked_navigation'],
                     'state_changed': before['sha256'] != after['sha256'], 'url_before': before.get('url'), 'url_after': after.get('url')}
            write_new(root/f'action-{step:04d}.json', event)
            before = after
            steps_completed += 1; step += 1
            if score['terminated'] or score.get('invalid_url'):
                stop = 'environment_terminal'; break
            if fin['terminate']:
                stop = 'agent_done'; break
            if fin['continuous_no_op'] >= MAX_CONTINUOUS_NO_OP:
                stop = 'seeact_max_no_op'; break
        if stop == 'budget' and time.monotonic()-budget.started >= budget.seconds:
            stop = 'wall_clock'
        result = {'episode_id': episode_id, 'system': SYSTEM, 'block': block, 'status': 'complete',
                  'completion': bool(score['terminated'] and not score.get('invalid_url') and score['raw_reward'] == 1.0),
                  'terminal_score': score, 'stop_reason': stop, 'counters': counters,
                  'model_calls': budget.calls, 'executor_requests': budget.executor_requests, 'recovery_attempts': 0,
                  'elapsed_seconds': time.monotonic()-budget.started, 'agent_steps': steps_completed}
        write_new(root/'result.json', result)
        return result
    except Exception as exc:
        write_new(root/'infrastructure-error.json', {'error_type': type(exc).__name__, 'error': str(exc),
                  'model_calls': budget.calls, 'executor_requests': budget.executor_requests})
        raise
    finally:
        if attached:
            try:
                summary = await asyncio.to_thread(seeact_worker.call, 'close')
                write_new(root/'seeact-summary.json', summary)
            except Exception:
                pass
        await asyncio.to_thread(browser_worker.call, 'close')
