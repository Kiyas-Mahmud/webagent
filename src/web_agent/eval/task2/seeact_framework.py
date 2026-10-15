"""SeeAct + the framework (system T): P1 failure detection, adaptive recovery v3.3, P4 memory.

The framework is the one used on Browser Use (live.py, system C): the same
trained heads, 0.9 trigger gate, plain-text recovery note, page facts,
numbered recovery options, repeat guard and experience memory, called through
the same modules. Only the agent adapter differs:
  * element identity for the guard is (tag_with_role, description) instead of an xpath;
  * the note is appended after SeeAct's "Previous Actions" block in its turn-0 prompt;
  * a chosen option is executed as a native SeeAct action (GOTO / TYPE / PRESS ENTER /
    PRESS HOME / GO BACK) and replaces that step's two actor turns;
  * the guard re-asks turn 1 (grounding) only, keeping SeeAct's turn-0 reasoning.
SeeAct's own code is unchanged; live.py is untouched (frozen record of web-v3).
"""
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import re
import time

from web_agent.eval.task1.core import write_new
from web_agent.eval.task2.assessment import ExecutedTransition, describe_action, recovery_note
from web_agent.eval.task2.live import Budget, BudgetExhausted, observation, page_path, same_action, search_query
from web_agent.eval.task2.seeact_common import action_type
from web_agent.eval.task2.seeact_live import MAX_CONTINUOUS_NO_OP, image_part, turn_messages

SYSTEM = 'T'
UP_ACTIONS = {'SCROLL UP', 'PRESS HOME', 'PRESS PAGEUP'}
ASSESSED = {'CLICK', 'TYPE', 'SELECT', 'SCROLL', 'NAVIGATE', 'PRESS_KEY'}  # HOVER / tabs / NONE: logged only (D5)


def identity(tag_with_role, description):
    """Stable element identity across steps (option letters change, descriptions do not)."""
    if not tag_with_role and not description:
        return None
    return f"{(tag_with_role or '').strip()}|{re.sub(r'\s+', ' ', (description or '').strip()).lower()[:200]}"


def element_summary(tag_with_role, description):
    """The note's element dict, built from SeeAct's element data (same keys as the Browser Use summary)."""
    if not tag_with_role and not description:
        return None
    tag = (tag_with_role or '').split()[0].lower() if tag_with_role else ''
    return {'tag': tag, 'text': (description or '').strip()[:120], 'attributes': {}}


def guard_value(seeact_action, value):
    """Value in the conventions same_action / describe_action expect."""
    kind = action_type(seeact_action)
    if kind == 'SCROLL':
        return json.dumps({'down': seeact_action not in UP_ACTIONS}, sort_keys=True)
    if kind == 'PRESS_KEY':
        return 'Enter'
    if seeact_action == 'GO BACK':
        return None
    if kind in ('TYPE', 'SELECT', 'NAVIGATE'):
        return value if value not in (None, 'None') else None
    return None


def recorded(seeact_action, value, tag_with_role, description):
    """Action record in the web-v3 vocabulary: used for P1, the guard and the action event."""
    return {'action_type': action_type(seeact_action), 'target': identity(tag_with_role, description),
            'value': guard_value(seeact_action, value)}


def search_box(element_list):
    """Option letter of a visible site-search input in SeeAct's own element list (analogue of live.search_box)."""
    for e in element_list or []:
        if (e.get('tag') or '').lower() not in ('input', 'textarea'):
            continue
        text = (e.get('tag_with_role') or '') + ' ' + (e.get('description') or '')
        if 'type="search"' in text or re.search(r'search|query|(^| )q( |$)', text, re.I):
            return e['option']
    return None


def insert_note(query_prompt, question_description, note_text):
    """D3: the note goes after SeeAct's 'Previous Actions' block, before its question description."""
    if question_description and question_description in query_prompt:
        head, tail = query_prompt.split(question_description, 1)
        return head + note_text + '\n\n' + question_description + tail
    return query_prompt + '\n\n' + note_text


def choice_text(context, options):
    lines = '\n'.join(f'{k}. {o["label"]}' for k, o in enumerate(options, 1))
    return (context + '\n\nRecovery options for the next action:\n' + lines +
            '\n0. None of these; I will choose my own action.\nReply with only the number of the best option.')


async def seeact_framework_episode(*, block, root, browser_worker, model_worker, seeact_worker, settings):
    if settings.get('recovery_mode') != 'v3' or settings.get('memory_mode') != 'experience':
        raise ValueError('System T requires the v3 recovery settings with experience memory (as system C)')
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    episode_id = f"{block['task']}-r{block['repeat_id']}-{SYSTEM}"
    write_new(root/'started.json', {'episode_id': episode_id, 'system': SYSTEM, 'block': block, 'settings': settings})
    reset = await asyncio.to_thread(browser_worker.call, 'reset', episode_id=episode_id, task=block['task'], seed=block['browser_reset_seed'])
    if reset['goal'] != block['goal']:
        raise RuntimeError('Frozen reset goal mismatch')
    await asyncio.to_thread(model_worker.call, 'reset')
    budget = Budget(**settings['budget'])
    gate = settings.get('recovery_trigger_probability'); regenerations = settings.get('guard_regenerations', 2)
    counters = {'executed': 0, 'rejected': 0, 'no_action': 0, 'invalid_outputs': 0, 'limit_hits': 0, 'blocked_navigations': 0,
                'executed_recoveries': 0, 'assessments': 0, 'memory_queries': 0, 'memory_exposures': 0, 'memory_writes': 0,
                'blocked_actions': 0, 'guard_blocks': 0, 'guard_overrides': 0, 'option_choices': 0, 'uncertain_failures': 0}
    stop = 'budget'; score = {'terminated': False, 'raw_reward': 0.0, 'invalid_url': False}; step = 0; steps_completed = 0
    calls = 0  # actor-side call counter for file prefixes (actor-NNNN-*), as on Browser Use
    blocked = []; succeeded = []
    incident = None; incident_attempts = 0; recovery_total = 0; next_is_recovery = False
    pending_memory = None; advice = None; followup = None; assessment_task = None
    attached = False

    async def close_incident(resolved):
        nonlocal pending_memory
        if pending_memory is None:
            return
        record = dict(pending_memory['record'], resolved=resolved)
        decision = {'incident_id': record['incident_id'], 'resolved': resolved, 'store_probability': pending_memory['store_probability'],
                    'stored': pending_memory['store_probability'] > 0.5 and bool(settings['experience_memory'].get('write'))}
        if decision['stored']:
            written = await asyncio.to_thread(model_worker.call, 'experience_write', store=settings['experience_memory'],
                                              record=record, vector=pending_memory['vector'])
            decision.update(memory_id=written['memory_id'], store_size=written['store_size'])
            counters['memory_writes'] += 1
        write_new(root/f"experience-{record['incident_id']}.json", dict(decision, record=record))
        pending_memory = None

    async def assess_step(step, action, element, before, after, terminal):
        """Verbatim logic of live.assess_step (v3 + experience memory), with SeeAct's element identity."""
        nonlocal incident, incident_attempts, next_is_recovery, pending_memory, advice
        transition = ExecutedTransition(episode_id=episode_id, task_id=reset.get('task_id', 'web.'+block['task']),
                                        action_id=f'a{step}', task_description=reset['goal'], website_domain=reset.get('website_domain', ''),
                                        before=observation(before), after=observation(after), **action,
                                        phase='recovery_assessment' if next_is_recovery else 'interaction_assessment',
                                        incident_id=incident if next_is_recovery else None, terminated=terminal)
        budget.charge('assessment')
        assessment = await asyncio.to_thread(model_worker.call, 'assess', transition=asdict(transition), image_root=reset['image_root'])
        counters['assessments'] += 1
        write_new(root/f'assessment-{step:04d}.json', {'transition': asdict(transition), 'assessment': assessment})
        failed = assessment['signals']['outcome_label'] == 'FAILURE'
        page_changed = before['sha256'] != after['sha256']
        p_fail = assessment.get('outcome_probabilities', {}).get('FAILURE')
        uncertain = failed and transition.phase == 'interaction_assessment' and None not in (p_fail, gate) and p_fail < gate
        if uncertain:
            counters['uncertain_failures'] += 1
            write_new(root/f'gate-{step:04d}.json', {'p_failure': p_fail, 'gate': gate, 'recovery_triggered': False})
        if failed and not uncertain and not (action['action_type'] == 'SCROLL' and page_changed):
            blocked.append(dict(action, xpath=action['target'], url=after.get('url'),
                                label=describe_action(action['action_type'], element, action['value'])))
            counters['blocked_actions'] += 1
        if pending_memory and transition.phase == 'recovery_assessment':
            pending_memory['record']['recovery_attempts'].append({'action_type': action['action_type'], 'element': element,
                                                                  'value': action['value'], 'assessed_outcome': assessment['signals']['outcome_label']})
        if not failed:
            succeeded.append({'action_type': action['action_type'], 'element': element, 'value': action['value']})
            await close_incident(resolved=True)
            incident = None; incident_attempts = 0; next_is_recovery = False
        elif uncertain:
            pass
        elif not terminal and recovery_total < settings['recovery_attempts_per_episode'] and (incident is None or incident_attempts < settings['recovery_attempts_per_incident']):
            new_incident = incident is None
            if new_incident:
                incident = f'incident-{step}'; incident_attempts = 0
            next_is_recovery = True
            memory = None
            if budget.available(3):
                budget.charge('memory_query')
                memory = await asyncio.to_thread(model_worker.call, 'experience', transition=asdict(transition), image_root=reset['image_root'],
                                                 store=settings['experience_memory'], episode_id=episode_id, controls=after['controls'], url=reset['url'])
                if new_incident:
                    signals = assessment['signals']
                    pending_memory = {'vector': memory['vector'], 'store_probability': memory['store_probability'], 'record': {
                        'episode_id': episode_id, 'incident_id': incident, 'task_goal': reset['goal'], 'page_url': reset['url'],
                        'failed_action': {'action_type': action['action_type'], 'element': element, 'value': action['value']},
                        'failure_type': signals.get('failure_type'), 'strategy': signals.get('recovery_strategy'), 'recovery_attempts': []}}
                write_new(root/f'memory-{step:04d}.json', {k: v for k, v in memory.items() if k != 'vector'})
                counters['memory_queries'] += 1
            facts = await asyncio.to_thread(browser_worker.call, 'facts', goal=reset['goal'])
            advice = recovery_note(transition, assessment, element=element, page_changed=page_changed, facts=facts,
                                   memory=(memory or {}).get('examples', []), succeeded=[x for x in succeeded if x['action_type'] != 'SCROLL'][-3:])
            if memory and memory.get('examples'):
                counters['memory_exposures'] += 1
            write_new(root/f'note-{step:04d}.json', advice)
        else:
            await close_incident(resolved=False)
            incident = None; incident_attempts = 0; next_is_recovery = False

    def link_options(facts):
        return [{'label': f"Open the link '{link['text'] or link['href']}' ({link['href']})",
                 'seeact': {'action': 'GOTO', 'value': link['href']}} for link in (facts.get('goal_links') or [])[:3]]

    def search_option(box, query):
        return {'label': f"Search this site for '{query}' (type it into the search box, then press Enter)",
                'seeact': {'action': 'TYPE', 'value': query, 'element_option': box},
                'then': {'label': 'Press Enter to submit the search', 'seeact': {'action': 'PRESS ENTER'}, 'then': {'resolve': 'results'}}}

    async def recovery_options(prep):
        """Same offers as live.recovery_options, executed as native SeeAct actions."""
        nonlocal followup
        task = f"Task: {reset['goal']}\n"
        query = search_query(reset['goal'], reset['url'])
        current, followup = followup, None
        if current and current.get('resolve') == 'search':
            box = search_box(prep['element_list'])
            return (task+'You scrolled to the top to use the search box.', [search_option(box, query)]) if box is not None and query else None
        if current and current.get('resolve') == 'results':
            facts = await asyncio.to_thread(browser_worker.call, 'facts', goal=reset['goal'])
            write_new(root/f'results-facts-{calls:04d}.json', facts)
            links = link_options(facts)
            return (task+'The search was submitted. These links on the results page match the task words.', links) if links else None
        if current:
            return (task+'Your recovery step typed the search text; it has not been submitted yet.', [current])
        if not (isinstance(advice, dict) and advice.get('schema') == 'task2.recovery_note.v3'):
            return None
        facts = advice.get('facts') or {}
        offered = link_options(facts)
        box = search_box(prep['element_list'])
        if box is not None and query:
            offered.append(search_option(box, query))
        elif facts.get('search_boxes') and query:
            offered.append({'label': f"Scroll to the top and search this site for '{query}'",
                            'seeact': {'action': 'PRESS HOME'}, 'then': {'resolve': 'search'}})
        field = facts.get('field') or {}
        if field.get('value'):
            offered.append({'label': f"Press Enter to submit '{field['value']}'", 'seeact': {'action': 'PRESS ENTER'}})
        if (facts.get('scroll') or {}).get('at_bottom'):
            offered.append({'label': 'Scroll back to the top of the page', 'seeact': {'action': 'PRESS HOME'}})
        if steps_completed > 1:
            offered.append({'label': 'Go back to the previous page', 'seeact': {'action': 'GO BACK'}})
        allowed = []
        by_option = {e['option']: e for e in prep['element_list']}
        for option in offered:
            s = option['seeact']; e = by_option.get(s.get('element_option')) or {}
            proposed = recorded(s['action'], s.get('value'), e.get('tag_with_role'), e.get('description'))
            if not any(same_action(item, proposed, proposed['target']) for item in blocked):
                allowed.append(option)
        if not allowed:
            return None
        return (task+advice['text'], allowed[:5])

    async def generate(kind, messages):
        nonlocal calls
        budget.charge(kind)
        result = await asyncio.to_thread(model_worker.call, 'generate', request={'messages': messages, 'output_schema': None, 'advice': None, 'plain': True})
        counters['limit_hits'] += int(bool(result.get('limit_hit')))
        return result

    try:
        attach = await asyncio.to_thread(seeact_worker.call, 'attach', cdp_url=reset['cdp_url'], goal=reset['goal'], start_url=reset['url'],
                                         viewport=reset['viewport'], allowed_domains=reset['allowed_domains'], root=str(root/'seeact'))
        attached = True
        write_new(root/'seeact-attach.json', attach)
        before = await asyncio.to_thread(browser_worker.call, 'observe', episode_id=episode_id)
        write_new(root/'observation-0000.json', before)
        while step < settings['max_agent_steps'] and budget.available(2) and budget.executor_requests < budget.max_executor_requests:
          try:
            prep = await asyncio.to_thread(seeact_worker.call, 'prepare')
            write_new(root/f'seeact-{step:04d}-prepare.json', {k: v for k, v in prep.items() if k != 'element_list'})
            if assessment_task is not None:  # overlapped with SeeAct's page processing, as on Browser Use
                await assessment_task; assessment_task = None
            if not budget.available(2):
                stop = 'wall_clock'; break
            is_recovery = next_is_recovery
            fin = None; actor_calls = 0; picked = None; note_used = advice
            offer = await recovery_options(prep) if settings.get('recovery_options') else None
            if offer:
                context, options = offer
                calls += 1; prefix = root/f'actor-{calls:04d}'
                messages = [{'role': 'user', 'content': [image_part(prep['screenshot']), {'type': 'text', 'text': choice_text(context, options)}]}]
                result = await generate('actor_choice', messages); actor_calls += 1
                match = re.search(r'\d+', result['raw_response'])
                number = int(match.group()) if match else 0
                picked = options[number-1] if 1 <= number <= len(options) else None
                write_new(str(prefix)+'-choice.json', {'context': context, 'options': options, 'raw_response': result['raw_response'], 'chosen': number, 'picked': picked})
                if picked is not None:
                    counters['option_choices'] += 1; followup = picked.get('then')
                    s = picked['seeact']
                    fin = await asyncio.to_thread(seeact_worker.call, 'act', action=s['action'], value=s.get('value'),
                                                  element_option=s.get('element_option'), label=picked['label'])
            if fin is None:
                prompt = list(prep['prompt'])
                if isinstance(advice, dict) and advice.get('schema') == 'task2.recovery_note.v3':
                    prompt[1] = insert_note(prompt[1], prep.get('question_description'), advice['text'])
                calls += 1; prefix = root/f'actor-{calls:04d}'
                out0 = await generate('seeact_turn0', turn_messages(prompt, prep['screenshot'])); actor_calls += 1
                write_new(str(prefix)+'-turn0.json', out0)
                messages = turn_messages(prompt, prep['screenshot'], out0['raw_response'])
                out1 = await generate('seeact_turn1', messages); actor_calls += 1
                write_new(str(prefix)+'-turn1.json', out1)
                for attempt in range(regenerations if blocked else 0):
                    parsed = await asyncio.to_thread(seeact_worker.call, 'parse', output1=out1['raw_response'])
                    proposed = recorded(parsed['parsed']['action'], parsed['parsed']['value'], parsed['element_tag'], parsed['element_description'])
                    reason = next((f"'{item['label']}' was already tried on this page and did not make progress. Choose a different action."
                                   for item in blocked if same_action(item, proposed, proposed['target'])), None)
                    if reason is None:
                        break
                    counters['guard_blocks'] += 1
                    write_new(str(prefix)+f'-guard-{attempt}.json', {'blocked_proposal': parsed, 'reason': reason})
                    if not budget.available():
                        break
                    messages = messages + [{'role': 'assistant', 'content': out1['raw_response']},
                                           {'role': 'user', 'content': 'PROPOSAL BLOCKED, NOT EXECUTED: '+reason}]
                    out1 = await generate('actor_guard', messages); actor_calls += 1
                    write_new(str(prefix)+f'-guard-{attempt}-raw.json', out1)
                else:
                    if blocked and regenerations:
                        parsed = await asyncio.to_thread(seeact_worker.call, 'parse', output1=out1['raw_response'])
                        proposed = recorded(parsed['parsed']['action'], parsed['parsed']['value'], parsed['element_tag'], parsed['element_description'])
                        if any(same_action(item, proposed, proposed['target']) for item in blocked):
                            counters['guard_overrides'] += 1
                fin = await asyncio.to_thread(seeact_worker.call, 'finish', output0=out0['raw_response'], output1=out1['raw_response'])
            advice = None
            budget.executor_requests += 1
            score = await asyncio.to_thread(browser_worker.call, 'score')
            write_new(root/f'score-{step:04d}.json', score)
            after = await asyncio.to_thread(browser_worker.call, 'observe', episode_id=episode_id)
            write_new(root/f'observation-{step+1:04d}.json', after)
            action = recorded(fin['action'], fin['value'], fin['element_tag'], fin['element_description'])
            element = element_summary(fin['element_tag'], fin['element_description'])
            status = 'executed' if fin['executed'] else ('rejected' if fin['error'] else 'none')
            if fin['parsed']['action'] == 'None':
                counters['invalid_outputs'] += 1
            if fin['blocked_navigation']:
                counters['blocked_navigations'] += 1
            counters['executed' if status == 'executed' else 'rejected' if status == 'rejected' else 'no_action'] += 1
            if is_recovery and actor_calls:
                incident_attempts += 1; recovery_total += 1
                if status == 'executed':
                    counters['executed_recoveries'] += 1
            event = {'step': step, 'actor_calls': actor_calls, 'action_type': action['action_type'], 'seeact_action': fin['action'],
                     'target': fin['element_option'], 'element': element, 'value': action['value'], 'execution_status': status,
                     'error': fin['error'], 'history_line': fin['history_line'], 'blocked_navigation': fin['blocked_navigation'],
                     'recovery_proposal': is_recovery, 'incident_id': incident if is_recovery else None,
                     'recovery_option': picked['label'] if picked else None, 'note_shown': bool(note_used) and picked is None,
                     'state_changed': before['sha256'] != after['sha256'], 'url_before': before.get('url'), 'url_after': after.get('url')}
            write_new(root/f'action-{step:04d}.json', event)
            if page_path(before.get('url')) != page_path(after.get('url')):
                blocked.clear()
            if status == 'rejected' and action['action_type'] != 'NONE':
                blocked.append(dict(action, xpath=action['target'], url=after.get('url'),
                                    label=describe_action(action['action_type'], element, action['value'])+' (it could not be executed)'))
                counters['blocked_actions'] += 1
            terminal = bool(score['terminated'] or score.get('invalid_url'))
            assessable = status == 'executed' and action['action_type'] in ASSESSED and budget.available()
            if assessable:
                assessment_task = asyncio.create_task(assess_step(step, action, element, before, after, terminal))
                if terminal:
                    await assessment_task; assessment_task = None
            elif is_recovery:
                if incident_attempts >= settings['recovery_attempts_per_incident'] or recovery_total >= settings['recovery_attempts_per_episode']:
                    await close_incident(resolved=False)
                    incident = None; incident_attempts = 0; next_is_recovery = False
            before = after
            steps_completed += 1; step += 1
            if terminal:
                stop = 'environment_terminal'; break
            if fin['terminate']:
                stop = 'agent_done'; break
            if fin['continuous_no_op'] >= MAX_CONTINUOUS_NO_OP:
                stop = 'seeact_max_no_op'; break
          except BudgetExhausted:
            # The 2700 s wall clock ran out inside a step (SeeAct's page processing or a model call):
            # a failed episode under the study's budget rule, as for system S.
            stop = 'wall_clock'; assessment_task = None; break
        if assessment_task is not None:
            try:
                await assessment_task
            except BudgetExhausted:
                pass
            assessment_task = None
        await close_incident(resolved=False)
        if stop == 'budget' and time.monotonic()-budget.started >= budget.seconds:
            stop = 'wall_clock'
        result = {'episode_id': episode_id, 'system': SYSTEM, 'block': block, 'status': 'complete',
                  'completion': bool(score['terminated'] and not score.get('invalid_url') and score['raw_reward'] == 1.0),
                  'terminal_score': score, 'stop_reason': stop, 'counters': counters,
                  'model_calls': budget.calls, 'executor_requests': budget.executor_requests, 'recovery_attempts': recovery_total,
                  'elapsed_seconds': time.monotonic()-budget.started, 'agent_steps': steps_completed}
        write_new(root/'result.json', result)
        return result
    except Exception as exc:
        write_new(root/'infrastructure-error.json', {'error_type': type(exc).__name__, 'error': str(exc),
                  'model_calls': budget.calls, 'executor_requests': budget.executor_requests})
        raise
    finally:
        if assessment_task is not None:
            assessment_task.cancel()
        if attached:
            try:
                summary = await asyncio.to_thread(seeact_worker.call, 'close')
                write_new(root/'seeact-summary.json', summary)
            except Exception:
                pass
        await asyncio.to_thread(browser_worker.call, 'close')
