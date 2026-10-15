"""SeeAct + framework (system T): hooks, options, guard, memory and result schema."""
import asyncio
import json
from pathlib import Path
import re
import pytest
from web_agent.eval.task2 import seeact_framework as fw

ROOT = Path(__file__).resolve().parents[2]
PNG = bytes.fromhex('89504e470d0a1a0a0000000d494844520000000100000001080600000' '01f15c4890000000d49444154789c636400010000050001' '0d0a2db40000000049454e44ae426082')
C_RESULT_KEYS = {'episode_id', 'system', 'block', 'status', 'completion', 'terminal_score', 'stop_reason', 'counters',
                 'model_calls', 'executor_requests', 'recovery_attempts', 'elapsed_seconds', 'agent_steps'}
C_COUNTERS = {'assessments', 'blocked_actions', 'executed', 'executed_recoveries', 'guard_blocks', 'guard_overrides',
              'memory_exposures', 'memory_queries', 'memory_writes', 'option_choices', 'rejected', 'uncertain_failures'}
QD = 'The screenshot below shows the webpage you see.'
ELEMENTS = [{'option': 'A', 'tag': 'a', 'tag_with_role': 'a', 'description': 'Products', 'center_point': [0.1, 0.1]},
            {'option': 'B', 'tag': 'a', 'tag_with_role': 'a', 'description': 'About us', 'center_point': [0.2, 0.1]},
            {'option': 'C', 'tag': 'input', 'tag_with_role': 'input type="search"', 'description': 'placeholder="Search"', 'center_point': [0.5, 0.1]}]


def settings(tmp_path, **extra):
    base = {'budget': {'max_model_calls': 102, 'max_executor_requests': 15, 'seconds': 2700}, 'max_agent_steps': 15,
            'recovery_mode': 'v3', 'memory_mode': 'experience', 'recovery_options': True, 'guard_regenerations': 2,
            'recovery_trigger_probability': 0.9, 'recovery_attempts_per_incident': 2, 'recovery_attempts_per_episode': 15,
            'experience_memory': {'directory': str(tmp_path/'memory'), 'threshold': 0.5, 'write': True}}
    base.update(extra); return base


class Scenario:
    """Scripted browser, model and SeeAct workers. `turns` maps step -> turn-1 answer (list for guard retries);
    `verdicts` maps assessed step -> (label, P(failure)); `facts` is what the page facts op returns."""

    def __init__(self, tmp_path, *, turns, verdicts, facts=None, terminal_after=None, page_changes=(), choice='1', elements=ELEMENTS):
        self.tmp = tmp_path; self.turns = turns; self.verdicts = verdicts; self.facts = facts or {}
        self.terminal_after = terminal_after; self.page_changes = set(page_changes); self.choice = choice; self.elements = elements
        self.shot = tmp_path/'shot.png'; self.shot.write_bytes(PNG)
        self.obs = 0; self.steps = 0; self.executed = []; self.requests = []; self.retries = {}; self.writes = []
        self.sha = 'p0'

    # browser worker
    def browser(self, op, **kw):
        if op == 'reset':
            return {'goal': 'Open the cPanel pricing page.', 'url': 'https://a.org/', 'cdp_url': 'cdp', 'viewport': {'width': 1280, 'height': 720},
                    'allowed_domains': ['*.a.org'], 'image_root': str(self.tmp), 'task_id': 'web.t', 'website_domain': 'a.org'}
        if op == 'observe':
            self.obs += 1
            return {'episode_id': 'e', 'observation_id': f'o{self.obs}', 'sequence': self.obs, 'image': 'shot.png', 'sha256': self.sha,
                    'url': 'https://a.org/'+self.sha, 'controls': []}
        if op == 'score':
            done = self.terminal_after is not None and self.steps >= self.terminal_after
            return {'terminated': done, 'raw_reward': 1.0 if done else 0.0, 'invalid_url': False, 'url': 'u'}
        if op == 'facts':
            return dict(self.facts, url='https://a.org/')
        if op == 'close':
            return {}
        raise AssertionError(op)

    # model worker
    def model(self, op, **kw):
        if op == 'reset':
            return {}
        if op == 'generate':
            text = json.dumps(kw['request']['messages']); self.requests.append(kw['request'])
            if 'Recovery options for the next action' in text:
                return {'raw_response': self.choice, 'limit_hit': False}
            if 'PROPOSAL BLOCKED' in text:
                step = self.steps; self.retries[step] = self.retries.get(step, 0)+1
                return {'raw_response': self.turns[step][self.retries[step]], 'limit_hit': False}
            last = kw['request']['messages'][-1]
            if last['role'] == 'user' and isinstance(last['content'], str):  # referring prompt = turn 1
                answer = self.turns[self.steps]
                return {'raw_response': answer[0] if isinstance(answer, list) else answer, 'limit_hit': False}
            return {'raw_response': 'thinking', 'limit_hit': False}
        if op == 'assess':
            t = kw['transition']; step = int(t['action_id'][1:])
            label, p = self.verdicts.get(step, ('SUCCESS', 0.01))
            if t['phase'] == 'recovery_assessment':
                signals = {'outcome_label': label}
            else:
                signals = {'outcome_label': label, 'failure_type': 'PERCEPTION_ERROR', 'recovery_strategy': 'REPLAN'}
            return {'schema': 'task2.assessment.v1', 'phase': t['phase'], 'episode_id': t['episode_id'], 'action_id': t['action_id'],
                    'incident_id': t['incident_id'], 'signals': signals, 'outcome_probabilities': {'FAILURE': p, 'SUCCESS': 1-p}, 'raw_logits': {}}
        if op == 'experience':
            return {'vector': [0.1, 0.2], 'store_probability': 0.7, 'store_size': 0, 'threshold': 0.5, 'write_enabled': True, 'candidates': [], 'examples': []}
        if op == 'experience_write':
            self.writes.append(kw['record']); return {'memory_id': 'm1', 'store_size': 1}
        raise AssertionError(op)

    # seeact worker
    def parse(self, output1):
        m = re.search(r'ELEMENT: ([A-Z]+)\s*ACTION: ([A-Z ]+?)\s*VALUE: (.*)$', output1, re.S)
        if not m:
            return {'parsed': {'element': 'Invalid', 'action': 'None', 'value': ''}, 'element_option': None, 'element_description': None, 'element_tag': None}
        letter, action, value = m.group(1), m.group(2).strip(), m.group(3).strip()
        e = next((x for x in self.elements if x['option'] == letter), None)
        return {'parsed': {'element': letter, 'action': action, 'value': value}, 'element_option': e['option'] if e else None,
                'element_description': e['description'] if e else None, 'element_tag': e['tag_with_role'] if e else None}

    def run(self, action, value, element):
        self.steps += 1
        if self.steps in self.page_changes:
            self.sha = f'p{self.steps}'
        self.executed.append((action, value, element['option'] if element else None))
        no_op = action in ('NONE', 'None')
        return {'parsed': {'element': element['option'] if element else 'Invalid', 'action': action, 'value': value or ''},
                'action': 'NONE' if no_op else action, 'element_option': element['option'] if element else None,
                'element_description': element['description'] if element else None, 'element_tag': element['tag_with_role'] if element else None,
                'value': value, 'executed': not no_op, 'error': None, 'history_line': 'h', 'blocked_navigation': None, 'url': 'u',
                'terminate': action == 'TERMINATE', 'continuous_no_op': 0, 'valid_op': self.steps}

    def seeact(self, op, **kw):
        if op == 'attach':
            return {'url': kw['start_url']}
        if op == 'prepare':
            return {'step': self.steps+1, 'url': 'u', 'screenshot': str(self.shot), 'elements': len(self.elements), 'element_list': list(self.elements),
                    'prompt': ['SYS', f'You are asked to complete the following task: G\n\nPrevious Actions:\n\n{QD} Think.', 'REFER'],
                    'question_description': QD, 'choices': [], 'options_text': '', 'previous_actions': []}
        if op == 'parse':
            return self.parse(kw['output1'])
        if op == 'finish':
            p = self.parse(kw['output1']); e = next((x for x in self.elements if x['option'] == p['element_option']), None)
            return self.run(p['parsed']['action'], p['parsed']['value'], e)
        if op == 'act':
            e = next((x for x in self.elements if x['option'] == kw.get('element_option')), None)
            return self.run(kw['action'], kw.get('value'), e)
        if op == 'close':
            return {'num_step': self.steps}
        raise AssertionError(op)


def run(tmp_path, scenario, **extra):
    worker = lambda fn: type('W', (), {'call': staticmethod(fn)})()
    block = {'task': 'cpanel-pricing', 'repeat_id': 0, 'browser_reset_seed': 0, 'goal': 'Open the cPanel pricing page.'}
    return asyncio.run(fw.seeact_framework_episode(block=block, root=tmp_path/'ep', browser_worker=worker(scenario.browser),
                                                   model_worker=worker(scenario.model), seeact_worker=worker(scenario.seeact),
                                                   settings=settings(tmp_path, **extra)))


def test_helpers_identity_values_and_note_position():
    assert fw.identity('a', ' Main   Page\n ') == 'a|main page' and fw.identity(None, None) is None
    assert fw.recorded('SCROLL UP', None, None, None) == {'action_type': 'SCROLL', 'target': None, 'value': '{"down": false}'}
    assert fw.recorded('PRESS HOME', None, None, None)['value'] == '{"down": false}' and fw.recorded('SCROLL DOWN', None, None, None)['value'] == '{"down": true}'
    assert fw.recorded('PRESS ENTER', 'None', None, None) == {'action_type': 'PRESS_KEY', 'target': None, 'value': 'Enter'}
    assert fw.recorded('TYPE', 'x', 'input', 'Search')['target'] == 'input|search'
    assert fw.search_box(ELEMENTS) == 'C' and fw.search_box(ELEMENTS[:2]) is None
    out = fw.insert_note('HEAD\n\nPrevious Actions:\nA\n\n' + QD + ' tail', QD, 'NOTE')
    assert out.index('Previous Actions') < out.index('NOTE') < out.index(QD)


def test_failure_opens_note_then_option_is_executed_as_seeact_action_and_memory_closes(tmp_path):
    # step 0: SCROLL DOWN, page unchanged -> confident failure; step 1: option 1 (goal link) -> GOTO -> page completes.
    sc = Scenario(tmp_path, turns={0: 'ELEMENT: ZZ\nACTION: SCROLL DOWN\nVALUE: None'}, verdicts={0: ('FAILURE', 0.99), 1: ('SUCCESS', 0.0)},
                  facts={'goal_links': [{'text': 'Pricing', 'href': 'https://a.org/pricing/', 'visible': False, 'score': 1.0}],
                         'scroll': {'at_bottom': True, 'at_top': False}, 'search_boxes': [], 'field': None},
                  terminal_after=2)
    r = run(tmp_path, sc)
    assert r['completion'] is True and r['stop_reason'] == 'environment_terminal' and r['agent_steps'] == 2 and r['system'] == 'T'
    assert sc.executed == [('SCROLL DOWN', 'None', None), ('GOTO', 'https://a.org/pricing/', None)]
    c = r['counters']
    assert c['assessments'] == 2 and c['blocked_actions'] == 1 and c['option_choices'] == 1 and c['executed_recoveries'] == 1
    assert c['memory_queries'] == 1 and c['memory_writes'] == 1 and c['uncertain_failures'] == 0 and r['recovery_attempts'] == 1
    assert set(r) == C_RESULT_KEYS and C_COUNTERS <= set(c)
    ep = tmp_path/'ep'
    note = json.loads((ep/'note-0000.json').read_text())
    assert note['schema'] == 'task2.recovery_note.v3' and "'Pricing' -> https://a.org/pricing/ (hidden)" in note['text']
    choice = json.loads((ep/'actor-0002-choice.json').read_text())
    labels = [o['label'] for o in choice['options']]
    assert labels[0].startswith("Open the link 'Pricing'") and any(l.startswith("Search this site for 'cPanel pricing'") for l in labels)
    assert any(l == 'Scroll back to the top of the page' for l in labels) and choice['picked']['seeact'] == {'action': 'GOTO', 'value': 'https://a.org/pricing/'}
    assert (ep/'experience-incident-0.json').exists() and sc.writes[0]['resolved'] is True
    event = json.loads((ep/'action-0001.json').read_text())
    assert event['recovery_proposal'] and event['recovery_option'].startswith('Open the link') and event['actor_calls'] == 1
    assert r['model_calls'] == {'seeact_turn0': 1, 'seeact_turn1': 1, 'assessment': 2, 'memory_query': 1, 'actor_choice': 1}
    sample = ROOT/'.task2-assets/comparison/web-v3/episodes/0-cpanel-pricing-C/result.json'
    if sample.exists():
        real = json.loads(sample.read_text())
        assert set(real) == set(r) and set(real['counters']) <= set(c)


def test_note_is_inserted_in_turn0_and_guard_blocks_the_repeated_click(tmp_path):
    # step 0: CLICK A, page unchanged -> failure; no facts -> no options; step 1: actor repeats CLICK A -> blocked -> CLICK B.
    sc = Scenario(tmp_path, turns={0: 'ELEMENT: A\nACTION: CLICK\nVALUE: None',
                                   1: ['ELEMENT: A\nACTION: CLICK\nVALUE: None', 'ELEMENT: B\nACTION: CLICK\nVALUE: None']},
                  verdicts={0: ('FAILURE', 0.97)}, facts={'goal_links': [], 'scroll': {}, 'search_boxes': [], 'field': None},
                  elements=ELEMENTS[:2], terminal_after=2)
    r = run(tmp_path, sc)
    assert sc.executed == [('CLICK', 'None', 'A'), ('CLICK', 'None', 'B')]
    c = r['counters']
    assert c['guard_blocks'] == 1 and c['guard_overrides'] == 0 and c['option_choices'] == 0 and c['blocked_actions'] == 1
    turn0 = [q for q in sc.requests if q['messages'][-1]['role'] == 'user' and isinstance(q['messages'][-1]['content'], list)]
    query = turn0[1]['messages'][1]['content'][0]['text']
    assert 'FAILURE MONITOR: your last action (click <a> \'Products\')' in query and query.index('Previous Actions') < query.index('FAILURE MONITOR') < query.index(QD)
    assert any('PROPOSAL BLOCKED, NOT EXECUTED' in json.dumps(q['messages'][-1]) for q in sc.requests)
    assert (tmp_path/'ep'/'actor-0002-guard-0.json').exists()
    assert r['model_calls']['actor_guard'] == 1


def test_uncertain_alarm_is_gated_and_scroll_that_moved_is_not_blocked(tmp_path):
    sc = Scenario(tmp_path, turns={0: 'ELEMENT: ZZ\nACTION: SCROLL DOWN\nVALUE: None', 1: 'ELEMENT: ZZ\nACTION: SCROLL DOWN\nVALUE: None'},
                  verdicts={0: ('FAILURE', 0.6), 1: ('FAILURE', 0.99)}, page_changes={2}, terminal_after=3,
                  facts={'goal_links': [], 'scroll': {}, 'search_boxes': [], 'field': None}, choice='0')
    sc.turns[2] = 'ELEMENT: ZZ\nACTION: TERMINATE\nVALUE: None'
    r = run(tmp_path, sc)
    c = r['counters']
    assert c['uncertain_failures'] == 1 and (tmp_path/'ep'/'gate-0000.json').exists() and not (tmp_path/'ep'/'note-0000.json').exists()
    assert c['blocked_actions'] == 0  # step 1 scrolled and the page changed: not a dead end
    # step 2: the page has a search box and the actor took >1 step, so search and 'Go back' are offered;
    # the actor answers 0 and decides itself (TERMINATE, not assessed)
    choice = json.loads((tmp_path/'ep'/'actor-0003-choice.json').read_text())
    labels = [o['label'] for o in choice['options']]
    assert labels[0].startswith("Search this site for 'cPanel pricing'") and labels[1] == 'Go back to the previous page' and len(labels) == 2
    assert choice['chosen'] == 0 and choice['picked'] is None
    assert c['assessments'] == 2 and c['option_choices'] == 0 and sc.executed[-1][0] == 'TERMINATE'
    assert json.loads((tmp_path/'ep'/'action-0002.json').read_text())['note_shown'] is True


def test_wall_clock_inside_a_step_is_a_failed_result(tmp_path):
    import time as _time
    sc = Scenario(tmp_path, turns={0: 'ELEMENT: A\nACTION: CLICK\nVALUE: None'}, verdicts={})
    original = sc.seeact
    def slow(op, **kw):
        if op == 'prepare':
            _time.sleep(0.1)
        return original(op, **kw)
    sc.seeact = slow
    r = run(tmp_path, sc, budget={'max_model_calls': 102, 'max_executor_requests': 15, 'seconds': 0.05})
    assert r['stop_reason'] == 'wall_clock' and r['completion'] is False and (tmp_path/'ep'/'result.json').exists()
    assert not (tmp_path/'ep'/'infrastructure-error.json').exists()


def test_settings_guard():
    with pytest.raises(ValueError):
        asyncio.run(fw.seeact_framework_episode(block={}, root='/nonexistent', browser_worker=None, model_worker=None, seeact_worker=None,
                                                settings={'recovery_mode': 'v2'}))
