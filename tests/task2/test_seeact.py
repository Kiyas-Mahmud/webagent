"""Raw SeeAct baseline (system S): parser, domain block, budget, result schema."""
import asyncio
import base64
import json
from pathlib import Path
import subprocess
import pytest
from web_agent.eval.task2 import seeact_live
from web_agent.eval.task2.seeact_common import action_type, host_allowed

ROOT = Path(__file__).resolve().parents[2]
SEEACT_PYTHON = ROOT/'.task2-assets/seeact-env/bin/python'
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==')
WEB_V3_RESULT_KEYS = {'episode_id', 'system', 'block', 'status', 'completion', 'terminal_score', 'stop_reason', 'counters',
                      'model_calls', 'executor_requests', 'recovery_attempts', 'elapsed_seconds', 'agent_steps'}
SETTINGS = {'budget': {'max_model_calls': 102, 'max_executor_requests': 15, 'seconds': 2700}, 'max_agent_steps': 15}


def test_host_allowed_matches_browser_use_semantics():
    assert host_allowed('https://en.wikipedia.org/wiki/X', ['*.wikipedia.org'])
    assert host_allowed('https://wikipedia.org/', ['*.wikipedia.org'])
    assert not host_allowed('https://www.google.com/', ['*.wikipedia.org'])
    assert not host_allowed('https://evilwikipedia.org/', ['*.wikipedia.org'])
    assert host_allowed('https://api.jquery.com/', ['api.jquery.com'])
    assert not host_allowed('https://jquery.com/', ['api.jquery.com'])
    assert host_allowed('about:blank', ['*.epa.gov']) and host_allowed('chrome-error://chromewebdata/', ['*.epa.gov'])


def test_action_type_mapping():
    assert action_type('CLICK') == 'CLICK' and action_type('TYPE') == 'TYPE' and action_type('PRESS ENTER') == 'PRESS_KEY'
    assert action_type('SCROLL DOWN') == 'SCROLL' and action_type('PRESS END') == 'SCROLL'
    assert action_type('GOTO') == 'NAVIGATE' and action_type('GO BACK') == 'NAVIGATE'
    assert action_type('TERMINATE') == 'DONE'
    for none in ('NONE', 'SAY', 'MEMORIZE', 'None', None, ''):
        assert action_type(none) == 'NONE'


@pytest.mark.skipif(not SEEACT_PYTHON.exists(), reason='seeact-env not installed')
def test_seeact_parser_on_model_style_outputs():
    cases = [
        ('(Reiteration)\nThe search box.\n\n(Final Answer)\nELEMENT: D\nACTION: TYPE\nVALUE: Alan Turing', ['D', 'TYPE', 'Alan Turing']),
        ('ELEMENT: AB\nACTION: CLICK\nVALUE: None', ['AB', 'CLICK', 'None']),
        ('ELEMENT: ZZ\nACTION: PRESS ENTER\nVALUE: None', ['ZZ', 'PRESS ENTER', 'None']),
        ('I think we should click the link.', ['Invalid', 'None', '']),
        ('ACTION: TERMINATE\nVALUE: None', ['Invalid', 'TERMINATE', 'None']),
    ]
    code = ('import json,sys\nfrom seeact.demo_utils.format_prompt import postprocess_action_lmm\n'
            'print(json.dumps([list(postprocess_action_lmm(t)) for t in json.load(sys.stdin)]))')
    out = subprocess.run([str(SEEACT_PYTHON), '-c', code], input=json.dumps([c for c, _ in cases]), capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == [e for _, e in cases]


def test_turn_messages_follow_seeact_two_turn_layout(tmp_path):
    shot = tmp_path/'s.png'; shot.write_bytes(PNG)
    prompt = ['SYS', 'QUERY', 'REFER']
    t0 = seeact_live.turn_messages(prompt, shot)
    assert [m['role'] for m in t0] == ['system', 'user'] and t0[0]['content'] == 'SYS'
    assert t0[1]['content'][0] == {'type': 'text', 'text': 'QUERY'} and t0[1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
    t1 = seeact_live.turn_messages(prompt, shot, 'THOUGHT')
    assert [m['role'] for m in t1] == ['system', 'user', 'assistant', 'user']
    assert t1[2]['content'] == '\n\nTHOUGHT' and t1[3]['content'] == 'REFER'


class Fakes:
    """Browser, model and SeeAct workers with scripted behaviour."""

    def __init__(self, tmp_path, *, terminal_after=None, terminate_at=None, no_op_from=None):
        self.tmp = tmp_path; self.terminal_after = terminal_after; self.terminate_at = terminate_at; self.no_op_from = no_op_from
        self.obs = 0; self.steps = 0; self.generated = 0; self.closed = []; self.no_op = 0
        self.shot = tmp_path/'shot.png'; self.shot.write_bytes(PNG)

    def browser(self, op, **kw):
        if op == 'reset':
            return {'goal': 'G', 'url': 'https://a.org/', 'cdp_url': 'http://127.0.0.1:1', 'viewport': {'width': 1280, 'height': 720},
                    'allowed_domains': ['*.a.org'], 'image_root': str(self.tmp)}
        if op == 'observe':
            self.obs += 1; return {'sha256': str(self.obs), 'url': 'https://a.org/p'+str(self.obs)}
        if op == 'score':
            done = self.terminal_after is not None and self.steps >= self.terminal_after
            return {'terminated': done, 'raw_reward': 1.0 if done else 0.0, 'invalid_url': False, 'url': 'u'}
        if op == 'close':
            self.closed.append('browser'); return {}
        raise AssertionError(op)

    def model(self, op, **kw):
        if op == 'reset':
            return {}
        assert op == 'generate' and kw['request']['plain'] and kw['request']['output_schema'] is None
        self.generated += 1
        return {'raw_response': 'ELEMENT: A\nACTION: CLICK\nVALUE: None', 'limit_hit': self.generated % 7 == 0}

    def seeact(self, op, **kw):
        if op == 'attach':
            assert kw['allowed_domains'] == ['*.a.org']; return {'url': kw['start_url']}
        if op == 'prepare':
            return {'prompt': ['S', 'Q', 'R'], 'screenshot': str(self.shot), 'elements': 3, 'choices': [], 'options_text': '', 'url': 'u', 'step': self.steps+1}
        if op == 'finish':
            self.steps += 1
            no_op = self.no_op_from is not None and self.steps >= self.no_op_from
            self.no_op = self.no_op+1 if no_op else 0
            return {'parsed': {'element': 'A', 'action': 'NONE' if no_op else 'CLICK', 'value': ''}, 'action': 'NONE' if no_op else 'CLICK',
                    'element_option': None if no_op else 'A', 'element_description': 'x', 'element_tag': 'a', 'value': '',
                    'executed': not no_op, 'error': None, 'history_line': 'h', 'blocked_navigation': None, 'url': 'u',
                    'terminate': self.terminate_at is not None and self.steps >= self.terminate_at,
                    'continuous_no_op': self.no_op, 'valid_op': self.steps}
        if op == 'close':
            self.closed.append('seeact'); return {'num_step': self.steps}
        raise AssertionError(op)


def run_episode(tmp_path, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    f = Fakes(tmp_path, **kw)
    worker = lambda fn: type('W', (), {'call': staticmethod(fn)})()
    block = {'task': 't', 'repeat_id': 0, 'browser_reset_seed': 0, 'goal': 'G'}
    result = asyncio.run(seeact_live.seeact_episode(block=block, root=tmp_path/'ep', browser_worker=worker(f.browser),
                                                    model_worker=worker(f.model), seeact_worker=worker(f.seeact), settings=SETTINGS))
    return f, result


def test_budget_stop_after_15_steps_and_two_calls_per_step(tmp_path):
    f, r = run_episode(tmp_path)
    assert r['stop_reason'] == 'budget' and r['agent_steps'] == 15 and f.generated == 30
    assert r['model_calls'] == {'seeact_turn0': 15, 'seeact_turn1': 15} and r['executor_requests'] == 15
    assert r['completion'] is False and r['counters']['executed'] == 15 and r['counters']['limit_hits'] == 4
    assert f.closed == ['seeact', 'browser']
    assert sorted(p.name for p in (tmp_path/'ep').glob('action-*.json')) == [f'action-{i:04d}.json' for i in range(15)]
    assert len(list((tmp_path/'ep').glob('observation-*.json'))) == 16


def test_environment_terminal_wins_and_result_matches_web_v3_schema(tmp_path):
    f, r = run_episode(tmp_path, terminal_after=3)
    assert r['completion'] is True and r['stop_reason'] == 'environment_terminal' and r['agent_steps'] == 3
    assert set(r) == WEB_V3_RESULT_KEYS and r['system'] == 'S' and r['recovery_attempts'] == 0
    assert r['counters']['executed_recoveries'] == 0 and r['counters']['memory_exposures'] == 0
    saved = json.loads((tmp_path/'ep'/'result.json').read_text())
    assert saved['completion'] is True
    sample = ROOT/'.task2-assets/comparison/web-v3/episodes/0-arxiv-list-cs-ai-A/result.json'
    if sample.exists():
        assert set(json.loads(sample.read_text())) == set(r)


def test_wall_clock_exhaustion_is_a_failed_result_not_an_error(tmp_path, monkeypatch):
    import time as _time
    settings = dict(SETTINGS, budget=dict(SETTINGS['budget'], seconds=0.05))
    f = Fakes(tmp_path)
    slow_prepare = f.seeact
    def seeact(op, **kw):
        if op == 'prepare':
            _time.sleep(0.1)  # SeeAct's element listing outlives the wall clock mid-step
        return slow_prepare(op, **kw)
    worker = lambda fn: type('W', (), {'call': staticmethod(fn)})()
    block = {'task': 't', 'repeat_id': 0, 'browser_reset_seed': 0, 'goal': 'G'}
    r = asyncio.run(seeact_live.seeact_episode(block=block, root=tmp_path/'ep', browser_worker=worker(f.browser),
                                               model_worker=worker(f.model), seeact_worker=worker(seeact), settings=settings))
    assert r['stop_reason'] == 'wall_clock' and r['completion'] is False and r['agent_steps'] == 0
    assert (tmp_path/'ep'/'result.json').exists() and not (tmp_path/'ep'/'infrastructure-error.json').exists()
    assert f.closed == ['seeact', 'browser']


def test_terminate_and_no_op_stops(tmp_path):
    _, r = run_episode(tmp_path, terminate_at=4)
    assert r['stop_reason'] == 'agent_done' and r['agent_steps'] == 4 and r['completion'] is False
    _, r = run_episode(tmp_path/'b', no_op_from=2)
    assert r['stop_reason'] == 'seeact_max_no_op' and r['agent_steps'] == 6 and r['counters']['no_action'] == 5
