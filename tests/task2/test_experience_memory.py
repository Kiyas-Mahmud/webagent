import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from web_agent.eval.task2 import live
from web_agent.memory.experience import ExperienceMemory, advice, element_summary

CHECKPOINT = 'a'*64


def vector(*head):
    v = np.zeros(768, np.float32); v[:len(head)] = head
    return v


def record(episode='other', text='Submit', resolved=True):
    return {'episode_id': episode, 'incident_id': 'incident-0', 'task_goal': 'Select words and click Submit.',
            'page_url': 'http://x/task.html', 'failure_type': 'PERCEPTION_ERROR', 'strategy': 'REPLAN',
            'failed_action': {'action_type': 'CLICK', 'element': {'tag': 'button', 'text': text, 'attributes': {}}, 'value': None},
            'recovery_attempts': [{'action_type': 'CLICK', 'element': {'tag': 'input', 'text': 'tragic', 'attributes': {'type': 'checkbox'}},
                                   'value': None, 'assessed_outcome': 'SUCCESS'}], 'resolved': resolved}


def test_store_persists_reloads_and_rejects_other_embedding_space(tmp_path):
    store = ExperienceMemory(tmp_path, threshold=0.8, checkpoint_sha256=CHECKPOINT)
    memory_id = store.add(record(), vector(1, 0))
    again = ExperienceMemory(tmp_path, threshold=0.8, checkpoint_sha256=CHECKPOINT)
    assert len(again) == 1 and again.records[0]['memory_id'] == memory_id
    assert np.allclose(again.vectors[0], vector(1, 0))
    with pytest.raises(ValueError, match='MEMORY_EMBEDDING_SPACE_MISMATCH'):
        ExperienceMemory(tmp_path, threshold=0.8, checkpoint_sha256='b'*64)
    with pytest.raises(RuntimeError, match='read-only'):
        ExperienceMemory(tmp_path, threshold=0.8, checkpoint_sha256=CHECKPOINT, write_enabled=False).add(record(), vector(1))


def test_admission_threshold_episode_and_observable_element(tmp_path):
    store = ExperienceMemory(tmp_path, threshold=0.8, checkpoint_sha256=CHECKPOINT)
    store.add(record(), vector(1, 0)); store.add(record(episode='now'), vector(1, 0.01)); store.add(record(), vector(0, 1))
    submit = [{'text': 'Submit', 'accessible_names': []}]
    reasons = {h.similarity > 0.9 and h.record['episode_id']: h.exclusion_reason for h in store.query(vector(1, 0), current_episode='now', current_controls=submit, current_url='http://x:8/task.html?q=1')}
    assert reasons == {'other': None, 'now': 'SAME_EPISODE', False: 'BELOW_THRESHOLD'}
    [hit] = [h for h in store.query(vector(1, 0), current_episode='now', current_controls=[{'text': 'Cancel'}], current_url='http://x/task.html') if h.record['episode_id'] == 'other' and h.similarity > .9]
    assert hit.exclusion_reason == 'NO_RECORDED_ELEMENT_ON_CURRENT_PAGE'
    shown = advice(store.query(vector(1, 0), current_episode='now', current_controls=submit, current_url='http://x:8/task.html?q=1')[0])
    assert shown['incident_outcome'] == 'resolved' and shown['recovery_attempts'][0]['element']['text'] == 'tragic'
    assert 'episode_id' not in shown and 'vector' not in shown


def test_element_summary_is_observable_only():
    summary = element_summary({'node_name': 'INPUT', 'ax_name': ' tragic ', 'x_path': '/html/body/input',
                               'attributes': {'type': 'checkbox', 'id': 'ch1', 'data-secret': 'x', 'style': 'y'}})
    assert summary == {'tag': 'input', 'text': 'tragic', 'attributes': {'id': 'ch1', 'type': 'checkbox'}}
    assert element_summary(None) is None


@pytest.mark.parametrize('store_probability,stored', [(0.9, True), (0.2, False)])
def test_live_loop_writes_incident_only_when_memory_head_says_store(tmp_path, monkeypatch, store_probability, stored):
    views = ModuleType('browser_use.agent.views'); views.AgentStepInfo = lambda **kw: kw
    monkeypatch.setitem(sys.modules, 'browser_use.agent.views', views)
    clicked = {'node_name': 'BUTTON', 'ax_name': 'Submit', 'attributes': {'id': 'subbtn'}}

    class Output:
        action = [SimpleNamespace(model_dump=lambda **kw: {'click': {'index': 4}})]
        def model_dump(self, **kw): return {'action': [{'click': {'index': 4}}]}
    result = SimpleNamespace(error=None, model_dump=lambda **kw: {'error': None})

    class Agent:
        def __init__(self, llm):
            self.llm = llm; self.contexts = []
            self.state = SimpleNamespace(consecutive_failures=0, last_model_output=None, last_result=[])
            self.history = SimpleNamespace(history=[])
        async def step(self, info):
            self.contexts.append(list(self.llm.memory_context))
            self.llm.budget.charge('actor'); self.llm.calls += 1
            self.state.last_model_output = Output(); self.state.last_result = [result]
            self.history.history.append(SimpleNamespace(state=SimpleNamespace(interacted_element=[SimpleNamespace(to_dict=lambda: clicked)])))
            (tmp_path/'episode'/f'actor-{self.llm.calls:04d}-parsed.json').write_text(json.dumps({'native_proposal': Output().model_dump()}))

    class Browser:
        async def stop(self): pass
    agents = []

    async def create(reset, llm, root):
        agents.append(Agent(llm)); return agents[-1], Browser()
    monkeypatch.setattr(live, 'create_agent', create)
    calls = []; seq = 0

    class Worker:
        def call(self, op, **kw):
            nonlocal seq; calls.append((op, kw))
            if op == 'reset': return {'goal': 'goal', 'image_root': str(tmp_path), 'url': 'http://x/task.html'}
            if op == 'observe':
                seq += 1
                return {'episode_id': kw['episode_id'], 'observation_id': f'o{seq}', 'sequence': seq, 'image': 'images/x.png', 'sha256': 'a'*64,
                        'controls': [{'text': 'Submit'}]}
            if op == 'score': return {'terminated': False, 'invalid_url': False, 'raw_reward': 0}
            if op == 'assess':
                t = kw['transition']
                return {k: t[k] for k in ('episode_id', 'action_id', 'phase', 'incident_id')} | {'signals': {
                    'outcome_label': 'FAILURE', 'failure_type': 'ACTION_MISMATCH', 'recovery_strategy': 'RETRY'}}
            if op == 'experience':
                return {'vector': [1.0]+[0.0]*767, 'store_probability': store_probability, 'candidates': [],
                        'examples': [{'past_task': 'earlier'}], 'store_size': 0, 'threshold': 0.8, 'write_enabled': True}
            if op == 'experience_write': return {'memory_id': 'm1', 'store_size': 1}
    settings = {'budget': {}, 'max_agent_steps': 3, 'recovery_attempts_per_episode': 4, 'recovery_attempts_per_incident': 2,
                'excluded_memory_tasks': [], 'memory_mode': 'experience',
                'experience_memory': {'directory': str(tmp_path/'mem'), 'threshold': 0.8, 'write': True}}
    block = {'task': 'test', 'repeat_id': 0, 'browser_reset_seed': 1, 'goal': 'goal'}
    r = asyncio.run(live.episode(system='C', block=block, root=tmp_path/'episode', browser_worker=Worker(), model_worker=Worker(), settings=settings))
    writes = [kw for op, kw in calls if op == 'experience_write']
    assert r['counters']['memory_writes'] == len(writes) == int(stored)
    decision = json.loads((tmp_path/'episode'/'experience-incident-0.json').read_text())
    assert decision['stored'] is stored and decision['resolved'] is False
    rec = decision['record']
    assert rec['failed_action']['element'] == {'tag': 'button', 'text': 'Submit', 'attributes': {'id': 'subbtn'}}
    assert [a['assessed_outcome'] for a in rec['recovery_attempts']] == ['FAILURE', 'FAILURE']
    assert agents[0].contexts[1] == [{'past_task': 'earlier'}] and agents[0].llm.memory_kind == 'experience'
    assert not list((tmp_path/'episode').glob('memory-*.json')) or 'vector' not in json.loads(next((tmp_path/'episode').glob('memory-*.json')).read_text())



def test_retrieval_is_scoped_to_the_same_observable_page(tmp_path):
    store = ExperienceMemory(tmp_path, threshold=0.8, checkpoint_sha256=CHECKPOINT)
    store.add(record(), vector(1, 0))
    other = dict(record(), page_url='http://x/other.html'); store.add(other, vector(1, 0))
    hits = store.query(vector(1, 0), current_episode='now', current_controls=[{'text': 'Submit'}], current_url='http://x:9/task.html#f')
    assert [h.record['page_url'] for h in hits] == ['http://x/task.html']
    assert store.query(vector(1, 0), current_episode='now', current_controls=[], current_url='http://y/task.html') == []


def test_live_loop_imports_in_the_isolated_browser_use_environment():
    import os, subprocess
    python = '.task2-assets/browser-use-env/bin/python'
    if not os.path.exists(python):
        pytest.skip('isolated Browser Use environment not installed')
    done = subprocess.run([python, '-c', 'import web_agent.eval.task2.live, web_agent.eval.task2.native_model'],
                          env=dict(os.environ, PYTHONPATH='src'), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]


def test_record_without_any_observable_element_is_never_admitted(tmp_path):
    store = ExperienceMemory(tmp_path, threshold=0.8, checkpoint_sha256=CHECKPOINT)
    blank = record(); blank['failed_action'] = dict(blank['failed_action'], element=None)
    blank['recovery_attempts'] = [dict(blank['recovery_attempts'][0], element=None)]
    store.add(blank, vector(1, 0))
    [hit] = store.query(vector(1, 0), current_episode='now', current_controls=[{'text': 'Submit'}], current_url='http://x/task.html')
    assert hit.exclusion_reason == 'NO_RECORDED_ELEMENT'


def test_web_worker_trims_only_rounding_overshoot():
    from web_agent.eval.task2.web_worker import clamp_rounding
    # Same shape as CONTROL_JAVASCRIPT's snapshot: {'controls': [...]}.
    [row] = clamp_rounding({'controls': [{'target_bbox': [0.03125, 0.951563, 0.210938, 0.048438]}]})['controls']
    assert row['target_bbox'][1] + row['target_bbox'][3] <= 1
    [big] = clamp_rounding({'controls': [{'target_bbox': [0.5, 0.5, 0.6, 0.1]}]})['controls']
    assert big['target_bbox'] == [0.5, 0.5, 0.6, 0.1]
