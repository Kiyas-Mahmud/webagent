import json
from types import SimpleNamespace

import pytest
import torch

from web_agent.eval.task1.core import file_hash
from web_agent.eval.task1.qwen25_backends import QWEN25_CHECKPOINT_HASH
from web_agent.eval.task2.assessment import Observation, ExecutedTransition, Qwen25TransitionAssessor
from web_agent.eval.task2.campaign_v2 import select_families
from web_agent.eval.task2.model_worker import BACKENDS, actor_batch
from web_agent.eval.task2.native_model import LocalInternVL

RULE = json.load(open('configs/eval/task2/qwen25_v1.json'))['selection_rule']


def screen(counts, episodes=5):
    return {'repeats': 5, 'per_family': [{'system': 'A', 'task': t, 'episodes': episodes, 'completed': c}
                                         for t, c in counts.items()]}


def test_selection_keeps_partial_families_closest_to_half():
    counts = {f'mid-{i:02d}': 2 + i % 2 for i in range(12)} | {'always': 5, 'never': 0, 'rare': 1}
    kept, screened = select_families(screen(counts), RULE)
    assert len(kept) == 10 and 'always' not in kept and 'never' not in kept and 'rare' not in kept
    assert screened['always'] == 5


def test_selection_fills_with_zero_families_but_never_all_success():
    kept, _ = select_families(screen({'a': 5, 'b': 0, 'c': 0, 'd': 3, 'e': 0, 'f': 0, 'g': 0, 'h': 0}), RULE)
    assert kept == ['b', 'c', 'd', 'e', 'f', 'g']


def test_selection_uses_eligible_denominator():
    kept, _ = select_families({'repeats': 5, 'per_family': [
        {'system': 'A', 'task': 'x', 'episodes': 4, 'completed': 4},
        {'system': 'A', 'task': 'y', 'episodes': 4, 'completed': 3}]}, dict(RULE, min_families=1))
    assert kept == ['y']


def test_selection_rejects_unscreened_family():
    with pytest.raises(ValueError, match='Unscreened'):
        select_families(screen({'x': 0}, episodes=0), RULE)


def test_qwen25_assessor_keeps_argmax_rule_and_logs_probabilities(tmp_path):
    from PIL import Image
    (tmp_path/'images').mkdir()
    for name in ('before', 'after'):
        Image.new('RGB', (8, 8)).save(tmp_path/'images'/f'{name}.png')
    obs = [Observation('e', n, i, f'images/{n}.png', file_hash(tmp_path/f'images/{n}.png'))
           for i, n in enumerate(('before', 'after'))]
    transition = ExecutedTransition('e', 't', 'a0', 'Click it', 'miniwob.local', *obs, 'CLICK', '3', None)
    backend = SimpleNamespace(root=tmp_path, torch=torch, batch=lambda r: r, model=lambda b: {
        'outcome': torch.tensor([[0.4, 0.0]]), 'failure_type': torch.tensor([[1., 0., 0., 0.]]),
        'recovery': torch.tensor([[1., 0., 0., 0., 0., 0.]])})
    result = Qwen25TransitionAssessor(backend).assess(transition)
    assert result['signals']['outcome_label'] == 'SUCCESS'
    assert result['checkpoint_sha256'] == QWEN25_CHECKPOINT_HASH
    assert abs(sum(result['outcome_probabilities'].values()) - 1) < 1e-6
    assert result['outcome_probabilities']['SUCCESS'] > 0.5


def test_qwen_actor_batch_requires_one_grid_per_screenshot():
    class Processor:
        def apply_chat_template(self, messages, **kw):return 'chat'
        def __call__(self, **kw):return {'image_grid_thw': torch.ones((2, 3))}
    with pytest.raises(ValueError, match='grid'):
        actor_batch('qwen25', SimpleNamespace(processor=Processor()), [], ['one'])
    assert actor_batch('qwen25', SimpleNamespace(processor=Processor()), [], ['one', 'two'])


def test_backend_labels_and_actor_identity():
    assert BACKENDS['qwen25']['label'] == json.load(open('configs/eval/task2/qwen25_v1.json'))['actor_backend']
    llm = LocalInternVL(None, '.', None, 'Qwen2.5-VL-7B-Instruct-local-base')
    assert llm.model == llm.name == llm.model_name == 'Qwen2.5-VL-7B-Instruct-local-base'
    assert LocalInternVL(None, '.', None).model == 'InternVL3.5-8B-HF-local-base'


def test_whole_response_fence_rule_matches_upstream_and_never_edits_body():
    import re
    from pathlib import Path
    from web_agent.eval.task2.native_model import _JSON_FENCE_RE, strict_object, unwrap_native_fence
    upstream = Path('.task2-assets/browser-use-env/lib/python3.12/site-packages/browser_use/llm/ollama/chat.py').read_text()
    assert "_JSON_FENCE_RE = re.compile(r'" + _JSON_FENCE_RE.pattern + "'" in upstream
    assert _JSON_FENCE_RE.flags & re.IGNORECASE and _JSON_FENCE_RE.flags & re.DOTALL
    assert strict_object(unwrap_native_fence('```json\n{"action": [{"click": {"index": 5}}]}\n```')) == {'action': [{'click': {'index': 5}}]}
    assert unwrap_native_fence('Answer: ```json\n{}\n```') == 'Answer: ```json\n{}\n```'
    with pytest.raises(ValueError):
        strict_object(unwrap_native_fence('```json\n{"a": 1}\n```\n```json\n{"a": 2}\n```'))


def test_host_memory_guard_and_allocator_setting(monkeypatch):
    import os
    from web_agent.eval.task2 import model_worker
    assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == 'expandable_segments:True'
    monkeypatch.setattr(model_worker, 'host_available_gb', lambda: 15.9)
    with pytest.raises(RuntimeError, match='HOST_MEMORY_GUARD'):
        model_worker.require_host_memory()
    monkeypatch.setattr(model_worker, 'host_available_gb', lambda: 40.0)
    model_worker.require_host_memory()
    assert model_worker.host_available_gb.__module__ or True
