from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from web_agent.eval.task1.core import CHECKPOINT_HASH, file_hash
from web_agent.eval.task2.assessment import (
    Observation, ExecutedTransition, InternVLTransitionAssessor,
    assess_for_system, recovery_advice,
)
from web_agent.eval.task2.memory_contract import verify_memory_query_identity


@pytest.fixture
def transition(tmp_path):
    from PIL import Image
    (tmp_path / 'images').mkdir()
    for name in ('before.png', 'after.png'):
        Image.new('RGB', (8, 8)).save(tmp_path / 'images' / name)
    def observation(name, seq):
        image = 'images/' + name + '.png'
        return Observation('episode', name, seq, image, file_hash(tmp_path / image))
    return ExecutedTransition('episode', 'task', 'action-1', 'Enter the supplied text',
                              'localhost', observation('before', 0), observation('after', 1),
                              'TYPE', 'textbox:1', 'Exact Text!')


class Backend:
    def __init__(self, root, recovery_logit=0.):
        self.root = root
        self.torch = torch
        self.recovery_logit = recovery_logit
        self.requests = []

    def batch(self, request):
        self.requests.append(request)
        return request

    def model(self, batch):
        return {'outcome': torch.tensor([[0., 1.]]),
                'failure_type': torch.tensor([[0., 0., 1., 0.]]),
                'recovery': torch.tensor([[0., 0., 1., 0., 0., 0.]]),
                'recovery_outcome': torch.tensor([[self.recovery_logit]])}


def test_preserves_exact_values_outside_unchanged_head_input(transition, tmp_path):
    backend = Backend(tmp_path)
    result = InternVLTransitionAssessor(backend).assess(transition)
    assert result['signals'] == {'outcome_label': 'FAILURE', 'failure_type': 'ACTION_MISMATCH', 'recovery_strategy': 'REPLAN'}
    assert result['actor_context']['value'] == 'Exact Text!'
    assert result['actor_context']['target'] == 'textbox:1'
    assert backend.requests[0]['executed_action'] == {'type': 'TYPE', 'value': None}
    assert result['model_calls'] == 1


@pytest.mark.parametrize('action', ['CLICK', 'TYPE', 'SELECT', 'SCROLL', 'NAVIGATE', 'PRESS_KEY'])
def test_all_six_classes_remain_available(transition, tmp_path, action):
    assert replace(transition, action_type=action).head_request(tmp_path)['executed_action']['type'] == action


@pytest.mark.parametrize('change', [
    {'execution_status': 'rejected'}, {'phase': 'recovery_assessment'},
    {'action_type': 'AUTO_TYPE'}, {'value': {'answer': 'secret'}},
])
def test_invalid_transition_rejected_before_backend(transition, tmp_path, change):
    backend = Backend(tmp_path)
    with pytest.raises(ValueError):
        InternVLTransitionAssessor(backend).assess(replace(transition, **change))
    assert backend.requests == []


@pytest.mark.parametrize('after_change', [{'episode_id': 'different'}, {'sequence': 3}, {'observation_id': 'before'}])
def test_temporal_and_episode_binding(transition, tmp_path, after_change):
    with pytest.raises(ValueError):
        replace(transition, after=replace(transition.after, **after_change)).head_request(tmp_path)


def test_changed_image_rejected(transition, tmp_path):
    (tmp_path / transition.after.image).write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        transition.head_request(tmp_path)


def test_labels_cannot_be_added_to_transition(transition):
    with pytest.raises(TypeError):
        replace(transition, reward=1.0)


@pytest.mark.parametrize('logit,expected', [(-1., 'FAILURE'), (0., 'FAILURE'), (1., 'SUCCESS')])
def test_recovery_uses_original_threshold_and_phase(transition, tmp_path, logit, expected):
    t = replace(transition, phase='recovery_assessment', incident_id='incident-1')
    b = Backend(tmp_path, logit)
    result = InternVLTransitionAssessor(b).assess(t)
    assert result['signals'] == {'outcome_label': expected}
    assert set(result['raw_logits']) == {'recovery_outcome'}
    assert b.requests[0]['phase'] == 'recovery_assessment'


def test_baseline_never_touches_assessor_and_c_fails_closed(transition):
    assert assess_for_system('A', transition, None) is None
    with pytest.raises(RuntimeError, match='C_NOT_READY'):
        assess_for_system('C', transition, None)


def test_advice_is_bound_and_never_replaces_action(transition, tmp_path):
    result = InternVLTransitionAssessor(Backend(tmp_path)).assess(transition)
    advice = recovery_advice(transition, result)
    assert advice['observed_action']['value'] == transition.value
    assert 'next_action' not in advice
    assert recovery_advice(replace(transition, terminated=True), result) is None
    assert recovery_advice(replace(transition, truncated=True), result) is None
    with pytest.raises(ValueError, match='another transition'):
        recovery_advice(replace(transition, action_id='future-action'), result)


def test_nonfinite_model_output_rejected(transition, tmp_path):
    b = Backend(tmp_path)
    b.model = lambda _: {'outcome': torch.tensor([[float('nan'), 0.]])}
    with pytest.raises(ValueError, match='head output'):
        InternVLTransitionAssessor(b).assess(transition)


def test_memory_rejects_equal_dimension_different_checkpoint(tmp_path):
    artifact = tmp_path / 'embeddings.bin'
    artifact.write_bytes(b'frozen fixture')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'checkpoint_sha256': 'qwen-checkpoint', 'dimension': 768,
                                   'count': 1974, 'admission_threshold': .737,
                                   'files': {artifact.name: file_hash(artifact)}}))
    args = dict(expected_manifest_sha256=file_hash(manifest), query_dimension=768)
    with pytest.raises(ValueError, match='MEMORY_EMBEDDING_SPACE_MISMATCH'):
        verify_memory_query_identity(tmp_path, query_checkpoint_sha256=CHECKPOINT_HASH, **args)
    before = artifact.read_bytes()
    result = verify_memory_query_identity(tmp_path, query_checkpoint_sha256='qwen-checkpoint', **args)
    assert result['write_enabled'] is False and result['live_query_parity_verified'] is False
    assert artifact.read_bytes() == before
    artifact.write_bytes(b'mutated')
    with pytest.raises(ValueError, match='artifact identity changed'):
        verify_memory_query_identity(tmp_path, query_checkpoint_sha256='qwen-checkpoint', **args)
