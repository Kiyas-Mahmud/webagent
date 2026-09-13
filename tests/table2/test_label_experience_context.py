"""Synthetic interface checks, not evidence of live memory benefit."""
from dataclasses import replace
import hashlib
import json
from threading import Lock
from types import SimpleNamespace as NS

import pytest
import torch

from web_agent.memory.label_backed_store import LabelMemoryHit
from web_agent.memory.label_experience import LabelExperienceMaterial
from web_agent.memory.label_runtime import LabelBackedMemoryAdapter
from web_agent.runtime.contracts import ActionType, RecoveryDecision, RecoveryStrategy
from web_agent.runtime.causal_recovery_planner import recovery_input_context
from web_agent.runtime.memory_adapter import MemoryBoundaryError


def setup(tmp_path, *, split='train'):
    rows = [dict(memory_id=f'memory-{i}', source_split=split,
                 evidence_basis='training_dataset_labels', reward='must-not-leak',
                 material=dict(canonical_task_id=f'train-{i}', failure_type='ERROR',
                               failed_action='NAVIGATE', executed_recovery_action=action,
                               recovery_strategy=strategy, recovery_action_value='',
                               reflection_text='', final_task_success='must-not-leak'))
            for i, (action, strategy) in enumerate([
                ('NAVIGATE', 'BACKTRACK'), ('CLICK', 'ALTERNATIVE_TARGET'), ('TYPE', 'RETRY')])]
    path = tmp_path / 'material.jsonl'
    raw = '\n'.join(json.dumps(row) for row in rows).encode()
    path.write_bytes(raw)
    hits = tuple(LabelMemoryHit(row['memory_id'], row['material']['canonical_task_id'],
                                'ERROR', row['material']['recovery_strategy'], score, admitted)
                 for row, score, admitted in zip(rows, [.95, .9, .5], [True, True, False]))
    store = NS(memory_input_sha256=hashlib.sha256(raw).hexdigest(),
               manifest_sha256='a'*64, checkpoint_sha256='b'*64,
               query=lambda *args, **kwargs: hits)
    return store, path, hits


def test_material_is_hash_bound_train_only_and_does_not_fabricate_parameters(tmp_path):
    store, path, hits = setup(tmp_path)
    material = LabelExperienceMaterial(path, store=store)
    item = material.for_hit(hits[0])
    assert item.recovery_action_value is None and item.reflection is None
    assert item.recovery_action is ActionType.NAVIGATE
    assert 'must-not-leak' not in json.dumps(item.to_dict())
    with pytest.raises(MemoryBoundaryError, match='identity mismatch'):
        material.for_hit(replace(hits[0], source_task='other'))
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(MemoryBoundaryError, match='embedding source'):
        LabelExperienceMaterial(path, store=store)


def test_nontraining_material_rejected_even_when_hash_matches(tmp_path):
    store, path, _ = setup(tmp_path, split='validation')
    with pytest.raises(MemoryBoundaryError, match='train-only'):
        LabelExperienceMaterial(path, store=store)


def test_explicitly_unavailable_failed_action_remains_null(tmp_path):
    store, path, hits = setup(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['material'].update(failed_action='UNAVAILABLE', failed_action_available=False)
    raw = '\n'.join(json.dumps(row) for row in rows).encode()
    path.write_bytes(raw)
    store.memory_input_sha256 = hashlib.sha256(raw).hexdigest()
    assert LabelExperienceMaterial(path, store=store).for_hit(hits[0]).failed_action is None
    rows[0]['material']['failed_action_available'] = True
    raw = '\n'.join(json.dumps(row) for row in rows).encode()
    path.write_bytes(raw)
    store.memory_input_sha256 = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match='valid ActionType'):
        LabelExperienceMaterial(path, store=store)


def test_context_exposes_threshold_results_without_forcing_strategy_or_action(tmp_path):
    store, path, hits = setup(tmp_path)
    material = LabelExperienceMaterial(path, store=store)
    runtime = NS(lock=Lock(), torch=torch, batch=NS(transition=lambda *args: {}),
                 model=NS(memory_embedding=lambda batch: torch.ones((1, 768))))
    adapter = LabelBackedMemoryAdapter(store=store, runtime=runtime, transitions={},
                                      recovery_context_material=material)
    shadow = RecoveryDecision(decision_id='shadow', incident_id='incident',
                              strategy=RecoveryStrategy.REPLAN, diagnosis='ERROR',
                              trigger_sources=('policy',))
    query = NS(incident_id='incident', query_id='query', post_failure_observation_id='post',
               failed_action_id='failed', post_action_input_sha256='c'*64,
               task_id='eval-task', record_sha256='d'*64)
    action = NS(action_id='failed', action_type=ActionType.NAVIGATE, bbox=None, parameters={})
    transition = NS(task_id='eval-task', post_observation=NS(goal='goal'),
                    executed_action=action, record_sha256='c'*64)
    embedding = NS(query_id='query', post_failure_observation_id='post', failed_action_id='failed',
                   checkpoint_sha256='b'*64, post_action_input_sha256='c'*64,
                   post_action_input=transition)
    decision = adapter.apply_post_failure(query=query, shadow_decision=shadow, rng=None,
                                         embedding_request=embedding)
    result = decision.query_result
    assert result.context_candidate_ids == ('memory-0', 'memory-1')
    assert result.intervened and result.changed_recovery_context
    assert not result.changed_strategy and not result.changed_target_or_parameters
    assert decision.final_decision.strategy is RecoveryStrategy.REPLAN
    assert decision.final_decision.planned_action is None
    assert shadow.memory_experiences == ()
    # The original BACKTRACK example is exposed honestly, never cherry-picked away.
    assert decision.final_decision.memory_experiences[0].recorded_strategy is RecoveryStrategy.BACKTRACK
    history = (NS(action_id='failed', action_type=ActionType.NAVIGATE,
                  execution_status=NS(value='rejected'), state_changed=False, environment_error=False),)
    obs = NS(task_id='eval-task', causal_history=history, current_page_state={})
    task = NS(task_id='eval-task')
    context = recovery_input_context(task, obs, decision.final_decision, action)
    assert len(context['retrieved_training_examples']) == 2
    assert 'retrieved_training_examples' not in recovery_input_context(task, obs, shadow, action)
    assert context['observed_failure_diagnosis'] == shadow.diagnosis
    with pytest.raises(ValueError, match='flag and candidate IDs'):
        replace(result, changed_recovery_context=False)
    with pytest.raises(MemoryBoundaryError, match='separate experiment modes'):
        LabelBackedMemoryAdapter(store=store, runtime=runtime, transitions={},
                                 recovery_context_material=material, require_strategy_applicability=True)
    store.query = lambda *args, **kwargs: tuple(replace(hit, admitted=False) for hit in hits)
    abstained = adapter.apply_post_failure(query=query, shadow_decision=shadow, rng=None,
                                          embedding_request=embedding)
    assert abstained.final_decision is shadow
    assert not abstained.query_result.intervened
