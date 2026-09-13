"""Causal planner wiring checks, not model or live completion results."""
import json
from types import SimpleNamespace as NS

import pytest

from web_agent.runtime.causal_recovery_planner import CausalRecoveryRuntime, recovery_input_context
from web_agent.runtime.contracts import ActionType, ExecutionStatus, RecoveryStrategy
from web_agent.runtime.named_target_policy import NamedTargetRuntime
from web_agent.runtime.policy import ActionParseError


def inputs(strategy=RecoveryStrategy.REPLAN):
    task = NS(task_id='task')
    history = (NS(action_id='failed', action_type=ActionType.NAVIGATE,
                  execution_status=ExecutionStatus.REJECTED, state_changed=False,
                  environment_error=False),)
    observation = NS(task_id='task', causal_history=history,
                     current_page_state={'visible_controls': [
                         {'tag': 'input', 'text': '', 'target_bbox': [.1,.2,.3,.1],
                          'oracle_success': 'must-not-leak'}], 'reward': 1})
    decision = NS(strategy=strategy, diagnosis='parameter_resolution_rejected')
    action = NS(action_id='failed', action_type=ActionType.NAVIGATE, bbox=None, parameters={})
    return task, observation, decision, action


def test_supplied_diagnosis_strategy_history_and_controls_reach_context():
    args = inputs()
    context = recovery_input_context(*args)
    assert context['selected_recovery_strategy'] == 'REPLAN'
    assert context['observed_failure_diagnosis'] == args[2].diagnosis
    assert context['failed_action']['action_type'] == 'NAVIGATE'
    assert context['completed_actions'][0]['execution_status'] == 'rejected'
    assert context['current_visible_controls'][0]['target_bbox'] == [.1,.2,.3,.1]
    assert 'oracle_success' not in json.dumps(context) and 'reward' not in json.dumps(context)
    assert recovery_input_context(*inputs(RecoveryStrategy.ALTERNATIVE_TARGET))['selected_recovery_strategy'] == 'ALTERNATIVE_TARGET'


def test_stale_or_cross_task_context_is_rejected():
    task, obs, decision, action = inputs()
    task.task_id = 'other'
    with pytest.raises(ValueError, match='mismatch'):
        recovery_input_context(task, obs, decision, action)
    task.task_id = 'task'; action.action_id = 'future'
    with pytest.raises(ValueError, match='absent'):
        recovery_input_context(task, obs, decision, action)


def test_generation_receives_context_and_clears_it_even_on_rejection(monkeypatch, tmp_path):
    # Exercise the actual predict_recovery -> predict_action -> _generate chain
    # while replacing only model generation; inspect the suffix at its boundary.
    def init(self, **kwargs): pass
    monkeypatch.setattr(NamedTargetRuntime, '__init__', init)
    seen = []
    def generate(self, **kwargs):
        seen.append(kwargs['suffix'])
        raise ValueError('model-output-rejected')
    monkeypatch.setattr(NamedTargetRuntime, '_generate', generate)
    runtime = CausalRecoveryRuntime()
    for strategy in (RecoveryStrategy.REPLAN, RecoveryStrategy.ALTERNATIVE_TARGET):
        with pytest.raises(ValueError, match='model-output-rejected'):
            runtime.predict_recovery(*inputs(strategy), rng=None)
        assert runtime._context_suffix is None
    assert 'REPLAN' in seen[0] and 'ALTERNATIVE_TARGET' in seen[1]
    assert 'ALTERNATIVE_TARGET' not in seen[0]
    with pytest.raises(ValueError, match='requires its causal inputs'):
        runtime.predict_action(inputs()[0], inputs()[1], None)


def feedback_inputs():
    task, obs, decision, action = inputs()
    task.goal = 'Enter a requested value, then submit.'
    obs.observation_id = 'post-failure'
    decision.incident_id = 'incident'
    return task, obs, decision, action


def test_existing_retry_receives_generation_error_and_success_clears_it(monkeypatch):
    monkeypatch.setattr(NamedTargetRuntime, '__init__', lambda *args, **kwargs: None)
    seen = []
    def predict(self, *args):
        seen.append(self._context_suffix)
        if len(seen) == 1:
            raise ActionParseError('not one strict JSON object')
        return 'model-decision'
    monkeypatch.setattr(NamedTargetRuntime, 'predict_action', predict)
    runtime = CausalRecoveryRuntime(use_step_request_feedback=True)
    args = feedback_inputs()
    with pytest.raises(ActionParseError):
        runtime.predict_recovery(*args, rng=None)
    assert runtime._context_suffix is None
    assert runtime.predict_recovery(*args, rng=None) == 'model-decision'
    runtime.predict_recovery(*args, rng=None)
    requests = [json.loads(s.split('\nnext_action_request: ')[1]) for s in seen]
    assert [r['previous_generation_rejection'] for r in requests] == [
        None, 'not one strict JSON object', None]
    assert all(r['current_goal'] == args[0].goal for r in requests)
    assert runtime._previous_generation_rejection is None
    assert runtime._context_suffix is None
    assert len(seen) == 3  # No hidden generation retry inside one call.


@pytest.mark.parametrize('changed', ['task', 'incident', 'observation'])
def test_feedback_does_not_leak_to_a_different_failure_context(monkeypatch, changed):
    monkeypatch.setattr(NamedTargetRuntime, '__init__', lambda *args, **kwargs: None)
    seen = []
    def predict(self, *args):
        seen.append(json.loads(self._context_suffix.split('\nnext_action_request: ')[1]))
        raise ActionParseError('invalid previous output')
    monkeypatch.setattr(NamedTargetRuntime, 'predict_action', predict)
    runtime = CausalRecoveryRuntime(use_step_request_feedback=True)
    args = feedback_inputs()
    with pytest.raises(ActionParseError):
        runtime.predict_recovery(*args, rng=None)
    if changed == 'task':
        args[0].task_id = args[1].task_id = 'another-task'
    elif changed == 'incident':
        args[2].incident_id = 'another-incident'
    else:
        args[1].observation_id = 'another-observation'
    with pytest.raises(ActionParseError):
        runtime.predict_recovery(*args, rng=None)
    assert all(r['previous_generation_rejection'] is None for r in seen)
