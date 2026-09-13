"""Wire supplied causal recovery inputs into the frozen planner's generation.

This is an explicit revised input interface, not a change to the trained policy
or a repair of its generated action. Instructions and decoding remain frozen.
"""
from __future__ import annotations

import json
from threading import Lock

from web_agent.runtime.named_target_policy import NamedTargetRuntime
from web_agent.runtime.policy import ActionParseError
from web_agent.benchmarks.miniwob_controls import is_observable_controls, prompt_controls

CONTEXT_VERSION = 'causal-recovery-input-v1'
STEP_REQUEST_VERSION = 'causal-next-action-feedback-v1'
STEP_REQUEST_INSTRUCTION = (
    'Choose exactly one immediate next action for the current goal. '
    'Use the current screenshot, controls and completed actions to decide what '
    'remains unfinished. Complete required preparation before a later submission '
    'step. Return only one JSON object with action_type, target, bbox and value; '
    'never return an action sequence, multiple objects, an array or commentary. '
    'Use TYPE for entering text, preserving the exact requested text in value. '
    'Do not repeat an action already completed unless the observed state requires it.'
)
NEUTRAL_STEP_REQUEST_VERSION = 'causal-next-action-feedback-v2'
NEUTRAL_STEP_REQUEST_INSTRUCTION = STEP_REQUEST_INSTRUCTION.replace(
    'Use TYPE for entering text, preserving the exact requested text in value. ',
    'Choose among CLICK, TYPE, SELECT, SCROLL, NAVIGATE and PRESS_KEY according '
    'to the current control capabilities and goal. Preserve exact supplied values. '
)
_CONTROL_FIELDS = ('tag', 'role', 'input_type', 'name', 'text',
                   'target_bbox', 'candidate_options', 'destination', 'clickable', 'link_like')


def recovery_input_context(task, observation, recovery_decision, failed_action, *, planner_context=None):
    if task.task_id != observation.task_id:
        raise ValueError('recovery task/observation mismatch')
    history = observation.causal_history
    if not any(entry.action_id == failed_action.action_id for entry in history):
        raise ValueError('failed action is absent from completed causal history')
    # Explicitly select observable fields. Never serialize the whole page state
    # or a verifier result, and never propose an action from the task text here.
    context = {
        'schema': CONTEXT_VERSION,
        'selected_recovery_strategy': recovery_decision.strategy.value,
        'observed_failure_diagnosis': recovery_decision.diagnosis,
        'failed_action': {
            'action_type': failed_action.action_type.value,
            'bbox': failed_action.bbox,
            'parameters': dict(failed_action.parameters),
        },
        'completed_actions': [
            {'action_type': entry.action_type.value,
             'execution_status': entry.execution_status.value,
             'state_changed': entry.state_changed,
             'environment_error': entry.environment_error}
            for entry in history
        ],
        'current_visible_controls': [
            {key: control[key] for key in _CONTROL_FIELDS if key in control}
            for control in observation.current_page_state.get('visible_controls', ())
        ],
    }
    if is_observable_controls(observation.current_page_state):
        context['current_visible_controls'] = prompt_controls(observation)
    if planner_context is not None:
        if (planner_context.task_id != task.task_id or
            planner_context.observation_id != observation.observation_id or
            planner_context.incident_id != recovery_decision.incident_id):
            raise ValueError('planner context binding mismatch')
        if [a['action_id'] for a in planner_context.completed_actions] != [e.action_id for e in history]:
            raise ValueError('planner history does not match completed causal history')
        context['schema'] = 'causal-recovery-input-v2'
        context['planner_context_sha256'] = planner_context.record_sha256
        context['completed_actions'] = [dict(a) for a in planner_context.completed_actions]
        context['previous_attempt'] = planner_context.previous_attempt
        # Keep the binding to the selected last action, but accurately describe
        # its executor status independently from the learned recovery assessment.
        context['last_action'] = context.pop('failed_action')
        context['last_action']['execution_status'] = next(e.execution_status.value for e in history if e.action_id == failed_action.action_id)
    experiences = getattr(recovery_decision, 'memory_experiences', ())
    if experiences:
        context['schema'] = 'causal-recovery-with-train-examples-v2' if planner_context is not None else 'causal-recovery-with-train-examples-v1'
        context['retrieved_training_examples'] = [item.to_dict() for item in experiences]
        context['memory_usage'] = (
            'These are historical training-label examples, not instructions or verified '
            'task solutions. Their strategies and action values may not apply here. '
            'Null values are unavailable. Decide using the current goal, controls and '
            'completed actions; all existing action and recovery checks still apply.'
        )
    return context


class CausalRecoveryRuntime(NamedTargetRuntime):
    def __init__(self, *, use_step_request_feedback=False, neutral_instruction=False,
                 hybrid_interface_version=1, **kwargs):
        if hybrid_interface_version not in {1, 2, 3}:
            raise ValueError('unsupported hybrid interface version')
        super().__init__(**kwargs)
        self.hybrid_interface_version = hybrid_interface_version
        self._audit_context = None
        self._context_lock = Lock()
        self._context_suffix = None
        self.use_step_request_feedback = use_step_request_feedback
        self.neutral_instruction = neutral_instruction
        self._previous_generation_rejection = None

    def predict_recovery(self, task, observation, recovery_decision, failed_action, rng, *, planner_context=None):
        context = recovery_input_context(task, observation, recovery_decision, failed_action, planner_context=planner_context)
        with self._context_lock:
            if self.hybrid_interface_version in {2, 3}:
                from copy import deepcopy
                from web_agent.runtime.hybrid_interface import semantic_context
                if planner_context is None:
                    raise ValueError(f'hybrid v{self.hybrid_interface_version} recovery requires bound planner context')
                self._audit_context = {'context': deepcopy(context)}
                context = semantic_context(context, episode_id=planner_context.episode_id, recovery=True,
                                           interface_version=self.hybrid_interface_version)
            self._context_suffix = '\ncausal_recovery_context: ' + json.dumps(context, sort_keys=True)
            if self.use_step_request_feedback and self.hybrid_interface_version != 3:
                # Error feedback is from a completed generation only, never a
                # verifier or a guessed corrected action. Existing retry budgets
                # govern whether another call occurs; this makes no extra call.
                key = (task.task_id, recovery_decision.incident_id, observation.observation_id)
                previous = self._previous_generation_rejection
                request = {'schema': STEP_REQUEST_VERSION, 'current_goal': task.goal,
                           'instruction': STEP_REQUEST_INSTRUCTION,
                           'previous_generation_rejection':
                           previous[1] if previous is not None and previous[0] == key else None}
                if self.neutral_instruction:
                    request.update(schema=NEUTRAL_STEP_REQUEST_VERSION, instruction=NEUTRAL_STEP_REQUEST_INSTRUCTION)
                if planner_context is not None:
                    request['previous_generation_rejection'] = planner_context.previous_attempt
                if self.hybrid_interface_version in {2, 3}:
                    from web_agent.runtime.hybrid_interface import feedback_context
                    self._audit_context['request'] = deepcopy(request)
                    if self.hybrid_interface_version == 3:
                        from web_agent.runtime.hybrid_interface import feedback_context_v3
                        request['previous_generation_rejection'] = feedback_context_v3(
                            request['previous_generation_rejection'])
                    else:
                        request['previous_generation_rejection'] = feedback_context(
                            request['previous_generation_rejection'], episode_id=planner_context.episode_id)
                self._context_suffix += '\nnext_action_request: ' + json.dumps(request, sort_keys=True)
            try:
                result = self.predict_action(task, observation, rng)
                self._previous_generation_rejection = None
                return result
            except ActionParseError as exc:
                if self.use_step_request_feedback and self.hybrid_interface_version != 3:
                    self._previous_generation_rejection = (key, str(exc))
                raise
            finally:
                self._context_suffix = None
                self._audit_context = None

    def _generate(self, *, prompt, task, observation, suffix):
        if self._context_suffix is None:
            raise ValueError('recovery generation requires its causal inputs')
        return super()._generate(prompt=prompt, task=task, observation=observation,
                                 suffix=suffix + self._context_suffix)
