"""Causal bridge for the existing InternVL heads; never selects browser actions.

This module is an integration boundary, not a native-agent runner. Live evidence
retains exact action arguments while head tensorization stays unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from web_agent.eval.task1.core import CHECKPOINT_HASH, file_hash, validate_request
from web_agent.labels import ACTION_TYPE, EXECUTION_OUTCOME_INV, FAILURE_TYPE_INV, RECOVERY_STRATEGY_INV


@dataclass(frozen=True)
class Observation:
    episode_id: str
    observation_id: str
    sequence: int
    image: str
    sha256: str

    def validate(self, image_root: Path) -> None:
        if not self.episode_id or not self.observation_id:
            raise ValueError('Observation identity is required')
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError('Invalid observation sequence')
        image = Path(self.image)
        if image.is_absolute() or '..' in image.parts or not image.parts or image.parts[0] != 'images':
            raise ValueError('Expected a relative images/ path')
        path = (image_root / image).resolve()
        if not path.is_relative_to(image_root.resolve()):
            raise ValueError('Image escaped evidence root')
        if not re.fullmatch(r'[0-9a-f]{64}', self.sha256) or file_hash(path) != self.sha256:
            raise ValueError('Observation image hash mismatch')


@dataclass(frozen=True)
class ExecutedTransition:
    episode_id: str
    task_id: str
    action_id: str
    task_description: str
    website_domain: str
    before: Observation
    after: Observation
    action_type: str
    target: str | None
    value: str | None
    phase: str = 'interaction_assessment'
    incident_id: str | None = None
    execution_status: str = 'executed'
    terminated: bool = False
    truncated: bool = False

    def validate(self, image_root: Path) -> None:
        for field in ('episode_id', 'task_id', 'action_id', 'task_description', 'website_domain'):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError('Missing live transition field: ' + field)
        if self.execution_status != 'executed':
            raise ValueError('Rejected/unexecuted actions must not be assessed as executed transitions')
        if self.action_type not in ACTION_TYPE:
            raise ValueError('Unsupported action class')
        if self.phase not in ('interaction_assessment', 'recovery_assessment'):
            raise ValueError('Unsupported assessment phase')
        if self.phase == 'recovery_assessment' and not self.incident_id:
            raise ValueError('Recovery must bind an existing incident')
        if self.incident_id is not None and not isinstance(self.incident_id, str):
            raise ValueError('Invalid incident identity')
        if any(v is not None and not isinstance(v, str) for v in (self.target, self.value)):
            raise ValueError('Issued target/value must be exact strings or null')
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError('Invalid terminal state')
        for observation in (self.before, self.after):
            observation.validate(image_root)
            if observation.episode_id != self.episode_id:
                raise ValueError('Cross-episode transition')
        if self.after.sequence != self.before.sequence + 1:
            raise ValueError('Only an adjacent, completed action transition is allowed')
        if self.before.observation_id == self.after.observation_id:
            raise ValueError('Pre/post observation identities must differ')

    def head_request(self, image_root: Path) -> dict:
        self.validate(image_root)
        # Original head processor uses the action TYPE, not its argument/target.
        # Do not expand trained inputs or falsely claim these fields reach it.
        request = dict(case_id=self.action_id, task_id=self.task_id, phase=self.phase,
                       task_description=self.task_description, website_domain=self.website_domain,
                       before_image=self.before.image, after_image=self.after.image,
                       executed_action={'type': self.action_type, 'value': None})
        validate_request(request)
        return request

    def actor_context(self) -> dict:
        # Explicit allowlist: no evaluator reward, answer labels or reference box.
        return dict(episode_id=self.episode_id, action_id=self.action_id,
                    incident_id=self.incident_id, task_description=self.task_description,
                    action_type=self.action_type, target=self.target, value=self.value,
                    before_observation_id=self.before.observation_id,
                    after_observation_id=self.after.observation_id,
                    execution_status=self.execution_status)


class InternVLTransitionAssessor:
    """Uses the Task 1 verified loader/batch contract without changing its source."""

    def __init__(self, backend):
        self.backend = backend

    @classmethod
    def load(cls, config):
        if file_hash(config['checkpoint']) != CHECKPOINT_HASH:
            raise ValueError('Task 2 requires the selected InternVL checkpoint')
        from web_agent.eval.task1.backends import PC03Assessor
        return cls(PC03Assessor(config))

    def assess(self, transition: ExecutedTransition) -> dict:
        request = transition.head_request(Path(self.backend.root))
        torch = self.backend.torch
        batch = self.backend.batch(request)
        with torch.inference_mode():
            predictions = self.backend.model(batch)
        if transition.phase == 'interaction_assessment':
            shapes = {'outcome': (1, 2), 'failure_type': (1, 4), 'recovery': (1, 6)}
        else:
            shapes = {'recovery_outcome': (1, 1)}
        for key, shape in shapes.items():
            if tuple(predictions[key].shape) != shape or not torch.isfinite(predictions[key]).all():
                raise ValueError('Invalid trained head output: ' + key)
        raw = {k: predictions[k].float().cpu().tolist() for k in shapes}
        if transition.phase == 'interaction_assessment':
            outcome = EXECUTION_OUTCOME_INV[int(predictions['outcome'].argmax(-1).item())]
            signals = dict(outcome_label=outcome,
                           failure_type=FAILURE_TYPE_INV[int(predictions['failure_type'].argmax(-1).item())],
                           recovery_strategy=RECOVERY_STRATEGY_INV[int(predictions['recovery'].argmax(-1).item())])
        else:
            # Exactly the original Task 1 >0 logit threshold, including the tie.
            signals = {'outcome_label': 'SUCCESS' if predictions['recovery_outcome'].item() > 0 else 'FAILURE'}
        return dict(schema='task2.assessment.v1', phase=transition.phase,
                    episode_id=transition.episode_id, action_id=transition.action_id,
                    incident_id=transition.incident_id, checkpoint_sha256=CHECKPOINT_HASH,
                    signals=signals, raw_logits=raw, model_calls=1,
                    head_input_fields=['task_description', 'website_domain', 'before_image', 'after_image', 'action_type'],
                    actor_context=transition.actor_context())


class Qwen25TransitionAssessor(InternVLTransitionAssessor):
    """Selected PC-02 Qwen2.5-VL-7B heads through the verified Task 1 batch contract."""

    @classmethod
    def load(cls, config):
        from web_agent.eval.task1.qwen25_backends import QWEN25_CHECKPOINT_HASH, Qwen25Assessor
        if file_hash(config['checkpoint']) != QWEN25_CHECKPOINT_HASH:
            raise ValueError('Task 2 requires the selected Qwen2.5 checkpoint')
        return cls(Qwen25Assessor(config))

    def assess(self, transition: ExecutedTransition) -> dict:
        from web_agent.eval.task1.qwen25_backends import QWEN25_CHECKPOINT_HASH
        result = super().assess(transition)
        result['checkpoint_sha256'] = QWEN25_CHECKPOINT_HASH
        if transition.phase == 'interaction_assessment':
            # Logged only; the decision rule stays the unchanged argmax.
            torch = self.backend.torch
            probability = torch.softmax(torch.tensor(result['raw_logits']['outcome']), -1)[0]
            result['outcome_probabilities'] = {label: float(probability[index])
                                               for index, label in EXECUTION_OUTCOME_INV.items()}
        return result


def assess_for_system(system, transition, assessor, *, memory_ready=False):
    """A never invokes our module. C is fail-closed until retrieval is integrated."""
    if system not in ('A', 'B', 'C'):
        raise ValueError('Unknown Task 2 system')
    if system == 'A':
        return None
    if system == 'C' and not memory_ready:
        raise RuntimeError('C_NOT_READY: compatible frozen retrieval must be wired first')
    return assessor.assess(transition)


FAILURE_MEANING = {
    'NONE': 'no specific failure type was identified',
    'PERCEPTION_ERROR': 'the action targeted the wrong element or misread the page',
    'ACTION_MISMATCH': 'the action did not have its intended effect on the page',
    'LOOP_DETECTED': 'the same action is being repeated without progress',
}
STRATEGY_MEANING = {
    'NONE': 'no recovery suggested',
    'RETRY': 'repeat the same action once more',
    'REPLAN': 'choose a different next action toward the goal',
    'BACKTRACK': 'go back to the previous page or state',
    'ALTERNATIVE_TARGET': 'do the same kind of action on a different element',
    'ABORT': 'stop; the task may not be achievable',
}


def describe_action(action_type, element=None, value=None):
    """Plain description of an executed action, e.g. "type 'serendipity' into <input> 'Search'"."""
    verb = {'CLICK': 'click', 'TYPE': 'type', 'SCROLL': 'scroll', 'SELECT': 'select',
            'NAVIGATE': 'navigate', 'PRESS_KEY': 'press'}.get(action_type, str(action_type).lower())
    if action_type == 'TYPE':
        verb += f" '{value or ''}' into"
    elif action_type == 'SCROLL':
        verb += ' up' if value and '"down": false' in value else ' down'
    elif action_type in ('NAVIGATE', 'PRESS_KEY') and value:
        verb += ' ' + str(value)
    elif action_type == 'SELECT' and value:
        verb += f" '{value}' in"
    if element:
        attributes = element.get('attributes') or {}
        label = element.get('text') or next((attributes[k] for k in ('aria-label', 'placeholder', 'title', 'name', 'id')
                                             if attributes.get(k)), '')
        verb += f" <{element.get('tag') or 'element'}>" + (f" '{label[:60]}'" if label else '')
    return verb


def recovery_note(transition, assessment, *, element=None, page_changed=None, facts=None, memory=(), succeeded=()):
    """v3 (2026-10-01): what the actor sees after a confident FAILURE, in plain text.

    web-v2 showed the actor ignored JSON advice and that the failure-type and
    strategy heads (macro-F1 0.54 / 0.49) misdirected it. The note keeps only the
    outcome head's verdict and observable page facts; labels are logged, not shown.
    It never selects an action.
    """
    if (assessment['episode_id'], assessment['action_id'], assessment['incident_id'], assessment['phase']) != (
            transition.episode_id, transition.action_id, transition.incident_id, transition.phase):
        raise ValueError('Assessment belongs to another transition')
    if transition.terminated or transition.truncated:
        return None
    p = assessment.get('outcome_probabilities', {}).get('FAILURE')
    did = describe_action(transition.action_type, element, transition.value)
    lines = [f"FAILURE MONITOR: your last action ({did}) did not make progress"
             + (f" (learned confidence {p:.2f})" if p is not None else '') + '.']
    if page_changed is False:
        lines.append('- The page did not change after it.')
    lines.append('- Repeating exactly that action will be blocked.')
    facts = facts or {}
    scroll = facts.get('scroll') or {}
    if transition.action_type == 'SCROLL' and scroll.get('at_bottom'):
        lines.append('- You are at the bottom of the page: there is nothing further down.')
    elif transition.action_type == 'SCROLL' and scroll.get('at_top'):
        lines.append('- You are at the top of the page.')
    if transition.action_type == 'TYPE' and not (transition.value or '').strip():
        lines.append('- The text you typed was empty.')
    field = facts.get('field')
    if field and field.get('value') and page_changed is False:
        lines.append(f"- The field '{field.get('label') or 'text field'}' contains '{field['value']}' but nothing was "
                     'submitted. A form is submitted with the Enter key'
                     + (f" or its '{field['submit']}' button" if field.get('submit') else '') + '.')
    links = facts.get('goal_links') or []
    if links:
        lines.append('- Links on this page whose words match the task (hidden = inside a closed menu; '
                     'any of them can be opened with navigate):')
        lines += [f"  * '{l['text'] or l['href']}' -> {l['href']}" + ('' if l.get('visible') else ' (hidden)') for l in links]
    for box in facts.get('search_boxes') or []:
        lines.append(f"- This page has a search box" + (f" ('{box['label']}')" if box.get('label') else '') + '.')
    for past in memory:
        tried = '; '.join(f"{describe_action(a['action_type'], a.get('element'), a.get('value'))} -> {a.get('assessed_outcome')}"
                          for a in past.get('recovery_attempts') or []) or 'nothing'
        failed = past.get('failed_action') or {}
        lines.append(f"- Earlier on this page: {describe_action(failed.get('action_type'), failed.get('element'), failed.get('value'))} "
                     f"failed; then tried: {tried} ({past.get('incident_outcome')}).")
    if succeeded:
        lines.append('- Already done successfully, do not redo: '
                     + '; '.join(describe_action(s['action_type'], s.get('element'), s.get('value')) for s in succeeded) + '.')
    lines.append('Choose your next action yourself from the current page.')
    return dict(schema='task2.recovery_note.v3', text='\n'.join(lines), p_failure=p,
                logged_signals=dict(assessment['signals']), facts=facts, memory=list(memory))


def recovery_advice(transition, assessment, *, failed_element=None, page_changed=None, succeeded=()):
    """Returns advisory signals for the next actor call, never a replacement action.

    v2 (2026-09-29): in the pilot a vague "FAILURE / REPLAN" made the actor redo
    a step that had already worked (it retyped "serendipity" into the same box,
    producing "serendipityserendipity"). The advice now names the failed step in
    observable terms, lists the steps P1 already judged successful, and explains
    the labels. It still never selects an action.
    """
    if (assessment['episode_id'], assessment['action_id'], assessment['incident_id'], assessment['phase']) != (
            transition.episode_id, transition.action_id, transition.incident_id, transition.phase):
        raise ValueError('Assessment belongs to another transition')
    if transition.terminated or transition.truncated:
        return None
    signals = dict(assessment['signals'])
    failure, strategy = signals.get('failure_type'), signals.get('recovery_strategy')
    return dict(schema='task2.advisory.v2', observed_action=transition.actor_context(),
                learned_assessment=signals,
                failed_step={'action_type': transition.action_type, 'element': failed_element,
                             'value': transition.value, 'page_changed': page_changed},
                diagnosis={'failure_type': failure, 'meaning': FAILURE_MEANING.get(failure),
                           'suggested_strategy': strategy, 'strategy_meaning': STRATEGY_MEANING.get(strategy)},
                already_succeeded=list(succeeded),
                instruction='The learned assessment refers only to failed_step, the most recent action. '
                            'Steps in already_succeeded were judged successful: do not redo them (for example, '
                            'do not type the same text again) unless the current page shows they were undone. '
                            'Choose your own next action, target and exact value from the current page. '
                            'A learned outcome is advisory evidence, not task completion.')
