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


def assess_for_system(system, transition, assessor, *, memory_ready=False):
    """A never invokes our module. C is fail-closed until retrieval is integrated."""
    if system not in ('A', 'B', 'C'):
        raise ValueError('Unknown Task 2 system')
    if system == 'A':
        return None
    if system == 'C' and not memory_ready:
        raise RuntimeError('C_NOT_READY: compatible frozen retrieval must be wired first')
    return assessor.assess(transition)


def recovery_advice(transition, assessment):
    """Returns advisory signals for the next actor call, never a replacement action.

    The future runner owns budgets/incident state and must persist both this
    advice and its actual inclusion in generation. Returning advice alone is not
    evidence of an executed recovery or integrated Browser Use extension.
    """
    if (assessment['episode_id'], assessment['action_id'], assessment['incident_id'], assessment['phase']) != (
            transition.episode_id, transition.action_id, transition.incident_id, transition.phase):
        raise ValueError('Assessment belongs to another transition')
    if transition.terminated or transition.truncated:
        return None
    return dict(schema='task2.advisory.v1', observed_action=transition.actor_context(),
                learned_assessment=dict(assessment['signals']),
                instruction='Use these learned signals as advisory evidence. Choose your own next action, target and exact value from the current observation. A learned outcome is not task completion.')
