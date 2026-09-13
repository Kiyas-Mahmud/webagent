"""Explicit visible-name action serialization, separate from the frozen parser.

Preserves action type and resolves only a model-supplied name. It never derives
an action/target from the task goal, outcome, or memory. Named targets take
precedence over approximate generated boxes under this versioned interface.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from collections.abc import Mapping
from web_agent.runtime.policy import ActionParseError
from web_agent.runtime.qwen2vl_pc01 import _UnadaptedBaseRuntime, _strict_json_object, parse_e0_action_output
from web_agent.benchmarks.miniwob_controls import CONTROL_INTERFACE, OBSERVED_CONTROL_INTERFACES, aliases, supported_actions, control_suffix, is_observable_controls

INTERFACE_ID = 'visible-name-action-interface-v2'
FENCED_INTERFACE_ID = 'visible-name-action-interface-v3'
ROLE_INTERFACE_ID = 'visible-description-action-interface-v4'
POINTER_INTERFACE_ID = 'visible-pointer-description-action-interface-v5'


def _observed_interface(observation, default):
    if is_observable_controls(observation.current_page_state):
        return 'visible-control-action-interface-v6'
    controls = observation.current_page_state.get('visible_controls', ())
    return (POINTER_INTERFACE_ID if isinstance(controls, (list, tuple))
            and any(isinstance(c, Mapping) and c.get('clickable') is True for c in controls)
            else default)


def _name(value):
    return ' '.join(value.casefold().split()) if isinstance(value, str) else ''


def _compatible(control, action):
    if control.get('control_interface') in OBSERVED_CONTROL_INTERFACES:
        return action in supported_actions(control)
    tag = control.get('tag')
    if action == 'CLICK':
        return (tag in {'a', 'button', 'input', 'textarea', 'select'}
                or control.get('role') in {'button', 'link'}
                or control.get('clickable') is True)
    if action == 'TYPE':
        return tag == 'textarea' or (tag == 'input' and control.get('input_type') not in {'button', 'checkbox', 'radio', 'submit', 'reset', 'file', 'image', 'hidden'})
    return action == 'SELECT' and tag == 'select'


def _description_matches(control, description, action, click_value):
    """Match a small explicit role grammar, then an exact visible label.

    No fuzzy matching, task-goal lookup or ordinal guessing. A bare role is
    allowed only if the caller subsequently establishes one compatible match.
    """
    tag = control.get('tag')
    roles = {
        'button': tag == 'button' or control.get('role') == 'button' or
                  (tag == 'input' and control.get('input_type') in {'button', 'submit', 'reset'}),
        'link': tag == 'a' or control.get('role') == 'link' or control.get('link_like') is True,
        'text field': _compatible(control, 'TYPE'),
        'textbox': _compatible(control, 'TYPE'),
        'input': tag == 'input',
        'textarea': tag == 'textarea',
        'select': tag == 'select',
        'checkbox': tag == 'input' and control.get('input_type') == 'checkbox',
        'radio button': tag == 'input' and control.get('input_type') == 'radio',
    }
    for role in sorted(roles, key=len, reverse=True):
        if description == role or description.startswith(role + ' '):
            label = description[len(role):].strip()
            if not label and action == 'CLICK':
                label = _name(click_value)
            return roles[role] and (not label or label in aliases(control))
    return False


def _current_id_target(value, controls, observation, action):
    """Resolve only the declared current-ID grammar, with no CSS execution."""
    target = value.get('target')
    if not isinstance(target, str):
        return None
    tagged = re.fullmatch(r'([a-z][a-z0-9-]*)#(o[^\s:#]+:c[0-9]+)', target)
    bare = re.fullmatch(r'o[^\s:#]+:c[0-9]+', target)
    # An exact supplied ID is still an ID if a nonstandard fixture generated it.
    exact = any(isinstance(c, Mapping) and c.get('control_id') == target for c in controls)
    if not (tagged or bare or exact):
        return None
    control_id = tagged.group(2) if tagged else target
    candidates = [c for c in controls if isinstance(c, Mapping) and c.get('control_id') == control_id]
    code = None
    if len(candidates) != 1:
        code = 'CURRENT_CONTROL_ID_NOT_UNIQUE'
    elif candidates[0].get('observation_id') != observation.observation_id:
        code = 'STALE_CONTROL_OBSERVATION'
    elif tagged and candidates[0].get('tag') != tagged.group(1):
        code = 'CURRENT_CONTROL_TAG_MISMATCH'
    elif not _compatible(candidates[0], action):
        code = 'ACTION_TARGET_INCOMPATIBLE'
    if code:
        exc = ActionParseError(f'{code}: action={action}; target={target!r}; current ID matches={len(candidates)}')
        exc.failure_stage = 'target_resolution'
        exc.diagnostic_code = code
        raise exc
    return candidates[0], ('unique_observed_tag_current_control_id' if tagged else 'unique_current_control_id')


def resolve_named_target(raw, observation, *, allow_json_fence=False, allow_role_target=False,
                         allow_tag_control_target=False):
    interface = ROLE_INTERFACE_ID if allow_role_target else (FENCED_INTERFACE_ID if allow_json_fence else INTERFACE_ID)
    interface = _observed_interface(observation, interface)
    envelope = 'bare_json'
    if allow_json_fence:
        match = re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```', raw.strip(), re.DOTALL)
        if match:
            raw = match.group(1)
            envelope = 'single_json_fence'
    value = _strict_json_object(raw, context=interface)
    expected = {'action_type', 'target', 'bbox', 'value'}
    if 'action_type' not in value or set(value) - expected:
        raise ActionParseError('named-target output has unexpected fields')
    defaulted = sorted(expected - set(value))
    value = {'target': None, 'bbox': None, 'value': None, **value}
    for field in ('target', 'value'):
        if value[field] is not None and not isinstance(value[field], str):
            raise ActionParseError(f'named-target {field} must be text or null')
    action = value.get('action_type')
    name = _name(value.get('target'))
    source = 'target'
    # CLICK has no text payload: explicitly support its emitted visible label.
    if not allow_tag_control_target and not name and action == 'CLICK' and value.get('target') is None:
        name = _name(value.get('value'));source = 'click_value'
    if action not in {'CLICK', 'TYPE', 'SELECT'} or not name:
        return parse_e0_action_output(json.dumps(value)), {'interface': interface, 'output_envelope': envelope, 'resolution': 'original_box', 'defaulted_nullable_fields': defaulted}
    controls = observation.current_page_state.get('visible_controls', ())
    if not isinstance(controls, (list, tuple)):
        raise ActionParseError('visible controls are unavailable')
    id_match = _current_id_target(value, controls, observation, action) if allow_tag_control_target else None
    matches = ([id_match[0]] if id_match else [
        c for c in controls if isinstance(c, Mapping) and _compatible(c, action)
        and (name in aliases(c) or (not allow_tag_control_target and name == _name(c.get('control_id')))
             or (allow_role_target and _description_matches(c, name, action, value.get('value'))))])
    if len(matches) != 1:
        exc = ActionParseError(f'model-named target has {len(matches)} exact compatible matches; action={action}; target={value.get("target")!r}')
        exc.failure_stage = 'target_resolution'
        exc.diagnostic_code = 'TARGET_NOT_UNIQUELY_COMPATIBLE'
        raise exc
    control = matches[0]
    if allow_tag_control_target and control.get('control_id') is not None:
        # A name cannot hide duplicate current IDs or an observation mismatch.
        _current_id_target({**value, 'target': control['control_id']}, controls, observation, action)
    if control.get('control_interface') in OBSERVED_CONTROL_INTERFACES and control['observation_id'] != observation.observation_id:
        raise ActionParseError('STALE_CONTROL_OBSERVATION')
    # The action payload keeps the model's own target string. Identity travels
    # with the resolved box, which is this interface's output for a named target
    # and is what downstream resolution matches the control on.
    identity = control.get('control_id') if control.get('control_interface') in OBSERVED_CONTROL_INTERFACES else None
    resolved = {**value, 'target': value['target'] if isinstance(value.get('target'), str) else value['value'],
                'bbox': list(control['target_bbox'])}
    if action == 'CLICK':resolved['value'] = None
    parsed = parse_e0_action_output(json.dumps(resolved))
    resolution = (id_match[1] if id_match else 'unique_visible_name'
                  if name in {_name(control.get('name')), _name(control.get('text'))}
                  else 'unique_visible_role_description')
    return parsed, {'interface': interface, 'output_envelope': envelope, 'resolution': resolution,
                    'name_source': source, 'normalized_name': name, 'defaulted_nullable_fields': defaulted,
                    'resolved_control_id': identity,
                    'original_bbox': value.get('bbox'), 'resolved_bbox': parsed['bbox'],
                    'action_type_preserved': parsed['action_type'] == action}


class NamedTargetRuntime(_UnadaptedBaseRuntime):
    def __init__(self, *, base, evidence_dir, action_prompt=None, allow_json_fence=False, allow_role_target=False, include_control_context=False):
        super().__init__(model=base.model, processor=base.processor, torch=base.torch)
        self.lock = base.lock
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.action_prompt = action_prompt
        self.include_control_context = include_control_context
        self.allow_json_fence = allow_json_fence
        self.allow_role_target = allow_role_target
        self.interface_id = ROLE_INTERFACE_ID if allow_role_target else (FENCED_INTERFACE_ID if allow_json_fence else INTERFACE_ID)
        self.calls = 0

    def _generate(self, *, prompt, task, observation, suffix):
        prompt = self.action_prompt or prompt
        v3 = getattr(self, 'hybrid_interface_version', 1) == 3
        if self.include_control_context and not v3:
            suffix += control_suffix(observation)
        metadata = None
        if v3:
            raw, metadata = super()._generate_with_metadata(
                prompt=prompt, task=task, observation=observation, suffix=suffix)
        else:
            raw = super()._generate(prompt=prompt, task=task, observation=observation, suffix=suffix)
        self.calls += 1
        receipt = {'interface': _observed_interface(observation, self.interface_id), 'task_id': task.task_id,
                   'observation_id': observation.observation_id,
                   'screenshot_sha256': observation.screenshot_sha256,
                   'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                   'input_suffix': suffix,
                   'input_suffix_sha256': hashlib.sha256(suffix.encode()).hexdigest(),
                   'raw_response': raw}
        if v3:
            receipt['generation_metadata'] = metadata
            self._last_resolution = None
        try:
            if getattr(self, 'hybrid_interface_version', 1) in {2, 3}:
                from copy import deepcopy
                from web_agent.runtime.hybrid_interface import parse_action
                receipt['hybrid_interface_version'] = self.hybrid_interface_version
                receipt['audit_context'] = deepcopy(self._audit_context)
                parsed, resolution = parse_action(raw, observation, interface_version=self.hybrid_interface_version)
            else:
                parsed, resolution = resolve_named_target(raw, observation, allow_json_fence=self.allow_json_fence, allow_role_target=self.allow_role_target)
            receipt.update(status='RESOLVED', resolved=parsed, resolution=resolution)
            if v3:
                self._last_resolution = resolution
            return json.dumps(parsed)
        except ActionParseError as exc:
            receipt.update(status='REJECTED', error=str(exc))
            if v3:
                receipt.update(failure_stage=getattr(exc, 'failure_stage', 'parsing'),
                               diagnostic_code=getattr(exc, 'diagnostic_code', None),
                               diagnostic=getattr(exc, 'diagnostic', None),
                               proposed_action=getattr(exc, 'proposed_action', None))
            raise
        finally:
            (self.evidence_dir / f'output-{self.calls:04d}.json').write_text(json.dumps(receipt, indent=2)+'\n')

    def predict_action(self, task, observation, rng):
        result = super().predict_action(task, observation, rng)
        if getattr(self, 'hybrid_interface_version', 1) == 3:
            from dataclasses import replace
            control_id = (self._last_resolution or {}).get('resolved_control_id')
            if control_id is not None:
                result = replace(result, parameter_hints={
                    **dict(result.parameter_hints), 'resolved_control_id': control_id})
        return result
