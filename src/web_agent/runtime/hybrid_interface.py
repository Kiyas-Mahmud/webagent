"""Version 2 semantic generation boundary; provenance remains in audit receipts.

No model, retrieval, action selection or execution validation is changed here.
Current control IDs are already observation-local (oN:cM) and stay unchanged.
"""
from copy import deepcopy
import json
import math
import re

from web_agent.runtime.policy import ActionParseError
from web_agent.runtime.qwen2vl_pc01 import _strict_json_object
from web_agent.runtime.named_target_policy import resolve_named_target

VERSION = 'hybrid-normal-action-v2'
ADVICE_FIELDS = ('action_probabilities', 'argmax', 'bbox',
                 'grounding_confidence', 'confidence_before', 'advisory_only')
ACTION_FIELDS = ('action_type', 'parameters', 'bbox', 'execution_status',
                 'state_changed', 'environment_error')


def feedback_context(feedback, *, episode_id):
    if feedback is None:
        return None
    # Only diagnostics are normalized. Never rewrite agent-issued text/values.
    result = {k: deepcopy(feedback[k]) for k in ('stage', 'code', 'detail') if k in feedback}
    if episode_id and isinstance(result.get('detail'), str):
        result['detail'] = result['detail'].replace(episode_id, '<episode>')
    return result


def semantic_context(context, *, episode_id, recovery=False, interface_version=2):
    """Allowlist semantic fields after the caller validates raw causal bindings."""
    if interface_version == 3:
        return _semantic_context_v3(context, recovery=recovery)
    if interface_version != 2:
        raise ValueError('unsupported hybrid semantic interface version')
    if recovery:
        fields = ('selected_recovery_strategy', 'observed_failure_diagnosis',
                  'current_visible_controls', 'retrieved_training_examples', 'memory_usage')
    else:
        fields = ('url', 'title', 'current_controls')
    result = {k: deepcopy(context[k]) for k in fields if k in context}
    result['schema'] = 'hybrid-semantic-recovery-v2' if recovery else 'hybrid-semantic-action-v2'
    result['completed_actions'] = [
        {k: deepcopy(a[k]) for k in ACTION_FIELDS if k in a}
        for a in context.get('completed_actions', ())]
    for key in ('last_action', 'failed_action') if recovery else ():
        if key in context:
            result[key] = {k: deepcopy(context[key][k]) for k in ACTION_FIELDS if k in context[key]}
    feedback_key = 'previous_attempt' if recovery else 'last_rejection'
    result[feedback_key] = feedback_context(context.get(feedback_key), episode_id=episode_id)
    if not recovery and 'trained_advice' in context:
        result['trained_advice'] = {k: deepcopy(context['trained_advice'][k]) for k in ADVICE_FIELDS}
    return result


def parse_action(raw, observation, *, interface_version=2):
    """Accept one bare/fenced four-field object; never infer an action or value."""
    if interface_version == 3:
        return _parse_action_v3(raw, observation)
    if interface_version != 2:
        raise ValueError('unsupported hybrid parsing interface version')
    match = re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```', raw.strip(), re.DOTALL)
    body = match.group(1) if match else raw
    value = _strict_json_object(body, context=VERSION)
    if set(value) != {'action_type', 'target', 'bbox', 'value'}:
        raise ActionParseError('hybrid output requires exactly four action fields')
    if not isinstance(value['action_type'], str) or value['action_type'] not in {
            'CLICK', 'TYPE', 'SELECT', 'SCROLL', 'NAVIGATE', 'PRESS_KEY'}:
        exc = ActionParseError('action_type must name exactly one registered action')
        exc.failure_stage = 'parsing'
        exc.diagnostic_code = 'INVALID_ACTION_TYPE'
        raise exc
    try:
        parsed, resolution = resolve_named_target(body, observation, allow_role_target=True)
    except ActionParseError as exc:
        exc.failure_stage = getattr(exc, 'failure_stage', 'target_resolution')
        raise
    # The CLICK label is an issued value too; preserve it in the decision receipt.
    parsed['value'] = value['value']
    resolution['output_envelope'] = 'single_json_fence' if match else 'bare_json'
    return parsed, resolution


V3_VERSION = 'hybrid-normal-action-v3'
PROPOSAL_FIELDS = ('action_type', 'target', 'bbox', 'value')
REGISTERED_ACTIONS = ('CLICK', 'TYPE', 'SELECT', 'SCROLL', 'NAVIGATE', 'PRESS_KEY')
TARGET_SYNTAX = ('current control_id; observed_tag#current_control_id with both '
                 'parts verified; unique observed name/role; or null with a '
                 'normalized bbox for coordinate-only CLICK/TYPE/SELECT')
EXPECTED_TYPES = {
    'action_type': 'one registered action string: ' + ', '.join(REGISTERED_ACTIONS),
    'target': 'string or null',
    'bbox': 'null or four finite JSON numbers [x,y,width,height], normalized to [0,1] with positive width/height',
    'value': 'string or null; preserve the exact issued text',
}


class TargetProposalResolutionError(ActionParseError):
    """A validated, known proposal rejected before any browser command.

    The runner may diagnose this rejected request using its real action label.
    Syntax/schema failures deliberately never receive ``proposed_action``.
    """

    def __init__(self, diagnostic, proposed_action):
        super().__init__(json.dumps(diagnostic, sort_keys=True, ensure_ascii=False))
        self.diagnostic = deepcopy(diagnostic)
        self.proposed_action = deepcopy(proposed_action)
        self.diagnostic_code = diagnostic['code']
        self.failure_stage = 'target_resolution'


def _proposal_diagnostic(raw, observation, code, detail, *, value=None, **extra):
    controls = observation.current_page_state.get('visible_controls', ())
    legal_ids = [c['control_id'] for c in controls
                 if isinstance(c, dict) and isinstance(c.get('control_id'), str)
                 and c.get('observation_id') == observation.observation_id]
    result = {
        'schema': 'hybrid-proposal-diagnostic-v3', 'code': code, 'detail': detail,
        'raw_response': raw, 'rejected_proposal': deepcopy(value),
        'required_fields': list(PROPOSAL_FIELDS), 'expected_types': dict(EXPECTED_TYPES),
        'accepted_target_syntax': TARGET_SYNTAX, 'current_control_ids': legal_ids,
    }
    result.update(extra)
    return result


def _parsing_rejection(diagnostic):
    exc = ActionParseError(json.dumps(diagnostic, sort_keys=True, ensure_ascii=False))
    exc.diagnostic = deepcopy(diagnostic)
    exc.diagnostic_code = diagnostic['code']
    exc.failure_stage = 'parsing'
    return exc


def _parse_action_v3(raw, observation):
    match = re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```', raw.strip(), re.DOTALL)
    body = match.group(1) if match else raw
    try:
        value = _strict_json_object(body, context=V3_VERSION)
        # Python's JSON reader accepts non-JSON NaN/Infinity constants. Reject
        # those without trying to embed nonfinite values in a strict receipt.
        json.dumps(value, allow_nan=False)
    except ActionParseError as exc:
        raise _parsing_rejection(_proposal_diagnostic(
            raw, observation, 'INVALID_JSON_OBJECT', str(exc))) from exc
    except ValueError as exc:
        raise _parsing_rejection(_proposal_diagnostic(
            raw, observation, 'INVALID_JSON_OBJECT', 'Nonfinite JSON numbers are not allowed.')) from exc
    missing, extra = sorted(set(PROPOSAL_FIELDS) - set(value)), sorted(set(value) - set(PROPOSAL_FIELDS))
    if missing or extra:
        raise _parsing_rejection(_proposal_diagnostic(
            raw, observation, 'ACTION_FIELDS_MISMATCH',
            'Supply exactly the four required fields; nullable fields must still be present.',
            value=value, missing_fields=missing, extra_fields=extra))
    errors = {}
    if type(value['action_type']) is not str or value['action_type'] not in REGISTERED_ACTIONS:
        errors['action_type'] = EXPECTED_TYPES['action_type']
    for field in ('target', 'value'):
        if value[field] is not None and type(value[field]) is not str:
            errors[field] = EXPECTED_TYPES[field]
    box = value['bbox']
    if box is not None and (type(box) is not list or len(box) != 4 or any(
            type(n) not in (int, float) or not math.isfinite(n) or not 0 <= n <= 1 for n in box)
            or box[2] <= 0 or box[3] <= 0 or box[0] + box[2] > 1 or box[1] + box[3] > 1):
        errors['bbox'] = EXPECTED_TYPES['bbox']
    if errors:
        raise _parsing_rejection(_proposal_diagnostic(
            raw, observation, 'INVALID_ACTION_FIELD_TYPES', 'Correct the listed fields.',
            value=value, field_errors=errors))
    if value['action_type'] not in {'CLICK', 'TYPE', 'SELECT'} and box is not None:
        raise _parsing_rejection(_proposal_diagnostic(
            raw, observation, 'NON_GROUNDED_BBOX_MUST_BE_NULL',
            'SCROLL, NAVIGATE and PRESS_KEY require bbox null.', value=value))
    try:
        parsed, resolution = resolve_named_target(
            body, observation, allow_role_target=True, allow_tag_control_target=True)
    except ActionParseError as exc:
        diagnostic = _proposal_diagnostic(
            raw, observation, getattr(exc, 'diagnostic_code', 'TARGET_RESOLUTION_REJECTED'),
            str(exc), value=value)
        raise TargetProposalResolutionError(diagnostic, value) from exc
    # Resolution supplies only an observed box/identity. Issued labels and values
    # remain byte-for-byte equal, including CLICK labels and whitespace.
    parsed['target'] = value['target']
    parsed['value'] = value['value']
    resolution['output_envelope'] = 'single_json_fence' if match else 'bare_json'
    resolution['hybrid_interface_version'] = 3
    return parsed, resolution


def feedback_context_v3(feedback):
    """Preserve rejected text exactly; never normalize inside issued values."""
    if feedback is None:
        return None
    result = {k: deepcopy(feedback[k]) for k in ('stage', 'code') if k in feedback}
    detail = feedback.get('detail')
    if isinstance(detail, str):
        try:
            diagnostic = json.loads(detail)
        except (ValueError, TypeError):
            diagnostic = None
        if isinstance(diagnostic, dict) and diagnostic.get('schema') == 'hybrid-proposal-diagnostic-v3':
            result['diagnostic'] = diagnostic
        else:
            result['detail'] = detail
    return result


def _causal_event_v3(action):
    # This is a record of an earlier attempt, deliberately distinct from the
    # requested action object. Stable identity is audit-only executor evidence.
    arguments = {k: deepcopy(v) for k, v in action.get('parameters', {}).items()
                 if k != 'target_stable_identity'}
    result = {'issued_kind': action.get('action_type'), 'issued_arguments': arguments}
    if 'bbox' in action:
        result['issued_box'] = deepcopy(action['bbox'])
    result['observed_result'] = {k: deepcopy(action[k]) for k in (
        'execution_status', 'state_changed', 'environment_error') if k in action}
    return result


def _semantic_context_v3(context, *, recovery):
    fields = (('selected_recovery_strategy', 'observed_failure_diagnosis',
               'retrieved_training_examples', 'memory_usage') if recovery else ('url', 'title'))
    result = {k: deepcopy(context[k]) for k in fields if k in context}
    result['schema'] = 'hybrid-semantic-recovery-v3' if recovery else 'hybrid-semantic-action-v3'
    # Both actors consume this one shared current-control projection. Recovery
    # must not append a second control_suffix with a competing representation.
    result['current_controls'] = deepcopy(context.get(
        'current_visible_controls' if recovery else 'current_controls', []))
    result['causal_history'] = [_causal_event_v3(a) for a in context.get('completed_actions', ())]
    for key in ('last_action', 'failed_action') if recovery else ():
        if key in context:
            result['last_attempt'] = _causal_event_v3(context[key])
    result['proposal_feedback'] = feedback_context_v3(context.get(
        'previous_attempt' if recovery else 'last_rejection'))
    if not recovery and 'trained_advice' in context:
        advice = context['trained_advice']
        result['trained_advice'] = {k: deepcopy(advice[k]) for k in ADVICE_FIELDS
                                   if k != 'grounding_confidence'}
        result['trained_advice']['action_class_confidence'] = deepcopy(advice['grounding_confidence'])
    return result
