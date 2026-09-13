"""Conservative action-effect evidence, separate from learned task assessment."""
from web_agent.runtime.contracts import canonical_sha256
from web_agent.benchmarks.miniwob_controls import is_observable_controls, validate_control_action

VERSION = 'observable-control-effect-v1'


def observable_step_effect(action, pre, post):
    record = {'schema': VERSION, 'task_id': pre.task_id,
              'action_id': action.action_id, 'pre_observation_id': pre.observation_id,
              'post_observation_id': post.observation_id, 'observed_effect': False,
              'effect_kind': None, 'before_state_sha256': None, 'after_state_sha256': None,
              'reason': 'NO_VERIFIABLE_CONTROL_EFFECT'}
    if pre.task_id != post.task_id:
        raise ValueError('progress observation task mismatch')
    if not is_observable_controls(post.current_page_state):
        return record
    try:
        before = validate_control_action(action.action_type.value, action.parameters, action.bbox, pre,
                                         control_id=action.parameters.get('target_control_id'))
    except ValueError:
        return record
    if before is None or not before.get('source_id'):
        return record
    matches = [c for c in post.current_page_state.get('visible_controls', ())
               if c.get('source_id') == before['source_id']]
    if len(matches) != 1:
        return record
    after = matches[0]
    if any(before.get(k) != after.get(k) for k in ('tag', 'input_type', 'name', 'accessible_names')):
        return record
    kind = action.action_type.value
    field = None
    if kind == 'TYPE' and before.get('input_type') != 'password':
        expected = action.parameters.get('text')
        if isinstance(expected, str) and expected and after.get('value') == expected:
            field = 'value'
    elif kind == 'SELECT' and action.parameters.get('option') in after.get('selected', ()):
        field = 'selected'
    elif kind == 'CLICK' and type(before.get('checked')) is bool and type(after.get('checked')) is bool:
        field = 'checked'
    # Focus, screenshot changes and button clicks alone do not establish useful
    # preparation. Masked password values cannot establish exact inserted text.
    if field is None or before.get(field) == after.get(field):
        return record
    identity = {k: before[k] for k in ('source_id', 'tag', 'input_type', 'name', 'accessible_names')}
    record.update(observed_effect=True, effect_kind=kind + '_' + field.upper(), reason='OBSERVED_ACTION_EFFECT',
        before_state_sha256=canonical_sha256({'control': identity, 'field': field, 'state': before[field]}),
        after_state_sha256=canonical_sha256({'control': identity, 'field': field, 'state': after[field]}))
    return record


def register_novel_effect(record, seen):
    """No continuation credit for repeated states or toggling back to a baseline."""
    novel = False
    if record['observed_effect']:
        seen.add(record['before_state_sha256'])
        novel = record['after_state_sha256'] not in seen
        seen.add(record['after_state_sha256'])
    return {**record, 'novel_effect': novel}
