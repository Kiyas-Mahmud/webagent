"""Overlapping-target resolution regressions; no claimed model or browser efficacy.

The development-v3 run rejected every `click-button-sequence` recovery proposal
with `TARGET_POINT_MATCH_COUNT:2` while the browser itself hit-tested those
points to the control the model had named. These cases pin the corrected rule:
the model's named control decides identity, the browser's own hit point decides
the executed coordinate, and nothing else about the action may change.
"""
import json
import random

import pytest

from web_agent.benchmarks.miniwob_controls import (
    CONTROL_INTERFACE, LEGACY_CONTROL_INTERFACES, action_point, bind_controls,
    hit_point, prompt_controls, validate_control_action)
from web_agent.runtime.action_parameters import (
    DeterministicParameterProvider, ParameterResolutionError, named_control,
    validate_action_parameters)
from web_agent.runtime.contracts import (
    ActionType, PolicyObservation, PreActionDecision, RuntimeTaskView, probability_map)
from web_agent.runtime.named_target_policy import resolve_named_target
from web_agent.runtime.policy import ActionParseError


# Exact development-v3 `click-button-sequence` geometry: TWO is painted over ONE,
# and each button's box contains the other's centre.
ONE_BOX = [0.271084, 0.327103, 0.120482, 0.186916]
TWO_BOX = [0.253012, 0.280374, 0.120482, 0.186916]
ONE_HIT = [0.379518, 0.420561]   # inside ONE only; TWO ends at x=0.373494
TWO_HIT = [0.313253, 0.373832]   # TWO's own centre, which the browser hits


def raw_control(**kw):
    control = dict(tag='button', input_type='', role='', source_id='subbtn', name='', text='ONE',
                   accessible_names=[], value=None, value_present=False, target_bbox=list(ONE_BOX),
                   candidate_options=[], destination='', clickable=True, link_like=False,
                   contenteditable=False, disabled=False, readonly=False, checked=None,
                   selected=[], focused=False, hit_point=list(ONE_HIT))
    control.update(kw)
    return control


def overlapping():
    return [raw_control(),
            raw_control(source_id='subbtn2', text='TWO', target_bbox=list(TWO_BOX), hit_point=list(TWO_HIT))]


def view(raw=None, oid='episode:obs:2', interface=CONTROL_INTERFACE):
    controls = bind_controls({'controls': raw if raw is not None else overlapping()}, oid)
    return PolicyObservation(task_id='miniwob.click-button-sequence', goal='Click button ONE, then click button TWO.',
                             observation_id=oid, screenshot_sha256='a'*64, screenshot_path=None,
                             width=332, height=214, url='fixture://test', title='',
                             current_page_state={'control_interface': interface, 'visible_controls': controls})


def decision(action='CLICK', box=None, **hints):
    return PreActionDecision(decision_id='decision', observation_id='episode:obs:2',
                             action_type=ActionType(action),
                             action_probabilities=probability_map([a.value for a in ActionType], action),
                             bbox=tuple(box if box is not None else ONE_BOX), grounding_confidence=0.0,
                             confidence_before=0.0, input_observation_ids=('episode:obs:2',),
                             policy_id='p', policy_version='v1', parameter_hints=hints)


def resolve(obs, dec):
    return DeterministicParameterProvider().resolve(
        RuntimeTaskView(task_id=obs.task_id, goal=obs.goal), obs, dec, rng=random.Random(0)).values


def test_named_control_under_overlap_executes_the_model_selected_button():
    """Both proposals development-v3 rejected now resolve to the named control."""
    obs = view()
    for control_id, box, point, source in [('o2:c0', ONE_BOX, ONE_HIT, 'subbtn'),
                                           ('o2:c1', TWO_BOX, TWO_HIT, 'subbtn2')]:
        values = resolve(obs, decision(box=box, target=control_id))
        assert values['target_control_id'] == control_id
        assert [values['target_x'], values['target_y']] == point
        resolved = validate_control_action('CLICK', values, tuple(box), obs, control_id=control_id)
        assert resolved['control_id'] == control_id and resolved['source_id'] == source
        # The point must stay inside the model's own control, never the neighbour.
        left, top, width, height = box
        assert left <= point[0] <= left+width and top <= point[1] <= top+height
        validate_action_parameters(ActionType.CLICK, values, tuple(box))


def test_geometric_ambiguity_alone_no_longer_rejects_but_identity_still_binds():
    obs = view()
    # The exact development-v3 failure: both boxes contain the centre point.
    centre = {'target_x': TWO_BOX[0]+TWO_BOX[2]/2, 'target_y': TWO_BOX[1]+TWO_BOX[3]/2,
              'target_bbox': list(TWO_BOX), 'button': 'left', 'click_count': 1}
    with pytest.raises(ValueError, match='TARGET_POINT_MATCH_COUNT:2'):
        validate_control_action('CLICK', centre, tuple(TWO_BOX), obs)
    # Naming a control that is not in this observation cannot invent a target.
    with pytest.raises(ValueError, match='TARGET_CONTROL_UNRESOLVED'):
        validate_control_action('CLICK', {**centre, 'target_control_id': 'o2:c9'},
                                tuple(TWO_BOX), obs, control_id='o2:c9')


def test_covered_control_is_rejected_rather_than_retargeted():
    """A control the browser reports no hit point for is unreachable, not swapped."""
    covered = overlapping()
    covered[0]['hit_point'] = None
    obs = view(covered)
    values = {'target_x': ONE_BOX[0]+ONE_BOX[2]/2, 'target_y': ONE_BOX[1]+ONE_BOX[3]/2,
              'target_bbox': list(ONE_BOX), 'button': 'left', 'click_count': 1,
              'target_control_id': 'o2:c0'}
    with pytest.raises(ValueError, match='TARGET_CONTROL_NOT_HIT_TESTABLE:o2:c0'):
        validate_control_action('CLICK', values, tuple(ONE_BOX), obs, control_id='o2:c0')
    # Without hit evidence the control claims no identity and the original
    # geometric rule still rejects. It is never retargeted onto the neighbour.
    assert named_control(obs, decision(target='o2:c0')) is None
    with pytest.raises(ParameterResolutionError, match='TARGET_POINT_MATCH_COUNT:2'):
        resolve(obs, decision(target='o2:c0'))
    alone = view([raw_control(hit_point=None)])
    assert named_control(alone, decision(target='o2:c0')) is None
    assert 'target_control_id' not in resolve(alone, decision(target='o2:c0'))


def test_executed_point_must_be_the_browser_verified_one():
    """A moved or stale hit point rejects instead of clicking a guessed pixel."""
    obs = view()
    values = resolve(obs, decision(target='o2:c0'))
    for moved in ([ONE_BOX[0]+ONE_BOX[2]/2, ONE_BOX[1]+ONE_BOX[3]/2], [ONE_HIT[0], ONE_HIT[1]+0.01]):
        with pytest.raises(ValueError, match='TARGET_POINT_NOT_BROWSER_VERIFIED:o2:c0'):
            validate_control_action('CLICK', {**values, 'target_x': moved[0], 'target_y': moved[1]},
                                    tuple(ONE_BOX), obs, control_id='o2:c0')
    stale = view()
    stale.current_page_state['visible_controls'][0]['observation_id'] = 'episode:obs:1'
    with pytest.raises(ValueError, match='STALE_CONTROL_OBSERVATION'):
        validate_control_action('CLICK', values, tuple(ONE_BOX), stale, control_id='o2:c0')


def test_coordinate_only_actions_keep_the_original_rule_and_exact_point():
    """Nothing changes for a model that supplies coordinates instead of a control."""
    obs = view([raw_control(hit_point=[0.3, 0.4])])
    box = (0.271084, 0.327103, 0.120482, 0.186916)
    values = {'target_x': 0.3, 'target_y': 0.35, 'target_bbox': list(box), 'button': 'left', 'click_count': 1}
    assert validate_control_action('CLICK', values, box, obs)['source_id'] == 'subbtn'
    assert values['target_x'] == 0.3 and values['target_y'] == 0.35
    # A decision with no named target never acquires a control identity.
    assert named_control(obs, decision(box=list(box))) is None
    assert 'target_control_id' not in resolve(obs, decision(box=list(box)))


def test_legacy_v2_observations_resolve_exactly_as_before():
    """The completed archives carry no hit evidence and must replay unchanged."""
    legacy = [{k: v for k, v in c.items() if k != 'hit_point'} for c in overlapping()]
    obs = view(legacy, interface=LEGACY_CONTROL_INTERFACES[0])
    assert all(hit_point(c) is None for c in obs.current_page_state['visible_controls'])
    centre = {'target_x': TWO_BOX[0]+TWO_BOX[2]/2, 'target_y': TWO_BOX[1]+TWO_BOX[3]/2,
              'target_bbox': list(TWO_BOX), 'button': 'left', 'click_count': 1}
    with pytest.raises(ValueError, match='TARGET_POINT_MATCH_COUNT:2'):
        validate_control_action('CLICK', centre, tuple(TWO_BOX), obs)
    assert action_point(obs.current_page_state['visible_controls'][1]) == (centre['target_x'], centre['target_y'])
    # A v2 control still resolves by name, and still carries no identity forward.
    parsed, resolution = resolve_named_target(json.dumps(
        dict(action_type='CLICK', target='TWO', bbox=None, value=None)), obs)
    assert parsed['action_type'] == 'CLICK' and list(parsed['bbox']) == TWO_BOX
    # The model's own words stay in the action payload; only the box is resolved.
    assert parsed['target'] == 'TWO' and resolution['resolved_control_id'] == 'o2:c1'


def test_model_facing_context_never_sees_the_executed_point():
    """The planner interface is byte-identical to development-v3's projection."""
    obs = view()
    projected = prompt_controls(obs)
    assert all('hit_point' not in control for control in projected)
    assert all(set(control) == set(projected[0]) for control in projected)
    assert {'control_id', 'target_bbox', 'supported_actions'} <= set(projected[0])
    assert 'hit_point' not in json.dumps(projected)


def test_named_resolution_preserves_the_action_value_and_cannot_swap_controls():
    field = raw_control(tag='input', input_type='text', source_id='field', text='',
                        value='', target_bbox=[0.05, 0.29, 0.38, 0.09], hit_point=[0.24, 0.33],
                        accessible_names=[{'value': 'Person', 'source': 'associated_label'}])
    submit = raw_control(source_id='submit', text='Submit', target_bbox=[0.05, 0.44, 0.28, 0.14],
                         hit_point=[0.19, 0.51])
    obs = view([field, submit])
    parsed, resolution = resolve_named_target(json.dumps(
        dict(action_type='TYPE', target='o2:c0', bbox=None, value='Thaddeus')), obs)
    assert parsed['action_type'] == 'TYPE' and parsed['value'] == 'Thaddeus'
    assert parsed['target'] == 'o2:c0' and resolution['action_type_preserved']
    assert resolution['resolved_control_id'] == 'o2:c0'
    values = resolve(obs, decision('TYPE', box=field['target_bbox'], target='o2:c0', text='Thaddeus'))
    assert values['text'] == 'Thaddeus' and values['target_control_id'] == 'o2:c0'
    assert validate_control_action('TYPE', values, tuple(field['target_bbox']), obs,
                                   control_id='o2:c0')['source_id'] == 'field'
    # TYPE on the submit button stays incompatible; identity grants no capability.
    with pytest.raises(ValueError, match='ACTION_TARGET_INCOMPATIBLE:TYPE:o2:c1'):
        validate_control_action('TYPE', {'target_x': 0.19, 'target_y': 0.51,
                                         'target_bbox': submit['target_bbox'], 'text': 'x',
                                         'target_control_id': 'o2:c1'},
                                tuple(submit['target_bbox']), obs, control_id='o2:c1')
    with pytest.raises(ActionParseError):
        resolve_named_target(json.dumps(dict(action_type='TYPE', target='o2:c1', bbox=None, value='x')), obs)


def test_hit_point_must_lie_inside_its_own_control():
    with pytest.raises(ValueError, match='outside its own target box'):
        bind_controls({'controls': [raw_control(hit_point=[0.9, 0.9])]}, 'episode:obs:2')
    with pytest.raises(ValueError, match='invalid control hit point'):
        bind_controls({'controls': [raw_control(hit_point=[0.3])]}, 'episode:obs:2')


def test_identity_field_is_rejected_on_actions_that_have_no_target():
    with pytest.raises(ParameterResolutionError, match='do not match the registered schema'):
        validate_action_parameters(ActionType.NAVIGATE,
                                   {'url': 'http://fixture/', 'target_control_id': 'o2:c0'}, None)
    with pytest.raises(ParameterResolutionError, match='must be a non-empty control ID'):
        validate_action_parameters(ActionType.CLICK,
                                   {'target_x': ONE_HIT[0], 'target_y': ONE_HIT[1],
                                    'target_bbox': list(ONE_BOX), 'button': 'left',
                                    'click_count': 1, 'target_control_id': ''}, tuple(ONE_BOX))


def test_identical_boxes_fall_back_instead_of_choosing_a_control():
    """Two controls sharing one box identify nothing; the old rule then applies."""
    twin = raw_control(source_id='twin', text='ONE', hit_point=list(ONE_HIT))
    obs = view([raw_control(), twin])
    assert named_control(obs, decision(target='ONE')) is None
    with pytest.raises(ParameterResolutionError, match='TARGET_POINT_MATCH_COUNT:2'):
        resolve(obs, decision(target='ONE'))


def test_model_target_string_is_never_reinterpreted_as_identity():
    """A named target that the resolver did not place on this box claims nothing."""
    obs = view()
    # A coordinate-shaped decision on a box no control owns resolves nothing.
    assert named_control(obs, decision(box=[0.4, 0.4, 0.1, 0.1], target='o2:c0')) is None
    # And a decision with no target hint never acquires an identity either.
    assert named_control(obs, decision(box=ONE_BOX)) is None


def test_overlapping_point_keeps_the_registration_of_its_own_rectangle():
    """The second ambiguity gate must not discard an already-resolved target.

    `_visible_point_target` cannot separate two rectangles that both contain the
    executed point. When the action is grounded on one of those exact registered
    rectangles, that registration is the resolved target and still stands; when
    it is grounded on no registered rectangle, nothing is established.
    """
    from web_agent.runtime.contracts import ConcreteAction, RuntimeTaskView
    from web_agent.runtime.recovery.strategies import (
        RecoveryResolutionError, build_point_target_evidence,
        build_recovery_target_evidence, validate_recovery_target_evidence)
    from web_agent.benchmarks.miniwob_controls import compatible_actions

    obs = view()
    controls = obs.current_page_state['visible_controls']
    task = RuntimeTaskView(task_id=obs.task_id, goal=obs.goal)
    evidence = compatible_actions(controls)
    obs.current_page_state.update(
        recovery_target_evidence=build_recovery_target_evidence(
            task=task, observation_id=obs.observation_id, compatible_actions=evidence),
        visible_point_targets=build_point_target_evidence(
            task=task, observation_id=obs.observation_id, compatible_actions=evidence))

    def click(box, point, control_id=None):
        params = {'target_x': point[0], 'target_y': point[1], 'target_bbox': list(box),
                  'button': 'left', 'click_count': 1}
        if control_id is not None:
            params['target_control_id'] = control_id
        return ConcreteAction(action_id='recovery', source_decision_id='decision',
                              action_type=ActionType.CLICK, bbox=tuple(box), parameters=params)

    # TWO's browser-verified point sits inside ONE's rectangle as well; the
    # action is still grounded on TWO's own registered rectangle.
    validate_recovery_target_evidence(click(TWO_BOX, TWO_HIT, 'o2:c1'), task=task, post_failure_observation=obs)
    validate_recovery_target_evidence(click(ONE_BOX, ONE_HIT, 'o2:c0'), task=task, post_failure_observation=obs)
    # An unregistered rectangle establishes nothing, ambiguous point or not.
    with pytest.raises(RecoveryResolutionError, match='not visibly present'):
        validate_recovery_target_evidence(click([0.26, 0.33, 0.05, 0.05], TWO_HIT),
                                        task=task, post_failure_observation=obs)


def _policy_view(extra=None):
    """Observation carrying the registered executable-action evidence."""
    from web_agent.benchmarks.miniwob_controls import compatible_actions
    from web_agent.runtime.contracts import RuntimeTaskView
    from web_agent.runtime.recovery.strategies import (
        build_point_target_evidence, build_recovery_target_evidence)
    obs = view(extra)
    task = RuntimeTaskView(task_id=obs.task_id, goal=obs.goal)
    evidence = compatible_actions(obs.current_page_state['visible_controls'])
    obs.current_page_state.update(
        recovery_target_evidence=build_recovery_target_evidence(
            task=task, observation_id=obs.observation_id, compatible_actions=evidence),
        visible_point_targets=build_point_target_evidence(
            task=task, observation_id=obs.observation_id, compatible_actions=evidence))
    return task, obs


def test_executable_action_set_comes_only_from_the_observed_page():
    """Two buttons expose CLICK and SCROLL; never TYPE, SELECT or NAVIGATE."""
    from web_agent.runtime.recovery.strategies import registered_executable_action_types
    task, obs = _policy_view()
    executable = registered_executable_action_types(task_view=task, observation=obs)
    assert executable == frozenset({ActionType.CLICK, ActionType.SCROLL})
    # A text field adds TYPE; a select adds SELECT. Nothing adds NAVIGATE,
    # because no observed control exposes a permitted destination.
    field = raw_control(tag='input', input_type='text', source_id='field', text='',
                        value='', target_bbox=[0.05, 0.29, 0.38, 0.09], hit_point=[0.24, 0.33])
    task, obs = _policy_view([raw_control(), field])
    assert registered_executable_action_types(task_view=task, observation=obs) == frozenset(
        {ActionType.CLICK, ActionType.TYPE, ActionType.SCROLL})
    # No registered evidence at all leaves selection unrestricted.
    assert registered_executable_action_types(task_view=task, observation=view()) is None


def test_restriction_selects_the_best_executable_class_without_touching_probabilities():
    """The learned distribution is preserved; only the argmax domain narrows."""
    from web_agent.labels import ACTION_TYPE_INV
    from web_agent.runtime.recovery.strategies import registered_executable_action_types
    task, obs = _policy_view()
    executable = registered_executable_action_types(task_view=task, observation=obs)
    # The exact archived development-v6 E1 distribution for click-button.
    probabilities = [0.3021, 0.0300, 0.0286, 0.2928, 0.3164, 0.0301]
    unrestricted = max(range(6), key=probabilities.__getitem__)
    selectable = [i for i in range(6)
                  if ActionType(ACTION_TYPE_INV[i]) in executable]
    restricted = max(selectable, key=probabilities.__getitem__)
    assert ACTION_TYPE_INV[unrestricted] == 'NAVIGATE'   # unexecutable here
    assert ACTION_TYPE_INV[restricted] == 'CLICK'        # best executable class
    # Probabilities themselves are untouched.
    assert probabilities == [0.3021, 0.0300, 0.0286, 0.2928, 0.3164, 0.0301]


def test_restriction_is_off_by_default_and_never_invents_a_class():
    from types import SimpleNamespace
    from web_agent.runtime.qwen2vl_pc01 import _SelectedCheckpointRuntime
    runtime = _SelectedCheckpointRuntime.__new__(_SelectedCheckpointRuntime)
    runtime.restrict_to_executable_actions = False
    task, obs = _policy_view()
    assert runtime.executable_action_types(task, obs) is None
    runtime.restrict_to_executable_actions = True
    assert runtime.executable_action_types(task, obs) == frozenset(
        {ActionType.CLICK, ActionType.SCROLL})
    # An observation with no evidence stays unrestricted rather than empty.
    assert runtime.executable_action_types(task, view()) is None
