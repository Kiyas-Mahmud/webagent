"""CPU-only identity contracts; no browser, model or archived evidence access."""
from copy import deepcopy
from dataclasses import replace
import random

import pytest

from web_agent.benchmarks.miniwob_controls import (
    CONTROL_INTERFACE, STABLE_TARGET_FIELD, bind_controls, compatible_actions, validate_control_action,
)
from web_agent.runtime.action_parameters import (
    DeterministicParameterProvider, ParameterResolutionError, concretize,
    validate_action_parameters, validate_observation_bound_action_parameters,
)
from web_agent.runtime.contracts import (
    ActionType, Observation, ObservationStage, PolicyObservation, PreActionDecision,
    RecoveryDecision, RecoveryStrategy, RuntimeTaskView, probability_map,
)
from web_agent.runtime.decision import LoopGuard
from web_agent.runtime.protocol import LoopRule
from web_agent.runtime.recovery.strategies import build_recovery_target_evidence, resolve_strategy


TASK = RuntimeTaskView(task_id='stable-target-engineering', goal='Use the visible control.')


def control(**changes):
    return dict(dict(tag='input', input_type='text', role='', source_id='field',
                     name='Person', text='', accessible_names=[], value='', value_present=False,
                     target_bbox=[.1, .2, .3, .1], candidate_options=[], destination='',
                     clickable=True, link_like=False, contenteditable=False, disabled=False,
                     readonly=False, checked=None, selected=[], focused=False,
                     hit_point=[.25, .25]), **changes)


def view(n=1, *, controls=None, stable=True, state='a'):
    oid = f'engineering:obs:{n}'
    visible = bind_controls({'controls': [control()] if controls is None else controls}, oid)
    page = {'control_interface': CONTROL_INTERFACE, 'visible_controls': visible}
    if stable:
        page['hybrid_interface_version'] = 3
    page['recovery_target_evidence'] = build_recovery_target_evidence(
        task=TASK, observation_id=oid, compatible_actions=compatible_actions(visible))
    selects = [dict(target_bbox=c['target_bbox'], candidate_options=c['candidate_options'])
               for c in visible if c['tag'] == 'select']
    if selects:
        page['observable_select_controls'] = selects
    return PolicyObservation(task_id=TASK.task_id, goal=TASK.goal, observation_id=oid,
                             screenshot_sha256=state * 64, screenshot_path=None,
                             width=1000, height=600, url='fixture://controls', title='',
                             current_page_state=page)


def action(obs, kind=ActionType.TYPE, *, index=0, target=None, **hints):
    selected = obs.current_page_state['visible_controls'][index]
    if kind is ActionType.TYPE:
        hints.setdefault('text', ' Exact Case \n \"value\" ')
    if kind is ActionType.SELECT:
        hints.setdefault('option', 'Exact Case')
        hints.setdefault('candidate_options', list(selected['candidate_options']))
    decision = PreActionDecision(
        decision_id='decision:' + obs.observation_id, observation_id=obs.observation_id,
        action_type=kind, action_probabilities=probability_map([a.value for a in ActionType], kind.value),
        bbox=tuple(selected['target_bbox']), grounding_confidence=0., confidence_before=0.,
        input_observation_ids=(obs.observation_id,), policy_id='scripted', policy_version='v3',
        parameter_hints={'target': target or selected['control_id'], **hints})
    parameters = DeterministicParameterProvider().resolve(TASK, obs, decision, rng=random.Random(0))
    return concretize(action_id='action:' + obs.observation_id, decision=decision, parameters=parameters)


def retry(failed, current, *, enabled=True):
    decision = RecoveryDecision(decision_id='retry-decision', incident_id='incident',
                                strategy=RecoveryStrategy.RETRY, trigger_sources=('executor',),
                                diagnosis='NO_EFFECT')
    return resolve_strategy(decision, failed, attempt_id='retry', incident_attempt_index=1,
                            episode_attempt_index=1, task=TASK, post_failure_observation=current,
                            stable_target_identity=enabled)


@pytest.mark.parametrize('kind', [ActionType.CLICK, ActionType.TYPE, ActionType.SELECT])
def test_retry_rebinds_only_capture_id_and_receipt_preserving_issued_action(kind):
    controls = [control(tag='select', candidate_options=['Exact Case', 'other'])] if kind is ActionType.SELECT else None
    before, current = view(1, controls=controls), view(2, controls=controls)
    failed = action(before, kind, **({'button': 'right', 'click_count': 2} if kind is ActionType.CLICK else {}))
    evidence_hashes = (before.record_sha256, current.record_sha256, failed.record_sha256)
    plan = retry(failed, current)
    assert plan.resolution_status == 'READY', plan.rejection_reason
    issued = plan.actions[0]
    assert issued.action_type is failed.action_type and issued.bbox == failed.bbox
    expected = deepcopy(dict(failed.parameters))
    expected['target_control_id'] = 'o2:c0'
    expected[STABLE_TARGET_FIELD]['observation_id'] = current.observation_id
    expected[STABLE_TARGET_FIELD]['control_id'] = 'o2:c0'
    assert issued.parameters == expected
    assert evidence_hashes == (before.record_sha256, current.record_sha256, failed.record_sha256)
    validate_observation_bound_action_parameters(kind, issued.parameters, issued.bbox, observation=current)


@pytest.mark.parametrize('controls', [
    [], [control(source_id='replacement')], [control(text='replacement')],
    [control(value='partially changed', value_present=True)], [control(disabled=True)],
    [control(readonly=True)], [control(hit_point=None)],
    [control(target_bbox=[.1, .2, .4, .1])],
    [control(), control()], [control(), control(text='different same source')],
])
def test_retry_rejects_removed_replaced_changed_ambiguous_and_unusable_controls(controls):
    failed = action(view())
    plan = retry(failed, view(2, controls=controls))
    assert plan.resolution_status == 'REJECTED' and plan.actions == ()


def test_retry_revalidates_current_browser_hit_point_without_repairing_coordinates():
    failed = action(view())
    failed = replace(failed, parameters={**failed.parameters, 'target_x': .2})
    plan = retry(failed, view(2))
    assert plan.resolution_status == 'REJECTED'
    assert 'TARGET_POINT_NOT_BROWSER_VERIFIED' in plan.rejection_reason


def test_missing_or_foreign_identity_evidence_cannot_authorize_rebinding():
    failed = action(view())
    parameters = deepcopy(dict(failed.parameters))
    parameters.pop(STABLE_TARGET_FIELD)
    assert retry(replace(failed, parameters=parameters), view(2)).resolution_status == 'REJECTED'
    parameters = deepcopy(dict(failed.parameters))
    parameters[STABLE_TARGET_FIELD]['task_id'] = 'foreign'
    assert retry(replace(failed, parameters=parameters), view(2)).resolution_status == 'REJECTED'
    anonymous = action(view(controls=[control(source_id='')]))
    assert STABLE_TARGET_FIELD not in anonymous.parameters
    assert retry(anonymous, view(2, controls=[control(source_id='')])).resolution_status == 'REJECTED'


def test_current_proposal_stale_id_is_rejected_even_when_old_geometry_matches():
    with pytest.raises(ParameterResolutionError, match='NOT_UNIQUELY_CURRENT'):
        action(view(2), target='o1:c0')


@pytest.mark.parametrize('changes', [{'value': 'changed'}, {'focused': True}, {'checked': True},
                                     {'destination': 'http://changed.example/'}])
def test_worker_current_capture_revalidation_rejects_observed_changes_before_dispatch(changes):
    """The worker reuses the capture ID, so checking that ID alone is insufficient."""
    before = view()
    issued = action(before)
    current = view(controls=[control(**changes)])
    with pytest.raises(ValueError, match='STABLE_TARGET_CHANGED_BEFORE_EXECUTION'):
        validate_control_action(issued.action_type.value, issued.parameters, issued.bbox,
                                current, control_id=issued.parameters['target_control_id'])


def test_receipt_is_observation_bound_and_defaults_preserve_legacy_behavior():
    old = action(view(stable=False))
    assert STABLE_TARGET_FIELD not in old.parameters
    assert retry(old, view(2, stable=False)).resolution_status == 'REJECTED'
    current = action(view())
    assert retry(current, view(2), enabled=False).resolution_status == 'REJECTED'
    with pytest.raises(ParameterResolutionError, match='unexpected'):
        validate_action_parameters(current.action_type, current.parameters, current.bbox)
    with pytest.raises(ParameterResolutionError, match='UNRESOLVED'):
        validate_observation_bound_action_parameters(current.action_type, current.parameters,
                                                     current.bbox, observation=view(2))


def loop_record(guard, obs, issued):
    raw = Observation(observation_id=obs.observation_id, episode_id='engineering',
                      stage=ObservationStage.PRE_ACTION, screenshot_sha256=obs.screenshot_sha256,
                      screenshot_path=None, width=obs.width, height=obs.height,
                      url=obs.url, title=obs.title, page_state=obs.current_page_state)
    return guard.record(raw, issued)


def test_abab_stops_at_fourth_capture_only_when_stable_identity_enabled():
    guards = [LoopGuard(LoopRule(), stable_target_identity=flag) for flag in (False, True)]
    results = [[], []]
    for n, state in enumerate('ababa', 1):
        obs = view(n, state=state)
        issued = action(obs)
        original = (obs.record_sha256, issued.record_sha256)
        for guard, result in zip(guards, results):
            result.append(loop_record(guard, obs, issued))
        assert (obs.record_sha256, issued.record_sha256) == original
    assert results[0] == [False, False, False, False, True]
    assert results[1] == [False, False, False, True, True]


@pytest.mark.parametrize('difference', ['value', 'target', 'unverified'])
def test_stable_loop_identity_never_collapses_different_values_controls_or_unverified_receipts(difference):
    guard = LoopGuard(LoopRule(), stable_target_identity=True)
    controls = [control(), control(source_id='second', name='Second',
                                   target_bbox=[.6, .2, .3, .1], hit_point=[.75, .25])]
    results = []
    for n, state in enumerate('abab', 1):
        obs = view(n, state=state, controls=controls)
        issued = action(obs, index=1 if difference == 'target' and n == 3 else 0,
                        text='different' if difference == 'value' and n == 3 else 'same')
        if difference == 'unverified':
            parameters = deepcopy(dict(issued.parameters))
            parameters[STABLE_TARGET_FIELD]['semantic_sha256'] = 'f' * 64
            issued = replace(issued, parameters=parameters)
        results.append(loop_record(guard, obs, issued))
    assert results == [False, False, False, False]
