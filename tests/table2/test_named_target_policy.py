import json
from types import SimpleNamespace
import pytest
from web_agent.runtime.named_target_policy import resolve_named_target
from web_agent.runtime.policy import ActionParseError


def obs(controls=None):
    return SimpleNamespace(current_page_state={'visible_controls':controls if controls is not None else [dict(tag='button',text='Next',name='',target_bbox=[.1,.2,.3,.1]) ]})


def test_click_label_resolves_only_unique_visible_control():
    raw=json.dumps(dict(action_type='CLICK',target=None,bbox=None,value='next'))
    parsed,receipt=resolve_named_target(raw,obs())
    assert parsed==dict(action_type='CLICK',target='next',bbox=[.1,.2,.3,.1],value=None)
    assert receipt['action_type_preserved']


def test_observed_custom_link_support_preserves_the_model_target():
    control=dict(tag='span', text='Details', name='', clickable=True,
                 link_like=True, target_bbox=[.1,.2,.3,.1])
    raw=json.dumps(dict(action_type='CLICK',target='link Details'))
    parsed,receipt=resolve_named_target(raw,obs([control]),allow_role_target=True)
    assert parsed['bbox']==control['target_bbox'] and parsed['action_type']=='CLICK'
    assert receipt['interface']=='visible-pointer-description-action-interface-v5'
    with pytest.raises(ActionParseError):
        resolve_named_target(json.dumps(dict(action_type='TYPE',target='Details',value='x')),obs([control]))
    with pytest.raises(ActionParseError):
        resolve_named_target(raw,obs([control,control]),allow_role_target=True)


def test_explicit_name_precedence_over_generated_box_is_logged():
    raw=json.dumps(dict(action_type='CLICK',target='Next',bbox=[.8,.8,.1,.1],value=None))
    parsed,receipt=resolve_named_target(raw,obs())
    assert receipt['original_bbox']==[.8,.8,.1,.1]
    assert parsed['bbox']==[.1,.2,.3,.1]


def test_ambiguous_or_missing_name_is_rejected():
    raw=json.dumps(dict(action_type='CLICK',target='Next',bbox=None,value=None))
    control=obs().current_page_state['visible_controls'][0]
    for controls in [[],[control,control]]:
        with pytest.raises(ActionParseError):resolve_named_target(raw,obs(controls))


def test_typed_text_is_not_reinterpreted_as_target():
    raw=json.dumps(dict(action_type='TYPE',target=None,bbox=None,value='Next'))
    with pytest.raises(ActionParseError):resolve_named_target(raw,obs())


def test_navigation_is_not_replaced_by_click():
    raw=json.dumps(dict(action_type='NAVIGATE',target=None,bbox=None,value=None))
    parsed,receipt=resolve_named_target(raw,obs())
    assert parsed['action_type']=='NAVIGATE' and parsed['value'] is None


def test_conflicting_explicit_target_cannot_fall_back_to_value():
    raw=json.dumps(dict(action_type='CLICK',target='Missing',bbox=None,value='Next'))
    with pytest.raises(ActionParseError):resolve_named_target(raw,obs())


def test_omitted_click_value_is_null_without_inventing_action_data():
    parsed,receipt=resolve_named_target(json.dumps(dict(action_type='CLICK',target='Next',bbox=[.1,.5,.2,.55])),obs())
    assert parsed['action_type']=='CLICK' and parsed['value'] is None
    assert receipt['defaulted_nullable_fields']==['value']


def test_unknown_fields_and_wrong_value_types_are_rejected():
    for value in [dict(action_type='CLICK',target='Next',extra='field'),dict(action_type='CLICK',target='Next',value=True)]:
        with pytest.raises(ActionParseError):resolve_named_target(json.dumps(value),obs())


def test_single_fenced_json_preserves_exact_action_under_explicit_new_interface():
    raw = json.dumps(dict(action_type='CLICK', target='Next', bbox=None, value=None))
    expected, _ = resolve_named_target(raw, obs())
    for prefix in ('```json\n', '```\n'):
        wrapped = prefix + raw + '\n```'
        parsed, receipt = resolve_named_target(wrapped, obs(), allow_json_fence=True)
        assert parsed == expected
        assert receipt['output_envelope'] == 'single_json_fence'
        assert receipt['interface'] == 'visible-name-action-interface-v3'
        with pytest.raises(ActionParseError):
            resolve_named_target(wrapped, obs())


@pytest.mark.parametrize('raw', [
    'Explanation\n```json\n{}\n```',
    '```json\n{}\n```\n```json\n{}\n```',
    '```json\n{"action_type":"CLICK","target":"Next","bbox":[0,0,.1,.1]"}\n```',
    '```json\n{"action_type":"CLICK","action_type":"TYPE"}\n```',
])
def test_fence_support_does_not_repair_json_or_extract_a_preferred_object(raw):
    with pytest.raises(ActionParseError):
        resolve_named_target(raw, obs(), allow_json_fence=True)


def test_role_qualified_label_preserves_the_model_named_target():
    controls = [dict(tag='button', text='ONE', target_bbox=[.1,.2,.2,.1]),
                dict(tag='button', text='TWO', target_bbox=[.1,.4,.2,.1])]
    parsed, receipt = resolve_named_target(json.dumps(dict(action_type='CLICK', target='button ONE')),
                                          obs(controls), allow_role_target=True)
    assert parsed['target'] == 'button ONE' and parsed['bbox'] == [.1,.2,.2,.1]
    assert receipt['resolution'] == 'unique_visible_role_description'


def test_bare_role_with_explicit_click_label_is_conjunctive():
    raw = json.dumps(dict(action_type='CLICK', target='button', value='Next'))
    parsed, _ = resolve_named_target(raw, obs(), allow_role_target=True)
    assert parsed['action_type'] == 'CLICK' and parsed['bbox'] == [.1,.2,.3,.1]
    with pytest.raises(ActionParseError):
        resolve_named_target(raw, obs([dict(tag='a', text='Next', target_bbox=[.1,.2,.3,.1])]), allow_role_target=True)


def test_ambiguous_role_and_unknown_label_do_not_select_an_arbitrary_control():
    control=obs().current_page_state['visible_controls'][0]
    for target, controls in [('button', [control,control]), ('button Missing', [control])]:
        with pytest.raises(ActionParseError):
            resolve_named_target(json.dumps(dict(action_type='CLICK', target=target)), obs(controls), allow_role_target=True)


def test_text_field_description_preserves_supplied_text_and_requires_unique_input():
    raw=json.dumps(dict(action_type='TYPE',target='text field',value='literal value'))
    control=dict(tag='input',input_type='text',text='',target_bbox=[.1,.2,.3,.1])
    parsed,_=resolve_named_target(raw,obs([control]),allow_role_target=True)
    assert parsed['value']=='literal value' and parsed['action_type']=='TYPE'
    with pytest.raises(ActionParseError):
        resolve_named_target(raw,obs([control,control]),allow_role_target=True)
