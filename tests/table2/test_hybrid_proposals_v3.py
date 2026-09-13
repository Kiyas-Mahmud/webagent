"""CPU-only checks of the opt-in proposal contract, not model capability."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
import random
from types import SimpleNamespace as NS

import pytest

from web_agent.runtime.hybrid_action_policy import HybridActionContext, HybridActionPolicy
from web_agent.runtime.hybrid_interface import (
    TargetProposalResolutionError, feedback_context_v3, parse_action, semantic_context,
)
from web_agent.runtime.policy import ActionParseError, CallablePolicyAdapter, PolicyKind
from tests.table2.test_development_v3 import observation
from tests.table2.test_hybrid_agent import advice, task, with_history


def proposal(kind='TYPE', target='input#o1:c0', value=' Exact  Case! \nΩ ', bbox=None):
    return dict(action_type=kind, target=target, bbox=bbox, value=value)


def parse(value, obs=None):
    return parse_action(json.dumps(value), obs or observation(), interface_version=3)


@pytest.mark.parametrize('kind,value', [
    ('CLICK', ' Explicit label '), ('TYPE', ' Exact  Case! \nΩ '), ('SELECT', 'two'),
    ('SCROLL', 'down'), ('NAVIGATE', 'https://example.org/A?value=Exact%20Case'), ('PRESS_KEY', 'TAB'),
])
def test_all_six_actions_bare_or_single_fence_preserve_issued_values(kind, value):
    obs = observation(kind='select' if kind == 'SELECT' else 'text')
    target = ('select' if kind == 'SELECT' else 'input') + '#o1:c0' if kind in {'CLICK', 'TYPE', 'SELECT'} else None
    issued = proposal(kind, target, value)
    raw = json.dumps(issued)
    parsed, resolution = parse_action(raw, obs, interface_version=3)
    fenced, fence_resolution = parse_action('```json\n' + raw + '\n```', obs, interface_version=3)
    assert parsed == fenced
    assert parsed['action_type'] == kind and parsed['target'] == target and parsed['value'] == value
    assert resolution['output_envelope'] == 'bare_json'
    assert fence_resolution['output_envelope'] == 'single_json_fence'
    if target:
        assert resolution['resolved_control_id'] == 'o1:c0'


@pytest.mark.parametrize('target', ['o1:c0', 'input#o1:c0', 'Field', 'text field Field'])
def test_only_uniquely_observed_current_targets_are_resolved(target):
    parsed, receipt = parse(proposal(target=target))
    assert parsed['target'] == target
    assert parsed['bbox'] == [.1, .2, .3, .1]
    assert receipt['resolved_control_id'] == 'o1:c0'


@pytest.mark.parametrize('target,code', [
    ('input#o0:c0', 'CURRENT_CONTROL_ID_NOT_UNIQUE'),
    ('o0:c0', 'CURRENT_CONTROL_ID_NOT_UNIQUE'),
    ('button#o1:c0', 'CURRENT_CONTROL_TAG_MISMATCH'),
    ('#field', 'TARGET_NOT_UNIQUELY_COMPATIBLE'),
    ('input[name=Field]', 'TARGET_NOT_UNIQUELY_COMPATIBLE'),
])
def test_target_rejection_keeps_known_exact_action_and_never_guesses(target, code):
    issued = proposal(target=target)
    raw = json.dumps(issued)
    with pytest.raises(TargetProposalResolutionError) as caught:
        parse_action(raw, observation(), interface_version=3)
    exc = caught.value
    assert exc.failure_stage == 'target_resolution'
    assert exc.diagnostic_code == code
    assert exc.proposed_action == issued
    assert exc.diagnostic['rejected_proposal'] == issued
    assert exc.diagnostic['raw_response'] == raw
    assert exc.diagnostic['current_control_ids'] == ['o1:c0']
    exc.proposed_action['value'] = 'mutated'
    assert issued['value'] == ' Exact  Case! \nΩ '


@pytest.mark.parametrize('change,code', [
    ('duplicate', 'CURRENT_CONTROL_ID_NOT_UNIQUE'), ('stale', 'STALE_CONTROL_OBSERVATION'),
    ('disabled', 'ACTION_TARGET_INCOMPATIBLE'), ('readonly', 'ACTION_TARGET_INCOMPATIBLE'),
])
def test_current_id_checks_identity_before_capability_filter(change, code):
    obs = observation()
    controls = obs.current_page_state['visible_controls']
    if change == 'duplicate':
        duplicate = deepcopy(controls[0]); duplicate['disabled'] = True; controls.append(duplicate)
    elif change == 'stale':
        controls[0]['observation_id'] = 'episode:obs:0'
    else:
        controls[0][change] = True
    with pytest.raises(TargetProposalResolutionError) as caught:
        parse(proposal(), obs)
    assert caught.value.diagnostic_code == code


@pytest.mark.parametrize('issued,missing,extra', [
    ({'action_type': 'CLICK', 'target': 'Field', 'bbox': None}, ['value'], []),
    ({**proposal(), 'parameters': {'text': 'x'}}, [], ['parameters']),
])
def test_schema_feedback_identifies_keys_without_defaulting_nullable_values(issued, missing, extra):
    with pytest.raises(ActionParseError) as caught:
        parse(issued)
    exc = caught.value
    assert not isinstance(exc, TargetProposalResolutionError)
    assert not hasattr(exc, 'proposed_action')
    assert exc.diagnostic['missing_fields'] == missing and exc.diagnostic['extra_fields'] == extra
    assert exc.diagnostic['rejected_proposal'] == issued


@pytest.mark.parametrize('field,value', [
    ('action_type', ['CLICK', 'TYPE']), ('action_type', 'CLICK|TYPE'),
    ('target', 7), ('value', ['text']), ('bbox', [True, 0, .2, .2]),
    ('bbox', ['0', 0, .2, .2]), ('bbox', [.9, 0, .2, .2]),
])
def test_type_diagnostics_have_no_learned_action_for_invalid_schema(field, value):
    with pytest.raises(ActionParseError) as caught:
        parse({**proposal(), field: value})
    exc = caught.value
    assert exc.failure_stage == 'parsing' and not hasattr(exc, 'proposed_action')
    assert field in exc.diagnostic['field_errors']
    assert field in exc.diagnostic['expected_types']


@pytest.mark.parametrize('raw', [
    '{} trailing', '[]', '```json\n{}\n```\n```json\n{}\n```',
    '{"action_type":"CLICK","action_type":"TYPE","target":null,"bbox":null,"value":null}',
    '{"action_type":"CLICK","target":null,"bbox":[NaN,0,1,1],"value":null}',
])
def test_malformed_json_has_no_fabricated_action(raw):
    with pytest.raises(ActionParseError) as caught:
        parse_action(raw, observation(), interface_version=3)
    assert not hasattr(caught.value, 'proposed_action')
    assert caught.value.diagnostic['raw_response'] == raw


def test_coordinate_target_and_explicit_missing_argument_are_not_invented():
    issued = proposal('CLICK', None, None, [.5, .5, .1, .1])
    assert parse(issued)[0] == issued
    assert parse(proposal(value=None))[0]['value'] is None
    with pytest.raises(TargetProposalResolutionError):
        parse(proposal('CLICK', None, 'Field'))
    with pytest.raises(ActionParseError):
        parse_action(json.dumps(proposal()), observation())  # v2 remains unchanged.


def build_actor(tmp_path, system, raw=None):
    seen = []
    def generate(**kwargs):
        seen.append(kwargs)
        return raw if raw is not None else json.dumps(proposal())
    trained = CallablePolicyAdapter(
        policy_id='trained', policy_version='v1', kind=PolicyKind.TRAINED,
        checkpoint_sha256='b' * 64,
        action_predictor=lambda t, o, rng: replace(advice(o), decision_id=system, latency_ms=float(system[-1]) + 1),
    )
    actor = HybridActionPolicy(
        system=system, episode_id=f'campaign:{system}', base=NS(_generate=generate),
        evidence_dir=tmp_path / system, trained_policy=None if system == 'H0' else trained,
        interface_version=3,
    )
    return actor, seen


def test_normal_h2_h3_equal_inputs_confidence_label_and_raw_advice_preservation(tmp_path):
    contexts = []; receipts = []
    for system in ('H0', 'H1', 'H2', 'H3'):
        actor, seen = build_actor(tmp_path, system)
        obs = observation(oid=f'campaign:{system}:obs:1')
        decision = actor.predict_action(task(obs), obs, rng=random.Random(42))
        assert decision.parameter_hints['text'] == proposal()['value']
        assert decision.parameter_hints['target'] == 'input#o1:c0'
        assert decision.parameter_hints['resolved_control_id'] == 'o1:c0'
        contexts.append(json.loads(seen[0]['suffix'].split(': ', 1)[1]))
        receipts.append(json.loads(next(actor.evidence_dir.glob('*.json')).read_text()))
    assert contexts[1] == contexts[2] == contexts[3]
    assert {k: v for k, v in contexts[1].items() if k != 'trained_advice'} == contexts[0]
    assert 'campaign:H' not in json.dumps(contexts)
    assert contexts[2]['trained_advice']['action_class_confidence'] == .85
    assert 'grounding_confidence' not in contexts[2]['trained_advice']
    assert receipts[2]['trained_advice']['grounding_confidence'] == .85
    assert receipts[2]['trained_advice'] != receipts[3]['trained_advice']
    assert receipts[2]['generation_metadata']['available'] is False


def test_rejection_receipt_and_correction_feedback_preserve_raw_text(tmp_path):
    raw = json.dumps(proposal(target='button#o1:c0', value=' campaign:H2 exact '))
    actor, _ = build_actor(tmp_path, 'H2', raw)
    obs = observation(oid='campaign:H2:obs:1')
    with pytest.raises(TargetProposalResolutionError) as caught:
        actor.predict_action(task(obs), obs, rng=random.Random(42))
    receipt = json.loads(next(actor.evidence_dir.glob('*.json')).read_text())
    assert receipt['raw_response'] == raw and receipt['proposed_action'] == json.loads(raw)
    feedback = feedback_context_v3(dict(stage='target_resolution', code=caught.value.diagnostic_code, detail=str(caught.value)))
    assert feedback['diagnostic']['raw_response'] == raw
    assert feedback['diagnostic']['rejected_proposal']['value'] == ' campaign:H2 exact '


def test_causal_history_preserves_issued_values_with_distinct_record_shape():
    obs, actions = with_history()
    raw = HybridActionContext('episode', obs.observation_id, actions).project(obs)
    raw['completed_actions'][0]['parameters']['target_stable_identity'] = {'source_id': 'audit-only'}
    clean = semantic_context(raw, episode_id='episode', interface_version=3)
    assert [e['issued_arguments']['text'] for e in clean['causal_history']] == [' Exact 1 ', ' Exact 2 ']
    assert [e['issued_arguments']['target_control_id'] for e in clean['causal_history']] == ['o1:c0', 'o2:c0']
    assert 'completed_actions' not in clean and 'target_stable_identity' not in json.dumps(clean)
    assert 'action_type' not in clean['causal_history'][0]
    assert clean['causal_history'][0]['observed_result']['execution_status'] == 'executed'
    assert raw['completed_actions'][0]['parameters']['target_stable_identity']


def test_recovery_v3_uses_one_shared_control_projection_and_isolates_memory(monkeypatch):
    from web_agent.runtime.causal_recovery_planner import CausalRecoveryRuntime
    from web_agent.runtime.named_target_policy import NamedTargetRuntime
    from web_agent.runtime.contracts import RecoveryStrategy
    monkeypatch.setattr(NamedTargetRuntime, '__init__', lambda *a, **k: None)
    seen = []
    monkeypatch.setattr(NamedTargetRuntime, 'predict_action', lambda self, *a: seen.append(self._context_suffix))
    obs, actions = with_history()
    normal = HybridActionContext('episode', obs.observation_id, actions).project(obs)
    bound = NS(task_id=obs.task_id, observation_id=obs.observation_id, incident_id='incident',
               episode_id='episode', completed_actions=normal['completed_actions'], previous_attempt=None,
               record_sha256='a' * 64)
    runtime = CausalRecoveryRuntime(hybrid_interface_version=3, use_step_request_feedback=True)
    for memory in ((), (NS(to_dict=lambda: {'recovery_action': 'TYPE', 'recovery_action_value': None, 'reflection': None}),)):
        decision = NS(strategy=RecoveryStrategy.REPLAN, diagnosis='observed rejection',
                      incident_id='incident', memory_experiences=memory)
        runtime.predict_recovery(task(obs), obs, decision, actions[-1], None, planner_context=bound)
    contexts = [json.loads(s.split(': ', 1)[1]) for s in seen]
    assert all('\nnext_action_request:' not in s and '\ncurrent_controls:' not in s for s in seen)
    assert contexts[0]['current_controls'] == semantic_context(normal, episode_id='episode', interface_version=3)['current_controls']
    assert contexts[1]['retrieved_training_examples'][0]['recovery_action_value'] is None
    contexts[1].pop('retrieved_training_examples'); contexts[1].pop('memory_usage')
    assert contexts[0] == contexts[1]


@pytest.mark.parametrize('tokens,eos,reason,capped', [
    ([10, 11, 99], 99, 'eos', False), ([10] * 128, [98, 99], 'max_new_tokens', True),
    ([10] * 127 + [99], 99, 'eos', True), ([10, 11], None, 'other', False),
])
def test_metadata_uses_actual_suffix_tokens_without_changing_decoder(monkeypatch, tokens, eos, reason, capped):
    from web_agent.runtime import qwen2vl_pc01 as module
    import numpy as np
    monkeypatch.setattr(module, '_load_image', lambda obs: 'same image')
    calls = []
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(('template', messages, kwargs)); return 'same template'
        def __call__(self, **kwargs):
            calls.append(('encode', kwargs)); return {'input_ids': np.array([[1, 2]])}
        def batch_decode(self, suffix, **kwargs):
            assert suffix.tolist() == [tokens]
            calls.append(('decode', kwargs)); return [' Exact output\n']
    def generate(**kwargs):
        calls.append(('generate', {k: v for k, v in kwargs.items() if k != 'input_ids'}))
        return np.array([[1, 2] + tokens])
    model = NS(training=False, parameters=lambda: iter([NS(device='cpu')]), generate=generate,
               generation_config=NS(eos_token_id=eos))
    runtime = module._UnadaptedBaseRuntime(model=model, processor=Processor(), torch=NS(inference_mode=nullcontext))
    obs = observation(); kwargs = dict(prompt='same prompt', task=task(obs), observation=obs, suffix='same suffix')
    text = runtime._generate(**kwargs); old_calls = deepcopy(calls); calls.clear()
    instrumented, metadata = runtime._generate_with_metadata(**kwargs)
    assert instrumented == text == ' Exact output\n' and calls == old_calls
    assert metadata['generated_token_count'] == len(tokens)
    assert metadata['stop_reason'] == reason and metadata['token_cap_reached'] is capped
    assert metadata['max_new_tokens'] == 128
    assert next(c[1] for c in calls if c[0] == 'generate') == dict(module.REGISTERED_GENERATION_KWARGS)
