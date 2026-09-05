from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import random

import pytest

from web_agent.benchmarks.fixture import DeterministicFixtureAdapter
from web_agent.runtime.action_parameters import (
    ActionParameterProvider,
    CallableActionParameterProvider,
    DeterministicParameterProvider,
    HybridParameterProvider,
    HybridParameterResolutionError,
    ParameterResolutionError,
    ParameterProvider,
    build_registered_hybrid_parameter_provider,
    raw_prompt_sha256,
    validate_action_parameters,
    validate_observation_bound_action_parameters,
)
from web_agent.runtime.contracts import (
    ActionParameters,
    ActionType,
    ConcreteAction,
    PolicyObservation,
    PreActionDecision,
    TaskSpecification,
    probability_map,
)
from web_agent.runtime.executor import Executor
from web_agent.runtime.protocol import REGISTERED_BUDGETS


SHA = "a" * 64
ROOT = Path(__file__).resolve().parents[2]


def test_parameter_provider_is_the_public_name_for_the_single_provider_contract():
    assert ParameterProvider is ActionParameterProvider


def _task() -> TaskSpecification:
    return TaskSpecification(
        task_id="fixture-six-actions",
        goal="complete the deterministic fixture",
        benchmark_id="fixture",
        benchmark_version="v1",
        start_state_id="s0",
    )


def _observation() -> PolicyObservation:
    return PolicyObservation(
        task_id="fixture-six-actions",
        goal="complete the deterministic fixture",
        observation_id="obs-1",
        screenshot_sha256=SHA,
        screenshot_path=None,
        width=1280,
        height=720,
        url="fixture://state/0",
        title="fixture",
        current_page_state={
            "visible": True,
            "observable_select_controls": [
                {
                    "target_bbox": [0.1, 0.1, 0.2, 0.1],
                    "candidate_options": ["other", "fixture-option"],
                }
            ],
        },
    )


def _decision(action_type: ActionType, hints: dict, bbox=(0.1, 0.1, 0.2, 0.1)):
    return PreActionDecision(
        decision_id=f"decision-{action_type.value}",
        observation_id="obs-1",
        action_type=action_type,
        action_probabilities=probability_map([item.value for item in ActionType], action_type.value),
        bbox=bbox if action_type in {ActionType.CLICK, ActionType.TYPE, ActionType.SELECT} else None,
        grounding_confidence=1.0,
        confidence_before=1.0,
        input_observation_ids=("obs-1",),
        policy_id="fixture-policy",
        policy_version="v1",
        parameter_hints=hints,
    )


@pytest.mark.parametrize(
    ("action_type", "hints", "required"),
    [
        (
            ActionType.CLICK,
            {},
            {"target_x", "target_y", "target_bbox", "button", "click_count"},
        ),
        (
            ActionType.TYPE,
            {"text": "fixture"},
            {"target_x", "target_y", "target_bbox", "text"},
        ),
        (
            ActionType.SELECT,
            {
                "option": "fixture-option",
                "candidate_options": ["other", "fixture-option"],
            },
            {
                "target_x",
                "target_y",
                "target_bbox",
                "option",
                "candidate_options",
            },
        ),
        (
            ActionType.SCROLL,
            {"direction": "down", "amount": 0.5},
            {"direction", "amount", "container"},
        ),
        (ActionType.NAVIGATE, {"url": "fixture://next"}, {"url"}),
        (ActionType.PRESS_KEY, {"key": "ENTER"}, {"key"}),
    ],
)
def test_deterministic_provider_supports_all_registered_actions(action_type, hints, required):
    result = DeterministicParameterProvider().resolve(
        _task(), _observation(), _decision(action_type, hints), rng=random.Random(1)
    )
    assert result.action_type is action_type
    assert required.issubset(result.values)
    assert result.resolution_source == "deterministic"


def test_hybrid_uses_one_frozen_fallback_without_changing_action_class():
    fallback = CallableActionParameterProvider(
        provider_id="frozen-base",
        provider_version="v1",
        prompt_sha256=SHA,
        resolver=lambda task, observation, decision, rng: {
            "target_x": 0.2,
            "target_y": 0.15,
            "target_bbox": [0.1, 0.1, 0.2, 0.1],
            "text": "fallback text",
        },
    )
    provider = HybridParameterProvider(
        deterministic=DeterministicParameterProvider(),
        frozen_base_fallback=fallback,
        prompt_sha256=SHA,
        decoding_parameters={"temperature": 0.0},
    )
    result = provider.resolve(
        _task(), _observation(), _decision(ActionType.TYPE, {}), rng=random.Random(4)
    )
    assert result.resolution_source == "frozen_base_fallback"
    assert result.attempted_sources == ("deterministic", "frozen_base_fallback")
    assert result.action_type is ActionType.TYPE
    assert result.provider_id == provider.provider_id
    assert result.provider_version == provider.provider_version
    assert result.prompt_sha256 == SHA
    assert result.decoding_parameters == {"temperature": 0.0}
    assert result.resolution_trace is not None
    assert result.latency_ms == result.resolution_trace.latency_ms
    assert [attempt.source for attempt in result.resolution_trace.attempts] == [
        "deterministic",
        "frozen_base_fallback",
    ]
    assert [attempt.status for attempt in result.resolution_trace.attempts] == [
        "REJECTED",
        "RESOLVED",
    ]
    assert result.resolution_trace.latency_ms >= sum(
        attempt.latency_ms for attempt in result.resolution_trace.attempts
    )


def test_registered_hybrid_uses_raw_prompt_file_sha256_not_json_string_hash():
    prompt_path = (
        ROOT / "configs" / "eval" / "table2" / "prompts"
        / "parameter_provider_v1.txt"
    )
    raw = prompt_path.read_bytes()
    provider = build_registered_hybrid_parameter_provider(
        fallback_resolver=lambda task, observation, decision, rng: {
            "target_x": 0.2,
            "target_y": 0.2,
        },
        fallback_policy_id="selected-unadapted-backbone",
        fallback_policy_version="v1",
        prompt_bytes=raw,
        decoding_parameters={"temperature": 0.0},
    )
    expected = hashlib.sha256(raw).hexdigest()
    assert provider.prompt_sha256 == expected
    assert provider.frozen_base_fallback.prompt_sha256 == expected
    assert raw_prompt_sha256(raw) == expected

    # JSON-string hashing adds quotes/escapes and is a distinct, forbidden
    # identity even when the visible prompt text is unchanged.
    from web_agent.runtime.contracts import canonical_sha256

    assert canonical_sha256(raw.decode("utf-8")) != expected


def test_registered_hybrid_rejects_ambiguous_prompt_sources():
    arguments = {
        "fallback_resolver": lambda task, observation, decision, rng: {},
        "fallback_policy_id": "selected-unadapted-backbone",
        "fallback_policy_version": "v1",
        "decoding_parameters": {},
    }
    with pytest.raises(ValueError, match="exactly one"):
        build_registered_hybrid_parameter_provider(**arguments)
    with pytest.raises(ValueError, match="exactly one"):
        build_registered_hybrid_parameter_provider(
            **arguments,
            prompt_text="same",
            prompt_bytes=b"same",
        )


def test_double_provider_rejection_consumes_exactly_one_executor_step():
    rejecting = CallableActionParameterProvider(
        provider_id="frozen-base",
        provider_version="v1",
        prompt_sha256=SHA,
        resolver=lambda task, observation, decision, rng: {},
    )
    provider = HybridParameterProvider(
        deterministic=DeterministicParameterProvider(),
        frozen_base_fallback=rejecting,
    )
    with pytest.raises(HybridParameterResolutionError) as caught:
        provider.resolve(
            _task(), _observation(), _decision(ActionType.TYPE, {}), rng=random.Random(4)
        )
    assert len(caught.value.attempts) == 2
    assert caught.value.trace.resolved is False
    assert caught.value.trace.resolution_source is None
    assert caught.value.total_latency_ms == caught.value.trace.latency_ms
    assert all(attempt.latency_ms >= 0.0 for attempt in caught.value.attempts)
    assert caught.value.trace.latency_ms >= sum(
        attempt.latency_ms for attempt in caught.value.attempts
    )

    executor = Executor(DeterministicFixtureAdapter(), budgets=REGISTERED_BUDGETS)
    executor.reset(_task(), episode_id="episode-provider-reject", seed=42)
    action = ConcreteAction(
        action_id="rejected-provider-request",
        source_decision_id="rejected-provider-decision",
        action_type=ActionType.TYPE,
        parameters={},
        bbox=(0.1, 0.1, 0.2, 0.1),
    )
    result = executor.reject_unresolved_request(
        action=action,
        reason=str(caught.value),
    )
    assert result.executor_step == 1
    assert executor.steps_used == 1
    assert result.status.value == "rejected"


@pytest.mark.parametrize(
    ("action_type", "values", "bbox"),
    [
        (
            ActionType.CLICK,
            {
                "target_x": True,
                "target_y": 0.5,
                "target_bbox": [0.0, 0.0, 1.0, 1.0],
                "button": "left",
                "click_count": 1,
            },
            (0.0, 0.0, 1.0, 1.0),
        ),
        (
            ActionType.CLICK,
            {
                "target_x": 0.5,
                "target_y": False,
                "target_bbox": [0.0, 0.0, 1.0, 1.0],
                "button": "left",
                "click_count": 1,
            },
            (0.0, 0.0, 1.0, 1.0),
        ),
        (
            ActionType.SELECT,
            {
                "target_x": 0.5,
                "target_y": 0.5,
                "target_bbox": [0.0, 0.0, 1.0, 1.0],
                "option": True,
                "candidate_options": [True],
            },
            (0.0, 0.0, 1.0, 1.0),
        ),
        (
            ActionType.SCROLL,
            {"direction": "down", "amount": True, "container": "viewport"},
            None,
        ),
    ],
)
def test_boolean_values_cannot_masquerade_as_numeric_parameters(
    action_type, values, bbox
):
    with pytest.raises(ParameterResolutionError):
        validate_action_parameters(action_type, values, bbox)


def _canonical_values(action_type: ActionType):
    target = {
        "target_x": 0.2,
        "target_y": 0.15,
        "target_bbox": [0.1, 0.1, 0.2, 0.1],
    }
    return {
        ActionType.CLICK: {
            **target,
            "button": "left",
            "click_count": 1,
        },
        ActionType.TYPE: {**target, "text": "fixture"},
        ActionType.SELECT: {
            **target,
            "option": "fixture-option",
            "candidate_options": ["other", "fixture-option", 3, 4.5],
        },
        ActionType.SCROLL: {
            "direction": "down",
            "amount": 0.75,
            "container": "viewport",
        },
        ActionType.NAVIGATE: {"url": "https://example.test/path?q=1"},
        ActionType.PRESS_KEY: {"key": "ALT+LEFT"},
    }[action_type]


@pytest.mark.parametrize("action_type", tuple(ActionType))
def test_registered_parameter_schemas_accept_only_canonical_shapes(action_type):
    bbox = (
        (0.1, 0.1, 0.2, 0.1)
        if action_type in {ActionType.CLICK, ActionType.TYPE, ActionType.SELECT}
        else None
    )
    validate_action_parameters(action_type, _canonical_values(action_type), bbox)


@pytest.mark.parametrize(
    ("action_type", "values", "bbox"),
    [
        (
            ActionType.CLICK,
            {
                **_canonical_values(ActionType.CLICK),
                "button": "primary",
            },
            (0.1, 0.1, 0.2, 0.1),
        ),
        (
            ActionType.CLICK,
            {
                **_canonical_values(ActionType.CLICK),
                "click_count": 1.0,
            },
            (0.1, 0.1, 0.2, 0.1),
        ),
        (
            ActionType.CLICK,
            {
                **_canonical_values(ActionType.CLICK),
                "click_count": 4,
            },
            (0.1, 0.1, 0.2, 0.1),
        ),
        (
            ActionType.CLICK,
            {
                **_canonical_values(ActionType.CLICK),
                "target_bbox": [0.0, 0.0, 1.0, 1.0],
            },
            (0.1, 0.1, 0.2, 0.1),
        ),
        (
            ActionType.CLICK,
            {
                **_canonical_values(ActionType.CLICK),
                "target_x": 0.9,
            },
            (0.1, 0.1, 0.2, 0.1),
        ),
        (
            ActionType.TYPE,
            {**_canonical_values(ActionType.TYPE), "text": 7},
            (0.1, 0.1, 0.2, 0.1),
        ),
        (
            ActionType.TYPE,
            {**_canonical_values(ActionType.TYPE), "text": ""},
            (0.1, 0.1, 0.2, 0.1),
        ),
        (
            ActionType.SELECT,
            {
                **_canonical_values(ActionType.SELECT),
                "candidate_options": [],
            },
            (0.1, 0.1, 0.2, 0.1),
        ),
        (
            ActionType.SELECT,
            {
                **_canonical_values(ActionType.SELECT),
                "candidate_options": ["other"],
            },
            (0.1, 0.1, 0.2, 0.1),
        ),
        (
            ActionType.SELECT,
            {
                **_canonical_values(ActionType.SELECT),
                "option": 3,
                "candidate_options": [3.0],
            },
            (0.1, 0.1, 0.2, 0.1),
        ),
        (
            ActionType.SELECT,
            {
                **_canonical_values(ActionType.SELECT),
                "option": "fixture-option",
                "candidate_options": ["fixture-option", float("nan")],
            },
            (0.1, 0.1, 0.2, 0.1),
        ),
        (
            ActionType.SCROLL,
            {**_canonical_values(ActionType.SCROLL), "amount": float("inf")},
            None,
        ),
        (
            ActionType.SCROLL,
            {**_canonical_values(ActionType.SCROLL), "amount": 1.01},
            None,
        ),
        (
            ActionType.SCROLL,
            {**_canonical_values(ActionType.SCROLL), "amount": 0.25},
            None,
        ),
        (
            ActionType.SCROLL,
            {**_canonical_values(ActionType.SCROLL), "container": ""},
            None,
        ),
        (
            ActionType.SCROLL,
            {**_canonical_values(ActionType.SCROLL), "container": "page"},
            None,
        ),
        (
            ActionType.NAVIGATE,
            {"url": "javascript:alert(1)"},
            None,
        ),
        (
            ActionType.NAVIGATE,
            {"url": "https://user:secret@example.test/path"},
            None,
        ),
        (
            ActionType.NAVIGATE,
            {"url": "https:///missing-host"},
            None,
        ),
        (
            ActionType.PRESS_KEY,
            {"key": "CTRL+ALT+DELETE"},
            None,
        ),
        (
            ActionType.PRESS_KEY,
            {"key": "enter"},
            None,
        ),
        (
            ActionType.CLICK,
            {**_canonical_values(ActionType.CLICK), "oracle_target": "hidden"},
            (0.1, 0.1, 0.2, 0.1),
        ),
    ],
)
def test_prior_accepted_bad_parameter_values_fail_closed(action_type, values, bbox):
    with pytest.raises(ParameterResolutionError):
        validate_action_parameters(action_type, values, bbox)


@pytest.mark.parametrize(
    ("action_type", "values"),
    [
        (
            ActionType.CLICK,
            {**_canonical_values(ActionType.CLICK), "button": "primary"},
        ),
        (
            ActionType.TYPE,
            {**_canonical_values(ActionType.TYPE), "text": 7},
        ),
        (
            ActionType.SELECT,
            {
                **_canonical_values(ActionType.SELECT),
                "candidate_options": ["different"],
            },
        ),
        (
            ActionType.SCROLL,
            {**_canonical_values(ActionType.SCROLL), "amount": float("nan")},
        ),
        (ActionType.NAVIGATE, {"url": "file:///etc/passwd"}),
        (ActionType.PRESS_KEY, {"key": "CTRL+X"}),
    ],
)
def test_frozen_fallback_rejects_the_same_noncanonical_values(action_type, values):
    provider = CallableActionParameterProvider(
        provider_id="frozen-base",
        provider_version="v1",
        prompt_sha256=SHA,
        resolver=lambda task, observation, decision, rng: values,
    )
    with pytest.raises(ParameterResolutionError):
        provider.resolve(
            _task(),
            _observation(),
            _decision(action_type, {}),
            rng=random.Random(1),
        )


@pytest.mark.parametrize(
    ("action_type", "hints"),
    [
        (ActionType.CLICK, {"button": "primary"}),
        (ActionType.TYPE, {"text": 7}),
        (
            ActionType.SELECT,
            {"option": "missing", "candidate_options": ["different"]},
        ),
        (ActionType.SCROLL, {"amount": float("nan")}),
        (ActionType.NAVIGATE, {"url": "data:text/plain,unsafe"}),
        (ActionType.PRESS_KEY, {"key": "CTRL+X"}),
    ],
)
def test_deterministic_provider_rejects_noncanonical_hints(action_type, hints):
    with pytest.raises(ParameterResolutionError):
        DeterministicParameterProvider().resolve(
            _task(),
            _observation(),
            _decision(action_type, hints),
            rng=random.Random(1),
        )


@pytest.mark.parametrize("amount", [0.5, 0.75])
def test_scroll_accepts_each_registered_amount_for_the_viewport(amount):
    validate_action_parameters(
        ActionType.SCROLL,
        {
            "direction": "down",
            "amount": amount,
            "container": "viewport",
        },
        None,
    )


def _select_observation(controls):
    return replace(
        _observation(),
        current_page_state={
            "visible": True,
            "observable_select_controls": controls,
        },
    )


def test_select_static_validation_remains_available_for_replay_without_observation():
    values = _canonical_values(ActionType.SELECT)
    validate_action_parameters(
        ActionType.SELECT,
        values,
        (0.1, 0.1, 0.2, 0.1),
    )


def test_select_provider_validation_binds_candidates_to_grounded_observation():
    values = _canonical_values(ActionType.SELECT)
    validate_observation_bound_action_parameters(
        ActionType.SELECT,
        values,
        (0.1, 0.1, 0.2, 0.1),
        observation=_select_observation(
            [
                {
                    "target_bbox": [0.1, 0.1, 0.2, 0.1],
                    "candidate_options": list(values["candidate_options"]),
                }
            ]
        ),
    )


@pytest.mark.parametrize(
    "controls",
    [
        [],
        [
            {
                "target_bbox": [0.6, 0.6, 0.2, 0.1],
                "candidate_options": ["other", "fixture-option", 3, 4.5],
            }
        ],
        [
            {
                "target_bbox": [0.1, 0.1, 0.2, 0.1],
                "candidate_options": ["fixture-option", "other", 3, 4.5],
            }
        ],
        [
            {
                "target_bbox": [0.1, 0.1, 0.2, 0.1],
                "candidate_options": ["other", "fixture-option", 3, 4.5],
            },
            {
                "target_bbox": [0.1, 0.1, 0.2, 0.1],
                "candidate_options": ["other", "fixture-option", 3, 4.5],
            },
        ],
        [
            {
                "target_bbox": [0.1, 0.1, 0.2, 0.1],
                "candidate_options": ["other", "fixture-option", 3, 4.5],
                "selected_option": "fixture-option",
            }
        ],
    ],
)
def test_select_observation_evidence_mismatches_fail_closed(controls):
    with pytest.raises(ParameterResolutionError):
        validate_observation_bound_action_parameters(
            ActionType.SELECT,
            _canonical_values(ActionType.SELECT),
            (0.1, 0.1, 0.2, 0.1),
            observation=_select_observation(controls),
        )


def test_select_observation_requires_the_registered_page_state_field():
    observation = replace(
        _observation(),
        current_page_state={"visible": True},
    )
    with pytest.raises(ParameterResolutionError, match="lacks observable_select_controls"):
        validate_observation_bound_action_parameters(
            ActionType.SELECT,
            _canonical_values(ActionType.SELECT),
            (0.1, 0.1, 0.2, 0.1),
            observation=observation,
        )


def test_deterministic_select_rejects_candidates_not_shown_at_grounded_control():
    with pytest.raises(ParameterResolutionError, match="grounded observable control"):
        DeterministicParameterProvider().resolve(
            _task(),
            _observation(),
            _decision(
                ActionType.SELECT,
                {
                    "option": "fixture-option",
                    "candidate_options": ["fixture-option", "hallucinated"],
                },
            ),
            rng=random.Random(1),
        )


def test_frozen_fallback_select_rejects_candidates_not_shown_at_grounded_control():
    provider = CallableActionParameterProvider(
        provider_id="frozen-base",
        provider_version="v1",
        prompt_sha256=SHA,
        resolver=lambda task, observation, decision, rng: {
            "target_x": 0.2,
            "target_y": 0.15,
            "target_bbox": [0.1, 0.1, 0.2, 0.1],
            "option": "fixture-option",
            "candidate_options": ["fixture-option", "hallucinated"],
        },
    )
    with pytest.raises(ParameterResolutionError, match="grounded observable control"):
        provider.resolve(
            _task(),
            _observation(),
            _decision(ActionType.SELECT, {}),
            rng=random.Random(1),
        )


class _UncheckedParameterProvider(ActionParameterProvider):
    provider_version = "v1"
    prompt_sha256 = SHA
    frozen = True
    decoding_parameters = {}

    def __init__(self, provider_id, policy_source, values):
        self.provider_id = provider_id
        self.policy_source = policy_source
        self.values = values

    def resolve(self, task, observation, decision, *, rng):
        del task, observation, rng
        return ActionParameters(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            action_type=decision.action_type,
            values=dict(self.values),
            prompt_sha256=self.prompt_sha256,
            resolution_source=self.provider_id,
        )


def test_hybrid_boundary_revalidates_select_evidence_from_custom_stages():
    values = {
        "target_x": 0.2,
        "target_y": 0.15,
        "target_bbox": [0.1, 0.1, 0.2, 0.1],
        "option": "hallucinated",
        "candidate_options": ["hallucinated"],
    }
    provider = HybridParameterProvider(
        deterministic=_UncheckedParameterProvider(
            "unchecked-deterministic",
            "deterministic",
            values,
        ),
        frozen_base_fallback=_UncheckedParameterProvider(
            "unchecked-fallback",
            "selected_backbone_unadapted",
            values,
        ),
    )
    with pytest.raises(HybridParameterResolutionError) as caught:
        provider.resolve(
            _task(),
            _observation(),
            _decision(ActionType.SELECT, {}),
            rng=random.Random(1),
        )
    assert [attempt.status for attempt in caught.value.attempts] == [
        "REJECTED",
        "REJECTED",
    ]
