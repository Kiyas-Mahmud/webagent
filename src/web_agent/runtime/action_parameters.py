"""Shared, oracle-blind conversion from policy decisions to concrete actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
import hashlib
import math
import random
from time import perf_counter
from typing import Mapping
from urllib.parse import urlparse

from web_agent.runtime.contracts import (
    ActionParameters,
    ActionType,
    ConcreteAction,
    JsonValue,
    ParameterResolutionAttempt,
    ParameterResolutionTrace,
    PolicyObservation,
    PreActionDecision,
    RuntimeTaskView,
    TaskSpecification,
    canonical_sha256,
    detached_record_copy,
    runtime_task_view,
)
from web_agent.runtime.observation import (
    assert_oracle_blind_mapping,
    safe_provider_context,
)
from web_agent.runtime.model_calls import record_model_call
from web_agent.runtime.protocol import (
    REGISTERED_INVALID_PROVIDER_OUTPUT_POLICY,
    REGISTERED_PARAMETER_PROVIDER_FALLBACK,
    REGISTERED_PARAMETER_PROVIDER_MODE,
)


class ParameterResolutionError(ValueError):
    """A provider could not resolve the selected action without hidden rescue."""


REGISTERED_CLICK_BUTTONS = ("left", "middle", "right")
REGISTERED_CLICK_COUNTS = (1, 2, 3)
REGISTERED_SCROLL_DIRECTIONS = ("up", "down", "left", "right")
REGISTERED_SCROLL_AMOUNTS = (0.5, 0.75)
REGISTERED_SCROLL_CONTAINER = "viewport"
REGISTERED_NAVIGATION_SCHEMES = ("http", "https", "fixture")
OBSERVABLE_SELECT_CONTROLS_FIELD = "observable_select_controls"
REGISTERED_PRESS_KEYS = (
    "ENTER",
    "TAB",
    "ESCAPE",
    "ARROWDOWN",
    "ARROWUP",
    "SPACE",
    "ALT+LEFT",
)


def raw_prompt_sha256(prompt: str | bytes) -> str:
    """Hash the exact UTF-8 prompt bytes used by the frozen campaign.

    Prompt files are frozen and attested with a raw file SHA-256.  Hashing the
    prompt as a JSON string would add quotes/escapes and produce a different
    identity, so runtime construction deliberately uses the same byte-level
    convention as protocol freezing and package validation.
    """

    raw = prompt if isinstance(prompt, bytes) else prompt.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


ResolutionAttempt = ParameterResolutionAttempt


class HybridParameterResolutionError(ParameterResolutionError):
    def __init__(self, message: str, trace: ParameterResolutionTrace):
        super().__init__(message)
        self.trace = trace
        self.attempts = trace.attempts
        self.total_latency_ms = trace.latency_ms


class ActionParameterProvider(ABC):
    provider_id: str
    provider_version: str
    prompt_sha256: str
    frozen: bool = True
    policy_source: str = "selected_backbone_unadapted"
    decoding_parameters: Mapping[str, JsonValue] = {}

    @abstractmethod
    def resolve(
        self,
        task: RuntimeTaskView,
        observation: PolicyObservation,
        decision: PreActionDecision,
        *,
        rng: random.Random,
    ) -> ActionParameters:
        raise NotImplementedError


# Public plan-level name retained as an exact compatibility alias so there is
# one provider hierarchy, not a competing implementation.
ParameterProvider = ActionParameterProvider


ProviderCallable = Callable[
    [RuntimeTaskView, PolicyObservation, PreActionDecision, random.Random],
    Mapping[str, JsonValue],
]


@dataclass(frozen=True, slots=True)
class CallableActionParameterProvider(ActionParameterProvider):
    """Wrap a frozen provider while preserving the common E0-E3 interface."""

    provider_id: str
    provider_version: str
    prompt_sha256: str
    resolver: ProviderCallable
    frozen: bool = True
    policy_source: str = "selected_backbone_unadapted"
    decoding_parameters: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider_id or not self.provider_version or not self.prompt_sha256:
            raise ValueError("provider identity, version, and prompt hash are required")
        if not self.frozen or self.policy_source != "selected_backbone_unadapted":
            raise ValueError("fallback provider must be the frozen unadapted base policy")

    def resolve(
        self,
        task: TaskSpecification | RuntimeTaskView,
        observation: PolicyObservation,
        decision: PreActionDecision,
        *,
        rng: random.Random,
    ) -> ActionParameters:
        started = perf_counter()
        task_view = runtime_task_view(task)
        if observation.task_id != task_view.task_id:
            raise ParameterResolutionError("provider observation belongs to another task")
        if decision.observation_id != observation.observation_id:
            raise ParameterResolutionError("provider decision cites another observation")
        safe_provider_context(observation, decision.parameter_hints)
        record_model_call(
            stage="action_parameter_fallback",
            component_id=f"{self.provider_id}@{self.provider_version}",
        )
        protected = (task_view, observation, decision)
        protected_hashes = tuple(item.record_sha256 for item in protected)
        callback_inputs = tuple(detached_record_copy(item) for item in protected)
        callback_hashes = tuple(item.record_sha256 for item in callback_inputs)
        callback_error: BaseException | None = None
        raw_values: object | None = None
        try:
            raw_values = self.resolver(
                callback_inputs[0],  # type: ignore[arg-type]
                callback_inputs[1],  # type: ignore[arg-type]
                callback_inputs[2],  # type: ignore[arg-type]
                rng,
            )
        except BaseException as exc:
            callback_error = exc
        mutated: list[str] = []
        for label, records, hashes in (
            ("runtime", protected, protected_hashes),
            ("callback", callback_inputs, callback_hashes),
        ):
            for index, (item, expected) in enumerate(zip(records, hashes)):
                try:
                    actual = item.record_sha256
                except Exception:
                    actual = "<unhashable>"
                if actual != expected:
                    mutated.append(f"{label}[{index}]")
        if mutated:
            mutation = ParameterResolutionError(
                "parameter provider mutated protected callback input: "
                + ", ".join(mutated)
            )
            if callback_error is not None:
                raise mutation from callback_error
            raise mutation
        if callback_error is not None:
            raise callback_error
        if not isinstance(raw_values, Mapping):
            raise ParameterResolutionError(
                "frozen parameter provider output must be a JSON object"
            )
        values = dict(raw_values)
        assert_oracle_blind_mapping(values, location="provider_output")
        validate_observation_bound_action_parameters(
            decision.action_type,
            values,
            decision.bbox,
            observation=observation,
        )
        latency_ms = (perf_counter() - started) * 1000.0
        trace = ParameterResolutionTrace(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            attempts=(
                ParameterResolutionAttempt(
                    source=self.provider_id,
                    status="RESOLVED",
                    latency_ms=latency_ms,
                ),
            ),
            resolved=True,
            resolution_source=self.provider_id,
            latency_ms=latency_ms,
        )
        return detached_record_copy(ActionParameters(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            action_type=decision.action_type,
            values=values,
            prompt_sha256=self.prompt_sha256,
            resolution_source=self.provider_id,
            attempted_sources=(self.provider_id,),
            decoding_parameters=dict(self.decoding_parameters),
            latency_ms=latency_ms,
            resolution_trace=trace,
        ))


@dataclass(frozen=True, slots=True)
class DeterministicParameterProvider(ActionParameterProvider):
    """Resolve registered actions from policy hints without an external model."""

    provider_id: str = "deterministic-parameter-provider"
    provider_version: str = "v1"
    prompt_sha256: str = canonical_sha256(
        "deterministic registered action-parameter rules v1"
    )
    permitted_keys: tuple[str, ...] = REGISTERED_PRESS_KEYS
    navigation_schemes: tuple[str, ...] = REGISTERED_NAVIGATION_SCHEMES
    frozen: bool = True
    policy_source: str = "deterministic"
    decoding_parameters: Mapping[str, JsonValue] = field(default_factory=dict)

    def resolve(
        self,
        task: TaskSpecification | RuntimeTaskView,
        observation: PolicyObservation,
        decision: PreActionDecision,
        *,
        rng: random.Random,
    ) -> ActionParameters:
        started = perf_counter()
        del rng  # Rules are deterministic; accepting RNG keeps provider parity.
        task_view = runtime_task_view(task)
        if observation.task_id != task_view.task_id:
            raise ParameterResolutionError("provider observation belongs to another task")
        if decision.observation_id != observation.observation_id:
            raise ParameterResolutionError("provider decision cites another observation")
        safe_provider_context(observation, decision.parameter_hints)
        hints = dict(decision.parameter_hints)
        values: dict[str, JsonValue]
        if decision.action_type is ActionType.CLICK:
            values = _target_values(decision)
            button = hints.get("button", "left")
            if isinstance(button, str):
                button = button.lower()
            values.update(
                {
                    "button": button,
                    "click_count": hints.get("click_count", 1),
                }
            )
        elif decision.action_type is ActionType.TYPE:
            text = hints.get("text")
            if not isinstance(text, str) or not text:
                raise ParameterResolutionError("TYPE requires a non-empty policy text hint")
            values = {**_target_values(decision), "text": text}
        elif decision.action_type is ActionType.SELECT:
            option = hints.get("option")
            candidate_options = hints.get("candidate_options")
            if not isinstance(candidate_options, list):
                raise ParameterResolutionError(
                    "SELECT requires explicit observable candidate_options evidence"
                )
            values = {
                **_target_values(decision),
                "option": option,
                "candidate_options": list(candidate_options),
            }
        elif decision.action_type is ActionType.SCROLL:
            direction = hints.get("direction", "down")
            if isinstance(direction, str):
                direction = direction.lower()
            values = {
                "direction": direction,
                "amount": hints.get("amount", 0.75),
                "container": hints.get("container", REGISTERED_SCROLL_CONTAINER),
            }
        elif decision.action_type is ActionType.NAVIGATE:
            url = hints.get("url")
            if not isinstance(url, str) or not url:
                raise ParameterResolutionError("NAVIGATE requires a URL hint")
            values = {"url": url}
        elif decision.action_type is ActionType.PRESS_KEY:
            key = hints.get("key", "")
            if isinstance(key, str):
                key = key.upper()
            values = {"key": key}
        else:  # pragma: no cover - exhaustive enum guard
            raise ParameterResolutionError(f"unsupported action: {decision.action_type}")
        assert_oracle_blind_mapping(values, location="provider_output")
        validate_observation_bound_action_parameters(
            decision.action_type,
            values,
            decision.bbox,
            observation=observation,
        )
        latency_ms = (perf_counter() - started) * 1000.0
        trace = ParameterResolutionTrace(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            attempts=(
                ParameterResolutionAttempt(
                    source="deterministic",
                    status="RESOLVED",
                    latency_ms=latency_ms,
                ),
            ),
            resolved=True,
            resolution_source="deterministic",
            latency_ms=latency_ms,
        )
        return ActionParameters(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            action_type=decision.action_type,
            values=values,
            prompt_sha256=self.prompt_sha256,
            resolution_source="deterministic",
            attempted_sources=("deterministic",),
            decoding_parameters={},
            latency_ms=latency_ms,
            resolution_trace=trace,
        )


@dataclass(frozen=True, slots=True)
class HybridParameterProvider(ActionParameterProvider):
    """Deterministic extraction followed by one frozen base-policy fallback.

    Both stages operate on the same oracle-blind input and selected action.
    There is no repair loop: after the single fallback rejects, the caller must
    submit one rejected executor request so budget accounting remains honest.
    """

    deterministic: ActionParameterProvider
    frozen_base_fallback: ActionParameterProvider
    provider_id: str = REGISTERED_PARAMETER_PROVIDER_MODE
    provider_version: str = "v1"
    prompt_sha256: str = canonical_sha256("hybrid parameter provider v1")
    frozen: bool = True
    policy_source: str = "deterministic_then_frozen_base_fallback"
    decoding_parameters: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.policy_source != REGISTERED_PARAMETER_PROVIDER_MODE:
            raise ValueError("hybrid provider mode differs from frozen protocol")
        if self.deterministic is self.frozen_base_fallback:
            raise ValueError("hybrid provider stages must be distinct")
        if not self.frozen_base_fallback.prompt_sha256:
            raise ValueError("frozen base fallback requires a registered prompt hash")
        if not self.frozen_base_fallback.frozen:
            raise ValueError("hybrid fallback must be frozen")
        if self.frozen_base_fallback.policy_source != "selected_backbone_unadapted":
            raise ValueError("hybrid fallback must use the unadapted selected backbone")
        if self.frozen_base_fallback.policy_source != REGISTERED_PARAMETER_PROVIDER_FALLBACK:
            raise ValueError("hybrid fallback source differs from frozen protocol")

    def resolve(
        self,
        task: TaskSpecification | RuntimeTaskView,
        observation: PolicyObservation,
        decision: PreActionDecision,
        *,
        rng: random.Random,
    ) -> ActionParameters:
        total_started = perf_counter()
        task_view = runtime_task_view(task)
        if observation.task_id != task_view.task_id:
            raise ParameterResolutionError("provider observation belongs to another task")
        if decision.observation_id != observation.observation_id:
            raise ParameterResolutionError("provider decision cites another observation")
        attempts: list[ResolutionAttempt] = []
        # Each registered stage receives an identical copy of the stage-keyed
        # stream. A rejecting deterministic implementation cannot silently
        # advance randomness before the one frozen-base fallback.
        rng_state = rng.getstate()
        deterministic_rng = random.Random()
        deterministic_rng.setstate(rng_state)
        fallback_rng = random.Random()
        fallback_rng.setstate(rng_state)
        attempt_started = perf_counter()
        try:
            result = self.deterministic.resolve(
                task_view,
                observation,
                decision,
                rng=deterministic_rng,
            )
            if not isinstance(result, ActionParameters):
                raise ParameterResolutionError(
                    "deterministic provider returned an invalid contract"
                )
            if result.action_type is not decision.action_type:
                raise ParameterResolutionError(
                    "deterministic provider changed the selected action class"
                )
            validate_observation_bound_action_parameters(
                decision.action_type,
                result.values,
                decision.bbox,
                observation=observation,
            )
            attempts.append(
                ResolutionAttempt(
                    "deterministic",
                    "RESOLVED",
                    (perf_counter() - attempt_started) * 1000.0,
                )
            )
            total_latency_ms = (perf_counter() - total_started) * 1000.0
            trace = ParameterResolutionTrace(
                provider_id=self.provider_id,
                provider_version=self.provider_version,
                attempts=tuple(attempts),
                resolved=True,
                resolution_source="deterministic",
                latency_ms=total_latency_ms,
            )
            return replace(
                result,
                provider_id=self.provider_id,
                provider_version=self.provider_version,
                prompt_sha256=self.prompt_sha256,
                resolution_source="deterministic",
                attempted_sources=tuple(item.source for item in attempts),
                decoding_parameters=dict(self.decoding_parameters),
                latency_ms=total_latency_ms,
                resolution_trace=trace,
            )
        except ParameterResolutionError as exc:
            attempts.append(
                ResolutionAttempt(
                    "deterministic",
                    "REJECTED",
                    (perf_counter() - attempt_started) * 1000.0,
                    str(exc),
                )
            )

        attempt_started = perf_counter()
        try:
            result = self.frozen_base_fallback.resolve(
                task_view,
                observation,
                decision,
                rng=fallback_rng,
            )
            if not isinstance(result, ActionParameters):
                raise ParameterResolutionError(
                    "frozen base fallback returned an invalid contract"
                )
            if result.action_type is not decision.action_type:
                raise ParameterResolutionError(
                    "frozen base fallback changed the selected action class"
                )
            validate_observation_bound_action_parameters(
                decision.action_type,
                result.values,
                decision.bbox,
                observation=observation,
            )
            attempts.append(
                ResolutionAttempt(
                    "frozen_base_fallback",
                    "RESOLVED",
                    (perf_counter() - attempt_started) * 1000.0,
                )
            )
            total_latency_ms = (perf_counter() - total_started) * 1000.0
            trace = ParameterResolutionTrace(
                provider_id=self.provider_id,
                provider_version=self.provider_version,
                attempts=tuple(attempts),
                resolved=True,
                resolution_source="frozen_base_fallback",
                latency_ms=total_latency_ms,
            )
            return replace(
                result,
                provider_id=self.provider_id,
                provider_version=self.provider_version,
                prompt_sha256=self.prompt_sha256,
                resolution_source="frozen_base_fallback",
                attempted_sources=tuple(item.source for item in attempts),
                decoding_parameters=dict(self.decoding_parameters),
                latency_ms=total_latency_ms,
                resolution_trace=trace,
            )
        except ParameterResolutionError as exc:
            attempts.append(
                ResolutionAttempt(
                    "frozen_base_fallback",
                    "REJECTED",
                    (perf_counter() - attempt_started) * 1000.0,
                    str(exc),
                )
            )
            trace = ParameterResolutionTrace(
                provider_id=self.provider_id,
                provider_version=self.provider_version,
                attempts=tuple(attempts),
                resolved=False,
                resolution_source=None,
                latency_ms=(perf_counter() - total_started) * 1000.0,
            )
            raise HybridParameterResolutionError(
                "both registered parameter-provider stages rejected; no repair allowed",
                trace,
            ) from exc


def build_registered_hybrid_parameter_provider(
    *,
    fallback_resolver: ProviderCallable,
    fallback_policy_id: str,
    fallback_policy_version: str,
    prompt_text: str | None = None,
    prompt_bytes: bytes | None = None,
    decoding_parameters: Mapping[str, JsonValue],
) -> HybridParameterProvider:
    """Bind deterministic extraction to one frozen unadapted-backbone fallback.

    The returned provider has no repair loop. ``EpisodeRunner`` applies the
    registered ``reject_and_consume_executor_step`` policy if both stages reject.
    """

    if REGISTERED_INVALID_PROVIDER_OUTPUT_POLICY != "reject_and_consume_executor_step":
        raise RuntimeError("runtime invalid-output policy registration changed")
    if (prompt_text is None) == (prompt_bytes is None):
        raise ValueError("supply exactly one of prompt_text or prompt_bytes")
    if prompt_bytes is not None:
        raw_prompt = prompt_bytes
    else:
        assert prompt_text is not None
        raw_prompt = prompt_text.encode("utf-8")
    try:
        decoded_prompt = raw_prompt.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("frozen base parameter prompt must be UTF-8") from exc
    if not decoded_prompt.strip():
        raise ValueError("frozen base parameter prompt cannot be empty")
    fallback = CallableActionParameterProvider(
        provider_id=fallback_policy_id,
        provider_version=fallback_policy_version,
        prompt_sha256=raw_prompt_sha256(raw_prompt),
        resolver=fallback_resolver,
        frozen=True,
        policy_source=REGISTERED_PARAMETER_PROVIDER_FALLBACK,
        decoding_parameters=dict(decoding_parameters),
    )
    return HybridParameterProvider(
        deterministic=DeterministicParameterProvider(),
        frozen_base_fallback=fallback,
        provider_id=REGISTERED_PARAMETER_PROVIDER_MODE,
        provider_version="v1",
        prompt_sha256=fallback.prompt_sha256,
        frozen=True,
        policy_source=REGISTERED_PARAMETER_PROVIDER_MODE,
        decoding_parameters=dict(decoding_parameters),
    )


def _target_values(decision: PreActionDecision) -> dict[str, JsonValue]:
    if decision.bbox is None:
        raise ParameterResolutionError(f"{decision.action_type.value} requires a grounded bbox")
    x, y, width, height = _validated_bbox(
        decision.bbox,
        name=f"{decision.action_type.value} decision bbox",
    )
    return {
        "target_x": x + width / 2.0,
        "target_y": y + height / 2.0,
        "target_bbox": [x, y, width, height],
    }


def validate_action_parameters(
    action_type: ActionType,
    values: Mapping[str, JsonValue],
    bbox: tuple[float, float, float, float] | None,
) -> None:
    if not isinstance(values, Mapping):
        raise ParameterResolutionError(
            f"{action_type.value} parameters must be a JSON object"
        )
    required = {
        ActionType.CLICK: {
            "target_x",
            "target_y",
            "target_bbox",
            "button",
            "click_count",
        },
        ActionType.TYPE: {"target_x", "target_y", "target_bbox", "text"},
        ActionType.SELECT: {
            "target_x",
            "target_y",
            "target_bbox",
            "option",
            "candidate_options",
        },
        ActionType.SCROLL: {"direction", "amount", "container"},
        ActionType.NAVIGATE: {"url"},
        ActionType.PRESS_KEY: {"key"},
    }[action_type]
    actual = set(values)
    if actual != required:
        missing = required - actual
        unexpected = actual - required
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unexpected:
            details.append(
                "unexpected=" + repr(sorted(str(name) for name in unexpected))
            )
        raise ParameterResolutionError(
            f"{action_type.value} parameters do not match the registered schema: "
            + ", ".join(details)
        )

    if action_type in {ActionType.CLICK, ActionType.TYPE, ActionType.SELECT}:
        _validate_grounded_target(action_type, values, bbox)

    if action_type is ActionType.CLICK:
        button = values["button"]
        if not isinstance(button, str) or button not in REGISTERED_CLICK_BUTTONS:
            raise ParameterResolutionError(
                "CLICK button must be one of "
                f"{list(REGISTERED_CLICK_BUTTONS)}"
            )
        click_count = values["click_count"]
        if type(click_count) is not int or click_count not in REGISTERED_CLICK_COUNTS:
            raise ParameterResolutionError(
                "CLICK click_count must be an exact integer in "
                f"{list(REGISTERED_CLICK_COUNTS)}"
            )
    elif action_type is ActionType.TYPE:
        text = values["text"]
        if not isinstance(text, str) or not text:
            raise ParameterResolutionError("TYPE text must be a non-empty string")
    elif action_type is ActionType.SELECT:
        option = values["option"]
        _validate_select_option(option, name="SELECT option")
        candidates = values["candidate_options"]
        if not isinstance(candidates, list) or not candidates:
            raise ParameterResolutionError(
                "SELECT candidate_options must be a non-empty list"
            )
        for index, candidate in enumerate(candidates):
            _validate_select_option(
                candidate,
                name=f"SELECT candidate_options[{index}]",
            )
        if not any(_same_select_option(option, candidate) for candidate in candidates):
            raise ParameterResolutionError(
                "SELECT option is not an exact member of candidate_options"
            )
    elif action_type is ActionType.SCROLL:
        direction = values["direction"]
        if (
            not isinstance(direction, str)
            or direction not in REGISTERED_SCROLL_DIRECTIONS
        ):
            raise ParameterResolutionError(
                "SCROLL direction must be one of "
                f"{list(REGISTERED_SCROLL_DIRECTIONS)}"
            )
        amount = _finite_number("SCROLL amount", values["amount"])
        if amount not in REGISTERED_SCROLL_AMOUNTS:
            raise ParameterResolutionError(
                "SCROLL amount must be one of the registered values "
                f"{list(REGISTERED_SCROLL_AMOUNTS)}"
            )
        container = values["container"]
        if container != REGISTERED_SCROLL_CONTAINER:
            raise ParameterResolutionError(
                "SCROLL container must be the registered viewport container"
            )
    elif action_type is ActionType.NAVIGATE:
        _validate_navigation_url(values["url"])
    elif action_type is ActionType.PRESS_KEY:
        key = values["key"]
        if not isinstance(key, str) or key not in REGISTERED_PRESS_KEYS:
            raise ParameterResolutionError(
                "PRESS_KEY key is not in the registered key/chord vocabulary"
            )


def validate_observation_bound_action_parameters(
    action_type: ActionType,
    values: Mapping[str, JsonValue],
    bbox: tuple[float, float, float, float] | None,
    *,
    observation: PolicyObservation,
) -> None:
    """Validate provider output against the exact current causal observation.

    Static replay validation remains in :func:`validate_action_parameters`.
    SELECT additionally needs live evidence: one observable select control must
    match the policy-grounded bbox and expose exactly the candidates claimed by
    the provider.  The observation schema is deliberately narrow so arbitrary
    provider output cannot serve as its own evidence.
    """

    validate_action_parameters(action_type, values, bbox)
    if action_type is not ActionType.SELECT:
        return
    if not isinstance(observation, PolicyObservation):
        raise ParameterResolutionError(
            "observation-bound parameter validation requires PolicyObservation"
        )
    assert_oracle_blind_mapping(
        observation.current_page_state,
        location="select_candidate_observation",
    )
    controls = observation.current_page_state.get(
        OBSERVABLE_SELECT_CONTROLS_FIELD
    )
    if not isinstance(controls, (tuple, list)) or not controls:
        raise ParameterResolutionError(
            "SELECT current observation lacks observable_select_controls evidence"
        )

    assert bbox is not None  # Enforced by static SELECT validation above.
    grounded_bbox = _validated_bbox(
        bbox,
        name="SELECT policy-grounded bbox",
    )
    matching_candidates: list[list[JsonValue]] = []
    for index, control in enumerate(controls):
        if not isinstance(control, Mapping):
            raise ParameterResolutionError(
                f"observable_select_controls[{index}] must be an object"
            )
        required = {"target_bbox", "candidate_options"}
        if set(control) != required:
            raise ParameterResolutionError(
                f"observable_select_controls[{index}] must contain exactly "
                "target_bbox and candidate_options"
            )
        observed_bbox = _validated_bbox(
            control["target_bbox"],
            name=f"observable_select_controls[{index}].target_bbox",
            require_json_list=True,
        )
        observed_candidates = control["candidate_options"]
        if not isinstance(observed_candidates, (tuple, list)) or not observed_candidates:
            raise ParameterResolutionError(
                f"observable_select_controls[{index}].candidate_options must "
                "be a non-empty list"
            )
        for candidate_index, candidate in enumerate(observed_candidates):
            _validate_select_option(
                candidate,
                name=(
                    f"observable_select_controls[{index}].candidate_options"
                    f"[{candidate_index}]"
                ),
            )
        if observed_bbox == grounded_bbox:
            matching_candidates.append(list(observed_candidates))

    if len(matching_candidates) != 1:
        raise ParameterResolutionError(
            "SELECT requires exactly one observable control at the grounded bbox"
        )
    provider_candidates = values["candidate_options"]
    assert isinstance(provider_candidates, list)  # Static validation above.
    observed_candidates = matching_candidates[0]
    if len(provider_candidates) != len(observed_candidates) or any(
        not _same_select_option(provider, observed)
        for provider, observed in zip(provider_candidates, observed_candidates)
    ):
        raise ParameterResolutionError(
            "SELECT candidate_options differ from the grounded observable control"
        )


def _finite_number(name: str, value: JsonValue) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterResolutionError(f"{name} must be a finite number")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as exc:
        raise ParameterResolutionError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ParameterResolutionError(f"{name} must be a finite number")
    return numeric


def _normalized_number(name: str, value: JsonValue) -> float:
    numeric = _finite_number(name, value)
    if not 0.0 <= numeric <= 1.0:
        raise ParameterResolutionError(f"{name} must be normalized to [0, 1]")
    return numeric


def _validated_bbox(
    value: object,
    *,
    name: str,
    require_json_list: bool = False,
) -> tuple[float, float, float, float]:
    if require_json_list and not isinstance(value, list):
        raise ParameterResolutionError(f"{name} must be a four-value JSON list")
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ParameterResolutionError(f"{name} must contain four values")
    result = tuple(
        _normalized_number(f"{name}[{index}]", item)
        for index, item in enumerate(value)
    )
    x, y, width, height = result
    if x + width > 1.0 or y + height > 1.0:
        raise ParameterResolutionError(
            f"{name} falls outside the normalized viewport"
        )
    return result


def _validate_grounded_target(
    action_type: ActionType,
    values: Mapping[str, JsonValue],
    bbox: tuple[float, float, float, float] | None,
) -> None:
    if bbox is None:
        raise ParameterResolutionError(f"{action_type.value} requires grounding")
    expected_bbox = _validated_bbox(
        bbox,
        name=f"{action_type.value} decision bbox",
    )

    target_bbox = values["target_bbox"]
    actual_bbox = _validated_bbox(
        target_bbox,
        name=f"{action_type.value} target_bbox",
        require_json_list=True,
    )
    actual_x, actual_y, actual_width, actual_height = actual_bbox
    if actual_bbox != expected_bbox:
        raise ParameterResolutionError(
            f"{action_type.value} target_bbox differs from the policy-grounded bbox"
        )

    target_x = _normalized_number("target_x", values["target_x"])
    target_y = _normalized_number("target_y", values["target_y"])
    if not (
        actual_x <= target_x <= actual_x + actual_width
        and actual_y <= target_y <= actual_y + actual_height
    ):
        raise ParameterResolutionError(
            f"{action_type.value} target point lies outside target_bbox"
        )


def _validate_select_option(value: JsonValue, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ParameterResolutionError(
            f"{name} must be a string, integer, or finite float"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ParameterResolutionError(
            f"{name} must be a string, integer, or finite float"
        )


def _same_select_option(left: JsonValue, right: JsonValue) -> bool:
    return type(left) is type(right) and left == right


def _validate_navigation_url(value: JsonValue) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ParameterResolutionError("NAVIGATE url must be a non-empty URL string")
    if "\\" in value or any(character.isspace() or ord(character) < 32 for character in value):
        raise ParameterResolutionError("NAVIGATE url contains unsafe characters")
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError as exc:
        raise ParameterResolutionError("NAVIGATE url is not parseable") from exc
    if parsed.scheme.lower() not in REGISTERED_NAVIGATION_SCHEMES:
        raise ParameterResolutionError(
            f"NAVIGATE scheme is not permitted: {parsed.scheme!r}"
        )
    if not parsed.netloc or not parsed.hostname:
        raise ParameterResolutionError("NAVIGATE url requires an absolute destination")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ParameterResolutionError("NAVIGATE url must not contain userinfo")
    if port is not None and port == 0:
        raise ParameterResolutionError("NAVIGATE url port must be in [1, 65535]")


def concretize(
    *,
    action_id: str,
    decision: PreActionDecision,
    parameters: ActionParameters,
    recovery_attempt_id: str | None = None,
) -> ConcreteAction:
    if parameters.action_type is not decision.action_type:
        raise ParameterResolutionError("provider changed the policy-selected action class")
    return ConcreteAction(
        action_id=action_id,
        source_decision_id=decision.decision_id,
        action_type=decision.action_type,
        parameters=dict(parameters.values),
        bbox=decision.bbox,
        recovery_attempt_id=recovery_attempt_id,
    )
