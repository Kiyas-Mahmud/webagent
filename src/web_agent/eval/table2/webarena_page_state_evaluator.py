"""Sealed, read-only WebArena page-state compatibility evaluator.

Only the deterministic page-state mechanisms needed by the proposed Table 2
development set are implemented: WebArena's ``url_match`` rule and a strict,
read-only subset of ``program_html``.  The implementation intentionally does
not claim to be the upstream official evaluator.  Its frozen classification is
``reviewed_compatibility_port_pending_external_review``; a campaign must not
promote it until the registered independent compatibility review is complete
and a separately authenticated process-isolated page broker is registered.

The evaluator never navigates, clicks, types, invokes helper functions, runs
configuration-supplied code, or opens credentials.  HTML selectors are parsed
from an allow-listed ``document.querySelector(...).property`` grammar and are
passed as data to one fixed read-only JavaScript expression.  The one-way page
broker additionally verifies that every bound evaluation leaves the complete
registered live-page digest unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import html
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from web_agent.benchmarks.base import EnvironmentAdapter
from web_agent.benchmarks.webarena import FrozenWebArenaObservationSnapshot
from web_agent.runtime.contracts import (
    EpisodeSummary,
    OpaqueTerminalSignal,
    RuntimeStartState,
    TaskSpecification,
    TerminalReason,
    VerifierReceiptBinding,
    canonical_sha256,
)
from web_agent.runtime.event_log import verify_event_log
from web_agent.runtime.protocol import REGISTERED_MEMORY_TOP_K

from .common import SchemaError, read_jsonl
from .live_deployment import REGISTERED_BROWSERGYM_VERSION
from .live_page_broker_assembly import (
    process_sealed_page_evaluator_capability,
)
from .production_runner import SealedEvaluatorBinding
from .sealed_page_broker import (
    SealedBoundEvaluation,
    SealedBoundPageEvaluationRequest,
    SealedFinalEvaluation,
    SealedFinalPageEvaluationRequest,
    SealedPageEvaluatorCapability,
)
from .sealed_verifier import SealedVerifierWriter


EVALUATOR_ID = "table2-sealed-webarena-page-state-compatibility-port"
EVALUATOR_VERSION = "reviewed_compatibility_port_pending_external_review"
IMPLEMENTATION_CLASSIFICATION = EVALUATOR_VERSION
COMPILE_REPORT_SCHEMA_VERSION = "table2-page-state-compile-report-v1"
COMPILE_REPORT_RECORD_TYPE = "WebArenaPageStateEvaluatorCompileReport"
COMPILE_REPORT_STATUS = "COMPILED_PENDING_EXTERNAL_REVIEW"
REQUIRED_COMPILE_TASK_COUNT = 50
SUPPORTED_EVAL_TYPES = frozenset({"url_match", "program_html"})
_SUPPORTED_EVAL_ORDERS = frozenset(
    {
        ("url_match",),
        ("program_html",),
        ("url_match", "program_html"),
    }
)
_CONFIG_BASE_FIELDS = frozenset(
    {"eval_types", "reference_answers", "reference_url", "program_html"}
)
_HTML_TARGET_FIELDS = frozenset({"url", "locator", "required_contents"})
_HTML_PROPERTIES = frozenset({"outerText", "value", "selectedIndex"})
_QUERY_SELECTOR = re.compile(
    r"^document\.querySelector\((?P<selector>.+)\)\."
    r"(?P<property>outerText|value|selectedIndex)$",
    re.DOTALL,
)
_MEMORY_QUERY_INDEX = re.compile(r":memory-query:(?P<index>[1-9][0-9]*)$")
_SENSITIVE_URL_KEY = re.compile(
    r"(?:password|passwd|secret|token|cookie|authorization|api[_-]?key|"
    r"access[_-]?key|credential|signature|session(?:id)?|sid)",
    re.IGNORECASE,
)
_READ_ONLY_PROPERTY_SCRIPTS = {
    "outerText": "selector => document.querySelector(selector).outerText",
    "value": "selector => document.querySelector(selector).value",
    "selectedIndex": "selector => document.querySelector(selector).selectedIndex",
}


class WebArenaPageStateEvaluatorError(SchemaError):
    """The task config, live page, or completed runtime evidence is unsafe."""


@dataclass(frozen=True, slots=True)
class _URLRule:
    references: tuple[str, ...]
    rule_sha256: str


@dataclass(frozen=True, slots=True)
class _HTMLRule:
    locator: str
    selector: str | None
    property_name: str | None
    match_kind: str
    expected_exact: str | None
    expected_groups: tuple[tuple[str, ...], ...]
    rule_sha256: str


@dataclass(frozen=True, slots=True)
class CompiledPageStateEvaluator:
    """Immutable evaluator config containing no task/runtime capability."""

    eval_types: tuple[str, ...]
    url_rule: _URLRule | None
    html_rules: tuple[_HTMLRule, ...]
    config_sha256: str

    @property
    def criterion_count(self) -> int:
        return int(self.url_rule is not None) + len(self.html_rules)


@dataclass(frozen=True, slots=True)
class _CriterionResult:
    criterion_id: str
    mechanism: str
    matched: bool
    observed_value_sha256: str
    rule_sha256: str
    component_matches: tuple[bool, ...]

    def sealed_evidence(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "mechanism": self.mechanism,
            "matched": self.matched,
            "observed_value_sha256": self.observed_value_sha256,
            "rule_sha256": self.rule_sha256,
            "component_matches": list(self.component_matches),
        }


@dataclass(frozen=True, slots=True)
class _PageEvaluation:
    criteria: tuple[_CriterionResult, ...]

    @property
    def vector(self) -> tuple[bool, ...]:
        return tuple(item.matched for item in self.criteria)

    @property
    def all_satisfied(self) -> bool:
        return bool(self.criteria) and all(self.vector)

    def sealed_evidence(self) -> dict[str, Any]:
        vector = list(self.vector)
        return {
            "criteria": [item.sealed_evidence() for item in self.criteria],
            "criterion_vector_sha256": canonical_sha256(vector),
            "satisfied_criterion_count": sum(vector),
            "criterion_count": len(vector),
            "all_criteria_satisfied": self.all_satisfied,
        }


@dataclass(frozen=True, slots=True)
class _BoundHistory:
    receipt_kind: str
    observation_id: str
    action_id: str | None
    evaluation: _PageEvaluation


@dataclass(frozen=True, slots=True)
class _RuntimeAction:
    action_id: str
    event_type: str
    recovery_attempt_id: str | None
    execution_status: str
    environment_error: bool
    executor_step: int


@dataclass(frozen=True, slots=True)
class _RuntimeAttempt:
    attempt_id: str
    incident_id: str
    incident_attempt_index: int
    strategy: str
    action_ids: tuple[str, ...]
    completed: bool
    trigger_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RuntimeQuery:
    query_id: str
    incident_id: str
    incident_attempt_index: int
    failed_action_id: str
    candidate_ids: tuple[str, ...]
    admitted_candidate_id: str | None
    admitted: bool
    changed_decision: bool


@dataclass(frozen=True, slots=True)
class _AttemptOutcome:
    attempt_id: str
    successful: bool
    strict_progress: bool
    strict_regression: bool
    reached_task_success: bool
    execution_error_resolved: bool


def _exact_bool(value: Any, *, context: str) -> bool:
    if type(value) is not bool:
        raise WebArenaPageStateEvaluatorError(f"{context} must be an exact boolean")
    return value


def _nonempty_text(value: Any, *, context: str) -> str:
    if type(value) is not str or not value.strip():
        raise WebArenaPageStateEvaluatorError(f"{context} must be nonempty text")
    return value


def _mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise WebArenaPageStateEvaluatorError(f"{context} must be an object")
    return dict(value)


def _clean_answer(value: str) -> str:
    result = value.strip()
    if len(result) >= 2 and (
        (result.startswith("'") and result.endswith("'"))
        or (result.startswith('"') and result.endswith('"'))
    ):
        result = result[1:-1]
    return result.lower()


def _exact_match(reference: str, prediction: str) -> bool:
    return _clean_answer(prediction) == _clean_answer(reference)


def _must_include(reference: str, prediction: str) -> bool:
    return _clean_answer(reference) in _clean_answer(prediction)


def _validate_reference_url(value: str) -> None:
    try:
        parsed = urlparse(value)
    except ValueError as exc:
        raise WebArenaPageStateEvaluatorError(
            "url_match reference contains a malformed URL"
        ) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or any(ord(character) < 32 for character in value)
    ):
        raise WebArenaPageStateEvaluatorError(
            "url_match references must be credential-free HTTP(S) URLs"
        )
    for key in parse_qs(parsed.query, keep_blank_values=True):
        if _SENSITIVE_URL_KEY.search(key):
            raise WebArenaPageStateEvaluatorError(
                "url_match reference contains a credential-bearing query key"
            )


def _parse_plain_javascript_string_literal(value: str) -> str:
    """Parse the approved no-escape JavaScript string-literal subset.

    ``ast.literal_eval`` is intentionally unsuitable here: it accepts Python
    raw/unicode prefixes, adjacent literals, and triple quotes that the pinned
    WebArena JavaScript expression does not.  Exact approved task selectors use
    only one ordinary single- or double-quoted literal and no escapes.  Reject
    anything outside that lossless subset instead of guessing ECMAScript
    decoding or silently changing the selector bytes.
    """

    if len(value) < 2 or value[0] not in {"'", '"'} or value[-1] != value[0]:
        raise WebArenaPageStateEvaluatorError(
            "program_html querySelector argument is not one plain JavaScript "
            "string literal"
        )
    body = value[1:-1]
    if (
        not body
        or value[0] in body
        or "\\" in body
        or any(
            character in body
            for character in ("\x00", "\n", "\r", "\u2028", "\u2029")
        )
    ):
        raise WebArenaPageStateEvaluatorError(
            "program_html querySelector literal requires a nonempty, unescaped "
            "single-line CSS selector"
        )
    return body


def _compile_html_rule(raw: Any, *, index: int) -> _HTMLRule:
    target = _mapping(raw, context=f"program_html[{index}]")
    if set(target) != _HTML_TARGET_FIELDS:
        raise WebArenaPageStateEvaluatorError(
            f"program_html[{index}] has missing/extra or mutation-capable fields"
        )
    if target["url"] != "last":
        raise WebArenaPageStateEvaluatorError(
            "program_html may inspect only target_url='last'; navigation is forbidden"
        )
    locator = target["locator"]
    if type(locator) is not str:
        raise WebArenaPageStateEvaluatorError("program_html locator must be text")
    selector: str | None = None
    property_name: str | None = None
    if locator.strip():
        if locator != locator.strip():
            raise WebArenaPageStateEvaluatorError(
                "program_html locator may not contain surrounding whitespace"
            )
        if locator.startswith(("func:", "eval:", "[...document.")):
            raise WebArenaPageStateEvaluatorError(
                "function/eval/spread program_html locators are forbidden"
            )
        match = _QUERY_SELECTOR.fullmatch(locator)
        if match is None:
            raise WebArenaPageStateEvaluatorError(
                "program_html locator is outside the read-only querySelector grammar"
            )
        selector = _parse_plain_javascript_string_literal(match.group("selector"))
        property_name = match.group("property")
        if property_name not in _HTML_PROPERTIES:  # pragma: no cover - regex invariant
            raise WebArenaPageStateEvaluatorError(
                "program_html property is outside the read-only allow-list"
            )

    required = _mapping(
        target["required_contents"],
        context=f"program_html[{index}].required_contents",
    )
    if set(required) not in ({"exact_match"}, {"must_include"}):
        raise WebArenaPageStateEvaluatorError(
            "program_html required_contents must contain exactly exact_match or must_include"
        )
    expected_exact: str | None = None
    expected_groups: tuple[tuple[str, ...], ...] = ()
    if "exact_match" in required:
        expected_exact = _nonempty_text(
            required["exact_match"],
            context=f"program_html[{index}].exact_match",
        )
        match_kind = "exact_match"
    else:
        values = required["must_include"]
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not str or not value for value in values)
        ):
            raise WebArenaPageStateEvaluatorError(
                f"program_html[{index}].must_include must be a nonempty string list"
            )
        groups: list[tuple[str, ...]] = []
        for value in values:
            alternatives = tuple(value.split(" |OR| "))
            if any(not alternative for alternative in alternatives):
                raise WebArenaPageStateEvaluatorError(
                    "program_html must_include contains an empty OR alternative"
                )
            groups.append(alternatives)
        expected_groups = tuple(groups)
        match_kind = "must_include"
    return _HTMLRule(
        locator=locator,
        selector=selector,
        property_name=property_name,
        match_kind=match_kind,
        expected_exact=expected_exact,
        expected_groups=expected_groups,
        rule_sha256=canonical_sha256(target),
    )


def compile_page_state_evaluator_config(
    raw: Mapping[str, Any],
) -> CompiledPageStateEvaluator:
    """Validate and freeze the supported WebArena evaluator subset."""

    config = _mapping(raw, context="WebArena evaluator config")
    eval_types = config.get("eval_types")
    if (
        not isinstance(eval_types, list)
        or not eval_types
        or any(type(value) is not str for value in eval_types)
        or len(eval_types) != len(set(eval_types))
    ):
        raise WebArenaPageStateEvaluatorError(
            "eval_types must be a nonempty unique string list"
        )
    unknown_types = set(eval_types) - SUPPORTED_EVAL_TYPES
    if unknown_types:
        raise WebArenaPageStateEvaluatorError(
            "string/fuzzy/unknown evaluator mechanisms are forbidden: "
            f"{sorted(unknown_types)}"
        )
    if tuple(eval_types) not in _SUPPORTED_EVAL_ORDERS:
        raise WebArenaPageStateEvaluatorError(
            "page-state evaluator mechanisms use an unsupported order"
        )
    expected_fields = set(_CONFIG_BASE_FIELDS)
    if "url_match" in eval_types:
        expected_fields.add("url_note")
    if set(config) != expected_fields:
        raise WebArenaPageStateEvaluatorError(
            "WebArena evaluator config contains missing or extra fields"
        )
    if config["reference_answers"] is not None:
        raise WebArenaPageStateEvaluatorError(
            "page-state evaluation forbids string/fuzzy reference answers"
        )

    url_rule: _URLRule | None = None
    if "url_match" in eval_types:
        if config.get("url_note") != "GOLD in PRED":
            raise WebArenaPageStateEvaluatorError(
                "only WebArena url_note='GOLD in PRED' is supported"
            )
        reference_url = _nonempty_text(
            config["reference_url"], context="url_match reference_url"
        )
        references = tuple(
            value.rstrip("/") for value in reference_url.split(" |OR| ")
        )
        if any(not value for value in references):
            raise WebArenaPageStateEvaluatorError(
                "url_match reference_url contains an empty OR alternative"
            )
        for reference in references:
            _validate_reference_url(reference)
        url_rule = _URLRule(
            references=references,
            rule_sha256=canonical_sha256(
                {
                    "reference_url": reference_url,
                    "url_note": config["url_note"],
                }
            ),
        )
    elif config["reference_url"] is not None and config["reference_url"] != "":
        raise WebArenaPageStateEvaluatorError(
            "reference_url is forbidden when url_match is disabled"
        )

    raw_html = config["program_html"]
    if not isinstance(raw_html, list):
        raise WebArenaPageStateEvaluatorError("program_html must be an array")
    if "program_html" in eval_types:
        if not raw_html:
            raise WebArenaPageStateEvaluatorError(
                "program_html evaluator requires at least one target"
            )
        html_rules = tuple(
            _compile_html_rule(value, index=index)
            for index, value in enumerate(raw_html)
        )
    else:
        if raw_html:
            raise WebArenaPageStateEvaluatorError(
                "program_html targets are forbidden when the mechanism is disabled"
            )
        html_rules = ()
    compiled = CompiledPageStateEvaluator(
        eval_types=tuple(eval_types),
        url_rule=url_rule,
        html_rules=html_rules,
        config_sha256=canonical_sha256(config),
    )
    if compiled.criterion_count <= 0:  # pragma: no cover - eval_types invariant
        raise WebArenaPageStateEvaluatorError("evaluator has no page-state criteria")
    return compiled


def _webarena_url_match(references: Sequence[str], prediction: str) -> bool:
    """Port WebArena 0.0.4's exact ``GOLD in PRED`` URL semantics."""

    def parse_one(url: str) -> tuple[str, dict[str, list[str]]]:
        parsed = urlparse(url)
        return parsed.netloc + parsed.path, parse_qs(parsed.query)

    reference_bases: list[str] = []
    reference_queries: dict[str, set[str]] = defaultdict(set)
    for reference in references:
        base, query = parse_one(reference.rstrip("/"))
        reference_bases.append(base)
        for key, values in query.items():
            reference_queries[key].update(values)
    predicted_base, predicted_query = parse_one(str(prediction).rstrip("/"))
    base_match = any(reference in predicted_base for reference in reference_bases)
    query_match = all(
        any(value in predicted_query.get(key, []) for value in values)
        for key, values in reference_queries.items()
    )
    return base_match and query_match


def _read_html_value(page: Any, rule: _HTMLRule) -> str:
    if rule.selector is None:
        value = page.content()
        if type(value) is not str:
            raise WebArenaPageStateEvaluatorError("page.content() returned non-text")
        return html.unescape(value)
    assert rule.property_name is not None
    script = _READ_ONLY_PROPERTY_SCRIPTS[rule.property_name]
    try:
        value = page.evaluate(script, rule.selector)
        selected = str(value)
        if not selected:
            selected = ""
    except Exception:
        # This is the pinned WebArena behavior for a missing/invalid DOM target.
        selected = ""
    return html.unescape(selected)


def evaluate_compiled_page_state(
    page: Any,
    compiled: CompiledPageStateEvaluator,
) -> _PageEvaluation:
    """Evaluate one page without navigation or configuration-supplied code."""

    if type(compiled) is not CompiledPageStateEvaluator:
        raise TypeError("page-state evaluation requires a compiled config")
    criteria: list[_CriterionResult] = []
    if compiled.url_rule is not None:
        current_url = str(getattr(page, "url", ""))
        matched = _webarena_url_match(compiled.url_rule.references, current_url)
        criteria.append(
            _CriterionResult(
                criterion_id="url:0",
                mechanism="url_match",
                matched=matched,
                observed_value_sha256=canonical_sha256(current_url.rstrip("/")),
                rule_sha256=compiled.url_rule.rule_sha256,
                component_matches=(matched,),
            )
        )
    for index, rule in enumerate(compiled.html_rules):
        selected = _read_html_value(page, rule)
        if rule.match_kind == "exact_match":
            assert rule.expected_exact is not None
            components = (_exact_match(rule.expected_exact, selected),)
        else:
            components = tuple(
                any(_must_include(option, selected) for option in alternatives)
                for alternatives in rule.expected_groups
            )
        criteria.append(
            _CriterionResult(
                criterion_id=f"html:{index}",
                mechanism=f"program_html:{rule.match_kind}",
                matched=all(components),
                observed_value_sha256=canonical_sha256(selected),
                rule_sha256=rule.rule_sha256,
                component_matches=components,
            )
        )
    return _PageEvaluation(criteria=tuple(criteria))


def live_page_session_id(
    *, episode_id: str, task_id: str, start_state_sha256: str
) -> str:
    """Reproduce BrowserGym's registered live-page session identity exactly."""

    return canonical_sha256(
        {
            "contract": "table2-live-page-session-v1",
            "episode_id": episode_id,
            "task_id": task_id,
            "start_state_sha256": start_state_sha256,
        }
    )


def _task_evaluator_spec(
    task_row: Mapping[str, Any],
) -> tuple[str, str, str, str, CompiledPageStateEvaluator]:
    row = _mapping(task_row, context="resolved WebArena task row")
    task_id = _nonempty_text(row.get("task_id"), context="task_id")
    evaluator = _mapping(row.get("evaluator"), context="task evaluator")
    if set(evaluator) != {"evaluator_id", "evaluator_version", "config"}:
        raise WebArenaPageStateEvaluatorError(
            "task evaluator contains missing or extra fields"
        )
    evaluator_id = _nonempty_text(
        evaluator["evaluator_id"], context="evaluator_id"
    )
    evaluator_version = _nonempty_text(
        evaluator["evaluator_version"], context="evaluator_version"
    )
    if evaluator_id != EVALUATOR_ID or evaluator_version != EVALUATOR_VERSION:
        raise WebArenaPageStateEvaluatorError(
            "page-state evaluator must be labelled as a compatibility port pending "
            "external review, never as the official evaluator"
        )
    start_state = row.get("start_state")
    if not isinstance(start_state, Mapping):
        raise WebArenaPageStateEvaluatorError("task row lacks six-field start_state")
    try:
        typed_start = RuntimeStartState.from_webarena_mapping(start_state)
    except (TypeError, ValueError) as exc:
        raise WebArenaPageStateEvaluatorError(
            "task row has an invalid six-field start_state"
        ) from exc
    config = _mapping(evaluator["config"], context="task evaluator config")
    task_config = _mapping(row.get("task_config"), context="task_config")
    if task_config.get("eval") != config:
        raise WebArenaPageStateEvaluatorError(
            "task_config.eval differs from the sealed evaluator config"
        )
    return (
        task_id,
        evaluator_id,
        evaluator_version,
        typed_start.start_state_sha256,
        compile_page_state_evaluator_config(config),
    )


def build_page_state_evaluator_compile_report(
    task_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile and bind the complete ordered 50-task evaluator authority.

    Returning a report proves only that every exact row was accepted by this
    compatibility port's strict grammar.  It is deliberately *not* a campaign
    readiness receipt: this implementation remains pending independent
    external review, and no field in this artifact can be interpreted as a
    handoff, freeze, or production ``PASS``.

    A caller must preserve both the JSON-file digest and ``report_sha256``.
    Downstream gates must reopen the exact task rows and call
    :func:`validate_page_state_evaluator_compile_report` rather than trusting
    counts or hashes copied from this artifact.
    """

    if (
        not isinstance(task_rows, Sequence)
        or isinstance(task_rows, (str, bytes, bytearray))
        or len(task_rows) != REQUIRED_COMPILE_TASK_COUNT
        or not all(isinstance(row, Mapping) for row in task_rows)
    ):
        raise WebArenaPageStateEvaluatorError(
            "page-state compile report requires exactly 50 resolved task rows"
        )

    identities: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    seen_upstream_indices: set[int] = set()
    seen_benchmark_task_ids: set[str] = set()
    criterion_count = 0
    for position, raw_row in enumerate(task_rows):
        row = dict(raw_row)
        task_id = _nonempty_text(
            row.get("task_id"), context=f"compile task {position}.task_id"
        )
        benchmark_task_id = _nonempty_text(
            row.get("benchmark_task_id"),
            context=f"compile task {position}.benchmark_task_id",
        )
        upstream_index = row.get("upstream_index")
        if (
            task_id != task_id.strip()
            or benchmark_task_id != benchmark_task_id.strip()
            or type(upstream_index) is not int
            or upstream_index < 0
        ):
            raise WebArenaPageStateEvaluatorError(
                f"compile task {position} has a non-canonical stable identity"
            )
        if (
            task_id in seen_task_ids
            or upstream_index in seen_upstream_indices
            or benchmark_task_id in seen_benchmark_task_ids
        ):
            raise WebArenaPageStateEvaluatorError(
                "page-state compile report contains a duplicate stable task identity"
            )
        seen_task_ids.add(task_id)
        seen_upstream_indices.add(upstream_index)
        seen_benchmark_task_ids.add(benchmark_task_id)

        (
            compiled_task_id,
            evaluator_id,
            evaluator_version,
            start_state_sha256,
            compiled,
        ) = _task_evaluator_spec(row)
        if compiled_task_id != task_id:  # pragma: no cover - same parsed field
            raise WebArenaPageStateEvaluatorError(
                f"compile task {position} changed identity during compilation"
            )
        identity = {
            "position": position,
            "task_id": task_id,
            "upstream_index": upstream_index,
            "benchmark_task_id": benchmark_task_id,
        }
        identities.append(identity)
        criterion_count += compiled.criterion_count
        tasks.append(
            {
                **identity,
                "evaluator_id": evaluator_id,
                "evaluator_version": evaluator_version,
                "start_state_sha256": start_state_sha256,
                "task_row_sha256": canonical_sha256(row),
                "evaluator_config_sha256": compiled.config_sha256,
                "eval_types": list(compiled.eval_types),
                "criterion_count": compiled.criterion_count,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": COMPILE_REPORT_SCHEMA_VERSION,
        "record_type": COMPILE_REPORT_RECORD_TYPE,
        "status": COMPILE_REPORT_STATUS,
        "implementation_classification": IMPLEMENTATION_CLASSIFICATION,
        "compatibility_claim": "LOCAL_PINNED_REFERENCE_PARITY_ONLY",
        "external_review_required": True,
        "external_review_status": "PENDING",
        "all_task_configs_compiled": True,
        "campaign_ready": False,
        "handoff_eligible": False,
        "freeze_eligible": False,
        "production_eligible": False,
        "task_count": len(tasks),
        "criterion_count": criterion_count,
        "supported_eval_types": sorted(SUPPORTED_EVAL_TYPES),
        "ordered_task_identities_sha256": canonical_sha256(identities),
        "exact_task_rows_sha256": canonical_sha256(
            [dict(row) for row in task_rows]
        ),
        "tasks": tasks,
    }
    payload["report_sha256"] = canonical_sha256(payload)
    return payload


def validate_page_state_evaluator_compile_report(
    value: Mapping[str, Any],
    *,
    task_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompile all 50 frozen rows and require exact report equality."""

    if not isinstance(value, Mapping):
        raise WebArenaPageStateEvaluatorError(
            "page-state compile report must be a JSON object"
        )
    expected = build_page_state_evaluator_compile_report(task_rows)
    if dict(value) != expected:
        raise WebArenaPageStateEvaluatorError(
            "page-state compile report differs from exact 50-task recomputation"
        )
    return expected


def require_page_state_evaluator_campaign_ready(
    value: Mapping[str, Any],
    *,
    task_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed: the locally reviewed compatibility port is not authority.

    This helper exists for handoff/freeze/production call sites.  A future
    independently reviewed evaluator must use a newly registered identity and
    readiness contract; mutating this pending report into ``PASS`` is forbidden.
    """

    validate_page_state_evaluator_compile_report(value, task_rows=task_rows)
    raise WebArenaPageStateEvaluatorError(
        "page-state evaluator compiled all 50 exact task configs, but remains "
        "reviewed_compatibility_port_pending_external_review and is not "
        "campaign-ready for handoff, freeze, or production"
    )


def _stream(
    runtime_dir: Path,
    name: str,
    *,
    episode_id: str,
    required: bool = True,
) -> list[dict[str, Any]]:
    path = runtime_dir / f"{name}.jsonl"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise WebArenaPageStateEvaluatorError(
            f"runtime {name} stream is not a regular owned file"
        )
    if not path.is_file():
        if required:
            raise WebArenaPageStateEvaluatorError(
                f"runtime evidence lacks {name}.jsonl"
            )
        return []
    try:
        verify_event_log(
            path,
            expected_stream=name,
            expected_episode_id=episode_id,
        )
    except (OSError, ValueError) as exc:
        raise WebArenaPageStateEvaluatorError(
            f"runtime {name} stream failed hash-chain validation"
        ) from exc
    return read_jsonl(path)


def _payload(record: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    return _mapping(record.get("payload"), context=context)


def _runtime_actions(
    records: Sequence[Mapping[str, Any]],
    summary: EpisodeSummary,
) -> tuple[list[_RuntimeAction], dict[str, _RuntimeAction]]:
    actions: list[_RuntimeAction] = []
    charged_normal = 0
    executor_steps: list[int] = []
    for row in records:
        event_type = str(row.get("event_type") or "")
        payload = _payload(row, context=f"actions.{event_type}")
        if event_type == "pre_action_parse_rejection":
            rejection = _mapping(
                payload.get("rejection"), context="pre_action_parse_rejection"
            )
            execution = _mapping(
                payload.get("execution"),
                context="pre_action_parse_rejection.execution",
            )
            if (
                payload.get("action") is not None
                or execution.get("action_id") != rejection.get("request_id")
                or execution.get("status") != "rejected"
                or _exact_bool(
                    execution.get("environment_error"),
                    context="parse rejection environment_error",
                )
            ):
                raise WebArenaPageStateEvaluatorError(
                    "pre-action parser rejection is not one local rejected request"
                )
            step = execution.get("executor_step")
            if type(step) is not int or step <= 0:
                raise WebArenaPageStateEvaluatorError(
                    "parse rejection has an invalid executor step"
                )
            executor_steps.append(step)
            charged_normal += 1
            continue
        if event_type not in {"normal_action", "recovery_action"}:
            raise WebArenaPageStateEvaluatorError(
                f"actions stream contains unknown event type {event_type!r}"
            )
        if event_type == "normal_action":
            charged_normal += 1
        action = _mapping(payload.get("action"), context=f"{event_type}.action")
        execution = _mapping(
            payload.get("execution"), context=f"{event_type}.execution"
        )
        action_id = _nonempty_text(action.get("action_id"), context="action_id")
        if execution.get("action_id") != action_id:
            raise WebArenaPageStateEvaluatorError(
                "runtime action/execution identities disagree"
            )
        status = str(execution.get("status") or "")
        if status not in {"executed", "rejected", "error"}:
            raise WebArenaPageStateEvaluatorError(
                f"runtime execution has unknown status {status!r}"
            )
        environment_error = _exact_bool(
            execution.get("environment_error"), context="execution.environment_error"
        )
        step = execution.get("executor_step")
        if type(step) is not int or step <= 0:
            raise WebArenaPageStateEvaluatorError(
                "runtime action has an invalid executor step"
            )
        executor_steps.append(step)
        attempt_id: str | None = None
        if event_type == "recovery_action":
            attempt_id = _nonempty_text(
                payload.get("attempt_id"), context="recovery_action.attempt_id"
            )
            if action.get("recovery_attempt_id") != attempt_id:
                raise WebArenaPageStateEvaluatorError(
                    "recovery action differs from its attempt identity"
                )
        elif action.get("recovery_attempt_id") is not None:
            raise WebArenaPageStateEvaluatorError(
                "normal action contains a recovery attempt identity"
            )
        actions.append(
            _RuntimeAction(
                action_id=action_id,
                event_type=event_type,
                recovery_attempt_id=attempt_id,
                execution_status=status,
                environment_error=environment_error,
                executor_step=step,
            )
        )
    by_id = {item.action_id: item for item in actions}
    if len(by_id) != len(actions):
        raise WebArenaPageStateEvaluatorError("runtime action IDs are not unique")
    recovery_count = sum(item.event_type == "recovery_action" for item in actions)
    if charged_normal != summary.normal_actions or recovery_count != summary.recovery_actions:
        raise WebArenaPageStateEvaluatorError(
            "runtime action stream does not reconcile with EpisodeSummary"
        )
    if summary.executor_steps != charged_normal + recovery_count:
        raise WebArenaPageStateEvaluatorError(
            "runtime executor-step accounting does not reconcile"
        )
    if executor_steps != list(range(1, summary.executor_steps + 1)):
        raise WebArenaPageStateEvaluatorError(
            "runtime executor-step identities are not complete and ordered"
        )
    return actions, by_id


def _runtime_attempts(
    records: Sequence[Mapping[str, Any]],
    summary: EpisodeSummary,
    actions_by_id: Mapping[str, _RuntimeAction],
) -> list[_RuntimeAttempt]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pending_plan: dict[str, Any] | None = None
    for row in records:
        event_type = str(row.get("event_type") or "")
        payload = _payload(row, context=f"recoveries.{event_type}")
        if event_type == "recovery_plan":
            if pending_plan is not None:
                raise WebArenaPageStateEvaluatorError(
                    "recovery stream contains two plans before one completed attempt"
                )
            plan = _mapping(payload.get("plan"), context="recovery_plan.plan")
            shadow = _mapping(
                payload.get("shadow_decision"),
                context="recovery_plan.shadow_decision",
            )
            sources = shadow.get("trigger_sources")
            if (
                not isinstance(sources, list)
                or not sources
                or any(
                    source not in {"policy", "executor", "loop_guard"}
                    for source in sources
                )
                or len(sources) != len(set(sources))
            ):
                raise WebArenaPageStateEvaluatorError(
                    "recovery plan has invalid oracle-independent trigger sources"
                )
            pending_plan = {**plan, "_trigger_sources": tuple(sources)}
        elif event_type == "recovery_attempt":
            if pending_plan is None:
                raise WebArenaPageStateEvaluatorError(
                    "recovery attempt lacks its immediately preceding plan"
                )
            pairs.append(
                (
                    pending_plan,
                    _mapping(
                        payload.get("attempt", payload),
                        context="recovery_attempt",
                    ),
                )
            )
            pending_plan = None
        else:
            raise WebArenaPageStateEvaluatorError(
                f"recoveries stream contains unknown event type {event_type!r}"
            )
    if pending_plan is not None or len(pairs) != summary.recovery_attempts:
        raise WebArenaPageStateEvaluatorError(
            "recovery plans/attempts do not reconcile with EpisodeSummary"
        )
    resolved: list[_RuntimeAttempt] = []
    seen: set[str] = set()
    for episode_attempt_index, (plan, attempt) in enumerate(pairs, start=1):
        attempt_id = _nonempty_text(attempt.get("attempt_id"), context="attempt_id")
        incident_id = _nonempty_text(
            attempt.get("incident_id"), context="attempt.incident_id"
        )
        if attempt_id in seen:
            raise WebArenaPageStateEvaluatorError("recovery attempt IDs are not unique")
        seen.add(attempt_id)
        if plan.get("attempt_id") != attempt_id or plan.get("incident_id") != incident_id:
            raise WebArenaPageStateEvaluatorError(
                "recovery plan and completed attempt identities disagree"
            )
        index = attempt.get("incident_attempt_index")
        if type(index) is not int or index <= 0:
            raise WebArenaPageStateEvaluatorError(
                "recovery attempt has an invalid incident index"
            )
        if attempt_id != f"{incident_id}:attempt:{index}":
            raise WebArenaPageStateEvaluatorError(
                "recovery attempt ID is not canonical for its incident/index"
            )
        if (
            attempt.get("episode_attempt_index") != episode_attempt_index
            or plan.get("incident_attempt_index") != index
            or plan.get("episode_attempt_index") != episode_attempt_index
        ):
            raise WebArenaPageStateEvaluatorError(
                "recovery attempt/plan indices do not match runtime order"
            )
        strategy = str(attempt.get("strategy") or "")
        if strategy not in {"RETRY", "REPLAN", "BACKTRACK", "ALTERNATIVE_TARGET", "ABORT"}:
            raise WebArenaPageStateEvaluatorError(
                f"recovery attempt has unknown strategy {strategy!r}"
            )
        if plan.get("strategy") != strategy:
            raise WebArenaPageStateEvaluatorError(
                "recovery plan and attempt strategies disagree"
            )
        resolution_status = str(plan.get("resolution_status") or "")
        if resolution_status not in {"READY", "ABORT", "REJECTED"}:
            raise WebArenaPageStateEvaluatorError(
                "recovery plan has an invalid resolution status"
            )
        if (strategy == "ABORT") != (resolution_status == "ABORT"):
            raise WebArenaPageStateEvaluatorError(
                "ABORT strategy/resolution status disagree"
            )
        raw_plan_actions = plan.get("actions")
        if not isinstance(raw_plan_actions, list) or any(
            not isinstance(value, Mapping) for value in raw_plan_actions
        ):
            raise WebArenaPageStateEvaluatorError(
                "recovery plan actions must be an object list"
            )
        plan_action_ids = tuple(
            _nonempty_text(value.get("action_id"), context="plan action_id")
            for value in raw_plan_actions
        )
        if len(plan_action_ids) != len(set(plan_action_ids)):
            raise WebArenaPageStateEvaluatorError(
                "recovery plan repeats a browser action"
            )
        if resolution_status == "READY" and len(plan_action_ids) != 1:
            raise WebArenaPageStateEvaluatorError(
                "ready recovery plan must contain exactly one bounded browser action"
            )
        if resolution_status in {"ABORT", "REJECTED"} and plan_action_ids:
            raise WebArenaPageStateEvaluatorError(
                "non-executable recovery plan cannot contain browser actions"
            )
        raw_action_ids = attempt.get("action_ids")
        if not isinstance(raw_action_ids, list) or any(
            type(value) is not str or not value for value in raw_action_ids
        ):
            raise WebArenaPageStateEvaluatorError(
                "recovery attempt action_ids must be a string list"
            )
        action_ids = tuple(raw_action_ids)
        if len(action_ids) != len(set(action_ids)):
            raise WebArenaPageStateEvaluatorError(
                "recovery attempt repeats a browser action"
            )
        completed = _exact_bool(
            attempt.get("completed"), context="attempt.completed"
        )
        if action_ids != plan_action_ids[: len(action_ids)]:
            raise WebArenaPageStateEvaluatorError(
                "recovery attempt actions are not an ordered prefix of its plan"
            )
        if completed and action_ids != plan_action_ids:
            raise WebArenaPageStateEvaluatorError(
                "completed recovery attempt did not account for its whole plan"
            )
        if resolution_status == "REJECTED" and (completed or plan_action_ids):
            raise WebArenaPageStateEvaluatorError(
                "rejected recovery plan cannot complete or contain actions"
            )
        if resolution_status == "ABORT" and not completed:
            raise WebArenaPageStateEvaluatorError(
                "ABORT is an immediately completed zero-action attempt"
            )
        for action_id in action_ids:
            action = actions_by_id.get(action_id)
            if action is None or action.recovery_attempt_id != attempt_id:
                raise WebArenaPageStateEvaluatorError(
                    "recovery attempt cites an absent or foreign action"
                )
        resolved.append(
            _RuntimeAttempt(
                attempt_id=attempt_id,
                incident_id=incident_id,
                incident_attempt_index=index,
                strategy=strategy,
                action_ids=action_ids,
                completed=completed,
                trigger_sources=tuple(plan["_trigger_sources"]),
            )
        )
    observed_recovery_actions = tuple(
        item.action_id
        for item in actions_by_id.values()
        if item.event_type == "recovery_action"
    )
    declared_recovery_actions = tuple(
        action_id for attempt in resolved for action_id in attempt.action_ids
    )
    if observed_recovery_actions != declared_recovery_actions:
        raise WebArenaPageStateEvaluatorError(
            "runtime recovery actions are not covered by attempts in order"
        )
    incidents = list(dict.fromkeys(attempt.incident_id for attempt in resolved))
    if len(incidents) > summary.failure_incidents:
        raise WebArenaPageStateEvaluatorError(
            "runtime attempts exceed the EpisodeSummary incident count"
        )
    for incident in incidents:
        indices = [
            attempt.incident_attempt_index
            for attempt in resolved
            if attempt.incident_id == incident
        ]
        if indices != list(range(1, len(indices) + 1)):
            raise WebArenaPageStateEvaluatorError(
                "recovery incident attempt indices are not contiguous"
            )
    return resolved


def _runtime_queries(
    records: Sequence[Mapping[str, Any]], summary: EpisodeSummary
) -> list[_RuntimeQuery]:
    pending_shadow = 0
    queries: list[_RuntimeQuery] = []
    for row in records:
        event_type = str(row.get("event_type") or "")
        payload = _payload(row, context=f"memory_queries.{event_type}")
        if event_type == "no_memory_shadow_before_retrieval":
            pending_shadow += 1
            continue
        if event_type != "post_failure_query" or pending_shadow <= 0:
            raise WebArenaPageStateEvaluatorError(
                "memory stream violates shadow-before-query ordering"
            )
        pending_shadow -= 1
        query = _mapping(payload.get("query"), context="memory query contract")
        result = _mapping(payload.get("query_result"), context="memory query result")
        query_id = _nonempty_text(result.get("query_id"), context="memory query_id")
        if query.get("query_id") != query_id:
            raise WebArenaPageStateEvaluatorError(
                "memory query contract/result identities disagree"
            )
        index_match = _MEMORY_QUERY_INDEX.search(query_id)
        if index_match is None:
            raise WebArenaPageStateEvaluatorError(
                "memory query ID lacks its incident attempt index"
            )
        incident_id = _nonempty_text(
            query.get("incident_id"), context="memory query incident_id"
        )
        candidates = result.get("candidate_ids")
        if not isinstance(candidates, list) or any(
            type(value) is not str or not value for value in candidates
        ):
            raise WebArenaPageStateEvaluatorError(
                "memory candidate_ids must be a string list"
            )
        if len(candidates) > REGISTERED_MEMORY_TOP_K or len(candidates) != len(
            set(candidates)
        ):
            raise WebArenaPageStateEvaluatorError(
                "memory candidates violate frozen top-k/uniqueness semantics"
            )
        admitted = _exact_bool(result.get("admitted"), context="memory.admitted")
        admitted_id = result.get("admitted_candidate_id")
        if admitted != (isinstance(admitted_id, str) and bool(admitted_id)):
            raise WebArenaPageStateEvaluatorError(
                "memory admission flag and candidate identity disagree"
            )
        if admitted and admitted_id not in candidates:
            raise WebArenaPageStateEvaluatorError(
                "admitted memory is absent from returned candidates"
            )
        changed = _exact_bool(
            result.get("changed_strategy"), context="memory.changed_strategy"
        ) or _exact_bool(
            result.get("changed_target_or_parameters"),
            context="memory.changed_target_or_parameters",
        )
        if changed and not admitted:
            raise WebArenaPageStateEvaluatorError(
                "a rejected memory candidate cannot change the recovery decision"
            )
        attempt_index = int(index_match.group("index"))
        if query_id != f"{incident_id}:memory-query:{attempt_index}":
            raise WebArenaPageStateEvaluatorError(
                "memory query ID does not bind its exact incident/attempt"
            )
        queries.append(
            _RuntimeQuery(
                query_id=query_id,
                incident_id=incident_id,
                incident_attempt_index=attempt_index,
                failed_action_id=_nonempty_text(
                    query.get("failed_action_id"),
                    context="memory query failed_action_id",
                ),
                candidate_ids=tuple(candidates),
                admitted_candidate_id=(str(admitted_id) if admitted else None),
                admitted=admitted,
                changed_decision=changed,
            )
        )
    if pending_shadow:
        raise WebArenaPageStateEvaluatorError(
            "memory stream contains an unused no-memory shadow"
        )
    if len(queries) != summary.memory_queries:
        raise WebArenaPageStateEvaluatorError(
            "memory query count does not reconcile with EpisodeSummary"
        )
    if sum(query.changed_decision for query in queries) != summary.memory_interventions:
        raise WebArenaPageStateEvaluatorError(
            "memory interventions do not reconcile with EpisodeSummary"
        )
    if len({item.query_id for item in queries}) != len(queries):
        raise WebArenaPageStateEvaluatorError("memory query IDs are not unique")
    return queries


def _strict_progress(before: tuple[bool, ...], after: tuple[bool, ...]) -> bool:
    if len(before) != len(after):
        raise WebArenaPageStateEvaluatorError("criterion vector width changed")
    return all((not prior) or current for prior, current in zip(before, after)) and any(
        (not prior) and current for prior, current in zip(before, after)
    )


def _strict_regression(before: tuple[bool, ...], after: tuple[bool, ...]) -> bool:
    if len(before) != len(after):
        raise WebArenaPageStateEvaluatorError("criterion vector width changed")
    return any(prior and not current for prior, current in zip(before, after))


class _EpisodePageStateEvaluator:
    """One task-bound sealed evaluator; never exposed to model callbacks."""

    def __init__(
        self,
        *,
        task_id: str,
        start_state_sha256: str,
        compiled: CompiledPageStateEvaluator,
        writer: SealedVerifierWriter,
        capability: SealedPageEvaluatorCapability,
        manual_rescue_identity_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.task_id = task_id
        self.start_state_sha256 = start_state_sha256
        self.compiled = compiled
        self.writer = writer
        self.capability = capability
        if manual_rescue_identity_provider is not None and not callable(
            manual_rescue_identity_provider
        ):
            raise TypeError("manual-rescue identity provider must be callable")
        self.manual_rescue_identity_provider = manual_rescue_identity_provider
        self.history: list[_BoundHistory] = []
        self.episode_id: str | None = None

    def _session_id(self, episode_id: str) -> str:
        return live_page_session_id(
            episode_id=episode_id,
            task_id=self.task_id,
            start_state_sha256=self.start_state_sha256,
        )

    def terminal_signal_mapper(
        self,
        snapshot: FrozenWebArenaObservationSnapshot,
        task: TaskSpecification,
        binding: VerifierReceiptBinding | None,
    ) -> OpaqueTerminalSignal:
        if type(snapshot) is not FrozenWebArenaObservationSnapshot:
            raise WebArenaPageStateEvaluatorError(
                "sealed evaluator requires a detached observation snapshot"
            )
        if type(task) is not TaskSpecification or task.task_id != self.task_id:
            raise WebArenaPageStateEvaluatorError(
                "sealed evaluator received another task"
            )
        if (
            task.runtime_start_state is None
            or task.runtime_start_state.start_state_sha256
            != self.start_state_sha256
        ):
            raise WebArenaPageStateEvaluatorError(
                "sealed evaluator task start-state identity changed"
            )
        if type(binding) is not VerifierReceiptBinding:
            raise WebArenaPageStateEvaluatorError(
                "sealed evaluator requires a causal verifier receipt binding"
            )
        episode_id = snapshot.observation.episode_id
        if self.episode_id is None:
            self.episode_id = episode_id
        elif self.episode_id != episode_id:
            raise WebArenaPageStateEvaluatorError(
                "sealed evaluator was reused across episodes"
            )
        request = SealedBoundPageEvaluationRequest(
            session_id=self._session_id(episode_id),
            episode_id=episode_id,
            task_id=self.task_id,
            binding=binding,
        )
        return self.capability.evaluate_bound(
            request,
            writer=self.writer,
            evaluator=self._evaluate_bound,
        )

    def _evaluate_bound(
        self,
        page: Any,
        request: SealedBoundPageEvaluationRequest,
    ) -> SealedBoundEvaluation:
        evaluation = evaluate_compiled_page_state(page, self.compiled)
        self.history.append(
            _BoundHistory(
                receipt_kind=request.binding.receipt_kind,
                observation_id=request.binding.observation_id,
                action_id=request.binding.action_id,
                evaluation=evaluation,
            )
        )
        return SealedBoundEvaluation(
            verification={
                "implementation_classification": IMPLEMENTATION_CLASSIFICATION,
                "evaluation_config_sha256": self.compiled.config_sha256,
                "observation_id": request.binding.observation_id,
                **evaluation.sealed_evidence(),
            },
            should_terminate=evaluation.all_satisfied,
        )

    def finalize_episode_evidence(
        self,
        summary: EpisodeSummary,
        writer: SealedVerifierWriter,
        runtime_dir: Path,
    ) -> OpaqueTerminalSignal:
        if type(summary) is not EpisodeSummary or summary.task_id != self.task_id:
            raise WebArenaPageStateEvaluatorError(
                "final evaluator received another episode summary"
            )
        if writer is not self.writer:
            raise WebArenaPageStateEvaluatorError(
                "final evaluator writer differs from its bound write-only capability"
            )
        if self.episode_id != summary.episode_id or not self.history:
            raise WebArenaPageStateEvaluatorError(
                "final evaluator lacks the episode's bound reset evidence"
            )
        request = SealedFinalPageEvaluationRequest(
            session_id=self._session_id(summary.episode_id),
            episode_id=summary.episode_id,
            task_id=self.task_id,
            summary=summary,
        )
        return self.capability.evaluate_final(
            request,
            writer=writer,
            evaluator=lambda page, current: self._evaluate_final(
                page, current, Path(runtime_dir)
            ),
        )

    def _evaluate_final(
        self,
        page: Any,
        request: SealedFinalPageEvaluationRequest,
        runtime_dir: Path,
    ) -> SealedFinalEvaluation:
        final_page = evaluate_compiled_page_state(page, self.compiled)
        summary = request.summary
        actions, actions_by_id = _runtime_actions(
            _stream(runtime_dir, "actions", episode_id=summary.episode_id),
            summary,
        )
        attempts = _runtime_attempts(
            _stream(runtime_dir, "recoveries", episode_id=summary.episode_id),
            summary,
            actions_by_id,
        )
        if summary.system_id.value == "E3":
            query_records = _stream(
                runtime_dir,
                "memory_queries",
                episode_id=summary.episode_id,
            )
        else:
            query_records = _stream(
                runtime_dir,
                "memory_queries",
                episode_id=summary.episode_id,
                required=False,
            )
            if query_records:
                raise WebArenaPageStateEvaluatorError(
                    f"{summary.system_id.value} emitted forbidden memory queries"
                )
        queries = _runtime_queries(query_records, summary)

        bound_action_ids = tuple(
            entry.action_id for entry in self.history if entry.action_id is not None
        )
        logged_action_ids = tuple(item.action_id for item in actions)
        if bound_action_ids != logged_action_ids:
            raise WebArenaPageStateEvaluatorError(
                "bound page evaluations do not cover browser actions exactly in order"
            )
        if self.history[0].receipt_kind != "after_reset" or self.history[0].action_id is not None:
            raise WebArenaPageStateEvaluatorError(
                "bound page evaluation history does not begin after reset"
            )
        if (
            sum(entry.action_id is None for entry in self.history) != 1
            or len({entry.observation_id for entry in self.history}) != len(self.history)
        ):
            raise WebArenaPageStateEvaluatorError(
                "bound page history repeats reset or observation identity"
            )
        for action, entry in zip(actions, self.history[1:], strict=True):
            expected_kind = (
                "after_recovery_action"
                if action.event_type == "recovery_action"
                else "after_normal_action"
            )
            if entry.action_id != action.action_id or entry.receipt_kind != expected_kind:
                raise WebArenaPageStateEvaluatorError(
                    "bound page receipt kind differs from its runtime action"
                )
        history_by_action = {
            entry.action_id: (index, entry)
            for index, entry in enumerate(self.history)
            if entry.action_id is not None
        }
        if len(history_by_action) != len(bound_action_ids):
            raise WebArenaPageStateEvaluatorError(
                "bound page history repeats an action identity"
            )

        queries_by_attempt = {
            (query.incident_id, query.incident_attempt_index): query
            for query in queries
        }
        if len(queries_by_attempt) != len(queries):
            raise WebArenaPageStateEvaluatorError(
                "memory queries repeat an incident attempt"
            )
        if summary.system_id.value == "E3" and set(queries_by_attempt) != {
            (attempt.incident_id, attempt.incident_attempt_index)
            for attempt in attempts
        }:
            raise WebArenaPageStateEvaluatorError(
                "E3 memory queries do not cover recovery attempts one-to-one"
            )
        attempts_by_key = {
            (attempt.incident_id, attempt.incident_attempt_index): attempt
            for attempt in attempts
        }
        for key, query in queries_by_attempt.items():
            failed_action = actions_by_id.get(query.failed_action_id)
            if failed_action is None or query.failed_action_id not in history_by_action:
                raise WebArenaPageStateEvaluatorError(
                    "memory query cites an absent failed browser action"
                )
            attempt = attempts_by_key[key]
            if attempt.action_ids:
                first_position, _ = history_by_action[attempt.action_ids[0]]
                if (
                    first_position <= 0
                    or self.history[first_position - 1].action_id
                    != query.failed_action_id
                ):
                    raise WebArenaPageStateEvaluatorError(
                        "memory query is not causally adjacent to its recovery attempt"
                    )

        attempts_by_incident: dict[str, list[_RuntimeAttempt]] = defaultdict(list)
        for attempt in attempts:
            attempts_by_incident[attempt.incident_id].append(attempt)
        expected_runtime_incident_ids = [
            f"{summary.episode_id}:incident:{index}"
            for index in range(1, summary.failure_incidents + 1)
        ]
        if not set(attempts_by_incident).issubset(expected_runtime_incident_ids):
            raise WebArenaPageStateEvaluatorError(
                "runtime recovery attempt uses a noncanonical incident identity"
            )
        if [
            incident_id
            for incident_id in expected_runtime_incident_ids
            if incident_id in attempts_by_incident
        ] != list(attempts_by_incident):
            raise WebArenaPageStateEvaluatorError(
                "runtime recovery incident order differs from EpisodeRunner order"
            )
        # A failure may reach a global recovery cap or timeout before
        # ``begin_attempt`` emits a plan.  The EpisodeSummary still commits its
        # deterministic incident identity, but there is no independent cause
        # evidence.  Retain it as explicitly unverified instead of inventing a
        # positive label or refusing to emit a schema-valid forced failure.
        for incident_id in expected_runtime_incident_ids:
            attempts_by_incident.setdefault(incident_id, [])

        failed_action_by_incident: dict[str, str] = {}
        used_executor_failures: set[str] = set()
        for incident_id, incident_attempts in attempts_by_incident.items():
            if not incident_attempts:
                continue
            first = incident_attempts[0]
            query = queries_by_attempt.get((incident_id, 1))
            if query is not None:
                failed_action_by_incident[incident_id] = query.failed_action_id
            else:
                first_action_id = next(
                    (
                        action_id
                        for attempt in incident_attempts
                        for action_id in attempt.action_ids
                    ),
                    None,
                )
                if first_action_id is not None:
                    position, _ = history_by_action[first_action_id]
                    if position <= 0 or self.history[position - 1].action_id is None:
                        raise WebArenaPageStateEvaluatorError(
                            "recovery action lacks a preceding failed action"
                        )
                    failed_action_by_incident[incident_id] = str(
                        self.history[position - 1].action_id
                    )
            if "executor" in first.trigger_sources:
                failed_id = failed_action_by_incident.get(incident_id)
                if failed_id is None:
                    failed_id = next(
                        (
                            action.action_id
                            for action in actions
                            if action.event_type == "normal_action"
                            and action.execution_status != "executed"
                            and action.action_id not in used_executor_failures
                        ),
                        None,
                    )
                    if failed_id is not None:
                        failed_action_by_incident[incident_id] = failed_id
                failed_action = actions_by_id.get(failed_id or "")
                if failed_action is None or failed_action.execution_status == "executed":
                    raise WebArenaPageStateEvaluatorError(
                        "executor-triggered incident lacks an independently logged execution error"
                    )
                used_executor_failures.add(failed_action.action_id)

        incident_verification: dict[str, tuple[bool, str, bool]] = {}
        for incident_id, incident_attempts in attempts_by_incident.items():
            if not incident_attempts:
                incident_verification[incident_id] = (
                    False,
                    "UNVERIFIABLE_NO_ATTEMPT_EVIDENCE",
                    False,
                )
                continue
            sources = incident_attempts[0].trigger_sources
            if any(attempt.trigger_sources != sources for attempt in incident_attempts):
                raise WebArenaPageStateEvaluatorError(
                    "recovery trigger sources changed within one incident"
                )
            regression = False
            failed_action_id = failed_action_by_incident.get(incident_id)
            if failed_action_id in history_by_action:
                failed_position, failed_entry = history_by_action[failed_action_id]
                if failed_position > 0:
                    regression = _strict_regression(
                        self.history[failed_position - 1].evaluation.vector,
                        failed_entry.evaluation.vector,
                    )
            if "loop_guard" in sources:
                verified, kind = True, "REPEATED_STATE_ACTION_LOOP"
            elif "executor" in sources:
                failed_action = actions_by_id.get(failed_action_id or "")
                if failed_action is not None and failed_action.environment_error:
                    verified, kind = False, "ENVIRONMENT_ERROR_NOT_AGENT_FAILURE"
                else:
                    verified, kind = True, "EXECUTION_ERROR"
            elif regression:
                verified, kind = True, "INDEPENDENT_CRITERION_REGRESSION"
            else:
                verified, kind = False, "POLICY_TRIGGER_FALSE_POSITIVE"
            incident_verification[incident_id] = (verified, kind, regression)

        attempt_outcomes: dict[str, _AttemptOutcome] = {}
        for incident_id, incident_attempts in attempts_by_incident.items():
            if not incident_attempts:
                continue
            verified = incident_verification[incident_id][0]
            for attempt in incident_attempts:
                strict_progress = False
                strict_regression = False
                reached_success = False
                execution_resolution = False
                if attempt.action_ids:
                    first_position, _ = history_by_action[attempt.action_ids[0]]
                    last_position, last_entry = history_by_action[attempt.action_ids[-1]]
                    if first_position <= 0 or last_position < first_position:
                        raise WebArenaPageStateEvaluatorError(
                            "recovery attempt has invalid bound-history ordering"
                        )
                    before = self.history[first_position - 1].evaluation.vector
                    after = last_entry.evaluation.vector
                    strict_progress = _strict_progress(before, after)
                    strict_regression = _strict_regression(before, after)
                    reached_success = last_entry.evaluation.all_satisfied
                    failed_id = (
                        queries_by_attempt[
                            (incident_id, attempt.incident_attempt_index)
                        ].failed_action_id
                        if (incident_id, attempt.incident_attempt_index)
                        in queries_by_attempt
                        else str(self.history[first_position - 1].action_id or "")
                    )
                    failed_action = actions_by_id.get(failed_id)
                    execution_resolution = bool(
                        failed_action is not None
                        and failed_action.execution_status != "executed"
                        and not failed_action.environment_error
                        and all(
                            actions_by_id[action_id].execution_status == "executed"
                            and not actions_by_id[action_id].environment_error
                            for action_id in attempt.action_ids
                        )
                    )
                candidate_success = bool(
                    verified
                    and attempt.completed
                    and attempt.strategy != "ABORT"
                    and (reached_success or strict_progress or execution_resolution)
                )
                attempt_outcomes[attempt.attempt_id] = _AttemptOutcome(
                    attempt_id=attempt.attempt_id,
                    successful=candidate_success,
                    strict_progress=strict_progress,
                    strict_regression=strict_regression,
                    reached_task_success=reached_success,
                    execution_error_resolved=execution_resolution,
                )

        recovery_verifications: list[dict[str, Any]] = []
        failure_incidents: list[dict[str, Any]] = []
        for incident_id, incident_attempts in attempts_by_incident.items():
            verified, failure_kind, regression = incident_verification[incident_id]
            successful = [
                attempt
                for attempt in incident_attempts
                if attempt_outcomes[attempt.attempt_id].successful
            ]
            resolved_attempt_index = (
                successful[0].incident_attempt_index if successful else None
            )
            failure_incidents.append(
                {
                    "failure_incident_id": incident_id,
                    "verified_agent_failure": verified,
                    "resolved": bool(successful),
                    "resolved_attempt_index": resolved_attempt_index,
                    "attempt_count": len(incident_attempts),
                    "failure_kind": failure_kind,
                    "trigger_sources": (
                        list(incident_attempts[0].trigger_sources)
                        if incident_attempts
                        else []
                    ),
                    "criterion_regression_measured": regression,
                }
            )
            for attempt in incident_attempts:
                outcome = attempt_outcomes[attempt.attempt_id]
                recovery_verifications.append(
                    {
                        "recovery_attempt_id": attempt.attempt_id,
                        "failure_incident_id": incident_id,
                        "verified_failure_present": verified,
                        "successful": outcome.successful,
                        "strict_criterion_progress": outcome.strict_progress,
                        "reached_task_success": outcome.reached_task_success,
                        "execution_error_resolved": outcome.execution_error_resolved,
                    }
                )

        # E0/E1 have no recovery controller, and an action can also fail after
        # the last controller incident.  Derive these failure incidents only
        # from executor receipts or a criterion regression measured across two
        # consecutive bound page states.  Policy diagnoses alone never create
        # an independently verified failure.
        covered_failed_actions = set(failed_action_by_incident.values())
        standalone_loop_added = any(
            verified and kind == "REPEATED_STATE_ACTION_LOOP"
            for verified, kind, _ in incident_verification.values()
        )
        for action in actions:
            if action.event_type != "normal_action" or action.action_id in covered_failed_actions:
                continue
            position, entry = history_by_action[action.action_id]
            regression = bool(
                position > 0
                and _strict_regression(
                    self.history[position - 1].evaluation.vector,
                    entry.evaluation.vector,
                )
            )
            executor_failure = bool(
                action.execution_status != "executed"
                and not action.environment_error
            )
            if not executor_failure and not regression:
                continue
            incident_id = (
                f"{summary.episode_id}:sealed-observed-failure:"
                f"{canonical_sha256(action.action_id)[:16]}"
            )
            failure_incidents.append(
                {
                    "failure_incident_id": incident_id,
                    "verified_agent_failure": True,
                    "resolved": False,
                    "resolved_attempt_index": None,
                    "attempt_count": 0,
                    "failure_kind": (
                        "EXECUTION_ERROR"
                        if executor_failure
                        else "INDEPENDENT_CRITERION_REGRESSION"
                    ),
                    "trigger_sources": (["executor"] if executor_failure else []),
                    "criterion_regression_measured": regression,
                }
            )
            incident_verification[incident_id] = (
                True,
                (
                    "EXECUTION_ERROR"
                    if executor_failure
                    else "INDEPENDENT_CRITERION_REGRESSION"
                ),
                regression,
            )
        if summary.terminal_reason is TerminalReason.LOOP and not standalone_loop_added:
            incident_id = f"{summary.episode_id}:sealed-terminal-loop"
            failure_incidents.append(
                {
                    "failure_incident_id": incident_id,
                    "verified_agent_failure": True,
                    "resolved": False,
                    "resolved_attempt_index": None,
                    "attempt_count": 0,
                    "failure_kind": "REPEATED_STATE_ACTION_LOOP",
                    "trigger_sources": ["loop_guard"],
                    "criterion_regression_measured": False,
                }
            )
            incident_verification[incident_id] = (
                True,
                "REPEATED_STATE_ACTION_LOOP",
                False,
            )

        memory_relevance: dict[str, dict[str, Any]] = {}
        for query in queries:
            attempt = attempts_by_key[(query.incident_id, query.incident_attempt_index)]
            outcome = attempt_outcomes[attempt.attempt_id]
            useful = bool(
                query.admitted and query.changed_decision and outcome.successful
            )
            harmful = bool(
                query.admitted
                and query.changed_decision
                and outcome.strict_regression
            )
            memory_relevance[query.query_id] = {
                # Page outcome cannot independently prove item-level semantic
                # relevance.  Candidate relevance remains empty until blinded
                # post-episode adjudication; no positive label is manufactured.
                "relevant_ids": [],
                "relevance_definition": (
                    "post-action conservative outcome label; task outcome does not "
                    "establish item-level semantic relevance"
                ),
                "useful_intervention": useful if query.admitted else None,
                "harmful_intervention": harmful if query.admitted else None,
            }

        raw_task_success = final_page.all_satisfied
        runtime_reason = summary.terminal_reason
        if runtime_reason is TerminalReason.OPAQUE_VERIFIER_TERMINAL:
            if not self.history[-1].evaluation.all_satisfied or not raw_task_success:
                raise WebArenaPageStateEvaluatorError(
                    "opaque verifier terminal is not reproducible from final page criteria"
                )
        forced_failure = runtime_reason in {
            TerminalReason.ABORT,
            TerminalReason.ACTION_BUDGET_EXHAUSTED,
            TerminalReason.RECOVERY_BUDGET_EXHAUSTED,
            TerminalReason.TIMEOUT,
            TerminalReason.LOOP,
            TerminalReason.POLICY_ERROR,
            TerminalReason.PROVIDER_ERROR,
        }
        task_success = bool(
            raw_task_success
            and runtime_reason is TerminalReason.OPAQUE_VERIFIER_TERMINAL
            and summary.valid_for_primary
            and not summary.environment_failure
            and not summary.reset_already_success
            and not forced_failure
        )
        if task_success:
            terminal_reason = "TASK_SUCCESS"
        elif summary.environment_failure:
            terminal_reason = "ENVIRONMENT_FAILURE"
        elif summary.reset_already_success:
            terminal_reason = "RESET_ALREADY_SUCCESS"
        elif forced_failure:
            terminal_reason = f"FORCED_RUNTIME_FAILURE:{runtime_reason.value.upper()}"
        else:
            terminal_reason = "TASK_FAILURE"

        verified_count = sum(
            item[0] for item in incident_verification.values()
        )
        loop_count = sum(
            item[0] and item[1] == "REPEATED_STATE_ACTION_LOOP"
            for item in incident_verification.values()
        )
        evidence = {
            "task_success": task_success,
            "terminal_reason": terminal_reason,
            "loop_detected": bool(
                runtime_reason is TerminalReason.LOOP or loop_count > 0
            ),
            "environment_failure": summary.environment_failure,
            "failure_incidents": failure_incidents,
            "recovery_verifications": recovery_verifications,
            "verified_failure_event_count": verified_count,
            "repeated_error_event_count": loop_count,
            "memory_relevance": memory_relevance,
            "implementation_classification": IMPLEMENTATION_CLASSIFICATION,
            "evaluation_config_sha256": self.compiled.config_sha256,
            "raw_page_criteria_satisfied": raw_task_success,
            "forced_failure_runtime_semantics_applied": forced_failure,
            "runtime_terminal_reason": runtime_reason.value,
            "final_page_evaluation": final_page.sealed_evidence(),
        }
        if self.manual_rescue_identity_provider is not None:
            identity = self.manual_rescue_identity_provider()
            expected_fields = {
                "schema_version",
                "record_type",
                "relative_path",
                "record_count",
                "tail_sha256",
                "content_sha256",
            }
            if not isinstance(identity, Mapping) or set(identity) != expected_fields:
                raise WebArenaPageStateEvaluatorError(
                    "manual-rescue sidecar identity fields differ"
                )
            count = identity.get("record_count")
            tail = identity.get("tail_sha256")
            if (
                identity.get("schema_version")
                != "table2-process-broker-manual-rescue-sidecar-identity-v1"
                or identity.get("record_type")
                != "ProcessBrokerManualRescueSidecarIdentity"
                or identity.get("relative_path")
                != "manual_rescue_guard.child.jsonl"
                or type(count) is not int
                or count < 0
                or (tail is None) is not (count == 0)
                or (
                    tail is not None
                    and (
                        type(tail) is not str
                        or len(tail) != 64
                        or any(character not in "0123456789abcdef" for character in tail)
                    )
                )
            ):
                raise WebArenaPageStateEvaluatorError(
                    "manual-rescue sidecar identity is invalid"
                )
            content_sha256 = identity.get("content_sha256")
            if (
                type(content_sha256) is not str
                or len(content_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in content_sha256
                )
            ):
                raise WebArenaPageStateEvaluatorError(
                    "manual-rescue sidecar content identity is invalid"
                )
            evidence["manual_rescue_guard_sidecar_identity"] = dict(identity)
        return SealedFinalEvaluation(verification=evidence)


def create_sealed_webarena_page_state_evaluator(
    task_row: Mapping[str, Any],
    writer: SealedVerifierWriter,
    *,
    manual_rescue_identity_provider: Callable[[], Mapping[str, Any]] | None = None,
) -> SealedEvaluatorBinding:
    """Engineering factory shape for one sealed task evaluator binding.

    The current same-process broker makes this implementation unpromotable for
    live campaign dispatch; see the frozen page-broker security binding.
    """

    if type(writer) is not SealedVerifierWriter:
        raise TypeError("sealed evaluator factory requires the exact writer capability")
    task_id, evaluator_id, evaluator_version, start_sha256, compiled = (
        _task_evaluator_spec(task_row)
    )
    callbacks = _EpisodePageStateEvaluator(
        task_id=task_id,
        start_state_sha256=start_sha256,
        compiled=compiled,
        writer=writer,
        capability=process_sealed_page_evaluator_capability(),
        manual_rescue_identity_provider=manual_rescue_identity_provider,
    )
    return SealedEvaluatorBinding(
        benchmark_version=REGISTERED_BROWSERGYM_VERSION,
        evaluator_id=evaluator_id,
        evaluator_version=evaluator_version,
        terminal_signal_mapper=callbacks.terminal_signal_mapper,
        finalize_episode_evidence=callbacks.finalize_episode_evidence,
        frozen=True,
        oracle_labels_exposed_to_runtime=False,
    )


_PROCESS_BROKER_TASK_ROW_FIELDS = frozenset(
    {
        "task_id",
        "upstream_index",
        "benchmark_task_id",
        "benchmark_task_version",
        "instruction",
        "start_state",
        "task_config",
        "evaluator",
        "source_content_sha256",
    }
)


def _load_process_broker_evaluator_task_row(
    *,
    task: TaskSpecification,
) -> Mapping[str, Any]:
    """Fail closed until a child-only resolved-task bootstrap is registered."""

    del task
    raise WebArenaPageStateEvaluatorError(
        "process-broker sealed task-row bootstrap is not registered; the full "
        "resolved evaluator row must be supplied by an external child-only "
        "deployment integration"
    )


def _process_broker_sealed_binding(
    *,
    task: TaskSpecification,
    adapter: EnvironmentAdapter,
    evidence_writer: SealedVerifierWriter,
    allow_create: bool,
) -> SealedEvaluatorBinding:
    if (
        type(task) is not TaskSpecification
        or not isinstance(adapter, EnvironmentAdapter)
        or type(evidence_writer) is not SealedVerifierWriter
    ):
        raise TypeError("process-broker sealed callback received the wrong contract")
    getter = getattr(adapter, "process_broker_sealed_evaluator", None)
    binder = getattr(adapter, "process_broker_bind_sealed_evaluator", None)
    if not callable(getter) or not callable(binder):
        raise WebArenaPageStateEvaluatorError(
            "process-broker browser adapter lacks its sealed callback bridge"
        )
    current = getter(evidence_writer=evidence_writer)
    if current is not None:
        if type(current) is not SealedEvaluatorBinding:
            raise WebArenaPageStateEvaluatorError(
                "process-broker sealed callback bridge returned a wrong binding"
            )
        return current
    if not allow_create:
        raise WebArenaPageStateEvaluatorError(
            "process-broker finalizer lacks prior causal transition binding"
        )
    row_value = _load_process_broker_evaluator_task_row(task=task)
    if not isinstance(row_value, Mapping):
        raise WebArenaPageStateEvaluatorError(
            "process-broker sealed task-row bootstrap returned non-mapping data"
        )
    row = dict(row_value)
    if set(row) != _PROCESS_BROKER_TASK_ROW_FIELDS:
        raise WebArenaPageStateEvaluatorError(
            "process-broker sealed task row fields differ from the frozen snapshot"
        )
    content = {
        key: row[key]
        for key in _PROCESS_BROKER_TASK_ROW_FIELDS
        if key != "source_content_sha256"
    }
    metadata = task.metadata
    try:
        typed_start = RuntimeStartState.from_webarena_mapping(row["start_state"])
    except (TypeError, ValueError) as exc:
        raise WebArenaPageStateEvaluatorError(
            "process-broker sealed task row has invalid start state"
        ) from exc
    if (
        row.get("source_content_sha256") != canonical_sha256(content)
        or row.get("source_content_sha256")
        != metadata.get("source_content_sha256")
        or row.get("task_id") != task.task_id
        or row.get("instruction") != task.goal
        or row.get("upstream_index") != metadata.get("upstream_index")
        or row.get("benchmark_task_id") != metadata.get("benchmark_task_id")
        or task.runtime_start_state is None
        or typed_start.start_state_sha256
        != task.runtime_start_state.start_state_sha256
    ):
        raise WebArenaPageStateEvaluatorError(
            "process-broker sealed task row differs from its oracle-blind task projection"
        )
    identity_provider = getattr(
        adapter,
        "process_broker_manual_rescue_sidecar_identity",
        None,
    )
    if not callable(identity_provider):
        raise WebArenaPageStateEvaluatorError(
            "process-broker adapter lacks child-owned manual-rescue evidence"
        )
    created = create_sealed_webarena_page_state_evaluator(
        row,
        evidence_writer,
        manual_rescue_identity_provider=identity_provider,
    )
    binder(evaluator=created, evidence_writer=evidence_writer)
    rebound = getter(evidence_writer=evidence_writer)
    if rebound is not created:
        raise WebArenaPageStateEvaluatorError(
            "process-broker sealed callback bridge did not retain exact binding"
        )
    return created


def evaluate_environment_transition(
    *,
    task: TaskSpecification,
    adapter: EnvironmentAdapter,
    receipt_binding: VerifierReceiptBinding,
    evidence_writer: SealedVerifierWriter,
) -> OpaqueTerminalSignal:
    """Source-attested transition entrypoint returning only an opaque signal."""

    if type(receipt_binding) is not VerifierReceiptBinding:
        raise TypeError("process-broker transition requires receipt binding")
    _process_broker_sealed_binding(
        task=task,
        adapter=adapter,
        evidence_writer=evidence_writer,
        allow_create=True,
    )
    signal = adapter.terminal_signal(task, receipt_binding)
    if type(signal) is not OpaqueTerminalSignal:
        raise WebArenaPageStateEvaluatorError(
            "process-broker transition exposed a non-opaque result"
        )
    return signal


def finalize_environment_episode(
    *,
    task: TaskSpecification,
    adapter: EnvironmentAdapter,
    episode_summary: EpisodeSummary,
    episode_runtime_dir: Path,
    evidence_writer: SealedVerifierWriter,
) -> OpaqueTerminalSignal:
    """Source-attested child finalizer over the already-bound sealed stream."""

    if type(episode_summary) is not EpisodeSummary:
        raise TypeError("process-broker finalizer requires EpisodeSummary")
    binding = _process_broker_sealed_binding(
        task=task,
        adapter=adapter,
        evidence_writer=evidence_writer,
        allow_create=False,
    )
    signal = binding.finalize_episode_evidence(
        episode_summary,
        evidence_writer,
        Path(episode_runtime_dir),
    )
    if type(signal) is not OpaqueTerminalSignal or signal.terminate is not True:
        raise WebArenaPageStateEvaluatorError(
            "process-broker finalizer exposed a non-opaque acknowledgement"
        )
    return signal
