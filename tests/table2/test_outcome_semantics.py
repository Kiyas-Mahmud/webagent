from __future__ import annotations

import pytest

from web_agent.eval.table2.common import SchemaError
from web_agent.eval.table2.outcome_semantics import (
    FORCED_FAILURE_RUNTIME_REASONS,
    PRIMARY_INVALID_RUNTIME_REASONS,
    validate_primary_runtime_outcome,
)
from web_agent.eval.table2.package_validator import _validate_final_evidence


def _evidence(*, task_success: bool, evaluator_reason: str) -> dict:
    return {
        "task_success": task_success,
        "terminal_reason": evaluator_reason,
        "loop_detected": False,
        "environment_failure": False,
        "failure_incidents": [],
        "recovery_verifications": [],
        "verified_failure_event_count": 0,
        "repeated_error_event_count": 0,
        "memory_relevance": {},
    }


@pytest.mark.parametrize("runtime_reason", sorted(FORCED_FAILURE_RUNTIME_REASONS))
def test_forced_failure_runtime_reasons_cannot_be_counted_as_success(
    runtime_reason: str,
) -> None:
    with pytest.raises(SchemaError, match="cannot be counted as task success"):
        validate_primary_runtime_outcome(
            runtime_terminal_reason=runtime_reason,
            task_success=True,
            context="fixture",
        )


@pytest.mark.parametrize(
    ("runtime_reason", "task_success"),
    [
        *((reason, False) for reason in sorted(FORCED_FAILURE_RUNTIME_REASONS)),
        ("OPAQUE_VERIFIER_TERMINAL", False),
        ("OPAQUE_VERIFIER_TERMINAL", True),
    ],
)
def test_registered_primary_outcome_truth_table_accepts_only_valid_cells(
    runtime_reason: str,
    task_success: bool,
) -> None:
    assert (
        validate_primary_runtime_outcome(
            runtime_terminal_reason=runtime_reason,
            task_success=task_success,
            context="fixture",
        )
        == runtime_reason
    )


@pytest.mark.parametrize(
    ("runtime_reason", "task_success"),
    [
        (reason, success)
        for reason in sorted(PRIMARY_INVALID_RUNTIME_REASONS | {"CLOSED"})
        for success in (False, True)
    ],
)
def test_nonterminal_or_primary_invalid_reasons_are_never_selected(
    runtime_reason: str,
    task_success: bool,
) -> None:
    with pytest.raises(SchemaError):
        validate_primary_runtime_outcome(
            runtime_terminal_reason=runtime_reason,
            task_success=task_success,
            context="fixture",
        )


def test_evaluator_task_success_remains_distinct_from_opaque_runtime_terminal() -> None:
    # TASK_SUCCESS belongs to the sealed evaluator taxonomy.  It is deliberately
    # not required to equal the runtime's OPAQUE_VERIFIER_TERMINAL reason.
    _validate_final_evidence(
        _evidence(task_success=True, evaluator_reason="TASK_SUCCESS"),
        system_id="E0",
        summary={
            "runtime_terminal_reason": "opaque_verifier_terminal",
            "infrastructure_invalid": False,
        },
        recovery_attempt_events=(),
        post_queries=(),
        context="fixture",
    )


def test_package_evidence_rejects_success_after_timeout() -> None:
    with pytest.raises(SchemaError, match="TIMEOUT cannot be counted as task success"):
        _validate_final_evidence(
            _evidence(task_success=True, evaluator_reason="TASK_SUCCESS"),
            system_id="E0",
            summary={
                "runtime_terminal_reason": "timeout",
                "infrastructure_invalid": False,
            },
            recovery_attempt_events=(),
            post_queries=(),
            context="fixture",
        )
