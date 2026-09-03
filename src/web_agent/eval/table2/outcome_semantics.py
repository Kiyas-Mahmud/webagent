"""Registered runtime-terminal semantics for primary Table 2 outcomes.

The runtime and the independent evaluator use different terminal taxonomies.
For example, a successful benchmark episode normally ends with runtime reason
``OPAQUE_VERIFIER_TERMINAL`` while the sealed evaluator may report
``TASK_SUCCESS``.  These fields must remain separate: evaluator truth supplies
the success label, while the runtime reason determines whether that label is
admissible under the preregistered protocol.
"""

from __future__ import annotations

from typing import Any

from .common import SchemaError


OPAQUE_VERIFIER_TERMINAL = "OPAQUE_VERIFIER_TERMINAL"

FORCED_FAILURE_RUNTIME_REASONS = frozenset(
    {
        "ABORT",
        "ACTION_BUDGET_EXHAUSTED",
        "RECOVERY_BUDGET_EXHAUSTED",
        "TIMEOUT",
        "LOOP",
        "POLICY_ERROR",
        "PROVIDER_ERROR",
    }
)

PRIMARY_INVALID_RUNTIME_REASONS = frozenset(
    {
        "ENVIRONMENT_FAILURE",
        "RESET_ALREADY_SUCCESS",
    }
)

REGISTERED_RUNTIME_TERMINAL_REASONS = frozenset(
    {
        OPAQUE_VERIFIER_TERMINAL,
        *FORCED_FAILURE_RUNTIME_REASONS,
        *PRIMARY_INVALID_RUNTIME_REASONS,
        "CLOSED",
    }
)


def normalize_runtime_terminal_reason(value: Any, *, context: str) -> str:
    """Return one canonical enum-name spelling or fail closed."""

    raw = getattr(value, "value", value)
    if not isinstance(raw, str) or not raw.strip():
        raise SchemaError(f"{context} lacks a runtime terminal reason")
    normalized = raw.strip().upper().replace("-", "_").replace(" ", "_")
    if normalized not in REGISTERED_RUNTIME_TERMINAL_REASONS:
        raise SchemaError(
            f"{context} has unregistered runtime terminal reason {raw!r}"
        )
    return normalized


def validate_primary_runtime_outcome(
    *,
    runtime_terminal_reason: Any,
    task_success: bool,
    context: str,
) -> str:
    """Enforce the preregistered success/terminal truth table.

    Only an opaque-verifier terminal may carry a positive official success
    label.  Budget exhaustion, timeout, unresolved loop, ABORT, and agent-side
    policy/provider errors remain failures even if the final page happens to
    satisfy the benchmark oracle.  Infrastructure/reset invalidity never enters
    an included primary package, and ``CLOSED`` is an internal continuation
    sentinel rather than a completed terminal outcome.
    """

    if type(task_success) is not bool:
        raise SchemaError(f"{context} task_success must be an exact boolean")
    reason = normalize_runtime_terminal_reason(
        runtime_terminal_reason,
        context=context,
    )
    if reason in PRIMARY_INVALID_RUNTIME_REASONS:
        raise SchemaError(
            f"{context} runtime reason {reason} cannot enter a primary-valid package"
        )
    if reason == "CLOSED":
        raise SchemaError(
            f"{context} runtime reason CLOSED is not a completed terminal outcome"
        )
    if task_success and reason != OPAQUE_VERIFIER_TERMINAL:
        raise SchemaError(
            f"{context} runtime reason {reason} cannot be counted as task success"
        )
    return reason
