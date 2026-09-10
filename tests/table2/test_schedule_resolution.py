from __future__ import annotations

import pytest

from web_agent.eval.table2.common import SchemaError
from web_agent.eval.table2.schedule import SYSTEM_IDS, resolve_block_attempts


SHA256 = "a" * 64


def _status(
    *,
    launched: bool = False,
    completed: bool = False,
    infrastructure_invalid: bool = False,
) -> dict[str, object]:
    value: dict[str, object] = {
        "launched": launched,
        "completed": completed,
        "infrastructure_invalid": infrastructure_invalid,
    }
    if infrastructure_invalid:
        value.update(
            {
                "infrastructure_reason": "BROWSER_RESET_FAILURE",
                "infrastructure_evidence_sha256": SHA256,
            }
        )
    return value


def _attempt(
    attempt_id: int, overrides: dict[str, dict[str, object]] | None = None
) -> dict[str, object]:
    systems = {system_id: _status() for system_id in SYSTEM_IDS}
    systems.update(overrides or {})
    return {"attempt_id": attempt_id, "systems": systems}


def _schedule() -> dict[str, object]:
    return {"block_id": "adversarial-block", "max_block_attempts": 2}


def test_two_all_unlaunched_attempts_cannot_become_infrastructure_exclusion() -> None:
    resolution = resolve_block_attempts(
        _schedule(),
        [_attempt(0), _attempt(1)],
    )

    assert resolution["status"] == "INTERRUPTED_UNAUTHORIZED"
    assert resolution["selected_attempt_id"] is None
    assert not any(
        attempt["infrastructure_invalid"] for attempt in resolution["attempts"]
    )


def test_partial_completed_attempts_without_infrastructure_are_unauthorized() -> None:
    first = _attempt(0, {"E0": _status(launched=True, completed=True)})
    second = _attempt(
        1,
        {
            "E0": _status(launched=True, completed=True),
            "E1": _status(launched=True, completed=True),
        },
    )

    resolution = resolve_block_attempts(_schedule(), [first, second])

    assert resolution["status"] == "INTERRUPTED_UNAUTHORIZED"
    assert resolution["selected_attempt_id"] is None


def test_unauthorized_partial_attempt_cannot_consume_rerun_then_select_valid() -> None:
    partial = _attempt(0, {"E0": _status(launched=True, completed=True)})
    complete = _attempt(
        1,
        {
            system_id: _status(launched=True, completed=True)
            for system_id in SYSTEM_IDS
        },
    )

    resolution = resolve_block_attempts(_schedule(), [partial, complete])

    assert resolution["status"] == "INTERRUPTED_UNAUTHORIZED"
    assert resolution["selected_attempt_id"] is None


def test_only_typed_infrastructure_attempts_can_rerun_or_reach_exclusion() -> None:
    first = _attempt(
        0,
        {"E0": _status(launched=True, infrastructure_invalid=True)},
    )
    assert resolve_block_attempts(_schedule(), [first])["status"] == "RERUN_REQUIRED"

    second = _attempt(
        1,
        {"E2": _status(launched=True, infrastructure_invalid=True)},
    )
    resolution = resolve_block_attempts(_schedule(), [first, second])
    assert resolution["status"] == "EXCLUDED_INFRASTRUCTURE"


def test_unlaunched_row_cannot_claim_infrastructure_invalidation() -> None:
    malformed = _attempt(0, {"E0": _status(infrastructure_invalid=True)})

    with pytest.raises(SchemaError, match="launched, incomplete"):
        resolve_block_attempts(_schedule(), [malformed])
