from __future__ import annotations

from web_agent.data.recovery_transitions import (
    build_recovery_transition_index,
    recovery_class_audit,
)


def _row(
    task: str,
    step: int,
    *,
    strategy: str = "NONE",
    recovery_success=None,
    action: str = "CLICK",
    failure: str = "NONE",
) -> dict:
    return {
        "inputs": {
            "state_before": f"{task}-{step}-before.png",
            "state_after": f"{task}-{step}-after.png",
            "task_description": "finish the task",
            "website_domain": "example.test",
        },
        "labels": {
            "action_type": action,
            "action_value": "target",
            "recovery_strategy": strategy,
            "recovery_success": recovery_success,
            "failure_type_4": failure,
        },
        "meta": {
            "task_id": task,
            "step_index": step,
            "sample_id": f"{task}-{step}",
        },
    }


def test_transition_uses_next_action_and_post_recovery_state():
    rows = [
        _row(
            "t1", 0, strategy="BACKTRACK", recovery_success=True,
            failure="ACTION_MISMATCH",
        ),
        _row("t1", 1, action="PRESS_KEY"),
    ]
    transitions, report = build_recovery_transition_index(rows)

    transition = transitions["t1-0"]
    assert transition["failure_state"] == "t1-0-after.png"
    assert transition["executed_recovery_action"] == "PRESS_KEY"
    assert transition["post_recovery_state"] == "t1-1-after.png"
    assert transition["recovery_success"] is True
    assert report["proper_transitions"] == 1
    assert report["coverage"] == 1.0


def test_terminal_attempt_is_reported_but_not_exported():
    rows = [_row("t1", 0, strategy="RETRY", recovery_success=False)]
    transitions, report = build_recovery_transition_index(rows)

    assert transitions == {}
    assert report["attempted_recoveries"] == 1
    assert report["missing_next_step"] == 1


def test_class_audit_exposes_targeted_missing_classes():
    rows = [
        _row("t1", 0, strategy="RETRY", recovery_success=False),
        _row("t1", 1, failure="LOOP_DETECTED"),
    ]
    audit = recovery_class_audit(rows)

    assert audit["targeted_counts"]["RETRY"] == 1
    assert audit["targeted_counts"]["ABORT"] == 0
    assert audit["targeted_counts"]["BACKTRACK"] == 0
    assert audit["targeted_counts"]["LOOP_DETECTED"] == 1
    assert audit["missing_claimed_classes"] == ["ABORT", "BACKTRACK"]
