from __future__ import annotations

from web_agent.eval.table2.summary import (
    MANUAL_AUDIT_CASE_CATEGORIES,
    _manual_audit_categories,
)


def test_manual_audit_categories_are_evidence_backed_without_fabrication():
    episodes = [
        {
            "episode_id": "case",
            "environment_failure": True,
            "loop_detected": True,
        },
        {
            "episode_id": "plain",
            "environment_failure": False,
            "loop_detected": False,
        },
    ]
    categories = _manual_audit_categories(
        episodes,
        recovery_attempts=[
            {
                "episode_id": "case",
                "successful": True,
                "verified_failure_present": True,
            },
            {
                "episode_id": "case",
                "successful": False,
                "verified_failure_present": False,
            },
        ],
        failure_incidents=[
            {
                "episode_id": "case",
                "failure_type": "INVALID_ACTION_PARAMETER",
            }
        ],
        memory_queries=[
            {
                "episode_id": "case",
                "useful_intervention": True,
                "harmful_intervention": True,
            }
        ],
    )
    assert categories["case"] == set(MANUAL_AUDIT_CASE_CATEGORIES)
    assert categories["plain"] == set()
