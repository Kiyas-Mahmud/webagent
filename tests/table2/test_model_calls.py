from __future__ import annotations

import pytest

from web_agent.runtime.model_calls import (
    REGISTERED_MAX_MODEL_CALLS_PER_EPISODE,
    ModelCallBudgetExceeded,
    ModelCallLedger,
    activate_model_call_ledger,
    deactivate_model_call_ledger,
    record_model_call,
)


def test_model_call_ledger_records_before_dispatch_and_enforces_cap() -> None:
    records: list[dict] = []
    ledger = ModelCallLedger(
        episode_id="episode-1",
        sink=records.append,
        maximum=2,
    )
    token = activate_model_call_ledger(ledger)
    try:
        record_model_call(stage="pre_action_policy", component_id="policy-v1")
        record_model_call(
            stage="action_parameter_fallback",
            component_id="provider-v1",
        )
        with pytest.raises(ModelCallBudgetExceeded, match="exceeded 2"):
            record_model_call(stage="post_action_assessment", component_id="policy-v1")
    finally:
        deactivate_model_call_ledger(token)

    assert ledger.count == 2
    assert [row["model_call_index"] for row in records] == [1, 2]
    assert all(row["maximum_model_calls"] == 2 for row in records)


def test_model_call_context_is_episode_scoped() -> None:
    first: list[dict] = []
    second: list[dict] = []
    outer = activate_model_call_ledger(
        ModelCallLedger(episode_id="outer", sink=first.append)
    )
    try:
        record_model_call(stage="pre_action_policy", component_id="outer-policy")
        inner = activate_model_call_ledger(
            ModelCallLedger(episode_id="inner", sink=second.append)
        )
        try:
            record_model_call(stage="memory_embedding", component_id="inner-model")
        finally:
            deactivate_model_call_ledger(inner)
        record_model_call(stage="recovery_assessment", component_id="outer-policy")
    finally:
        deactivate_model_call_ledger(outer)

    assert [row["episode_id"] for row in first] == ["outer", "outer"]
    assert [row["episode_id"] for row in second] == ["inner"]


def test_registered_model_call_cap_covers_frozen_structural_maximum() -> None:
    assert REGISTERED_MAX_MODEL_CALLS_PER_EPISODE == 102
