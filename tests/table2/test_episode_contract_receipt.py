from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from web_agent.eval.table2.common import SchemaError, read_jsonl, sha256_json
from web_agent.eval.table2.package_validator import (
    _validate_episode_contract_validation_receipt,
)
from web_agent.runtime.contracts import (
    EPISODE_FOREIGN_KEY_RECEIPT_VERSION,
    EpisodeForeignKeyValidation,
    canonical_sha256,
)
from web_agent.runtime.event_log import EpisodeEventLogs


SHA = "a" * 64
SHA_B = "b" * 64
EPISODE_ID = "campaign:E2:fixture:repeat-0:seed-42"


def _fixture(
    tmp_path: Path,
) -> tuple[Path, dict, list[dict], list[dict], list[dict], list[dict]]:
    validation = EpisodeForeignKeyValidation(
        episode_id=EPISODE_ID,
        status="PASS",
        observations=2,
        actions=1,
        executions=1,
        transition_assessments=1,
        recovery_attempts=0,
        recovery_assessments=0,
        memory_queries=0,
    )
    runtime = tmp_path / "runtime"
    logs = EpisodeEventLogs(
        runtime,
        episode_id=EPISODE_ID,
        include_memory=False,
        system_id="E2",
        task_id="fixture",
        repeat_id=0,
        matched_seed=42,
        timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
    )
    logs.append("environment_events", "reset", {})
    logs.append("environment_events", "post_action_observation", {})
    logs.append("actions", "normal_action", {})
    logs.append("transitions", "post_action_assessment", {})
    causal_snapshot = canonical_sha256(logs.file_hashes())
    receipt = {
        "receipt_version": EPISODE_FOREIGN_KEY_RECEIPT_VERSION,
        "episode_id": EPISODE_ID,
        "bundle_record_type": "EpisodeContractBundle",
        "bundle_sha256": SHA,
        "summary_sha256": SHA,
        "causal_log_snapshot_sha256": causal_snapshot,
        "validation": validation.to_dict(),
        "validation_sha256": validation.record_sha256,
    }
    logs.append("environment_events", "episode_contract_validation", receipt)
    logs.append("environment_events", "sealed_evaluator_backend_guard", {})
    summary = {
        "episode_id": EPISODE_ID,
        "memory_queries": 0,
        "event_log_sha256": causal_snapshot,
    }
    return (
        runtime,
        summary,
        read_jsonl(runtime / "actions.jsonl"),
        read_jsonl(runtime / "transitions.jsonl"),
        read_jsonl(runtime / "recoveries.jsonl"),
        read_jsonl(runtime / "environment_events.jsonl"),
    )


def _validate(fixture, *, required: bool = True) -> None:
    runtime, summary, actions, transitions, recoveries, environment = fixture
    _validate_episode_contract_validation_receipt(
        runtime=runtime,
        summary=summary,
        actions=actions,
        transitions=transitions,
        recovery_events=recoveries,
        environment_events=environment,
        required=required,
    )


def test_production_receipt_reconciles_hash_and_observable_causal_counts(
    tmp_path: Path,
) -> None:
    _validate(_fixture(tmp_path))


def test_production_receipt_is_required_but_nonproduction_absence_is_allowed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture[-1][:] = fixture[-1][:-2]
    with pytest.raises(SchemaError, match="lacks in-memory contract validation"):
        _validate(fixture)
    _validate(fixture, required=False)


def test_receipt_tamper_fails_hash_and_causal_count_replay(tmp_path: Path) -> None:
    bad_hash = deepcopy(_fixture(tmp_path / "bad-hash"))
    bad_hash[-1][2]["payload"]["validation"]["actions"] = 0
    with pytest.raises(SchemaError, match="result hash differs"):
        _validate(bad_hash)

    causal_tamper = deepcopy(_fixture(tmp_path / "causal-tamper"))
    receipt = causal_tamper[-1][2]["payload"]
    receipt["validation"]["actions"] = 0
    receipt["validation_sha256"] = sha256_json(receipt["validation"])
    with pytest.raises(SchemaError, match="actions differs from hash-chained"):
        _validate(causal_tamper)


def test_receipt_must_precede_only_sealed_finalizer_guards(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture[-1].append({"event_type": "model_call_dispatch", "payload": {}})
    with pytest.raises(SchemaError, match="evidence was appended after"):
        _validate(fixture)


def test_causal_log_snapshot_excludes_receipt_but_is_rebuilt_from_prefix(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture[1]["event_log_sha256"] = SHA_B
    fixture[-1][2]["payload"]["causal_log_snapshot_sha256"] = SHA_B
    with pytest.raises(SchemaError, match="pre-receipt causal-log snapshot differs"):
        _validate(fixture)
