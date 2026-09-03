from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from web_agent.benchmarks.fixture import run_all_systems_fixture_smoke
from web_agent.benchmarks.recovery_fixture import (
    RecoveryFixtureCampaignRunner,
    load_registered_recovery_diagnostics,
    run_failure_memory_intervention_smoke,
)
from web_agent.eval.table2.common import SchemaError
from web_agent.eval.table2.evidence_validation import (
    validate_e2_e3_causal_trace,
    validate_e1_e2_e3_first_pre_action_equivalence,
    validate_ordered_verifier_receipts,
    validate_runtime_screenshot_artifacts,
)
from web_agent.eval.table2.sealed_verifier import SealedVerifierSink
from web_agent.runtime.event_log import EpisodeEventLogs
from web_agent.runtime.protocol import RuntimeProtocol


def test_e2_e3_trace_matches_through_first_admitted_intervention(tmp_path: Path) -> None:
    root = tmp_path / "failure-memory"
    run_failure_memory_intervention_smoke(root)
    result = validate_e2_e3_causal_trace(
        root / "E2" / "runtime",
        root / "E3" / "runtime",
    )
    assert result["comparison_scope"] == (
        "PREFIX_THROUGH_FIRST_ADMITTED_INTERVENTION"
    )
    assert result["first_admitted_query_index"] == 0


def test_first_trained_pre_action_output_matches_e1_e2_e3_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "first-decision"
    run_all_systems_fixture_smoke(root)
    result = validate_e1_e2_e3_first_pre_action_equivalence(root)
    assert result["status"] == "PASS"
    assert result["system_count"] == 3
    assert result["event_type"] == "normal_action"

    actions_path = root / "E3" / "runtime" / "actions.jsonl"
    rows = _read_jsonl(actions_path)
    rows[0]["payload"]["decision"]["confidence_before"] = 0.123
    _write_jsonl(actions_path, rows)
    with pytest.raises(SchemaError, match="first trained pre-action outputs differ"):
        validate_e1_e2_e3_first_pre_action_equivalence(root)


def test_e2_e3_trace_requires_full_equality_when_e3_abstains(tmp_path: Path) -> None:
    root = tmp_path / "no-memory-intervention"
    run_all_systems_fixture_smoke(root)
    result = validate_e2_e3_causal_trace(
        root / "E2" / "runtime",
        root / "E3" / "runtime",
    )
    assert result["comparison_scope"] == "FULL_TRACE_E3_ABSTAINED"

    changed = tmp_path / "changed-e3"
    shutil.copytree(root / "E3" / "runtime", changed)
    actions_path = changed / "actions.jsonl"
    rows = _read_jsonl(actions_path)
    rows[0]["payload"]["decision"]["confidence_before"] = 0.123
    _write_jsonl(actions_path, rows)
    with pytest.raises(SchemaError, match="semantic actions"):
        validate_e2_e3_causal_trace(root / "E2" / "runtime", changed)


def test_admitted_trace_requires_query_action_and_observation_binding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "failure-memory"
    run_failure_memory_intervention_smoke(root)
    changed = tmp_path / "changed-e3"
    shutil.copytree(root / "E3" / "runtime", changed)
    path = changed / "memory_queries.jsonl"
    rows = _read_jsonl(path)
    del rows[1]["payload"]["query"]
    _write_jsonl(path, rows)
    with pytest.raises(SchemaError, match="lacks causal"):
        validate_e2_e3_causal_trace(root / "E2" / "runtime", changed)


def test_ordered_verifier_receipts_cover_reset_and_every_action(tmp_path: Path) -> None:
    campaign, runtime, sealed, episode_id = _run_canonical_recovery_bridge(
        tmp_path / "campaign"
    )
    result = validate_ordered_verifier_receipts(
        runtime,
        sealed,
        episode_id=episode_id,
    )
    action_count = len(_read_jsonl(runtime / "actions.jsonl"))
    assert result["bound_receipt_count"] == action_count + 1
    assert result["action_receipt_count"] == action_count

    changed = tmp_path / "changed-runtime"
    shutil.copytree(runtime, changed)
    terminal_path = changed / "terminal_signals.jsonl"
    terminal = _read_jsonl(terminal_path)
    terminal[1]["payload"]["receipt_binding"]["action_sha256"] = "f" * 64
    _write_jsonl(terminal_path, terminal)
    with pytest.raises(SchemaError, match="not bound"):
        validate_ordered_verifier_receipts(
            changed,
            sealed,
            episode_id=episode_id,
        )

    # A separately valid HMAC stream containing only episode_final is still
    # insufficient: final scoring cannot stand in for transition receipts.
    final_only_root = tmp_path / "final-only-campaign"
    final_sink = _sink_for(final_only_root, episode_id)
    final_sink.record_episode_final(_minimal_final_evidence())
    with pytest.raises(SchemaError, match="final-only|misses an action"):
        validate_ordered_verifier_receipts(
            runtime,
            final_sink.system_root / "sealed",
            episode_id=episode_id,
        )
    assert campaign.is_dir()


def test_receipt_commits_complete_observation_not_only_screenshot(
    tmp_path: Path,
) -> None:
    _, runtime, sealed, episode_id = _run_canonical_recovery_bridge(
        tmp_path / "campaign"
    )
    changed = tmp_path / "changed-runtime"
    shutil.copytree(runtime, changed)
    environment_path = changed / "environment_events.jsonl"
    rows = _read_jsonl(environment_path)
    reset = next(row for row in rows if row.get("event_type") == "reset")
    original_screenshot = reset["payload"]["screenshot_sha256"]
    reset["payload"]["title"] = "tampered title with identical screenshot"
    assert reset["payload"]["screenshot_sha256"] == original_screenshot
    _write_jsonl(environment_path, rows)

    with pytest.raises(SchemaError, match="complete record SHA-256 mismatch"):
        validate_ordered_verifier_receipts(
            changed,
            sealed,
            episode_id=episode_id,
        )


def test_runtime_screenshot_validator_binds_logged_path_hash_and_bytes(
    tmp_path: Path,
) -> None:
    runtime, artifact, _ = _screenshot_validator_fixture(tmp_path / "runtime")
    result = validate_runtime_screenshot_artifacts(runtime)
    assert result == {
        "status": "PASS",
        "observation_count": 1,
        "screenshot_count": 1,
    }

    artifact.chmod(0o600)
    artifact.write_bytes(b"mutated-after-log")
    with pytest.raises(SchemaError, match="read-only|byte SHA-256 mismatch"):
        validate_runtime_screenshot_artifacts(runtime)


@pytest.mark.parametrize(
    ("canary", "message"),
    (
        ("wrong_hash", "byte SHA-256 mismatch"),
        ("path_escape", "escapes"),
        ("symlink", "must not be a symlink"),
        ("future_path", "absent"),
        ("shared_latest", "share one mutable"),
    ),
)
def test_runtime_screenshot_validator_rejects_path_and_identity_canaries(
    tmp_path: Path,
    canary: str,
    message: str,
) -> None:
    runtime, artifact, observation = _screenshot_validator_fixture(
        tmp_path / canary
    )
    events_path = runtime / "environment_events.jsonl"
    events = _read_jsonl(events_path)
    if canary == "wrong_hash":
        events[0]["payload"]["screenshot_sha256"] = "0" * 64
    elif canary == "path_escape":
        outside = tmp_path / "outside.png"
        outside.write_bytes(artifact.read_bytes())
        events[0]["payload"]["screenshot_path"] = str(outside.resolve())
    elif canary == "symlink":
        outside = tmp_path / "outside.png"
        outside.write_bytes(artifact.read_bytes())
        artifact.unlink()
        artifact.symlink_to(outside)
    elif canary == "future_path":
        events[0]["payload"]["screenshot_path"] = str(
            (runtime / "screenshots" / "future.png").resolve()
        )
    elif canary == "shared_latest":
        events.append(
            {
                "event_type": "post_action_observation",
                "payload": {
                    **observation,
                    "observation_id": "obs-2",
                    "stage": "post_action",
                },
            }
        )
    _write_jsonl(events_path, events)
    with pytest.raises(SchemaError, match=message):
        validate_runtime_screenshot_artifacts(runtime)


def _run_canonical_recovery_bridge(
    campaign: Path,
) -> tuple[Path, Path, Path, str]:
    scenario = load_registered_recovery_diagnostics()[0][0]
    episode_id = f"receipt-campaign:E2:{scenario.scenario_id}:repeat-0:seed-42"
    sink = _sink_for(campaign, episode_id)
    runtime = sink.system_root / "runtime"
    logs = EpisodeEventLogs(
        runtime,
        episode_id=episode_id,
        include_memory=False,
        system_id="E2",
        task_id=scenario.scenario_id,
        repeat_id=0,
        matched_seed=42,
        timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
    )
    protocol = RuntimeProtocol(
        protocol_id="receipt-protocol",
        campaign_id="receipt-campaign",
    )
    stage_seeds = protocol.rng_factory().seed_plan(
        task_id=scenario.scenario_id,
        repeat_id=0,
        matched_seed=42,
    ).stage_seeds
    RecoveryFixtureCampaignRunner().run(
        task={
            "task_id": scenario.scenario_id,
            "task_partition": "recovery_diagnostic",
            "failure_kind": scenario.failure_kind,
            "expected_strategy": scenario.expected_strategy.value,
            "resolution_event": scenario.resolution_event,
        },
        repeat_id=0,
        model_seed=42,
        system_id="E2",
        protocol={
            "protocol_id": protocol.protocol_id,
            "statistics": {"seed": protocol.campaign_seed},
        },
        stage_seeds=stage_seeds,
        runtime_dir=runtime,
        event_logs=logs,
        verifier_sink=sink,
        episode_id=episode_id,
    )
    return campaign, runtime, sink.system_root / "sealed", episode_id


def _sink_for(campaign: Path, episode_id: str) -> SealedVerifierSink:
    scenario_id = episode_id.split(":E2:", 1)[1].split(":repeat-", 1)[0]
    return SealedVerifierSink(
        campaign,
        block_id=f"task_{scenario_id}__seed_42__repeat_0",
        attempt_id=0,
        system_id="E2",
        episode_id=episode_id,
        matched_seed=42,
        task_id=scenario_id,
        repeat_id=0,
    )


def _minimal_final_evidence() -> dict[str, object]:
    return {
        "task_success": False,
        "terminal_reason": "TERMINAL_FAILURE",
        "loop_detected": False,
        "environment_failure": False,
        "failure_incidents": [],
        "recovery_verifications": [],
        "verified_failure_event_count": 0,
        "repeated_error_event_count": 0,
        "memory_relevance": {},
    }


def _screenshot_validator_fixture(
    runtime: Path,
) -> tuple[Path, Path, dict[str, object]]:
    screenshot_root = runtime / "screenshots"
    screenshot_root.mkdir(parents=True)
    screenshot_bytes = b"validator-screenshot-bytes"
    digest = hashlib.sha256(screenshot_bytes).hexdigest()
    artifact = screenshot_root / f"000001-{digest}.png"
    artifact.write_bytes(screenshot_bytes)
    artifact.chmod(0o400)
    observation: dict[str, object] = {
        "observation_id": "obs-1",
        "screenshot_sha256": digest,
        "screenshot_path": str(artifact.resolve()),
        "stage": "reset",
    }
    _write_jsonl(
        runtime / "environment_events.jsonl",
        [{"event_type": "reset", "payload": observation}],
    )
    (runtime / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "schema_version": "table2.v1",
                "hash_algorithm": "sha256",
                "files": {f"screenshots/{artifact.name}": digest},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return runtime, artifact, observation


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
