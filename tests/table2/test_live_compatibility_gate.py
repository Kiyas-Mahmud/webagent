from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any

import pytest

from web_agent.eval.table2 import campaign as campaign_module
from web_agent.eval.table2 import live_compatibility as readiness
from web_agent.eval.table2 import package_validator as package_validator_module
from web_agent.eval.table2.campaign import CampaignRunner
from web_agent.eval.table2.common import (
    SchemaError,
    Table2Error,
    read_json,
    sha256_file,
)
from web_agent.eval.table2.schedule import (
    SYSTEM_IDS,
    build_paired_schedule,
    resolve_block_attempts,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _refresh_runtime_artifact_hashes(runtime: Path) -> None:
    files = {
        path.name: sha256_file(path)
        for path in sorted(runtime.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(runtime / "artifact_hashes.json", {"files": files})


def _manifest(*, campaign_id: str, schedule_sha256: str) -> dict[str, Any]:
    return {
        "campaign_id": campaign_id,
        "campaign_kind": "engineering_pilot",
        "campaign_mode": "evaluation",
        "evidence_label": "PILOT_ONLY",
        "publication_status": "DRAFT_PILOT_ONLY",
        "paper_table_status": "N/R",
        "matched_seeds": [42],
        "normal_block_count": 50,
        "recovery_block_count": 15,
        "scheduled_block_count": 65,
        "planned_episode_count": 260,
        "manual_rescue": "forbidden",
        "systems": list(SYSTEM_IDS),
        "repository_commit": "0123456789abcdef" * 2 + "01234567",
        "protocol_id": "table2-pc01-pilot-v1",
        "campaign_seed": 42,
        "repeat_ids": [0],
        "max_block_attempts": 2,
        "runner_identity_scope": "FROZEN_EVALUATION_RUNNER",
        "runner_entrypoint": "fixture:runner",
        "schedule_sha256": schedule_sha256,
    }


def _make_campaign(root: Path, *, campaign_id: str) -> tuple[Path, dict[str, Any]]:
    root.mkdir(parents=True)
    tasks = [
        {
            "task_id": f"recovery-{index:02d}",
            "task_partition": "recovery_diagnostic",
        }
        for index in range(15)
    ] + [
        {"task_id": f"webarena-{index:02d}", "task_partition": "normal"}
        for index in range(50)
    ]
    schedule = build_paired_schedule(
        tasks,
        campaign_id=campaign_id,
        campaign_seed=42,
        protocol_id="table2-pc01-pilot-v1",
    )
    _write_jsonl(root / "schedule/schedule.jsonl", schedule)

    for relative, content in {
        "frozen/campaign.yaml": "protocol_id: table2-pc01-pilot-v1\n",
        "frozen/protocol.yaml": (
            "loop_rule:\n"
            "  state_fingerprint_fields:\n"
            "    - url\n"
            "    - dom_hash\n"
        ),
        "frozen/environment.json": '{"environment_id":"fixture-webarena"}\n',
        "frozen/task_manifest.json": '{"task_registry":"public-0-49"}\n',
        "frozen/model/checkpoint.sha256": "checkpoint-sha\n",
        "memory/store.bin": "immutable-memory-fixture\n",
        "access_ledger.jsonl": "",
        "deviation_ledger.jsonl": "",
    }.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

    source_rows: list[dict[str, str]] = []
    for relative in readiness.LIVE_COMPATIBILITY_REQUIRED_SOURCE_RELATIVE_PATHS:
        live = REPOSITORY_ROOT / relative
        frozen = root / "frozen/runner_source" / relative
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(live, frozen)
        source_rows.append(
            {"relative_path": relative, "sha256": sha256_file(live)}
        )
    _write_json(
        root / "frozen/runner_attestation.json",
        {
            "repository_commit": "0123456789abcdef" * 2 + "01234567",
            "runner_entrypoint": "fixture:runner",
            "source_files": source_rows,
        },
    )

    immutable = [
        path
        for path in sorted((root / "frozen").rglob("*"))
        if path.is_file()
    ] + [root / "memory/store.bin"]
    artifact_files = {
        path.relative_to(root).as_posix(): sha256_file(path) for path in immutable
    }
    _write_json(root / "artifact_hashes.json", {"files": artifact_files})
    manifest = _manifest(
        campaign_id=campaign_id,
        schedule_sha256=sha256_file(root / "schedule/schedule.jsonl"),
    )
    manifest["runner_attestation_sha256"] = sha256_file(
        root / "frozen/runner_attestation.json"
    )
    _write_json(root / "campaign_manifest.json", manifest)
    for directory in ("paired_blocks", "aggregate", "manual_audit", "component_test"):
        (root / directory).mkdir()
    return root, schedule[15]


def _make_probe_block(probe: Path, row: dict[str, Any]) -> Path:
    base = readiness._block_base(probe, row)
    rerun = base / "rerun_0"
    block_manifest = {
        "block_id": row["block_id"],
        "attempt_id": 0,
        "systems": {
            system_id: {
                "launched": True,
                "completed": True,
                "infrastructure_invalid": False,
                "episode_id": f"readiness-{system_id}",
            }
            for system_id in SYSTEM_IDS
        },
    }
    _write_json(rerun / "block_manifest.json", block_manifest)
    _write_json(base / "resolution.json", resolve_block_attempts(row, [block_manifest]))
    for system_id in SYSTEM_IDS:
        episode_id = f"readiness-{system_id}"
        _write_json(
            rerun / system_id / "runtime/episode_summary.json",
            {
                "system_id": system_id,
                "episode_id": episode_id,
                "completed": True,
            },
        )
        _write_jsonl(
            rerun / system_id / f"sealed/{episode_id}/verifier_events.jsonl",
            [],
        )
        seal_key = rerun / system_id / "sealed/.seal_key"
        seal_key.write_bytes(b"k" * 32)
        seal_key.chmod(0o600)
    _write_jsonl(rerun / "E3/runtime/memory_queries.jsonl", [])
    for system_id in SYSTEM_IDS:
        _refresh_runtime_artifact_hashes(rerun / system_id / "runtime")
    return base


def _prepend_infrastructure_rerun(block: Path, row: dict[str, Any]) -> None:
    """Retain one legitimate failed whole block before the included attempt."""

    complete = block / "rerun_0"
    selected = block / "rerun_1"
    complete.rename(selected)
    selected_manifest = read_json(selected / "block_manifest.json")
    selected_manifest["attempt_id"] = 1
    _write_json(selected / "block_manifest.json", selected_manifest)

    failed = block / "rerun_0"
    failed_statuses: dict[str, dict[str, Any]] = {}
    for system_id in SYSTEM_IDS:
        (failed / system_id / "runtime").mkdir(parents=True)
        (failed / system_id / "sealed").mkdir()
        failed_statuses[system_id] = {
            "launched": False,
            "completed": False,
            "infrastructure_invalid": False,
            "fatal_noninfrastructure_error": False,
            "episode_id": None,
            "infrastructure_reason": None,
            "infrastructure_evidence_sha256": None,
        }
    failed_statuses["E0"].update(
        {
            "launched": True,
            "episode_id": "readiness-infrastructure-E0",
            "infrastructure_invalid": True,
            "infrastructure_reason": "BROWSER_UNAVAILABLE",
            "infrastructure_evidence_sha256": "a" * 64,
        }
    )
    failed_manifest = {
        "block_id": row["block_id"],
        "attempt_id": 0,
        "systems": failed_statuses,
    }
    _write_json(failed / "block_manifest.json", failed_manifest)

    runtime = failed / "E0/runtime"
    _write_json(
        runtime / "episode_summary.json",
        {"episode_id": "readiness-infrastructure-E0", "completed": False},
    )
    _write_jsonl(runtime / "terminal_signals.jsonl", [])
    _refresh_runtime_artifact_hashes(runtime)
    sealed = failed / "E0/sealed"
    seal_key = sealed / ".seal_key"
    seal_key.write_bytes(b"i" * 32)
    seal_key.chmod(0o600)
    _write_json(sealed / "infrastructure_invalid.json", {"fixture": True})
    (sealed / "runtime_exception.txt").write_text(
        "fixture infrastructure interruption\n", encoding="utf-8"
    )
    _write_json(
        block / "resolution.json",
        resolve_block_attempts(row, [failed_manifest, selected_manifest]),
    )


@pytest.fixture
def readiness_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, Any], Path]:
    target, target_first = _make_campaign(tmp_path / "target", campaign_id="target")
    probe, probe_first = _make_campaign(tmp_path / "probe", campaign_id="probe")
    assert probe_first["task_id"] == target_first["task_id"]
    block = _make_probe_block(probe, probe_first)

    def _validate_fixture(root: Path, *, field: str) -> dict[str, Any]:
        manifest = read_json(root / "campaign_manifest.json")
        readiness._validate_pilot_manifest(manifest, field=field)
        return manifest

    monkeypatch.setattr(readiness, "_validated_campaign", _validate_fixture)
    monkeypatch.setattr(
        readiness,
        "_validate_registered_included_block_schema",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        readiness,
        "validate_included_block_causal_trace",
        lambda *args, **kwargs: {
            "status": "PASS",
            "comparison_scope": "PREFIX_THROUGH_FIRST_ADMITTED_INTERVENTION",
            "first_admitted_query_index": 0,
            "e3_query_count": 1,
            "paired_reset_fingerprint": "a" * 64,
            "paired_reset_system_count": 4,
            "trained_first_pre_action_sha256": "b" * 64,
            "trained_first_pre_action_system_count": 3,
            "trained_causal_prefix": {
                "status": "PASS",
                "comparison_scope": "FULL_SHARED_TRACE_NO_P1_RECOVERY",
                "p1_recovery_intervention_present": False,
                "normal_action_count": 2,
                "observation_count": 3,
                "normalized_causal_prefix_sha256": "c" * 64,
                "first_pre_action_sha256": "b" * 64,
                "first_pre_action_event_type": "normal_action",
                "system_count": 3,
            },
        },
    )
    return probe, target, probe_first, block


def test_receipt_is_non_scored_read_only_and_exactly_replayable(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
) -> None:
    probe, target, first, _ = readiness_pair
    readiness.authorize_live_compatibility_probe(
        probe,
        target,
        block_id=first["block_id"],
    )
    receipt_path = readiness.write_live_compatibility_receipt(
        probe_campaign_dir=probe,
        target_campaign_dir=target,
    )

    receipt = readiness.validate_live_compatibility_receipt(target)
    assert receipt_path == target / readiness.LIVE_COMPATIBILITY_RELATIVE_PATH
    assert receipt["evidence_role"] == "NON_SCORED_RUNTIME_READINESS_ONLY"
    assert receipt["paper_table_eligible"] is False
    assert receipt["paper_table_status"] == "N/R"
    assert receipt["matched_block"]["systems"] == list(SYSTEM_IDS)
    assert receipt["matched_block"]["e2_memory_query_count"] == 0
    assert receipt["matched_block"]["e3_memory_write_count"] == 0
    assert "probe_campaign_path" not in receipt
    assert receipt["probe_evidence"]["relative_path"] == (
        readiness.LIVE_COMPATIBILITY_PROBE_EVIDENCE_RELATIVE_PATH.as_posix()
    )
    evidence_root = target / receipt["probe_evidence"]["relative_path"]
    assert evidence_root.is_dir()
    assert receipt["probe_evidence"]["tree"] == readiness._tree_binding(
        evidence_root,
        field="test portable probe evidence",
    )
    causal = receipt["matched_block"]["causal_validation"]
    assert causal["trained_first_pre_action_sha256"] == "b" * 64
    assert causal["trained_first_pre_action_system_count"] == 3
    assert causal["comparison_scope"] == (
        "PREFIX_THROUGH_FIRST_ADMITTED_INTERVENTION"
    )
    assert causal["trained_causal_prefix"]["normal_action_count"] == 2
    assert causal["trained_causal_prefix"]["normalized_causal_prefix_sha256"] == (
        "c" * 64
    )
    assert receipt_path.stat().st_mode & 0o222 == 0
    sidecar = target / readiness.LIVE_COMPATIBILITY_SIDECAR_RELATIVE_PATH
    assert sidecar.stat().st_mode & 0o222 == 0


@pytest.mark.parametrize(
    "canary",
    (
        "shadow_trial",
        "empty_directory",
        "symlink",
        "additional_rerun",
        "unregistered_rerun",
        "rerun_extra",
        "system_extra",
        "runtime_extra",
        "runtime_empty_directory",
        "sealed_extra",
        "sealed_empty_directory",
        "nested_symlink",
    ),
)
def test_selected_probe_block_has_exact_registered_directory_closure(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
    canary: str,
) -> None:
    probe, target, _, block = readiness_pair
    rerun = block / "rerun_0"
    if canary == "shadow_trial":
        _write_json(block / "shadow_trial/E0/outcome.json", {"selected": False})
    elif canary == "empty_directory":
        (block / "unregistered-empty").mkdir()
    elif canary == "symlink":
        (block / "unregistered-link").symlink_to(rerun, target_is_directory=True)
    elif canary == "additional_rerun":
        (block / "rerun_1").mkdir()
    elif canary == "unregistered_rerun":
        (block / "rerun_2").mkdir()
    elif canary == "rerun_extra":
        _write_json(rerun / "unregistered.json", {"selected": False})
    elif canary == "system_extra":
        (rerun / "E1/unregistered-empty").mkdir()
    elif canary == "runtime_extra":
        _write_json(rerun / "E1/runtime/unregistered.json", {"selected": False})
    elif canary == "runtime_empty_directory":
        (rerun / "E1/runtime/unregistered-empty").mkdir()
    elif canary == "sealed_extra":
        _write_json(rerun / "E1/sealed/unregistered.json", {"selected": False})
    elif canary == "sealed_empty_directory":
        (rerun / "E1/sealed/unregistered-empty").mkdir()
    else:
        (rerun / "E1/runtime/unregistered-link").symlink_to(
            rerun / "E1/runtime/episode_summary.json"
        )

    with pytest.raises(
        SchemaError,
        match=(
            "exact registered entries|exact registered tree|contains a symlink|"
            "physical attempt lacks|out-of-range physical attempt"
        ),
    ):
        readiness.write_live_compatibility_receipt(
            probe_campaign_dir=probe,
            target_campaign_dir=target,
        )


def test_registered_infrastructure_rerun_is_preserved_and_portable(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
) -> None:
    probe, target, row, block = readiness_pair
    _prepend_infrastructure_rerun(block, row)

    readiness.write_live_compatibility_receipt(
        probe_campaign_dir=probe,
        target_campaign_dir=target,
    )
    receipt = readiness.validate_live_compatibility_receipt(target)
    assert receipt["matched_block"]["selected_attempt_id"] == 1


@pytest.mark.parametrize("subtree", ("runtime", "sealed"))
def test_unlaunched_system_in_prior_rerun_must_remain_exactly_empty(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
    subtree: str,
) -> None:
    probe, target, row, block = readiness_pair
    _prepend_infrastructure_rerun(block, row)
    (block / f"rerun_0/E2/{subtree}/unregistered-empty").mkdir()

    with pytest.raises(SchemaError, match="exact registered tree"):
        readiness.write_live_compatibility_receipt(
            probe_campaign_dir=probe,
            target_campaign_dir=target,
        )


def test_live_gate_delegates_completed_block_to_registered_package_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = {"block_id": "registered-block"}
    calls: list[tuple[Path, dict[str, Any], bool]] = []

    def validate_block(
        root: Path,
        schedule_row: dict[str, Any],
        report: Any,
        *,
        require_complete: bool,
    ) -> str:
        calls.append((root, schedule_row, require_complete))
        assert report.campaign_dir == str(tmp_path)
        return "INCLUDED"

    monkeypatch.setattr(package_validator_module, "_validate_block", validate_block)
    readiness._validate_registered_included_block_schema(tmp_path, row)

    assert calls == [(tmp_path, row, True)]


def test_receipt_replay_rejects_probe_evidence_tampering(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
) -> None:
    probe, target, _, _ = readiness_pair
    readiness.write_live_compatibility_receipt(
        probe_campaign_dir=probe,
        target_campaign_dir=target,
    )
    evidence_root = target / readiness.LIVE_COMPATIBILITY_PROBE_EVIDENCE_RELATIVE_PATH
    evidence_manifest = read_json(
        evidence_root / readiness.PROBE_EVIDENCE_MANIFEST_NAME
    )
    summary = (
        evidence_root
        / evidence_manifest["selected_block_relative_path"]
        / "rerun_0/E1/runtime/episode_summary.json"
    )
    os.chmod(summary, 0o644)
    _write_json(summary, {"system_id": "E1", "completed": False})

    with pytest.raises(
        SchemaError,
        match="read-only|differs from its exact file inventory",
    ):
        readiness.validate_live_compatibility_receipt(target)


def test_receipt_replay_rejects_uninventoried_empty_block_directory(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
) -> None:
    probe, target, _, _ = readiness_pair
    readiness.write_live_compatibility_receipt(
        probe_campaign_dir=probe,
        target_campaign_dir=target,
    )
    evidence_root = target / readiness.LIVE_COMPATIBILITY_PROBE_EVIDENCE_RELATIVE_PATH
    evidence_manifest = read_json(
        evidence_root / readiness.PROBE_EVIDENCE_MANIFEST_NAME
    )
    selected_block = evidence_root / evidence_manifest["selected_block_relative_path"]
    (selected_block / "unregistered-empty").mkdir()

    with pytest.raises(SchemaError, match="exact registered entries"):
        readiness.validate_live_compatibility_receipt(target)


@pytest.mark.parametrize("location", ("root", "ancestor"))
def test_receipt_replay_rejects_uninventoried_empty_evidence_directory(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
    location: str,
) -> None:
    probe, target, _, _ = readiness_pair
    readiness.write_live_compatibility_receipt(
        probe_campaign_dir=probe,
        target_campaign_dir=target,
    )
    evidence_root = target / readiness.LIVE_COMPATIBILITY_PROBE_EVIDENCE_RELATIVE_PATH
    destination = (
        evidence_root / "unregistered-empty"
        if location == "root"
        else evidence_root / "paired_blocks/unregistered-empty"
    )
    destination.mkdir()

    with pytest.raises(SchemaError, match="exact registered entries"):
        readiness.validate_live_compatibility_receipt(target)


def test_receipt_replay_rejects_target_runtime_rebinding(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
) -> None:
    probe, target, _, _ = readiness_pair
    readiness.write_live_compatibility_receipt(
        probe_campaign_dir=probe,
        target_campaign_dir=target,
    )
    (target / "frozen/environment.json").write_text(
        '{"environment_id":"substituted"}\n', encoding="utf-8"
    )

    with pytest.raises(SchemaError, match="runtime input hash mismatch"):
        readiness.validate_live_compatibility_receipt(target)


def test_probe_must_be_first_normal_block_and_target_must_be_unstarted(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
) -> None:
    probe, target, first, _ = readiness_pair
    schedule = [
        json.loads(line)
        for line in (probe / "schedule/schedule.jsonl").read_text().splitlines()
        if line
    ]
    for forbidden in (schedule[0], schedule[16]):
        with pytest.raises(SchemaError, match="first registered normal"):
            readiness.authorize_live_compatibility_probe(
                probe,
                target,
                block_id=forbidden["block_id"],
            )

    started = target / readiness._block_base(target, first).relative_to(target)
    _write_json(started / "unexpected.json", {"started": True})
    with pytest.raises(SchemaError, match="before any target pilot block starts"):
        readiness.authorize_live_compatibility_probe(
            probe,
            target,
            block_id=first["block_id"],
        )


def test_probe_rejects_an_empty_second_block_and_empty_target_start(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
) -> None:
    probe, target, first, _ = readiness_pair
    (probe / "paired_blocks/seed_42/unregistered-empty").mkdir()
    with pytest.raises(SchemaError, match="another attempted/scored block"):
        readiness.authorize_live_compatibility_probe(
            probe,
            target,
            block_id=first["block_id"],
        )

    shutil.rmtree(probe / "paired_blocks/seed_42/unregistered-empty")
    (target / "paired_blocks/unregistered-empty").mkdir()
    with pytest.raises(SchemaError, match="before any target pilot block starts"):
        readiness.authorize_live_compatibility_probe(
            probe,
            target,
            block_id=first["block_id"],
        )


def test_target_access_ledger_proves_it_was_already_started(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
) -> None:
    probe, target, first, _ = readiness_pair
    _write_jsonl(
        target / "access_ledger.jsonl",
        [{"event_type": "episode_task_load", "payload": {"task_id": "removed"}}],
    )
    with pytest.raises(SchemaError, match="before any target pilot block starts"):
        readiness.authorize_live_compatibility_probe(
            probe,
            target,
            block_id=first["block_id"],
        )


@pytest.mark.parametrize(
    ("system_id", "event_type", "message"),
    [
        ("E2", "post_failure_query", "E2 performed forbidden memory retrieval"),
        ("E3", "memory_write", "E3 performed a forbidden memory write"),
    ],
)
def test_memory_role_canaries_fail_receipt_issuance(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
    system_id: str,
    event_type: str,
    message: str,
) -> None:
    probe, target, _, block = readiness_pair
    _write_jsonl(
        block / f"rerun_0/{system_id}/runtime/memory_queries.jsonl",
        [{"event_type": event_type}],
    )
    _refresh_runtime_artifact_hashes(block / f"rerun_0/{system_id}/runtime")
    with pytest.raises(SchemaError, match=message):
        readiness.write_live_compatibility_receipt(
            probe_campaign_dir=probe,
            target_campaign_dir=target,
        )


def test_missing_attested_gate_source_fails_before_probe(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
) -> None:
    probe, target, first, _ = readiness_pair
    required = (
        target
        / "frozen/runner_source"
        / readiness.LIVE_COMPATIBILITY_SOURCE_RELATIVE_PATH
    )
    required.unlink()
    with pytest.raises(SchemaError, match="runtime input|source is not frozen"):
        readiness.authorize_live_compatibility_probe(
            probe,
            target,
            block_id=first["block_id"],
        )


def test_real_pilot_requires_receipt_but_engineering_smoke_does_not(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    pilot = _manifest(campaign_id="target", schedule_sha256="0" * 64)
    with pytest.raises(SchemaError, match="runtime_readiness|live-readiness receipt"):
        readiness.require_live_compatibility_before_execution(root, pilot)

    assert (
        readiness.require_live_compatibility_before_execution(
            root,
            {
                "campaign_kind": "engineering_pilot",
                "campaign_mode": "smoke",
                "evidence_label": "PILOT_ONLY",
            },
        )
        is None
    )


def test_package_boundary_rejects_started_or_complete_pilot_without_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    (root / "paired_blocks").mkdir(parents=True)
    manifest = _manifest(campaign_id="target", schedule_sha256="0" * 64)
    package_validator_module._validate_live_readiness_boundary(
        root,
        manifest,
        access_records=[],
        require_complete=False,
    )
    with pytest.raises(SchemaError, match="live-readiness receipt"):
        package_validator_module._validate_live_readiness_boundary(
            root,
            manifest,
            access_records=[],
            require_complete=True,
        )
    _write_json(root / "paired_blocks" / "started.json", {"started": True})
    with pytest.raises(SchemaError, match="live-readiness receipt"):
        package_validator_module._validate_live_readiness_boundary(
            root,
            manifest,
            access_records=[],
            require_complete=False,
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        package_validator_module.validate_campaign(
            root,
            require_complete=False,
            _skip_live_readiness=True,
        )


def test_package_boundary_replays_receipt_and_rejects_smoke_or_final_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    readiness_root = root / readiness.LIVE_COMPATIBILITY_RELATIVE_PATH.parent
    _write_json(readiness_root / "receipt-fixture.json", {"status": "PASS"})
    calls: list[Path] = []
    monkeypatch.setattr(
        package_validator_module,
        "validate_live_compatibility_receipt",
        lambda value: calls.append(Path(value)),
    )
    pilot = _manifest(campaign_id="target", schedule_sha256="0" * 64)
    package_validator_module._validate_live_readiness_boundary(
        root,
        pilot,
        access_records=[],
        require_complete=False,
    )
    assert calls == [root]

    for manifest in (
        {**pilot, "campaign_mode": "smoke"},
        {
            **pilot,
            "campaign_kind": "locked_final",
            "evidence_label": "FINAL_LOCKED",
        },
    ):
        with pytest.raises(SchemaError, match="cannot contain"):
            package_validator_module._validate_live_readiness_boundary(
                root,
                manifest,
                access_records=[],
                require_complete=False,
            )


@pytest.mark.parametrize(
    "relative",
    (
        readiness.LIVE_COMPATIBILITY_RELATIVE_PATH,
        readiness.LIVE_COMPATIBILITY_PROBE_EVIDENCE_RELATIVE_PATH / "canary.json",
    ),
)
def test_final_campaign_rejects_provisional_pc01_receipt(
    tmp_path: Path,
    relative: Path,
) -> None:
    root = tmp_path / "final"
    (root / relative).parent.mkdir(parents=True)
    _write_json(root / relative, {"status": "PASS"})

    with pytest.raises(Table2Error, match="cannot accept provisional PC-01"):
        readiness.require_live_compatibility_before_execution(
            root,
            {
                "campaign_kind": "locked_final",
                "campaign_mode": "evaluation",
                "evidence_label": "FINAL_LOCKED",
            },
        )


def test_campaign_runner_checks_receipt_adjacent_to_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = object.__new__(CampaignRunner)
    runner.root = tmp_path
    runner.runner = object()
    runner.runner_entrypoint = "fixture:runner"
    runner.live_readiness_probe_target = None
    runner.manifest = {
        "campaign_kind": "engineering_pilot",
        "campaign_mode": "evaluation",
        "evidence_label": "PILOT_ONLY",
    }
    row = {
        "block_id": "block-1",
        "task_id": "task-1",
    }
    runner.schedule = [row]
    runner.tasks = {"task-1": {"task_id": "task-1"}}
    runner.protocol = {}
    runner.systems = {}

    dispatched = False

    def _must_not_dispatch(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal dispatched
        dispatched = True
        return {}

    monkeypatch.setattr(campaign_module, "verify_runner_before_execution", lambda *a, **k: None)
    monkeypatch.setattr(campaign_module, "_run_block", _must_not_dispatch)
    monkeypatch.setattr(
        campaign_module,
        "require_live_compatibility_before_execution",
        lambda *a, **k: (_ for _ in ()).throw(Table2Error("missing readiness")),
    )

    with pytest.raises(Table2Error, match="missing readiness"):
        runner.run_block(row)
    assert dispatched is False


def test_campaign_runner_engineering_smoke_still_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = object.__new__(CampaignRunner)
    runner.root = tmp_path
    runner.runner = object()
    runner.runner_entrypoint = "fixture:runner"
    runner.live_readiness_probe_target = None
    runner.manifest = {
        "campaign_kind": "engineering_pilot",
        "campaign_mode": "smoke",
        "evidence_label": "PILOT_ONLY",
    }
    row = {"block_id": "block-1", "task_id": "task-1"}
    runner.schedule = [row]
    runner.tasks = {"task-1": {"task_id": "task-1"}}
    runner.protocol = {}
    runner.systems = {}
    monkeypatch.setattr(
        campaign_module,
        "verify_runner_before_execution",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        campaign_module,
        "_run_block",
        lambda *args, **kwargs: {"status": "INCLUDED"},
    )

    assert runner.run_block(row) == {"status": "INCLUDED"}


def test_receipt_sidecar_is_mandatory(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
) -> None:
    probe, target, _, _ = readiness_pair
    readiness.write_live_compatibility_receipt(
        probe_campaign_dir=probe,
        target_campaign_dir=target,
    )
    sidecar = target / readiness.LIVE_COMPATIBILITY_SIDECAR_RELATIVE_PATH
    os.chmod(sidecar, 0o644)
    sidecar.unlink()

    with pytest.raises(
        SchemaError,
        match="receipt/probe-evidence/sidecar closure|receipt sidecar",
    ):
        readiness.validate_live_compatibility_receipt(target)


@pytest.mark.parametrize("canary", ["extra", "symlink"])
def test_runtime_readiness_directory_has_exact_file_closure(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
    canary: str,
) -> None:
    probe, target, _, _ = readiness_pair
    receipt = readiness.write_live_compatibility_receipt(
        probe_campaign_dir=probe,
        target_campaign_dir=target,
    )
    readiness_root = receipt.parent
    if canary == "extra":
        _write_json(readiness_root / "unregistered.json", {"status": "PASS"})
    else:
        sidecar = target / readiness.LIVE_COMPATIBILITY_SIDECAR_RELATIVE_PATH
        sidecar.unlink()
        sidecar.symlink_to(receipt)

    with pytest.raises(SchemaError, match="exact canonical"):
        readiness.validate_live_compatibility_receipt(target)


def test_portable_probe_survives_source_deletion_and_target_relocation(
    readiness_pair: tuple[Path, Path, dict[str, Any], Path],
) -> None:
    probe, target, _, _ = readiness_pair
    readiness.write_live_compatibility_receipt(
        probe_campaign_dir=probe,
        target_campaign_dir=target,
    )
    shutil.rmtree(probe)
    relocated = target.parent / "relocated-target"
    target.rename(relocated)

    receipt = readiness.require_live_compatibility_before_execution(
        relocated,
        read_json(relocated / "campaign_manifest.json"),
    )
    assert receipt is not None
    assert "probe_campaign_path" not in receipt
    assert receipt["probe_evidence"]["relative_path"] == (
        readiness.LIVE_COMPATIBILITY_PROBE_EVIDENCE_RELATIVE_PATH.as_posix()
    )
