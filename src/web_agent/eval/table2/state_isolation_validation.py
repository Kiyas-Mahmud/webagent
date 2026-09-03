"""Fail-closed validation for production episode backend-state isolation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from web_agent.runtime.contracts import SystemID
from web_agent.runtime.state_reset import (
    EpisodeStateResetEvidence,
    EpisodeStateResetRequest,
    PreBrowserSetupEvidence,
    WebArenaResetStateReceipt,
)
from web_agent.runtime.protocol import REGISTERED_PRE_BROWSER_SETUP_TIMEOUT_SECONDS

from .common import SchemaError, read_json, read_jsonl
from .schedule import SYSTEM_IDS


RESET_EVENT_TYPE = "episode_state_reset"
PRE_BROWSER_SETUP_EVENT_TYPE = "pre_browser_setup_boundary"
WEBARENA_RESET_RECEIPT_EVENT_TYPE = "webarena_reset_state_receipt"
_RESET_PAYLOAD_FIELDS = frozenset(
    {"request", "request_sha256", "evidence", "evidence_sha256"}
)


def _lowercase_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def frozen_runner_source_hashes(campaign_root: str | Path) -> frozenset[str]:
    """Load the exact source hash of the frozen runtime integration."""

    attestation = read_json(Path(campaign_root) / "frozen" / "runner_attestation.json")
    rows = attestation.get("source_files")
    if not isinstance(rows, list) or not rows:
        raise SchemaError("frozen runner attestation has no source-file evidence")
    hashes: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not _lowercase_sha256(row.get("sha256")):
            raise SchemaError("frozen runner attestation contains an invalid source hash")
        hashes.add(str(row["sha256"]))
    runtime_identity = attestation.get("runtime_identity")
    integration = (
        runtime_identity.get("runtime_integration")
        if isinstance(runtime_identity, Mapping)
        else None
    )
    integration_source = (
        integration.get("source_sha256") if isinstance(integration, Mapping) else None
    )
    if not _lowercase_sha256(integration_source) or integration_source not in hashes:
        raise SchemaError(
            "frozen runtime-integration source is absent from attested source files"
        )
    return frozenset({str(integration_source)})


def _payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("payload")
    if not isinstance(value, Mapping):
        raise SchemaError("episode-state reset event payload must be an object")
    if set(value) != _RESET_PAYLOAD_FIELDS:
        raise SchemaError(
            "episode-state reset payload must contain only the registered request/evidence"
        )
    return value


def validate_episode_state_reset(
    runtime_dir: str | Path,
    *,
    expected_episode_id: str,
    expected_system_id: str,
    expected_reset_stage_seed: int,
    attested_source_sha256: frozenset[str],
) -> EpisodeStateResetEvidence:
    """Require one first-event reset whose typed evidence matches this package."""

    runtime = Path(runtime_dir)
    records = read_jsonl(runtime / "environment_events.jsonl")
    reset_records = [
        record
        for record in records
        if str(record.get("event_type", "")) == RESET_EVENT_TYPE
    ]
    if len(reset_records) != 1:
        raise SchemaError(
            "production episode must contain exactly one episode-state reset event"
        )
    record = reset_records[0]
    if records[0] is not record:
        raise SchemaError("episode-state reset must be the first environment event")
    if type(record.get("sequence")) is not int or record.get("sequence") != 1:
        raise SchemaError("episode-state reset must have exact event sequence 1")
    if record.get("stream") != "environment_events":
        raise SchemaError("episode-state reset is recorded in the wrong stream")
    if record.get("episode_id") != expected_episode_id:
        raise SchemaError("episode-state reset event cites another episode")
    if record.get("system_id") != expected_system_id:
        raise SchemaError("episode-state reset event cites another system")
    if (
        isinstance(expected_reset_stage_seed, bool)
        or not isinstance(expected_reset_stage_seed, int)
        or expected_reset_stage_seed < 0
    ):
        raise SchemaError("registered reset-stage seed is not an exact nonnegative integer")
    try:
        expected_system = SystemID(expected_system_id)
    except ValueError as exc:
        raise SchemaError("episode-state reset has an unregistered system") from exc

    payload = _payload(record)
    request_value = payload["request"]
    evidence_value = payload["evidence"]
    if not isinstance(request_value, Mapping) or not isinstance(
        evidence_value, Mapping
    ):
        raise SchemaError("episode-state reset typed records must be objects")
    for field in ("request_sha256", "evidence_sha256"):
        if not _lowercase_sha256(payload[field]):
            raise SchemaError(f"episode-state reset {field} is not lowercase SHA-256")
    try:
        request = EpisodeStateResetRequest.from_dict(request_value)
        evidence = EpisodeStateResetEvidence.from_dict(evidence_value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"invalid typed episode-state reset evidence: {exc}") from exc
    if not isinstance(request, EpisodeStateResetRequest) or not isinstance(
        evidence, EpisodeStateResetEvidence
    ):
        raise SchemaError("episode-state reset record reconstruction changed type")
    if payload["request_sha256"] != request.record_sha256:
        raise SchemaError("episode-state reset request hash mismatch")
    if payload["evidence_sha256"] != evidence.record_sha256:
        raise SchemaError("episode-state reset evidence hash mismatch")

    expected = (expected_episode_id, expected_system, expected_reset_stage_seed)
    request_identity = (
        request.episode_id,
        request.system_id,
        request.reset_stage_seed,
    )
    evidence_identity = (
        evidence.episode_id,
        evidence.system_id,
        evidence.reset_stage_seed,
    )
    if request_identity != expected:
        raise SchemaError("episode-state reset request differs from scheduled identity")
    if evidence_identity != expected or evidence_identity != request_identity:
        raise SchemaError("episode-state reset evidence differs from its request")
    if (
        not attested_source_sha256
        or any(not _lowercase_sha256(item) for item in attested_source_sha256)
        or evidence.resetter_source_sha256 not in attested_source_sha256
    ):
        raise SchemaError(
            "episode-state resetter source is absent from the frozen runner attestation"
        )
    return evidence


def validate_webarena_reset_state_receipt(
    runtime_dir: str | Path,
    *,
    expected_episode_id: str,
    expected_system_id: str,
    expected_task_id: str,
    expected_benchmark_version: str | None,
    expected_start_state_id: str | None,
    expected_reset_stage_seed: int,
    attested_source_sha256: frozenset[str],
) -> WebArenaResetStateReceipt:
    """Validate the setup boundary and one hashes-only hidden-state receipt."""

    records = read_jsonl(Path(runtime_dir) / "environment_events.jsonl")
    setup_rows = [
        row
        for row in records
        if str(row.get("event_type", "")) == PRE_BROWSER_SETUP_EVENT_TYPE
    ]
    receipt_rows = [
        row
        for row in records
        if str(row.get("event_type", "")) == WEBARENA_RESET_RECEIPT_EVENT_TYPE
    ]
    if len(setup_rows) != 1:
        raise SchemaError(
            "ordinary WebArena episode must contain exactly one pre-browser setup boundary"
        )
    if len(receipt_rows) != 1:
        raise SchemaError(
            "ordinary WebArena episode must contain exactly one reset-state receipt"
        )
    setup_row = setup_rows[0]
    receipt_row = receipt_rows[0]
    if (
        type(setup_row.get("sequence")) is not int
        or type(receipt_row.get("sequence")) is not int
        or receipt_row["sequence"] != setup_row["sequence"] + 1
    ):
        raise SchemaError(
            "WebArena reset-state receipt must immediately follow the browser-reset boundary"
        )
    for row, label in ((setup_row, "pre-browser setup"), (receipt_row, "reset-state")):
        if row.get("stream") != "environment_events":
            raise SchemaError(f"{label} evidence is recorded in the wrong stream")
        if row.get("episode_id") != expected_episode_id:
            raise SchemaError(f"{label} evidence cites another episode")

    setup_payload = setup_row.get("payload")
    if not isinstance(setup_payload, Mapping) or set(setup_payload) != {
        "evidence",
        "evidence_sha256",
    }:
        raise SchemaError("pre-browser setup payload has unregistered fields")
    setup_value = setup_payload["evidence"]
    if not isinstance(setup_value, Mapping) or not _lowercase_sha256(
        setup_payload["evidence_sha256"]
    ):
        raise SchemaError("pre-browser setup evidence/hash is malformed")
    try:
        setup = PreBrowserSetupEvidence.from_dict(setup_value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"invalid typed pre-browser setup evidence: {exc}") from exc
    if not isinstance(setup, PreBrowserSetupEvidence):
        raise SchemaError("pre-browser setup evidence reconstruction changed type")
    if setup_payload["evidence_sha256"] != setup.record_sha256:
        raise SchemaError("pre-browser setup evidence hash mismatch")
    if setup.episode_id != expected_episode_id:
        raise SchemaError("pre-browser setup evidence cites another episode")
    try:
        expected_system = SystemID(expected_system_id)
    except ValueError as exc:
        raise SchemaError("pre-browser setup has an unregistered system") from exc
    if setup.system_id is not expected_system:
        raise SchemaError("pre-browser setup evidence cites another system")
    if setup.timeout_seconds != REGISTERED_PRE_BROWSER_SETUP_TIMEOUT_SECONDS:
        raise SchemaError("pre-browser setup timeout differs from registration")

    receipt_payload = receipt_row.get("payload")
    if not isinstance(receipt_payload, Mapping) or set(receipt_payload) != {
        "receipt",
        "receipt_sha256",
    }:
        raise SchemaError("WebArena reset-state receipt payload has unregistered fields")
    receipt_value = receipt_payload["receipt"]
    if not isinstance(receipt_value, Mapping) or not _lowercase_sha256(
        receipt_payload["receipt_sha256"]
    ):
        raise SchemaError("WebArena reset-state receipt/hash is malformed")
    try:
        receipt = WebArenaResetStateReceipt.from_dict(receipt_value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"invalid typed WebArena reset-state receipt: {exc}") from exc
    if not isinstance(receipt, WebArenaResetStateReceipt):
        raise SchemaError("WebArena reset-state receipt reconstruction changed type")
    if receipt_payload["receipt_sha256"] != receipt.record_sha256:
        raise SchemaError("WebArena reset-state receipt hash mismatch")
    expected_identity = (
        expected_episode_id,
        expected_task_id,
        expected_reset_stage_seed,
    )
    actual_identity = (
        receipt.episode_id,
        receipt.task_id,
        receipt.reset_stage_seed,
    )
    if actual_identity != expected_identity:
        raise SchemaError("WebArena reset-state receipt differs from scheduled identity")
    if (
        expected_benchmark_version is not None
        and receipt.benchmark_version != expected_benchmark_version
    ):
        raise SchemaError("WebArena reset-state receipt changed benchmark version")
    if (
        expected_start_state_id is not None
        and receipt.start_state_id != expected_start_state_id
    ):
        raise SchemaError("WebArena reset-state receipt changed frozen start state")
    if (
        not attested_source_sha256
        or any(not _lowercase_sha256(item) for item in attested_source_sha256)
        or receipt.attester_source_sha256 not in attested_source_sha256
    ):
        raise SchemaError(
            "WebArena reset-state attester source is absent from the frozen runner "
            "attestation"
        )
    return receipt


def validate_included_block_state_isolation(
    selected_attempt_dir: str | Path,
    *,
    schedule_row: Mapping[str, Any],
    attested_source_sha256: frozenset[str],
) -> None:
    """Reject schedule-order contamination across the matched E0--E3 block."""

    attempt = Path(selected_attempt_dir)
    stage_seeds = schedule_row.get("stage_seeds")
    if not isinstance(stage_seeds, Mapping) or "reset" not in stage_seeds:
        raise SchemaError("schedule row lacks its registered reset-stage seed")
    reset_seed = stage_seeds["reset"]
    commitments: dict[str, tuple[Any, ...]] = {}
    environment_commitments: dict[str, tuple[Any, ...]] = {}
    for system_id in SYSTEM_IDS:
        runtime = attempt / system_id / "runtime"
        manifest = read_json(runtime / "episode_manifest.json")
        evidence = validate_episode_state_reset(
            runtime,
            expected_episode_id=str(manifest.get("episode_id", "")),
            expected_system_id=system_id,
            expected_reset_stage_seed=reset_seed,
            attested_source_sha256=attested_source_sha256,
        )
        commitments[system_id] = (
            evidence.resetter_id,
            evidence.resetter_version,
            evidence.resetter_source_sha256,
            evidence.reset_stage_seed,
            tuple(sorted(dict(evidence.backend_state_sha256).items())),
            evidence.initial_state_sha256,
        )
        if str(schedule_row.get("task_partition")) == "normal":
            reset_receipt = validate_webarena_reset_state_receipt(
                runtime,
                expected_episode_id=str(manifest.get("episode_id", "")),
                expected_system_id=system_id,
                expected_task_id=str(schedule_row.get("task_id", "")),
                expected_benchmark_version=None,
                expected_start_state_id=None,
                expected_reset_stage_seed=reset_seed,
                attested_source_sha256=attested_source_sha256,
            )
            environment_commitments[system_id] = (
                reset_receipt.attester_id,
                reset_receipt.attester_version,
                reset_receipt.attester_source_sha256,
                reset_receipt.task_id,
                reset_receipt.benchmark_version,
                reset_receipt.start_state_id,
                reset_receipt.reset_stage_seed,
                tuple(sorted(dict(reset_receipt.commitments_sha256).items())),
                reset_receipt.reset_state_sha256,
            )
    reference = commitments["E0"]
    contaminated = [
        system_id
        for system_id in SYSTEM_IDS
        if commitments[system_id] != reference
    ]
    if contaminated:
        raise SchemaError(
            "matched E0--E3 initial backend state differs after reset; "
            "possible cross-system/schedule-order contamination: "
            + ", ".join(contaminated)
        )
    if environment_commitments:
        environment_reference = environment_commitments["E0"]
        reset_contaminated = [
            system_id
            for system_id in SYSTEM_IDS
            if environment_commitments[system_id] != environment_reference
        ]
        if reset_contaminated:
            raise SchemaError(
                "matched E0--E3 hidden WebArena reset state differs; possible "
                "service/account/database/start-state contamination: "
                + ", ".join(reset_contaminated)
            )
