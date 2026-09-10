from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import base64
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from web_agent.eval.table2 import deployment_authority as authority
from web_agent.eval.table2.common import SchemaError, canonical_json_bytes, sha256_json


ROOT = Path(__file__).resolve().parents[2]
FIXED_NOW = datetime(2026, 9, 5, 0, 0, 0, tzinfo=timezone.utc)
VERIFY_NOW = datetime(2026, 9, 5, 0, 0, 3, tzinfo=timezone.utc)
PUBLIC_KEY_BASE64 = "A6EHv/POEL4dcN0Y50vAmWfk1jCbpQ1fHdyGZBJVMbg="
PUBLIC_KEY_SHA256 = "56475aa75463474c0285df5dbf2bcab73da651358839e9b77481b2eab107708c"
PROFILE_SHA256 = "7" * 64
CAMPAIGN_ID = "table2-pc01-pilot-fixture"
BLOCK_ID = "block-fixture-0"
RERUN_ID = 0
DGX_SERVICE_SESSION_ID = "dgx-session-fixture-1"
VALID_SIGNATURE_BASE64 = (
    "id6fO63tzEy7vom8PFYPgPCv+LNXtDIeRzZY2mQ1fdov30UmMqfjwOJclJt/"
    "cE3EGUu+1SNg7qNSKpJ66btoBw=="
)
BLOCK_CLOSE_SIGNATURE_BASE64 = (
    "A9yXtHMYZV4bHejBssjy+RNmkQcT6kGDBq1A0WGkZHvZs/UxtmdPqdMOQMw1NR69"
    "bsjs3ftJDSpzEys3Cw5KDA=="
)


def _registry() -> dict:
    return {
        "schema_version": authority.AUTHORITY_REGISTRY_SCHEMA_VERSION,
        "registry_id": authority.AUTHORITY_REGISTRY_ID,
        "status": authority.ACTIVE_REGISTRY_STATUS,
        "authorities": [
            {
                "authority_id": "independent-lab-a",
                "key_id": "lab-key-2026-01",
                "algorithm": authority.SIGNATURE_ALGORITHM,
                "public_key_base64": PUBLIC_KEY_BASE64,
                "public_key_sha256": PUBLIC_KEY_SHA256,
                "allowed_roles": sorted(
                    [
                        authority.DGX_DEPLOYMENT_ROLE,
                        authority.RUNTIME_VALUE_PROVENANCE_ROLE,
                    ]
                ),
                "allowed_phases": sorted(authority.AUTHORITY_PHASES),
                "allowed_topologies": [authority.SPLIT_DEPLOYMENT_TOPOLOGY],
                "measurement_profile_id": "lab-independent-observer-v1",
                "measurement_profile_sha256": PROFILE_SHA256,
                "valid_from_utc": "2026-01-01T00:00:00Z",
                "valid_until_utc": "2027-01-01T00:00:00Z",
                "revoked": False,
            }
        ],
    }


def _startup_challenge() -> dict:
    return {
        "schema_version": authority.CHALLENGE_SCHEMA_VERSION,
        "record_type": authority.CHALLENGE_RECORD_TYPE,
        "registry_sha256": sha256_json(_registry()),
        "campaign_id": CAMPAIGN_ID,
        "handoff_manifest_sha256": "1" * 64,
        "runner_attestation_sha256": "2" * 64,
        "semantic_dependency_lock_sha256": "3" * 64,
        "deployment_topology": authority.SPLIT_DEPLOYMENT_TOPOLOGY,
        "authority_id": "independent-lab-a",
        "key_id": "lab-key-2026-01",
        "authority_role": authority.DGX_DEPLOYMENT_ROLE,
        "measurement_profile_id": "lab-independent-observer-v1",
        "measurement_profile_sha256": PROFILE_SHA256,
        "phase": authority.DGX_SERVICE_STARTUP_PHASE,
        "block_id": None,
        "rerun_id": None,
        "challenge_nonce": "4" * 64,
        "issued_at_utc": "2026-09-05T00:00:00Z",
        "expires_at_utc": "2026-09-05T00:05:00Z",
        "previous_challenge_sha256": None,
        "previous_receipt_sha256": None,
    }


def _challenge_for_phase(
    phase: str,
    *,
    previous_receipt_sha256: str | None,
    block_id: str | None,
    rerun_id: int | None,
) -> dict:
    challenge = _startup_challenge()
    challenge.update(
        {
            "authority_role": authority.PHASE_ROLES[phase],
            "phase": phase,
            "block_id": block_id,
            "rerun_id": rerun_id,
            "previous_receipt_sha256": previous_receipt_sha256,
        }
    )
    return challenge


def _common_payload(
    phase: str,
    *,
    previous_receipt_sha256: str | None,
    block_id: str | None,
    rerun_id: int | None,
) -> dict:
    role = authority.PHASE_ROLES[phase]
    challenge = _challenge_for_phase(
        phase,
        previous_receipt_sha256=previous_receipt_sha256,
        block_id=block_id,
        rerun_id=rerun_id,
    )
    return {
        "schema_version": authority.PAYLOAD_SCHEMA_BY_PHASE[phase],
        "record_type": authority.PAYLOAD_RECORD_TYPE_BY_PHASE[phase],
        "phase": phase,
        "registry_sha256": sha256_json(_registry()),
        "authority_id": "independent-lab-a",
        "key_id": "lab-key-2026-01",
        "authority_role": role,
        "measurement_profile_id": "lab-independent-observer-v1",
        "measurement_profile_sha256": PROFILE_SHA256,
        "deployment_topology": authority.SPLIT_DEPLOYMENT_TOPOLOGY,
        "campaign_id": CAMPAIGN_ID,
        "handoff_manifest_sha256": "1" * 64,
        "runner_attestation_sha256": "2" * 64,
        "semantic_dependency_lock_sha256": "3" * 64,
        "challenge_sha256": sha256_json(challenge),
        "challenge_nonce": "4" * 64,
        "block_id": block_id,
        "rerun_id": rerun_id,
        "issued_at_utc": "2026-09-05T00:00:02Z",
        "measured_at_utc": "2026-09-05T00:00:01Z",
        "previous_receipt_sha256": previous_receipt_sha256,
        "authority_sequence": 1,
        "dgx_service_session_id": DGX_SERVICE_SESSION_ID,
        "dgx_host_identity_sha256": "5" * 64,
        "dgx_runtime_identity_sha256": "6" * 64,
        "bridge_identity_sha256": "8" * 64,
        "measurements": {},
    }


def _startup_payload() -> dict:
    payload = _common_payload(
        authority.DGX_SERVICE_STARTUP_PHASE,
        previous_receipt_sha256=None,
        block_id=None,
        rerun_id=None,
    )
    payload["measurements"] = {
        "service_executable_sha256": "9" * 64,
        "container_image_digest": "sha256:" + "a" * 64,
        "runtime_source_set_sha256": "b" * 64,
        "cuda_runtime_identity_sha256": "c" * 64,
        "dependency_identity_sha256": "d" * 64,
        "endpoint_identity_sha256": "e" * 64,
        "service_process_identity_sha256": "f" * 64,
        "independent_observation_method": authority.INDEPENDENT_OBSERVATION_METHOD,
    }
    return payload


def _startup_envelope() -> dict:
    payload = _startup_payload()
    return {
        "schema_version": authority.SIGNED_ENVELOPE_SCHEMA_VERSION,
        "record_type": authority.SIGNED_ENVELOPE_RECORD_TYPE,
        "algorithm": authority.SIGNATURE_ALGORITHM,
        "canonicalization": authority.CANONICAL_JSON_ALGORITHM,
        "authority_id": "independent-lab-a",
        "key_id": "lab-key-2026-01",
        "payload_sha256": sha256_json(payload),
        "payload": payload,
        "signature_base64": VALID_SIGNATURE_BASE64,
    }


def _signed_envelope(payload: dict) -> dict:
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    signature = private_key.sign(
        authority.SIGNATURE_DOMAIN + canonical_json_bytes(payload)
    )
    return {
        "schema_version": authority.SIGNED_ENVELOPE_SCHEMA_VERSION,
        "record_type": authority.SIGNED_ENVELOPE_RECORD_TYPE,
        "algorithm": authority.SIGNATURE_ALGORITHM,
        "canonicalization": authority.CANONICAL_JSON_ALGORITHM,
        "authority_id": payload["authority_id"],
        "key_id": payload["key_id"],
        "payload_sha256": sha256_json(payload),
        "payload": payload,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def _payload_for_issued_challenge(
    challenge: dict, *, authority_sequence: int
) -> dict:
    phase = challenge["phase"]
    payload = deepcopy(_phase_payloads()[phase])
    for field in authority._CHALLENGE_TO_PAYLOAD_FIELDS:
        payload[field] = challenge[field]
    payload["challenge_sha256"] = sha256_json(challenge)
    payload["issued_at_utc"] = challenge["issued_at_utc"]
    payload["measured_at_utc"] = challenge["issued_at_utc"]
    payload["authority_sequence"] = authority_sequence
    predecessor_measurement = {
        authority.DGX_MODEL_LOAD_PHASE: "startup_receipt_sha256",
        authority.BLOCK_OPEN_PHASE: "model_load_receipt_sha256",
        authority.BLOCK_CLOSE_PHASE: "block_open_receipt_sha256",
    }.get(phase)
    if predecessor_measurement is not None:
        payload["measurements"][predecessor_measurement] = challenge[
            "previous_receipt_sha256"
        ]
    return payload


def _block_close_challenge() -> dict:
    return _challenge_for_phase(
        authority.BLOCK_CLOSE_PHASE,
        previous_receipt_sha256="c" * 64,
        block_id=BLOCK_ID,
        rerun_id=RERUN_ID,
    )


def _block_close_envelope() -> dict:
    payload = _phase_payloads()[authority.BLOCK_CLOSE_PHASE]
    return {
        "schema_version": authority.SIGNED_ENVELOPE_SCHEMA_VERSION,
        "record_type": authority.SIGNED_ENVELOPE_RECORD_TYPE,
        "algorithm": authority.SIGNATURE_ALGORITHM,
        "canonicalization": authority.CANONICAL_JSON_ALGORITHM,
        "authority_id": "independent-lab-a",
        "key_id": "lab-key-2026-01",
        "payload_sha256": sha256_json(payload),
        "payload": payload,
        "signature_base64": BLOCK_CLOSE_SIGNATURE_BASE64,
    }


def _phase_payloads() -> dict[str, dict]:
    startup = _startup_payload()
    model = _common_payload(
        authority.DGX_MODEL_LOAD_PHASE,
        previous_receipt_sha256="a" * 64,
        block_id=None,
        rerun_id=None,
    )
    model["measurements"] = {
        "startup_receipt_sha256": "a" * 64,
        "e0_base_snapshot_sha256": "b" * 64,
        "e0_processor_contract_sha256": "c" * 64,
        "trained_checkpoint_sha256": "d" * 64,
        "resolved_config_sha256": "e" * 64,
        "trained_processor_contract_sha256": "f" * 64,
        "p4_manifest_sha256": "1" * 64,
        "p4_index_sha256": "2" * 64,
        "p4_embeddings_sha256": "3" * 64,
        "loaded_weights_identity_sha256": "4" * 64,
        "model_process_identity_sha256": "5" * 64,
        "model_evaluation_mode": True,
        "memory_read_only": True,
        "independent_observation_method": authority.INDEPENDENT_OBSERVATION_METHOD,
    }
    block_open = _common_payload(
        authority.BLOCK_OPEN_PHASE,
        previous_receipt_sha256="b" * 64,
        block_id=BLOCK_ID,
        rerun_id=RERUN_ID,
    )
    block_open["measurements"] = {
        "model_load_receipt_sha256": "b" * 64,
        "schedule_row_sha256": "c" * 64,
        "task_id": "webarena-0",
        "repeat_id": 0,
        "matched_model_seed": 42,
        "execution_order": ["E2", "E0", "E3", "E1"],
        "model_call_count": 0,
        "transcript_entry_count": 0,
        "memory_manifest_sha256": "d" * 64,
        "memory_read_only": True,
        "evaluation_memory_writes_enabled": False,
        "independent_observation_method": authority.INDEPENDENT_OBSERVATION_METHOD,
    }
    runtime_entries = _transcript_entries("PAGE_BROKER")
    bridge_entries = _transcript_entries("DGX_BRIDGE")
    runtime_root = _transcript_root(runtime_entries, "PAGE_BROKER")
    bridge_root = _transcript_root(bridge_entries, "DGX_BRIDGE")
    block_close = _common_payload(
        authority.BLOCK_CLOSE_PHASE,
        previous_receipt_sha256="c" * 64,
        block_id=BLOCK_ID,
        rerun_id=RERUN_ID,
    )
    block_close["measurements"] = {
        "block_open_receipt_sha256": "c" * 64,
        "episode_ids": {system: f"episode-{system}" for system in ("E0", "E1", "E2", "E3")},
        "system_completion": {system: True for system in ("E0", "E1", "E2", "E3")},
        "local_broker_launch_receipt_sha256s": {
            system: str(index + 1) * 64
            for index, system in enumerate(("E0", "E1", "E2", "E3"))
        },
        "local_broker_cleanup_receipt_sha256s": {
            system: chr(ord("a") + index) * 64
            for index, system in enumerate(("E0", "E1", "E2", "E3"))
        },
        "runtime_transcript_root_sha256": runtime_root["root_sha256"],
        "runtime_transcript_entry_count": runtime_root["entry_count"],
        "bridge_transcript_root_sha256": bridge_root["root_sha256"],
        "bridge_transcript_entry_count": bridge_root["entry_count"],
        "transcript_chain_algorithm": authority.TRANSCRIPT_CHAIN_ALGORITHM,
        "independent_observation_method": authority.INDEPENDENT_OBSERVATION_METHOD,
    }
    return {
        authority.DGX_SERVICE_STARTUP_PHASE: startup,
        authority.DGX_MODEL_LOAD_PHASE: model,
        authority.BLOCK_OPEN_PHASE: block_open,
        authority.BLOCK_CLOSE_PHASE: block_close,
    }


def _transcript_entries(channel: str) -> list[dict]:
    first = authority.build_transcript_entry(
        campaign_id=CAMPAIGN_ID,
        block_id=BLOCK_ID,
        rerun_id=RERUN_ID,
        dgx_service_session_id=DGX_SERVICE_SESSION_ID,
        sequence=0,
        channel=channel,
        direction="REQUEST",
        operation="fixture_operation",
        value_sha256="1" * 64,
        previous_entry_sha256=None,
    )
    second = authority.build_transcript_entry(
        campaign_id=CAMPAIGN_ID,
        block_id=BLOCK_ID,
        rerun_id=RERUN_ID,
        dgx_service_session_id=DGX_SERVICE_SESSION_ID,
        sequence=1,
        channel=channel,
        direction="RESPONSE",
        operation="fixture_operation",
        value_sha256="2" * 64,
        previous_entry_sha256=first["entry_sha256"],
    )
    return [first, second]


def _transcript_root(entries: list[dict], channel: str) -> dict:
    return authority.reconstruct_transcript_root(
        entries,
        campaign_id=CAMPAIGN_ID,
        block_id=BLOCK_ID,
        rerun_id=RERUN_ID,
        dgx_service_session_id=DGX_SERVICE_SESSION_ID,
        expected_channel=channel,
    )


def _issue_startup(monkeypatch: pytest.MonkeyPatch, ledger: Path) -> dict:
    monkeypatch.setattr(authority, "_utc_now", lambda: FIXED_NOW)
    monkeypatch.setattr(authority.secrets, "token_hex", lambda size: "4" * 64)
    issued = authority._issue_challenge_against_registry(
        ledger,
        registry_value=_registry(),
        authority_id="independent-lab-a",
        key_id="lab-key-2026-01",
        phase=authority.DGX_SERVICE_STARTUP_PHASE,
        campaign_id=CAMPAIGN_ID,
        handoff_manifest_sha256="1" * 64,
        runner_attestation_sha256="2" * 64,
        semantic_dependency_lock_sha256="3" * 64,
    )
    assert issued["challenge"] == _startup_challenge()
    return issued


def test_production_registry_is_exact_empty_source_pin() -> None:
    registry = authority.load_production_authority_registry()
    assert registry == {
        "schema_version": authority.AUTHORITY_REGISTRY_SCHEMA_VERSION,
        "registry_id": authority.AUTHORITY_REGISTRY_ID,
        "status": authority.EMPTY_REGISTRY_STATUS,
        "authorities": [],
    }
    resource = authority.production_authority_registry_resource()
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == (
        authority.PRODUCTION_AUTHORITY_REGISTRY_FILE_SHA256
    )
    assert not (
        ROOT / "configs/eval/table2/deployment_authority_registry_v1.json"
    ).exists()
    assert (
        ROOT
        / "src/web_agent/eval/table2/deployment_authority_registry_v1.json"
    ).is_file()


def test_empty_production_registry_rejects_issue_and_signed_receipt(
    tmp_path: Path,
) -> None:
    with pytest.raises(SchemaError, match="not registered"):
        authority.issue_production_challenge(
            tmp_path / "authority.jsonl",
            authority_id="independent-lab-a",
            key_id="lab-key-2026-01",
            phase=authority.DGX_SERVICE_STARTUP_PHASE,
            campaign_id="table2-pc01-pilot-fixture",
            handoff_manifest_sha256="1" * 64,
            runner_attestation_sha256="2" * 64,
            semantic_dependency_lock_sha256="3" * 64,
        )
    assert not (tmp_path / "authority.jsonl").exists()
    with pytest.raises(SchemaError):
        authority.inspect_production_receipt(
            _startup_envelope(),
            challenge_value=_startup_challenge(),
            active_at=VERIFY_NOW,
        )


def test_precomputed_ed25519_signature_verifies_canonical_payload() -> None:
    verified = authority._verify_receipt_against_registry(
        _startup_envelope(),
        registry_value=_registry(),
        challenge_value=_startup_challenge(),
        active_at=VERIFY_NOW,
    )
    assert verified.phase == authority.DGX_SERVICE_STARTUP_PHASE
    assert verified.receipt_sha256 == sha256_json(_startup_envelope())
    assert verified.to_dict()["claim_scope"].endswith("NOT_DISPATCH_AUTHORITY")
    assert verified.to_dict()["external_global_replay_anchor_present"] is False
    assert verified.to_dict()["production_dispatch_authorized"] is False


def test_signature_and_payload_mutations_fail_closed() -> None:
    envelope = _startup_envelope()
    signature = bytearray(base64.b64decode(envelope["signature_base64"]))
    signature[0] ^= 1
    envelope["signature_base64"] = base64.b64encode(signature).decode("ascii")
    with pytest.raises(SchemaError, match="signature verification failed"):
        authority._verify_receipt_against_registry(
            envelope,
            registry_value=_registry(),
            challenge_value=_startup_challenge(),
            active_at=VERIFY_NOW,
        )

    envelope = _startup_envelope()
    envelope["payload"]["dgx_host_identity_sha256"] = "a" * 64
    envelope["payload_sha256"] = sha256_json(envelope["payload"])
    with pytest.raises(SchemaError, match="signature verification failed"):
        authority._verify_receipt_against_registry(
            envelope,
            registry_value=_registry(),
            challenge_value=_startup_challenge(),
            active_at=VERIFY_NOW,
        )


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda row: row.update(public_key_base64=row["public_key_base64"] + " "), "base64"),
        (
            lambda row: row.update(
                public_key_base64=base64.b64encode(b"short").decode("ascii")
            ),
            "wrong decoded length",
        ),
        (lambda row: row.update(public_key_sha256="0" * 64), "fingerprint"),
    ],
)
def test_registry_rejects_malformed_public_keys(mutation, match: str) -> None:
    registry = _registry()
    mutation(registry["authorities"][0])
    with pytest.raises(SchemaError, match=match):
        authority.validate_authority_registry(registry)


def test_signature_base64_is_strict_and_exact_length() -> None:
    for value in (
        VALID_SIGNATURE_BASE64.rstrip("="),
        base64.b64encode(b"short").decode("ascii"),
        VALID_SIGNATURE_BASE64 + " ",
    ):
        envelope = _startup_envelope()
        envelope["signature_base64"] = value
        with pytest.raises(SchemaError, match="signature_base64"):
            authority._verify_receipt_against_registry(
                envelope,
                registry_value=_registry(),
                challenge_value=_startup_challenge(),
                active_at=VERIFY_NOW,
            )


def test_registry_permission_validity_revocation_and_profile_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(authority, "_utc_now", lambda: FIXED_NOW)
    cases = []
    revoked = _registry()
    revoked["authorities"][0]["revoked"] = True
    cases.append((revoked, "revoked"))
    expired = _registry()
    expired["authorities"][0]["valid_until_utc"] = "2026-08-01T00:00:00Z"
    cases.append((expired, "validity"))
    wrong_role = _registry()
    wrong_role["authorities"][0]["allowed_roles"] = [
        authority.RUNTIME_VALUE_PROVENANCE_ROLE
    ]
    cases.append((wrong_role, "not permitted for this role"))
    wrong_phase = _registry()
    wrong_phase["authorities"][0]["allowed_phases"] = sorted(
        phase
        for phase in authority.AUTHORITY_PHASES
        if phase != authority.DGX_SERVICE_STARTUP_PHASE
    )
    cases.append((wrong_phase, "not permitted for this phase"))
    for index, (registry, match) in enumerate(cases):
        with pytest.raises(SchemaError, match=match):
            authority._issue_challenge_against_registry(
                tmp_path / f"authority-{index}.jsonl",
                registry_value=registry,
                authority_id="independent-lab-a",
                key_id="lab-key-2026-01",
                phase=authority.DGX_SERVICE_STARTUP_PHASE,
                campaign_id="table2-pc01-pilot-fixture",
                handoff_manifest_sha256="1" * 64,
                runner_attestation_sha256="2" * 64,
                semantic_dependency_lock_sha256="3" * 64,
            )

    challenge = _startup_challenge()
    payload = _startup_payload()
    challenge["measurement_profile_id"] = "unregistered-profile-v2"
    payload["measurement_profile_id"] = "unregistered-profile-v2"
    payload["challenge_sha256"] = sha256_json(challenge)
    envelope = _startup_envelope()
    envelope["payload"] = payload
    envelope["payload_sha256"] = sha256_json(payload)
    with pytest.raises(SchemaError, match="measurement profile"):
        authority._verify_receipt_against_registry(
            envelope,
            registry_value=_registry(),
            challenge_value=challenge,
            active_at=VERIFY_NOW,
        )


def test_duplicate_json_keys_are_rejected_at_every_depth(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="duplicate JSON key"):
        authority.loads_strict_json('{"outer":{"value":1,"value":2}}')
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value":1,"value":2}\n', encoding="utf-8")
    with pytest.raises(SchemaError, match="duplicate JSON key"):
        authority.read_strict_json(duplicate)


@pytest.mark.parametrize("value", ["1e999", "-1e999"])
def test_strict_json_rejects_exponent_overflow(value: str) -> None:
    with pytest.raises(SchemaError, match="non-finite JSON number"):
        authority.loads_strict_json(value)


def test_strict_json_rejects_registered_nesting_overflow(tmp_path: Path) -> None:
    deep = "[" * (authority.MAX_AUTHORITY_JSON_NESTING + 1) + "0" + "]" * (
        authority.MAX_AUTHORITY_JSON_NESTING + 1
    )
    with pytest.raises(SchemaError, match="nesting bound"):
        authority.loads_strict_json(deep)

    source = tmp_path / "deep.json"
    source.write_text(deep, encoding="utf-8")
    with pytest.raises(SchemaError, match="nesting bound"):
        authority.read_strict_json(source)

    transcript = tmp_path / "deep.jsonl"
    transcript.write_text('{"nested":' + deep + "}\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="nesting bound"):
        authority.read_runtime_transcript_jsonl(transcript)

    cli = ROOT / "scripts/validate_table2_deployment_authority.py"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(cli),
            "reconstruct-transcript",
            "--transcript-jsonl",
            str(transcript),
            "--campaign-id",
            CAMPAIGN_ID,
            "--block-id",
            BLOCK_ID,
            "--rerun-id",
            str(RERUN_ID),
            "--dgx-service-session-id",
            DGX_SERVICE_SESSION_ID,
            "--channel",
            "PAGE_BROKER",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "nesting bound" in json.loads(completed.stdout)["error"]


def test_strict_json_reader_enforces_byte_bound_and_no_symlink_ancestry(
    tmp_path: Path,
) -> None:
    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as handle:
        handle.truncate(authority.MAX_AUTHORITY_JSON_BYTES + 1)
    with pytest.raises(SchemaError, match="byte bound"):
        authority.read_strict_json(oversized)

    real_directory = tmp_path / "real"
    real_directory.mkdir()
    (real_directory / "receipt.json").write_text("{}\n", encoding="utf-8")
    linked_directory = tmp_path / "linked"
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(SchemaError, match="securely resolve"):
        authority.read_strict_json(linked_directory / "receipt.json")


def test_special_files_are_rejected_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "untrusted.fifo"
    os.mkfifo(fifo)
    code = (
        "from web_agent.eval.table2.deployment_authority import read_strict_json;"
        "from web_agent.eval.table2.common import SchemaError;"
        f"p={str(fifo)!r};"
        "\ntry: read_strict_json(p)\n"
        "except SchemaError: raise SystemExit(0)\n"
        "raise SystemExit(2)\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        timeout=2,
        check=False,
    )
    assert completed.returncode == 0


def test_failed_path_resolution_and_fdopen_do_not_leak_descriptors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    descriptor_directory = Path("/proc/self/fd")
    baseline = len(list(descriptor_directory.iterdir()))
    missing = tmp_path / "missing" / "nested" / "receipt.json"
    for _ in range(50):
        with pytest.raises(SchemaError, match="securely resolve"):
            authority.read_strict_json(missing)
    assert len(list(descriptor_directory.iterdir())) == baseline

    source = tmp_path / "receipt.json"
    source.write_text("{}\n", encoding="utf-8")
    real_fdopen = authority.os.fdopen

    def failed_fdopen(*args, **kwargs):
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(authority.os, "fdopen", failed_fdopen)
    for _ in range(50):
        with pytest.raises(OSError, match="injected"):
            authority.read_strict_json(source)
    monkeypatch.setattr(authority.os, "fdopen", real_fdopen)
    assert len(list(descriptor_directory.iterdir())) == baseline


def test_strict_json_reader_detects_ancestor_swap_during_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    safe = tmp_path / "safe"
    evil = tmp_path / "evil"
    moved = tmp_path / "moved"
    safe.mkdir()
    evil.mkdir()
    (safe / "receipt.json").write_text('{"source":"safe"}\n', encoding="utf-8")
    (evil / "receipt.json").write_text('{"source":"evil"}\n', encoding="utf-8")
    real_open = authority.os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "receipt.json" and dir_fd is not None and not swapped:
            swapped = True
            safe.rename(moved)
            safe.symlink_to(evil, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(authority.os, "open", swapping_open)
    with pytest.raises(SchemaError, match="ancestry|securely resolve"):
        authority.read_strict_json(safe / "receipt.json")


def test_nonfilesystem_registry_reader_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OversizedResource:
        def open(self, mode: str):
            assert mode == "rb"
            return io.BytesIO(b"x" * (authority.MAX_AUTHORITY_JSON_BYTES + 1))

    monkeypatch.setattr(
        authority,
        "production_authority_registry_resource",
        lambda: OversizedResource(),
    )
    with pytest.raises(SchemaError, match="byte bound"):
        authority.load_production_authority_registry()


def test_runtime_transcript_reader_requires_bounded_canonical_lf_jsonl(
    tmp_path: Path,
) -> None:
    entry = _transcript_entries("PAGE_BROKER")[0]
    canonical = canonical_json_bytes(entry)
    transcript = tmp_path / "runtime.jsonl"
    transcript.write_bytes(canonical + b"\n")
    assert authority.read_runtime_transcript_jsonl(transcript) == [entry]

    for malformed in (canonical, canonical + b"\r\n", b" " + canonical + b"\n"):
        transcript.write_bytes(malformed)
        with pytest.raises(SchemaError, match="LF|canonical"):
            authority.read_runtime_transcript_jsonl(transcript)

    with transcript.open("wb") as handle:
        handle.truncate(authority.MAX_RUNTIME_TRANSCRIPT_BYTES + 1)
    with pytest.raises(SchemaError, match="byte bound"):
        authority.read_runtime_transcript_jsonl(transcript)


def test_all_four_phase_payload_schemas_are_exact_and_cross_bound() -> None:
    payloads = _phase_payloads()
    assert set(payloads) == set(authority.AUTHORITY_PHASES)
    for phase, payload in payloads.items():
        assert authority.validate_receipt_payload(payload)["phase"] == phase
        extra = deepcopy(payload)
        extra["unregistered"] = True
        with pytest.raises(SchemaError, match="fields differ"):
            authority.validate_receipt_payload(extra)

    model = deepcopy(payloads[authority.DGX_MODEL_LOAD_PHASE])
    model["measurements"]["startup_receipt_sha256"] = "0" * 64
    with pytest.raises(SchemaError, match="prior receipt"):
        authority.validate_receipt_payload(model)
    block_open = deepcopy(payloads[authority.BLOCK_OPEN_PHASE])
    block_open["measurements"]["model_call_count"] = False
    with pytest.raises(SchemaError, match="integer"):
        authority.validate_receipt_payload(block_open)


def test_signed_transcript_chain_comparison_is_context_and_channel_bound() -> None:
    runtime = _transcript_entries("PAGE_BROKER")
    bridge = _transcript_entries("DGX_BRIDGE")
    comparison = authority._verify_block_close_transcript_roots_against_registry(
        _block_close_envelope(),
        registry_value=_registry(),
        challenge_value=_block_close_challenge(),
        runtime_entries=runtime,
        bridge_entries=bridge,
        active_at=VERIFY_NOW,
    )
    assert comparison["cryptographic_receipt_verified"] is True
    assert comparison["local_roots_equal_authenticated_claim"] is True
    assert comparison["claim_scope"].endswith("NOT_VALUE_ORIGIN_OR_DISPATCH_AUTHORITY")
    assert comparison["external_global_replay_anchor_present"] is False
    assert comparison["production_dispatch_authorized"] is False

    tampered = deepcopy(runtime)
    tampered[1]["value_sha256"] = "f" * 64
    with pytest.raises(SchemaError, match="entry hash differs"):
        _transcript_root(tampered, "PAGE_BROKER")

    with pytest.raises(SchemaError, match="signed envelope"):
        authority._verify_block_close_transcript_roots_against_registry(
            _phase_payloads()[authority.BLOCK_CLOSE_PHASE],
            registry_value=_registry(),
            challenge_value=_block_close_challenge(),
            runtime_entries=runtime,
            bridge_entries=bridge,
            active_at=VERIFY_NOW,
        )
    with pytest.raises(SchemaError, match="wrong channel chain"):
        authority._verify_block_close_transcript_roots_against_registry(
            _block_close_envelope(),
            registry_value=_registry(),
            challenge_value=_block_close_challenge(),
            runtime_entries=bridge,
            bridge_entries=runtime,
            active_at=VERIFY_NOW,
        )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("campaign_id", "different-campaign"),
        ("block_id", "different-block"),
        ("rerun_id", 1),
        ("dgx_service_session_id", "different-session"),
    ],
)
def test_transcript_root_rejects_mixed_context(field: str, wrong_value: object) -> None:
    context = {
        "campaign_id": CAMPAIGN_ID,
        "block_id": BLOCK_ID,
        "rerun_id": RERUN_ID,
        "dgx_service_session_id": DGX_SERVICE_SESSION_ID,
    }
    context[field] = wrong_value
    with pytest.raises(SchemaError, match=f"context differs: {field}"):
        authority.reconstruct_transcript_root(
            _transcript_entries("PAGE_BROKER"),
            expected_channel="PAGE_BROKER",
            **context,
        )


def test_empty_production_registry_rejects_signed_block_close_comparison() -> None:
    with pytest.raises(SchemaError, match="registry hash differs|not registered"):
        authority.verify_production_block_close_transcript_roots(
            _block_close_envelope(),
            challenge_value=_block_close_challenge(),
            runtime_entries=_transcript_entries("PAGE_BROKER"),
            bridge_entries=_transcript_entries("DGX_BRIDGE"),
            active_at=VERIFY_NOW,
        )


def test_challenge_ledger_consumes_once_and_rejects_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "authority.jsonl"
    issued = _issue_startup(monkeypatch, ledger)
    monkeypatch.setattr(authority, "_utc_now", lambda: VERIFY_NOW)
    verified = authority._consume_receipt_against_registry(
        ledger,
        registry_value=_registry(),
        challenge_sha256=issued["challenge_sha256"],
        envelope_value=_startup_envelope(),
    )
    assert verified.receipt_sha256 == sha256_json(_startup_envelope())
    events = authority.read_authority_ledger(ledger)
    assert [event["event_type"] for event in events] == [
        authority.CHALLENGE_ISSUED_EVENT,
        authority.RECEIPT_CONSUMED_EVENT,
    ]
    assert all(
        event["claim_scope"] == authority.LOCAL_LEDGER_CLAIM_SCOPE
        for event in events
    )
    consumed = events[1]["payload"]
    assert consumed["challenge"] == _startup_challenge()
    assert consumed["envelope"] == _startup_envelope()
    assert consumed["dgx_service_session_id"] == DGX_SERVICE_SESSION_ID
    assert consumed["dgx_host_identity_sha256"] == "5" * 64
    assert consumed["dgx_runtime_identity_sha256"] == "6" * 64
    assert consumed["bridge_identity_sha256"] == "8" * 64
    revalidated = authority._validate_authority_ledger_against_registry(
        ledger,
        registry_value=_registry(),
    )
    assert revalidated["claim_scope"] == authority.LOCAL_LEDGER_CLAIM_SCOPE
    assert revalidated["consumed_receipt_count"] == 1
    assert revalidated["external_global_replay_anchor_present"] is False
    assert revalidated["production_dispatch_authorized"] is False
    with pytest.raises(SchemaError, match="already been consumed"):
        authority._consume_receipt_against_registry(
            ledger,
            registry_value=_registry(),
            challenge_sha256=issued["challenge_sha256"],
            envelope_value=_startup_envelope(),
        )


@pytest.mark.parametrize(
    "field",
    [
        "handoff_manifest_sha256",
        "runner_attestation_sha256",
        "semantic_dependency_lock_sha256",
        "dgx_service_session_id",
        "dgx_host_identity_sha256",
        "dgx_runtime_identity_sha256",
        "bridge_identity_sha256",
    ],
)
@pytest.mark.parametrize(
    ("predecessor_phase", "phase"),
    [
        (
            authority.DGX_SERVICE_STARTUP_PHASE,
            authority.DGX_MODEL_LOAD_PHASE,
        ),
        (authority.DGX_MODEL_LOAD_PHASE, authority.BLOCK_OPEN_PHASE),
        (authority.BLOCK_OPEN_PHASE, authority.BLOCK_CLOSE_PHASE),
    ],
)
def test_phase_identity_continuity_rejects_every_identity_change(
    field: str, predecessor_phase: str, phase: str
) -> None:
    payloads = _phase_payloads()
    predecessor = payloads[predecessor_phase]
    current = payloads[phase]
    authority._validate_identity_continuity([predecessor], current)
    changed = deepcopy(current)
    changed[field] = "different-session" if field.endswith("session_id") else "0" * 64
    with pytest.raises(SchemaError, match=f"continuity differs: {field}"):
        authority._validate_identity_continuity([predecessor], changed)


def test_identity_continuity_allows_next_physical_block_progression() -> None:
    payloads = _phase_payloads()
    predecessor = payloads[authority.BLOCK_CLOSE_PHASE]
    next_block = deepcopy(payloads[authority.BLOCK_OPEN_PHASE])
    next_block["block_id"] = "block-fixture-1"
    next_block["rerun_id"] = 1

    authority._validate_identity_continuity([predecessor], next_block)


@pytest.mark.parametrize(
    "field",
    [
        "handoff_manifest_sha256",
        "runner_attestation_sha256",
        "semantic_dependency_lock_sha256",
    ],
)
def test_ledger_replay_rejects_resigned_deployment_identity_drift(
    field: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "authority.jsonl"
    startup_issued = _issue_startup(monkeypatch, ledger)
    monkeypatch.setattr(authority, "_utc_now", lambda: VERIFY_NOW)
    startup_envelope = _startup_envelope()
    authority._consume_receipt_against_registry(
        ledger,
        registry_value=_registry(),
        challenge_sha256=startup_issued["challenge_sha256"],
        envelope_value=startup_envelope,
    )

    model_issue_time = VERIFY_NOW + timedelta(seconds=1)
    monkeypatch.setattr(authority, "_utc_now", lambda: model_issue_time)
    monkeypatch.setattr(authority.secrets, "token_hex", lambda size: "5" * 64)
    model_issued = authority._issue_challenge_against_registry(
        ledger,
        registry_value=_registry(),
        authority_id="independent-lab-a",
        key_id="lab-key-2026-01",
        phase=authority.DGX_MODEL_LOAD_PHASE,
        campaign_id=CAMPAIGN_ID,
        handoff_manifest_sha256="1" * 64,
        runner_attestation_sha256="2" * 64,
        semantic_dependency_lock_sha256="3" * 64,
        previous_receipt_sha256=sha256_json(startup_envelope),
    )
    model_payload = _payload_for_issued_challenge(
        model_issued["challenge"], authority_sequence=2
    )
    model_envelope = _signed_envelope(model_payload)
    monkeypatch.setattr(
        authority, "_utc_now", lambda: model_issue_time + timedelta(seconds=1)
    )
    authority._consume_receipt_against_registry(
        ledger,
        registry_value=_registry(),
        challenge_sha256=model_issued["challenge_sha256"],
        envelope_value=model_envelope,
    )
    assert authority._validate_authority_ledger_against_registry(
        ledger, registry_value=_registry()
    )["consumed_receipt_count"] == 2

    events = authority.read_authority_ledger(ledger)
    drifted_value = "0" * 64
    issued_payload = events[2]["payload"]
    issued_payload["challenge"][field] = drifted_value
    issued_payload["challenge_sha256"] = sha256_json(issued_payload["challenge"])
    unsigned_issued = dict(events[2])
    unsigned_issued.pop("event_sha256")
    events[2]["event_sha256"] = sha256_json(unsigned_issued)

    consumed_payload = events[3]["payload"]
    consumed_payload["challenge"] = deepcopy(issued_payload["challenge"])
    consumed_payload["challenge_sha256"] = issued_payload["challenge_sha256"]
    signed_payload = consumed_payload["envelope"]["payload"]
    signed_payload[field] = drifted_value
    signed_payload["challenge_sha256"] = issued_payload["challenge_sha256"]
    consumed_payload["envelope"] = _signed_envelope(signed_payload)
    consumed_payload["payload_sha256"] = consumed_payload["envelope"][
        "payload_sha256"
    ]
    consumed_payload["receipt_sha256"] = sha256_json(
        consumed_payload["envelope"]
    )
    consumed_payload[field] = drifted_value
    events[3]["previous_event_sha256"] = events[2]["event_sha256"]
    unsigned_consumed = dict(events[3])
    unsigned_consumed.pop("event_sha256")
    events[3]["event_sha256"] = sha256_json(unsigned_consumed)

    authority._verify_receipt_against_registry(
        consumed_payload["envelope"],
        registry_value=_registry(),
        challenge_value=consumed_payload["challenge"],
        active_at=model_issue_time,
    )
    ledger.write_text(
        "".join(
            canonical_json_bytes(event).decode("utf-8") + "\n"
            for event in events
        ),
        encoding="utf-8",
    )

    with pytest.raises(SchemaError, match=f"continuity differs: {field}"):
        authority.read_authority_ledger(ledger)
    with pytest.raises(SchemaError, match=f"continuity differs: {field}"):
        authority._validate_authority_ledger_against_registry(
            ledger, registry_value=_registry()
        )


def test_ledger_revalidation_detects_rehashed_local_signature_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "authority.jsonl"
    issued = _issue_startup(monkeypatch, ledger)
    monkeypatch.setattr(authority, "_utc_now", lambda: VERIFY_NOW)
    authority._consume_receipt_against_registry(
        ledger,
        registry_value=_registry(),
        challenge_sha256=issued["challenge_sha256"],
        envelope_value=_startup_envelope(),
    )
    events = authority.read_authority_ledger(ledger)
    consumed = events[1]["payload"]
    signed_payload = consumed["envelope"]["payload"]
    signed_payload["dgx_host_identity_sha256"] = "0" * 64
    consumed["envelope"]["payload_sha256"] = sha256_json(signed_payload)
    consumed["payload_sha256"] = consumed["envelope"]["payload_sha256"]
    consumed["receipt_sha256"] = sha256_json(consumed["envelope"])
    consumed["dgx_host_identity_sha256"] = "0" * 64
    unsigned_event = dict(events[1])
    unsigned_event.pop("event_sha256")
    events[1]["event_sha256"] = sha256_json(unsigned_event)
    ledger.write_text(
        "".join(
            canonical_json_bytes(event).decode("utf-8") + "\n"
            for event in events
        ),
        encoding="utf-8",
    )

    # The local structure was coherently rehashed, but the preserved independent
    # signature still exposes the mutation when the evidence is replay-verified.
    assert len(authority.read_authority_ledger(ledger)) == 2
    before = ledger.read_bytes()
    with pytest.raises(SchemaError, match="signature verification failed"):
        authority._issue_challenge_against_registry(
            ledger,
            registry_value=_registry(),
            authority_id="independent-lab-a",
            key_id="lab-key-2026-01",
            phase=authority.DGX_MODEL_LOAD_PHASE,
            campaign_id=CAMPAIGN_ID,
            handoff_manifest_sha256="1" * 64,
            runner_attestation_sha256="2" * 64,
            semantic_dependency_lock_sha256="3" * 64,
            previous_receipt_sha256=consumed["receipt_sha256"],
        )
    assert ledger.read_bytes() == before
    with pytest.raises(SchemaError, match="signature verification failed"):
        authority._consume_receipt_against_registry(
            ledger,
            registry_value=_registry(),
            challenge_sha256=issued["challenge_sha256"],
            envelope_value=_startup_envelope(),
        )
    assert ledger.read_bytes() == before
    with pytest.raises(SchemaError, match="signature verification failed"):
        authority._validate_authority_ledger_against_registry(
            ledger,
            registry_value=_registry(),
        )


def test_issue_clock_begins_only_after_ledger_lock_acquisition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_locked_ledger = authority._locked_ledger
    lock_state = {"held": False}
    lock_time = datetime(2026, 9, 5, 0, 4, 0, tzinfo=timezone.utc)

    @contextmanager
    def delayed_lock(path, *, allow_create):
        with real_locked_ledger(path, allow_create=allow_create) as locked:
            lock_state["held"] = True
            try:
                yield locked
            finally:
                lock_state["held"] = False

    monkeypatch.setattr(authority, "_locked_ledger", delayed_lock)
    monkeypatch.setattr(
        authority,
        "_utc_now",
        lambda: lock_time if lock_state["held"] else FIXED_NOW,
    )
    monkeypatch.setattr(authority.secrets, "token_hex", lambda size: "4" * 64)
    ledger = tmp_path / "authority.jsonl"
    issued = authority._issue_challenge_against_registry(
        ledger,
        registry_value=_registry(),
        authority_id="independent-lab-a",
        key_id="lab-key-2026-01",
        phase=authority.DGX_SERVICE_STARTUP_PHASE,
        campaign_id=CAMPAIGN_ID,
        handoff_manifest_sha256="1" * 64,
        runner_attestation_sha256="2" * 64,
        semantic_dependency_lock_sha256="3" * 64,
        ttl_seconds=60,
    )
    assert issued["challenge"]["issued_at_utc"] == "2026-09-05T00:04:00Z"
    assert issued["challenge"]["expires_at_utc"] == "2026-09-05T00:05:00Z"
    assert authority.read_authority_ledger(ledger)[0]["recorded_at_utc"] == (
        "2026-09-05T00:04:00Z"
    )


def test_consume_clock_is_captured_while_ledger_lock_is_held(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "authority.jsonl"
    issued = _issue_startup(monkeypatch, ledger)
    real_locked_ledger = authority._locked_ledger
    lock_state = {"held": False}

    @contextmanager
    def observed_lock(path, *, allow_create):
        with real_locked_ledger(path, allow_create=allow_create) as locked:
            lock_state["held"] = True
            try:
                yield locked
            finally:
                lock_state["held"] = False

    def locked_clock() -> datetime:
        assert lock_state["held"] is True
        return VERIFY_NOW

    monkeypatch.setattr(authority, "_locked_ledger", observed_lock)
    monkeypatch.setattr(authority, "_utc_now", locked_clock)
    verified = authority._consume_receipt_against_registry(
        ledger,
        registry_value=_registry(),
        challenge_sha256=issued["challenge_sha256"],
        envelope_value=_startup_envelope(),
    )
    assert verified.receipt_sha256 == sha256_json(_startup_envelope())


def test_issue_permission_uses_post_contention_clock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_locked_ledger = authority._locked_ledger
    lock_state = {"held": False}
    lock_time = datetime(2026, 9, 5, 0, 4, 0, tzinfo=timezone.utc)

    @contextmanager
    def delayed_lock(path, *, allow_create):
        with real_locked_ledger(path, allow_create=allow_create) as locked:
            lock_state["held"] = True
            try:
                yield locked
            finally:
                lock_state["held"] = False

    monkeypatch.setattr(authority, "_locked_ledger", delayed_lock)
    monkeypatch.setattr(
        authority,
        "_utc_now",
        lambda: lock_time if lock_state["held"] else FIXED_NOW,
    )
    expired_during_wait = _registry()
    expired_during_wait["authorities"][0]["valid_until_utc"] = (
        "2026-09-05T00:02:00Z"
    )
    ledger = tmp_path / "authority.jsonl"
    with pytest.raises(SchemaError, match="validity window"):
        authority._issue_challenge_against_registry(
            ledger,
            registry_value=expired_during_wait,
            authority_id="independent-lab-a",
            key_id="lab-key-2026-01",
            phase=authority.DGX_SERVICE_STARTUP_PHASE,
            campaign_id=CAMPAIGN_ID,
            handoff_manifest_sha256="1" * 64,
            runner_attestation_sha256="2" * 64,
            semantic_dependency_lock_sha256="3" * 64,
            ttl_seconds=60,
        )
    assert ledger.read_bytes() == b""


def test_copied_ledger_is_explicitly_not_a_global_replay_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = tmp_path / "authority.jsonl"
    issued = _issue_startup(monkeypatch, original)
    monkeypatch.setattr(authority, "_utc_now", lambda: VERIFY_NOW)
    authority._consume_receipt_against_registry(
        original,
        registry_value=_registry(),
        challenge_sha256=issued["challenge_sha256"],
        envelope_value=_startup_envelope(),
    )
    copied = tmp_path / "copied-authority.jsonl"
    copied.write_bytes(original.read_bytes())
    copied.chmod(0o600)
    original_result = authority._validate_authority_ledger_against_registry(
        original,
        registry_value=_registry(),
    )
    copied_result = authority._validate_authority_ledger_against_registry(
        copied,
        registry_value=_registry(),
    )
    assert original_result == copied_result
    assert copied_result["claim_scope"] == (
        "LOCAL_PER_LEDGER_SERIALIZATION_NOT_DISPATCH_AUTHORITY"
    )
    assert copied_result["external_global_replay_anchor_present"] is False
    assert copied_result["production_dispatch_authorized"] is False


def test_challenge_chain_requires_consumed_predecessor_and_fresh_nonce(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "authority.jsonl"
    issued = _issue_startup(monkeypatch, ledger)
    with pytest.raises(SchemaError, match="nonce collision"):
        authority._issue_challenge_against_registry(
            ledger,
            registry_value=_registry(),
            authority_id="independent-lab-a",
            key_id="lab-key-2026-01",
            phase=authority.DGX_SERVICE_STARTUP_PHASE,
            campaign_id="second-campaign",
            handoff_manifest_sha256="1" * 64,
            runner_attestation_sha256="2" * 64,
            semantic_dependency_lock_sha256="3" * 64,
        )
    monkeypatch.setattr(authority, "_utc_now", lambda: VERIFY_NOW)
    authority._consume_receipt_against_registry(
        ledger,
        registry_value=_registry(),
        challenge_sha256=issued["challenge_sha256"],
        envelope_value=_startup_envelope(),
    )
    nonce_values = iter(["5" * 64, "6" * 64])
    monkeypatch.setattr(authority.secrets, "token_hex", lambda size: next(nonce_values))
    model = authority._issue_challenge_against_registry(
        ledger,
        registry_value=_registry(),
        authority_id="independent-lab-a",
        key_id="lab-key-2026-01",
        phase=authority.DGX_MODEL_LOAD_PHASE,
        campaign_id="table2-pc01-pilot-fixture",
        handoff_manifest_sha256="1" * 64,
        runner_attestation_sha256="2" * 64,
        semantic_dependency_lock_sha256="3" * 64,
        previous_receipt_sha256=sha256_json(_startup_envelope()),
    )
    assert model["challenge"]["previous_challenge_sha256"] == (
        issued["challenge_sha256"]
    )
    with pytest.raises(SchemaError, match="block-open challenge must follow"):
        authority._issue_challenge_against_registry(
            ledger,
            registry_value=_registry(),
            authority_id="independent-lab-a",
            key_id="lab-key-2026-01",
            phase=authority.BLOCK_OPEN_PHASE,
            campaign_id="table2-pc01-pilot-fixture",
            handoff_manifest_sha256="1" * 64,
            runner_attestation_sha256="2" * 64,
            semantic_dependency_lock_sha256="3" * 64,
            block_id="block-fixture-0",
            rerun_id=0,
            previous_receipt_sha256=sha256_json(_startup_envelope()),
        )


def test_expired_challenge_and_future_receipt_are_rejected() -> None:
    with pytest.raises(SchemaError, match="not active"):
        authority._verify_receipt_against_registry(
            _startup_envelope(),
            registry_value=_registry(),
            challenge_value=_startup_challenge(),
            active_at=datetime(2026, 9, 5, 0, 6, tzinfo=timezone.utc),
        )
    with pytest.raises(SchemaError, match="future"):
        authority._verify_receipt_against_registry(
            _startup_envelope(),
            registry_value=_registry(),
            challenge_value=_startup_challenge(),
            active_at=datetime(2026, 9, 5, 0, 0, 1, tzinfo=timezone.utc),
        )


def test_ledger_rejects_tampering_symlinks_and_hardlinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "authority.jsonl"
    _issue_startup(monkeypatch, ledger)
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(text.replace('"sequence":0', '"sequence":1'), encoding="utf-8")
    with pytest.raises(SchemaError):
        authority.read_authority_ledger(ledger)

    target = tmp_path / "target.jsonl"
    target.write_text("", encoding="utf-8")
    symlink = tmp_path / "symlink.jsonl"
    symlink.symlink_to(target)
    with pytest.raises(SchemaError, match="symlink"):
        authority.read_authority_ledger(symlink)
    hardlink = tmp_path / "hardlink.jsonl"
    hardlink.hardlink_to(target)
    with pytest.raises(SchemaError, match="hard-linked"):
        authority.read_authority_ledger(hardlink)


def test_ledger_rejects_non_lf_framing_before_another_append(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "authority.jsonl"
    _issue_startup(monkeypatch, ledger)
    canonical = ledger.read_bytes()

    ledger.write_bytes(canonical.removesuffix(b"\n"))
    with pytest.raises(SchemaError, match="end with LF"):
        authority.read_authority_ledger(ledger)

    ledger.write_bytes(canonical.replace(b"\n", b"\r\n"))
    with pytest.raises(SchemaError, match="LF rather than CRLF"):
        authority.read_authority_ledger(ledger)


def test_ledger_enforces_byte_line_event_and_mode_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "authority.jsonl"
    _issue_startup(monkeypatch, ledger)

    monkeypatch.setattr(authority, "MAX_AUTHORITY_LEDGER_EVENTS", 0)
    with pytest.raises(SchemaError, match="event bound"):
        authority.read_authority_ledger(ledger)
    monkeypatch.undo()

    monkeypatch.setattr(authority, "MAX_AUTHORITY_JSONL_LINE_BYTES", 8)
    with pytest.raises(SchemaError, match="line.*bound"):
        authority.read_authority_ledger(ledger)
    monkeypatch.undo()

    ledger.chmod(0o620)
    with pytest.raises(SchemaError, match="group/world writable"):
        authority.read_authority_ledger(ledger)
    ledger.chmod(0o600)

    with ledger.open("wb") as handle:
        handle.truncate(authority.MAX_AUTHORITY_LEDGER_BYTES + 1)
    with pytest.raises(SchemaError, match="byte bound"):
        authority.read_authority_ledger(ledger)


def test_ledger_rejects_insecure_or_symlinked_parent(tmp_path: Path) -> None:
    insecure = tmp_path / "insecure"
    insecure.mkdir()
    insecure.chmod(0o777)
    with pytest.raises(SchemaError, match="owner-controlled"):
        with authority._locked_ledger(insecure / "ledger.jsonl", allow_create=True):
            pass

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(SchemaError, match="securely resolve"):
        with authority._locked_ledger(
            linked_parent / "ledger.jsonl", allow_create=True
        ):
            pass


def test_ledger_detects_leaf_replacement_while_locked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "authority.jsonl"
    _issue_startup(monkeypatch, ledger)
    displaced = tmp_path / "displaced.jsonl"
    real_parse = authority._parse_ledger_text

    def replacing_parse(payload):
        events = real_parse(payload)
        original = ledger.read_bytes()
        ledger.rename(displaced)
        ledger.write_bytes(original)
        ledger.chmod(0o600)
        return events

    monkeypatch.setattr(authority, "_parse_ledger_text", replacing_parse)
    with pytest.raises(SchemaError, match="path changed"):
        authority.read_authority_ledger(ledger)


def test_ledger_detects_parent_replacement_while_locked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent = tmp_path / "active"
    parent.mkdir()
    parent.chmod(0o700)
    ledger = parent / "authority.jsonl"
    _issue_startup(monkeypatch, ledger)
    displaced_parent = tmp_path / "displaced"
    real_parse = authority._parse_ledger_text

    def replacing_parent(payload):
        events = real_parse(payload)
        original = ledger.read_bytes()
        parent.rename(displaced_parent)
        parent.mkdir()
        ledger.write_bytes(original)
        ledger.chmod(0o600)
        return events

    monkeypatch.setattr(authority, "_parse_ledger_text", replacing_parent)
    with pytest.raises(SchemaError, match="parent changed"):
        authority.read_authority_ledger(ledger)


def test_ledger_append_rejects_short_write_and_restores_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "authority.jsonl"
    real_fdopen = authority.os.fdopen

    class ShortWriter:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def write(self, payload):
            return self.handle.write(payload[:-1])

    def short_fdopen(descriptor, mode, *args, **kwargs):
        handle = real_fdopen(descriptor, mode, *args, **kwargs)
        return ShortWriter(handle) if mode == "r+b" else handle

    monkeypatch.setattr(authority.os, "fdopen", short_fdopen)
    with pytest.raises(SchemaError, match="append was short"):
        _issue_startup(monkeypatch, ledger)
    assert ledger.read_bytes() == b""


def test_ledger_append_rejects_same_inode_corruption_after_fsync(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "authority.jsonl"
    _issue_startup(monkeypatch, ledger)
    monkeypatch.setattr(authority.secrets, "token_hex", lambda size: "5" * 64)
    real_fsync = authority.os.fsync
    corrupted = False

    def corrupting_fsync(descriptor):
        nonlocal corrupted
        real_fsync(descriptor)
        metadata = os.fstat(descriptor)
        if stat.S_ISREG(metadata.st_mode) and metadata.st_size and not corrupted:
            corrupted = True
            second = os.open(ledger, os.O_WRONLY)
            try:
                os.pwrite(second, b"X", 0)
                real_fsync(second)
            finally:
                os.close(second)

    monkeypatch.setattr(authority.os, "fsync", corrupting_fsync)
    with pytest.raises(SchemaError, match="final content differs"):
        authority._issue_challenge_against_registry(
            ledger,
            registry_value=_registry(),
            authority_id="independent-lab-a",
            key_id="lab-key-2026-01",
            phase=authority.DGX_SERVICE_STARTUP_PHASE,
            campaign_id="second-campaign",
            handoff_manifest_sha256="1" * 64,
            runner_attestation_sha256="2" * 64,
            semantic_dependency_lock_sha256="3" * 64,
        )
    assert corrupted is True


def test_ledger_context_rejects_same_inode_change_after_append_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "authority.jsonl"
    real_append = authority._append_ledger_event

    def append_then_change(*args, **kwargs):
        result = real_append(*args, **kwargs)
        second = os.open(ledger, os.O_WRONLY)
        try:
            os.pwrite(second, b"X", 0)
            os.fsync(second)
        finally:
            os.close(second)
        return result

    monkeypatch.setattr(authority, "_append_ledger_event", append_then_change)
    with pytest.raises(SchemaError, match="final content differs"):
        _issue_startup(monkeypatch, ledger)


def test_ledger_lock_wait_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "authority.jsonl"
    ledger.write_bytes(b"")
    ledger.chmod(0o600)
    descriptor = os.open(ledger, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(authority, "AUTHORITY_LEDGER_LOCK_TIMEOUT_SECONDS", 0.0)
        with pytest.raises(SchemaError, match="lock timed out"):
            authority.read_authority_ledger(ledger)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_cli_has_no_key_material_or_registry_override_and_stays_blocked(
    tmp_path: Path,
) -> None:
    cli = ROOT / "scripts/validate_table2_deployment_authority.py"
    source = cli.read_text(encoding="utf-8")
    module_source = Path(authority.__file__).read_text(encoding="utf-8")
    assert "Ed25519" + "PrivateKey" not in source + module_source
    assert "--public-key" not in source
    assert "--registry" not in source
    assert "sign-receipt" not in source
    assert "generate-key" not in source
    assert '"AUTHENTICATED_AND_CONSUMED"' not in source
    assert "matches_signed_payload" not in source + module_source
    assert not hasattr(authority, "validate_block_close_transcript_roots")
    completed = subprocess.run(
        [sys.executable, str(cli), "registry-status"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result["registered_authority_count"] == 0
    assert result["production_dispatch_authorized"] is False
    ledger = tmp_path / "authority.jsonl"
    blocked = subprocess.run(
        [
            sys.executable,
            str(cli),
            "issue-challenge",
            "--ledger",
            str(ledger),
            "--authority-id",
            "unregistered-lab",
            "--key-id",
            "unregistered-key",
            "--phase",
            authority.DGX_SERVICE_STARTUP_PHASE,
            "--campaign-id",
            "table2-pc01-pilot-fixture",
            "--handoff-manifest-sha256",
            "1" * 64,
            "--runner-attestation-sha256",
            "2" * 64,
            "--semantic-dependency-lock-sha256",
            "3" * 64,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert blocked.returncode == 1
    assert json.loads(blocked.stdout)["status"] == "FAIL"
    assert not ledger.exists()


def test_canonical_signature_input_is_domain_separated() -> None:
    payload = _startup_payload()
    signed_bytes = authority.SIGNATURE_DOMAIN + canonical_json_bytes(payload)
    assert signed_bytes.startswith(b"TABLE2_EXTERNAL_DEPLOYMENT_RECEIPT_V1\x00{")
    assert b" " not in canonical_json_bytes(payload)
