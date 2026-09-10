"""Independent deployment-receipt verification for Table 2.

This module deliberately contains no signing or key-generation capability. A
receipt can be authenticated only by an Ed25519 public key already present in
the fixed, source-pinned registry.  The production registry is initially empty,
so all production challenge issuance and receipt verification fail closed until
an independently controlled lab key is added in a reviewed source commit.

Local transcript chains, broker receipts, and the JSONL consumption ledger are
integrity evidence only. A copied, renamed, restored, or independently created
ledger can replay the same bytes; consequently this module makes only a
``LOCAL_PER_LEDGER_SERIALIZATION_NOT_DISPATCH_AUTHORITY`` claim. Global replay
exclusion requires an independently maintained authority nonce/sequence anchor.
Transcript roots become externally attributed evidence only when an allowed
authority signs the matching block-close payload after independently observing
the bytes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import base64
import binascii
import errno
import fcntl
import hashlib
from importlib.resources import files
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import time
from typing import Any, BinaryIO, Iterator

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .common import SchemaError, canonical_json_bytes, sha256_json


AUTHORITY_REGISTRY_SCHEMA_VERSION = "table2-deployment-authority-registry-v1"
AUTHORITY_REGISTRY_ID = "table2-independent-deployment-authorities-v1"
EMPTY_REGISTRY_STATUS = "NO_AUTHORITIES_REGISTERED"
ACTIVE_REGISTRY_STATUS = "ACTIVE_AUTHORITIES_REGISTERED"
PRODUCTION_AUTHORITY_REGISTRY_RESOURCE = "deployment_authority_registry_v1.json"
PRODUCTION_AUTHORITY_REGISTRY_FILE_SHA256 = (
    "638e5910eae541d169a2ff8b76776600690098af42e501802f04b17428298750"
)

SIGNED_ENVELOPE_SCHEMA_VERSION = "table2-external-deployment-envelope-v1"
SIGNED_ENVELOPE_RECORD_TYPE = "ExternalDeploymentAuthoritySignedEnvelope"
SIGNATURE_ALGORITHM = "Ed25519"
SIGNATURE_DOMAIN = b"TABLE2_EXTERNAL_DEPLOYMENT_RECEIPT_V1\x00"
CANONICAL_JSON_ALGORITHM = "UTF8_ASCII_VALUES_SORTED_KEYS_COMPACT_NO_NAN_V1"

CHALLENGE_SCHEMA_VERSION = "table2-deployment-authority-challenge-v1"
CHALLENGE_RECORD_TYPE = "DeploymentAuthorityChallenge"
LEDGER_EVENT_SCHEMA_VERSION = "table2-deployment-authority-ledger-event-v1"
LEDGER_EVENT_RECORD_TYPE = "DeploymentAuthorityLedgerEvent"
CHALLENGE_ISSUED_EVENT = "CHALLENGE_ISSUED"
RECEIPT_CONSUMED_EVENT = "RECEIPT_CONSUMED"
MAX_CHALLENGE_TTL_SECONDS = 600
LOCAL_LEDGER_CLAIM_SCOPE = "LOCAL_PER_LEDGER_SERIALIZATION_NOT_DISPATCH_AUTHORITY"

# Inputs crossing the filesystem boundary are deliberately bounded.  The values
# are large enough for a full Table 2 campaign while preventing malformed local
# evidence from causing unbounded allocation or replay work.
MAX_AUTHORITY_JSON_BYTES = 1_048_576
MAX_AUTHORITY_JSON_NESTING = 64
MAX_AUTHORITY_JSONL_LINE_BYTES = 1_048_576
MAX_AUTHORITY_LEDGER_BYTES = 67_108_864
MAX_AUTHORITY_LEDGER_EVENTS = 100_000
MAX_RUNTIME_TRANSCRIPT_BYTES = 67_108_864
MAX_RUNTIME_TRANSCRIPT_EVENTS = 100_000
AUTHORITY_LEDGER_LOCK_TIMEOUT_SECONDS = 30.0

SPLIT_DEPLOYMENT_TOPOLOGY = "SPLIT_LOCAL_BROWSER_DGX_INFERENCE"
DGX_DEPLOYMENT_ROLE = "DGX_DEPLOYMENT_ATTESTOR"
RUNTIME_VALUE_PROVENANCE_ROLE = "RUNTIME_VALUE_PROVENANCE_ATTESTOR"
INDEPENDENT_OBSERVATION_METHOD = (
    "INDEPENDENT_EXTERNAL_BRIDGE_AND_HOST_MEASUREMENT_V1"
)

DGX_SERVICE_STARTUP_PHASE = "DGX_INFERENCE_SERVICE_STARTUP"
DGX_MODEL_LOAD_PHASE = "DGX_MODEL_LOAD_COMPLETE"
BLOCK_OPEN_PHASE = "BEFORE_PHYSICAL_BLOCK"
BLOCK_CLOSE_PHASE = "AFTER_PHYSICAL_BLOCK"
AUTHORITY_PHASES = (
    DGX_SERVICE_STARTUP_PHASE,
    DGX_MODEL_LOAD_PHASE,
    BLOCK_OPEN_PHASE,
    BLOCK_CLOSE_PHASE,
)
PHASE_ROLES = {
    DGX_SERVICE_STARTUP_PHASE: DGX_DEPLOYMENT_ROLE,
    DGX_MODEL_LOAD_PHASE: DGX_DEPLOYMENT_ROLE,
    BLOCK_OPEN_PHASE: DGX_DEPLOYMENT_ROLE,
    BLOCK_CLOSE_PHASE: RUNTIME_VALUE_PROVENANCE_ROLE,
}

PAYLOAD_SCHEMA_BY_PHASE = {
    DGX_SERVICE_STARTUP_PHASE: "table2-dgx-service-startup-receipt-v1",
    DGX_MODEL_LOAD_PHASE: "table2-dgx-model-load-complete-receipt-v1",
    BLOCK_OPEN_PHASE: "table2-physical-block-open-receipt-v1",
    BLOCK_CLOSE_PHASE: "table2-physical-block-close-provenance-receipt-v1",
}
PAYLOAD_RECORD_TYPE_BY_PHASE = {
    DGX_SERVICE_STARTUP_PHASE: "DGXInferenceServiceStartupReceipt",
    DGX_MODEL_LOAD_PHASE: "DGXModelLoadCompleteReceipt",
    BLOCK_OPEN_PHASE: "PhysicalBlockOpenReceipt",
    BLOCK_CLOSE_PHASE: "PhysicalBlockCloseValueProvenanceReceipt",
}

TRANSCRIPT_ENTRY_SCHEMA_VERSION = "table2-runtime-value-transcript-entry-v1"
TRANSCRIPT_ROOT_SCHEMA_VERSION = "table2-runtime-value-transcript-root-v1"
TRANSCRIPT_CHAIN_ALGORITHM = "sha256_canonical_json_previous_entry_v1"
LOCAL_TRANSCRIPT_CLAIM_SCOPE = (
    "LOCAL_HASH_RECONSTRUCTION_NOT_EXTERNAL_VALUE_PROVENANCE"
)
TRANSCRIPT_CHANNELS = frozenset({"PAGE_BROKER", "DGX_BRIDGE"})
TRANSCRIPT_DIRECTIONS = frozenset({"REQUEST", "RESPONSE"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$")
_SYSTEM_IDS = ("E0", "E1", "E2", "E3")

_REGISTRY_FIELDS = {
    "schema_version",
    "registry_id",
    "status",
    "authorities",
}
_AUTHORITY_FIELDS = {
    "authority_id",
    "key_id",
    "algorithm",
    "public_key_base64",
    "public_key_sha256",
    "allowed_roles",
    "allowed_phases",
    "allowed_topologies",
    "measurement_profile_id",
    "measurement_profile_sha256",
    "valid_from_utc",
    "valid_until_utc",
    "revoked",
}
_CHALLENGE_FIELDS = {
    "schema_version",
    "record_type",
    "registry_sha256",
    "campaign_id",
    "handoff_manifest_sha256",
    "runner_attestation_sha256",
    "semantic_dependency_lock_sha256",
    "deployment_topology",
    "authority_id",
    "key_id",
    "authority_role",
    "measurement_profile_id",
    "measurement_profile_sha256",
    "phase",
    "block_id",
    "rerun_id",
    "challenge_nonce",
    "issued_at_utc",
    "expires_at_utc",
    "previous_challenge_sha256",
    "previous_receipt_sha256",
}
_PAYLOAD_COMMON_FIELDS = {
    "schema_version",
    "record_type",
    "phase",
    "registry_sha256",
    "authority_id",
    "key_id",
    "authority_role",
    "measurement_profile_id",
    "measurement_profile_sha256",
    "deployment_topology",
    "campaign_id",
    "handoff_manifest_sha256",
    "runner_attestation_sha256",
    "semantic_dependency_lock_sha256",
    "challenge_sha256",
    "challenge_nonce",
    "block_id",
    "rerun_id",
    "issued_at_utc",
    "measured_at_utc",
    "previous_receipt_sha256",
    "authority_sequence",
    "dgx_service_session_id",
    "dgx_host_identity_sha256",
    "dgx_runtime_identity_sha256",
    "bridge_identity_sha256",
    "measurements",
}
_ENVELOPE_FIELDS = {
    "schema_version",
    "record_type",
    "algorithm",
    "canonicalization",
    "authority_id",
    "key_id",
    "payload_sha256",
    "payload",
    "signature_base64",
}
_STARTUP_MEASUREMENT_FIELDS = {
    "service_executable_sha256",
    "container_image_digest",
    "runtime_source_set_sha256",
    "cuda_runtime_identity_sha256",
    "dependency_identity_sha256",
    "endpoint_identity_sha256",
    "service_process_identity_sha256",
    "independent_observation_method",
}
_MODEL_LOAD_MEASUREMENT_FIELDS = {
    "startup_receipt_sha256",
    "e0_base_snapshot_sha256",
    "e0_processor_contract_sha256",
    "trained_checkpoint_sha256",
    "resolved_config_sha256",
    "trained_processor_contract_sha256",
    "p4_manifest_sha256",
    "p4_index_sha256",
    "p4_embeddings_sha256",
    "loaded_weights_identity_sha256",
    "model_process_identity_sha256",
    "model_evaluation_mode",
    "memory_read_only",
    "independent_observation_method",
}
_BLOCK_OPEN_MEASUREMENT_FIELDS = {
    "model_load_receipt_sha256",
    "schedule_row_sha256",
    "task_id",
    "repeat_id",
    "matched_model_seed",
    "execution_order",
    "model_call_count",
    "transcript_entry_count",
    "memory_manifest_sha256",
    "memory_read_only",
    "evaluation_memory_writes_enabled",
    "independent_observation_method",
}
_BLOCK_CLOSE_MEASUREMENT_FIELDS = {
    "block_open_receipt_sha256",
    "episode_ids",
    "system_completion",
    "local_broker_launch_receipt_sha256s",
    "local_broker_cleanup_receipt_sha256s",
    "runtime_transcript_root_sha256",
    "runtime_transcript_entry_count",
    "bridge_transcript_root_sha256",
    "bridge_transcript_entry_count",
    "transcript_chain_algorithm",
    "independent_observation_method",
}
_LEDGER_EVENT_FIELDS = {
    "schema_version",
    "record_type",
    "claim_scope",
    "sequence",
    "event_type",
    "recorded_at_utc",
    "previous_event_sha256",
    "payload",
    "event_sha256",
}
_ISSUED_EVENT_PAYLOAD_FIELDS = {"challenge", "challenge_sha256"}
_CONSUMED_EVENT_PAYLOAD_FIELDS = {
    "challenge",
    "challenge_sha256",
    "envelope",
    "payload_sha256",
    "receipt_sha256",
    "phase",
    "campaign_id",
    "block_id",
    "rerun_id",
    "authority_id",
    "key_id",
    "authority_sequence",
    "handoff_manifest_sha256",
    "runner_attestation_sha256",
    "semantic_dependency_lock_sha256",
    "dgx_service_session_id",
    "dgx_host_identity_sha256",
    "dgx_runtime_identity_sha256",
    "bridge_identity_sha256",
}
_TRANSCRIPT_ENTRY_FIELDS = {
    "schema_version",
    "claim_scope",
    "campaign_id",
    "block_id",
    "rerun_id",
    "dgx_service_session_id",
    "sequence",
    "channel",
    "direction",
    "operation",
    "value_sha256",
    "previous_entry_sha256",
    "entry_sha256",
}


@dataclass(frozen=True, slots=True)
class VerifiedDeploymentReceipt:
    """A signature result, not a self-contained production authorization."""

    phase: str
    authority_id: str
    key_id: str
    registry_sha256: str
    challenge_sha256: str
    payload_sha256: str
    receipt_sha256: str
    authority_sequence: int
    campaign_id: str
    block_id: str | None
    rerun_id: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_scope": "SIGNATURE_AUTHENTICITY_ONLY_NOT_DISPATCH_AUTHORITY",
            "external_global_replay_anchor_present": False,
            "production_dispatch_authorized": False,
            "phase": self.phase,
            "authority_id": self.authority_id,
            "key_id": self.key_id,
            "registry_sha256": self.registry_sha256,
            "challenge_sha256": self.challenge_sha256,
            "payload_sha256": self.payload_sha256,
            "receipt_sha256": self.receipt_sha256,
            "authority_sequence": self.authority_sequence,
            "campaign_id": self.campaign_id,
            "block_id": self.block_id,
            "rerun_id": self.rerun_id,
        }


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def loads_strict_json(text: str, *, context: str = "JSON") -> Any:
    """Parse JSON while rejecting duplicate keys and non-standard numbers."""

    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda value: (_raise_invalid_constant(value)),
            parse_float=_finite_json_float,
        )
    except SchemaError:
        raise
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise SchemaError(f"{context} is not strict JSON") from exc
    _validate_json_nesting(value, context=context)
    return value


def _raise_invalid_constant(value: str) -> None:
    raise SchemaError(f"non-finite JSON number is forbidden: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SchemaError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _validate_json_nesting(value: Any, *, context: str) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if isinstance(current, dict):
            nested = current.values()
        elif isinstance(current, list):
            nested = current
        else:
            continue
        next_depth = depth + 1
        if next_depth > MAX_AUTHORITY_JSON_NESTING:
            raise SchemaError(f"{context} exceeds the registered JSON nesting bound")
        pending.extend((item, next_depth) for item in nested)


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


@contextmanager
def _opened_secure_parent(
    path: str | Path, *, label: str
) -> Iterator[tuple[Path, int, str]]:
    """Resolve every ancestor through no-follow directory descriptors."""

    candidate = _lexical_absolute(path)
    if not candidate.name:
        raise SchemaError(f"{label} must name a file")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(candidate.anchor or os.sep, directory_flags)
        for component in candidate.parts[1:-1]:
            next_descriptor = -1
            try:
                next_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
            except OSError:
                if next_descriptor >= 0:
                    os.close(next_descriptor)
                raise
            descriptor = next_descriptor
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        raise SchemaError(f"cannot securely resolve {label} ancestry") from exc
    try:
        yield candidate, descriptor, candidate.name
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _bound_leaf_metadata(
    parent_descriptor: int,
    leaf: str,
    opened: os.stat_result,
    *,
    label: str,
) -> os.stat_result:
    try:
        current = os.stat(leaf, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise SchemaError(f"{label} path changed while open") from exc
    if stat.S_ISLNK(current.st_mode) or not _unchanged_file_metadata(
        opened, current
    ):
        raise SchemaError(f"{label} path changed while open")
    return current


def _same_parent_binding(
    before: os.stat_result,
    after: os.stat_result,
) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_uid,
        before.st_gid,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_uid,
        after.st_gid,
    )


def _assert_reopened_path_binding(
    path: Path,
    original_parent: os.stat_result,
    opened_leaf: os.stat_result,
    *,
    label: str,
) -> None:
    with _opened_secure_parent(path, label=label) as (_, parent_descriptor, leaf):
        reopened_parent = os.fstat(parent_descriptor)
        if not _same_parent_binding(original_parent, reopened_parent):
            raise SchemaError(f"{label} parent changed while open")
        try:
            metadata = os.stat(
                leaf,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise SchemaError(f"{label} path changed while open") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SchemaError(f"{label} must not use symlinks")
        if not _unchanged_file_metadata(opened_leaf, metadata):
            raise SchemaError(f"{label} path changed while open")


def _unchanged_file_metadata(
    before: os.stat_result,
    after: os.stat_result,
) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_uid,
        before.st_gid,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_uid,
        after.st_gid,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _read_bounded_regular_file(
    path: str | Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    with _opened_secure_parent(path, label=label) as (
        candidate,
        parent_descriptor,
        leaf,
    ):
        original_parent = os.fstat(parent_descriptor)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(leaf, flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise SchemaError(f"cannot open {label}: {candidate}") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise SchemaError(f"{label} must be a non-hard-linked regular file")
            if before.st_size > max_bytes:
                raise SchemaError(f"{label} exceeds the registered byte bound")
            _bound_leaf_metadata(parent_descriptor, leaf, before, label=label)
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read(max_bytes + 1)
            if len(payload) > max_bytes:
                raise SchemaError(f"{label} exceeds the registered byte bound")
            after = os.fstat(descriptor)
            if len(payload) != before.st_size or not _unchanged_file_metadata(
                before, after
            ):
                raise SchemaError(f"{label} changed during bounded read")
            _bound_leaf_metadata(parent_descriptor, leaf, after, label=label)
            if not _same_parent_binding(
                original_parent, os.fstat(parent_descriptor)
            ):
                raise SchemaError(f"{label} parent changed during bounded read")
            _assert_reopened_path_binding(
                candidate,
                original_parent,
                after,
                label=label,
            )
            final = os.fstat(descriptor)
            if not _unchanged_file_metadata(after, final):
                raise SchemaError(f"{label} changed during path rebound")
            _bound_leaf_metadata(parent_descriptor, leaf, final, label=label)
            return payload
        finally:
            os.close(descriptor)


def _read_strict_json_with_sha256(path: str | Path) -> tuple[Any, str]:
    candidate = _lexical_absolute(path)
    payload = _read_bounded_regular_file(
        candidate,
        label="authority JSON",
        max_bytes=MAX_AUTHORITY_JSON_BYTES,
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaError(f"authority JSON is not UTF-8: {candidate}") from exc
    return (
        loads_strict_json(text, context=str(candidate)),
        hashlib.sha256(payload).hexdigest(),
    )


def read_strict_json(path: str | Path) -> Any:
    return _read_strict_json_with_sha256(path)[0]


def _require_exact_fields(value: object, expected: set[str], *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SchemaError(f"{context} fields differ from the registered schema")
    return dict(value)


def _text(value: object, *, field: str, identifier: bool = False) -> str:
    if type(value) is not str or not value or value != value.strip() or len(value) > 512:
        raise SchemaError(f"{field} must be non-empty canonical text")
    if not value.isascii():
        raise SchemaError(f"{field} must be ASCII for cross-host canonical signing")
    if any(ord(character) < 0x20 for character in value):
        raise SchemaError(f"{field} contains a control character")
    if identifier and _IDENTIFIER_RE.fullmatch(value) is None:
        raise SchemaError(f"{field} is not a canonical identifier")
    return value


def _sha256(value: object, *, field: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SchemaError(f"{field} must be lowercase SHA-256")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SchemaError(f"{field} must be an integer >= {minimum}")
    return value


def _strict_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise SchemaError(f"{field} must be a Boolean")
    return value


def _parse_utc(value: object, *, field: str) -> datetime:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        raise SchemaError(f"{field} must be UTC at whole-second precision")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise SchemaError(f"{field} is not a valid UTC timestamp") from exc


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise SchemaError("authority clock must be timezone-aware")
    return value.astimezone(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _require_aware_utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SchemaError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc).replace(microsecond=0)


def _strict_base64(value: object, *, field: str, expected_length: int) -> bytes:
    if type(value) is not str or not value or value != value.strip():
        raise SchemaError(f"{field} must be canonical base64 text")
    expected_encoded_length = ((expected_length + 2) // 3) * 4
    if len(value) != expected_encoded_length:
        raise SchemaError(f"{field} has the wrong decoded length")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SchemaError(f"{field} is not strict base64") from exc
    if len(decoded) != expected_length:
        raise SchemaError(f"{field} has the wrong decoded length")
    if base64.b64encode(decoded).decode("ascii") != value:
        raise SchemaError(f"{field} is not canonical padded base64")
    return decoded


def _ordered_unique_texts(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise SchemaError(f"{field} must be a non-empty array")
    rows = [_text(item, field=f"{field}[]", identifier=True) for item in value]
    if rows != sorted(set(rows)):
        raise SchemaError(f"{field} must be sorted and unique")
    return rows


def validate_authority_registry(value: object) -> dict[str, Any]:
    registry = _require_exact_fields(value, _REGISTRY_FIELDS, context="authority registry")
    if registry["schema_version"] != AUTHORITY_REGISTRY_SCHEMA_VERSION:
        raise SchemaError("authority registry schema version differs")
    if registry["registry_id"] != AUTHORITY_REGISTRY_ID:
        raise SchemaError("authority registry ID differs")
    authorities = registry["authorities"]
    if not isinstance(authorities, list):
        raise SchemaError("authority registry authorities must be an array")
    expected_status = EMPTY_REGISTRY_STATUS if not authorities else ACTIVE_REGISTRY_STATUS
    if registry["status"] != expected_status:
        raise SchemaError("authority registry status is inconsistent with its entries")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(authorities):
        row = _require_exact_fields(
            raw, _AUTHORITY_FIELDS, context=f"authority registry entry {index}"
        )
        authority_id = _text(
            row["authority_id"], field="authority_id", identifier=True
        )
        key_id = _text(row["key_id"], field="key_id", identifier=True)
        identity = (authority_id, key_id)
        if identity in seen:
            raise SchemaError("authority registry contains a duplicate authority/key")
        seen.add(identity)
        if row["algorithm"] != SIGNATURE_ALGORITHM:
            raise SchemaError("authority registry algorithm must be Ed25519")
        public_key = _strict_base64(
            row["public_key_base64"],
            field="public_key_base64",
            expected_length=32,
        )
        if hashlib.sha256(public_key).hexdigest() != row["public_key_sha256"]:
            raise SchemaError("authority public-key fingerprint differs")
        _sha256(row["public_key_sha256"], field="public_key_sha256")
        roles = _ordered_unique_texts(row["allowed_roles"], field="allowed_roles")
        if not set(roles).issubset({DGX_DEPLOYMENT_ROLE, RUNTIME_VALUE_PROVENANCE_ROLE}):
            raise SchemaError("authority registry contains an unknown role")
        phases = _ordered_unique_texts(row["allowed_phases"], field="allowed_phases")
        if not set(phases).issubset(set(AUTHORITY_PHASES)):
            raise SchemaError("authority registry contains an unknown phase")
        topologies = _ordered_unique_texts(
            row["allowed_topologies"], field="allowed_topologies"
        )
        if topologies != [SPLIT_DEPLOYMENT_TOPOLOGY]:
            raise SchemaError("phase-one authority permits only the split topology")
        _text(
            row["measurement_profile_id"],
            field="measurement_profile_id",
            identifier=True,
        )
        _sha256(
            row["measurement_profile_sha256"],
            field="measurement_profile_sha256",
        )
        valid_from = _parse_utc(row["valid_from_utc"], field="valid_from_utc")
        valid_until = _parse_utc(row["valid_until_utc"], field="valid_until_utc")
        if valid_until <= valid_from:
            raise SchemaError("authority validity interval is empty")
        _strict_bool(row["revoked"], field="revoked")
        normalized.append(row)
    if [(row["authority_id"], row["key_id"]) for row in normalized] != sorted(seen):
        raise SchemaError("authority registry entries must be sorted by authority/key")
    return json.loads(canonical_json_bytes(registry))


def production_authority_registry_resource() -> Any:
    """Return the single package-owned authority-registry resource."""

    return files("web_agent.eval.table2").joinpath(
        PRODUCTION_AUTHORITY_REGISTRY_RESOURCE
    )


def load_production_authority_registry() -> dict[str, Any]:
    resource = production_authority_registry_resource()
    if isinstance(resource, Path):
        value, file_sha256 = _read_strict_json_with_sha256(resource)
    else:  # Supports non-filesystem package-resource loaders.
        try:
            with resource.open("rb") as handle:
                payload = handle.read(MAX_AUTHORITY_JSON_BYTES + 1)
            if len(payload) > MAX_AUTHORITY_JSON_BYTES:
                raise SchemaError(
                    "packaged authority registry exceeds the registered byte bound"
                )
            text = payload.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SchemaError("cannot read the packaged authority registry") from exc
        value = loads_strict_json(text, context="packaged authority registry")
        file_sha256 = hashlib.sha256(payload).hexdigest()
    if file_sha256 != PRODUCTION_AUTHORITY_REGISTRY_FILE_SHA256:
        raise SchemaError("production authority registry differs from its source pin")
    registry = validate_authority_registry(value)
    if registry["status"] != EMPTY_REGISTRY_STATUS or registry["authorities"] != []:
        raise SchemaError(
            "this source version permits only the initially empty authority registry"
        )
    return registry


def _find_authority(
    registry: Mapping[str, Any], *, authority_id: str, key_id: str
) -> dict[str, Any]:
    validated = validate_authority_registry(registry)
    matches = [
        dict(row)
        for row in validated["authorities"]
        if row["authority_id"] == authority_id and row["key_id"] == key_id
    ]
    if len(matches) != 1:
        raise SchemaError("receipt authority/key is not registered")
    return matches[0]


def _validate_authority_permission(
    authority: Mapping[str, Any],
    *,
    role: str,
    phase: str,
    topology: str,
    profile_id: str,
    profile_sha256: str,
    at: datetime,
) -> None:
    if authority["revoked"] is True:
        raise SchemaError("receipt authority key is revoked")
    if not (
        _parse_utc(authority["valid_from_utc"], field="valid_from_utc")
        <= at
        <= _parse_utc(authority["valid_until_utc"], field="valid_until_utc")
    ):
        raise SchemaError("receipt was issued outside the authority key validity window")
    if role not in authority["allowed_roles"]:
        raise SchemaError("receipt authority is not permitted for this role")
    if phase not in authority["allowed_phases"]:
        raise SchemaError("receipt authority is not permitted for this phase")
    if topology not in authority["allowed_topologies"]:
        raise SchemaError("receipt authority is not permitted for this topology")
    if (
        profile_id != authority["measurement_profile_id"]
        or profile_sha256 != authority["measurement_profile_sha256"]
    ):
        raise SchemaError("receipt measurement profile differs from the registered profile")


def validate_challenge(
    value: object, *, active_at: datetime | None = None
) -> dict[str, Any]:
    challenge = _require_exact_fields(value, _CHALLENGE_FIELDS, context="authority challenge")
    if challenge["schema_version"] != CHALLENGE_SCHEMA_VERSION:
        raise SchemaError("authority challenge schema version differs")
    if challenge["record_type"] != CHALLENGE_RECORD_TYPE:
        raise SchemaError("authority challenge record type differs")
    for field in (
        "registry_sha256",
        "handoff_manifest_sha256",
        "runner_attestation_sha256",
        "semantic_dependency_lock_sha256",
    ):
        _sha256(challenge[field], field=field)
    _sha256(
        challenge["previous_challenge_sha256"],
        field="previous_challenge_sha256",
        nullable=True,
    )
    _sha256(
        challenge["previous_receipt_sha256"],
        field="previous_receipt_sha256",
        nullable=True,
    )
    for field in (
        "campaign_id",
        "authority_id",
        "key_id",
        "measurement_profile_id",
    ):
        _text(challenge[field], field=field, identifier=True)
    if challenge["deployment_topology"] != SPLIT_DEPLOYMENT_TOPOLOGY:
        raise SchemaError("authority challenge topology differs")
    phase = challenge["phase"]
    if phase not in AUTHORITY_PHASES:
        raise SchemaError("authority challenge phase is unknown")
    if challenge["authority_role"] != PHASE_ROLES[phase]:
        raise SchemaError("authority challenge role differs from its phase")
    _sha256(
        challenge["measurement_profile_sha256"],
        field="measurement_profile_sha256",
    )
    nonce = challenge["challenge_nonce"]
    if type(nonce) is not str or _SHA256_RE.fullmatch(nonce) is None:
        raise SchemaError("challenge nonce must contain exactly 256 random bits")
    issued = _parse_utc(challenge["issued_at_utc"], field="issued_at_utc")
    expires = _parse_utc(challenge["expires_at_utc"], field="expires_at_utc")
    ttl = (expires - issued).total_seconds()
    if ttl <= 0 or ttl > MAX_CHALLENGE_TTL_SECONDS:
        raise SchemaError("authority challenge validity period is invalid")
    if active_at is not None and not (issued <= active_at <= expires):
        raise SchemaError("authority challenge is not active")
    if phase == DGX_SERVICE_STARTUP_PHASE:
        if (
            challenge["block_id"] is not None
            or challenge["rerun_id"] is not None
            or challenge["previous_receipt_sha256"] is not None
        ):
            raise SchemaError("startup challenge must begin a campaign receipt chain")
    elif phase == DGX_MODEL_LOAD_PHASE:
        if challenge["block_id"] is not None or challenge["rerun_id"] is not None:
            raise SchemaError("model-load challenge cannot name a physical block")
        _sha256(
            challenge["previous_receipt_sha256"], field="previous_receipt_sha256"
        )
    else:
        _text(challenge["block_id"], field="block_id", identifier=True)
        _integer(challenge["rerun_id"], field="rerun_id")
        _sha256(
            challenge["previous_receipt_sha256"], field="previous_receipt_sha256"
        )
    return json.loads(canonical_json_bytes(challenge))


def _validate_sha256_map(value: object, *, field: str) -> None:
    mapping = _require_exact_fields(value, set(_SYSTEM_IDS), context=field)
    for system_id in _SYSTEM_IDS:
        _sha256(mapping[system_id], field=f"{field}.{system_id}")


def _validate_text_map(value: object, *, field: str) -> None:
    mapping = _require_exact_fields(value, set(_SYSTEM_IDS), context=field)
    for system_id in _SYSTEM_IDS:
        _text(mapping[system_id], field=f"{field}.{system_id}")


def _validate_bool_map(value: object, *, field: str, require_all_true: bool) -> None:
    mapping = _require_exact_fields(value, set(_SYSTEM_IDS), context=field)
    for system_id in _SYSTEM_IDS:
        result = _strict_bool(mapping[system_id], field=f"{field}.{system_id}")
        if require_all_true and not result:
            raise SchemaError(f"{field} must close one complete matched E0-E3 block")


def _validate_measurements(phase: str, value: object) -> dict[str, Any]:
    if phase == DGX_SERVICE_STARTUP_PHASE:
        measurements = _require_exact_fields(
            value, _STARTUP_MEASUREMENT_FIELDS, context="startup measurements"
        )
        for field in _STARTUP_MEASUREMENT_FIELDS - {
            "container_image_digest",
            "independent_observation_method",
        }:
            _sha256(measurements[field], field=f"measurements.{field}")
        _text(
            measurements["container_image_digest"],
            field="measurements.container_image_digest",
        )
    elif phase == DGX_MODEL_LOAD_PHASE:
        measurements = _require_exact_fields(
            value, _MODEL_LOAD_MEASUREMENT_FIELDS, context="model-load measurements"
        )
        for field in _MODEL_LOAD_MEASUREMENT_FIELDS - {
            "model_evaluation_mode",
            "memory_read_only",
            "independent_observation_method",
        }:
            _sha256(measurements[field], field=f"measurements.{field}")
        if not _strict_bool(
            measurements["model_evaluation_mode"],
            field="measurements.model_evaluation_mode",
        ):
            raise SchemaError("model-load receipt must attest evaluation mode")
        if not _strict_bool(
            measurements["memory_read_only"], field="measurements.memory_read_only"
        ):
            raise SchemaError("model-load receipt must attest read-only memory")
    elif phase == BLOCK_OPEN_PHASE:
        measurements = _require_exact_fields(
            value, _BLOCK_OPEN_MEASUREMENT_FIELDS, context="block-open measurements"
        )
        for field in (
            "model_load_receipt_sha256",
            "schedule_row_sha256",
            "memory_manifest_sha256",
        ):
            _sha256(measurements[field], field=f"measurements.{field}")
        _text(measurements["task_id"], field="measurements.task_id")
        _integer(measurements["repeat_id"], field="measurements.repeat_id")
        _integer(
            measurements["matched_model_seed"], field="measurements.matched_model_seed"
        )
        order = measurements["execution_order"]
        if (
            not isinstance(order, list)
            or any(type(item) is not str for item in order)
            or len(order) != 4
            or set(order) != set(_SYSTEM_IDS)
        ):
            raise SchemaError("block-open execution order must be one E0-E3 permutation")
        if _integer(
            measurements["model_call_count"], field="measurements.model_call_count"
        ) != 0:
            raise SchemaError("block-open model-call counter must be zero")
        if _integer(
            measurements["transcript_entry_count"],
            field="measurements.transcript_entry_count",
        ) != 0:
            raise SchemaError("block-open transcript counter must be zero")
        if not _strict_bool(
            measurements["memory_read_only"], field="measurements.memory_read_only"
        ):
            raise SchemaError("block-open receipt must attest read-only memory")
        if _strict_bool(
            measurements["evaluation_memory_writes_enabled"],
            field="measurements.evaluation_memory_writes_enabled",
        ):
            raise SchemaError("evaluation memory writes must remain disabled")
    elif phase == BLOCK_CLOSE_PHASE:
        measurements = _require_exact_fields(
            value, _BLOCK_CLOSE_MEASUREMENT_FIELDS, context="block-close measurements"
        )
        for field in (
            "block_open_receipt_sha256",
            "runtime_transcript_root_sha256",
            "bridge_transcript_root_sha256",
        ):
            _sha256(measurements[field], field=f"measurements.{field}")
        _validate_text_map(measurements["episode_ids"], field="measurements.episode_ids")
        _validate_bool_map(
            measurements["system_completion"],
            field="measurements.system_completion",
            require_all_true=True,
        )
        _validate_sha256_map(
            measurements["local_broker_launch_receipt_sha256s"],
            field="measurements.local_broker_launch_receipt_sha256s",
        )
        _validate_sha256_map(
            measurements["local_broker_cleanup_receipt_sha256s"],
            field="measurements.local_broker_cleanup_receipt_sha256s",
        )
        _integer(
            measurements["runtime_transcript_entry_count"],
            field="measurements.runtime_transcript_entry_count",
            minimum=1,
        )
        _integer(
            measurements["bridge_transcript_entry_count"],
            field="measurements.bridge_transcript_entry_count",
            minimum=1,
        )
        if measurements["transcript_chain_algorithm"] != TRANSCRIPT_CHAIN_ALGORITHM:
            raise SchemaError("block-close transcript chain algorithm differs")
    else:  # pragma: no cover - guarded by the payload validator
        raise SchemaError("receipt payload phase is unknown")
    if measurements["independent_observation_method"] != INDEPENDENT_OBSERVATION_METHOD:
        raise SchemaError("receipt lacks the registered independent observation method")
    return measurements


def validate_receipt_payload(value: object) -> dict[str, Any]:
    payload = _require_exact_fields(value, _PAYLOAD_COMMON_FIELDS, context="receipt payload")
    phase = payload["phase"]
    if phase not in AUTHORITY_PHASES:
        raise SchemaError("receipt payload phase is unknown")
    if payload["schema_version"] != PAYLOAD_SCHEMA_BY_PHASE[phase]:
        raise SchemaError("receipt payload schema version differs from its phase")
    if payload["record_type"] != PAYLOAD_RECORD_TYPE_BY_PHASE[phase]:
        raise SchemaError("receipt payload record type differs from its phase")
    if payload["authority_role"] != PHASE_ROLES[phase]:
        raise SchemaError("receipt payload authority role differs from its phase")
    if payload["deployment_topology"] != SPLIT_DEPLOYMENT_TOPOLOGY:
        raise SchemaError("receipt payload topology differs")
    for field in (
        "registry_sha256",
        "handoff_manifest_sha256",
        "runner_attestation_sha256",
        "semantic_dependency_lock_sha256",
        "challenge_sha256",
        "dgx_host_identity_sha256",
        "dgx_runtime_identity_sha256",
        "bridge_identity_sha256",
        "measurement_profile_sha256",
    ):
        _sha256(payload[field], field=field)
    _sha256(
        payload["previous_receipt_sha256"],
        field="previous_receipt_sha256",
        nullable=phase == DGX_SERVICE_STARTUP_PHASE,
    )
    if phase == DGX_SERVICE_STARTUP_PHASE and payload["previous_receipt_sha256"] is not None:
        raise SchemaError("startup receipt must begin a receipt chain")
    for field in (
        "campaign_id",
        "authority_id",
        "key_id",
        "measurement_profile_id",
        "dgx_service_session_id",
    ):
        _text(payload[field], field=field, identifier=True)
    nonce = payload["challenge_nonce"]
    if type(nonce) is not str or _SHA256_RE.fullmatch(nonce) is None:
        raise SchemaError("receipt challenge nonce is malformed")
    issued = _parse_utc(payload["issued_at_utc"], field="issued_at_utc")
    measured = _parse_utc(payload["measured_at_utc"], field="measured_at_utc")
    if measured > issued:
        raise SchemaError("receipt measurement cannot occur after receipt issuance")
    _integer(payload["authority_sequence"], field="authority_sequence", minimum=1)
    if phase in {DGX_SERVICE_STARTUP_PHASE, DGX_MODEL_LOAD_PHASE}:
        if payload["block_id"] is not None or payload["rerun_id"] is not None:
            raise SchemaError("startup/model-load receipt cannot name a physical block")
    else:
        _text(payload["block_id"], field="block_id", identifier=True)
        _integer(payload["rerun_id"], field="rerun_id")
    measurements = _validate_measurements(phase, payload["measurements"])
    prior_measurement_field = {
        DGX_MODEL_LOAD_PHASE: "startup_receipt_sha256",
        BLOCK_OPEN_PHASE: "model_load_receipt_sha256",
        BLOCK_CLOSE_PHASE: "block_open_receipt_sha256",
    }.get(phase)
    if (
        prior_measurement_field is not None
        and measurements[prior_measurement_field]
        != payload["previous_receipt_sha256"]
    ):
        raise SchemaError("phase measurement does not bind the prior receipt")
    return json.loads(canonical_json_bytes(payload))


def _validate_envelope(value: object) -> dict[str, Any]:
    envelope = _require_exact_fields(value, _ENVELOPE_FIELDS, context="signed envelope")
    if envelope["schema_version"] != SIGNED_ENVELOPE_SCHEMA_VERSION:
        raise SchemaError("signed envelope schema version differs")
    if envelope["record_type"] != SIGNED_ENVELOPE_RECORD_TYPE:
        raise SchemaError("signed envelope record type differs")
    if envelope["algorithm"] != SIGNATURE_ALGORITHM:
        raise SchemaError("signed envelope algorithm must be Ed25519")
    if envelope["canonicalization"] != CANONICAL_JSON_ALGORITHM:
        raise SchemaError("signed envelope canonicalization algorithm differs")
    _text(envelope["authority_id"], field="authority_id", identifier=True)
    _text(envelope["key_id"], field="key_id", identifier=True)
    payload = validate_receipt_payload(envelope["payload"])
    if (
        envelope["authority_id"] != payload["authority_id"]
        or envelope["key_id"] != payload["key_id"]
    ):
        raise SchemaError("signed envelope authority differs from its payload")
    expected_payload_sha256 = sha256_json(payload)
    if envelope["payload_sha256"] != expected_payload_sha256:
        raise SchemaError("signed envelope payload hash differs")
    _strict_base64(
        envelope["signature_base64"],
        field="signature_base64",
        expected_length=64,
    )
    return json.loads(canonical_json_bytes(envelope))


_CHALLENGE_TO_PAYLOAD_FIELDS = {
    "registry_sha256",
    "campaign_id",
    "handoff_manifest_sha256",
    "runner_attestation_sha256",
    "semantic_dependency_lock_sha256",
    "deployment_topology",
    "authority_id",
    "key_id",
    "authority_role",
    "measurement_profile_id",
    "measurement_profile_sha256",
    "phase",
    "block_id",
    "rerun_id",
    "challenge_nonce",
    "previous_receipt_sha256",
}


def _verify_receipt_against_registry(
    envelope_value: object,
    *,
    registry_value: object,
    challenge_value: object,
    active_at: datetime | None = None,
) -> VerifiedDeploymentReceipt:
    """Internal verifier used by fixed-registry entrypoints and synthetic tests.

    This lower-level function exists for deterministic tests. Offline inspection
    may use :func:`inspect_production_receipt`. Local serialization may use
    :func:`consume_production_receipt`, which reloads the fixed source registry
    and currently rejects every receipt. Neither path authorizes dispatch; that
    additionally requires an independent external replay anchor.
    """

    now = _require_aware_utc(active_at or _utc_now(), field="active_at")
    registry = validate_authority_registry(registry_value)
    challenge = validate_challenge(challenge_value, active_at=now)
    envelope = _validate_envelope(envelope_value)
    payload = dict(envelope["payload"])
    registry_sha256 = sha256_json(registry)
    if challenge["registry_sha256"] != registry_sha256:
        raise SchemaError("challenge registry hash differs")
    if payload["challenge_sha256"] != sha256_json(challenge):
        raise SchemaError("receipt does not bind the exact challenge")
    for field in _CHALLENGE_TO_PAYLOAD_FIELDS:
        if payload[field] != challenge[field]:
            raise SchemaError(f"receipt/challenge binding differs: {field}")
    challenge_issued = _parse_utc(challenge["issued_at_utc"], field="issued_at_utc")
    challenge_expires = _parse_utc(challenge["expires_at_utc"], field="expires_at_utc")
    measured = _parse_utc(payload["measured_at_utc"], field="measured_at_utc")
    issued = _parse_utc(payload["issued_at_utc"], field="issued_at_utc")
    if not (challenge_issued <= measured <= issued <= challenge_expires):
        raise SchemaError("receipt measurement/issuance is outside its challenge window")
    if issued > now:
        raise SchemaError("receipt issuance time is in the verifier's future")

    authority = _find_authority(
        registry,
        authority_id=payload["authority_id"],
        key_id=payload["key_id"],
    )
    _validate_authority_permission(
        authority,
        role=payload["authority_role"],
        phase=payload["phase"],
        topology=payload["deployment_topology"],
        profile_id=payload["measurement_profile_id"],
        profile_sha256=payload["measurement_profile_sha256"],
        at=issued,
    )
    public_key = _strict_base64(
        authority["public_key_base64"],
        field="public_key_base64",
        expected_length=32,
    )
    signature = _strict_base64(
        envelope["signature_base64"],
        field="signature_base64",
        expected_length=64,
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            SIGNATURE_DOMAIN + canonical_json_bytes(payload),
        )
    except (InvalidSignature, ValueError) as exc:
        raise SchemaError("deployment authority signature verification failed") from exc
    return VerifiedDeploymentReceipt(
        phase=payload["phase"],
        authority_id=payload["authority_id"],
        key_id=payload["key_id"],
        registry_sha256=registry_sha256,
        challenge_sha256=payload["challenge_sha256"],
        payload_sha256=envelope["payload_sha256"],
        receipt_sha256=sha256_json(envelope),
        authority_sequence=payload["authority_sequence"],
        campaign_id=payload["campaign_id"],
        block_id=payload["block_id"],
        rerun_id=payload["rerun_id"],
    )


def inspect_production_receipt(
    envelope_value: object,
    *,
    challenge_value: object,
    active_at: datetime | None = None,
) -> VerifiedDeploymentReceipt:
    """Inspect authenticity through the fixed registry without consuming it.

    This is explicitly not replay-safe dispatch authority. Even local consumption
    cannot authorize dispatch without an independently anchored nonce/sequence.
    """

    return _verify_receipt_against_registry(
        envelope_value,
        registry_value=load_production_authority_registry(),
        challenge_value=challenge_value,
        active_at=active_at,
    )


def _parse_canonical_jsonl_objects(
    payload: bytes,
    *,
    context: str,
    max_events: int,
) -> list[dict[str, Any]]:
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        raise SchemaError(f"{context} must end with LF")
    if b"\r" in payload:
        raise SchemaError(f"{context} must use LF rather than CRLF")
    lines = payload[:-1].split(b"\n")
    if len(lines) > max_events:
        raise SchemaError(f"{context} exceeds the registered event bound")
    rows: list[dict[str, Any]] = []
    for line_number, encoded in enumerate(lines, start=1):
        if not encoded:
            raise SchemaError(f"{context} contains a blank line")
        if len(encoded) > MAX_AUTHORITY_JSONL_LINE_BYTES:
            raise SchemaError(f"{context} line exceeds the registered byte bound")
        try:
            line = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SchemaError(f"{context} is not UTF-8") from exc
        value = loads_strict_json(line, context=f"{context} line {line_number}")
        if not isinstance(value, dict):
            raise SchemaError(f"{context} entries must be objects")
        try:
            canonical = canonical_json_bytes(value)
        except (RecursionError, TypeError, ValueError) as exc:
            raise SchemaError(f"{context} line cannot be canonicalized") from exc
        if canonical != encoded:
            raise SchemaError(f"{context} line is not canonical JSON")
        rows.append(value)
    return rows


def read_runtime_transcript_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a bounded, canonical runtime-value transcript from a stable inode."""

    payload = _read_bounded_regular_file(
        path,
        label="runtime-value transcript",
        max_bytes=MAX_RUNTIME_TRANSCRIPT_BYTES,
    )
    return _parse_canonical_jsonl_objects(
        payload,
        context="runtime-value transcript",
        max_events=MAX_RUNTIME_TRANSCRIPT_EVENTS,
    )


def _validate_transcript_entry(
    value: object,
    *,
    expected_sequence: int,
    previous: str | None,
    campaign_id: str,
    block_id: str,
    rerun_id: int,
    dgx_service_session_id: str,
    expected_channel: str,
) -> dict[str, Any]:
    entry = _require_exact_fields(
        value, _TRANSCRIPT_ENTRY_FIELDS, context="runtime-value transcript entry"
    )
    if entry["schema_version"] != TRANSCRIPT_ENTRY_SCHEMA_VERSION:
        raise SchemaError("runtime-value transcript schema differs")
    if entry["claim_scope"] != LOCAL_TRANSCRIPT_CLAIM_SCOPE:
        raise SchemaError("runtime-value transcript claim scope differs")
    for field, expected in (
        ("campaign_id", campaign_id),
        ("block_id", block_id),
        ("rerun_id", rerun_id),
        ("dgx_service_session_id", dgx_service_session_id),
    ):
        if entry[field] != expected:
            raise SchemaError(f"runtime-value transcript context differs: {field}")
    _text(entry["campaign_id"], field="transcript.campaign_id", identifier=True)
    _text(entry["block_id"], field="transcript.block_id", identifier=True)
    _integer(entry["rerun_id"], field="transcript.rerun_id")
    _text(
        entry["dgx_service_session_id"],
        field="transcript.dgx_service_session_id",
        identifier=True,
    )
    sequence = _integer(entry["sequence"], field="transcript.sequence")
    if sequence != expected_sequence:
        raise SchemaError("runtime-value transcript sequence is not contiguous")
    channel = _text(entry["channel"], field="transcript.channel", identifier=True)
    if channel not in TRANSCRIPT_CHANNELS:
        raise SchemaError("runtime-value transcript channel is unknown")
    if channel != expected_channel:
        raise SchemaError("runtime-value transcript entered the wrong channel chain")
    direction = _text(
        entry["direction"], field="transcript.direction", identifier=True
    )
    if direction not in TRANSCRIPT_DIRECTIONS:
        raise SchemaError("runtime-value transcript direction is unknown")
    _text(entry["operation"], field="transcript.operation", identifier=True)
    _sha256(entry["value_sha256"], field="transcript.value_sha256")
    _sha256(
        entry["previous_entry_sha256"],
        field="transcript.previous_entry_sha256",
        nullable=expected_sequence == 0,
    )
    if entry["previous_entry_sha256"] != previous:
        raise SchemaError("runtime-value transcript previous hash differs")
    unsigned = dict(entry)
    supplied = unsigned.pop("entry_sha256")
    expected = sha256_json(unsigned)
    if supplied != expected:
        raise SchemaError("runtime-value transcript entry hash differs")
    return entry


def build_transcript_entry(
    *,
    campaign_id: str,
    block_id: str,
    rerun_id: int,
    dgx_service_session_id: str,
    sequence: int,
    channel: str,
    direction: str,
    operation: str,
    value_sha256: str,
    previous_entry_sha256: str | None,
) -> dict[str, Any]:
    """Build local hash evidence without claiming external provenance."""

    unsigned = {
        "schema_version": TRANSCRIPT_ENTRY_SCHEMA_VERSION,
        "claim_scope": LOCAL_TRANSCRIPT_CLAIM_SCOPE,
        "campaign_id": campaign_id,
        "block_id": block_id,
        "rerun_id": rerun_id,
        "dgx_service_session_id": dgx_service_session_id,
        "sequence": sequence,
        "channel": channel,
        "direction": direction,
        "operation": operation,
        "value_sha256": value_sha256,
        "previous_entry_sha256": previous_entry_sha256,
    }
    entry = {**unsigned, "entry_sha256": sha256_json(unsigned)}
    return _validate_transcript_entry(
        entry,
        expected_sequence=sequence,
        previous=previous_entry_sha256,
        campaign_id=campaign_id,
        block_id=block_id,
        rerun_id=rerun_id,
        dgx_service_session_id=dgx_service_session_id,
        expected_channel=channel,
    )


def reconstruct_transcript_root(
    entries: Sequence[Mapping[str, Any]],
    *,
    campaign_id: str,
    block_id: str,
    rerun_id: int,
    dgx_service_session_id: str,
    expected_channel: str,
) -> dict[str, Any]:
    _text(campaign_id, field="campaign_id", identifier=True)
    _text(block_id, field="block_id", identifier=True)
    _integer(rerun_id, field="rerun_id")
    _text(
        dgx_service_session_id,
        field="dgx_service_session_id",
        identifier=True,
    )
    if expected_channel not in TRANSCRIPT_CHANNELS:
        raise SchemaError("expected transcript channel is unknown")
    previous: str | None = None
    for index, raw in enumerate(entries):
        if index >= MAX_RUNTIME_TRANSCRIPT_EVENTS:
            raise SchemaError("runtime-value transcript exceeds the registered event bound")
        entry = _validate_transcript_entry(
            raw,
            expected_sequence=index,
            previous=previous,
            campaign_id=campaign_id,
            block_id=block_id,
            rerun_id=rerun_id,
            dgx_service_session_id=dgx_service_session_id,
            expected_channel=expected_channel,
        )
        previous = entry["entry_sha256"]
    root = previous if previous is not None else sha256_json([])
    return {
        "schema_version": TRANSCRIPT_ROOT_SCHEMA_VERSION,
        "claim_scope": LOCAL_TRANSCRIPT_CLAIM_SCOPE,
        "chain_algorithm": TRANSCRIPT_CHAIN_ALGORITHM,
        "campaign_id": campaign_id,
        "block_id": block_id,
        "rerun_id": rerun_id,
        "dgx_service_session_id": dgx_service_session_id,
        "channel": expected_channel,
        "entry_count": len(entries),
        "root_sha256": root,
        "external_value_provenance_attested": False,
        "production_dispatch_authorized": False,
    }


def _verify_block_close_transcript_roots_against_registry(
    envelope_value: object,
    *,
    registry_value: object,
    challenge_value: object,
    runtime_entries: Sequence[Mapping[str, Any]],
    bridge_entries: Sequence[Mapping[str, Any]],
    active_at: datetime | None = None,
) -> dict[str, Any]:
    """Authenticate a block-close envelope before comparing transcript roots."""

    envelope = _validate_envelope(envelope_value)
    verified = _verify_receipt_against_registry(
        envelope,
        registry_value=registry_value,
        challenge_value=challenge_value,
        active_at=active_at,
    )
    payload = envelope["payload"]
    if payload["phase"] != BLOCK_CLOSE_PHASE:
        raise SchemaError("transcript-root validation requires a block-close payload")
    context = {
        "campaign_id": payload["campaign_id"],
        "block_id": payload["block_id"],
        "rerun_id": payload["rerun_id"],
        "dgx_service_session_id": payload["dgx_service_session_id"],
    }
    runtime_root = reconstruct_transcript_root(
        runtime_entries,
        expected_channel="PAGE_BROKER",
        **context,
    )
    bridge_root = reconstruct_transcript_root(
        bridge_entries,
        expected_channel="DGX_BRIDGE",
        **context,
    )
    measurements = payload["measurements"]
    if (
        measurements["runtime_transcript_root_sha256"]
        != runtime_root["root_sha256"]
        or measurements["runtime_transcript_entry_count"]
        != runtime_root["entry_count"]
        or measurements["bridge_transcript_root_sha256"]
        != bridge_root["root_sha256"]
        or measurements["bridge_transcript_entry_count"]
        != bridge_root["entry_count"]
    ):
        raise SchemaError("block-close transcript root/count differs from local evidence")
    return {
        "schema_version": "table2-authenticated-block-close-transcript-comparison-v1",
        "claim_scope": (
            "AUTHENTICATED_SIGNER_AND_LOCAL_ROOT_EQUALITY_"
            "NOT_VALUE_ORIGIN_OR_DISPATCH_AUTHORITY"
        ),
        "authenticated_receipt_sha256": verified.receipt_sha256,
        "authenticated_authority_id": verified.authority_id,
        "authenticated_key_id": verified.key_id,
        "runtime": runtime_root,
        "bridge": bridge_root,
        "cryptographic_receipt_verified": True,
        "local_roots_equal_authenticated_claim": True,
        "external_global_replay_anchor_present": False,
        "production_dispatch_authorized": False,
    }


def verify_production_block_close_transcript_roots(
    envelope_value: object,
    *,
    challenge_value: object,
    runtime_entries: Sequence[Mapping[str, Any]],
    bridge_entries: Sequence[Mapping[str, Any]],
    active_at: datetime | None = None,
) -> dict[str, Any]:
    """Use only the packaged registry; success still could not authorize dispatch.

    The empty registry currently rejects. A future authenticated receipt would
    attribute a root claim to its signer, but dispatch additionally requires an
    independent global challenge/sequence anchor.
    """

    return _verify_block_close_transcript_roots_against_registry(
        envelope_value,
        registry_value=load_production_authority_registry(),
        challenge_value=challenge_value,
        runtime_entries=runtime_entries,
        bridge_entries=bridge_entries,
        active_at=active_at,
    )


def _validate_ledger_event(
    value: object, *, expected_sequence: int, previous_event_sha256: str | None
) -> dict[str, Any]:
    event = _require_exact_fields(
        value, _LEDGER_EVENT_FIELDS, context="deployment-authority ledger event"
    )
    if event["schema_version"] != LEDGER_EVENT_SCHEMA_VERSION:
        raise SchemaError("deployment-authority ledger schema differs")
    if event["record_type"] != LEDGER_EVENT_RECORD_TYPE:
        raise SchemaError("deployment-authority ledger record type differs")
    if event["claim_scope"] != LOCAL_LEDGER_CLAIM_SCOPE:
        raise SchemaError("deployment-authority ledger claim scope differs")
    if event["sequence"] != expected_sequence:
        raise SchemaError("deployment-authority ledger sequence is not contiguous")
    _parse_utc(event["recorded_at_utc"], field="recorded_at_utc")
    if event["previous_event_sha256"] != previous_event_sha256:
        raise SchemaError("deployment-authority ledger hash chain differs")
    event_type = event["event_type"]
    payload = event["payload"]
    if event_type == CHALLENGE_ISSUED_EVENT:
        issued = _require_exact_fields(
            payload,
            _ISSUED_EVENT_PAYLOAD_FIELDS,
            context="challenge-issued ledger payload",
        )
        challenge = validate_challenge(issued["challenge"])
        if issued["challenge_sha256"] != sha256_json(challenge):
            raise SchemaError("challenge-issued ledger payload hash differs")
    elif event_type == RECEIPT_CONSUMED_EVENT:
        consumed = _require_exact_fields(
            payload,
            _CONSUMED_EVENT_PAYLOAD_FIELDS,
            context="receipt-consumed ledger payload",
        )
        challenge = validate_challenge(consumed["challenge"])
        envelope = _validate_envelope(consumed["envelope"])
        for field in (
            "challenge_sha256",
            "payload_sha256",
            "receipt_sha256",
            "handoff_manifest_sha256",
            "runner_attestation_sha256",
            "semantic_dependency_lock_sha256",
            "dgx_host_identity_sha256",
            "dgx_runtime_identity_sha256",
            "bridge_identity_sha256",
        ):
            _sha256(consumed[field], field=field)
        if consumed["challenge_sha256"] != sha256_json(challenge):
            raise SchemaError("receipt-consumed challenge hash differs")
        if consumed["payload_sha256"] != envelope["payload_sha256"]:
            raise SchemaError("receipt-consumed payload hash differs")
        if consumed["receipt_sha256"] != sha256_json(envelope):
            raise SchemaError("receipt-consumed envelope hash differs")
        signed_payload = envelope["payload"]
        if consumed["phase"] not in AUTHORITY_PHASES:
            raise SchemaError("receipt-consumed phase is unknown")
        _text(consumed["campaign_id"], field="campaign_id", identifier=True)
        _text(consumed["authority_id"], field="authority_id", identifier=True)
        _text(consumed["key_id"], field="key_id", identifier=True)
        _text(
            consumed["dgx_service_session_id"],
            field="dgx_service_session_id",
            identifier=True,
        )
        _integer(consumed["authority_sequence"], field="authority_sequence", minimum=1)
        if consumed["phase"] in {BLOCK_OPEN_PHASE, BLOCK_CLOSE_PHASE}:
            _text(consumed["block_id"], field="block_id", identifier=True)
            _integer(consumed["rerun_id"], field="rerun_id")
        elif consumed["block_id"] is not None or consumed["rerun_id"] is not None:
            raise SchemaError("startup/model-load consumption cannot name a block")
        for field in (
            "phase",
            "campaign_id",
            "block_id",
            "rerun_id",
            "authority_id",
            "key_id",
            "authority_sequence",
            "handoff_manifest_sha256",
            "runner_attestation_sha256",
            "semantic_dependency_lock_sha256",
            "dgx_service_session_id",
            "dgx_host_identity_sha256",
            "dgx_runtime_identity_sha256",
            "bridge_identity_sha256",
        ):
            if consumed[field] != signed_payload[field]:
                raise SchemaError(
                    f"receipt-consumed metadata differs from envelope: {field}"
                )
    else:
        raise SchemaError("deployment-authority ledger event type is unknown")
    unsigned = dict(event)
    supplied_hash = unsigned.pop("event_sha256")
    expected_hash = sha256_json(unsigned)
    if supplied_hash != expected_hash:
        raise SchemaError("deployment-authority ledger event hash differs")
    return event


def _parse_ledger_text(text: str | bytes) -> list[dict[str, Any]]:
    try:
        payload = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    except UnicodeEncodeError as exc:
        raise SchemaError("deployment-authority ledger is not UTF-8") from exc
    if len(payload) > MAX_AUTHORITY_LEDGER_BYTES:
        raise SchemaError("deployment-authority ledger exceeds the registered byte bound")
    rows = _parse_canonical_jsonl_objects(
        payload,
        context="deployment-authority ledger",
        max_events=MAX_AUTHORITY_LEDGER_EVENTS,
    )
    events: list[dict[str, Any]] = []
    previous: str | None = None
    for value in rows:
        event = _validate_ledger_event(
            value,
            expected_sequence=len(events),
            previous_event_sha256=previous,
        )
        previous = event["event_sha256"]
        events.append(event)
    _validate_ledger_semantics(events)
    return events


def _validate_ledger_semantics(events: Sequence[Mapping[str, Any]]) -> None:
    issued_by_hash: dict[str, dict[str, Any]] = {}
    seen_nonces: set[str] = set()
    consumed_challenges: set[str] = set()
    consumed_receipts: set[str] = set()
    last_authority_sequences: dict[tuple[str, str], int] = {}
    latest_consumed_by_campaign: dict[str, dict[str, Any]] = {}
    previous_challenge_sha256: str | None = None
    for event in events:
        payload = event["payload"]
        if event["event_type"] == CHALLENGE_ISSUED_EVENT:
            challenge = payload["challenge"]
            challenge_sha256 = payload["challenge_sha256"]
            if challenge_sha256 in issued_by_hash:
                raise SchemaError("deployment-authority ledger repeats a challenge")
            nonce = challenge["challenge_nonce"]
            if nonce in seen_nonces:
                raise SchemaError("deployment-authority ledger repeats a nonce")
            if challenge["previous_challenge_sha256"] != previous_challenge_sha256:
                raise SchemaError("deployment-authority challenge chain differs")
            issued_by_hash[challenge_sha256] = payload
            seen_nonces.add(nonce)
            previous_challenge_sha256 = challenge_sha256
            continue
        challenge_sha256 = payload["challenge_sha256"]
        receipt_sha256 = payload["receipt_sha256"]
        issued = issued_by_hash.get(challenge_sha256)
        if issued is None:
            raise SchemaError("deployment-authority ledger consumes an absent challenge")
        if challenge_sha256 in consumed_challenges:
            raise SchemaError("deployment-authority ledger consumes a challenge twice")
        if receipt_sha256 in consumed_receipts:
            raise SchemaError("deployment-authority ledger repeats a signed receipt")
        challenge = issued["challenge"]
        if payload["challenge"] != challenge:
            raise SchemaError("consumed receipt does not preserve its issued challenge")
        latest_consumed = latest_consumed_by_campaign.get(challenge["campaign_id"])
        _validate_challenge_predecessor(latest_consumed, challenge)
        signed_payload = payload["envelope"]["payload"]
        if signed_payload["challenge_sha256"] != challenge_sha256:
            raise SchemaError("consumed envelope does not hash-bind its challenge")
        for field in _CHALLENGE_TO_PAYLOAD_FIELDS:
            if signed_payload[field] != challenge[field]:
                raise SchemaError(
                    f"consumed envelope/challenge binding differs: {field}"
                )
        _validate_identity_after(latest_consumed, signed_payload)
        for field in (
            "phase",
            "campaign_id",
            "block_id",
            "rerun_id",
            "authority_id",
            "key_id",
        ):
            if payload[field] != challenge[field]:
                raise SchemaError(
                    f"deployment-authority consumption differs from challenge: {field}"
                )
        key = (payload["authority_id"], payload["key_id"])
        sequence = payload["authority_sequence"]
        if sequence <= last_authority_sequences.get(key, 0):
            raise SchemaError("deployment-authority ledger sequence is stale")
        last_authority_sequences[key] = sequence
        consumed_challenges.add(challenge_sha256)
        consumed_receipts.add(receipt_sha256)
        latest_consumed_by_campaign[payload["campaign_id"]] = dict(payload)


def _validate_identity_continuity(
    consumed_rows: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> None:
    prior = [
        row
        for row in consumed_rows
        if row["campaign_id"] == payload["campaign_id"]
    ]
    _validate_identity_after(prior[-1] if prior else None, payload)


def _validate_identity_after(
    latest: Mapping[str, Any] | None,
    payload: Mapping[str, Any],
) -> None:
    if payload["phase"] == DGX_SERVICE_STARTUP_PHASE:
        if latest is not None:
            raise SchemaError("startup receipt cannot restart an identity chain")
        return
    if latest is None:
        raise SchemaError("receipt identity continuity lacks a startup predecessor")
    for field in (
        "handoff_manifest_sha256",
        "runner_attestation_sha256",
        "semantic_dependency_lock_sha256",
        "dgx_service_session_id",
        "dgx_host_identity_sha256",
        "dgx_runtime_identity_sha256",
        "bridge_identity_sha256",
    ):
        if payload[field] != latest[field]:
            raise SchemaError(f"deployment identity continuity differs: {field}")


@contextmanager
def _locked_ledger(
    path: str | Path, *, allow_create: bool
) -> Iterator[tuple[BinaryIO, list[dict[str, Any]]]]:
    label = "deployment-authority ledger"
    with _opened_secure_parent(path, label=label) as (
        candidate,
        parent_descriptor,
        leaf,
    ):
        parent_metadata = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(parent_metadata.st_mode) & 0o022
        ):
            raise SchemaError(
                "deployment-authority ledger parent must be owner-controlled"
            )
        flags = (
            os.O_RDWR
            | os.O_APPEND
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        created = False
        try:
            initial_path_metadata = os.stat(
                leaf,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            initial_path_metadata = None
        except OSError as exc:
            raise SchemaError("cannot inspect deployment-authority ledger") from exc
        if initial_path_metadata is not None and stat.S_ISLNK(
            initial_path_metadata.st_mode
        ):
            raise SchemaError("deployment-authority ledger must not be a symlink")
        try:
            descriptor = os.open(leaf, flags, dir_fd=parent_descriptor)
        except FileNotFoundError as absent:
            if not allow_create or initial_path_metadata is not None:
                raise SchemaError("cannot open deployment-authority ledger") from absent
            try:
                descriptor = os.open(
                    leaf,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=parent_descriptor,
                )
                created = True
            except OSError as exc:
                raise SchemaError("cannot create deployment-authority ledger") from exc
        except OSError as exc:
            raise SchemaError("cannot open deployment-authority ledger") from exc
        try:
            metadata = _validate_ledger_metadata(os.fstat(descriptor))
            if initial_path_metadata is None and not created:
                raise SchemaError("deployment-authority ledger appeared during open")
            if initial_path_metadata is not None and not _unchanged_file_metadata(
                initial_path_metadata, metadata
            ):
                raise SchemaError("deployment-authority ledger changed during open")
            if created and stat.S_IMODE(metadata.st_mode) != 0o600:
                os.fchmod(descriptor, 0o600)
                metadata = _validate_ledger_metadata(os.fstat(descriptor))
            _bound_leaf_metadata(parent_descriptor, leaf, metadata, label=label)
            if not _same_parent_binding(parent_metadata, os.fstat(parent_descriptor)):
                raise SchemaError("deployment-authority ledger parent changed during open")
            _assert_reopened_path_binding(
                candidate,
                parent_metadata,
                metadata,
                label=label,
            )
            if created:
                os.fsync(parent_descriptor)
            _acquire_bounded_ledger_lock(descriptor)
            locked = _validate_ledger_metadata(os.fstat(descriptor))
            if not _unchanged_file_metadata(metadata, locked):
                raise SchemaError("deployment-authority ledger changed before lock")
            _bound_leaf_metadata(parent_descriptor, leaf, locked, label=label)
            handle = os.fdopen(descriptor, "r+b")
            descriptor = -1
            events: list[dict[str, Any]] | None = None
            try:
                handle.seek(0)
                payload = handle.read(MAX_AUTHORITY_LEDGER_BYTES + 1)
                if len(payload) > MAX_AUTHORITY_LEDGER_BYTES:
                    raise SchemaError(
                        "deployment-authority ledger exceeds the registered byte bound"
                    )
                after_read = _validate_ledger_metadata(os.fstat(handle.fileno()))
                if len(payload) != locked.st_size or not _unchanged_file_metadata(
                    locked, after_read
                ):
                    raise SchemaError(
                        "deployment-authority ledger changed during bounded read"
                    )
                events = _parse_ledger_text(payload)
                yield handle, events
            finally:
                try:
                    if events is not None:
                        try:
                            expected_final_payload = b"".join(
                                canonical_json_bytes(event) + b"\n"
                                for event in events
                            )
                        except (RecursionError, TypeError, ValueError) as exc:
                            raise SchemaError(
                                "deployment-authority ledger final state cannot be canonicalized"
                            ) from exc
                        if len(expected_final_payload) > MAX_AUTHORITY_LEDGER_BYTES:
                            raise SchemaError(
                                "deployment-authority ledger exceeds the registered byte bound"
                            )
                        before_final_read = _validate_ledger_metadata(
                            os.fstat(handle.fileno())
                        )
                        handle.seek(0)
                        observed_final_payload = handle.read(
                            MAX_AUTHORITY_LEDGER_BYTES + 1
                        )
                        after_final_read = _validate_ledger_metadata(
                            os.fstat(handle.fileno())
                        )
                        if not _unchanged_file_metadata(
                            before_final_read, after_final_read
                        ):
                            raise SchemaError(
                                "deployment-authority ledger changed during final read"
                            )
                        if observed_final_payload != expected_final_payload:
                            raise SchemaError(
                                "deployment-authority ledger final content differs"
                            )
                        final_metadata = after_final_read
                    else:
                        final_metadata = _validate_ledger_metadata(
                            os.fstat(handle.fileno())
                        )
                    _bound_leaf_metadata(
                        parent_descriptor,
                        leaf,
                        final_metadata,
                        label=label,
                    )
                    if not _same_parent_binding(
                        parent_metadata, os.fstat(parent_descriptor)
                    ):
                        raise SchemaError(
                            "deployment-authority ledger parent changed while locked"
                        )
                    _assert_reopened_path_binding(
                        candidate,
                        parent_metadata,
                        final_metadata,
                        label=label,
                    )
                    rebound_final = _validate_ledger_metadata(
                        os.fstat(handle.fileno())
                    )
                    if not _unchanged_file_metadata(
                        final_metadata, rebound_final
                    ):
                        raise SchemaError(
                            "deployment-authority ledger changed during path rebound"
                        )
                    _bound_leaf_metadata(
                        parent_descriptor,
                        leaf,
                        rebound_final,
                        label=label,
                    )
                finally:
                    handle.close()
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _validate_ledger_metadata(metadata: os.stat_result) -> os.stat_result:
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise SchemaError(
            "deployment-authority ledger must be a non-hard-linked regular file"
        )
    if metadata.st_uid != os.geteuid():
        raise SchemaError("deployment-authority ledger must be owned by the current user")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise SchemaError(
            "deployment-authority ledger must not be group/world writable"
        )
    if metadata.st_size > MAX_AUTHORITY_LEDGER_BYTES:
        raise SchemaError("deployment-authority ledger exceeds the registered byte bound")
    return metadata


def _acquire_bounded_ledger_lock(descriptor: int) -> None:
    deadline = time.monotonic() + AUTHORITY_LEDGER_LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise SchemaError("cannot lock deployment-authority ledger") from exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SchemaError("deployment-authority ledger lock timed out") from exc
            time.sleep(min(0.05, remaining))


def read_authority_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Read structural ledger integrity without authenticating stored signatures."""

    with _locked_ledger(path, allow_create=False) as (_, events):
        return [dict(event) for event in events]


def _revalidate_authority_events_against_registry(
    events: Sequence[Mapping[str, Any]],
    *,
    registry_value: object,
) -> dict[str, Any]:
    """Cryptographically replay structurally validated in-memory ledger events.

    The caller must hold the ledger lock for any operational use. This helper
    deliberately performs no file access so issuance and consumption can replay
    preserved signatures without recursively acquiring that lock.
    """

    registry = validate_authority_registry(registry_value)
    receipt_hashes: list[str] = []
    for event in events:
        if event["event_type"] != RECEIPT_CONSUMED_EVENT:
            continue
        consumed = event["payload"]
        envelope = consumed["envelope"]
        payload = envelope["payload"]
        verified = _verify_receipt_against_registry(
            envelope,
            registry_value=registry,
            challenge_value=consumed["challenge"],
            # Historical replay checks the receipt at its signed issuance time.
            # Expiry after valid consumption must not erase prior evidence.
            active_at=_parse_utc(payload["issued_at_utc"], field="issued_at_utc"),
        )
        if verified.receipt_sha256 != consumed["receipt_sha256"]:
            raise SchemaError("replayed ledger receipt hash differs")
        receipt_hashes.append(verified.receipt_sha256)
    return {
        "schema_version": "table2-local-authority-ledger-revalidation-v1",
        "claim_scope": LOCAL_LEDGER_CLAIM_SCOPE,
        "registry_sha256": sha256_json(registry),
        "event_count": len(events),
        "consumed_receipt_count": len(receipt_hashes),
        "consumed_receipt_set_sha256": sha256_json(receipt_hashes),
        "external_global_replay_anchor_present": False,
        "production_dispatch_authorized": False,
    }


def _validate_authority_ledger_against_registry(
    path: str | Path,
    *,
    registry_value: object,
) -> dict[str, Any]:
    """Re-verify every preserved envelope; still only a local-ledger claim."""

    registry = validate_authority_registry(registry_value)
    with _locked_ledger(path, allow_create=False) as (_, events):
        return _revalidate_authority_events_against_registry(
            events,
            registry_value=registry,
        )


def validate_production_authority_ledger(path: str | Path) -> dict[str, Any]:
    """Re-verify with the packaged registry; never grants dispatch authority."""

    return _validate_authority_ledger_against_registry(
        path,
        registry_value=load_production_authority_registry(),
    )


def _append_ledger_event(
    handle: BinaryIO,
    events: list[dict[str, Any]],
    *,
    event_type: str,
    payload: Mapping[str, Any],
    recorded_at: datetime,
) -> dict[str, Any]:
    unsigned = {
        "schema_version": LEDGER_EVENT_SCHEMA_VERSION,
        "record_type": LEDGER_EVENT_RECORD_TYPE,
        "claim_scope": LOCAL_LEDGER_CLAIM_SCOPE,
        "sequence": len(events),
        "event_type": event_type,
        "recorded_at_utc": _format_utc(recorded_at),
        "previous_event_sha256": (
            events[-1]["event_sha256"] if events else None
        ),
        "payload": dict(payload),
    }
    event = {**unsigned, "event_sha256": sha256_json(unsigned)}
    _validate_ledger_event(
        event,
        expected_sequence=len(events),
        previous_event_sha256=unsigned["previous_event_sha256"],
    )
    try:
        encoded = canonical_json_bytes(event) + b"\n"
        expected_prefix = b"".join(
            canonical_json_bytes(existing) + b"\n" for existing in events
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise SchemaError(
            "deployment-authority ledger event cannot be canonicalized"
        ) from exc
    if len(encoded) - 1 > MAX_AUTHORITY_JSONL_LINE_BYTES:
        raise SchemaError("deployment-authority ledger event exceeds the line bound")
    if len(events) >= MAX_AUTHORITY_LEDGER_EVENTS:
        raise SchemaError("deployment-authority ledger exceeds the registered event bound")
    before = _validate_ledger_metadata(os.fstat(handle.fileno()))
    handle.seek(0, os.SEEK_END)
    pre_size = handle.tell()
    if pre_size != before.st_size or pre_size != len(expected_prefix):
        raise SchemaError("deployment-authority ledger pre-append size differs")
    if pre_size + len(encoded) > MAX_AUTHORITY_LEDGER_BYTES:
        raise SchemaError("deployment-authority ledger exceeds the registered byte bound")
    handle.seek(0)
    if handle.read(pre_size + 1) != expected_prefix:
        raise SchemaError("deployment-authority ledger pre-append content differs")
    handle.seek(pre_size)
    if handle.read(1) != b"":
        raise SchemaError("deployment-authority ledger pre-append EOF differs")
    handle.seek(0, os.SEEK_END)
    written = handle.write(encoded)
    if written != len(encoded):
        handle.flush()
        os.ftruncate(handle.fileno(), pre_size)
        os.fsync(handle.fileno())
        raise SchemaError("deployment-authority ledger append was short")
    handle.flush()
    os.fsync(handle.fileno())
    expected_final = expected_prefix + encoded
    after = _validate_ledger_metadata(os.fstat(handle.fileno()))
    if after.st_size != len(expected_final):
        raise SchemaError("deployment-authority ledger post-append size differs")
    handle.seek(pre_size)
    if handle.read(len(encoded) + 1) != encoded:
        raise SchemaError("deployment-authority ledger appended range or EOF differs")
    handle.seek(0)
    final_payload = handle.read(len(expected_final) + 1)
    if final_payload != expected_final:
        raise SchemaError("deployment-authority ledger final content differs")
    final_events = _parse_ledger_text(final_payload)
    if (
        len(final_events) != len(events) + 1
        or final_events[:-1] != events
        or final_events[-1] != event
    ):
        raise SchemaError("deployment-authority ledger append count differs")
    final_metadata = _validate_ledger_metadata(os.fstat(handle.fileno()))
    if not _unchanged_file_metadata(after, final_metadata):
        raise SchemaError("deployment-authority ledger changed during final verification")
    events.append(event)
    return event


def _consumed_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(event["payload"])
        for event in events
        if event["event_type"] == RECEIPT_CONSUMED_EVENT
    ]


def _issued_challenges(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(event["payload"])
        for event in events
        if event["event_type"] == CHALLENGE_ISSUED_EVENT
    ]


def _validate_new_challenge_chain(
    events: Sequence[Mapping[str, Any]], challenge: Mapping[str, Any]
) -> None:
    consumed = [
        row
        for row in _consumed_events(events)
        if row["campaign_id"] == challenge["campaign_id"]
    ]
    _validate_challenge_predecessor(consumed[-1] if consumed else None, challenge)


def _validate_challenge_predecessor(
    latest: Mapping[str, Any] | None,
    challenge: Mapping[str, Any],
) -> None:
    previous = challenge["previous_receipt_sha256"]
    phase = challenge["phase"]
    if phase == DGX_SERVICE_STARTUP_PHASE:
        if latest is not None or previous is not None:
            raise SchemaError("startup challenge cannot restart an existing campaign chain")
        return
    if latest is None or latest["receipt_sha256"] != previous:
        raise SchemaError("challenge does not extend the latest consumed campaign receipt")
    if phase == DGX_MODEL_LOAD_PHASE and latest["phase"] != DGX_SERVICE_STARTUP_PHASE:
        raise SchemaError("model-load challenge must follow the startup receipt")
    if phase == BLOCK_OPEN_PHASE and latest["phase"] not in {
        DGX_MODEL_LOAD_PHASE,
        BLOCK_CLOSE_PHASE,
    }:
        raise SchemaError("block-open challenge must follow model-load or prior block-close")
    if phase == BLOCK_CLOSE_PHASE and (
        latest["phase"] != BLOCK_OPEN_PHASE
        or latest["block_id"] != challenge["block_id"]
        or latest["rerun_id"] != challenge["rerun_id"]
    ):
        raise SchemaError("block-close challenge must follow its matching block-open")


def _issue_challenge_against_registry(
    ledger_path: str | Path,
    *,
    registry_value: object,
    authority_id: str,
    key_id: str,
    phase: str,
    campaign_id: str,
    handoff_manifest_sha256: str,
    runner_attestation_sha256: str,
    semantic_dependency_lock_sha256: str,
    block_id: str | None = None,
    rerun_id: int | None = None,
    previous_receipt_sha256: str | None = None,
    ttl_seconds: int = 300,
) -> dict[str, Any]:
    """Issue a challenge that is single-use only within this validated ledger."""

    registry = validate_authority_registry(registry_value)
    authority = _find_authority(
        registry, authority_id=authority_id, key_id=key_id
    )
    if phase not in AUTHORITY_PHASES:
        raise SchemaError("cannot issue a challenge for an unknown phase")
    if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= MAX_CHALLENGE_TTL_SECONDS:
        raise SchemaError("challenge TTL is outside the registered bound")
    role = PHASE_ROLES[phase]
    with _locked_ledger(ledger_path, allow_create=True) as (handle, events):
        # Structural parsing alone is not trusted for operational state. Replay
        # every preserved independent signature while the same lock is held.
        _revalidate_authority_events_against_registry(
            events,
            registry_value=registry,
        )
        # Capture time only after lock acquisition and replay. A caller waiting
        # on contention must not append a challenge whose TTL began beforehand.
        now = _utc_now()
        _validate_authority_permission(
            authority,
            role=role,
            phase=phase,
            topology=SPLIT_DEPLOYMENT_TOPOLOGY,
            profile_id=authority["measurement_profile_id"],
            profile_sha256=authority["measurement_profile_sha256"],
            at=now,
        )
        issued_rows = _issued_challenges(events)
        previous_challenge = (
            issued_rows[-1]["challenge_sha256"] if issued_rows else None
        )
        challenge = {
            "schema_version": CHALLENGE_SCHEMA_VERSION,
            "record_type": CHALLENGE_RECORD_TYPE,
            "registry_sha256": sha256_json(registry),
            "campaign_id": campaign_id,
            "handoff_manifest_sha256": handoff_manifest_sha256,
            "runner_attestation_sha256": runner_attestation_sha256,
            "semantic_dependency_lock_sha256": semantic_dependency_lock_sha256,
            "deployment_topology": SPLIT_DEPLOYMENT_TOPOLOGY,
            "authority_id": authority_id,
            "key_id": key_id,
            "authority_role": role,
            "measurement_profile_id": authority["measurement_profile_id"],
            "measurement_profile_sha256": authority[
                "measurement_profile_sha256"
            ],
            "phase": phase,
            "block_id": block_id,
            "rerun_id": rerun_id,
            "challenge_nonce": secrets.token_hex(32),
            "issued_at_utc": _format_utc(now),
            "expires_at_utc": _format_utc(now + timedelta(seconds=ttl_seconds)),
            "previous_challenge_sha256": previous_challenge,
            "previous_receipt_sha256": previous_receipt_sha256,
        }
        challenge = validate_challenge(challenge, active_at=now)
        _validate_new_challenge_chain(events, challenge)
        challenge_sha256 = sha256_json(challenge)
        consumed_hashes = {
            row["challenge_sha256"] for row in _consumed_events(events)
        }
        for issued in _issued_challenges(events):
            existing = issued["challenge"]
            if existing["challenge_nonce"] == challenge["challenge_nonce"]:
                raise SchemaError("challenge nonce collision/reuse is forbidden")
            if (
                issued["challenge_sha256"] not in consumed_hashes
                and _parse_utc(existing["expires_at_utc"], field="expires_at_utc") >= now
                and existing["campaign_id"] == challenge["campaign_id"]
            ):
                raise SchemaError("an active challenge already exists for this campaign")
        _append_ledger_event(
            handle,
            events,
            event_type=CHALLENGE_ISSUED_EVENT,
            payload={
                "challenge": challenge,
                "challenge_sha256": challenge_sha256,
            },
            recorded_at=now,
        )
        return {
            "claim_scope": LOCAL_LEDGER_CLAIM_SCOPE,
            "external_global_replay_anchor_present": False,
            "production_dispatch_authorized": False,
            "challenge": challenge,
            "challenge_sha256": challenge_sha256,
        }


def issue_production_challenge(
    ledger_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Issue locally through the source registry; never authorize dispatch."""

    return _issue_challenge_against_registry(
        ledger_path,
        registry_value=load_production_authority_registry(),
        **kwargs,
    )


def _consume_receipt_against_registry(
    ledger_path: str | Path,
    *,
    registry_value: object,
    challenge_sha256: str,
    envelope_value: object,
) -> VerifiedDeploymentReceipt:
    """Authenticate and serialize one consumption within the selected ledger."""

    _sha256(challenge_sha256, field="challenge_sha256")
    registry = validate_authority_registry(registry_value)
    with _locked_ledger(ledger_path, allow_create=False) as (handle, events):
        # Consumption may use no phase, identity, sequence, or replay state until
        # every preserved receipt has been cryptographically authenticated.
        _revalidate_authority_events_against_registry(
            events,
            registry_value=registry,
        )
        now = _utc_now()
        matches = [
            row
            for row in _issued_challenges(events)
            if row["challenge_sha256"] == challenge_sha256
        ]
        if len(matches) != 1:
            raise SchemaError("receipt challenge is absent or duplicated in the ledger")
        consumed = _consumed_events(events)
        if any(row["challenge_sha256"] == challenge_sha256 for row in consumed):
            raise SchemaError(
                "receipt challenge has already been consumed in this local ledger"
            )
        envelope = _validate_envelope(envelope_value)
        receipt_sha256 = sha256_json(envelope)
        if any(row["receipt_sha256"] == receipt_sha256 for row in consumed):
            raise SchemaError(
                "signed deployment receipt repeats within this local ledger"
            )
        _validate_new_challenge_chain(events, matches[0]["challenge"])
        verified = _verify_receipt_against_registry(
            envelope,
            registry_value=registry,
            challenge_value=matches[0]["challenge"],
            active_at=now,
        )
        previous_for_key = [
            row
            for row in consumed
            if row["authority_id"] == verified.authority_id
            and row["key_id"] == verified.key_id
        ]
        if previous_for_key and verified.authority_sequence <= previous_for_key[-1][
            "authority_sequence"
        ]:
            raise SchemaError("authority sequence is stale or replayed")
        payload = dict(envelope["payload"])
        _validate_identity_continuity(consumed, payload)
        _append_ledger_event(
            handle,
            events,
            event_type=RECEIPT_CONSUMED_EVENT,
            payload={
                "challenge": matches[0]["challenge"],
                "challenge_sha256": challenge_sha256,
                "envelope": envelope,
                "payload_sha256": verified.payload_sha256,
                "receipt_sha256": verified.receipt_sha256,
                "phase": verified.phase,
                "campaign_id": payload["campaign_id"],
                "block_id": payload["block_id"],
                "rerun_id": payload["rerun_id"],
                "authority_id": verified.authority_id,
                "key_id": verified.key_id,
                "authority_sequence": verified.authority_sequence,
                "handoff_manifest_sha256": payload[
                    "handoff_manifest_sha256"
                ],
                "runner_attestation_sha256": payload[
                    "runner_attestation_sha256"
                ],
                "semantic_dependency_lock_sha256": payload[
                    "semantic_dependency_lock_sha256"
                ],
                "dgx_service_session_id": payload["dgx_service_session_id"],
                "dgx_host_identity_sha256": payload[
                    "dgx_host_identity_sha256"
                ],
                "dgx_runtime_identity_sha256": payload[
                    "dgx_runtime_identity_sha256"
                ],
                "bridge_identity_sha256": payload["bridge_identity_sha256"],
            },
            recorded_at=now,
        )
        return verified


def consume_production_receipt(
    ledger_path: str | Path,
    *,
    challenge_sha256: str,
    envelope_value: object,
) -> VerifiedDeploymentReceipt:
    """Locally consume via the fixed registry; this never authorizes dispatch."""

    return _consume_receipt_against_registry(
        ledger_path,
        registry_value=load_production_authority_registry(),
        challenge_sha256=challenge_sha256,
        envelope_value=envelope_value,
    )
