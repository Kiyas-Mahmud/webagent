"""One-way bridge between the independent verifier and the agent runtime.

Verifier evidence is append-only below ``sealed/``.  The runtime receives only
an opaque token and a stop/no-stop bit.  In particular it never receives task
success, progress, failure labels, recovery correctness, relevance labels, or
oracle explanations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import inspect
import os
from pathlib import Path
import secrets
from typing import Any

from .common import (
    SCHEMA_VERSION,
    SchemaError,
    append_jsonl,
    as_mapping,
    canonical_json_bytes,
    read_jsonl,
    safe_relative_path,
    sha256_json,
)

try:  # Runtime contracts can land before or after this evaluation package.
    from web_agent.runtime.contracts import OpaqueTerminalSignal as _RuntimeSignal
except (ImportError, AttributeError):  # pragma: no cover - exercised during integration
    _RuntimeSignal = None


@dataclass(frozen=True)
class _FallbackOpaqueTerminalSignal:
    """Dependency-free mirror of the runtime's deliberately tiny contract."""

    event_id: str
    token_sha256: str
    terminate: bool


OpaqueTerminalSignal = _RuntimeSignal or _FallbackOpaqueTerminalSignal


FORBIDDEN_RUNTIME_EVIDENCE_KEYS = frozenset(
    {
        "task_success",
        "task_progress",
        "oracle_success",
        "oracle_failure",
        "oracle_progress",
        "oracle_result",
        "verified_failure",
        "verified_agent_failure",
        "failure_ground_truth",
        "incident_resolved",
        "registered_progress",
        "recovery_success",
        "recovery_verifications",
        "memory_relevance",
        "relevance_labels",
        "relevance_label",
        "relevant_ids",
        "reference_trajectory",
        "reference_action",
        "reference_answer",
        "oracle_evidence",
        "ground_truth",
        "success_evidence",
        "progress_evidence",
    }
)
_NORMALIZED_FORBIDDEN_RUNTIME_EVIDENCE_KEYS = frozenset(
    "".join(character for character in key.lower() if character.isalnum())
    for key in FORBIDDEN_RUNTIME_EVIDENCE_KEYS
)


class SealedVerifierSink:
    """Append full verifier evidence and return only an opaque terminal signal."""

    def __init__(
        self,
        campaign_dir: str | Path,
        *,
        block_id: str,
        attempt_id: int,
        system_id: str,
        episode_id: str,
        matched_seed: int | None = None,
        task_id: str | None = None,
        repeat_id: int | None = None,
        seal_key: bytes | None = None,
    ) -> None:
        self.campaign_dir = Path(campaign_dir).resolve()
        parsed = _parse_registered_block_id(block_id)
        resolved_seed = int(matched_seed if matched_seed is not None else parsed["matched_seed"])
        resolved_task = str(task_id if task_id is not None else parsed["task_id"])
        resolved_repeat = int(repeat_id if repeat_id is not None else parsed["repeat_id"])
        if resolved_seed != int(parsed["matched_seed"]):
            raise SchemaError("sealed verifier seed disagrees with registered block ID")
        # The registered block ID uses the same filesystem-safe encoding as
        # the artifact tree (for example ``webarena.0`` -> ``webarena-0``).
        # Compare encoded identities; the explicit raw task ID remains the
        # authoritative value stored in sealed evidence.
        if _path_id(resolved_task) != str(parsed["task_id"]):
            raise SchemaError("sealed verifier task disagrees with registered block ID")
        if resolved_repeat != int(parsed["repeat_id"]):
            raise SchemaError("sealed verifier repeat disagrees with registered block ID")
        if system_id not in {"E0", "E1", "E2", "E3"}:
            raise SchemaError(f"sealed verifier has unknown system ID: {system_id!r}")
        if int(attempt_id) < 0:
            raise SchemaError("sealed verifier attempt ID cannot be negative")
        self.system_root = (
            self.campaign_dir
            / "paired_blocks"
            / f"seed_{resolved_seed}"
            / _path_id(resolved_task)
            / f"repeat_{resolved_repeat}"
            / f"rerun_{int(attempt_id)}"
            / _path_id(system_id)
        ).resolve()
        self.sealed_root = (self.system_root / "sealed").resolve()
        if self.campaign_dir not in self.sealed_root.parents:
            raise SchemaError("sealed verifier root escaped the campaign directory")
        relative = safe_relative_path(Path(_path_id(episode_id)) / "verifier_events.jsonl")
        self.path = self.sealed_root / relative
        self.block_id = str(block_id)
        self.attempt_id = int(attempt_id)
        self.system_id = str(system_id)
        self.episode_id = str(episode_id)
        self.matched_seed = resolved_seed
        self.task_id = resolved_task
        self.repeat_id = resolved_repeat
        self._key = seal_key or self._load_or_create_key()
        self._previous_hash = "0" * 64
        self._next_index = 0
        if self.path.exists():
            existing = verify_sealed_stream(self.path)
            if existing:
                self._previous_hash = str(existing[-1]["record_hash"])
                self._next_index = int(existing[-1]["event_index"]) + 1

    def record(
        self,
        verification: Mapping[str, Any] | Any,
        *,
        should_terminate: bool,
        event_kind: str = "transition_verification",
    ) -> Any:
        """Seal one verifier result; never return its evidence to the runtime."""

        evidence = as_mapping(verification)
        if not event_kind:
            raise SchemaError("sealed verifier event_kind is required")
        if type(should_terminate) is not bool:
            raise SchemaError("sealed verifier should_terminate must be an exact boolean")
        event_id = f"{self.episode_id}:verifier:{self._next_index:06d}"
        body = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "event_index": self._next_index,
            "event_kind": event_kind,
            "block_id": self.block_id,
            "attempt_id": self.attempt_id,
            "system_id": self.system_id,
            "episode_id": self.episode_id,
            "should_terminate": should_terminate,
            "evidence": evidence,
            "previous_record_hash": self._previous_hash,
        }
        record_hash = sha256_json(body)
        token = hmac.new(
            self._key,
            canonical_json_bytes([event_id, record_hash]),
            hashlib.sha256,
        ).hexdigest()
        record = {**body, "opaque_token_sha256": token, "record_hash": record_hash}
        append_jsonl(self.path, record, mode=0o600)
        self._previous_hash = record_hash
        self._next_index += 1
        return _make_runtime_signal(
            should_terminate=should_terminate,
            opaque_token=token,
            verifier_event_id=event_id,
        )

    def record_bound_receipt(
        self,
        binding: Mapping[str, Any] | Any,
        verification: Mapping[str, Any] | Any,
        *,
        should_terminate: bool,
    ) -> Any:
        """Seal one evaluator result against an exact causal runtime artifact.

        This is the canonical transition API.  It prevents a benchmark wrapper
        from emitting a plausible final record without proving that the sealed
        evaluator ran after reset and after each requested browser action.
        Only the opaque signal returned by :meth:`record` crosses back into the
        runtime decision path.
        """

        raw_binding = as_mapping(binding)
        if raw_binding.get("schema_version") not in {None, SCHEMA_VERSION}:
            raise SchemaError("verifier receipt binding has the wrong schema")
        if raw_binding.get("record_type") not in {None, "VerifierReceiptBinding"}:
            raise SchemaError("verifier receipt binding has the wrong record type")
        allowed = {
            "schema_version",
            "record_type",
            "receipt_kind",
            "observation_id",
            "observation_sha256",
            "action_id",
            "action_sha256",
        }
        unknown = set(raw_binding) - allowed
        if unknown:
            raise SchemaError(
                f"verifier receipt binding contains unknown fields: {sorted(unknown)}"
            )
        normalized = {
            key: raw_binding.get(key)
            for key in (
                "receipt_kind",
                "observation_id",
                "observation_sha256",
                "action_id",
                "action_sha256",
            )
        }
        kind = str(normalized["receipt_kind"] or "")
        if kind not in {
            "after_reset",
            "after_normal_action",
            "after_recovery_action",
        }:
            raise SchemaError(f"unknown bound verifier receipt kind: {kind!r}")
        if not str(normalized["observation_id"] or "").strip():
            raise SchemaError("bound verifier receipt lacks observation_id")
        _require_sha256(
            normalized["observation_sha256"],
            context="bound verifier observation",
        )
        if kind == "after_reset":
            if normalized["action_id"] is not None or normalized["action_sha256"] is not None:
                raise SchemaError("reset verifier receipt cannot cite an action")
        else:
            if not str(normalized["action_id"] or "").strip():
                raise SchemaError("action verifier receipt lacks action_id")
            _require_sha256(
                normalized["action_sha256"],
                context="bound verifier action",
            )
        evidence = as_mapping(verification)
        if "runtime_binding" in evidence:
            raise SchemaError("verifier evidence cannot override runtime_binding")
        return self.record(
            {**evidence, "runtime_binding": normalized},
            should_terminate=should_terminate,
            event_kind=kind,
        )

    def record_episode_final(self, verification: Mapping[str, Any] | Any) -> Any:
        """Seal the single final analysis record required for a valid episode."""

        evidence = as_mapping(verification)
        required = {
            "task_success",
            "terminal_reason",
            "loop_detected",
            "environment_failure",
            "failure_incidents",
            "recovery_verifications",
            "verified_failure_event_count",
            "repeated_error_event_count",
            "memory_relevance",
        }
        missing = sorted(required - set(evidence))
        if missing:
            raise SchemaError(f"episode_final evidence is missing: {', '.join(missing)}")
        if self.has_episode_final:
            raise SchemaError("episode_final may be sealed exactly once")
        for key in ("task_success", "loop_detected", "environment_failure"):
            if type(evidence[key]) is not bool:
                raise SchemaError(f"episode_final {key} must be an exact boolean")
        if not isinstance(evidence["terminal_reason"], str) or not evidence[
            "terminal_reason"
        ].strip():
            raise SchemaError("episode_final terminal_reason must be nonempty text")
        for key in ("failure_incidents", "recovery_verifications"):
            value = evidence[key]
            if not isinstance(value, list) or not all(
                isinstance(item, Mapping) for item in value
            ):
                raise SchemaError(f"episode_final {key} must be an object list")
        for key in ("verified_failure_event_count", "repeated_error_event_count"):
            if type(evidence[key]) is not int or evidence[key] < 0:
                raise SchemaError(f"episode_final {key} must be a nonnegative integer")
        if not isinstance(evidence["memory_relevance"], Mapping):
            raise SchemaError("episode_final memory_relevance must be an object")
        return self.record(evidence, should_terminate=True, event_kind="episode_final")

    @property
    def has_episode_final(self) -> bool:
        if not self.path.exists():
            return False
        return any(
            record.get("event_kind") == "episode_final"
            for record in verify_sealed_stream(self.path)
        )

    def _load_or_create_key(self) -> bytes:
        key_path = self.sealed_root / ".seal_key"
        key_path.parent.mkdir(parents=True, exist_ok=True)
        if key_path.exists():
            key = key_path.read_bytes()
            if len(key) < 32:
                raise SchemaError("existing verifier seal key is too short")
            return key
        key = secrets.token_bytes(32)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(key_path, flags, 0o600)
        try:
            os.write(fd, key)
            os.fsync(fd)
        finally:
            os.close(fd)
        return key


# Public architecture name; retain ``SealedVerifierSink`` for compatibility
# with existing integrations that use the implementation-oriented name.
SealedVerifier = SealedVerifierSink


class SealedVerifierWriter:
    """Narrow write-only capability exposed to an evaluator integration.

    The evaluator receives neither the campaign root, sealed path, HMAC key,
    nor a method for reading prior records.  Runtime orchestration retains the
    full :class:`SealedVerifierSink` and performs all postconditions itself.
    This is a capability boundary for frozen, source-attested integration code;
    it is intentionally not a general Python sandbox.
    """

    __slots__ = ("__record_bound", "__record_final")

    def __init__(self, sink: SealedVerifierSink) -> None:
        if type(sink) is not SealedVerifierSink:
            raise TypeError("sealed verifier writer requires the exact sink type")
        object.__setattr__(self, "_SealedVerifierWriter__record_bound", sink.record_bound_receipt)
        object.__setattr__(self, "_SealedVerifierWriter__record_final", sink.record_episode_final)

    def __getattribute__(self, name: str) -> Any:
        if name in {
            "record_bound_receipt",
            "record_episode_final",
            "__class__",
            "__doc__",
        }:
            return object.__getattribute__(self, name)
        raise AttributeError("sealed verifier writer exposes write methods only")

    def record_bound_receipt(
        self,
        binding: Mapping[str, Any] | Any,
        verification: Mapping[str, Any] | Any,
        *,
        should_terminate: bool,
    ) -> Any:
        callback = object.__getattribute__(
            self, "_SealedVerifierWriter__record_bound"
        )
        return callback(
            binding,
            verification,
            should_terminate=should_terminate,
        )

    def record_episode_final(self, verification: Mapping[str, Any] | Any) -> Any:
        callback = object.__getattribute__(
            self, "_SealedVerifierWriter__record_final"
        )
        return callback(verification)


def verify_sealed_stream(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    key_path = source.parent.parent / ".seal_key"
    if not key_path.is_file():
        raise SchemaError(f"sealed stream has no HMAC key: {key_path}")
    key = key_path.read_bytes()
    if len(key) < 32:
        raise SchemaError(f"sealed stream HMAC key is too short: {key_path}")
    if key_path.stat().st_mode & 0o077 or source.stat().st_mode & 0o077:
        raise SchemaError(f"sealed verifier key/evidence permissions are not private: {source}")
    records = read_jsonl(source)
    previous_hash = "0" * 64
    for index, record in enumerate(records):
        required = {
            "schema_version",
            "event_id",
            "event_index",
            "event_kind",
            "block_id",
            "attempt_id",
            "system_id",
            "episode_id",
            "should_terminate",
            "evidence",
            "previous_record_hash",
            "opaque_token_sha256",
            "record_hash",
        }
        missing = sorted(required - set(record))
        if missing:
            raise SchemaError(f"sealed stream {path}:{index + 1} missing {missing}")
        if record["schema_version"] != SCHEMA_VERSION:
            raise SchemaError(f"sealed stream {path}:{index + 1} has wrong schema")
        if type(record["should_terminate"]) is not bool:
            raise SchemaError(
                f"sealed stream {path}:{index + 1} has non-boolean termination flag"
            )
        if int(record["event_index"]) != index:
            raise SchemaError(f"sealed stream {path} has non-monotonic event index")
        if record["previous_record_hash"] != previous_hash:
            raise SchemaError(f"sealed stream {path} broke its previous-record chain")
        body = {
            key: value
            for key, value in record.items()
            if key not in {"record_hash", "opaque_token_sha256"}
        }
        expected = sha256_json(body)
        if not hmac.compare_digest(str(record["record_hash"]), expected):
            raise SchemaError(f"sealed stream {path}:{index + 1} has a bad record hash")
        expected_token = hmac.new(
            key,
            canonical_json_bytes([record["event_id"], expected]),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(
            str(record["opaque_token_sha256"]), expected_token
        ):
            raise SchemaError(f"sealed stream {path}:{index + 1} has a bad opaque token")
        previous_hash = expected
    return records


def assert_no_verifier_evidence(record: Mapping[str, Any], *, context: str) -> None:
    """Recursively fail if an unsealed runtime record contains oracle evidence."""

    def walk(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            normalized = {
                "".join(character for character in str(key).lower() if character.isalnum())
                for key in value
            }
            overlap = _NORMALIZED_FORBIDDEN_RUNTIME_EVIDENCE_KEYS.intersection(normalized)
            if overlap:
                raise SchemaError(
                    f"{context}{location} leaks sealed verifier keys: {sorted(overlap)}"
                )
            for key, item in value.items():
                walk(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{location}[{index}]")

    walk(record, "")


def _make_runtime_signal(**values: Any) -> Any:
    signal_type = OpaqueTerminalSignal
    runtime_values = {
        **values,
        "event_id": values["verifier_event_id"],
        "token_sha256": values["opaque_token"],
        "terminate": values["should_terminate"],
    }
    try:
        signature = inspect.signature(signal_type)
        accepted = {
            name: value
            for name, value in runtime_values.items()
            if name in signature.parameters
        }
        if accepted:
            return signal_type(**accepted)
    except (TypeError, ValueError):
        pass
    return _FallbackOpaqueTerminalSignal(
        event_id=runtime_values["event_id"],
        token_sha256=runtime_values["token_sha256"],
        terminate=runtime_values["terminate"],
    )


def _require_sha256(value: Any, *, context: str) -> None:
    text = str(value or "")
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise SchemaError(f"{context} hash must be lowercase hexadecimal SHA-256")


def _path_id(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in str(value))
    cleaned = cleaned.strip("-")
    if not cleaned:
        raise SchemaError(f"unsafe empty verifier path ID: {value!r}")
    return cleaned[:160]


def _parse_registered_block_id(block_id: str) -> dict[str, Any]:
    """Parse IDs emitted by :func:`build_paired_schedule` as a compatibility aid."""

    value = str(block_id)
    marker_seed = "__seed_"
    marker_repeat = "__repeat_"
    if not value.startswith("task_") or marker_seed not in value or marker_repeat not in value:
        raise SchemaError(
            "matched_seed, task_id, and repeat_id are required when block_id is not "
            "the registered task_<id>__seed_<n>__repeat_<n> form"
        )
    task_part, tail = value[len("task_") :].rsplit(marker_seed, 1)
    seed_part, repeat_part = tail.split(marker_repeat, 1)
    try:
        return {
            "task_id": task_part,
            "matched_seed": int(seed_part),
            "repeat_id": int(repeat_part),
        }
    except ValueError as exc:
        raise SchemaError(f"invalid registered block ID: {block_id}") from exc
