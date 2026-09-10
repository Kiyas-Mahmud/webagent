"""One-way bridge between the independent verifier and the agent runtime.

Verifier evidence is append-only below ``sealed/``.  The runtime receives only
an opaque token and a stop/no-stop bit.  In particular it never receives task
success, progress, failure labels, recovery correctness, relevance labels, or
oracle explanations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import fcntl
import hashlib
import hmac
import inspect
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any

from .common import (
    SCHEMA_VERSION,
    SchemaError,
    as_mapping,
    canonical_json_bytes,
    safe_relative_path,
    sha256_file,
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


# The process-isolated evaluator child owns this sink implementation for the
# whole episode. Capture the loaded bytes so the broker can prove that its
# child-owned sealed stream belongs to the authenticated source set.
PROCESS_BROKER_IMPORT_SOURCE_SHA256 = sha256_file(Path(__file__).resolve())

SEALED_VERIFIER_STREAM_TARGET_SCHEMA_VERSION = (
    "table2-child-owned-sealed-verifier-stream-target-v1"
)


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
        "reward",
        "done",
        "terminated",
        "truncated",
        "evaluator_reward",
    }
)
_NORMALIZED_FORBIDDEN_RUNTIME_EVIDENCE_KEYS = frozenset(
    "".join(character for character in key.lower() if character.isalnum())
    for key in FORBIDDEN_RUNTIME_EVIDENCE_KEYS
)
# Opaque verifier references and learned causal predictions are admitted only
# through the exact allowlist below; evaluator counts and memory-eligibility
# truth are never globally exempted. Token-aware matching rejects cosmetic
# wrappers without treating words such as ``task_successor`` as hidden truth.
_FORBIDDEN_RUNTIME_IDENTIFIER_PHRASES = (
    ("oracle",),
    ("verifier",),
    ("task", "success"),
    ("task", "progress"),
    ("ground", "truth"),
    ("relevance", "label"),
    ("reference", "action"),
    ("reference", "answer"),
    ("reference", "trajectory"),
    ("expected", "action"),
    ("expected", "target"),
    ("correct", "action"),
    ("correct", "target"),
    ("success", "label"),
    ("failure", "label"),
    ("recovery", "label"),
    ("task", "reward"),
    ("evaluator", "reward"),
    ("verified", "failure"),
    ("verified", "agent", "failure"),
    ("verified", "success"),
    ("verified", "progress"),
    ("recovery", "success"),
    ("failure", "resolved"),
    ("incident", "resolved"),
    ("registered", "progress"),
    ("memory", "relevance"),
    ("relevant", "ids"),
    ("success", "evidence"),
    ("progress", "evidence"),
    ("future", "state"),
    ("future", "screenshot"),
)
_RUNTIME_COSMETIC_WRAPPER_PREFIXES = frozenset(
    {
        "env",
        "environment",
        "episode",
        "evaluator",
        "final",
        "hidden",
        "is",
        "official",
        "oracle",
        "posthoc",
        "reported",
        "sealed",
    }
)
_RUNTIME_COSMETIC_WRAPPER_SUFFIXES = frozenset(
    {
        "annotation",
        "answer",
        "evidence",
        "flag",
        "label",
        "output",
        "result",
        "reward",
        "score",
        "trajectory",
        "truth",
        "value",
    }
)
_NORMALIZED_ALLOWED_RUNTIME_EVIDENCE_KEYS = frozenset(
    {
        "verifiereventid",
        "verifiertokensha256",
        "predictedfailureresolved",
        # Structural pilot assurance flag only. Package validation requires
        # the exact boolean value and its source/deployment cross-bindings;
        # nested content is still recursively inspected.
        "sourceattestedoraclefreevalueoriginevidence",
    }
)
_NORMALIZED_INDIRECT_CONTROL_KEYS = frozenset(
    {
        "attribute",
        "attributename",
        "attributes",
        "attributenames",
        "column",
        "columnname",
        "columns",
        "columnnames",
        "field",
        "fieldname",
        "fields",
        "fieldnames",
        "feature",
        "featurename",
        "features",
        "featurenames",
        "keyname",
        "keynames",
        "label",
        "labelname",
        "labels",
        "labelnames",
        "metric",
        "metricname",
        "metrics",
        "metricnames",
        "property",
        "propertyname",
        "properties",
        "propertynames",
    }
)


def _runtime_identifier_tokens(value: object) -> tuple[str, ...]:
    rendered = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
    return tuple(token.lower() for token in re.findall(r"[A-Za-z0-9]+", rendered))


def _runtime_contains_phrase(
    tokens: tuple[str, ...], phrase: tuple[str, ...]
) -> bool:
    width = len(phrase)
    return any(tokens[index : index + width] == phrase for index in range(len(tokens) - width + 1))


def _runtime_wrapped_compound(normalized: str, compound: str) -> bool:
    start = normalized.find(compound)
    if start < 0:
        return False
    prefix = normalized[:start]
    suffix = normalized[start + len(compound) :]
    return bool(prefix or suffix) and _runtime_segmented_affix(
        prefix, _RUNTIME_COSMETIC_WRAPPER_PREFIXES
    ) and _runtime_segmented_affix(
        suffix, _RUNTIME_COSMETIC_WRAPPER_SUFFIXES
    )


def _runtime_segmented_affix(value: str, vocabulary: frozenset[str]) -> bool:
    if not value:
        return True
    reachable = {0}
    for start in range(len(value)):
        if start not in reachable:
            continue
        for token in vocabulary:
            if value.startswith(token, start):
                reachable.add(start + len(token))
    return len(value) in reachable


def _runtime_is_indirect_control_key(key_token: str) -> bool:
    if key_token in _NORMALIZED_INDIRECT_CONTROL_KEYS:
        return True
    return any(
        key_token.endswith(base)
        and _runtime_segmented_affix(
            key_token[: -len(base)],
            _RUNTIME_COSMETIC_WRAPPER_PREFIXES,
        )
        for base in _NORMALIZED_INDIRECT_CONTROL_KEYS
    )


def _runtime_evidence_key_matches(raw_key: object) -> set[str]:
    normalized = "".join(
        character for character in str(raw_key).lower() if character.isalnum()
    )
    if normalized in _NORMALIZED_ALLOWED_RUNTIME_EVIDENCE_KEYS:
        return set()
    tokens = _runtime_identifier_tokens(raw_key)
    matches: set[str] = set()
    if normalized in _NORMALIZED_FORBIDDEN_RUNTIME_EVIDENCE_KEYS:
        matches.add(normalized)
    for phrase in _FORBIDDEN_RUNTIME_IDENTIFIER_PHRASES:
        compound = "".join(phrase)
        if _runtime_contains_phrase(tokens, phrase) or compound in tokens:
            matches.add(compound)
        elif _runtime_wrapped_compound(normalized, compound):
            matches.add(compound)
    if _runtime_wrapped_compound(normalized, "verification"):
        matches.add("verification")
    for raw_signal in ("done", "reward", "terminated", "truncated"):
        if _runtime_wrapped_compound(normalized, raw_signal):
            matches.add(raw_signal)
    # Causal post-state hashes may be logged. Only evaluator-like wrappers of
    # that phrase are rejected here; the pre-action policy guard is stricter.
    if _runtime_wrapped_compound(normalized, "stateafter"):
        matches.add("stateafter")
    return matches


def _runtime_indirect_identifiers(key_token: str, value: Any) -> tuple[str, ...]:
    if not _runtime_is_indirect_control_key(key_token):
        return ()

    def collect(item: Any) -> list[str]:
        identifiers: list[str] = []
        if isinstance(item, str):
            identifiers.append(item)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                if isinstance(nested, (str, Mapping, list, tuple)):
                    identifiers.extend(collect(nested))
        elif isinstance(item, Mapping):
            for nested_key, nested_value in item.items():
                nested_token = "".join(
                    character
                    for character in str(nested_key).lower()
                    if character.isalnum()
                )
                if nested_token in {"name", "names"} or _runtime_is_indirect_control_key(
                    nested_token
                ):
                    identifiers.extend(collect(nested_value))
                elif isinstance(nested_value, (Mapping, list, tuple)):
                    identifiers.extend(collect(nested_value))
        return identifiers

    return tuple(collect(value))


def _validate_private_regular_file(path: Path, *, context: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SchemaError(f"{context} is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise SchemaError(f"{context} is not a private owner-controlled file")
    return metadata


def _inode_identity(metadata: os.stat_result) -> tuple[int, int]:
    return (metadata.st_dev, metadata.st_ino)


def _directory_open_flags() -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _validate_opened_directory(
    metadata: os.stat_result,
    *,
    context: str,
) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o002
    ):
        raise SchemaError(f"{context} is not an owner-controlled directory")


def _open_directory_below_exact(
    anchor: Path,
    directory: Path,
    *,
    create: bool,
    context: str,
) -> int:
    """Open a directory below ``anchor`` without following path components."""

    try:
        relative = directory.relative_to(anchor)
    except ValueError as exc:
        raise SchemaError(f"{context} escaped its trusted directory") from exc
    if not anchor.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise SchemaError(f"{context} is not a canonical relative directory")

    flags = _directory_open_flags()
    try:
        anchor_before = anchor.lstat()
        current_fd = os.open(anchor, flags)
    except OSError as exc:
        raise SchemaError(f"{context} anchor cannot be opened safely") from exc
    try:
        anchor_opened = os.fstat(current_fd)
        anchor_after = anchor.lstat()
        _validate_opened_directory(anchor_opened, context=f"{context} anchor")
        if not (
            _inode_identity(anchor_before)
            == _inode_identity(anchor_opened)
            == _inode_identity(anchor_after)
        ):
            raise SchemaError(f"{context} anchor identity changed")

        for component in relative.parts:
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise SchemaError(
                        f"{context} component cannot be created safely"
                    ) from exc
            try:
                entry_before = os.stat(
                    component,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as exc:
                raise SchemaError(
                    f"{context} component cannot be opened safely"
                ) from exc
            try:
                opened = os.fstat(next_fd)
                entry_after = os.stat(
                    component,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
                _validate_opened_directory(opened, context=context)
                if not (
                    _inode_identity(entry_before)
                    == _inode_identity(opened)
                    == _inode_identity(entry_after)
                ):
                    raise SchemaError(f"{context} component identity changed")
            except BaseException:
                os.close(next_fd)
                raise
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _open_optional_directory_at_exact(
    parent_fd: int,
    name: str,
    *,
    context: str,
) -> int | None:
    """Open one existing child directory, distinguishing only true absence."""

    flags = _directory_open_flags()
    try:
        entry_before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SchemaError(f"{context} cannot be inspected safely") from exc
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise SchemaError(f"{context} cannot be opened safely") from exc
    try:
        opened = os.fstat(fd)
        entry_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        _validate_opened_directory(opened, context=context)
        if not (
            _inode_identity(entry_before)
            == _inode_identity(opened)
            == _inode_identity(entry_after)
        ):
            raise SchemaError(f"{context} identity changed")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_private_regular_file_at_exact(
    parent_fd: int,
    name: str,
    *,
    context: str,
    maximum_bytes: int = 4096,
    missing_ok: bool = False,
) -> bytes | None:
    """Read one leaf relative to an already authenticated directory inode."""

    try:
        entry_before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise SchemaError(f"{context} is unavailable") from None
    except OSError as exc:
        raise SchemaError(f"{context} cannot be inspected safely") from exc
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise SchemaError(f"{context} cannot be opened safely") from exc
    try:
        opened_before = os.fstat(fd)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_nlink != 1
            or opened_before.st_uid != os.getuid()
            or opened_before.st_mode & 0o077
            or opened_before.st_size <= 0
            or opened_before.st_size > maximum_bytes
        ):
            raise SchemaError(f"{context} is not a bounded private regular file")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 4096))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        opened_after = os.fstat(fd)
        entry_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise SchemaError(f"{context} changed while being read") from exc
    finally:
        os.close(fd)
    if not (
        _inode_identity(entry_before)
        == _inode_identity(opened_before)
        == _inode_identity(opened_after)
        == _inode_identity(entry_after)
    ) or len(value) != opened_after.st_size:
        raise SchemaError(f"{context} identity changed while being read")
    return value


def _read_private_regular_file_exact(
    path: Path,
    *,
    context: str,
    maximum_bytes: int = 4096,
) -> bytes:
    """Read one private file without following or retaining a swapped leaf."""

    parent_before = path.parent.lstat()
    if not stat.S_ISDIR(parent_before.st_mode) or stat.S_ISLNK(
        parent_before.st_mode
    ):
        raise SchemaError(f"{context} parent is not a real directory")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SchemaError(f"{context} cannot be opened safely") from exc
    try:
        opened_before = os.fstat(fd)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_nlink != 1
            or opened_before.st_uid != os.getuid()
            or opened_before.st_mode & 0o077
            or opened_before.st_size <= 0
            or opened_before.st_size > maximum_bytes
        ):
            raise SchemaError(f"{context} is not a bounded private regular file")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 4096))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        opened_after = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        path_after = path.lstat()
        parent_after = path.parent.lstat()
    except OSError as exc:
        raise SchemaError(f"{context} changed while being read") from exc
    if (
        _inode_identity(parent_before) != _inode_identity(parent_after)
        or _inode_identity(opened_before) != _inode_identity(opened_after)
        or _inode_identity(opened_after) != _inode_identity(path_after)
        or len(value) != opened_after.st_size
    ):
        raise SchemaError(f"{context} identity changed while being read")
    return value


def _append_sealed_record_exact(
    path: Path,
    record: Mapping[str, Any],
    *,
    anchor: Path,
    expected_event_count: int,
    seal_key: bytes,
) -> None:
    """Durably append one complete record and verify the resulting stream."""

    parent_fd = _open_directory_below_exact(
        anchor,
        path.parent,
        create=True,
        context="sealed verifier stream directory tree",
    )
    flags = os.O_APPEND | os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        entry_before: os.stat_result | None
        try:
            entry_before = os.stat(
                path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            entry_before = None
        fd = os.open(path.name, flags, 0o600, dir_fd=parent_fd)
    except OSError as exc:
        os.close(parent_fd)
        raise SchemaError("sealed verifier stream cannot be opened safely") from exc
    try:
        metadata = os.fstat(fd)
        entry_opened = os.stat(
            path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
            or _inode_identity(metadata) != _inode_identity(entry_opened)
            or (
                entry_before is not None
                and _inode_identity(entry_before) != _inode_identity(metadata)
            )
        ):
            raise SchemaError(
                "sealed verifier stream is not an owner-controlled regular file"
            )
        payload = canonical_json_bytes(record) + b"\n"
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise SchemaError("sealed verifier stream append was incomplete")
            view = view[written:]
        os.fchmod(fd, 0o600)
        os.fsync(fd)
        metadata_after = os.fstat(fd)
        entry_after = os.stat(
            path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            metadata_after.st_nlink != 1
            or _inode_identity(metadata) != _inode_identity(metadata_after)
            or _inode_identity(metadata_after) != _inode_identity(entry_after)
        ):
            raise SchemaError("sealed verifier stream identity changed during append")
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = 128 * 1024 * 1024 + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(fd)
        os.close(parent_fd)
    records = _verify_sealed_stream_bytes(path, raw, seal_key)
    if len(records) != expected_event_count or records[-1] != dict(record):
        raise SchemaError("sealed verifier stream append verification failed")


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
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        self.system_root = (
            self.campaign_dir
            / "paired_blocks"
            / f"seed_{resolved_seed}"
            / _path_id(resolved_task)
            / f"repeat_{resolved_repeat}"
            / f"rerun_{int(attempt_id)}"
            / _path_id(system_id)
        )
        self.sealed_root = self.system_root / "sealed"
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
        sealed_fd = _open_directory_below_exact(
            self.campaign_dir,
            self.sealed_root,
            create=True,
            context="sealed verifier directory tree",
        )
        try:
            episode_fd = _open_optional_directory_at_exact(
                sealed_fd,
                self.path.parent.name,
                context="sealed verifier episode directory",
            )
            if episode_fd is None:
                initial_stream = None
            else:
                try:
                    initial_stream = _read_private_regular_file_at_exact(
                        episode_fd,
                        self.path.name,
                        context="sealed verifier stream",
                        maximum_bytes=128 * 1024 * 1024,
                        missing_ok=True,
                    )
                finally:
                    os.close(episode_fd)
        finally:
            os.close(sealed_fd)
        self._key = seal_key or self._load_or_create_key()
        self._previous_hash = "0" * 64
        self._next_index = 0
        self._write_disabled = False
        self._child_ownership_fd: int | None = None
        self._child_ownership_directory_fd: int | None = None
        if initial_stream is not None:
            existing = _verify_sealed_stream_bytes(
                self.path,
                initial_stream,
                self._key,
            )
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

        if self._write_disabled:
            raise SchemaError(
                "sealed verifier write ownership was transferred to the child"
            )
        evidence = as_mapping(verification)
        if type(event_kind) is not str or not event_kind.strip():
            raise SchemaError("sealed verifier event_kind must be nonempty text")
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
        _append_sealed_record_exact(
            self.path,
            record,
            anchor=self.campaign_dir,
            expected_event_count=self._next_index + 1,
            seal_key=self._key,
        )
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
        return any(
            record.get("event_kind") == "episode_final"
            for record in self.verified_records()
        )

    def verified_records(self) -> list[dict[str, Any]]:
        """Read and authenticate this stream through anchored descriptors."""

        raw = self._read_existing_stream_exact()
        if raw is None:
            return []
        return _verify_sealed_stream_bytes(self.path, raw, self._key)

    def _read_existing_stream_exact(self) -> bytes | None:
        sealed_fd = _open_directory_below_exact(
            self.campaign_dir,
            self.sealed_root,
            create=False,
            context="sealed verifier directory tree",
        )
        try:
            episode_fd = _open_optional_directory_at_exact(
                sealed_fd,
                self.path.parent.name,
                context="sealed verifier episode directory",
            )
            if episode_fd is None:
                return None
            try:
                return _read_private_regular_file_at_exact(
                    episode_fd,
                    self.path.name,
                    context="sealed verifier stream",
                    maximum_bytes=128 * 1024 * 1024,
                    missing_ok=True,
                )
            finally:
                os.close(episode_fd)
        finally:
            os.close(sealed_fd)

    def _load_or_create_key(self) -> bytes:
        key_path = self.sealed_root / ".seal_key"
        sealed_fd = _open_directory_below_exact(
            self.campaign_dir,
            self.sealed_root,
            create=False,
            context="sealed verifier directory tree",
        )
        try:
            existing = _read_private_regular_file_at_exact(
                sealed_fd,
                key_path.name,
                context="sealed verifier key",
                missing_ok=True,
            )
            if existing is not None:
                if len(existing) < 32:
                    raise SchemaError("existing verifier seal key is too short")
                return existing
            key = secrets.token_bytes(32)
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
            )
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                fd = os.open(key_path.name, flags, 0o600, dir_fd=sealed_fd)
            except FileExistsError:
                raced = _read_private_regular_file_at_exact(
                    sealed_fd,
                    key_path.name,
                    context="sealed verifier key",
                )
                if raced is None or len(raced) < 32:
                    raise SchemaError("existing verifier seal key is too short")
                return raced
            except OSError as exc:
                raise SchemaError("sealed verifier key cannot be created safely") from exc
            try:
                opened = os.fstat(fd)
                entry = os.stat(
                    key_path.name,
                    dir_fd=sealed_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or opened.st_uid != os.getuid()
                    or opened.st_mode & 0o077
                    or _inode_identity(opened) != _inode_identity(entry)
                ):
                    raise SchemaError(
                        "sealed verifier key is not an owner-controlled regular file"
                    )
                view = memoryview(key)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise SchemaError("sealed verifier key write was incomplete")
                    view = view[written:]
                os.fchmod(fd, 0o600)
                os.fsync(fd)
                opened_after = os.fstat(fd)
                entry_after = os.stat(
                    key_path.name,
                    dir_fd=sealed_fd,
                    follow_symlinks=False,
                )
                if not (
                    _inode_identity(opened)
                    == _inode_identity(opened_after)
                    == _inode_identity(entry_after)
                ) or opened_after.st_nlink != 1:
                    raise SchemaError("sealed verifier key identity changed during write")
            finally:
                os.close(fd)
            return key
        finally:
            os.close(sealed_fd)

    def release_child_ownership(self) -> None:
        """Release an isolated child's exclusive stream claim during cleanup."""

        fd = self._child_ownership_fd
        self._child_ownership_fd = None
        if fd is not None:
            os.close(fd)
        directory_fd = self._child_ownership_directory_fd
        self._child_ownership_directory_fd = None
        if directory_fd is not None:
            os.close(directory_fd)


# Public architecture name; retain ``SealedVerifierSink`` for compatibility
# with existing integrations that use the implementation-oriented name.
SealedVerifier = SealedVerifierSink


@dataclass(frozen=True, slots=True)
class SealedVerifierStreamTarget:
    """Secret-free locator for a child-owned sealed verifier stream.

    Campaign orchestration constructs this record from its readable sink and
    passes only the target to the process-isolated episode factory.  It binds
    the exact output identity and key *digest* but deliberately contains no
    seal-key bytes and no writer callback.
    """

    schema_version: str
    campaign_dir: str
    block_id: str
    attempt_id: int
    system_id: str
    episode_id: str
    matched_seed: int
    task_id: str
    repeat_id: int
    seal_key_sha256: str
    initial_event_count: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != SEALED_VERIFIER_STREAM_TARGET_SCHEMA_VERSION:
            raise SchemaError("sealed verifier stream target version differs")
        campaign = Path(self.campaign_dir)
        if (
            not campaign.is_absolute()
            or not campaign.is_dir()
            or campaign.is_symlink()
            or campaign.resolve(strict=True) != campaign
        ):
            raise SchemaError(
                "sealed verifier stream target campaign directory is not canonical"
            )
        for name in ("block_id", "system_id", "episode_id", "task_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise SchemaError(f"sealed verifier stream target {name} is invalid")
        if self.system_id not in {"E0", "E1", "E2", "E3"}:
            raise SchemaError("sealed verifier stream target system is invalid")
        for name in ("attempt_id", "matched_seed", "repeat_id"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SchemaError(
                    f"sealed verifier stream target {name} is invalid"
                )
        _require_sha256(
            self.seal_key_sha256,
            context="sealed verifier stream target key",
        )
        if self.initial_event_count != 0:
            raise SchemaError(
                "child-owned sealed verifier target must begin with an empty stream"
            )

    @classmethod
    def from_sink(cls, sink: SealedVerifierSink) -> "SealedVerifierStreamTarget":
        if type(sink) is not SealedVerifierSink:
            raise TypeError("sealed verifier stream target requires the exact sink")
        if sink._write_disabled:  # noqa: SLF001 - exact owner handoff
            raise SchemaError("sealed verifier stream ownership was already transferred")
        existing = sink._read_existing_stream_exact()  # noqa: SLF001
        if existing is not None or sink._next_index != 0:  # noqa: SLF001
            raise SchemaError(
                "child-owned sealed verifier target stream is not empty"
            )
        key_path = sink.sealed_root / ".seal_key"
        sealed_fd = _open_directory_below_exact(
            sink.campaign_dir,
            sink.sealed_root,
            create=False,
            context="sealed verifier stream target directory tree",
        )
        try:
            key = _read_private_regular_file_at_exact(
                sealed_fd,
                key_path.name,
                context="sealed verifier stream target key",
            )
        finally:
            os.close(sealed_fd)
        if key is None:  # pragma: no cover - missing_ok is deliberately false
            raise SchemaError("sealed verifier stream target key is unavailable")
        target = cls(
            schema_version=SEALED_VERIFIER_STREAM_TARGET_SCHEMA_VERSION,
            campaign_dir=str(sink.campaign_dir),
            block_id=sink.block_id,
            attempt_id=sink.attempt_id,
            system_id=sink.system_id,
            episode_id=sink.episode_id,
            matched_seed=sink.matched_seed,
            task_id=sink.task_id,
            repeat_id=sink.repeat_id,
            seal_key_sha256=hashlib.sha256(key).hexdigest(),
            initial_event_count=0,
        )
        # The parent retains read/verification access but can no longer append
        # through this stale sink or a writer derived from it.
        sink._write_disabled = True  # noqa: SLF001 - exact owner handoff
        return target

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | Any) -> "SealedVerifierStreamTarget":
        raw = as_mapping(value)
        expected = {
            "schema_version",
            "campaign_dir",
            "block_id",
            "attempt_id",
            "system_id",
            "episode_id",
            "matched_seed",
            "task_id",
            "repeat_id",
            "seal_key_sha256",
            "initial_event_count",
        }
        if set(raw) != expected:
            raise SchemaError("sealed verifier stream target fields differ")
        return cls(**{name: raw[name] for name in expected})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_dir": self.campaign_dir,
            "block_id": self.block_id,
            "attempt_id": self.attempt_id,
            "system_id": self.system_id,
            "episode_id": self.episode_id,
            "matched_seed": self.matched_seed,
            "task_id": self.task_id,
            "repeat_id": self.repeat_id,
            "seal_key_sha256": self.seal_key_sha256,
            "initial_event_count": self.initial_event_count,
        }

    @property
    def record_sha256(self) -> str:
        return sha256_json(self.to_dict())

    def open_child_sink(self, *, episode_runtime_dir: str | Path) -> SealedVerifierSink:
        """Open the exact empty stream from the isolated evaluator process."""

        runtime = Path(episode_runtime_dir)
        if (
            not runtime.is_absolute()
            or not runtime.is_dir()
            or runtime.is_symlink()
            or runtime.resolve(strict=True) != runtime
        ):
            raise SchemaError(
                "child-owned sealed verifier runtime directory is not canonical"
            )
        system_root = (
            Path(self.campaign_dir)
            / "paired_blocks"
            / f"seed_{self.matched_seed}"
            / _path_id(self.task_id)
            / f"repeat_{self.repeat_id}"
            / f"rerun_{self.attempt_id}"
            / _path_id(self.system_id)
        )
        if runtime != system_root / "runtime":
            raise SchemaError(
                "child-owned sealed verifier target differs from runtime directory"
            )
        campaign = Path(self.campaign_dir)
        sealed_root = system_root / "sealed"
        key_path = sealed_root / ".seal_key"
        lock_path = sealed_root / ".child_owner.lock"
        ownership_directory_fd = _open_directory_below_exact(
            campaign,
            sealed_root,
            create=False,
            context="child-owned sealed verifier directory tree",
        )
        ownership_fd = -1
        try:
            try:
                # The directory inode is the durable ownership authority.  A
                # same-user unlink/recreate of the marker cannot manufacture a
                # second directory flock while the first child remains alive.
                fcntl.flock(
                    ownership_directory_fd,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except OSError as exc:
                raise SchemaError(
                    "child-owned sealed verifier stream already has an owner"
                ) from exc

            lock_flags = (
                os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            )
            if hasattr(os, "O_NOFOLLOW"):
                lock_flags |= os.O_NOFOLLOW
            try:
                try:
                    lock_before = os.stat(
                        lock_path.name,
                        dir_fd=ownership_directory_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    lock_before = None
                ownership_fd = os.open(
                    lock_path.name,
                    lock_flags,
                    0o600,
                    dir_fd=ownership_directory_fd,
                )
            except OSError as exc:
                raise SchemaError(
                    "child-owned sealed verifier ownership lock cannot be opened"
                ) from exc
            metadata = os.fstat(ownership_fd)
            lock_opened = os.stat(
                lock_path.name,
                dir_fd=ownership_directory_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.getuid()
                or metadata.st_mode & 0o077
                or _inode_identity(metadata) != _inode_identity(lock_opened)
                or (
                    lock_before is not None
                    and _inode_identity(lock_before) != _inode_identity(metadata)
                )
            ):
                raise SchemaError(
                    "child-owned sealed verifier ownership lock is unsafe"
                )
            fcntl.flock(ownership_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            metadata_after_lock = os.fstat(ownership_fd)
            lock_after = os.stat(
                lock_path.name,
                dir_fd=ownership_directory_fd,
                follow_symlinks=False,
            )
            if not (
                _inode_identity(metadata)
                == _inode_identity(metadata_after_lock)
                == _inode_identity(lock_after)
            ) or metadata_after_lock.st_nlink != 1:
                raise SchemaError(
                    "child-owned sealed verifier ownership lock changed"
                )

            key = _read_private_regular_file_at_exact(
                ownership_directory_fd,
                key_path.name,
                context="child-owned sealed verifier key",
            )
            if key is None:  # pragma: no cover - missing_ok is deliberately false
                raise SchemaError("child-owned sealed verifier key is unavailable")
            if hashlib.sha256(key).hexdigest() != self.seal_key_sha256:
                raise SchemaError("child-owned sealed verifier key digest differs")
            sink = SealedVerifierSink(
                self.campaign_dir,
                block_id=self.block_id,
                attempt_id=self.attempt_id,
                system_id=self.system_id,
                episode_id=self.episode_id,
                matched_seed=self.matched_seed,
                task_id=self.task_id,
                repeat_id=self.repeat_id,
                seal_key=key,
            )
            if sink.system_root != system_root:
                raise SchemaError("child-owned sealed verifier system root differs")
            if sink.sealed_root / ".seal_key" != key_path:
                raise SchemaError("child-owned sealed verifier key path differs")
            if sink._next_index != self.initial_event_count:  # noqa: SLF001
                raise SchemaError(
                    "child-owned sealed verifier stream is not empty"
                )
            sink._child_ownership_fd = ownership_fd  # noqa: SLF001
            sink._child_ownership_directory_fd = ownership_directory_fd  # noqa: SLF001
            ownership_fd = -1
            ownership_directory_fd = -1
            return sink
        finally:
            if ownership_fd >= 0:
                os.close(ownership_fd)
            if ownership_directory_fd >= 0:
                os.close(ownership_directory_fd)


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


def _verify_sealed_stream_bytes(
    source: Path,
    raw: bytes,
    key: bytes,
) -> list[dict[str, Any]]:
    """Verify authenticated stream bytes without reopening a mutable pathname."""

    if len(key) < 32:
        raise SchemaError("sealed stream HMAC key is too short")
    if not raw.endswith(b"\n"):
        raise SchemaError(f"sealed stream has an incomplete final record: {source}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise SchemaError(
                f"sealed stream {source}:{line_number} has an empty record"
            )
        try:
            record = json.loads(
                line.decode("utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-standard JSON constant: {token}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise SchemaError(
                f"sealed stream {source}:{line_number} is not JSON"
            ) from exc
        try:
            canonical = canonical_json_bytes(record)
        except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise SchemaError(
                f"sealed stream {source}:{line_number} is not canonical JSON"
            ) from exc
        if not isinstance(record, dict) or canonical != line:
            raise SchemaError(
                f"sealed stream {source}:{line_number} is not canonical JSON"
            )
        records.append(record)
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
        fields = set(record)
        if fields != required:
            raise SchemaError(
                f"sealed stream {source}:{index + 1} fields differ: "
                f"missing={sorted(required - fields)}, "
                f"unknown={sorted(fields - required)}"
            )
        text_fields = (
            "schema_version",
            "event_id",
            "event_kind",
            "block_id",
            "system_id",
            "episode_id",
            "previous_record_hash",
            "opaque_token_sha256",
            "record_hash",
        )
        for field in text_fields:
            if type(record[field]) is not str or not record[field].strip():
                raise SchemaError(
                    f"sealed stream {source}:{index + 1} has invalid {field}"
                )
        if record["schema_version"] != SCHEMA_VERSION:
            raise SchemaError(f"sealed stream {source}:{index + 1} has wrong schema")
        if type(record["attempt_id"]) is not int or record["attempt_id"] < 0:
            raise SchemaError(
                f"sealed stream {source}:{index + 1} has invalid attempt_id"
            )
        if type(record["should_terminate"]) is not bool:
            raise SchemaError(
                f"sealed stream {source}:{index + 1} has non-boolean termination flag"
            )
        if type(record["evidence"]) is not dict:
            raise SchemaError(
                f"sealed stream {source}:{index + 1} evidence is not an object"
            )
        if type(record["event_index"]) is not int or record["event_index"] < 0:
            raise SchemaError(
                f"sealed stream {source}:{index + 1} has invalid event index"
            )
        if record["event_index"] != index:
            raise SchemaError(f"sealed stream {source} has non-monotonic event index")
        for field in (
            "previous_record_hash",
            "opaque_token_sha256",
            "record_hash",
        ):
            _require_sha256(
                record[field],
                context=f"sealed stream {source}:{index + 1} {field}",
            )
        if record["previous_record_hash"] != previous_hash:
            raise SchemaError(f"sealed stream {source} broke its previous-record chain")
        body = {
            key: value
            for key, value in record.items()
            if key not in {"record_hash", "opaque_token_sha256"}
        }
        expected = sha256_json(body)
        if not hmac.compare_digest(record["record_hash"], expected):
            raise SchemaError(
                f"sealed stream {source}:{index + 1} has a bad record hash"
            )
        expected_token = hmac.new(
            key,
            canonical_json_bytes([record["event_id"], expected]),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(
            record["opaque_token_sha256"], expected_token
        ):
            raise SchemaError(
                f"sealed stream {source}:{index + 1} has a bad opaque token"
            )
        previous_hash = expected
    return records


def _verify_sealed_stream_with_key(
    source: Path,
    key: bytes,
) -> list[dict[str, Any]]:
    """Verify one safely opened stream against already authenticated key bytes."""

    raw = _read_private_regular_file_exact(
        source,
        context="sealed verifier stream",
        maximum_bytes=128 * 1024 * 1024,
    )
    return _verify_sealed_stream_bytes(source, raw, key)


def verify_sealed_stream(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    key_path = source.parent.parent / ".seal_key"
    key = _read_private_regular_file_exact(
        key_path,
        context="sealed stream HMAC key",
    )
    return _verify_sealed_stream_with_key(source, key)


def assert_no_verifier_evidence(record: Mapping[str, Any], *, context: str) -> None:
    """Recursively fail if an unsealed runtime record contains oracle evidence."""

    def walk(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not str(key).isascii():
                    raise SchemaError(
                        f"{context}{location}.{key} uses a non-ASCII control-plane key"
                    )
                normalized = "".join(
                    character
                    for character in str(key).lower()
                    if character.isalnum()
                )
                if normalized in _NORMALIZED_ALLOWED_RUNTIME_EVIDENCE_KEYS:
                    walk(item, f"{location}.{key}")
                    continue
                matches = _runtime_evidence_key_matches(key)
                for identifier in _runtime_indirect_identifiers(normalized, item):
                    if not identifier.isascii():
                        matches.add("nonasciiidentifier")
                    else:
                        matches.update(_runtime_evidence_key_matches(identifier))
                if matches:
                    raise SchemaError(
                        f"{context}{location}.{key} leaks sealed verifier keys: "
                        f"{sorted(matches)}"
                    )
                walk(item, f"{location}.{key}")
        elif isinstance(value, (list, tuple)):
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
