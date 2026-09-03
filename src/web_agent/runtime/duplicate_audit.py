"""Frozen duplicate-audit evidence required before every E3 query.

The pilot registration can state that evidence is still pending, but pending
rows are never accepted by the runtime.  Synthetic diagnostic rows are kept in
their own evidence class and cannot be reused for ordinary WebArena episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from web_agent.runtime.contracts import VersionedRecord, canonical_sha256


DUPLICATE_AUDIT_SCHEMA_VERSION = "table2-duplicate-audit-v1"
DUPLICATE_CLUSTER_NAMESPACE_SCHEMA_VERSION = (
    "table2-joint-duplicate-cluster-namespace-v1"
)
DUPLICATE_AUDIT_RECORD_SCHEMA_VERSION = "table2-duplicate-audit-task-record-v1"


class DuplicateAuditError(RuntimeError):
    """Duplicate evidence is absent, incomplete, unverified, or mismatched."""


class DuplicateAuditStatus(str, Enum):
    VERIFIED = "VERIFIED"
    PENDING_EXTERNAL_AUDIT = "PENDING_EXTERNAL_AUDIT"
    SYNTHETIC_DIAGNOSTIC_ONLY = "SYNTHETIC_DIAGNOSTIC_ONLY"


@dataclass(frozen=True, slots=True)
class JointDuplicateClusterNamespace:
    """One exact clustering identity shared by train memory and WebArena.

    A pending tracked registration may omit both implementation hashes.  Such
    a registration is useful for smoke/configuration validation but is never
    evaluation-ready.  Frozen memory and VERIFIED task evidence must provide
    both hashes.
    """

    namespace_id: str
    audit_tool_id: str
    audit_tool_version: str
    audit_tool_config_sha256: str | None
    audit_tool_source_sha256: str | None

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        require_hashes: bool,
    ) -> "JointDuplicateClusterNamespace":
        if not isinstance(value, Mapping):
            raise DuplicateAuditError("duplicate cluster namespace must be an object")
        expected = {
            "schema_version",
            "namespace_id",
            "audit_tool_id",
            "audit_tool_version",
            "audit_tool_config_sha256",
            "audit_tool_source_sha256",
        }
        unknown = set(value) - expected
        missing = expected - set(value)
        if unknown or missing:
            raise DuplicateAuditError(
                "duplicate cluster namespace fields mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if value.get("schema_version") != DUPLICATE_CLUSTER_NAMESPACE_SCHEMA_VERSION:
            raise DuplicateAuditError("duplicate cluster namespace schema mismatch")
        text = {
            field: str(value.get(field) or "").strip()
            for field in ("namespace_id", "audit_tool_id", "audit_tool_version")
        }
        if any(not item for item in text.values()):
            raise DuplicateAuditError(
                "duplicate cluster namespace identity/tool/version are required"
            )
        config_hash = _optional_sha256(
            value.get("audit_tool_config_sha256"),
            field="duplicate_cluster_namespace.audit_tool_config_sha256",
        )
        source_hash = _optional_sha256(
            value.get("audit_tool_source_sha256"),
            field="duplicate_cluster_namespace.audit_tool_source_sha256",
        )
        if (config_hash is None) != (source_hash is None):
            raise DuplicateAuditError(
                "duplicate cluster namespace config/source hashes must be supplied together"
            )
        if require_hashes and (config_hash is None or source_hash is None):
            raise DuplicateAuditError(
                "evaluation duplicate cluster namespace needs config/source hashes"
            )
        return cls(
            namespace_id=text["namespace_id"],
            audit_tool_id=text["audit_tool_id"],
            audit_tool_version=text["audit_tool_version"],
            audit_tool_config_sha256=config_hash,
            audit_tool_source_sha256=source_hash,
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "schema_version": DUPLICATE_CLUSTER_NAMESPACE_SCHEMA_VERSION,
            "namespace_id": self.namespace_id,
            "audit_tool_id": self.audit_tool_id,
            "audit_tool_version": self.audit_tool_version,
            "audit_tool_config_sha256": self.audit_tool_config_sha256,
            "audit_tool_source_sha256": self.audit_tool_source_sha256,
        }

    @property
    def is_evaluation_ready(self) -> bool:
        return (
            self.audit_tool_config_sha256 is not None
            and self.audit_tool_source_sha256 is not None
        )


@dataclass(frozen=True, slots=True)
class DuplicateAuditEntry(VersionedRecord):
    task_id: str
    task_partition: str
    status: DuplicateAuditStatus
    cluster_ids: tuple[str, ...]
    content_sha256: str | None
    train_corpus_manifest_sha256: str | None
    audit_tool_id: str
    audit_tool_version: str
    evidence_sha256: str | None
    cluster_namespace_id: str | None = None
    audit_tool_config_sha256: str | None = None
    audit_tool_source_sha256: str | None = None
    audit_record: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.task_partition.strip():
            raise ValueError("duplicate audit task/partition IDs are required")
        if not self.audit_tool_id.strip() or not self.audit_tool_version.strip():
            raise ValueError("duplicate audit tool identity/version are required")
        if len(self.cluster_ids) != len(set(self.cluster_ids)) or any(
            not str(item).strip() for item in self.cluster_ids
        ):
            raise ValueError("duplicate audit cluster IDs must be unique/non-empty")
        hashes = (
            self.content_sha256,
            self.train_corpus_manifest_sha256,
            self.evidence_sha256,
            self.audit_tool_config_sha256,
            self.audit_tool_source_sha256,
        )
        if any(value is not None and not _is_sha256(value) for value in hashes):
            raise ValueError("duplicate audit hashes must be 64-digit hexadecimal")
        if self.status is DuplicateAuditStatus.PENDING_EXTERNAL_AUDIT:
            evidence_hashes = (
                self.content_sha256,
                self.train_corpus_manifest_sha256,
                self.evidence_sha256,
            )
            if self.cluster_ids or any(value is not None for value in evidence_hashes):
                raise ValueError("pending duplicate audit rows cannot claim evidence")
            if self.audit_record is not None:
                raise ValueError("pending duplicate audit rows cannot claim an audit record")
        elif self.status is DuplicateAuditStatus.VERIFIED:
            if not self.cluster_ids or any(value is None for value in hashes):
                raise ValueError("verified duplicate audit rows need complete evidence")
            if self.task_partition == "recovery_diagnostic":
                raise ValueError("synthetic recovery rows cannot claim VERIFIED status")
            if not isinstance(self.audit_record, Mapping):
                raise ValueError("verified duplicate audit rows need an audit record")
        elif self.status is DuplicateAuditStatus.SYNTHETIC_DIAGNOSTIC_ONLY:
            if self.task_partition != "recovery_diagnostic":
                raise ValueError("synthetic audit evidence is diagnostic-only")
            if not self.cluster_ids or self.evidence_sha256 is None:
                raise ValueError("synthetic diagnostic evidence must be hash-bound")
            if self.content_sha256 is None:
                raise ValueError("synthetic diagnostic content must be hash-bound")
            if self.train_corpus_manifest_sha256 is not None:
                raise ValueError("synthetic evidence cannot claim a train-corpus audit")

    @property
    def binding_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


class FrozenDuplicateAuditManifest:
    """Read-only, coverage-checked duplicate-audit registry."""

    __slots__ = (
        "manifest_id",
        "manifest_sha256",
        "duplicate_cluster_namespace",
        "_entries",
    )

    def __init__(
        self,
        *,
        manifest_id: str,
        entries: Sequence[DuplicateAuditEntry],
        manifest_sha256: str,
        duplicate_cluster_namespace: JointDuplicateClusterNamespace | None = None,
    ) -> None:
        if not manifest_id.strip() or not _is_sha256(manifest_sha256):
            raise ValueError("duplicate audit manifest identity/hash are invalid")
        indexed = {entry.task_id: entry for entry in entries}
        if len(indexed) != len(entries):
            raise ValueError("duplicate audit manifest has duplicate task IDs")
        if not indexed:
            raise ValueError("duplicate audit manifest cannot be empty")
        normal_entries = tuple(
            entry for entry in entries if entry.task_partition == "normal"
        )
        if normal_entries and duplicate_cluster_namespace is None:
            raise DuplicateAuditError(
                "normal duplicate-audit rows require a joint cluster namespace"
            )
        if duplicate_cluster_namespace is not None:
            for entry in normal_entries:
                _validate_entry_namespace(entry, duplicate_cluster_namespace)
                if entry.status is DuplicateAuditStatus.VERIFIED:
                    _validate_verified_task_audit_record(
                        entry,
                        duplicate_cluster_namespace,
                    )
        self.manifest_id = manifest_id
        self.manifest_sha256 = manifest_sha256
        self.duplicate_cluster_namespace = duplicate_cluster_namespace
        self._entries = indexed

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        expected_task_ids: Sequence[str] | None = None,
    ) -> "FrozenDuplicateAuditManifest":
        source = Path(path)
        raw_bytes = source.read_bytes()
        try:
            payload = json.loads(raw_bytes)
        except json.JSONDecodeError as exc:
            raise DuplicateAuditError(f"invalid duplicate audit JSON: {source}") from exc
        if not isinstance(payload, Mapping):
            raise DuplicateAuditError("duplicate audit manifest root must be an object")
        if payload.get("schema_version") != DUPLICATE_AUDIT_SCHEMA_VERSION:
            raise DuplicateAuditError("duplicate audit schema version mismatch")
        if payload.get("manifest_state") != "FROZEN_REGISTRATION":
            raise DuplicateAuditError("duplicate audit manifest is not frozen")
        rows = payload.get("entries")
        if not isinstance(rows, list):
            raise DuplicateAuditError("duplicate audit entries must be an array")
        entries = tuple(_entry_from_mapping(row, index) for index, row in enumerate(rows))
        has_normal_rows = any(entry.task_partition == "normal" for entry in entries)
        raw_namespace = payload.get("duplicate_cluster_namespace")
        namespace = (
            JointDuplicateClusterNamespace.from_mapping(
                raw_namespace,
                require_hashes=False,
            )
            if raw_namespace is not None
            else None
        )
        if has_normal_rows and namespace is None:
            raise DuplicateAuditError(
                "normal duplicate-audit manifest lacks duplicate_cluster_namespace"
            )
        result = cls(
            manifest_id=str(payload.get("manifest_id") or ""),
            entries=entries,
            manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            duplicate_cluster_namespace=namespace,
        )
        if expected_task_ids is not None:
            result.require_exact_coverage(expected_task_ids)
        return result

    @classmethod
    def synthetic_diagnostic(
        cls,
        *,
        task_id: str,
        cluster_ids: Sequence[str],
        content_identity: Mapping[str, Any],
    ) -> "FrozenDuplicateAuditManifest":
        content_sha256 = canonical_sha256(content_identity)
        evidence_sha256 = canonical_sha256(
            {
                "scope": "synthetic_diagnostic_only",
                "task_id": task_id,
                "cluster_ids": list(cluster_ids),
                "content_sha256": content_sha256,
            }
        )
        entry = DuplicateAuditEntry(
            task_id=task_id,
            task_partition="recovery_diagnostic",
            status=DuplicateAuditStatus.SYNTHETIC_DIAGNOSTIC_ONLY,
            cluster_ids=tuple(str(item) for item in cluster_ids),
            content_sha256=content_sha256,
            train_corpus_manifest_sha256=None,
            audit_tool_id="deterministic-fixture-registration",
            audit_tool_version="v1",
            evidence_sha256=evidence_sha256,
        )
        manifest_hash = canonical_sha256(
            {"manifest_id": f"synthetic-{task_id}", "entries": [entry.to_dict()]}
        )
        return cls(
            manifest_id=f"synthetic-{task_id}",
            entries=(entry,),
            manifest_sha256=manifest_hash,
        )

    def require_exact_coverage(self, expected_task_ids: Sequence[str]) -> None:
        expected = tuple(str(item) for item in expected_task_ids)
        if len(expected) != len(set(expected)):
            raise DuplicateAuditError("expected duplicate-audit task IDs are not unique")
        missing = set(expected) - set(self._entries)
        extra = set(self._entries) - set(expected)
        if missing or extra:
            raise DuplicateAuditError(
                "duplicate audit coverage mismatch: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

    def require_evaluation_ready(self, expected_task_ids: Sequence[str]) -> None:
        self.require_exact_coverage(expected_task_ids)
        invalid = {
            task_id: self._entries[task_id].status.value
            for task_id in expected_task_ids
            if self._entries[task_id].status is not DuplicateAuditStatus.VERIFIED
        }
        if invalid:
            raise DuplicateAuditError(
                "primary evaluation duplicate audit is not verified: "
                + ", ".join(f"{key}={value}" for key, value in sorted(invalid.items()))
            )
        if (
            self.duplicate_cluster_namespace is None
            or not self.duplicate_cluster_namespace.is_evaluation_ready
        ):
            raise DuplicateAuditError(
                "evaluation duplicate cluster namespace is not fully hash-bound"
            )

    def clusters_for(
        self,
        task_id: str,
        *,
        task_partition: str,
        allow_synthetic_diagnostic: bool = False,
    ) -> tuple[str, ...]:
        try:
            entry = self._entries[task_id]
        except KeyError as exc:
            raise DuplicateAuditError(f"duplicate audit has no task {task_id!r}") from exc
        if entry.task_partition != task_partition:
            raise DuplicateAuditError("duplicate audit task partition mismatch")
        if entry.status is DuplicateAuditStatus.PENDING_EXTERNAL_AUDIT:
            raise DuplicateAuditError(
                f"duplicate audit remains pending for task {task_id!r}"
            )
        if entry.status is DuplicateAuditStatus.SYNTHETIC_DIAGNOSTIC_ONLY:
            if not allow_synthetic_diagnostic or task_partition != "recovery_diagnostic":
                raise DuplicateAuditError(
                    "synthetic duplicate evidence cannot enter ordinary evaluation"
                )
        elif entry.status is not DuplicateAuditStatus.VERIFIED:
            raise DuplicateAuditError("unregistered duplicate audit status")
        return entry.cluster_ids

    def entry_binding_sha256(self, task_id: str) -> str:
        try:
            return self._entries[task_id].binding_sha256
        except KeyError as exc:
            raise DuplicateAuditError(f"duplicate audit has no task {task_id!r}") from exc

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)


def validate_pilot_duplicate_audit_manifest(
    path: str | Path,
    *,
    normal_task_ids: Sequence[str],
    recovery_scenario_ids: Sequence[str],
    require_normal_verified: bool = False,
) -> FrozenDuplicateAuditManifest:
    """Validate exact 50+15 registration coverage and evidence separation."""

    if len(normal_task_ids) != 50 or len(recovery_scenario_ids) != 15:
        raise DuplicateAuditError("pilot duplicate audit requires exactly 50+15 IDs")
    all_ids = tuple(normal_task_ids) + tuple(recovery_scenario_ids)
    manifest = FrozenDuplicateAuditManifest.from_path(path, expected_task_ids=all_ids)
    for scenario_id in recovery_scenario_ids:
        manifest.clusters_for(
            scenario_id,
            task_partition="recovery_diagnostic",
            allow_synthetic_diagnostic=True,
        )
    if require_normal_verified:
        for task_id in normal_task_ids:
            manifest.clusters_for(task_id, task_partition="normal")
    return manifest


def _entry_from_mapping(value: Any, index: int) -> DuplicateAuditEntry:
    if not isinstance(value, Mapping):
        raise DuplicateAuditError(f"duplicate audit entry[{index}] must be an object")
    expected = {
        "task_id",
        "task_partition",
        "status",
        "cluster_ids",
        "content_sha256",
        "train_corpus_manifest_sha256",
        "audit_tool_id",
        "audit_tool_version",
        "evidence_sha256",
    }
    if value.get("task_partition") == "normal":
        expected.update(
            {
                "cluster_namespace_id",
                "audit_tool_config_sha256",
                "audit_tool_source_sha256",
                "audit_record",
            }
        )
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise DuplicateAuditError(
            f"duplicate audit entry[{index}] fields mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    clusters = value["cluster_ids"]
    if not isinstance(clusters, list):
        raise DuplicateAuditError(f"duplicate audit entry[{index}].cluster_ids must be array")
    try:
        status = DuplicateAuditStatus(value["status"])
    except ValueError as exc:
        raise DuplicateAuditError(
            f"duplicate audit entry[{index}] has unknown status"
        ) from exc
    return DuplicateAuditEntry(
        task_id=str(value["task_id"]),
        task_partition=str(value["task_partition"]),
        status=status,
        cluster_ids=tuple(str(item) for item in clusters),
        content_sha256=_optional_string(value["content_sha256"]),
        train_corpus_manifest_sha256=_optional_string(
            value["train_corpus_manifest_sha256"]
        ),
        audit_tool_id=str(value["audit_tool_id"]),
        audit_tool_version=str(value["audit_tool_version"]),
        evidence_sha256=_optional_string(value["evidence_sha256"]),
        cluster_namespace_id=_optional_string(value.get("cluster_namespace_id")),
        audit_tool_config_sha256=_optional_string(
            value.get("audit_tool_config_sha256")
        ),
        audit_tool_source_sha256=_optional_string(
            value.get("audit_tool_source_sha256")
        ),
        audit_record=value.get("audit_record"),
    )


def canonical_task_audit_record(
    *,
    task_id: str,
    content_sha256: str,
    train_corpus_manifest_sha256: str,
    cluster_ids: Sequence[str],
    namespace: JointDuplicateClusterNamespace,
) -> dict[str, Any]:
    """Return the only record shape accepted for normal VERIFIED evidence."""

    if not namespace.is_evaluation_ready:
        raise DuplicateAuditError("task audit record needs a fully hash-bound namespace")
    content_hash = _required_sha256(content_sha256, field="audit_record.content_sha256")
    corpus_hash = _required_sha256(
        train_corpus_manifest_sha256,
        field="audit_record.train_corpus_manifest_sha256",
    )
    normalized_clusters = tuple(str(item).strip() for item in cluster_ids)
    if (
        not normalized_clusters
        or any(not item for item in normalized_clusters)
        or len(normalized_clusters) != len(set(normalized_clusters))
        or normalized_clusters != tuple(sorted(normalized_clusters))
    ):
        raise DuplicateAuditError(
            "task audit record cluster IDs must be non-empty, unique, and sorted"
        )
    normalized_task_id = str(task_id).strip()
    if not normalized_task_id:
        raise DuplicateAuditError("task audit record task_id is required")
    return {
        "schema_version": DUPLICATE_AUDIT_RECORD_SCHEMA_VERSION,
        "task_id": normalized_task_id,
        "content_sha256": content_hash,
        "train_corpus_manifest_sha256": corpus_hash,
        "cluster_namespace_id": namespace.namespace_id,
        "cluster_ids": list(normalized_clusters),
        "audit_tool_id": namespace.audit_tool_id,
        "audit_tool_version": namespace.audit_tool_version,
        "audit_tool_config_sha256": namespace.audit_tool_config_sha256,
        "audit_tool_source_sha256": namespace.audit_tool_source_sha256,
    }


def _validate_entry_namespace(
    entry: DuplicateAuditEntry,
    namespace: JointDuplicateClusterNamespace,
) -> None:
    expected = {
        "cluster_namespace_id": namespace.namespace_id,
        "audit_tool_id": namespace.audit_tool_id,
        "audit_tool_version": namespace.audit_tool_version,
        "audit_tool_config_sha256": namespace.audit_tool_config_sha256,
        "audit_tool_source_sha256": namespace.audit_tool_source_sha256,
    }
    actual = {
        "cluster_namespace_id": entry.cluster_namespace_id,
        "audit_tool_id": entry.audit_tool_id,
        "audit_tool_version": entry.audit_tool_version,
        "audit_tool_config_sha256": entry.audit_tool_config_sha256,
        "audit_tool_source_sha256": entry.audit_tool_source_sha256,
    }
    if actual != expected:
        raise DuplicateAuditError(
            f"duplicate audit row {entry.task_id!r} cites a different cluster namespace"
        )


def _validate_verified_task_audit_record(
    entry: DuplicateAuditEntry,
    namespace: JointDuplicateClusterNamespace,
) -> None:
    assert entry.content_sha256 is not None
    assert entry.train_corpus_manifest_sha256 is not None
    expected = canonical_task_audit_record(
        task_id=entry.task_id,
        content_sha256=entry.content_sha256,
        train_corpus_manifest_sha256=entry.train_corpus_manifest_sha256,
        cluster_ids=entry.cluster_ids,
        namespace=namespace,
    )
    record = entry.audit_record
    if not isinstance(record, Mapping) or dict(record) != expected:
        raise DuplicateAuditError(
            f"duplicate audit record differs from declared row for {entry.task_id!r}"
        )
    expected_evidence = canonical_sha256(expected)
    if entry.evidence_sha256 != expected_evidence:
        raise DuplicateAuditError(
            f"duplicate audit evidence hash is not canonical for {entry.task_id!r}"
        )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DuplicateAuditError("duplicate audit hash must be a string or null")
    return value


def _optional_sha256(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_sha256(value, field=field)


def _required_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or value != value.lower() or not _is_sha256(value):
        raise DuplicateAuditError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
