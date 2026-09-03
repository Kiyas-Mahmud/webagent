"""Immutable independent-verification evidence for admitted P4 memories.

Declared success booleans are eligibility gates, not scientific evidence.  A
potentially admitted memory must additionally bind versioned independent
recovery and final-task verification records.  Their canonical digests are
carried into the frozen item and closed again at the store-manifest level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from web_agent.memory.manifest import canonical_sha256, require_sha256, sha256_file


RECOVERY_VERIFICATION_SCHEMA_VERSION = (
    "table2-memory-recovery-verification-evidence-v1"
)
FINAL_TASK_VERIFICATION_SCHEMA_VERSION = (
    "table2-memory-final-task-verification-evidence-v1"
)
VERIFICATION_BUNDLE_SCHEMA_VERSION = "table2-memory-verification-bundle-v1"
VERIFICATION_MANIFEST_SCHEMA_VERSION = "table2-memory-verification-manifest-v1"
VERIFICATION_RECORDS_SCHEMA_VERSION = "table2-memory-verification-records-v1"
VERIFICATION_DIGEST_ALGORITHM = "canonical_json_sha256_v1"
VERIFICATION_AUTHORITY_TYPES = frozenset({"verifier", "reviewer"})


class VerificationEvidenceError(ValueError):
    """Independent recovery/final-task evidence is absent or inconsistent."""


@dataclass(frozen=True, slots=True)
class VerifiedMemoryEvidence:
    recovery_evidence_sha256: str
    final_task_evidence_sha256: str
    bundle_sha256: str
    recovery_record: Mapping[str, Any]
    final_task_record: Mapping[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """Return the exact evidence subset safe to freeze with a memory item."""

        return {
            "recovery_verification": dict(self.recovery_record),
            "final_task_verification": dict(self.final_task_record),
            "verification_evidence_sha256": self.bundle_sha256,
        }


def _nonempty(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise VerificationEvidenceError(f"verification evidence requires {field}")
    return text


def _strict_true(value: object, *, field: str) -> None:
    if value is not True or type(value) is not bool:
        raise VerificationEvidenceError(f"verification evidence requires {field}=true")


def _require_exact_fields(
    value: object,
    *,
    required: frozenset[str],
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VerificationEvidenceError(f"{field} must be an object")
    if set(value) != set(required):
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        raise VerificationEvidenceError(
            f"{field} fields differ from the registered schema: "
            f"missing={missing}, extra={extra}"
        )
    return value


def _validate_authority(value: Mapping[str, Any], *, field: str) -> None:
    authority_type = _nonempty(
        value.get("authority_type"), field=f"{field}.authority_type"
    )
    if authority_type not in VERIFICATION_AUTHORITY_TYPES:
        raise VerificationEvidenceError(
            f"{field}.authority_type must be verifier or reviewer"
        )
    _nonempty(value.get("authority_id"), field=f"{field}.authority_id")
    _nonempty(value.get("authority_version"), field=f"{field}.authority_version")
    _strict_true(
        value.get("independent_verification"),
        field=f"{field}.independent_verification",
    )


def _validate_record_digest(value: Mapping[str, Any], *, field: str) -> str:
    registered = require_sha256(
        value.get("evidence_sha256"), field=f"{field}.evidence_sha256"
    )
    content = {key: item for key, item in value.items() if key != "evidence_sha256"}
    try:
        actual = canonical_sha256(content)
    except (TypeError, ValueError) as error:
        raise VerificationEvidenceError(
            f"{field} must contain finite canonical JSON evidence"
        ) from error
    if registered != actual:
        raise VerificationEvidenceError(
            f"{field}.evidence_sha256 differs from its canonical evidence"
        )
    return registered


_RECOVERY_FIELDS = frozenset(
    {
        "schema_version",
        "authority_type",
        "authority_id",
        "authority_version",
        "independent_verification",
        "source_sample_id",
        "recovery_sample_id",
        "canonical_task_id",
        "episode_id",
        "pre_recovery_state_sha256",
        "executed_recovery_action_sha256",
        "post_recovery_state_sha256",
        "verified_recovery_success",
        "evidence_sha256",
    }
)

_FINAL_TASK_FIELDS = frozenset(
    {
        "schema_version",
        "authority_type",
        "authority_id",
        "authority_version",
        "independent_verification",
        "source_sample_id",
        "canonical_task_id",
        "episode_id",
        "task_specification_sha256",
        "terminal_state_sha256",
        "terminal_verifier_output_sha256",
        "verified_final_task_success",
        "evidence_sha256",
    }
)


def recovery_action_evidence_sha256(transition: Mapping[str, Any]) -> str:
    """Hash the exact recovery-action projection bound by provenance."""

    return canonical_sha256(
        {
            "recovery_sample_id": _nonempty(
                transition.get("recovery_sample_id"),
                field="transition.recovery_sample_id",
            ),
            "recovery_strategy": _nonempty(
                transition.get("recovery_strategy"),
                field="transition.recovery_strategy",
            ),
            "executed_recovery_action": _nonempty(
                transition.get("executed_recovery_action"),
                field="transition.executed_recovery_action",
            ),
            "recovery_action_value": str(
                transition.get("recovery_action_value") or ""
            ),
        }
    )


def recovery_state_evidence_sha256(
    transition: Mapping[str, Any],
    *,
    field: str,
) -> str:
    """Hash state artifact bytes when a production data root is available.

    Synthetic fixtures may provide an in-memory JSON state with no data root;
    production memory construction injects an explicit root and therefore
    fails closed unless the referenced artifact exists beneath that root.
    """

    if field not in {"failure_state", "post_recovery_state"}:
        raise VerificationEvidenceError(f"unsupported recovery-state field: {field}")
    state = transition.get(field)
    if state is None or (isinstance(state, str) and not state.strip()):
        raise VerificationEvidenceError(f"transition.{field} cannot be empty")
    raw_root = transition.get("_data_root")
    if raw_root is None:
        return canonical_sha256(state)
    root = Path(_nonempty(raw_root, field="transition._data_root")).resolve()
    if not root.is_dir():
        raise VerificationEvidenceError(
            f"transition._data_root is not a directory: {root}"
        )
    if not isinstance(state, str):
        raise VerificationEvidenceError(
            f"transition.{field} must be an artifact path when _data_root is set"
        )
    candidate = Path(state)
    artifact = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        artifact.relative_to(root)
    except ValueError as error:
        raise VerificationEvidenceError(
            f"transition.{field} escapes the registered data root"
        ) from error
    if not artifact.is_file():
        raise VerificationEvidenceError(
            f"transition.{field} artifact does not exist: {artifact}"
        )
    return sha256_file(artifact)


def verification_bundle_sha256(
    *,
    source_sample_id: str,
    recovery_sample_id: str,
    canonical_task_id: str,
    episode_id: str,
    recovery_evidence_sha256: str,
    final_task_evidence_sha256: str,
) -> str:
    """Close both independent records into one re-playable item digest."""

    return canonical_sha256(
        {
            "schema_version": VERIFICATION_BUNDLE_SCHEMA_VERSION,
            "source_sample_id": _nonempty(
                source_sample_id, field="bundle.source_sample_id"
            ),
            "recovery_sample_id": _nonempty(
                recovery_sample_id, field="bundle.recovery_sample_id"
            ),
            "canonical_task_id": _nonempty(
                canonical_task_id, field="bundle.canonical_task_id"
            ),
            "episode_id": _nonempty(episode_id, field="bundle.episode_id"),
            "recovery_verification_evidence_sha256": require_sha256(
                recovery_evidence_sha256,
                field="bundle.recovery_verification_evidence_sha256",
            ),
            "final_task_verification_evidence_sha256": require_sha256(
                final_task_evidence_sha256,
                field="bundle.final_task_verification_evidence_sha256",
            ),
        }
    )


def validate_provenance_verification_evidence(
    evidence: Mapping[str, Any],
    *,
    source_sample_id: str,
    recovery_sample_id: str,
    canonical_task_id: str,
    episode_id: str,
    transition: Mapping[str, Any] | None = None,
) -> VerifiedMemoryEvidence:
    """Validate both independent evidence records and their bundle digest."""

    recovery = _require_exact_fields(
        evidence.get("recovery_verification"),
        required=_RECOVERY_FIELDS,
        field=f"{source_sample_id}.recovery_verification",
    )
    final_task = _require_exact_fields(
        evidence.get("final_task_verification"),
        required=_FINAL_TASK_FIELDS,
        field=f"{source_sample_id}.final_task_verification",
    )
    expected_identity = {
        "source_sample_id": source_sample_id,
        "canonical_task_id": canonical_task_id,
        "episode_id": episode_id,
    }
    for field, expected in expected_identity.items():
        if recovery.get(field) != expected or final_task.get(field) != expected:
            raise VerificationEvidenceError(
                f"{source_sample_id} verification evidence differs at {field}"
            )
    if recovery.get("recovery_sample_id") != recovery_sample_id:
        raise VerificationEvidenceError(
            f"{source_sample_id} recovery verification cites another recovery sample"
        )
    if recovery.get("schema_version") != RECOVERY_VERIFICATION_SCHEMA_VERSION:
        raise VerificationEvidenceError(
            f"{source_sample_id} uses an unsupported recovery-verification schema"
        )
    if final_task.get("schema_version") != FINAL_TASK_VERIFICATION_SCHEMA_VERSION:
        raise VerificationEvidenceError(
            f"{source_sample_id} uses an unsupported final-task-verification schema"
        )
    _validate_authority(recovery, field=f"{source_sample_id}.recovery_verification")
    _validate_authority(
        final_task, field=f"{source_sample_id}.final_task_verification"
    )
    _strict_true(
        recovery.get("verified_recovery_success"),
        field=f"{source_sample_id}.verified_recovery_success",
    )
    _strict_true(
        final_task.get("verified_final_task_success"),
        field=f"{source_sample_id}.verified_final_task_success",
    )
    for field in (
        "pre_recovery_state_sha256",
        "executed_recovery_action_sha256",
        "post_recovery_state_sha256",
    ):
        require_sha256(recovery.get(field), field=f"recovery_verification.{field}")
    for field in (
        "task_specification_sha256",
        "terminal_state_sha256",
        "terminal_verifier_output_sha256",
    ):
        require_sha256(final_task.get(field), field=f"final_task_verification.{field}")

    if transition is not None:
        expected_transition_hashes = {
            "pre_recovery_state_sha256": recovery_state_evidence_sha256(
                transition,
                field="failure_state",
            ),
            "executed_recovery_action_sha256": recovery_action_evidence_sha256(
                transition
            ),
            "post_recovery_state_sha256": recovery_state_evidence_sha256(
                transition,
                field="post_recovery_state",
            ),
        }
        for field, expected in expected_transition_hashes.items():
            if recovery.get(field) != expected:
                raise VerificationEvidenceError(
                    f"{source_sample_id} recovery verification differs from "
                    f"transition at {field}"
                )

    recovery_digest = _validate_record_digest(
        recovery, field=f"{source_sample_id}.recovery_verification"
    )
    final_digest = _validate_record_digest(
        final_task, field=f"{source_sample_id}.final_task_verification"
    )
    bundle_digest = verification_bundle_sha256(
        source_sample_id=source_sample_id,
        recovery_sample_id=recovery_sample_id,
        canonical_task_id=canonical_task_id,
        episode_id=episode_id,
        recovery_evidence_sha256=recovery_digest,
        final_task_evidence_sha256=final_digest,
    )
    registered_bundle = require_sha256(
        evidence.get("verification_evidence_sha256"),
        field=f"{source_sample_id}.verification_evidence_sha256",
    )
    if registered_bundle != bundle_digest:
        raise VerificationEvidenceError(
            f"{source_sample_id} verification evidence bundle digest mismatch"
        )
    return VerifiedMemoryEvidence(
        recovery_evidence_sha256=recovery_digest,
        final_task_evidence_sha256=final_digest,
        bundle_sha256=bundle_digest,
        recovery_record=dict(recovery),
        final_task_record=dict(final_task),
    )


def validate_frozen_item_verification(item: Mapping[str, Any]) -> str:
    """Recompute one frozen item's combined evidence digest."""

    expected = verification_bundle_sha256(
        source_sample_id=str(item.get("source_sample_id") or ""),
        recovery_sample_id=str(item.get("recovery_sample_id") or ""),
        canonical_task_id=str(item.get("source_task_id") or ""),
        episode_id=str(item.get("source_episode_id") or ""),
        recovery_evidence_sha256=str(
            item.get("recovery_verification_evidence_sha256") or ""
        ),
        final_task_evidence_sha256=str(
            item.get("final_task_verification_evidence_sha256") or ""
        ),
    )
    registered = require_sha256(
        item.get("verification_evidence_sha256"),
        field="item.verification_evidence_sha256",
    )
    if registered != expected:
        raise VerificationEvidenceError(
            "frozen memory verification evidence bundle digest mismatch"
        )
    return expected


def verification_manifest_payload(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the deterministic per-item verification closure for a store."""

    rows: list[dict[str, str]] = []
    memory_ids: set[str] = set()
    for item in items:
        memory_id = _nonempty(item.get("memory_id"), field="item.memory_id")
        if memory_id in memory_ids:
            raise VerificationEvidenceError(
                f"duplicate memory ID in verification closure: {memory_id}"
            )
        memory_ids.add(memory_id)
        validate_frozen_item_verification(item)
        rows.append(
            {
                "memory_id": memory_id,
                "source_sample_id": _nonempty(
                    item.get("source_sample_id"), field="item.source_sample_id"
                ),
                "recovery_verification_evidence_sha256": require_sha256(
                    item.get("recovery_verification_evidence_sha256"),
                    field="item.recovery_verification_evidence_sha256",
                ),
                "final_task_verification_evidence_sha256": require_sha256(
                    item.get("final_task_verification_evidence_sha256"),
                    field="item.final_task_verification_evidence_sha256",
                ),
                "verification_evidence_sha256": require_sha256(
                    item.get("verification_evidence_sha256"),
                    field="item.verification_evidence_sha256",
                ),
            }
        )
    rows.sort(key=lambda row: row["memory_id"])
    return {
        "schema_version": VERIFICATION_MANIFEST_SCHEMA_VERSION,
        "digest_algorithm": VERIFICATION_DIGEST_ALGORITHM,
        "item_count": len(rows),
        "items": rows,
        "items_sha256": canonical_sha256(rows),
    }


_FROZEN_RECORD_FIELDS = frozenset(
    {
        "memory_id",
        "source_sample_id",
        "recovery_sample_id",
        "canonical_task_id",
        "episode_id",
        "source_step_index",
        "recovery_verification",
        "final_task_verification",
        "recovery_verification_evidence_sha256",
        "final_task_verification_evidence_sha256",
        "verification_evidence_sha256",
        "record_sha256",
    }
)


def _strict_step(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise VerificationEvidenceError(f"{field} must be a non-negative integer")
    return value


def _frozen_verification_record(
    item: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and close the full evidence records for one frozen item."""

    memory_id = _nonempty(item.get("memory_id"), field="item.memory_id")
    source_sample_id = _nonempty(
        item.get("source_sample_id"), field=f"{memory_id}.source_sample_id"
    )
    recovery_sample_id = _nonempty(
        item.get("recovery_sample_id"), field=f"{memory_id}.recovery_sample_id"
    )
    canonical_task_id = _nonempty(
        item.get("source_task_id"), field=f"{memory_id}.source_task_id"
    )
    episode_id = _nonempty(
        item.get("source_episode_id"), field=f"{memory_id}.source_episode_id"
    )
    source_step_index = _strict_step(
        item.get("step_index"), field=f"{memory_id}.step_index"
    )
    verified = validate_provenance_verification_evidence(
        evidence,
        source_sample_id=source_sample_id,
        recovery_sample_id=recovery_sample_id,
        canonical_task_id=canonical_task_id,
        episode_id=episode_id,
    )
    expected_digests = {
        "recovery_verification_evidence_sha256": (
            verified.recovery_evidence_sha256
        ),
        "final_task_verification_evidence_sha256": (
            verified.final_task_evidence_sha256
        ),
        "verification_evidence_sha256": verified.bundle_sha256,
    }
    for field, expected in expected_digests.items():
        if item.get(field) != expected:
            raise VerificationEvidenceError(
                f"{memory_id} item/evidence mismatch at {field}"
            )
    row: dict[str, Any] = {
        "memory_id": memory_id,
        "source_sample_id": source_sample_id,
        "recovery_sample_id": recovery_sample_id,
        "canonical_task_id": canonical_task_id,
        "episode_id": episode_id,
        "source_step_index": source_step_index,
        "recovery_verification": dict(verified.recovery_record),
        "final_task_verification": dict(verified.final_task_record),
        **expected_digests,
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def verification_records_payload(
    item_evidence_pairs: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any]]
    ],
) -> dict[str, Any]:
    """Build the full, reloadable independent-evidence package for a store."""

    records = [
        _frozen_verification_record(item, evidence)
        for item, evidence in item_evidence_pairs
    ]
    records.sort(key=lambda row: row["memory_id"])
    memory_ids = [row["memory_id"] for row in records]
    if len(memory_ids) != len(set(memory_ids)):
        raise VerificationEvidenceError(
            "duplicate memory ID in frozen verification records"
        )
    return {
        "schema_version": VERIFICATION_RECORDS_SCHEMA_VERSION,
        "digest_algorithm": VERIFICATION_DIGEST_ALGORITHM,
        "record_count": len(records),
        "records": records,
        "records_sha256": canonical_sha256(records),
    }


def validate_verification_records_payload(
    payload: Mapping[str, Any],
    *,
    expected_item_count: int,
    items: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Replay full evidence schemas/digests and, when present, item bindings."""

    required = {
        "schema_version",
        "digest_algorithm",
        "record_count",
        "records",
        "records_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise VerificationEvidenceError(
            "frozen verification-record fields differ from the registered schema"
        )
    if payload.get("schema_version") != VERIFICATION_RECORDS_SCHEMA_VERSION:
        raise VerificationEvidenceError(
            "unsupported frozen verification-record schema"
        )
    if payload.get("digest_algorithm") != VERIFICATION_DIGEST_ALGORITHM:
        raise VerificationEvidenceError(
            "unsupported frozen verification-record digest algorithm"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise VerificationEvidenceError("frozen verification records must be an array")
    if (
        type(payload.get("record_count")) is not int
        or payload["record_count"] != len(records)
        or payload["record_count"] != expected_item_count
    ):
        raise VerificationEvidenceError(
            "frozen verification-record count is inconsistent"
        )
    if require_sha256(
        payload.get("records_sha256"), field="verification_records.records_sha256"
    ) != canonical_sha256(records):
        raise VerificationEvidenceError(
            "frozen verification-record aggregate hash mismatch"
        )

    items_by_id: dict[str, Mapping[str, Any]] | None = None
    if items is not None:
        items_by_id = {}
        for item in items:
            memory_id = _nonempty(item.get("memory_id"), field="item.memory_id")
            if memory_id in items_by_id:
                raise VerificationEvidenceError(
                    f"duplicate frozen memory ID: {memory_id}"
                )
            items_by_id[memory_id] = item
        if len(items_by_id) != expected_item_count:
            raise VerificationEvidenceError(
                "frozen verification records do not align with memory items"
            )

    prior_memory_id = ""
    for index, row in enumerate(records):
        if not isinstance(row, Mapping) or set(row) != _FROZEN_RECORD_FIELDS:
            raise VerificationEvidenceError(
                f"verification_records.records[{index}] fields are invalid"
            )
        memory_id = _nonempty(
            row.get("memory_id"), field=f"verification_records.records[{index}].memory_id"
        )
        if memory_id <= prior_memory_id:
            raise VerificationEvidenceError(
                "frozen verification records must have unique ascending memory IDs"
            )
        prior_memory_id = memory_id
        registered_record_hash = require_sha256(
            row.get("record_sha256"),
            field=f"verification_records.records[{index}].record_sha256",
        )
        record_content = {
            key: value for key, value in row.items() if key != "record_sha256"
        }
        if registered_record_hash != canonical_sha256(record_content):
            raise VerificationEvidenceError(
                f"verification record {memory_id} hash mismatch"
            )
        evidence = {
            "recovery_verification": row["recovery_verification"],
            "final_task_verification": row["final_task_verification"],
            "verification_evidence_sha256": row[
                "verification_evidence_sha256"
            ],
        }
        verified = validate_provenance_verification_evidence(
            evidence,
            source_sample_id=_nonempty(
                row.get("source_sample_id"), field=f"{memory_id}.source_sample_id"
            ),
            recovery_sample_id=_nonempty(
                row.get("recovery_sample_id"), field=f"{memory_id}.recovery_sample_id"
            ),
            canonical_task_id=_nonempty(
                row.get("canonical_task_id"), field=f"{memory_id}.canonical_task_id"
            ),
            episode_id=_nonempty(
                row.get("episode_id"), field=f"{memory_id}.episode_id"
            ),
        )
        _strict_step(
            row.get("source_step_index"), field=f"{memory_id}.source_step_index"
        )
        expected_digests = {
            "recovery_verification_evidence_sha256": (
                verified.recovery_evidence_sha256
            ),
            "final_task_verification_evidence_sha256": (
                verified.final_task_evidence_sha256
            ),
            "verification_evidence_sha256": verified.bundle_sha256,
        }
        for field, expected in expected_digests.items():
            if row.get(field) != expected:
                raise VerificationEvidenceError(
                    f"verification record {memory_id} differs at {field}"
                )

        if items_by_id is not None:
            item = items_by_id.pop(memory_id, None)
            if item is None:
                raise VerificationEvidenceError(
                    f"verification record has no frozen item: {memory_id}"
                )
            expected_item = {
                "source_sample_id": row["source_sample_id"],
                "recovery_sample_id": row["recovery_sample_id"],
                "source_task_id": row["canonical_task_id"],
                "source_episode_id": row["episode_id"],
                "step_index": row["source_step_index"],
                **expected_digests,
            }
            for field, expected in expected_item.items():
                if item.get(field) != expected:
                    raise VerificationEvidenceError(
                        f"frozen item {memory_id} differs from verification record at {field}"
                    )
    if items_by_id:
        raise VerificationEvidenceError(
            "one or more frozen items lack a verification record"
        )
