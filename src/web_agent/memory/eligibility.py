"""Fail-closed eligibility and deduplication for Table 2 corrective memory.

Only reviewed, causally complete, successful training recoveries may enter the
primary P4 store.  Records that clearly fail a registered gate are excluded and
counted.  A record that otherwise qualifies but lacks required provenance is a
hard error: silently dropping such a row would make the frozen corpus depend on
accidental metadata availability.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

from web_agent.memory.manifest import (
    ELIGIBILITY_POLICY_VERSION,
    ManifestError,
    PROVENANCE_SCHEMA_VERSION,
    require_sha256,
)
from web_agent.runtime.duplicate_audit import (
    DuplicateAuditError,
    JointDuplicateClusterNamespace,
)
from web_agent.memory.verification import (
    VerificationEvidenceError,
    validate_provenance_verification_evidence,
)


class MemoryEligibilityError(ValueError):
    """Raised when a potentially eligible memory lacks required evidence."""


EXCLUSION_REASONS = (
    "quarantined_or_non_admitted",
    "memory_update_flag_missing",
    "memory_update_flag_false",
    "recovery_success_missing",
    "recovery_not_successful",
    "final_task_not_successful",
    "provenance_invalid",
    "exact_duplicate",
    "near_duplicate",
)

_NON_ADMISSION_TEXT_FIELDS = (
    "status",
    "final_review_decision",
    "review_decision",
    "admission_status",
    "memory_admission_status",
    "disposition",
)
_NON_ADMISSION_TEXT_VALUES = frozenset(
    {
        "quarantine",
        "quarantined",
        "quarantine_policy",
        "reject_recollect",
        "rejected",
        "excluded",
        "non_admitted",
        "nonadmitted",
        "do_not_admit",
    }
)
_QUARANTINE_BOOLEAN_FIELDS = (
    "quarantine",
    "quarantined",
    "excluded_from_memory",
)
_ADMISSION_BOOLEAN_FIELDS = (
    "memory_admission_allowed",
    "admission_allowed",
    "memory_eligible",
)


def _view(record: Mapping[str, Any]) -> tuple[Mapping, Mapping, Mapping]:
    if "inputs" in record and "labels" in record:
        inputs = record.get("inputs")
        labels = record.get("labels")
        meta = record.get("meta", {})
    else:
        inputs = labels = meta = record
    if not all(isinstance(value, Mapping) for value in (inputs, labels, meta)):
        raise MemoryEligibilityError("record inputs/labels/meta must be mappings")
    return inputs, labels, meta


def _nonempty(value: object, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise MemoryEligibilityError(f"missing required memory evidence: {field}")
    return normalized


def _strict_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise MemoryEligibilityError(f"{field} must be a JSON boolean")
    return value


def _strict_step(value: object) -> int:
    if isinstance(value, bool):
        raise MemoryEligibilityError("step_index must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise MemoryEligibilityError("step_index must be an integer") from error
    if parsed < 0:
        raise MemoryEligibilityError("step_index must be non-negative")
    return parsed


def _is_explicitly_non_admitted(
    meta: Mapping[str, Any], evidence: Mapping[str, Any] | None
) -> bool:
    """Reject a quarantine/non-admission marker without relying on an overlay.

    The review overlay normally removes quarantined rows before this package is
    called.  Memory construction nevertheless treats the source row and frozen
    provenance as independent fail-closed authorities, so omitting that
    optional loader step can never turn an explicitly excluded row into P4
    memory.
    """

    authorities: list[tuple[str, Mapping[str, Any]]] = [("meta", meta)]
    if evidence is not None:
        authorities.append(("provenance", evidence))
    overlay = meta.get("targeted_review_overlay")
    if overlay is not None:
        if not isinstance(overlay, Mapping):
            raise MemoryEligibilityError(
                "meta.targeted_review_overlay must be a mapping"
            )
        authorities.append(("meta.targeted_review_overlay", overlay))

    for authority_name, authority in authorities:
        review_status = str(authority.get("review_status", "")).strip().lower()
        if review_status in _NON_ADMISSION_TEXT_VALUES:
            return True
        for field in _NON_ADMISSION_TEXT_FIELDS:
            value = str(authority.get(field, "")).strip().lower().replace("-", "_")
            if value in _NON_ADMISSION_TEXT_VALUES:
                return True
        for field in _QUARANTINE_BOOLEAN_FIELDS:
            if field not in authority:
                continue
            value = authority[field]
            if type(value) is not bool:
                raise MemoryEligibilityError(
                    f"{authority_name}.{field} must be a JSON boolean"
                )
            if value:
                return True
        for field in _ADMISSION_BOOLEAN_FIELDS:
            if field not in authority:
                continue
            value = authority[field]
            if type(value) is not bool:
                raise MemoryEligibilityError(
                    f"{authority_name}.{field} must be a JSON boolean"
                )
            if not value:
                return True
    return False


@dataclass(frozen=True)
class ProvenanceManifest:
    """Frozen evidence external to the supervised row schema.

    ``records`` is keyed by source sample ID.  A potentially admitted sample
    must provide ``provenance_valid``, ``final_task_success``,
    ``canonical_task_id``, ``episode_id``, a SHA-256 ``exact_duplicate_key``,
    ``near_duplicate_cluster_id``, and the registered joint duplicate-cluster
    namespace ID. A potentially admitted sample must also bind independently
    produced recovery and final-task verification records plus their combined
    evidence digest. Missing fields reject the complete build rather than
    silently weakening P4 eligibility.
    """

    dataset_id: str
    dataset_version: str
    dataset_artifacts_sha256: str
    records_sha256: str
    duplicate_cluster_namespace: JointDuplicateClusterNamespace
    records: Mapping[str, Mapping[str, Any]]
    validation_rows_read: int
    test_rows_read: int
    locked_test_rows_read: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProvenanceManifest":
        if not isinstance(payload, Mapping):
            raise ManifestError("memory provenance manifest must be an object")
        if payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
            raise ManifestError(
                "unsupported memory provenance schema: "
                f"{payload.get('schema_version')!r}"
            )
        if payload.get("source_split") != "train":
            raise ManifestError("memory provenance source_split must be 'train'")
        try:
            duplicate_cluster_namespace = JointDuplicateClusterNamespace.from_mapping(
                payload.get("duplicate_cluster_namespace"),
                require_hashes=True,
            )
        except DuplicateAuditError as error:
            raise ManifestError(
                f"invalid memory duplicate-cluster namespace: {error}"
            ) from error
        read_counts: dict[str, int] = {}
        for field in (
            "validation_rows_read",
            "test_rows_read",
            "locked_test_rows_read",
        ):
            raw_value = payload.get(field)
            if type(raw_value) is not int:
                raise ManifestError(f"{field} must be an explicit JSON integer")
            if raw_value != 0:
                raise ManifestError(
                    f"memory provenance requires explicit {field}=0"
                )
            read_counts[field] = raw_value
        records = payload.get("records")
        if not isinstance(records, Mapping):
            raise ManifestError("memory provenance records must be keyed by sample ID")
        normalized_records: dict[str, Mapping[str, Any]] = {}
        for raw_id, evidence in records.items():
            sample_id = str(raw_id).strip()
            if not sample_id or not isinstance(evidence, Mapping):
                raise ManifestError("invalid sample entry in provenance manifest")
            if sample_id in normalized_records:
                raise ManifestError(f"duplicate provenance sample ID: {sample_id}")
            if evidence.get("source_split") != "train":
                raise ManifestError(
                    f"provenance sample {sample_id} is not source_split=train"
                )
            declared_sample_id = str(evidence.get("source_sample_id") or "").strip()
            if declared_sample_id != sample_id:
                raise ManifestError(
                    f"provenance sample key/ID mismatch for {sample_id}"
                )
            if (
                evidence.get("duplicate_cluster_namespace_id")
                != duplicate_cluster_namespace.namespace_id
            ):
                raise ManifestError(
                    f"provenance sample {sample_id} cites a different "
                    "duplicate-cluster namespace"
                )
            normalized_records[sample_id] = evidence
        return cls(
            dataset_id=_nonempty(payload.get("dataset_id"), field="dataset_id"),
            dataset_version=_nonempty(
                payload.get("dataset_version"), field="dataset_version"
            ),
            dataset_artifacts_sha256=require_sha256(
                payload.get("dataset_artifacts_sha256"),
                field="dataset_artifacts_sha256",
            ),
            records_sha256=require_sha256(
                payload.get("records_sha256"), field="records_sha256"
            ),
            duplicate_cluster_namespace=duplicate_cluster_namespace,
            records=normalized_records,
            validation_rows_read=read_counts["validation_rows_read"],
            test_rows_read=read_counts["test_rows_read"],
            locked_test_rows_read=read_counts["locked_test_rows_read"],
        )


@dataclass(frozen=True)
class EligibleMemoryCandidate:
    record_index: int
    memory_id: str
    source_sample_id: str
    recovery_sample_id: str
    canonical_task_id: str
    episode_id: str
    step_index: int
    exact_duplicate_key: str
    near_duplicate_cluster_id: str
    duplicate_cluster_namespace_id: str
    recovery_verification_evidence_sha256: str
    final_task_verification_evidence_sha256: str
    verification_evidence_sha256: str
    verification_evidence: Mapping[str, Any]
    item: Mapping[str, Any]


@dataclass(frozen=True)
class EligibilitySelection:
    candidates: tuple[EligibleMemoryCandidate, ...]
    exclusion_counts: Mapping[str, int]
    input_rows: int
    pre_dedup_eligible_rows: int
    duplicate_cluster_namespace: Mapping[str, Any]


def _sample_id(meta: Mapping[str, Any], index: int) -> str:
    return _nonempty(
        meta.get("sample_id") or meta.get("row_id") or meta.get("id"),
        field=f"record[{index}].sample_id",
    )


def _memory_id(dataset_id: str, source_sample_id: str, recovery_sample_id: str) -> str:
    raw = f"{dataset_id}\0{source_sample_id}\0{recovery_sample_id}".encode("utf-8")
    return f"mem_{hashlib.sha256(raw).hexdigest()}"


def _candidate_from_record(
    record: Mapping[str, Any],
    *,
    index: int,
    transition: Mapping[str, Any] | None,
    provenance: ProvenanceManifest,
) -> EligibleMemoryCandidate:
    inputs, labels, meta = _view(record)
    sample_id = _sample_id(meta, index)
    evidence = provenance.records.get(sample_id)
    if evidence is None:
        raise MemoryEligibilityError(
            f"potential memory {sample_id} is absent from provenance manifest"
        )
    if transition is None:
        raise MemoryEligibilityError(
            f"potential memory {sample_id} lacks a causal recovery transition"
        )
    transition_source_id = _nonempty(
        transition.get("source_sample_id"),
        field=f"{sample_id}.transition.source_sample_id",
    )
    if transition_source_id != sample_id:
        raise MemoryEligibilityError(
            f"potential memory {sample_id} is bound to another transition source"
        )
    if _strict_bool(
        evidence.get("provenance_valid"), field=f"{sample_id}.provenance_valid"
    ) is not True:
        raise MemoryEligibilityError(
            f"potential memory {sample_id} has invalid provenance"
        )
    final_success = _strict_bool(
        evidence.get("final_task_success"),
        field=f"{sample_id}.final_task_success",
    )
    if not final_success:
        raise MemoryEligibilityError(
            f"potential memory {sample_id} is not a successful final episode"
        )

    canonical_task_id = _nonempty(
        evidence.get("canonical_task_id"),
        field=f"{sample_id}.canonical_task_id",
    )
    episode_id = _nonempty(
        evidence.get("episode_id"), field=f"{sample_id}.episode_id"
    )
    source_task_id = _nonempty(
        meta.get("task_id") or meta.get("original_task_id"),
        field=f"{sample_id}.meta.task_id",
    )
    transition_task_id = _nonempty(
        transition.get("task_id"), field=f"{sample_id}.transition.task_id"
    )
    if canonical_task_id not in {source_task_id, transition_task_id} or (
        source_task_id != transition_task_id
    ):
        raise MemoryEligibilityError(
            f"potential memory {sample_id} task identity differs across "
            "record, transition, and provenance"
        )
    source_episode_id = _nonempty(
        meta.get("episode_id") or meta.get("trajectory_id") or source_task_id,
        field=f"{sample_id}.meta.episode_id",
    )
    transition_episode_value = transition.get("episode_id") or transition.get(
        "trajectory_id"
    )
    if episode_id != source_episode_id or (
        transition_episode_value is not None
        and _nonempty(
            transition_episode_value,
            field=f"{sample_id}.transition.episode_id",
        )
        != episode_id
    ):
        raise MemoryEligibilityError(
            f"potential memory {sample_id} episode identity differs across "
            "record, transition, and provenance"
        )
    exact_duplicate_key = require_sha256(
        evidence.get("exact_duplicate_key"),
        field=f"{sample_id}.exact_duplicate_key",
    )
    near_duplicate_cluster_id = _nonempty(
        evidence.get("near_duplicate_cluster_id"),
        field=f"{sample_id}.near_duplicate_cluster_id",
    )
    duplicate_cluster_namespace_id = _nonempty(
        evidence.get("duplicate_cluster_namespace_id"),
        field=f"{sample_id}.duplicate_cluster_namespace_id",
    )
    if duplicate_cluster_namespace_id != provenance.duplicate_cluster_namespace.namespace_id:
        raise MemoryEligibilityError(
            f"potential memory {sample_id} cites a different duplicate-cluster namespace"
        )
    if str(meta.get("review_status", "")).lower() != "approved":
        raise MemoryEligibilityError(
            f"potential memory {sample_id} is not review_status=approved"
        )

    outcome = _nonempty(labels.get("outcome_label"), field=f"{sample_id}.outcome")
    failure_type = _nonempty(
        labels.get("failure_type_4"), field=f"{sample_id}.failure_type"
    )
    recovery_strategy = _nonempty(
        labels.get("recovery_strategy"), field=f"{sample_id}.recovery_strategy"
    )
    if outcome != "FAILURE" or failure_type == "NONE":
        raise MemoryEligibilityError(
            f"potential memory {sample_id} is not an observed agent failure"
        )
    if recovery_strategy == "NONE":
        raise MemoryEligibilityError(
            f"potential memory {sample_id} has no recovery strategy"
        )
    if "recovery_attempted" in labels and labels["recovery_attempted"] is not True:
        raise MemoryEligibilityError(
            f"potential memory {sample_id} was not an executed recovery"
        )
    if transition.get("recovery_success") is not True:
        raise MemoryEligibilityError(
            f"potential memory {sample_id} transition is not verified successful"
        )
    transition_strategy = _nonempty(
        transition.get("recovery_strategy"),
        field=f"{sample_id}.transition.recovery_strategy",
    )
    if transition_strategy != recovery_strategy:
        raise MemoryEligibilityError(
            f"potential memory {sample_id} strategy conflicts with transition"
        )
    recovery_sample_id = _nonempty(
        transition.get("recovery_sample_id"),
        field=f"{sample_id}.transition.recovery_sample_id",
    )
    executed_recovery_action = _nonempty(
        transition.get("executed_recovery_action"),
        field=f"{sample_id}.transition.executed_recovery_action",
    )
    _nonempty(
        transition.get("failure_state"),
        field=f"{sample_id}.transition.failure_state",
    )
    _nonempty(
        transition.get("post_recovery_state"),
        field=f"{sample_id}.transition.post_recovery_state",
    )
    step_index = _strict_step(
        meta.get("step_index", transition.get("source_step_index"))
    )
    transition_step_index = _strict_step(transition.get("source_step_index"))
    if step_index != transition_step_index:
        raise MemoryEligibilityError(
            f"potential memory {sample_id} source step differs from transition"
        )
    try:
        verified_evidence = validate_provenance_verification_evidence(
            evidence,
            source_sample_id=sample_id,
            recovery_sample_id=recovery_sample_id,
            canonical_task_id=canonical_task_id,
            episode_id=episode_id,
            transition=transition,
        )
    except VerificationEvidenceError as error:
        raise MemoryEligibilityError(str(error)) from error

    memory_id = _memory_id(provenance.dataset_id, sample_id, recovery_sample_id)
    item = {
        "schema_version": ELIGIBILITY_POLICY_VERSION,
        "memory_id": memory_id,
        "source_split": "train",
        "source_dataset_id": provenance.dataset_id,
        "source_dataset_version": provenance.dataset_version,
        "source_sample_id": sample_id,
        "recovery_sample_id": recovery_sample_id,
        "source_task_id": canonical_task_id,
        "source_episode_id": episode_id,
        "step_index": step_index,
        "website_domain": str(inputs.get("website_domain") or ""),
        "failure_type": failure_type,
        "failed_action": str(labels.get("action_type") or ""),
        "strategy": recovery_strategy,
        "executed_recovery_action": executed_recovery_action,
        "recovery_action_value": str(
            transition.get("recovery_action_value") or ""
        ),
        "reflection_text": str(labels.get("reflection_text") or ""),
        "memory_update_flag": True,
        "verified_recovery_success": True,
        "final_task_success": True,
        "provenance_valid": True,
        "recovery_verification_evidence_sha256": (
            verified_evidence.recovery_evidence_sha256
        ),
        "final_task_verification_evidence_sha256": (
            verified_evidence.final_task_evidence_sha256
        ),
        "verification_evidence_sha256": verified_evidence.bundle_sha256,
        "exact_duplicate_key": exact_duplicate_key,
        "duplicate_cluster_id": near_duplicate_cluster_id,
        "duplicate_cluster_namespace_id": duplicate_cluster_namespace_id,
    }
    return EligibleMemoryCandidate(
        record_index=index,
        memory_id=memory_id,
        source_sample_id=sample_id,
        recovery_sample_id=recovery_sample_id,
        canonical_task_id=canonical_task_id,
        episode_id=episode_id,
        step_index=step_index,
        exact_duplicate_key=exact_duplicate_key,
        near_duplicate_cluster_id=near_duplicate_cluster_id,
        duplicate_cluster_namespace_id=duplicate_cluster_namespace_id,
        recovery_verification_evidence_sha256=(
            verified_evidence.recovery_evidence_sha256
        ),
        final_task_verification_evidence_sha256=(
            verified_evidence.final_task_evidence_sha256
        ),
        verification_evidence_sha256=verified_evidence.bundle_sha256,
        verification_evidence=verified_evidence.to_payload(),
        item=item,
    )


def select_eligible_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    source_split: str,
    transitions: Mapping[str, Mapping[str, Any]],
    provenance: ProvenanceManifest,
) -> EligibilitySelection:
    """Select eligible memories and deterministically remove duplicate clusters."""
    if source_split != "train":
        raise MemoryEligibilityError(
            "corrective memory can be built only from source_split='train'"
        )
    if any((
        provenance.validation_rows_read,
        provenance.test_rows_read,
        provenance.locked_test_rows_read,
    )):
        raise MemoryEligibilityError("provenance reports non-training split access")

    exclusions: Counter[str] = Counter()
    candidates: list[EligibleMemoryCandidate] = []
    fatal: list[str] = []
    for index, record in enumerate(records):
        try:
            _, labels, meta = _view(record)
            sample_id = _sample_id(meta, index)
            evidence = provenance.records.get(sample_id)
            if _is_explicitly_non_admitted(meta, evidence):
                exclusions["quarantined_or_non_admitted"] += 1
                continue
            if "memory_update_flag" not in labels:
                exclusions["memory_update_flag_missing"] += 1
                continue
            memory_flag = _strict_bool(
                labels.get("memory_update_flag"),
                field=f"{sample_id}.memory_update_flag",
            )
            if "recovery_success" not in labels:
                exclusions["recovery_success_missing"] += 1
                continue
            recovery_success = labels.get("recovery_success")
            if recovery_success is not None and type(recovery_success) is not bool:
                raise MemoryEligibilityError(
                    f"{sample_id}.recovery_success must be true, false, or null"
                )
            if not memory_flag:
                exclusions["memory_update_flag_false"] += 1
                continue
            if recovery_success is not True:
                exclusions["recovery_not_successful"] += 1
                continue
            if evidence is not None and evidence.get("final_task_success") is False:
                exclusions["final_task_not_successful"] += 1
                continue
            if evidence is not None and evidence.get("provenance_valid") is False:
                exclusions["provenance_invalid"] += 1
                continue
            candidates.append(_candidate_from_record(
                record,
                index=index,
                transition=transitions.get(sample_id),
                provenance=provenance,
            ))
        except (MemoryEligibilityError, ManifestError) as error:
            fatal.append(str(error))

    if fatal:
        examples = "\n - ".join(fatal[:25])
        raise MemoryEligibilityError(
            f"memory eligibility failed for {len(fatal)} record(s):\n - {examples}"
        )
    if not candidates:
        raise MemoryEligibilityError("no eligible successful training memories")

    pre_dedup = len(candidates)
    kept: list[EligibleMemoryCandidate] = []
    seen_exact: set[str] = set()
    seen_near: set[str] = set()
    for candidate in sorted(candidates, key=lambda value: value.memory_id):
        if candidate.exact_duplicate_key in seen_exact:
            exclusions["exact_duplicate"] += 1
            continue
        if candidate.near_duplicate_cluster_id in seen_near:
            exclusions["near_duplicate"] += 1
            continue
        seen_exact.add(candidate.exact_duplicate_key)
        seen_near.add(candidate.near_duplicate_cluster_id)
        kept.append(candidate)
    if not kept:
        raise MemoryEligibilityError("deduplication removed every eligible memory")
    return EligibilitySelection(
        candidates=tuple(kept),
        exclusion_counts={
            reason: int(exclusions[reason]) for reason in EXCLUSION_REASONS
        },
        input_rows=len(records),
        pre_dedup_eligible_rows=pre_dedup,
        duplicate_cluster_namespace=provenance.duplicate_cluster_namespace.to_dict(),
    )
