"""Train-only P4 preparation and cross-host transfer integrity.

This module has two deliberately separate responsibilities:

* inspect the Gold and optional recovery-supplement *training* JSON files and
  prepare a queue for independent recovery/final-task review; and
* describe and verify the exact files of an already-built frozen memory store
  before they are copied between the data/GPU host and the campaign host.

The preparation queue is not a provenance manifest.  In particular, this
module never creates an independent authority, a final-task-success decision,
duplicate-cluster evidence, or a verification-evidence digest.  The existing
``ProvenanceManifest`` and ``select_eligible_candidates`` path remains the
only admission authority used by the production memory builder.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from web_agent.data.recovery_transitions import build_recovery_transition_index
from web_agent.memory.eligibility import (
    EXCLUSION_REASONS,
    ELIGIBILITY_POLICY_VERSION,
    EligibilitySelection,
    EligibleMemoryCandidate,
    ProvenanceManifest,
    canonical_memory_id,
    is_explicitly_non_admitted,
)
from web_agent.memory.frozen_store import FrozenMemoryStore
from web_agent.memory.manifest import ManifestError, canonical_sha256, sha256_file
from web_agent.memory.verification import (
    FINAL_TASK_VERIFICATION_SCHEMA_VERSION,
    P4_LABEL_REVIEW_SCHEMA_VERSION,
    RECOVERY_VERIFICATION_SCHEMA_VERSION,
    VerificationEvidenceError,
    canonical_p4_memory_source_material,
    recovery_action_evidence_sha256,
    recovery_state_evidence_sha256,
    validate_p4_label_review_evidence,
    validate_provenance_verification_evidence,
)


PREPARATION_PACKAGE_SCHEMA_VERSION = "table2-p4-preparation-package-v5"
PREPARATION_AUDIT_SCHEMA_VERSION = "table2-p4-candidate-audit-v1"
PREPARATION_READ_LEDGER_SCHEMA_VERSION = "table2-p4-read-ledger-v1"
REVIEW_QUEUE_SCHEMA_VERSION = "table2-p4-review-provenance-input-queue-v3"
REVIEW_QUEUE_ROW_SCHEMA_VERSION = "table2-p4-review-provenance-input-v3"
TRANSFER_SCHEMA_VERSION = "table2-memory-cross-host-transfer-v1"
P4_SOURCE_AUTHORITY_SCHEMA_VERSION = "table2-p4-source-authority-v1"
P4_SOURCE_AUTHORITY_FILE_NAME = "source_authority.json"
P4_REGISTERED_SOURCE_AUTHORITY_SHA256 = (
    "c0870409ac84f46fec581d1e9ea94f3cd66e30716995752b94ea710f99284d34"
)
P4_DATASET_ARTIFACT_BINDING_SCHEMA_VERSION = (
    "table2-p4-dataset-artifact-binding-v1"
)

_PREPARATION_FILES = (
    "candidate_audit.json",
    "read_ledger.json",
    "review_queue.jsonl",
    P4_SOURCE_AUTHORITY_FILE_NAME,
)
_STORE_FILES = (
    "manifest.json",
    "manifest.sha256",
    "embeddings.npy",
    "items.jsonl",
    "verification_evidence.json",
    "calibration_evidence.json",
    "threshold_calibration.json",
)
_ZERO_READ_FIELDS = (
    "validation_rows_read",
    "test_rows_read",
    "locked_test_rows_read",
)
_PENDING_EXTERNAL_FIELDS = (
    "provenance_valid",
    "final_task_success",
    "exact_duplicate_key",
    "near_duplicate_cluster_id",
    "duplicate_cluster_namespace_id",
    "recovery_verification",
    "final_task_verification",
    "verification_evidence_sha256",
    "p4_label_review",
    "p4_label_review_evidence_sha256",
)
_QUEUE_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "queue_id",
        "status",
        "source_split",
        "source_dataset_role",
        "source_transition_kind",
        "source_review_status",
        "source_sample_id",
        "recovery_sample_id",
        "canonical_task_id",
        "episode_id",
        "source_step_index",
        "source_record_sha256",
        "transition_sha256",
        "duplicate_audit_material",
        "memory_item_source_material",
        "declared_training_labels",
        "causal_material",
        "required_external_evidence",
        "required_recovery_schema",
        "required_final_task_schema",
        "required_p4_label_review_schema",
        "eligibility_notice",
    }
)
_DECLARED_LABEL_FIELDS = frozenset(
    {
        "outcome_label",
        "failure_type_4",
        "action_type",
        "recovery_strategy",
        "recovery_success",
        "memory_update_flag",
        "reflection_text",
    }
)
_CAUSAL_MATERIAL_FIELDS = frozenset(
    {
        "failure_state_reference",
        "pre_recovery_state_sha256",
        "executed_recovery_action",
        "recovery_action_value",
        "executed_recovery_action_sha256",
        "post_recovery_state_reference",
        "post_recovery_state_sha256",
    }
)
_DUPLICATE_AUDIT_MATERIAL_FIELDS = frozenset(
    {"task_description", "website_domain"}
)
_READ_LEDGER_FIELDS = frozenset(
    {
        "schema_version",
        "source_split",
        *_ZERO_READ_FIELDS,
        "reader_contract",
        "train_json_files",
        "train_rows_read",
        "source_authority_sha256",
        "candidate_state_artifact_references",
        "candidate_state_artifact_unique_hashes",
        "candidate_state_artifacts_sha256",
    }
)
_AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "evidence_role",
        "dataset_id",
        "dataset_version",
        "source_split",
        *_ZERO_READ_FIELDS,
        "input_rows",
        "source_rows",
        "records_sha256",
        "transition_report",
        "local_gate_counts",
        "review_queue_rows",
        "suppressed_queue_rows_due_to_fatal_errors",
        "fatal_error_count",
        "fatal_errors",
        "independent_verification_records_created",
        "provenance_manifest_created",
        "eligible_memory_items_claimed",
    }
)
_READER_CONTRACT = (
    "Only explicit split_train.json and optional supplement_train.json plus "
    "state artifacts referenced by their local candidates are read."
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


class P4PreparationError(ValueError):
    """A preparation input or integrity contract is invalid."""


def _source_authority_payload(
    path: str | Path,
    *,
    source_descriptors: Sequence[Mapping[str, Any]],
    dataset_id: str,
    dataset_version: str,
) -> dict[str, Any]:
    """Load an explicit source authority and match every supplied train source."""

    expected_sources = [
        {
            "file_name": str(row["file_name"]),
            "records": int(row["records"]),
            "role": str(row["role"]),
            "sha256": str(row["sha256"]),
        }
        for row in source_descriptors
    ]
    authority_path = Path(path)
    if authority_path.is_symlink() or not authority_path.is_file():
        raise P4PreparationError(
            "P4 source authority must be a regular non-symlink file"
        )
    payload: object = _load_json(authority_path)
    required = {
        "authority_id",
        "authority_version",
        "dataset_id",
        "dataset_version",
        "schema_version",
        "sources",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise P4PreparationError("P4 source-authority fields differ from schema")
    if payload.get("schema_version") != P4_SOURCE_AUTHORITY_SCHEMA_VERSION:
        raise P4PreparationError("unsupported P4 source-authority schema")
    for field in ("authority_id", "authority_version"):
        _nonempty(payload.get(field), field=f"source_authority.{field}")
    if payload.get("dataset_id") != dataset_id or payload.get(
        "dataset_version"
    ) != dataset_version:
        raise P4PreparationError(
            "dataset identity differs from the P4 source authority"
        )
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise P4PreparationError("P4 source authority has no sources")
    normalized: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    for index, row in enumerate(sources):
        if not isinstance(row, Mapping) or set(row) != {
            "file_name", "records", "role", "sha256"
        }:
            raise P4PreparationError(
                f"source_authority.sources[{index}] fields differ from schema"
            )
        role = _nonempty(row.get("role"), field=f"source_authority.sources[{index}].role")
        if role in seen_roles or role not in {
            "original_gold", "retry_abort_supplement_v2"
        }:
            raise P4PreparationError(f"invalid source-authority role: {role}")
        seen_roles.add(role)
        file_name = _nonempty(
            row.get("file_name"), field=f"source_authority.sources[{index}].file_name"
        )
        records = _nonnegative_int(
            row.get("records"), field=f"source_authority.sources[{index}].records"
        )
        digest = _require_sha256(
            row.get("sha256"), field=f"source_authority.sources[{index}].sha256"
        )
        normalized.append(
            {"file_name": file_name, "records": records, "role": role, "sha256": digest}
        )
    if normalized != expected_sources:
        raise P4PreparationError(
            "training source bytes/counts/roles differ from P4 source authority"
        )
    return dict(payload)


@dataclass(frozen=True, slots=True)
class P4PreparationPackage:
    """A written, hash-closed review-input package."""

    root: Path
    manifest: Mapping[str, Any]

    @property
    def status(self) -> str:
        return str(self.manifest["package_status"])

    @property
    def candidate_count(self) -> int:
        return int(self.manifest["review_queue_rows"])


def _json_pairs_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise P4PreparationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_pairs_without_duplicates,
        )
    except FileNotFoundError:
        raise P4PreparationError(f"missing JSON file: {path}") from None
    except json.JSONDecodeError as error:
        raise P4PreparationError(f"invalid JSON file {path}: {error}") from error


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise P4PreparationError(f"training source must be a JSON array: {path}")
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(payload):
        if not isinstance(value, dict):
            raise P4PreparationError(
                f"training source row {index} is not a JSON object: {path}"
            )
        rows.append(value)
    return rows


def _file_descriptor(path: Path) -> dict[str, int | str]:
    if path.is_symlink() or not path.is_file():
        raise P4PreparationError(f"expected a regular non-symlink file: {path}")
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise P4PreparationError(f"temporary output already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _dataset_artifacts_sha256(
    *, source_authority_sha256: str, candidate_state_artifacts_sha256: str
) -> str:
    return canonical_sha256(
        {
            "schema_version": P4_DATASET_ARTIFACT_BINDING_SCHEMA_VERSION,
            "source_authority_sha256": _require_sha256(
                source_authority_sha256, field="source_authority_sha256"
            ),
            "candidate_state_artifacts_sha256": _require_sha256(
                candidate_state_artifacts_sha256,
                field="candidate_state_artifacts_sha256",
            ),
        }
    )


def _public_value(value: Any) -> Any:
    """Match the production builder's logical-record hash projection."""

    if isinstance(value, Mapping):
        return {
            str(key): _public_value(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    if isinstance(value, tuple):
        return [_public_value(item) for item in value]
    return value


def _resolve_train_source(
    json_path: str | Path,
    data_root: str | Path,
    *,
    expected_name: str,
    role: str,
) -> tuple[Path, Path]:
    root = Path(data_root).resolve()
    requested_source = Path(json_path)
    if requested_source.is_symlink():
        raise P4PreparationError(f"{role} training JSON must not be a symlink")
    source = requested_source.resolve()
    if not root.is_dir():
        raise P4PreparationError(f"{role} data root is not a directory: {root}")
    if source.name != expected_name:
        raise P4PreparationError(
            f"{role} accepts only {expected_name}; got {source.name!r}"
        )
    try:
        source.relative_to(root)
    except ValueError as error:
        raise P4PreparationError(
            f"{role} training JSON is outside its registered data root"
        ) from error
    forbidden_parts = {
        "locked_benchmark_mount",
        "locked-test",
        "locked_test",
    }
    if any(part.lower() in forbidden_parts for part in source.parts):
        raise P4PreparationError(
            f"{role} training JSON resolves beneath a locked path"
        )
    if not source.is_file():
        raise P4PreparationError(f"missing {role} training JSON: {source}")
    return source, root


def _view(record: Mapping[str, Any], *, index: int) -> tuple[dict, dict, dict]:
    inputs = record.get("inputs")
    labels = record.get("labels")
    meta = record.get("meta")
    if not all(isinstance(value, dict) for value in (inputs, labels, meta)):
        raise P4PreparationError(
            f"record[{index}] must contain current Gold inputs/labels/meta objects"
        )
    return inputs, labels, meta


def _nonempty(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise P4PreparationError(f"missing required field: {field}")
    return text


def _require_sha256(value: object, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise P4PreparationError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise P4PreparationError(f"{field} must be a non-negative JSON integer")
    return value


def _require_zero_reads(value: Mapping[str, Any], *, context: str) -> None:
    for field in _ZERO_READ_FIELDS:
        if type(value.get(field)) is not int or value[field] != 0:
            raise P4PreparationError(f"{context} requires explicit {field}=0")


def _strict_step(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise P4PreparationError(f"{field} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise P4PreparationError(
            f"{field} must be a non-negative integer"
        ) from error
    if result < 0:
        raise P4PreparationError(f"{field} must be a non-negative integer")
    return result


def _declared_split_is_train(
    record: Mapping[str, Any], meta: Mapping[str, Any]
) -> bool:
    declarations: list[str] = []
    for authority in (record, meta):
        for field in ("split", "source_split", "partition"):
            value = authority.get(field)
            if value not in (None, ""):
                declarations.append(str(value).strip().lower())
    return all(value == "train" for value in declarations)


def _is_non_admitted(meta: Mapping[str, Any]) -> bool:
    authorities: list[tuple[str, Mapping[str, Any]]] = [("meta", meta)]
    overlay = meta.get("targeted_review_overlay")
    if overlay is not None:
        if not isinstance(overlay, Mapping):
            raise P4PreparationError(
                "meta.targeted_review_overlay must be an object"
            )
        authorities.append(("meta.targeted_review_overlay", overlay))
    for authority_name, authority in authorities:
        if str(authority.get("review_status", "")).strip().lower() in (
            _NON_ADMISSION_TEXT_VALUES
        ):
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
                raise P4PreparationError(
                    f"{authority_name}.{field} must be a JSON boolean"
                )
            if value:
                return True
        for field in _ADMISSION_BOOLEAN_FIELDS:
            if field not in authority:
                continue
            value = authority[field]
            if type(value) is not bool:
                raise P4PreparationError(
                    f"{authority_name}.{field} must be a JSON boolean"
                )
            if not value:
                return True
    return False


def _annotate_source_rows(
    records: Sequence[dict[str, Any]],
    *,
    role: str,
    data_root: Path,
    direct_recovery_transition: bool,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        copied = deepcopy(record)
        _, _, meta = _view(copied, index=index)
        meta["_source_dataset"] = role
        meta["_data_root"] = str(data_root)
        # The caller-assigned, source-authority-bound role is authoritative;
        # raw rows cannot spoof the private direct-transition marker.
        meta["_direct_recovery_transition"] = bool(direct_recovery_transition)
        annotated.append(copied)
    return annotated


def _local_gate_reason(
    labels: Mapping[str, Any],
    meta: Mapping[str, Any],
    *,
    sample_id: str,
) -> str | None:
    if _is_non_admitted(meta):
        return "quarantined_or_non_admitted"
    review_status = meta.get("review_status")
    if not isinstance(review_status, str) or not review_status.strip():
        raise P4PreparationError(
            f"{sample_id}.review_status must be a non-empty string"
        )
    normalized_review_status = review_status.strip().lower()
    if normalized_review_status not in {"approved", "pending"}:
        raise P4PreparationError(
            f"{sample_id}.review_status is not registered: {review_status!r}"
        )
    if "memory_update_flag" not in labels:
        return "memory_update_flag_missing"
    memory_flag = labels.get("memory_update_flag")
    if type(memory_flag) is not bool:
        raise P4PreparationError(
            f"{sample_id}.memory_update_flag must be a JSON boolean"
        )
    if not memory_flag:
        return "memory_update_flag_false"
    if "recovery_success" not in labels:
        return "recovery_success_missing"
    recovery_success = labels.get("recovery_success")
    if recovery_success is not None and type(recovery_success) is not bool:
        raise P4PreparationError(
            f"{sample_id}.recovery_success must be true, false, or null"
        )
    if recovery_success is not True:
        return "recovery_not_successful"
    return None


def _queue_row(
    record: Mapping[str, Any],
    *,
    index: int,
    source_role: str,
    transition: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], dict[str, Any]]]:
    inputs, labels, meta = _view(record, index=index)
    direct_transition = source_role == "retry_abort_supplement_v2"
    if meta.get("_direct_recovery_transition") is not direct_transition:
        raise P4PreparationError(
            "potential memory source role/direct-transition marker mismatch"
        )
    transition_kind = (
        "direct_recovery_from_observed_failure_state"
        if direct_transition
        else "adjacent_failure_then_recovery"
    )
    sample_id = _nonempty(
        meta.get("sample_id") or meta.get("row_id") or meta.get("id"),
        field=f"record[{index}].sample_id",
    )
    if transition is None:
        raise P4PreparationError(
            f"potential memory {sample_id} lacks a causal recovery transition"
        )
    if transition.get("source_sample_id") != sample_id:
        raise P4PreparationError(
            f"potential memory {sample_id} is bound to another transition source"
        )
    task_id = _nonempty(
        meta.get("task_id") or meta.get("original_task_id"),
        field=f"{sample_id}.meta.task_id",
    )
    if transition.get("task_id") != task_id:
        raise P4PreparationError(
            f"potential memory {sample_id} task differs from its transition"
        )
    episode_id = _nonempty(
        meta.get("episode_id") or meta.get("trajectory_id") or task_id,
        field=f"{sample_id}.meta.episode_id",
    )
    source_step_index = _strict_step(
        meta.get("step_index", transition.get("source_step_index")),
        field=f"{sample_id}.step_index",
    )
    if _strict_step(
        transition.get("source_step_index"),
        field=f"{sample_id}.transition.source_step_index",
    ) != source_step_index:
        raise P4PreparationError(
            f"potential memory {sample_id} step differs from its transition"
        )
    source_review_status = str(meta.get("review_status", "")).strip().lower()
    if source_review_status not in {"approved", "pending"}:
        raise P4PreparationError(
            f"potential memory {sample_id} has no registered review status"
        )
    outcome = _nonempty(
        labels.get("outcome_label"), field=f"{sample_id}.outcome_label"
    )
    failure_type = _nonempty(
        labels.get("failure_type_4"), field=f"{sample_id}.failure_type_4"
    )
    if not direct_transition and (outcome != "FAILURE" or failure_type == "NONE"):
        raise P4PreparationError(
            f"potential memory {sample_id} is not an adjacent observed failure"
        )
    if direct_transition and outcome not in {"SUCCESS", "FAILURE"}:
        raise P4PreparationError(
            f"potential memory {sample_id} has invalid direct-recovery outcome"
        )
    strategy = _nonempty(
        labels.get("recovery_strategy"), field=f"{sample_id}.recovery_strategy"
    )
    if strategy == "NONE" or transition.get("recovery_strategy") != strategy:
        raise P4PreparationError(
            f"potential memory {sample_id} lacks a matching recovery strategy"
        )
    if "recovery_attempted" in labels and labels["recovery_attempted"] is not True:
        raise P4PreparationError(
            f"potential memory {sample_id} was not an executed recovery"
        )
    if transition.get("recovery_success") is not True:
        raise P4PreparationError(
            f"potential memory {sample_id} transition is not successful"
        )
    recovery_sample_id = _nonempty(
        transition.get("recovery_sample_id"),
        field=f"{sample_id}.transition.recovery_sample_id",
    )
    executed_action = _nonempty(
        transition.get("executed_recovery_action"),
        field=f"{sample_id}.transition.executed_recovery_action",
    )
    pre_hash = recovery_state_evidence_sha256(
        transition, field="failure_state"
    )
    action_hash = recovery_action_evidence_sha256(transition)
    post_hash = recovery_state_evidence_sha256(
        transition, field="post_recovery_state"
    )
    public_record = _public_value(record)
    public_transition = _public_value(
        {key: value for key, value in transition.items() if key != "source_meta"}
    )
    source_record_sha256 = canonical_sha256(public_record)
    transition_sha256 = canonical_sha256(public_transition)
    duplicate_audit_material = {
        "task_description": _nonempty(
            inputs.get("task_description"),
            field=f"{sample_id}.inputs.task_description",
        ),
        "website_domain": str(inputs.get("website_domain") or "").strip(),
    }
    reflection_text = str(labels.get("reflection_text") or "")
    source_action_type = _nonempty(
        labels.get("action_type"), field=f"{sample_id}.action_type"
    )
    source_material = canonical_p4_memory_source_material(
        {
            "source_dataset_role": source_role,
            "source_review_status": source_review_status,
            "source_sample_id": sample_id,
            "recovery_sample_id": recovery_sample_id,
            "canonical_task_id": task_id,
            "episode_id": episode_id,
            "source_step_index": source_step_index,
            "source_record_sha256": source_record_sha256,
            "transition_sha256": transition_sha256,
            "source_transition_kind": transition_kind,
            "task_description": duplicate_audit_material["task_description"],
            "website_domain": duplicate_audit_material["website_domain"],
            "observed_failure_basis": (
                "source_attested_direct_pre_recovery_state"
                if direct_transition
                else "source_outcome_failure"
            ),
            "source_outcome_label": outcome,
            "source_failure_type": failure_type,
            "source_action_type": source_action_type,
            "failure_type": "UNAVAILABLE" if direct_transition else failure_type,
            "failure_type_available": not direct_transition,
            "failed_action": "UNAVAILABLE" if direct_transition else source_action_type,
            "failed_action_available": not direct_transition,
            "recovery_strategy": strategy,
            "reflection_text": reflection_text,
            "pre_recovery_state_sha256": pre_hash,
            "executed_recovery_action": executed_action,
            "recovery_action_value": str(
                transition.get("recovery_action_value") or ""
            ),
            "executed_recovery_action_sha256": action_hash,
            "post_recovery_state_sha256": post_hash,
        }
    )
    identity = {
        "source_dataset_role": source_role,
        "source_transition_kind": transition_kind,
        "source_review_status": source_review_status,
        "source_sample_id": sample_id,
        "recovery_sample_id": recovery_sample_id,
        "canonical_task_id": task_id,
        "episode_id": episode_id,
        "source_step_index": source_step_index,
        "source_record_sha256": source_record_sha256,
        "pre_recovery_state_sha256": pre_hash,
        "executed_recovery_action_sha256": action_hash,
        "post_recovery_state_sha256": post_hash,
        "duplicate_audit_material": duplicate_audit_material,
        "reflection_text": reflection_text,
        "memory_item_source_material_sha256": canonical_sha256(source_material),
    }
    queue_id = f"p4-review-{canonical_sha256(identity)[:24]}"
    pending = {field: None for field in _PENDING_EXTERNAL_FIELDS}
    row = {
        "schema_version": REVIEW_QUEUE_ROW_SCHEMA_VERSION,
        "queue_id": queue_id,
        "status": "REQUIRES_INDEPENDENT_REVIEW_AND_JOINT_DUPLICATE_AUDIT",
        "source_split": "train",
        "source_dataset_role": source_role,
        "source_transition_kind": transition_kind,
        "source_review_status": source_review_status,
        "source_sample_id": sample_id,
        "recovery_sample_id": recovery_sample_id,
        "canonical_task_id": task_id,
        "episode_id": episode_id,
        "source_step_index": source_step_index,
        "source_record_sha256": identity["source_record_sha256"],
        "transition_sha256": transition_sha256,
        "duplicate_audit_material": duplicate_audit_material,
        "memory_item_source_material": source_material,
        "declared_training_labels": {
            "outcome_label": outcome,
            "failure_type_4": failure_type,
            "action_type": str(labels.get("action_type") or ""),
            "recovery_strategy": strategy,
            "recovery_success": True,
            "memory_update_flag": True,
            "reflection_text": reflection_text,
        },
        "causal_material": {
            "failure_state_reference": str(transition.get("failure_state")),
            "pre_recovery_state_sha256": pre_hash,
            "executed_recovery_action": executed_action,
            "recovery_action_value": str(
                transition.get("recovery_action_value") or ""
            ),
            "executed_recovery_action_sha256": action_hash,
            "post_recovery_state_reference": str(
                transition.get("post_recovery_state")
            ),
            "post_recovery_state_sha256": post_hash,
        },
        "required_external_evidence": pending,
        "required_recovery_schema": RECOVERY_VERIFICATION_SCHEMA_VERSION,
        "required_final_task_schema": FINAL_TASK_VERIFICATION_SCHEMA_VERSION,
        "required_p4_label_review_schema": P4_LABEL_REVIEW_SCHEMA_VERSION,
        "eligibility_notice": (
            "This is deterministic review input, not independent evidence or an "
            "eligible memory item. The production provenance manifest and "
            "existing strict memory builder must validate it after external review."
        ),
    }
    artifact_rows = (
        {
            "source_dataset_role": source_role,
            "reference": str(transition.get("failure_state")),
            "sha256": pre_hash,
        },
        {
            "source_dataset_role": source_role,
            "reference": str(transition.get("post_recovery_state")),
            "sha256": post_hash,
        },
    )
    return row, artifact_rows


def _prepare_p4_candidate_audit(
    *,
    gold_train_json: str | Path,
    gold_data_root: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    dataset_version: str,
    source_authority_path: str | Path,
    supplement_train_json: str | Path | None = None,
    supplement_data_root: str | Path | None = None,
) -> P4PreparationPackage:
    """Audit deterministic train-only gates and write an external-review queue.

    A queue row means only that the source labels and local causal material pass
    the pre-review gates.  It is *not* an eligible memory.  Any structural error
    suppresses the complete queue and marks the package ``FAIL``.
    """

    normalized_dataset_id = str(dataset_id).strip()
    normalized_dataset_version = str(dataset_version).strip()
    if not normalized_dataset_id or not normalized_dataset_version:
        raise P4PreparationError("dataset_id and dataset_version must be non-empty")
    if (supplement_train_json is None) != (supplement_data_root is None):
        raise P4PreparationError(
            "supplement_train_json and supplement_data_root must be supplied together"
        )
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise P4PreparationError(
            f"refusing to overwrite P4 preparation package: {destination}"
        )

    gold_json, gold_root = _resolve_train_source(
        gold_train_json,
        gold_data_root,
        expected_name="split_train.json",
        role="original_gold",
    )
    raw_gold = _load_json_array(gold_json)
    source_descriptors: list[dict[str, Any]] = [
        {
            "role": "original_gold",
            "file_name": gold_json.name,
            "records": len(raw_gold),
            **_file_descriptor(gold_json),
        }
    ]
    records = _annotate_source_rows(
        raw_gold,
        role="original_gold",
        data_root=gold_root,
        direct_recovery_transition=False,
    )
    if supplement_train_json is not None and supplement_data_root is not None:
        supplement_json, supplement_root = _resolve_train_source(
            supplement_train_json,
            supplement_data_root,
            expected_name="supplement_train.json",
            role="retry_abort_supplement_v2",
        )
        raw_supplement = _load_json_array(supplement_json)
        source_descriptors.append(
            {
                "role": "retry_abort_supplement_v2",
                "file_name": supplement_json.name,
                "records": len(raw_supplement),
                **_file_descriptor(supplement_json),
            }
        )
        records.extend(
            _annotate_source_rows(
                raw_supplement,
                role="retry_abort_supplement_v2",
                data_root=supplement_root,
                direct_recovery_transition=True,
            )
        )

    source_authority = _source_authority_payload(
        source_authority_path,
        source_descriptors=source_descriptors,
        dataset_id=normalized_dataset_id,
        dataset_version=normalized_dataset_version,
    )

    for index, record in enumerate(records):
        _, _, meta = _view(record, index=index)
        if not _declared_split_is_train(record, meta):
            raise P4PreparationError(
                "refusing a training source containing a row that declares "
                "validation, test, or another non-training split"
            )

    records_sha256 = canonical_sha256(_public_value(records))
    transitions, transition_report = build_recovery_transition_index(records)
    source_by_sample: dict[str, str] = {}
    root_by_source = {"original_gold": gold_root}
    if supplement_train_json is not None and supplement_data_root is not None:
        root_by_source["retry_abort_supplement_v2"] = Path(
            supplement_data_root
        ).resolve()

    fatal_errors: list[str] = []
    source_counts: Counter[str] = Counter(
        {
            str(descriptor["role"]): int(descriptor["records"])
            for descriptor in source_descriptors
        }
    )
    gate_counts: Counter[str] = Counter()
    sample_ids: set[str] = set()
    for index, record in enumerate(records):
        try:
            _, _, meta = _view(record, index=index)
            sample_id = _nonempty(
                meta.get("sample_id") or meta.get("row_id") or meta.get("id"),
                field=f"record[{index}].sample_id",
            )
            if sample_id in sample_ids:
                raise P4PreparationError(f"duplicate source sample ID: {sample_id}")
            sample_ids.add(sample_id)
            source_role = _nonempty(
                meta.get("_source_dataset"),
                field=f"{sample_id}._source_dataset",
            )
            if source_role not in root_by_source:
                raise P4PreparationError(
                    f"{sample_id} has an unregistered source role: {source_role}"
                )
            if not _declared_split_is_train(record, meta):
                raise P4PreparationError(
                    f"{sample_id} declares a validation/test/non-training split"
                )
            source_by_sample[sample_id] = source_role
        except P4PreparationError as error:
            fatal_errors.append(str(error))

    for sample_id, transition in transitions.items():
        source_role = source_by_sample.get(sample_id)
        if source_role is None:
            fatal_errors.append(
                f"transition source {sample_id} has no registered training row"
            )
            continue
        transition["_data_root"] = str(root_by_source[source_role])

    queue_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        try:
            _, labels, meta = _view(record, index=index)
            sample_id = _nonempty(
                meta.get("sample_id") or meta.get("row_id") or meta.get("id"),
                field=f"record[{index}].sample_id",
            )
            if sample_id not in source_by_sample:
                continue
            reason = _local_gate_reason(labels, meta, sample_id=sample_id)
            if reason is not None:
                gate_counts[reason] += 1
                continue
            row, row_artifacts = _queue_row(
                record,
                index=index,
                source_role=source_by_sample[sample_id],
                transition=transitions.get(sample_id),
            )
            queue_rows.append(row)
            artifact_rows.extend(row_artifacts)
            gate_counts["requires_independent_review"] += 1
        except (P4PreparationError, VerificationEvidenceError, ValueError) as error:
            fatal_errors.append(str(error))

    queue_rows.sort(
        key=lambda row: (row["source_dataset_role"], row["source_sample_id"])
    )
    artifact_rows.sort(
        key=lambda row: (row["source_dataset_role"], row["reference"], row["sha256"])
    )
    prepared_before_failure = len(queue_rows)
    if fatal_errors:
        queue_rows = []
        package_status = "FAIL"
    else:
        package_status = "REVIEW_REQUIRED"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.building-", dir=destination.parent
        )
    )
    try:
        _atomic_write_json(
            temporary / P4_SOURCE_AUTHORITY_FILE_NAME,
            source_authority,
        )
        queue_path = temporary / "review_queue.jsonl"
        with queue_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in queue_rows:
                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                stream.write("\n")

        read_ledger = {
            "schema_version": PREPARATION_READ_LEDGER_SCHEMA_VERSION,
            "source_split": "train",
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "reader_contract": _READER_CONTRACT,
            "train_json_files": source_descriptors,
            "train_rows_read": len(records),
            "source_authority_sha256": sha256_file(
                temporary / P4_SOURCE_AUTHORITY_FILE_NAME
            ),
            "candidate_state_artifact_references": len(artifact_rows),
            "candidate_state_artifact_unique_hashes": len(
                {row["sha256"] for row in artifact_rows}
            ),
            "candidate_state_artifacts_sha256": canonical_sha256(artifact_rows),
        }
        _atomic_write_json(temporary / "read_ledger.json", read_ledger)

        audit = {
            "schema_version": PREPARATION_AUDIT_SCHEMA_VERSION,
            "status": package_status,
            "evidence_role": "PREPARATION_ONLY_NOT_PROVENANCE",
            "dataset_id": normalized_dataset_id,
            "dataset_version": normalized_dataset_version,
            "source_split": "train",
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "input_rows": len(records),
            "source_rows": dict(sorted(source_counts.items())),
            "records_sha256": records_sha256,
            "transition_report": transition_report,
            "local_gate_counts": dict(sorted(gate_counts.items())),
            "review_queue_rows": len(queue_rows),
            "suppressed_queue_rows_due_to_fatal_errors": (
                prepared_before_failure if fatal_errors else 0
            ),
            "fatal_error_count": len(fatal_errors),
            "fatal_errors": fatal_errors,
            "independent_verification_records_created": 0,
            "provenance_manifest_created": False,
            "eligible_memory_items_claimed": 0,
        }
        _atomic_write_json(temporary / "candidate_audit.json", audit)

        descriptors = {
            name: _file_descriptor(temporary / name) for name in _PREPARATION_FILES
        }
        manifest = {
            "schema_version": PREPARATION_PACKAGE_SCHEMA_VERSION,
            "package_status": package_status,
            "source_split": "train",
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "dataset_id": normalized_dataset_id,
            "dataset_version": normalized_dataset_version,
            "records_sha256": records_sha256,
            "source_authority_sha256": sha256_file(
                temporary / P4_SOURCE_AUTHORITY_FILE_NAME
            ),
            "dataset_artifacts_sha256": _dataset_artifacts_sha256(
                source_authority_sha256=sha256_file(
                    temporary / P4_SOURCE_AUTHORITY_FILE_NAME
                ),
                candidate_state_artifacts_sha256=read_ledger[
                    "candidate_state_artifacts_sha256"
                ],
            ),
            "review_queue_schema_version": REVIEW_QUEUE_SCHEMA_VERSION,
            "review_queue_rows": len(queue_rows),
            "review_queue_sha256": canonical_sha256(queue_rows),
            "independent_verification_records_created": 0,
            "provenance_manifest_created": False,
            "eligible_memory_items_claimed": 0,
            "files": descriptors,
        }
        _atomic_write_json(temporary / "preparation_manifest.json", manifest)
        (temporary / "preparation_manifest.sha256").write_text(
            sha256_file(temporary / "preparation_manifest.json") + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return validate_p4_preparation_package(destination)


def prepare_p4_candidate_audit(
    *,
    gold_train_json: str | Path,
    gold_data_root: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    dataset_version: str,
    source_authority_path: str | Path,
    supplement_train_json: str | Path | None = None,
    supplement_data_root: str | Path | None = None,
) -> P4PreparationPackage:
    """Build a production PC-01 preparation from the registered train corpus."""

    authority = Path(source_authority_path)
    if sha256_file(authority) != P4_REGISTERED_SOURCE_AUTHORITY_SHA256:
        raise P4PreparationError(
            "production P4 preparation requires the registered PC-01 source authority"
        )
    return _prepare_p4_candidate_audit(
        gold_train_json=gold_train_json,
        gold_data_root=gold_data_root,
        output_dir=output_dir,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        source_authority_path=authority,
        supplement_train_json=supplement_train_json,
        supplement_data_root=supplement_data_root,
    )


def _prepare_p4_candidate_audit_fixture(
    *,
    gold_train_json: str | Path,
    gold_data_root: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    dataset_version: str,
    source_authority_path: str | Path,
    supplement_train_json: str | Path | None = None,
    supplement_data_root: str | Path | None = None,
) -> P4PreparationPackage:
    """Explicit non-production seam for synthetic unit fixtures only."""

    return _prepare_p4_candidate_audit(
        gold_train_json=gold_train_json,
        gold_data_root=gold_data_root,
        output_dir=output_dir,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        source_authority_path=source_authority_path,
        supplement_train_json=supplement_train_json,
        supplement_data_root=supplement_data_root,
    )


def _validate_count_mapping(value: object, *, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise P4PreparationError(f"{field} must be an object")
    normalized: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        key = _nonempty(raw_key, field=f"{field} key")
        normalized[key] = _nonnegative_int(raw_count, field=f"{field}.{key}")
    return normalized


def _validate_source_descriptors(value: object) -> dict[str, int]:
    if not isinstance(value, list) or len(value) not in {1, 2}:
        raise P4PreparationError(
            "read ledger must describe Gold train and at most one supplement train"
        )
    expected_names = {
        "original_gold": "split_train.json",
        "retry_abort_supplement_v2": "supplement_train.json",
    }
    result: dict[str, int] = {}
    for index, descriptor in enumerate(value):
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "role",
            "file_name",
            "records",
            "sha256",
            "bytes",
        }:
            raise P4PreparationError(
                f"read ledger train_json_files[{index}] has invalid fields"
            )
        role = _nonempty(
            descriptor.get("role"), field=f"train_json_files[{index}].role"
        )
        if role not in expected_names or role in result:
            raise P4PreparationError(
                f"read ledger contains an invalid or duplicate source role: {role}"
            )
        if descriptor.get("file_name") != expected_names[role]:
            raise P4PreparationError(
                f"read ledger {role} does not identify its exact training filename"
            )
        result[role] = _nonnegative_int(
            descriptor.get("records"), field=f"train_json_files[{index}].records"
        )
        _nonnegative_int(
            descriptor.get("bytes"), field=f"train_json_files[{index}].bytes"
        )
        _require_sha256(
            descriptor.get("sha256"), field=f"train_json_files[{index}].sha256"
        )
    if "original_gold" not in result:
        raise P4PreparationError("read ledger omits the Gold training source")
    return result


def _validate_read_ledger(
    payload: object,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, int]:
    if not isinstance(payload, Mapping) or set(payload) != set(_READ_LEDGER_FIELDS):
        raise P4PreparationError("P4 read-ledger fields differ from the schema")
    if payload.get("schema_version") != PREPARATION_READ_LEDGER_SCHEMA_VERSION:
        raise P4PreparationError("unsupported P4 read-ledger schema")
    if payload.get("source_split") != "train":
        raise P4PreparationError("P4 read ledger must be train-only")
    _require_zero_reads(payload, context="P4 read ledger")
    if payload.get("reader_contract") != _READER_CONTRACT:
        raise P4PreparationError("P4 read-ledger reader contract changed")
    sources = _validate_source_descriptors(payload.get("train_json_files"))
    train_rows = _nonnegative_int(
        payload.get("train_rows_read"), field="read_ledger.train_rows_read"
    )
    if train_rows != sum(sources.values()):
        raise P4PreparationError("read-ledger training row total is inconsistent")
    authority_sha256 = _require_sha256(
        payload.get("source_authority_sha256"),
        field="read_ledger.source_authority_sha256",
    )
    if authority_sha256 != manifest.get("source_authority_sha256"):
        raise P4PreparationError("read-ledger/source-authority binding mismatch")
    references = _nonnegative_int(
        payload.get("candidate_state_artifact_references"),
        field="read_ledger.candidate_state_artifact_references",
    )
    unique_hashes = _nonnegative_int(
        payload.get("candidate_state_artifact_unique_hashes"),
        field="read_ledger.candidate_state_artifact_unique_hashes",
    )
    if unique_hashes > references:
        raise P4PreparationError("read-ledger unique artifact count exceeds references")
    _require_sha256(
        payload.get("candidate_state_artifacts_sha256"),
        field="read_ledger.candidate_state_artifacts_sha256",
    )
    if manifest["package_status"] == "REVIEW_REQUIRED" and references != (
        2 * int(manifest["review_queue_rows"])
    ):
        raise P4PreparationError(
            "read-ledger artifact references do not match the review queue"
        )
    return sources


def _validate_audit(
    payload: object,
    *,
    manifest: Mapping[str, Any],
    source_records: Mapping[str, int],
    train_rows_read: int,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != set(_AUDIT_FIELDS):
        raise P4PreparationError("P4 candidate-audit fields differ from the schema")
    if payload.get("schema_version") != PREPARATION_AUDIT_SCHEMA_VERSION:
        raise P4PreparationError("unsupported P4 candidate-audit schema")
    if payload.get("status") != manifest["package_status"]:
        raise P4PreparationError("candidate-audit/package status mismatch")
    if payload.get("evidence_role") != "PREPARATION_ONLY_NOT_PROVENANCE":
        raise P4PreparationError("candidate audit improperly claims evidence authority")
    if payload.get("source_split") != "train":
        raise P4PreparationError("candidate audit must be train-only")
    _require_zero_reads(payload, context="P4 candidate audit")
    for field in ("dataset_id", "dataset_version", "records_sha256"):
        if payload.get(field) != manifest[field]:
            raise P4PreparationError(f"candidate-audit/package mismatch at {field}")
    _require_sha256(payload.get("records_sha256"), field="audit.records_sha256")
    input_rows = _nonnegative_int(
        payload.get("input_rows"), field="audit.input_rows"
    )
    if input_rows != train_rows_read:
        raise P4PreparationError("candidate-audit/read-ledger row mismatch")
    audit_sources = _validate_count_mapping(
        payload.get("source_rows"), field="audit.source_rows"
    )
    if audit_sources != dict(source_records):
        raise P4PreparationError("candidate-audit source counts are inconsistent")
    if not isinstance(payload.get("transition_report"), Mapping):
        raise P4PreparationError("candidate-audit transition report must be an object")
    gate_counts = _validate_count_mapping(
        payload.get("local_gate_counts"), field="audit.local_gate_counts"
    )
    queue_rows = _nonnegative_int(
        payload.get("review_queue_rows"), field="audit.review_queue_rows"
    )
    if queue_rows != manifest["review_queue_rows"]:
        raise P4PreparationError("candidate-audit review queue count mismatch")
    suppressed = _nonnegative_int(
        payload.get("suppressed_queue_rows_due_to_fatal_errors"),
        field="audit.suppressed_queue_rows_due_to_fatal_errors",
    )
    fatal_count = _nonnegative_int(
        payload.get("fatal_error_count"), field="audit.fatal_error_count"
    )
    fatal_errors = payload.get("fatal_errors")
    if not isinstance(fatal_errors, list) or fatal_count != len(fatal_errors):
        raise P4PreparationError("candidate-audit fatal-error count is inconsistent")
    if any(not isinstance(item, str) or not item.strip() for item in fatal_errors):
        raise P4PreparationError(
            "candidate-audit fatal errors must be non-empty strings"
        )
    if manifest["package_status"] == "REVIEW_REQUIRED":
        if fatal_count != 0 or suppressed != 0:
            raise P4PreparationError(
                "review-required package cannot contain fatal errors"
            )
        if gate_counts.get("requires_independent_review", 0) != queue_rows:
            raise P4PreparationError("candidate-audit review gate count mismatch")
    elif fatal_count == 0 or queue_rows != 0:
        raise P4PreparationError("failed package must contain an error and no queue")
    if (
        payload.get("independent_verification_records_created") != 0
        or payload.get("provenance_manifest_created") is not False
        or payload.get("eligible_memory_items_claimed") != 0
    ):
        raise P4PreparationError(
            "candidate audit must not claim independent provenance or eligibility"
        )


def _validate_queue_row(row: object, *, line_number: int) -> dict[str, Any]:
    prefix = f"review queue line {line_number}"
    if not isinstance(row, dict) or set(row) != set(_QUEUE_ROW_FIELDS):
        raise P4PreparationError(f"{prefix} fields differ from the schema")
    if row.get("schema_version") != REVIEW_QUEUE_ROW_SCHEMA_VERSION:
        raise P4PreparationError(f"{prefix} has an unsupported schema")
    if row.get("status") != "REQUIRES_INDEPENDENT_REVIEW_AND_JOINT_DUPLICATE_AUDIT":
        raise P4PreparationError(f"{prefix} improperly claims completed review")
    if row.get("source_split") != "train":
        raise P4PreparationError(f"{prefix} is not train-only")
    if row.get("source_dataset_role") not in {
        "original_gold",
        "retry_abort_supplement_v2",
    }:
        raise P4PreparationError(f"{prefix} has an unregistered source role")
    transition_kind = row.get("source_transition_kind")
    expected_transition_kind = (
        "direct_recovery_from_observed_failure_state"
        if row.get("source_dataset_role") == "retry_abort_supplement_v2"
        else "adjacent_failure_then_recovery"
    )
    if transition_kind != expected_transition_kind:
        raise P4PreparationError(f"{prefix} source transition kind is invalid")
    if row.get("source_review_status") not in {"approved", "pending"}:
        raise P4PreparationError(f"{prefix} source review status is invalid")
    for field in (
        "queue_id",
        "source_sample_id",
        "recovery_sample_id",
        "canonical_task_id",
        "episode_id",
        "eligibility_notice",
    ):
        _nonempty(row.get(field), field=f"{prefix}.{field}")
    _nonnegative_int(
        row.get("source_step_index"), field=f"{prefix}.source_step_index"
    )
    _require_sha256(
        row.get("source_record_sha256"), field=f"{prefix}.source_record_sha256"
    )
    _require_sha256(
        row.get("transition_sha256"), field=f"{prefix}.transition_sha256"
    )
    duplicate_material = row.get("duplicate_audit_material")
    if not isinstance(duplicate_material, Mapping) or set(
        duplicate_material
    ) != set(_DUPLICATE_AUDIT_MATERIAL_FIELDS):
        raise P4PreparationError(
            f"{prefix} duplicate-audit material is invalid"
        )
    _nonempty(
        duplicate_material.get("task_description"),
        field=f"{prefix}.duplicate_audit_material.task_description",
    )
    website_domain = duplicate_material.get("website_domain")
    if not isinstance(website_domain, str) or website_domain != website_domain.strip():
        raise P4PreparationError(
            f"{prefix}.duplicate_audit_material.website_domain is not canonical"
        )
    labels = row.get("declared_training_labels")
    if not isinstance(labels, Mapping) or set(labels) != set(_DECLARED_LABEL_FIELDS):
        raise P4PreparationError(f"{prefix} declared labels are invalid")
    if (
        labels.get("recovery_strategy") in {None, "", "NONE"}
        or labels.get("recovery_success") is not True
        or labels.get("memory_update_flag") is not True
    ):
        raise P4PreparationError(f"{prefix} does not pass local training-label gates")
    if transition_kind == "adjacent_failure_then_recovery" and (
        labels.get("outcome_label") != "FAILURE"
        or labels.get("failure_type_4") in {None, "", "NONE"}
    ):
        raise P4PreparationError(f"{prefix} lacks an adjacent failure label")
    if transition_kind == "direct_recovery_from_observed_failure_state" and (
        labels.get("outcome_label") not in {"SUCCESS", "FAILURE"}
        or labels.get("failure_type_4") in {None, ""}
    ):
        raise P4PreparationError(f"{prefix} has invalid direct-recovery labels")
    if not isinstance(labels.get("reflection_text"), str):
        raise P4PreparationError(f"{prefix} reflection_text is not a string")
    causal = row.get("causal_material")
    if not isinstance(causal, Mapping) or set(causal) != set(_CAUSAL_MATERIAL_FIELDS):
        raise P4PreparationError(f"{prefix} causal material is invalid")
    for field in (
        "failure_state_reference",
        "executed_recovery_action",
        "post_recovery_state_reference",
    ):
        _nonempty(causal.get(field), field=f"{prefix}.causal_material.{field}")
    for field in (
        "pre_recovery_state_sha256",
        "executed_recovery_action_sha256",
        "post_recovery_state_sha256",
    ):
        _require_sha256(causal.get(field), field=f"{prefix}.causal_material.{field}")
    pending = row.get("required_external_evidence")
    if not isinstance(pending, dict) or set(pending) != set(
        _PENDING_EXTERNAL_FIELDS
    ):
        raise P4PreparationError(f"{prefix} external-evidence fields are invalid")
    if any(value is not None for value in pending.values()):
        raise P4PreparationError(f"{prefix} fabricates external evidence")
    if row.get("required_recovery_schema") != RECOVERY_VERIFICATION_SCHEMA_VERSION:
        raise P4PreparationError(f"{prefix} recovery-evidence schema changed")
    if row.get("required_final_task_schema") != FINAL_TASK_VERIFICATION_SCHEMA_VERSION:
        raise P4PreparationError(f"{prefix} final-task-evidence schema changed")
    if row.get("required_p4_label_review_schema") != P4_LABEL_REVIEW_SCHEMA_VERSION:
        raise P4PreparationError(f"{prefix} P4 label-review schema changed")
    try:
        source_material = canonical_p4_memory_source_material(
            row.get("memory_item_source_material")
        )
    except VerificationEvidenceError as error:
        raise P4PreparationError(f"{prefix} has invalid P4 source material: {error}") from error
    expected_material_projection = {
        "source_dataset_role": row["source_dataset_role"],
        "source_review_status": row["source_review_status"],
        "source_sample_id": row["source_sample_id"],
        "recovery_sample_id": row["recovery_sample_id"],
        "canonical_task_id": row["canonical_task_id"],
        "episode_id": row["episode_id"],
        "source_step_index": row["source_step_index"],
        "source_record_sha256": row["source_record_sha256"],
        "transition_sha256": row["transition_sha256"],
        "source_transition_kind": transition_kind,
        "task_description": duplicate_material["task_description"],
        "website_domain": duplicate_material["website_domain"],
        "source_outcome_label": labels["outcome_label"],
        "source_failure_type": labels["failure_type_4"],
        "source_action_type": labels["action_type"],
        "recovery_strategy": labels["recovery_strategy"],
        "reflection_text": labels["reflection_text"],
        "pre_recovery_state_sha256": causal["pre_recovery_state_sha256"],
        "executed_recovery_action": causal["executed_recovery_action"],
        "recovery_action_value": causal["recovery_action_value"],
        "executed_recovery_action_sha256": causal[
            "executed_recovery_action_sha256"
        ],
        "post_recovery_state_sha256": causal["post_recovery_state_sha256"],
    }
    for field, expected in expected_material_projection.items():
        if source_material.get(field) != expected:
            raise P4PreparationError(
                f"{prefix} source material differs at {field}"
            )
    identity = {
        "source_dataset_role": row["source_dataset_role"],
        "source_transition_kind": transition_kind,
        "source_review_status": row["source_review_status"],
        "source_sample_id": row["source_sample_id"],
        "recovery_sample_id": row["recovery_sample_id"],
        "canonical_task_id": row["canonical_task_id"],
        "episode_id": row["episode_id"],
        "source_step_index": row["source_step_index"],
        "source_record_sha256": row["source_record_sha256"],
        "pre_recovery_state_sha256": causal["pre_recovery_state_sha256"],
        "executed_recovery_action_sha256": causal[
            "executed_recovery_action_sha256"
        ],
        "post_recovery_state_sha256": causal["post_recovery_state_sha256"],
        "duplicate_audit_material": dict(duplicate_material),
        "reflection_text": labels["reflection_text"],
        "memory_item_source_material_sha256": canonical_sha256(source_material),
    }
    expected_id = f"p4-review-{canonical_sha256(identity)[:24]}"
    if row["queue_id"] != expected_id:
        raise P4PreparationError(f"{prefix} deterministic queue ID mismatch")
    return row


def validate_p4_preparation_package(
    root: str | Path,
) -> P4PreparationPackage:
    """Validate that a review-input package stayed train-only and non-authoritative."""

    directory = Path(root).resolve()
    manifest_path = directory / "preparation_manifest.json"
    manifest = _load_json(manifest_path)
    required = {
        "schema_version",
        "package_status",
        "source_split",
        *_ZERO_READ_FIELDS,
        "dataset_id",
        "dataset_version",
        "records_sha256",
        "source_authority_sha256",
        "dataset_artifacts_sha256",
        "review_queue_schema_version",
        "review_queue_rows",
        "review_queue_sha256",
        "independent_verification_records_created",
        "provenance_manifest_created",
        "eligible_memory_items_claimed",
        "files",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise P4PreparationError(
            "P4 preparation manifest fields differ from the registered schema"
        )
    if manifest.get("schema_version") != PREPARATION_PACKAGE_SCHEMA_VERSION:
        raise P4PreparationError("unsupported P4 preparation package schema")
    if manifest.get("package_status") not in {"REVIEW_REQUIRED", "FAIL"}:
        raise P4PreparationError("invalid P4 preparation package status")
    if manifest.get("source_split") != "train":
        raise P4PreparationError("P4 preparation package must be train-only")
    _require_zero_reads(manifest, context="P4 preparation")
    if (
        manifest.get("independent_verification_records_created") != 0
        or manifest.get("provenance_manifest_created") is not False
        or manifest.get("eligible_memory_items_claimed") != 0
    ):
        raise P4PreparationError(
            "preparation package must not claim independent provenance or eligibility"
        )
    descriptors = manifest.get("files")
    if not isinstance(descriptors, dict) or set(descriptors) != set(
        _PREPARATION_FILES
    ):
        raise P4PreparationError("P4 preparation file set is invalid")
    for name in _PREPARATION_FILES:
        if descriptors[name] != _file_descriptor(directory / name):
            raise P4PreparationError(f"P4 preparation payload mismatch: {name}")
    if _require_sha256(
        manifest.get("source_authority_sha256"),
        field="preparation.source_authority_sha256",
    ) != sha256_file(directory / P4_SOURCE_AUTHORITY_FILE_NAME):
        raise P4PreparationError("P4 preparation source-authority hash mismatch")
    sidecar = directory / "preparation_manifest.sha256"
    if sidecar.is_symlink() or not sidecar.is_file():
        raise P4PreparationError("missing preparation_manifest.sha256")
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(manifest_path):
        raise P4PreparationError("P4 preparation manifest hash mismatch")

    read_ledger = _load_json(directory / "read_ledger.json")
    source_records = _validate_read_ledger(read_ledger, manifest=manifest)
    _source_authority_payload(
        directory / P4_SOURCE_AUTHORITY_FILE_NAME,
        source_descriptors=read_ledger["train_json_files"],
        dataset_id=str(manifest["dataset_id"]),
        dataset_version=str(manifest["dataset_version"]),
    )
    expected_dataset_artifacts_sha256 = _dataset_artifacts_sha256(
        source_authority_sha256=str(manifest["source_authority_sha256"]),
        candidate_state_artifacts_sha256=str(
            read_ledger["candidate_state_artifacts_sha256"]
        ),
    )
    if manifest.get("dataset_artifacts_sha256") != expected_dataset_artifacts_sha256:
        raise P4PreparationError("P4 dataset-artifact binding is inconsistent")
    candidate_audit = _load_json(directory / "candidate_audit.json")
    _validate_audit(
        candidate_audit,
        manifest=manifest,
        source_records=source_records,
        train_rows_read=int(read_ledger["train_rows_read"]),
    )

    rows = _validated_review_queue_rows(directory)
    _validate_queue_artifact_ledger(rows, read_ledger)
    if type(manifest.get("review_queue_rows")) is not int or manifest[
        "review_queue_rows"
    ] != len(rows):
        raise P4PreparationError("review queue row count mismatch")
    if manifest.get("review_queue_sha256") != canonical_sha256(rows):
        raise P4PreparationError("review queue canonical hash mismatch")
    if manifest["package_status"] == "FAIL" and rows:
        raise P4PreparationError("failed preparation package cannot expose a queue")
    return P4PreparationPackage(root=directory, manifest=manifest)


def _validated_review_queue_rows(directory: Path) -> list[dict[str, Any]]:
    """Reload the already hash-closed queue for cross-artifact validation."""

    rows: list[dict[str, Any]] = []
    queue_ids: set[str] = set()
    source_sample_ids: set[str] = set()
    for line_number, raw in enumerate(
        (directory / "review_queue.jsonl").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            raise P4PreparationError(f"blank review queue line {line_number}")
        try:
            row = json.loads(raw, object_pairs_hook=_json_pairs_without_duplicates)
        except json.JSONDecodeError as error:
            raise P4PreparationError(
                f"invalid review queue line {line_number}: {error}"
            ) from error
        verified_row = _validate_queue_row(row, line_number=line_number)
        queue_id = str(verified_row["queue_id"])
        sample_id = str(verified_row["source_sample_id"])
        if queue_id in queue_ids:
            raise P4PreparationError(f"duplicate review queue ID at line {line_number}")
        if sample_id in source_sample_ids:
            raise P4PreparationError(
                f"duplicate review source sample at line {line_number}"
            )
        queue_ids.add(queue_id)
        source_sample_ids.add(sample_id)
        rows.append(verified_row)
    return rows


def _validate_queue_artifact_ledger(
    rows: Sequence[Mapping[str, Any]],
    read_ledger: Mapping[str, Any],
) -> None:
    artifact_rows: list[dict[str, Any]] = []
    for row in rows:
        causal = row["causal_material"]
        artifact_rows.extend(
            (
                {
                    "source_dataset_role": row["source_dataset_role"],
                    "reference": causal["failure_state_reference"],
                    "sha256": causal["pre_recovery_state_sha256"],
                },
                {
                    "source_dataset_role": row["source_dataset_role"],
                    "reference": causal["post_recovery_state_reference"],
                    "sha256": causal["post_recovery_state_sha256"],
                },
            )
        )
    artifact_rows.sort(
        key=lambda row: (row["source_dataset_role"], row["reference"], row["sha256"])
    )
    expected = {
        "candidate_state_artifact_references": len(artifact_rows),
        "candidate_state_artifact_unique_hashes": len(
            {row["sha256"] for row in artifact_rows}
        ),
        "candidate_state_artifacts_sha256": canonical_sha256(artifact_rows),
    }
    for field, value in expected.items():
        if read_ledger.get(field) != value:
            raise P4PreparationError(
                f"P4 read ledger differs from review queue at {field}"
            )


def validate_p4_provenance_against_preparation(
    *,
    package_root: str | Path,
    provenance_manifest_path: str | Path,
) -> Mapping[str, Any]:
    """Validate externally produced provenance against its review-input queue.

    This function deliberately has no evidence-authoring path.  It cannot decide
    whether a recovery or final task actually succeeded and it cannot establish
    that a named authority is operationally independent.  It only validates the
    completed external records and proves that an admitted record is attached to
    the exact train-only candidate and causal hashes emitted by preparation.
    """

    package = validate_p4_preparation_package(package_root)
    if package.status != "REVIEW_REQUIRED":
        raise P4PreparationError(
            "only a REVIEW_REQUIRED preparation package can receive provenance"
        )
    queue_rows = _validated_review_queue_rows(package.root)
    if not queue_rows:
        raise P4PreparationError(
            "P4 provenance validation requires at least one review candidate"
        )

    provenance_path = Path(provenance_manifest_path)
    if provenance_path.is_symlink():
        raise P4PreparationError("P4 provenance manifest must not be a symlink")
    provenance_path = provenance_path.resolve()
    payload = _load_json(provenance_path)
    if not isinstance(payload, Mapping):
        raise P4PreparationError("P4 provenance manifest must be a JSON object")
    try:
        provenance = ProvenanceManifest.from_mapping(payload)
    except (ManifestError, TypeError, ValueError) as error:
        raise P4PreparationError(f"invalid P4 provenance manifest: {error}") from error

    expected_top_level = {
        "dataset_id": package.manifest["dataset_id"],
        "dataset_version": package.manifest["dataset_version"],
        "records_sha256": package.manifest["records_sha256"],
        "dataset_artifacts_sha256": package.manifest[
            "dataset_artifacts_sha256"
        ],
    }
    actual_top_level = {
        "dataset_id": provenance.dataset_id,
        "dataset_version": provenance.dataset_version,
        "records_sha256": provenance.records_sha256,
        "dataset_artifacts_sha256": provenance.dataset_artifacts_sha256,
    }
    for field, expected in expected_top_level.items():
        if actual_top_level[field] != expected:
            raise P4PreparationError(
                f"P4 provenance/preparation mismatch at {field}"
            )

    queue_by_sample = {
        str(row["source_sample_id"]): row for row in queue_rows
    }
    provenance_ids = set(provenance.records)
    queue_ids = set(queue_by_sample)
    missing_ids = queue_ids - provenance_ids
    if missing_ids:
        raise P4PreparationError(
            "P4 provenance candidate coverage is incomplete: "
            f"missing={sorted(missing_ids)}"
        )

    admitted_records = 0
    externally_excluded_records = 0
    for sample_id, queue_row in queue_by_sample.items():
        evidence = provenance.records[sample_id]
        for field in ("provenance_valid", "final_task_success"):
            if type(evidence.get(field)) is not bool:
                raise P4PreparationError(
                    f"P4 provenance {sample_id}.{field} must be a JSON boolean"
                )
        identity = {
            "canonical_task_id": queue_row["canonical_task_id"],
            "episode_id": queue_row["episode_id"],
        }
        for field, expected in identity.items():
            if evidence.get(field) != expected:
                raise P4PreparationError(
                    f"P4 provenance {sample_id} differs from its queue at {field}"
                )
        _require_sha256(
            evidence.get("exact_duplicate_key"),
            field=f"provenance.{sample_id}.exact_duplicate_key",
        )
        _nonempty(
            evidence.get("near_duplicate_cluster_id"),
            field=f"provenance.{sample_id}.near_duplicate_cluster_id",
        )

        if not (
            evidence["provenance_valid"] and evidence["final_task_success"]
        ):
            externally_excluded_records += 1
            continue
        label_review = evidence.get("p4_label_review")
        label_review_digest = evidence.get("p4_label_review_evidence_sha256")
        if queue_row["source_review_status"] == "pending" or label_review is not None:
            if not isinstance(label_review, Mapping):
                raise P4PreparationError(
                    f"P4 provenance {sample_id} lacks independent label review"
                )
            try:
                expected_label_review_digest = validate_p4_label_review_evidence(
                    label_review,
                    source_sample_id=sample_id,
                    source_record_sha256=str(queue_row["source_record_sha256"]),
                    memory_item_source_material_sha256=canonical_sha256(
                        queue_row["memory_item_source_material"]
                    ),
                )
            except (VerificationEvidenceError, ManifestError) as error:
                raise P4PreparationError(
                    f"invalid P4 label review for {sample_id}: {error}"
                ) from error
            if label_review_digest != expected_label_review_digest:
                raise P4PreparationError(
                    f"P4 provenance {sample_id} label-review digest mismatch"
                )
        try:
            verified = validate_provenance_verification_evidence(
                evidence,
                source_sample_id=sample_id,
                recovery_sample_id=str(queue_row["recovery_sample_id"]),
                canonical_task_id=str(queue_row["canonical_task_id"]),
                episode_id=str(queue_row["episode_id"]),
            )
        except (VerificationEvidenceError, ManifestError) as error:
            raise P4PreparationError(
                f"invalid independent verification for {sample_id}: {error}"
            ) from error
        causal = queue_row["causal_material"]
        recovery_record = verified.recovery_record
        for field in (
            "pre_recovery_state_sha256",
            "executed_recovery_action_sha256",
            "post_recovery_state_sha256",
        ):
            if recovery_record.get(field) != causal[field]:
                raise P4PreparationError(
                    f"P4 provenance {sample_id} differs from its queue at {field}"
                )
        admitted_records += 1

    return {
        "status": "PASS",
        "evidence_role": "STRUCTURAL_VALIDATION_ONLY_NOT_REVIEW_AUTHORITY",
        "independent_review_performed_by_tool": False,
        "package_manifest_sha256": sha256_file(
            package.root / "preparation_manifest.json"
        ),
        "provenance_manifest_sha256": sha256_file(provenance_path),
        "dataset_id": provenance.dataset_id,
        "dataset_version": provenance.dataset_version,
        "records_sha256": provenance.records_sha256,
        "dataset_artifacts_sha256": provenance.dataset_artifacts_sha256,
        "duplicate_cluster_namespace": (
            provenance.duplicate_cluster_namespace.to_dict()
        ),
        "review_queue_rows": len(queue_rows),
        "additional_nonqueue_provenance_records": len(provenance_ids - queue_ids),
        "admitted_records_with_validated_evidence": admitted_records,
        "externally_excluded_records": externally_excluded_records,
        "validation_rows_read": provenance.validation_rows_read,
        "test_rows_read": provenance.test_rows_read,
        "locked_test_rows_read": provenance.locked_test_rows_read,
    }


def reconstruct_p4_selection_from_preparation(
    *,
    package_root: str | Path,
    provenance_manifest_path: str | Path,
) -> EligibilitySelection:
    """Replay the exact eligible/deduplicated store corpus without Gold images.

    The data host must first create and validate the preparation package from
    the authenticated Gold train sources.  This compact replay then uses only
    that immutable queue and the completed external provenance manifest.  It
    deliberately has no route that can approve a pending label or create
    recovery/final-task evidence.
    """

    validate_p4_provenance_against_preparation(
        package_root=package_root,
        provenance_manifest_path=provenance_manifest_path,
    )
    package = validate_p4_preparation_package(package_root)
    queue_rows = _validated_review_queue_rows(package.root)
    provenance_payload = _load_json(Path(provenance_manifest_path).resolve())
    try:
        provenance = ProvenanceManifest.from_mapping(provenance_payload)
    except (ManifestError, TypeError, ValueError) as error:
        raise P4PreparationError(f"invalid P4 provenance manifest: {error}") from error

    audit = _load_json(package.root / "candidate_audit.json")
    local_counts = _validate_count_mapping(
        audit.get("local_gate_counts"), field="audit.local_gate_counts"
    )
    allowed_local = {
        "requires_independent_review",
        "quarantined_or_non_admitted",
        "memory_update_flag_missing",
        "memory_update_flag_false",
        "recovery_success_missing",
        "recovery_not_successful",
    }
    unknown_local = sorted(set(local_counts) - allowed_local)
    if unknown_local:
        raise P4PreparationError(
            f"preparation contains unregistered local gate counts: {unknown_local}"
        )
    if local_counts.get("requires_independent_review", 0) != len(queue_rows):
        raise P4PreparationError("preparation review count differs from queue")
    if sum(local_counts.values()) != int(audit["input_rows"]):
        raise P4PreparationError(
            "preparation local gate accounting does not equal input rows"
        )

    exclusions: Counter[str] = Counter(
        {
            reason: int(local_counts.get(reason, 0))
            for reason in EXCLUSION_REASONS
            if reason in allowed_local
        }
    )
    candidates: list[EligibleMemoryCandidate] = []
    namespace_id = provenance.duplicate_cluster_namespace.namespace_id
    for index, queue_row in enumerate(queue_rows):
        sample_id = str(queue_row["source_sample_id"])
        evidence = provenance.records[sample_id]
        if is_explicitly_non_admitted({}, evidence):
            exclusions["quarantined_or_non_admitted"] += 1
            continue
        if evidence.get("final_task_success") is False:
            exclusions["final_task_not_successful"] += 1
            continue
        if evidence.get("provenance_valid") is False:
            exclusions["provenance_invalid"] += 1
            continue
        if evidence.get("final_task_success") is not True or evidence.get(
            "provenance_valid"
        ) is not True:
            raise P4PreparationError(
                f"P4 provenance {sample_id} lacks explicit admission booleans"
            )

        source_material = canonical_p4_memory_source_material(
            queue_row["memory_item_source_material"]
        )
        if (
            source_material["source_transition_kind"]
            == "direct_recovery_from_observed_failure_state"
            and source_material["recovery_strategy"] == "ABORT"
        ):
            raise P4PreparationError(
                f"P4 provenance {sample_id} claims final success for direct ABORT"
            )
        label_review = evidence.get("p4_label_review")
        label_review_digest: str | None = None
        if queue_row["source_review_status"] == "pending" or label_review is not None:
            if not isinstance(label_review, Mapping):
                raise P4PreparationError(
                    f"P4 provenance {sample_id} lacks independent label review"
                )
            try:
                label_review_digest = validate_p4_label_review_evidence(
                    label_review,
                    source_sample_id=sample_id,
                    source_record_sha256=str(queue_row["source_record_sha256"]),
                    memory_item_source_material_sha256=canonical_sha256(
                        source_material
                    ),
                )
            except (VerificationEvidenceError, ManifestError) as error:
                raise P4PreparationError(
                    f"invalid P4 label review for {sample_id}: {error}"
                ) from error
            if evidence.get(
                "p4_label_review_evidence_sha256"
            ) != label_review_digest:
                raise P4PreparationError(
                    f"P4 provenance {sample_id} label-review digest mismatch"
                )
        try:
            verified = validate_provenance_verification_evidence(
                evidence,
                source_sample_id=sample_id,
                recovery_sample_id=str(queue_row["recovery_sample_id"]),
                canonical_task_id=str(queue_row["canonical_task_id"]),
                episode_id=str(queue_row["episode_id"]),
            )
        except (VerificationEvidenceError, ManifestError) as error:
            raise P4PreparationError(
                f"invalid independent verification for {sample_id}: {error}"
            ) from error
        recovery_record = verified.recovery_record
        causal = queue_row["causal_material"]
        for field in (
            "pre_recovery_state_sha256",
            "executed_recovery_action_sha256",
            "post_recovery_state_sha256",
        ):
            if recovery_record.get(field) != causal[field]:
                raise P4PreparationError(
                    f"P4 provenance {sample_id} differs from preparation at {field}"
                )
        exact_key = _require_sha256(
            evidence.get("exact_duplicate_key"),
            field=f"provenance.{sample_id}.exact_duplicate_key",
        )
        near_cluster = _nonempty(
            evidence.get("near_duplicate_cluster_id"),
            field=f"provenance.{sample_id}.near_duplicate_cluster_id",
        )
        if evidence.get("duplicate_cluster_namespace_id") != namespace_id:
            raise P4PreparationError(
                f"P4 provenance {sample_id} cites another duplicate namespace"
            )
        memory_id = canonical_memory_id(
            provenance.dataset_id,
            sample_id,
            str(queue_row["recovery_sample_id"]),
        )
        material_sha256 = canonical_sha256(source_material)
        item = {
            "schema_version": ELIGIBILITY_POLICY_VERSION,
            "memory_id": memory_id,
            "source_split": "train",
            "source_dataset_id": provenance.dataset_id,
            "source_dataset_version": provenance.dataset_version,
            "source_sample_id": sample_id,
            "recovery_sample_id": queue_row["recovery_sample_id"],
            "source_task_id": queue_row["canonical_task_id"],
            "source_episode_id": queue_row["episode_id"],
            "step_index": queue_row["source_step_index"],
            "website_domain": source_material["website_domain"],
            "source_transition_kind": source_material["source_transition_kind"],
            "source_review_status": source_material["source_review_status"],
            "source_record_sha256": source_material["source_record_sha256"],
            "memory_item_source_material_sha256": material_sha256,
            "failure_type": source_material["failure_type"],
            "failure_type_available": source_material["failure_type_available"],
            "failed_action": source_material["failed_action"],
            "failed_action_available": source_material["failed_action_available"],
            "strategy": source_material["recovery_strategy"],
            "executed_recovery_action": source_material[
                "executed_recovery_action"
            ],
            "recovery_action_value": source_material["recovery_action_value"],
            "reflection_text": source_material["reflection_text"],
            "p4_label_review_evidence_sha256": label_review_digest,
            "memory_update_flag": True,
            "verified_recovery_success": True,
            "final_task_success": True,
            "provenance_valid": True,
            "recovery_verification_evidence_sha256": (
                verified.recovery_evidence_sha256
            ),
            "final_task_verification_evidence_sha256": (
                verified.final_task_evidence_sha256
            ),
            "verification_evidence_sha256": verified.bundle_sha256,
            "exact_duplicate_key": exact_key,
            "duplicate_cluster_id": near_cluster,
            "duplicate_cluster_namespace_id": namespace_id,
        }
        candidates.append(
            EligibleMemoryCandidate(
                record_index=index,
                memory_id=memory_id,
                source_sample_id=sample_id,
                recovery_sample_id=str(queue_row["recovery_sample_id"]),
                canonical_task_id=str(queue_row["canonical_task_id"]),
                episode_id=str(queue_row["episode_id"]),
                step_index=int(queue_row["source_step_index"]),
                exact_duplicate_key=exact_key,
                near_duplicate_cluster_id=near_cluster,
                duplicate_cluster_namespace_id=namespace_id,
                recovery_verification_evidence_sha256=(
                    verified.recovery_evidence_sha256
                ),
                final_task_verification_evidence_sha256=(
                    verified.final_task_evidence_sha256
                ),
                verification_evidence_sha256=verified.bundle_sha256,
                verification_evidence={
                    **verified.to_payload(),
                    "p4_label_review": (
                        dict(label_review)
                        if isinstance(label_review, Mapping)
                        else None
                    ),
                    "p4_label_review_evidence_sha256": label_review_digest,
                },
                item=item,
            )
        )

    if not candidates:
        raise P4PreparationError("no eligible successful training memories")
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
        raise P4PreparationError("deduplication removed every eligible memory")
    return EligibilitySelection(
        candidates=tuple(kept),
        exclusion_counts={
            reason: int(exclusions[reason]) for reason in EXCLUSION_REASONS
        },
        input_rows=int(audit["input_rows"]),
        pre_dedup_eligible_rows=pre_dedup,
        duplicate_cluster_namespace=(
            provenance.duplicate_cluster_namespace.to_dict()
        ),
    )


def _ensure_transfer_manifest_outside_store(
    store_root: Path, manifest_path: Path
) -> None:
    try:
        manifest_path.resolve().relative_to(store_root.resolve())
    except ValueError:
        return
    raise P4PreparationError(
        "cross-host transfer manifest must be outside the immutable store"
    )


def _transfer_core(
    store: FrozenMemoryStore,
    *,
    files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_manifest = store.manifest
    return {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "artifact_kind": "frozen_table2_memory_store",
        "source_role": "p4_build_host",
        "destination_role": "table2_campaign_host",
        "store_id": source_manifest["store_id"],
        "store_manifest_sha256": store.manifest_sha256,
        "model_seed": source_manifest["model_seed"],
        "checkpoint_sha256": source_manifest["checkpoint_sha256"],
        "records_sha256": source_manifest["records_sha256"],
        "dataset_artifacts_sha256": source_manifest[
            "dataset_artifacts_sha256"
        ],
        "resolved_config_sha256": source_manifest["resolved_config_sha256"],
        "resolved_config_record_sha256": source_manifest[
            "resolved_config_record_sha256"
        ],
        "protocol_sha256": source_manifest["protocol_sha256"],
        "provenance_manifest_sha256": source_manifest[
            "provenance_manifest_sha256"
        ],
        "source_split": source_manifest["source_split"],
        "validation_rows_read": source_manifest["validation_rows_read"],
        "test_rows_read": source_manifest["test_rows_read"],
        "locked_test_rows_read": source_manifest["locked_test_rows_read"],
        "runtime_writes_allowed": source_manifest["runtime_writes_allowed"],
        "item_count": source_manifest["item_count"],
        "files": dict(files),
        "total_bytes": sum(int(item["bytes"]) for item in files.values()),
    }


def create_memory_store_transfer_manifest(
    *,
    store_root: str | Path,
    manifest_path: str | Path,
) -> Mapping[str, Any]:
    """Validate a frozen store and write its portable cross-host file manifest."""

    root = Path(store_root).resolve()
    destination = Path(manifest_path).resolve()
    _ensure_transfer_manifest_outside_store(root, destination)
    if destination.exists():
        raise P4PreparationError(
            f"refusing to overwrite transfer manifest: {destination}"
        )
    store = FrozenMemoryStore.load(root)
    files = {name: _file_descriptor(root / name) for name in _STORE_FILES}
    core = _transfer_core(store, files=files)
    digest = canonical_sha256(core)
    payload = {
        **core,
        "transfer_id": f"table2-memory-transfer-{store.model_seed}-{digest[:16]}",
        "transfer_sha256": digest,
    }
    _atomic_write_json(destination, payload)
    return validate_memory_store_transfer_manifest(
        store_root=root,
        manifest_path=destination,
    )


def validate_memory_store_transfer_manifest(
    *,
    store_root: str | Path,
    manifest_path: str | Path,
) -> Mapping[str, Any]:
    """Replay both frozen-store validation and its cross-host descriptors."""

    root = Path(store_root).resolve()
    path = Path(manifest_path).resolve()
    _ensure_transfer_manifest_outside_store(root, path)
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise P4PreparationError("memory transfer manifest must be a JSON object")
    expected_fields = set(
        _transfer_core_fields()
    ) | {"transfer_id", "transfer_sha256"}
    if set(payload) != expected_fields:
        raise P4PreparationError(
            "memory transfer manifest fields differ from the registered schema"
        )
    if payload.get("schema_version") != TRANSFER_SCHEMA_VERSION:
        raise P4PreparationError("unsupported memory transfer schema")
    if payload.get("artifact_kind") != "frozen_table2_memory_store":
        raise P4PreparationError("unexpected memory transfer artifact kind")
    if payload.get("source_role") != "p4_build_host" or payload.get(
        "destination_role"
    ) != "table2_campaign_host":
        raise P4PreparationError("memory transfer host roles are not registered")
    if payload.get("source_split") != "train":
        raise P4PreparationError("memory transfer store must be train-only")
    for field in _ZERO_READ_FIELDS:
        if type(payload.get(field)) is not int or payload[field] != 0:
            raise P4PreparationError(f"memory transfer requires explicit {field}=0")
    if payload.get("runtime_writes_allowed") is not False:
        raise P4PreparationError("memory transfer store must be read-only")

    store = FrozenMemoryStore.load(root)
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != set(_STORE_FILES):
        raise P4PreparationError("memory transfer file set is invalid")
    actual_files = {name: _file_descriptor(root / name) for name in _STORE_FILES}
    if files != actual_files:
        raise P4PreparationError("memory transfer payload descriptors do not match")
    expected_core = _transfer_core(store, files=actual_files)
    for field, expected in expected_core.items():
        if payload.get(field) != expected:
            raise P4PreparationError(
                f"memory transfer differs from frozen store at {field}"
            )
    expected_digest = canonical_sha256(expected_core)
    if payload.get("transfer_sha256") != expected_digest:
        raise P4PreparationError("memory transfer canonical hash mismatch")
    expected_id = f"table2-memory-transfer-{store.model_seed}-{expected_digest[:16]}"
    if payload.get("transfer_id") != expected_id:
        raise P4PreparationError("memory transfer ID mismatch")
    return payload


def _transfer_core_fields() -> tuple[str, ...]:
    return (
        "schema_version",
        "artifact_kind",
        "source_role",
        "destination_role",
        "store_id",
        "store_manifest_sha256",
        "model_seed",
        "checkpoint_sha256",
        "records_sha256",
        "dataset_artifacts_sha256",
        "resolved_config_sha256",
        "resolved_config_record_sha256",
        "protocol_sha256",
        "provenance_manifest_sha256",
        "source_split",
        "validation_rows_read",
        "test_rows_read",
        "locked_test_rows_read",
        "runtime_writes_allowed",
        "item_count",
        "files",
        "total_bytes",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit-candidates",
        help="audit explicit training sources and create a pending-review queue",
    )
    audit.add_argument("--gold-train-json", type=Path, required=True)
    audit.add_argument("--gold-data-root", type=Path, required=True)
    audit.add_argument("--supplement-train-json", type=Path)
    audit.add_argument("--supplement-data-root", type=Path)
    audit.add_argument("--dataset-id", required=True)
    audit.add_argument("--dataset-version", required=True)
    audit.add_argument("--source-authority", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)

    validate_preparation = subparsers.add_parser(
        "validate-preparation",
        help="verify a P4 pending-review package after cross-host transfer",
    )
    validate_preparation.add_argument("--package-dir", type=Path, required=True)

    validate_provenance = subparsers.add_parser(
        "validate-provenance",
        help=(
            "validate separately produced provenance against the exact "
            "pending-review package without authoring evidence"
        ),
    )
    validate_provenance.add_argument("--package-dir", type=Path, required=True)
    validate_provenance.add_argument(
        "--provenance-manifest", type=Path, required=True
    )

    create_transfer = subparsers.add_parser(
        "create-transfer-manifest",
        help="validate and describe an existing immutable memory store",
    )
    create_transfer.add_argument("--store-dir", type=Path, required=True)
    create_transfer.add_argument("--manifest", type=Path, required=True)

    validate_transfer = subparsers.add_parser(
        "validate-transfer",
        help="validate a copied immutable store against its transfer manifest",
    )
    validate_transfer.add_argument("--store-dir", type=Path, required=True)
    validate_transfer.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "audit-candidates":
        result = prepare_p4_candidate_audit(
            gold_train_json=args.gold_train_json,
            gold_data_root=args.gold_data_root,
            supplement_train_json=args.supplement_train_json,
            supplement_data_root=args.supplement_data_root,
            dataset_id=args.dataset_id,
            dataset_version=args.dataset_version,
            source_authority_path=args.source_authority,
            output_dir=args.output_dir,
        )
        output = {
            "status": result.status,
            "package_dir": str(result.root),
            "review_queue_rows": result.candidate_count,
            "source_split": "train",
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "provenance_manifest_created": False,
        }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0 if result.status == "REVIEW_REQUIRED" else 2
    if args.command == "validate-preparation":
        result = validate_p4_preparation_package(args.package_dir)
        output = {
            "status": "PASS",
            "package_status": result.status,
            "package_dir": str(result.root),
            "review_queue_rows": result.candidate_count,
        }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-provenance":
        output = validate_p4_provenance_against_preparation(
            package_root=args.package_dir,
            provenance_manifest_path=args.provenance_manifest,
        )
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    if args.command == "create-transfer-manifest":
        payload = create_memory_store_transfer_manifest(
            store_root=args.store_dir,
            manifest_path=args.manifest,
        )
    else:
        payload = validate_memory_store_transfer_manifest(
            store_root=args.store_dir,
            manifest_path=args.manifest,
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "transfer_id": payload["transfer_id"],
                "store_id": payload["store_id"],
                "model_seed": payload["model_seed"],
                "item_count": payload["item_count"],
                "total_bytes": payload["total_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
