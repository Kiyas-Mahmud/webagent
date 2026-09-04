"""Build an atomic, immutable corrective-memory store for one model seed."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np

from web_agent.memory.calibration_builder import (
    CALIBRATION_EMBEDDING_HASH,
    CALIBRATION_EMBEDDING_STAGE,
    CALIBRATION_EVIDENCE_SCHEMA_VERSION,
    CALIBRATION_NORMALIZATION,
    CALIBRATION_PAIR_POLICY,
    CALIBRATION_RELEVANCE_DEFINITION,
    CALIBRATION_SIMILARITY,
)
from web_agent.memory.eligibility import (
    ELIGIBILITY_POLICY_VERSION,
    EligibilitySelection,
    EligibleMemoryCandidate,
)
from web_agent.memory.frozen_store import FrozenMemoryStore
from web_agent.memory.manifest import (
    EMBEDDING_DIMENSION,
    ManifestError,
    RUNTIME_TOP_K,
    SIMILARITY_METRIC,
    STORE_SCHEMA_VERSION,
    canonical_sha256,
    file_descriptor,
    require_sha256,
    sha256_file,
    validate_threshold_calibration_payload,
)
from web_agent.runtime.duplicate_audit import (
    DuplicateAuditError,
    JointDuplicateClusterNamespace,
)
from web_agent.memory.verification import (
    VERIFICATION_MANIFEST_SCHEMA_VERSION,
    VerificationEvidenceError,
    validate_frozen_item_verification,
    validate_provenance_verification_evidence,
    verification_manifest_payload,
    verification_records_payload,
)


class MemoryBuildError(ValueError):
    """Raised when a frozen store cannot be built without weakening a gate."""


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIMENSION:
        raise MemoryBuildError(
            f"memory embeddings must have shape [N, {EMBEDDING_DIMENSION}], "
            f"got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise MemoryBuildError("memory embeddings contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms <= 1e-12).any() or not np.isfinite(norms).all():
        raise MemoryBuildError("memory embeddings contain zero/invalid vectors")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_frozen_store(
    output_dir: str | Path,
    *,
    selection: EligibilitySelection,
    embeddings: np.ndarray,
    model_seed: int,
    checkpoint_sha256: str,
    records_sha256: str,
    dataset_id: str,
    dataset_version: str,
    dataset_artifacts_sha256: str,
    resolved_config_sha256: str,
    resolved_config_record_sha256: str,
    protocol_sha256: str,
    provenance_manifest_sha256: str,
    threshold_calibration: Mapping[str, Any],
    calibration_evidence: Mapping[str, Any],
    transition_report: Mapping[str, Any],
    joint_duplicate_audit_binding: Mapping[str, Any],
) -> FrozenMemoryStore:
    """Write a new content-verified store and immediately reload-audit it.

    ``embeddings`` must align with ``selection.candidates``.  Existing output
    paths are never replaced, which prevents accidental mutation of a frozen
    seed store. ``resolved_config_sha256`` is the exact selected-config file
    hash; ``resolved_config_record_sha256`` is its canonical mapping hash.
    """
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen memory output: {destination}"
        )
    if not selection.candidates:
        raise MemoryBuildError("cannot build an empty memory store")
    try:
        duplicate_namespace = JointDuplicateClusterNamespace.from_mapping(
            selection.duplicate_cluster_namespace,
            require_hashes=True,
        )
    except DuplicateAuditError as error:
        raise MemoryBuildError(
            f"invalid joint duplicate-cluster namespace: {error}"
        ) from error
    for candidate in selection.candidates:
        if (
            candidate.duplicate_cluster_namespace_id
            != duplicate_namespace.namespace_id
            or candidate.item.get("duplicate_cluster_namespace_id")
            != duplicate_namespace.namespace_id
        ):
            raise MemoryBuildError(
                f"memory {candidate.memory_id} cites a different "
                "duplicate-cluster namespace"
            )
        try:
            item_verification_digest = validate_frozen_item_verification(
                candidate.item
            )
        except (VerificationEvidenceError, ManifestError) as error:
            raise MemoryBuildError(
                f"memory {candidate.memory_id} lacks valid verification evidence: "
                f"{error}"
            ) from error
        candidate_bindings = {
            "recovery_verification_evidence_sha256": (
                candidate.recovery_verification_evidence_sha256
            ),
            "final_task_verification_evidence_sha256": (
                candidate.final_task_verification_evidence_sha256
            ),
            "verification_evidence_sha256": candidate.verification_evidence_sha256,
        }
        for field, expected in candidate_bindings.items():
            if candidate.item.get(field) != expected:
                raise MemoryBuildError(
                    f"memory {candidate.memory_id} item/candidate verification "
                    f"binding differs at {field}"
                )
        if item_verification_digest != candidate.verification_evidence_sha256:
            raise MemoryBuildError(
                f"memory {candidate.memory_id} verification bundle is inconsistent"
            )
        try:
            candidate_evidence = validate_provenance_verification_evidence(
                candidate.verification_evidence,
                source_sample_id=candidate.source_sample_id,
                recovery_sample_id=candidate.recovery_sample_id,
                canonical_task_id=candidate.canonical_task_id,
                episode_id=candidate.episode_id,
            )
        except (VerificationEvidenceError, ManifestError) as error:
            raise MemoryBuildError(
                f"memory {candidate.memory_id} does not retain its independent "
                f"verification records: {error}"
            ) from error
        if (
            candidate_evidence.recovery_evidence_sha256
            != candidate.recovery_verification_evidence_sha256
            or candidate_evidence.final_task_evidence_sha256
            != candidate.final_task_verification_evidence_sha256
            or candidate_evidence.bundle_sha256
            != candidate.verification_evidence_sha256
        ):
            raise MemoryBuildError(
                f"memory {candidate.memory_id} evidence records differ from candidate"
            )
    if not isinstance(model_seed, int) or isinstance(model_seed, bool) or model_seed < 0:
        raise MemoryBuildError("model_seed must be a non-negative integer")
    checkpoint_hash = require_sha256(
        checkpoint_sha256, field="checkpoint_sha256"
    )
    records_hash = require_sha256(records_sha256, field="records_sha256")
    dataset_artifacts_hash = require_sha256(
        dataset_artifacts_sha256, field="dataset_artifacts_sha256"
    )
    config_hash = require_sha256(
        resolved_config_sha256, field="resolved_config_sha256"
    )
    config_record_hash = require_sha256(
        resolved_config_record_sha256,
        field="resolved_config_record_sha256",
    )
    protocol_hash = require_sha256(protocol_sha256, field="protocol_sha256")
    provenance_hash = require_sha256(
        provenance_manifest_sha256, field="provenance_manifest_sha256"
    )
    expected_binding_fields = {
        "schema_version",
        "preparation_manifest_sha256",
        "preparation_execution_receipt_status",
        "preparation_execution_receipt_sha256",
        "preparation_executed_source_set_sha256",
        "preparation_source_commit",
        "assignment_manifest_sha256",
        "entities_sha256",
        "clusters_sha256",
        "audit_config_sha256",
        "audit_source_sha256",
        "recovery_scenarios_sha256",
        "duplicate_audit_registration_sha256",
        "source_authority_sha256",
        "final_duplicate_audit_sha256",
        "provenance_manifest_sha256",
        "duplicate_cluster_namespace",
    }
    if not isinstance(joint_duplicate_audit_binding, Mapping) or set(
        joint_duplicate_audit_binding
    ) != expected_binding_fields:
        raise MemoryBuildError(
            "joint duplicate-audit binding fields differ from schema"
        )
    if joint_duplicate_audit_binding.get("schema_version") != (
        "table2-memory-joint-duplicate-evidence-binding-v3"
    ):
        raise MemoryBuildError("unsupported joint duplicate-audit binding")
    for field in expected_binding_fields - {
        "schema_version",
        "duplicate_cluster_namespace",
        "preparation_execution_receipt_status",
        "preparation_execution_receipt_sha256",
        "preparation_executed_source_set_sha256",
        "preparation_source_commit",
    }:
        require_sha256(
            joint_duplicate_audit_binding.get(field),
            field=f"joint_duplicate_audit_binding.{field}",
        )
    preparation_receipt_status = joint_duplicate_audit_binding.get(
        "preparation_execution_receipt_status"
    )
    if preparation_receipt_status == (
        "VALIDATED_REGISTERED_KAGGLE_PREPARE_ONLY_RECEIPT"
    ):
        for field in (
            "preparation_execution_receipt_sha256",
            "preparation_executed_source_set_sha256",
        ):
            require_sha256(
                joint_duplicate_audit_binding.get(field),
                field=f"joint_duplicate_audit_binding.{field}",
            )
        source_commit = joint_duplicate_audit_binding.get(
            "preparation_source_commit"
        )
        if (
            type(source_commit) is not str
            or len(source_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in source_commit
            )
        ):
            raise MemoryBuildError(
                "joint duplicate-audit binding has an invalid preparation source commit"
            )
    elif preparation_receipt_status == (
        "NOT_APPLICABLE_NONREGISTERED_SOURCE_AUTHORITY"
    ):
        if any(
            joint_duplicate_audit_binding.get(field) is not None
            for field in (
                "preparation_execution_receipt_sha256",
                "preparation_executed_source_set_sha256",
                "preparation_source_commit",
            )
        ):
            raise MemoryBuildError(
                "nonregistered preparation cannot claim Kaggle execution evidence"
            )
    else:
        raise MemoryBuildError(
            "joint duplicate-audit binding has an invalid preparation receipt status"
        )
    if joint_duplicate_audit_binding.get(
        "provenance_manifest_sha256"
    ) != provenance_hash:
        raise MemoryBuildError(
            "joint duplicate-audit binding cites another provenance manifest"
        )
    try:
        binding_namespace = JointDuplicateClusterNamespace.from_mapping(
            joint_duplicate_audit_binding.get("duplicate_cluster_namespace"),
            require_hashes=True,
        )
    except DuplicateAuditError as error:
        raise MemoryBuildError(
            f"invalid joint duplicate-audit binding namespace: {error}"
        ) from error
    if binding_namespace.to_dict() != duplicate_namespace.to_dict():
        raise MemoryBuildError(
            "joint duplicate-audit binding cites another cluster namespace"
        )
    registered_duplicate_binding = dict(joint_duplicate_audit_binding)
    normalized_dataset_id = str(dataset_id).strip()
    normalized_dataset_version = str(dataset_version).strip()
    if not normalized_dataset_id or not normalized_dataset_version:
        raise MemoryBuildError("dataset_id and dataset_version must be non-empty")
    calibration_replay = validate_threshold_calibration_payload(
        threshold_calibration,
        checkpoint_sha256=checkpoint_hash,
        records_sha256=records_hash,
        dataset_artifacts_sha256=dataset_artifacts_hash,
        protocol_sha256=protocol_hash,
    )
    threshold = calibration_replay.admission_threshold
    threshold_calibration_hash = canonical_sha256(threshold_calibration)
    if not isinstance(calibration_evidence, Mapping):
        raise MemoryBuildError("calibration_evidence must be a verified mapping")
    expected_evidence = {
        "schema_version": CALIBRATION_EVIDENCE_SCHEMA_VERSION,
        "source_split": "train",
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "relevance_definition": CALIBRATION_RELEVANCE_DEFINITION,
        "pair_policy": CALIBRATION_PAIR_POLICY,
        "embedding_stage": CALIBRATION_EMBEDDING_STAGE,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "normalization": CALIBRATION_NORMALIZATION,
        "similarity": CALIBRATION_SIMILARITY,
        "embedding_hash_algorithm": CALIBRATION_EMBEDDING_HASH,
        "model_seed": int(model_seed),
        "checkpoint_sha256": checkpoint_hash,
        "records_sha256": records_hash,
        "dataset_artifacts_sha256": dataset_artifacts_hash,
        "resolved_config_sha256": config_hash,
        "resolved_config_record_sha256": config_record_hash,
        "protocol_sha256": protocol_hash,
        "provenance_manifest_sha256": provenance_hash,
        "duplicate_cluster_namespace_id": duplicate_namespace.namespace_id,
        "retrieval_top_k": RUNTIME_TOP_K,
        "retrieval_tie_break": "memory_id_ascending",
    }
    for field, expected in expected_evidence.items():
        if calibration_evidence.get(field) != expected:
            raise MemoryBuildError(
                f"calibration evidence differs from memory build at {field}"
            )
    embedded_threshold = calibration_evidence.get("threshold_calibration")
    if not isinstance(embedded_threshold, Mapping) or canonical_sha256(
        embedded_threshold
    ) != threshold_calibration_hash:
        raise MemoryBuildError(
            "calibration evidence does not embed the supplied threshold replay"
        )
    if calibration_evidence.get(
        "threshold_calibration_sha256"
    ) != threshold_calibration_hash:
        raise MemoryBuildError("calibration evidence threshold hash mismatch")
    evidence_rows = calibration_evidence.get("calibration_rows")
    selection_rows = calibration_evidence.get("eligible_selection")
    if not isinstance(evidence_rows, list) or not isinstance(selection_rows, list):
        raise MemoryBuildError("calibration evidence lacks row-level bindings")
    if calibration_evidence.get("calibration_rows_sha256") != canonical_sha256(
        evidence_rows
    ):
        raise MemoryBuildError("calibration evidence row hash mismatch")
    if calibration_evidence.get("eligible_selection_sha256") != canonical_sha256(
        selection_rows
    ):
        raise MemoryBuildError("calibration eligible-selection hash mismatch")
    if (
        calibration_evidence.get("eligible_item_count")
        != len(selection.candidates)
        or calibration_evidence.get("query_count") != len(selection.candidates)
        or calibration_evidence.get("calibration_sample_count")
        != len(evidence_rows)
        or len(evidence_rows) > RUNTIME_TOP_K * len(selection.candidates)
    ):
        raise MemoryBuildError("calibration evidence counts differ from memory corpus")
    calibration_evidence_hash = canonical_sha256(calibration_evidence)

    matrix = _normalize_embeddings(embeddings)
    if matrix.shape[0] != len(selection.candidates):
        raise MemoryBuildError(
            "embedding/candidate count mismatch: "
            f"{matrix.shape[0]} vs {len(selection.candidates)}"
        )

    pairs: list[tuple[EligibleMemoryCandidate, np.ndarray]] = list(
        zip(selection.candidates, matrix, strict=True)
    )
    pairs.sort(key=lambda value: value[0].memory_id)
    ids = [candidate.memory_id for candidate, _ in pairs]
    if len(ids) != len(set(ids)):
        raise MemoryBuildError("eligible candidates contain duplicate memory IDs")
    sorted_matrix = np.stack([vector for _, vector in pairs]).astype(
        np.float32, copy=False
    )
    try:
        verification_evidence = verification_manifest_payload(
            [candidate.item for candidate, _ in pairs]
        )
    except (VerificationEvidenceError, ManifestError) as error:
        raise MemoryBuildError(
            f"cannot close independent memory-verification evidence: {error}"
        ) from error
    if verification_evidence.get("schema_version") != (
        VERIFICATION_MANIFEST_SCHEMA_VERSION
    ):
        raise MemoryBuildError("unexpected memory-verification manifest schema")
    verification_evidence_hash = canonical_sha256(verification_evidence)
    try:
        verification_records = verification_records_payload(
            [
                (candidate.item, candidate.verification_evidence)
                for candidate, _ in pairs
            ]
        )
    except (VerificationEvidenceError, ManifestError) as error:
        raise MemoryBuildError(
            f"cannot freeze independent memory-verification records: {error}"
        ) from error
    verification_records_hash = canonical_sha256(verification_records)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.building-",
        dir=destination.parent,
    ))
    try:
        embedding_path = temporary / "embeddings.npy"
        with embedding_path.open("wb") as stream:
            np.save(stream, sorted_matrix, allow_pickle=False)

        item_path = temporary / "items.jsonl"
        with item_path.open("w", encoding="utf-8", newline="\n") as stream:
            for candidate, _ in pairs:
                stream.write(json.dumps(
                    dict(candidate.item),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ))
                stream.write("\n")

        calibration_path = temporary / "threshold_calibration.json"
        _write_json(calibration_path, threshold_calibration)
        calibration_evidence_path = temporary / "calibration_evidence.json"
        _write_json(calibration_evidence_path, calibration_evidence)
        verification_records_path = temporary / "verification_evidence.json"
        _write_json(verification_records_path, verification_records)

        store_identity = canonical_sha256({
            "checkpoint_sha256": checkpoint_hash,
            "records_sha256": records_hash,
            "dataset_artifacts_sha256": dataset_artifacts_hash,
            "resolved_config_sha256": config_hash,
            "resolved_config_record_sha256": config_record_hash,
            "protocol_sha256": protocol_hash,
            "provenance_manifest_sha256": provenance_hash,
            "threshold_calibration_sha256": threshold_calibration_hash,
            "calibration_evidence_sha256": calibration_evidence_hash,
            "verification_evidence_sha256": verification_evidence_hash,
            "verification_records_sha256": verification_records_hash,
            "model_seed": int(model_seed),
            "admission_threshold": threshold,
            "memory_ids": ids,
            "duplicate_cluster_namespace": duplicate_namespace.to_dict(),
            "joint_duplicate_audit_binding": registered_duplicate_binding,
        })
        manifest = {
            "schema_version": STORE_SCHEMA_VERSION,
            "store_id": f"table2-memory-{int(model_seed)}-{store_identity[:16]}",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_split": "train",
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "model_seed": int(model_seed),
            "checkpoint_sha256": checkpoint_hash,
            "records_sha256": records_hash,
            "dataset_artifacts_sha256": dataset_artifacts_hash,
            "dataset_id": normalized_dataset_id,
            "dataset_version": normalized_dataset_version,
            "resolved_config_sha256": config_hash,
            "resolved_config_record_sha256": config_record_hash,
            "protocol_sha256": protocol_hash,
            "provenance_manifest_sha256": provenance_hash,
            "threshold_calibration_sha256": threshold_calibration_hash,
            "calibration_evidence_sha256": calibration_evidence_hash,
            "verification_evidence_sha256": verification_evidence_hash,
            "verification_records_sha256": verification_records_hash,
            "verification_evidence": verification_evidence,
            "embedding_stage": "post_action_memory_task_adapter",
            "embedding_dimension": EMBEDDING_DIMENSION,
            "normalization": "l2",
            "similarity": SIMILARITY_METRIC,
            "top_k": RUNTIME_TOP_K,
            "tie_break": "memory_id_ascending",
            "admission_threshold_source": "train_only_calibration",
            "admission_threshold": threshold,
            "same_task_exclusion": True,
            "duplicate_exclusion": True,
            "duplicate_cluster_namespace": duplicate_namespace.to_dict(),
            "runtime_writes_allowed": False,
            "item_count": len(pairs),
            "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
            "eligibility": {
                "input_rows": selection.input_rows,
                "pre_dedup_eligible_rows": selection.pre_dedup_eligible_rows,
                "stored_rows": len(pairs),
                "exclusion_counts": dict(selection.exclusion_counts),
            },
            "transition_report_sha256": canonical_sha256(transition_report),
            "files": {
                "embeddings.npy": file_descriptor(embedding_path),
                "items.jsonl": file_descriptor(item_path),
                "threshold_calibration.json": file_descriptor(calibration_path),
                "calibration_evidence.json": file_descriptor(
                    calibration_evidence_path
                ),
                "verification_evidence.json": file_descriptor(
                    verification_records_path
                ),
            },
        }
        manifest["joint_duplicate_audit_binding"] = registered_duplicate_binding
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "manifest.sha256").write_text(
            sha256_file(temporary / "manifest.json") + "\n",
            encoding="utf-8",
        )

        # Validate the complete temporary package before atomic publication, so
        # a malformed item can never become visible at the frozen destination.
        FrozenMemoryStore.load(temporary)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return FrozenMemoryStore.load(destination)
