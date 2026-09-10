"""Build provenance-bound, train-only P4 threshold-calibration evidence.

The compact threshold manifest in :mod:`web_agent.memory.manifest` can replay
threshold selection, but it cannot on its own prove where its scores or labels
came from. This offline builder closes that gap without accepting caller-
supplied score rows or relevance booleans.

Calibration treats every eligible training memory as a leave-one-out query,
applies the runtime-equivalent same-task and duplicate exclusions, and retains
only the deterministic cosine top-3 under the runtime memory-ID tie-break. A
retrieved pair is relevant exactly when both verified successful corrections
have the same registered recovery strategy. This matches the frozen runtime
intervention, which may replace the strategy but does not copy an action,
target, or parameter payload from memory. Each row still binds the executed
correction as provenance, but that field is not used as a relevance label.
Validation rebuilds the whole package from the actual memory-build inputs.

This module is construction-only. It is never imported by browser/runtime
decision code and creates no mutation API on the frozen memory store.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from web_agent.memory.eligibility import (
    EligibilitySelection,
    EligibleMemoryCandidate,
    ProvenanceManifest,
)
from web_agent.memory.index import deterministic_cosine_top_k
from web_agent.memory.manifest import (
    EMBEDDING_DIMENSION,
    ManifestError,
    RUNTIME_TOP_K,
    build_threshold_calibration_payload,
    canonical_json_bytes,
    canonical_sha256,
    require_sha256,
    validate_threshold_calibration_payload,
)
from web_agent.memory.verification import (
    VerificationEvidenceError,
    validate_provenance_verification_evidence,
)
from web_agent.runtime.duplicate_audit import (
    DuplicateAuditError,
    JointDuplicateClusterNamespace,
)


CALIBRATION_EVIDENCE_SCHEMA_VERSION = "table2-memory-calibration-evidence-v1"
CALIBRATION_RELEVANCE_DEFINITION = (
    "matching_verified_successful_correction_strategy_v1"
)
CALIBRATION_PAIR_POLICY = (
    "leave_one_out_runtime_exclusions_then_cosine_top3_memory_id_tie_break_v1"
)
CALIBRATION_EMBEDDING_STAGE = "post_action_memory_task_adapter"
CALIBRATION_NORMALIZATION = "l2"
CALIBRATION_SIMILARITY = "cosine"
CALIBRATION_EMBEDDING_HASH = "sha256_float32_little_endian_l2_vector_v1"


class CalibrationEvidenceError(ValueError):
    """Calibration evidence cannot be derived without weakening a P4 gate."""


@dataclass(frozen=True, slots=True)
class VerifiedCalibrationEvidence:
    """The reconstructed evidence identity and compact threshold payload."""

    evidence_sha256: str
    threshold_calibration: Mapping[str, Any]


def _normalize_embeddings(embeddings: np.ndarray, *, expected_rows: int) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.shape != (expected_rows, EMBEDDING_DIMENSION):
        raise CalibrationEvidenceError(
            "calibration embeddings must align with the eligible selection and "
            f"have shape [{expected_rows}, {EMBEDDING_DIMENSION}], got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise CalibrationEvidenceError("calibration embeddings contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or bool((norms <= 1e-12).any()):
        raise CalibrationEvidenceError("calibration embeddings contain zero/invalid vectors")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _embedding_sha256(vector: np.ndarray) -> str:
    # Explicit endian conversion makes the commitment portable while retaining
    # the exact float32 values written into the frozen store.
    little_endian = np.ascontiguousarray(vector, dtype="<f4")
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


def _candidate_descriptor(
    candidate: EligibleMemoryCandidate,
    *,
    vector: np.ndarray,
    provenance: ProvenanceManifest,
    field: str,
) -> dict[str, Any]:
    evidence = provenance.records.get(candidate.source_sample_id)
    if not isinstance(evidence, Mapping):
        raise CalibrationEvidenceError(
            f"{field} sample {candidate.source_sample_id!r} lacks train provenance"
        )
    if evidence.get("source_split") != "train":
        raise CalibrationEvidenceError(f"{field} provenance is not source_split=train")
    try:
        verified_evidence = validate_provenance_verification_evidence(
            evidence,
            source_sample_id=candidate.source_sample_id,
            recovery_sample_id=candidate.recovery_sample_id,
            canonical_task_id=candidate.canonical_task_id,
            episode_id=candidate.episode_id,
        )
    except (VerificationEvidenceError, ManifestError) as error:
        raise CalibrationEvidenceError(
            f"{field} lacks valid independent verification evidence: {error}"
        ) from error
    expected_provenance = {
        "source_sample_id": candidate.source_sample_id,
        "canonical_task_id": candidate.canonical_task_id,
        "episode_id": candidate.episode_id,
        "exact_duplicate_key": candidate.exact_duplicate_key,
        "near_duplicate_cluster_id": candidate.near_duplicate_cluster_id,
        "duplicate_cluster_namespace_id": candidate.duplicate_cluster_namespace_id,
        "provenance_valid": True,
        "final_task_success": True,
    }
    for name, value in expected_provenance.items():
        if evidence.get(name) != value:
            raise CalibrationEvidenceError(
                f"{field} candidate/provenance mismatch for {name}"
            )

    item = candidate.item
    expected_item = {
        "memory_id": candidate.memory_id,
        "source_split": "train",
        "source_sample_id": candidate.source_sample_id,
        "recovery_sample_id": candidate.recovery_sample_id,
        "source_task_id": candidate.canonical_task_id,
        "source_episode_id": candidate.episode_id,
        "exact_duplicate_key": candidate.exact_duplicate_key,
        "duplicate_cluster_id": candidate.near_duplicate_cluster_id,
        "duplicate_cluster_namespace_id": candidate.duplicate_cluster_namespace_id,
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
    }
    for name, value in expected_item.items():
        if item.get(name) != value:
            raise CalibrationEvidenceError(
                f"{field} candidate/item mismatch for {name}"
            )
    expected_candidate_evidence = {
        "recovery_verification_evidence_sha256": (
            verified_evidence.recovery_evidence_sha256
        ),
        "final_task_verification_evidence_sha256": (
            verified_evidence.final_task_evidence_sha256
        ),
        "verification_evidence_sha256": verified_evidence.bundle_sha256,
    }
    for name, value in expected_candidate_evidence.items():
        if getattr(candidate, name) != value:
            raise CalibrationEvidenceError(
                f"{field} candidate verification binding differs at {name}"
            )
    strategy = str(item.get("strategy") or "").strip()
    executed_action = str(item.get("executed_recovery_action") or "").strip()
    if not strategy or strategy == "NONE" or not executed_action:
        raise CalibrationEvidenceError(
            f"{field} lacks a verified correction strategy/executed action"
        )

    return {
        "memory_id": candidate.memory_id,
        "source_sample_id": candidate.source_sample_id,
        "recovery_sample_id": candidate.recovery_sample_id,
        "source_task_id": candidate.canonical_task_id,
        "source_episode_id": candidate.episode_id,
        "exact_duplicate_key": candidate.exact_duplicate_key,
        "duplicate_cluster_id": candidate.near_duplicate_cluster_id,
        "duplicate_cluster_namespace_id": candidate.duplicate_cluster_namespace_id,
        "strategy": strategy,
        "executed_recovery_action": executed_action,
        "recovery_verification_evidence_sha256": (
            verified_evidence.recovery_evidence_sha256
        ),
        "final_task_verification_evidence_sha256": (
            verified_evidence.final_task_evidence_sha256
        ),
        "verification_evidence_sha256": verified_evidence.bundle_sha256,
        "provenance_record_sha256": canonical_sha256(dict(evidence)),
        "memory_item_sha256": canonical_sha256(dict(item)),
        "embedding_sha256": _embedding_sha256(vector),
    }


def _selection_rows(
    selection: EligibilitySelection,
    matrix: np.ndarray,
    provenance: ProvenanceManifest,
) -> list[tuple[EligibleMemoryCandidate, np.ndarray, dict[str, Any]]]:
    try:
        namespace = JointDuplicateClusterNamespace.from_mapping(
            selection.duplicate_cluster_namespace,
            require_hashes=True,
        )
    except DuplicateAuditError as error:
        raise CalibrationEvidenceError(
            f"invalid calibration duplicate-cluster namespace: {error}"
        ) from error
    if namespace.namespace_id != provenance.duplicate_cluster_namespace.namespace_id:
        raise CalibrationEvidenceError(
            "eligible selection and provenance use different duplicate namespaces"
        )

    rows: list[tuple[EligibleMemoryCandidate, np.ndarray, dict[str, Any]]] = []
    source_ids: set[str] = set()
    memory_ids: set[str] = set()
    for candidate, vector in zip(selection.candidates, matrix, strict=True):
        if candidate.source_sample_id in source_ids:
            raise CalibrationEvidenceError(
                "eligible selection contains duplicate source sample: "
                f"{candidate.source_sample_id}"
            )
        if candidate.memory_id in memory_ids:
            raise CalibrationEvidenceError(
                f"eligible selection contains duplicate memory ID: {candidate.memory_id}"
            )
        source_ids.add(candidate.source_sample_id)
        memory_ids.add(candidate.memory_id)
        descriptor = _candidate_descriptor(
            candidate,
            vector=vector,
            provenance=provenance,
            field=f"selection[{candidate.source_sample_id}]",
        )
        rows.append((candidate, vector, descriptor))
    rows.sort(key=lambda row: row[0].memory_id)
    return rows


def _pair_is_eligible(
    query: EligibleMemoryCandidate,
    candidate: EligibleMemoryCandidate,
) -> bool:
    return (
        query.memory_id != candidate.memory_id
        and query.canonical_task_id != candidate.canonical_task_id
        and query.exact_duplicate_key != candidate.exact_duplicate_key
        and query.near_duplicate_cluster_id != candidate.near_duplicate_cluster_id
    )


def build_calibration_evidence(
    *,
    selection: EligibilitySelection,
    embeddings: np.ndarray,
    provenance: ProvenanceManifest,
    provenance_manifest_sha256: str,
    model_seed: int,
    checkpoint_sha256: str,
    records_sha256: str,
    dataset_artifacts_sha256: str,
    resolved_config_sha256: str,
    resolved_config_record_sha256: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Derive calibration evidence from the actual eligible P4 build corpus.

    The function accepts no row scores, pair list, or relevance labels. Pair
    enumeration and labels are deterministic consequences of frozen training
    inputs, while similarity is recomputed from normalized embeddings.
    """

    if type(model_seed) is not int or model_seed < 0:
        raise CalibrationEvidenceError("model_seed must be a non-negative integer")
    checkpoint_hash = require_sha256(checkpoint_sha256, field="checkpoint_sha256")
    records_hash = require_sha256(records_sha256, field="records_sha256")
    dataset_hash = require_sha256(
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
    if records_hash != provenance.records_sha256:
        raise CalibrationEvidenceError(
            "calibration records hash differs from train provenance"
        )
    if dataset_hash != provenance.dataset_artifacts_sha256:
        raise CalibrationEvidenceError(
            "calibration dataset-artifact hash differs from train provenance"
        )
    if any(
        (
            provenance.validation_rows_read,
            provenance.test_rows_read,
            provenance.locked_test_rows_read,
        )
    ):
        raise CalibrationEvidenceError("calibration provenance reports non-training reads")

    matrix = _normalize_embeddings(
        embeddings, expected_rows=len(selection.candidates)
    )
    selected = _selection_rows(selection, matrix, provenance)
    if len(selected) < 2:
        raise CalibrationEvidenceError(
            "leave-one-out calibration requires at least two eligible train memories"
        )

    evidence_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    eligible_pair_count = 0
    queries_with_eligible_candidates = 0
    sorted_matrix = np.stack([row[1] for row in selected]).astype(
        np.float32,
        copy=False,
    )
    memory_ids = tuple(row[0].memory_id for row in selected)
    for query, query_vector, query_descriptor in selected:
        eligible_mask = np.fromiter(
            (_pair_is_eligible(query, row[0]) for row in selected),
            dtype=np.bool_,
            count=len(selected),
        )
        query_eligible_count = int(np.count_nonzero(eligible_mask))
        eligible_pair_count += query_eligible_count
        if query_eligible_count == 0:
            continue
        queries_with_eligible_candidates += 1
        ranked = deterministic_cosine_top_k(
            sorted_matrix,
            query_vector,
            memory_ids,
            k=RUNTIME_TOP_K,
            eligible_mask=eligible_mask,
        )
        for rank, (similarity, memory_id, selected_index) in enumerate(
            ranked,
            start=1,
        ):
            candidate, _, candidate_descriptor = selected[selected_index]
            if memory_id != candidate.memory_id:  # pragma: no cover - primitive contract
                raise CalibrationEvidenceError(
                    "shared cosine primitive returned an inconsistent memory ID"
                )
            relevant = (
                query_descriptor["strategy"] == candidate_descriptor["strategy"]
            )
            row_identity = {
                "query_memory_id": query.memory_id,
                "candidate_memory_id": candidate.memory_id,
                "pair_policy": CALIBRATION_PAIR_POLICY,
            }
            row_id = f"cal_{canonical_sha256(row_identity)}"
            evidence_rows.append(
                {
                    "row_id": row_id,
                    "source_split": "train",
                    "rank": rank,
                    "query": query_descriptor,
                    "candidate": candidate_descriptor,
                    "cosine_similarity": similarity,
                    "relevant": relevant,
                    "relevance_components": {
                        "query_strategy": query_descriptor["strategy"],
                        "candidate_strategy": candidate_descriptor["strategy"],
                        "runtime_intervention_field": "strategy",
                    },
                }
            )
            threshold_rows.append(
                {
                    "row_id": row_id,
                    "source_split": "train",
                    "cosine_similarity": similarity,
                    "relevant": relevant,
                }
            )

    if not evidence_rows:
        raise CalibrationEvidenceError(
            "same-task/duplicate exclusions removed every calibration pair"
        )
    # The compact replay also enforces the scientifically necessary presence of
    # both positive and negative examples.
    try:
        threshold_calibration = build_threshold_calibration_payload(
            threshold_rows,
            checkpoint_sha256=checkpoint_hash,
            records_sha256=records_hash,
            dataset_artifacts_sha256=dataset_hash,
            protocol_sha256=protocol_hash,
        )
    except ManifestError as error:
        raise CalibrationEvidenceError(
            f"eligible train pairs cannot calibrate a binary threshold: {error}"
        ) from error

    selection_descriptors = [row[2] for row in selected]
    payload: dict[str, Any] = {
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
        "model_seed": model_seed,
        "checkpoint_sha256": checkpoint_hash,
        "records_sha256": records_hash,
        "dataset_artifacts_sha256": dataset_hash,
        "resolved_config_sha256": config_hash,
        "resolved_config_record_sha256": config_record_hash,
        "protocol_sha256": protocol_hash,
        "provenance_manifest_sha256": provenance_hash,
        "duplicate_cluster_namespace_id": (
            provenance.duplicate_cluster_namespace.namespace_id
        ),
        "eligible_item_count": len(selection_descriptors),
        "eligible_selection": selection_descriptors,
        "eligible_selection_sha256": canonical_sha256(selection_descriptors),
        "query_count": len(selected),
        "queries_with_eligible_candidates": queries_with_eligible_candidates,
        "queries_without_eligible_candidates": (
            len(selected) - queries_with_eligible_candidates
        ),
        "eligible_pair_count": eligible_pair_count,
        "retrieval_top_k": RUNTIME_TOP_K,
        "retrieval_tie_break": "memory_id_ascending",
        "calibration_sample_count": len(evidence_rows),
        "calibration_rows": evidence_rows,
        "calibration_rows_sha256": canonical_sha256(evidence_rows),
        "threshold_calibration": threshold_calibration,
        "threshold_calibration_sha256": canonical_sha256(threshold_calibration),
    }
    return payload


def validate_calibration_evidence(
    payload: Mapping[str, Any],
    *,
    selection: EligibilitySelection,
    embeddings: np.ndarray,
    provenance: ProvenanceManifest,
    provenance_manifest_sha256: str,
    model_seed: int,
    checkpoint_sha256: str,
    records_sha256: str,
    dataset_artifacts_sha256: str,
    resolved_config_sha256: str,
    resolved_config_record_sha256: str,
    protocol_sha256: str,
) -> VerifiedCalibrationEvidence:
    """Rebuild evidence from source inputs and reject any stored-field drift."""

    if not isinstance(payload, Mapping):
        raise CalibrationEvidenceError("calibration evidence must be a JSON object")
    expected = build_calibration_evidence(
        selection=selection,
        embeddings=embeddings,
        provenance=provenance,
        provenance_manifest_sha256=provenance_manifest_sha256,
        model_seed=model_seed,
        checkpoint_sha256=checkpoint_sha256,
        records_sha256=records_sha256,
        dataset_artifacts_sha256=dataset_artifacts_sha256,
        resolved_config_sha256=resolved_config_sha256,
        resolved_config_record_sha256=resolved_config_record_sha256,
        protocol_sha256=protocol_sha256,
    )
    try:
        matches = canonical_json_bytes(dict(payload)) == canonical_json_bytes(expected)
    except (TypeError, ValueError) as error:
        raise CalibrationEvidenceError(
            "calibration evidence is not canonical finite JSON"
        ) from error
    if not matches:
        raise CalibrationEvidenceError(
            "calibration evidence differs from recomputed train/model/provenance inputs"
        )
    threshold = expected["threshold_calibration"]
    try:
        validate_threshold_calibration_payload(
            threshold,
            checkpoint_sha256=checkpoint_sha256,
            records_sha256=records_sha256,
            dataset_artifacts_sha256=dataset_artifacts_sha256,
            protocol_sha256=protocol_sha256,
        )
    except ManifestError as error:  # defensive: build already performs this replay
        raise CalibrationEvidenceError(
            f"embedded threshold calibration is invalid: {error}"
        ) from error
    return VerifiedCalibrationEvidence(
        evidence_sha256=canonical_sha256(expected),
        threshold_calibration=dict(threshold),
    )


def validate_frozen_calibration_bindings(
    payload: Mapping[str, Any],
    *,
    items: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
) -> None:
    """Recompute stored top-3 rows from immutable items and embeddings.

    The original provenance document is intentionally external to the compact
    store, so its per-record hashes remain commitments. Every runtime-relevant
    endpoint field, item hash, embedding hash, exclusion, rank, score, and
    derived relevance label is nevertheless replayed from frozen payloads.
    """

    if not isinstance(payload, Mapping):
        raise CalibrationEvidenceError("stored calibration evidence must be an object")
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.shape != (len(items), EMBEDDING_DIMENSION):
        raise CalibrationEvidenceError(
            "stored calibration embeddings do not match frozen item count/dimension"
        )
    if not np.isfinite(matrix).all():
        raise CalibrationEvidenceError(
            "stored calibration embeddings contain non-finite values"
        )
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
        raise CalibrationEvidenceError(
            "stored calibration embeddings are not L2 normalized"
        )
    # Do not normalize a second time: endpoint hashes commit to the exact
    # float32 vectors already written by the store builder.
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    descriptors = payload.get("eligible_selection")
    if not isinstance(descriptors, list) or len(descriptors) != len(items):
        raise CalibrationEvidenceError(
            "stored calibration selection does not match frozen item count"
        )
    required_descriptor_fields = {
        "memory_id",
        "source_sample_id",
        "recovery_sample_id",
        "source_task_id",
        "source_episode_id",
        "exact_duplicate_key",
        "duplicate_cluster_id",
        "duplicate_cluster_namespace_id",
        "strategy",
        "executed_recovery_action",
        "recovery_verification_evidence_sha256",
        "final_task_verification_evidence_sha256",
        "verification_evidence_sha256",
        "provenance_record_sha256",
        "memory_item_sha256",
        "embedding_sha256",
    }
    frozen_rows = sorted(
        zip(items, matrix, strict=True),
        key=lambda row: str(row[0].get("memory_id") or ""),
    )
    for index, (descriptor, frozen_row) in enumerate(
        zip(descriptors, frozen_rows, strict=True)
    ):
        item, vector = frozen_row
        if not isinstance(descriptor, Mapping) or set(descriptor) != (
            required_descriptor_fields
        ):
            raise CalibrationEvidenceError(
                f"eligible_selection[{index}] descriptor fields are invalid"
            )
        expected = {
            "memory_id": item.get("memory_id"),
            "source_sample_id": item.get("source_sample_id"),
            "recovery_sample_id": item.get("recovery_sample_id"),
            "source_task_id": item.get("source_task_id"),
            "source_episode_id": item.get("source_episode_id"),
            "exact_duplicate_key": item.get("exact_duplicate_key"),
            "duplicate_cluster_id": item.get("duplicate_cluster_id"),
            "duplicate_cluster_namespace_id": item.get(
                "duplicate_cluster_namespace_id"
            ),
            "strategy": item.get("strategy"),
            "executed_recovery_action": item.get("executed_recovery_action"),
            "recovery_verification_evidence_sha256": item.get(
                "recovery_verification_evidence_sha256"
            ),
            "final_task_verification_evidence_sha256": item.get(
                "final_task_verification_evidence_sha256"
            ),
            "verification_evidence_sha256": item.get(
                "verification_evidence_sha256"
            ),
            "memory_item_sha256": canonical_sha256(dict(item)),
            "embedding_sha256": _embedding_sha256(vector),
        }
        for field, value in expected.items():
            if descriptor.get(field) != value:
                raise CalibrationEvidenceError(
                    f"eligible_selection[{index}] differs from frozen {field}"
                )
        try:
            require_sha256(
                descriptor.get("provenance_record_sha256"),
                field=f"eligible_selection[{index}].provenance_record_sha256",
            )
        except ManifestError as error:
            raise CalibrationEvidenceError(str(error)) from error

    memory_ids = tuple(str(row["memory_id"]) for row in descriptors)
    expected_rows: list[dict[str, Any]] = []
    eligible_pair_count = 0
    queries_with_candidates = 0
    for query_index, query in enumerate(descriptors):
        eligible_mask = np.fromiter(
            (
                index != query_index
                and candidate["source_task_id"] != query["source_task_id"]
                and candidate["exact_duplicate_key"]
                != query["exact_duplicate_key"]
                and candidate["duplicate_cluster_id"]
                != query["duplicate_cluster_id"]
                for index, candidate in enumerate(descriptors)
            ),
            dtype=np.bool_,
            count=len(descriptors),
        )
        eligible_count = int(np.count_nonzero(eligible_mask))
        eligible_pair_count += eligible_count
        if not eligible_count:
            continue
        queries_with_candidates += 1
        ranked = deterministic_cosine_top_k(
            matrix,
            matrix[query_index],
            memory_ids,
            k=RUNTIME_TOP_K,
            eligible_mask=eligible_mask,
        )
        for rank, (similarity, _, candidate_index) in enumerate(ranked, start=1):
            candidate = descriptors[candidate_index]
            relevant = query["strategy"] == candidate["strategy"]
            row_id = f"cal_{canonical_sha256({
                'query_memory_id': query['memory_id'],
                'candidate_memory_id': candidate['memory_id'],
                'pair_policy': CALIBRATION_PAIR_POLICY,
            })}"
            expected_rows.append(
                {
                    "row_id": row_id,
                    "source_split": "train",
                    "rank": rank,
                    "query": query,
                    "candidate": candidate,
                    "cosine_similarity": similarity,
                    "relevant": relevant,
                    "relevance_components": {
                        "query_strategy": query["strategy"],
                        "candidate_strategy": candidate["strategy"],
                        "runtime_intervention_field": "strategy",
                    },
                }
            )

    stored_rows = payload.get("calibration_rows")
    if not isinstance(stored_rows, list) or canonical_json_bytes(
        stored_rows
    ) != canonical_json_bytes(expected_rows):
        raise CalibrationEvidenceError(
            "stored calibration rows differ from frozen top-3 replay"
        )
    expected_counts = {
        "eligible_item_count": len(descriptors),
        "query_count": len(descriptors),
        "queries_with_eligible_candidates": queries_with_candidates,
        "queries_without_eligible_candidates": len(descriptors)
        - queries_with_candidates,
        "eligible_pair_count": eligible_pair_count,
        "calibration_sample_count": len(expected_rows),
    }
    for field, expected in expected_counts.items():
        if payload.get(field) != expected:
            raise CalibrationEvidenceError(
                f"stored calibration count differs at {field}"
            )
    compact_rows = [
        {
            "row_id": row["row_id"],
            "source_split": "train",
            "cosine_similarity": row["cosine_similarity"],
            "relevant": row["relevant"],
        }
        for row in expected_rows
    ]
    expected_threshold = build_threshold_calibration_payload(
        compact_rows,
        checkpoint_sha256=str(payload.get("checkpoint_sha256") or ""),
        records_sha256=str(payload.get("records_sha256") or ""),
        dataset_artifacts_sha256=str(
            payload.get("dataset_artifacts_sha256") or ""
        ),
        protocol_sha256=str(payload.get("protocol_sha256") or ""),
    )
    if canonical_json_bytes(payload.get("threshold_calibration")) != (
        canonical_json_bytes(expected_threshold)
    ):
        raise CalibrationEvidenceError(
            "stored threshold calibration differs from frozen top-3 replay"
        )
