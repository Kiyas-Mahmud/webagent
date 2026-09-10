from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from web_agent.memory.calibration_builder import (
    CALIBRATION_PAIR_POLICY,
    CALIBRATION_RELEVANCE_DEFINITION,
    CalibrationEvidenceError,
    build_calibration_evidence,
    validate_calibration_evidence,
)
from web_agent.memory.eligibility import (
    EligibilitySelection,
    EligibleMemoryCandidate,
    ProvenanceManifest,
)
from web_agent.memory.manifest import (
    EMBEDDING_DIMENSION,
    build_threshold_calibration_payload,
    canonical_sha256,
)
from web_agent.memory.index import deterministic_cosine_top_k
from web_agent.memory.verification import (
    FINAL_TASK_VERIFICATION_SCHEMA_VERSION,
    RECOVERY_VERIFICATION_SCHEMA_VERSION,
    recovery_action_evidence_sha256,
    verification_bundle_sha256,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
NAMESPACE = {
    "schema_version": "table2-joint-duplicate-cluster-namespace-v1",
    "namespace_id": "fixture-gold-v2.8-webarena-joint-v1",
    "audit_tool_id": "fixture-joint-duplicate-audit",
    "audit_tool_version": "v1",
    "audit_tool_config_sha256": SHA_A,
    "audit_tool_source_sha256": SHA_B,
}


def _verification_evidence(
    *,
    source_id: str,
    recovery_id: str,
    task_id: str,
    episode_id: str,
    strategy: str,
    action: str,
) -> dict:
    transition = {
        "recovery_sample_id": recovery_id,
        "recovery_strategy": strategy,
        "executed_recovery_action": action,
        "recovery_action_value": "",
        "failure_state": f"failure-{source_id}",
        "post_recovery_state": f"recovered-{source_id}",
    }
    recovery = {
        "schema_version": RECOVERY_VERIFICATION_SCHEMA_VERSION,
        "authority_type": "verifier",
        "authority_id": "fixture-recovery-verifier",
        "authority_version": "v1",
        "independent_verification": True,
        "source_sample_id": source_id,
        "recovery_sample_id": recovery_id,
        "canonical_task_id": task_id,
        "episode_id": episode_id,
        "pre_recovery_state_sha256": canonical_sha256(
            transition["failure_state"]
        ),
        "executed_recovery_action_sha256": recovery_action_evidence_sha256(
            transition
        ),
        "post_recovery_state_sha256": canonical_sha256(
            transition["post_recovery_state"]
        ),
        "verified_recovery_success": True,
    }
    recovery["evidence_sha256"] = canonical_sha256(recovery)
    final_task = {
        "schema_version": FINAL_TASK_VERIFICATION_SCHEMA_VERSION,
        "authority_type": "reviewer",
        "authority_id": "fixture-final-reviewer",
        "authority_version": "rubric-v1",
        "independent_verification": True,
        "source_sample_id": source_id,
        "canonical_task_id": task_id,
        "episode_id": episode_id,
        "task_specification_sha256": SHA_A,
        "terminal_state_sha256": SHA_B,
        "terminal_verifier_output_sha256": SHA_C,
        "verified_final_task_success": True,
    }
    final_task["evidence_sha256"] = canonical_sha256(final_task)
    return {
        "recovery_verification": recovery,
        "final_task_verification": final_task,
        "verification_evidence_sha256": verification_bundle_sha256(
            source_sample_id=source_id,
            recovery_sample_id=recovery_id,
            canonical_task_id=task_id,
            episode_id=episode_id,
            recovery_evidence_sha256=recovery["evidence_sha256"],
            final_task_evidence_sha256=final_task["evidence_sha256"],
        ),
    }


def _candidate(
    index: int,
    *,
    strategy: str,
    action: str,
    task_id: str | None = None,
    duplicate_cluster_id: str | None = None,
) -> EligibleMemoryCandidate:
    source_id = f"sample-{index}"
    memory_id = f"mem_{index:02d}"
    task = task_id or f"train-task-{index}"
    cluster = duplicate_cluster_id or f"near-{index}"
    exact_key = f"{index}" * 64
    recovery_id = f"recovery-{index}"
    episode_id = f"train-episode-{index}"
    verification = _verification_evidence(
        source_id=source_id,
        recovery_id=recovery_id,
        task_id=task,
        episode_id=episode_id,
        strategy=strategy,
        action=action,
    )
    recovery_verification_sha = verification["recovery_verification"][
        "evidence_sha256"
    ]
    final_task_verification_sha = verification["final_task_verification"][
        "evidence_sha256"
    ]
    item = {
        "schema_version": "table2-memory-eligibility-v1",
        "memory_id": memory_id,
        "source_split": "train",
        "source_dataset_id": "gold-v2.8",
        "source_dataset_version": "fixture",
        "source_sample_id": source_id,
        "recovery_sample_id": recovery_id,
        "source_task_id": task,
        "source_episode_id": episode_id,
        "step_index": index,
        "website_domain": "fixture.test",
        "failure_type": "NO_EFFECT",
        "failed_action": "CLICK",
        "strategy": strategy,
        "executed_recovery_action": action,
        "recovery_action_value": "",
        "reflection_text": "verified correction",
        "memory_update_flag": True,
        "verified_recovery_success": True,
        "final_task_success": True,
        "provenance_valid": True,
        "recovery_verification_evidence_sha256": recovery_verification_sha,
        "final_task_verification_evidence_sha256": final_task_verification_sha,
        "verification_evidence_sha256": verification[
            "verification_evidence_sha256"
        ],
        "exact_duplicate_key": exact_key,
        "duplicate_cluster_id": cluster,
        "duplicate_cluster_namespace_id": NAMESPACE["namespace_id"],
    }
    return EligibleMemoryCandidate(
        record_index=index - 1,
        memory_id=memory_id,
        source_sample_id=source_id,
        recovery_sample_id=recovery_id,
        canonical_task_id=task,
        episode_id=episode_id,
        step_index=index,
        exact_duplicate_key=exact_key,
        near_duplicate_cluster_id=cluster,
        duplicate_cluster_namespace_id=NAMESPACE["namespace_id"],
        recovery_verification_evidence_sha256=recovery_verification_sha,
        final_task_verification_evidence_sha256=final_task_verification_sha,
        verification_evidence_sha256=verification["verification_evidence_sha256"],
        verification_evidence=verification,
        item=item,
    )


def _fixtures(
    *,
    all_same_label: bool = False,
) -> tuple[EligibilitySelection, np.ndarray, ProvenanceManifest]:
    third_strategy = "RETRY" if all_same_label else "REPLAN"
    third_action = "CLICK" if all_same_label else "TYPE"
    candidates = (
        _candidate(1, strategy="RETRY", action="CLICK"),
        # The runtime consumes only strategy, so a different concrete action
        # with the same strategy remains a positive calibration pair.
        _candidate(2, strategy="RETRY", action="TYPE"),
        _candidate(3, strategy=third_strategy, action=third_action),
    )
    selection = EligibilitySelection(
        candidates=candidates,
        exclusion_counts={},
        input_rows=3,
        pre_dedup_eligible_rows=3,
        duplicate_cluster_namespace=dict(NAMESPACE),
    )
    embeddings = np.zeros((3, EMBEDDING_DIMENSION), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 0] = 0.9
    embeddings[1, 1] = np.sqrt(1.0 - 0.9**2)
    embeddings[2, 0] = 0.2
    embeddings[2, 2] = np.sqrt(1.0 - 0.2**2)
    provenance = ProvenanceManifest.from_mapping(
        {
            "schema_version": "table2-memory-provenance-v1",
            "source_split": "train",
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "dataset_id": "gold-v2.8",
            "dataset_version": "fixture",
            "dataset_artifacts_sha256": SHA_C,
            "records_sha256": SHA_B,
            "duplicate_cluster_namespace": dict(NAMESPACE),
            "records": {
                candidate.source_sample_id: {
                    "source_split": "train",
                    "source_sample_id": candidate.source_sample_id,
                    "provenance_valid": True,
                    "final_task_success": True,
                    "canonical_task_id": candidate.canonical_task_id,
                    "episode_id": candidate.episode_id,
                    "exact_duplicate_key": candidate.exact_duplicate_key,
                    "near_duplicate_cluster_id": (
                        candidate.near_duplicate_cluster_id
                    ),
                    "duplicate_cluster_namespace_id": NAMESPACE["namespace_id"],
                    **_verification_evidence(
                        source_id=candidate.source_sample_id,
                        recovery_id=candidate.recovery_sample_id,
                        task_id=candidate.canonical_task_id,
                        episode_id=candidate.episode_id,
                        strategy=str(candidate.item["strategy"]),
                        action=str(candidate.item["executed_recovery_action"]),
                    ),
                }
                for candidate in candidates
            },
        }
    )
    return selection, embeddings, provenance


def _build(
    selection: EligibilitySelection,
    embeddings: np.ndarray,
    provenance: ProvenanceManifest,
) -> dict:
    return build_calibration_evidence(
        selection=selection,
        embeddings=embeddings,
        provenance=provenance,
        provenance_manifest_sha256=SHA_D,
        model_seed=42,
        checkpoint_sha256=SHA_A,
        records_sha256=SHA_B,
        dataset_artifacts_sha256=SHA_C,
        resolved_config_sha256=SHA_E,
        resolved_config_record_sha256=SHA_A,
        protocol_sha256=SHA_D,
    )


def _validate(
    payload: dict,
    selection: EligibilitySelection,
    embeddings: np.ndarray,
    provenance: ProvenanceManifest,
):
    return validate_calibration_evidence(
        payload,
        selection=selection,
        embeddings=embeddings,
        provenance=provenance,
        provenance_manifest_sha256=SHA_D,
        model_seed=42,
        checkpoint_sha256=SHA_A,
        records_sha256=SHA_B,
        dataset_artifacts_sha256=SHA_C,
        resolved_config_sha256=SHA_E,
        resolved_config_record_sha256=SHA_A,
        protocol_sha256=SHA_D,
    )


def test_calibration_is_derived_from_bound_train_pairs_and_actual_embeddings():
    selection, embeddings, provenance = _fixtures()
    payload = _build(selection, embeddings, provenance)

    assert payload["source_split"] == "train"
    assert payload["validation_rows_read"] == 0
    assert payload["test_rows_read"] == 0
    assert payload["locked_test_rows_read"] == 0
    assert payload["pair_policy"] == CALIBRATION_PAIR_POLICY
    assert payload["relevance_definition"] == CALIBRATION_RELEVANCE_DEFINITION
    assert payload["eligible_item_count"] == 3
    assert payload["eligible_selection_sha256"] == canonical_sha256(
        payload["eligible_selection"]
    )
    assert payload["query_count"] == 3
    assert payload["eligible_pair_count"] == 6
    assert payload["retrieval_top_k"] == 3
    assert payload["calibration_sample_count"] == 6
    assert payload["calibration_rows_sha256"] == canonical_sha256(
        payload["calibration_rows"]
    )

    positives = [row for row in payload["calibration_rows"] if row["relevant"]]
    negatives = [row for row in payload["calibration_rows"] if not row["relevant"]]
    assert len(positives) == 2
    assert len(negatives) == 4
    for row in payload["calibration_rows"]:
        query = row["query"]
        candidate = row["candidate"]
        assert query["source_task_id"] != candidate["source_task_id"]
        assert query["exact_duplicate_key"] != candidate["exact_duplicate_key"]
        assert query["duplicate_cluster_id"] != candidate["duplicate_cluster_id"]
        assert len(query["embedding_sha256"]) == 64
        assert len(candidate["embedding_sha256"]) == 64
        expected_relevance = query["strategy"] == candidate["strategy"]
        assert row["relevant"] is expected_relevance
        assert row["relevance_components"]["runtime_intervention_field"] == "strategy"

    compact = payload["threshold_calibration"]
    assert compact["calibration_sample_count"] == 6
    assert {row["row_id"] for row in compact["calibration_rows"]} == {
        row["row_id"] for row in payload["calibration_rows"]
    }
    verified = _validate(payload, selection, embeddings, provenance)
    assert verified.evidence_sha256 == canonical_sha256(payload)
    assert dict(verified.threshold_calibration) == compact


def test_calibration_is_invariant_to_selection_iteration_order():
    selection, embeddings, provenance = _fixtures()
    expected = _build(selection, embeddings, provenance)
    reversed_selection = replace(
        selection, candidates=tuple(reversed(selection.candidates))
    )
    reversed_embeddings = embeddings[::-1].copy()

    assert _build(reversed_selection, reversed_embeddings, provenance) == expected


def test_calibration_stores_at_most_runtime_top3_per_query_without_matrix_slices():
    selection, embeddings, provenance = _fixtures()
    extra = (
        _candidate(4, strategy="RETRY", action="CLICK"),
        _candidate(5, strategy="REPLAN", action="TYPE"),
    )
    expanded_selection = replace(
        selection, candidates=selection.candidates + extra, input_rows=5,
        pre_dedup_eligible_rows=5,
    )
    expanded_embeddings = np.zeros((5, EMBEDDING_DIMENSION), dtype=np.float32)
    expanded_embeddings[:3] = embeddings
    expanded_embeddings[3, 3] = 1.0
    expanded_embeddings[4, 0] = 0.8
    expanded_embeddings[4, 4] = 0.6
    records = dict(provenance.records)
    for candidate in extra:
        records[candidate.source_sample_id] = {
            "source_split": "train",
            "source_sample_id": candidate.source_sample_id,
            "provenance_valid": True,
            "final_task_success": True,
            "canonical_task_id": candidate.canonical_task_id,
            "episode_id": candidate.episode_id,
            "exact_duplicate_key": candidate.exact_duplicate_key,
            "near_duplicate_cluster_id": candidate.near_duplicate_cluster_id,
            "duplicate_cluster_namespace_id": NAMESPACE["namespace_id"],
            **_verification_evidence(
                source_id=candidate.source_sample_id,
                recovery_id=candidate.recovery_sample_id,
                task_id=candidate.canonical_task_id,
                episode_id=candidate.episode_id,
                strategy=str(candidate.item["strategy"]),
                action=str(candidate.item["executed_recovery_action"]),
            ),
        }
    expanded_provenance = replace(provenance, records=records)

    payload = _build(
        expanded_selection,
        expanded_embeddings,
        expanded_provenance,
    )

    assert payload["query_count"] == 5
    assert payload["eligible_pair_count"] == 20
    assert payload["calibration_sample_count"] == 15
    ranks_by_query: dict[str, list[int]] = {}
    for row in payload["calibration_rows"]:
        ranks_by_query.setdefault(row["query"]["memory_id"], []).append(row["rank"])
    assert set(ranks_by_query) == {candidate.memory_id for candidate in extra}.union(
        candidate.memory_id for candidate in selection.candidates
    )
    assert all(sorted(ranks) == [1, 2, 3] for ranks in ranks_by_query.values())


def test_shared_cosine_eligible_mask_filters_before_exact_memory_id_tie_break():
    matrix = np.zeros((5, EMBEDDING_DIMENSION), dtype=np.float32)
    matrix[:, 0] = 1.0
    query = matrix[0]
    ids = ("mem-z", "mem-d", "mem-a", "mem-c", "mem-b")

    ranked = deterministic_cosine_top_k(
        matrix,
        query,
        ids,
        k=3,
        eligible_mask=np.array([False, True, True, True, True]),
    )

    assert [memory_id for _, memory_id, _ in ranked] == ["mem-a", "mem-b", "mem-c"]
    with pytest.raises(ValueError, match="eligible_mask"):
        deterministic_cosine_top_k(
            matrix,
            query,
            ids,
            eligible_mask=np.array([True, False]),
        )


def test_calibration_rejects_self_consistent_score_and_label_tampering():
    selection, embeddings, provenance = _fixtures()
    payload = _build(selection, embeddings, provenance)

    score_tamper = deepcopy(payload)
    score_tamper["calibration_rows"][0]["cosine_similarity"] = 0.123
    score_tamper["calibration_rows_sha256"] = canonical_sha256(
        score_tamper["calibration_rows"]
    )
    compact_rows = [
        {
            "row_id": row["row_id"],
            "source_split": "train",
            "cosine_similarity": row["cosine_similarity"],
            "relevant": row["relevant"],
        }
        for row in score_tamper["calibration_rows"]
    ]
    score_tamper["threshold_calibration"] = build_threshold_calibration_payload(
        compact_rows,
        checkpoint_sha256=SHA_A,
        records_sha256=SHA_B,
        dataset_artifacts_sha256=SHA_C,
        protocol_sha256=SHA_D,
    )
    score_tamper["threshold_calibration_sha256"] = canonical_sha256(
        score_tamper["threshold_calibration"]
    )
    with pytest.raises(CalibrationEvidenceError, match="recomputed"):
        _validate(score_tamper, selection, embeddings, provenance)

    label_tamper = deepcopy(payload)
    label_tamper["calibration_rows"][0]["relevant"] = not label_tamper[
        "calibration_rows"
    ][0]["relevant"]
    label_tamper["calibration_rows_sha256"] = canonical_sha256(
        label_tamper["calibration_rows"]
    )
    with pytest.raises(CalibrationEvidenceError, match="recomputed"):
        _validate(label_tamper, selection, embeddings, provenance)


def test_calibration_requires_bound_independent_memory_verification():
    selection, embeddings, provenance = _fixtures()
    records = deepcopy(dict(provenance.records))
    records["sample-1"].pop("final_task_verification")
    missing = replace(provenance, records=records)

    with pytest.raises(
        CalibrationEvidenceError,
        match="valid independent verification evidence",
    ):
        _build(selection, embeddings, missing)


def test_calibration_rejects_embedding_endpoint_and_provenance_canaries():
    selection, embeddings, provenance = _fixtures()
    payload = _build(selection, embeddings, provenance)

    changed_embeddings = embeddings.copy()
    changed_embeddings[0, 4] = 0.25
    with pytest.raises(CalibrationEvidenceError, match="recomputed"):
        _validate(payload, selection, changed_embeddings, provenance)

    endpoint_tamper = deepcopy(payload)
    endpoint_tamper["calibration_rows"][0]["query"]["source_sample_id"] = (
        "another-train-sample"
    )
    endpoint_tamper["calibration_rows_sha256"] = canonical_sha256(
        endpoint_tamper["calibration_rows"]
    )
    with pytest.raises(CalibrationEvidenceError, match="recomputed"):
        _validate(endpoint_tamper, selection, embeddings, provenance)

    provenance_canary = replace(provenance, validation_rows_read=1)
    with pytest.raises(CalibrationEvidenceError, match="non-training"):
        _build(selection, embeddings, provenance_canary)


def test_calibration_fails_without_both_relevance_classes_or_cross_task_pairs():
    selection, embeddings, provenance = _fixtures(all_same_label=True)
    with pytest.raises(CalibrationEvidenceError, match="binary threshold"):
        _build(selection, embeddings, provenance)

    same_task_candidates = []
    same_task_records = {}
    for candidate in selection.candidates:
        verification = _verification_evidence(
            source_id=candidate.source_sample_id,
            recovery_id=candidate.recovery_sample_id,
            task_id="one-train-task",
            episode_id=candidate.episode_id,
            strategy=str(candidate.item["strategy"]),
            action=str(candidate.item["executed_recovery_action"]),
        )
        recovery_digest = verification["recovery_verification"][
            "evidence_sha256"
        ]
        final_digest = verification["final_task_verification"]["evidence_sha256"]
        same_task_candidates.append(
            replace(
                candidate,
                canonical_task_id="one-train-task",
                recovery_verification_evidence_sha256=recovery_digest,
                final_task_verification_evidence_sha256=final_digest,
                verification_evidence_sha256=verification[
                    "verification_evidence_sha256"
                ],
                item={
                    **candidate.item,
                    "source_task_id": "one-train-task",
                    "recovery_verification_evidence_sha256": recovery_digest,
                    "final_task_verification_evidence_sha256": final_digest,
                    "verification_evidence_sha256": verification[
                        "verification_evidence_sha256"
                    ],
                },
            )
        )
        same_task_records[candidate.source_sample_id] = {
            **provenance.records[candidate.source_sample_id],
            "canonical_task_id": "one-train-task",
            **verification,
        }
    same_task_candidates = tuple(same_task_candidates)
    same_task_selection = replace(selection, candidates=same_task_candidates)
    same_task_provenance = replace(provenance, records=same_task_records)
    with pytest.raises(CalibrationEvidenceError, match="removed every"):
        _build(same_task_selection, embeddings, same_task_provenance)
