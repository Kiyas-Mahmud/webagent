from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from web_agent.memory import (
    FINAL_TASK_VERIFICATION_SCHEMA_VERSION,
    FrozenMemoryStore,
    ManifestError,
    MemoryEligibilityError,
    ProvenanceManifest,
    QueryExclusions,
    RECOVERY_VERIFICATION_SCHEMA_VERSION,
    build_frozen_store,
    build_threshold_calibration_payload,
    recovery_action_evidence_sha256,
    recovery_state_evidence_sha256,
    select_eligible_candidates,
    verification_bundle_sha256,
)
from web_agent.memory.calibration_builder import build_calibration_evidence
from web_agent.memory.build_pipeline import _validate_protocol, parse_args
from web_agent.eval.table2.resolved_config import (
    ResolvedConfigIdentityError,
    assert_checkpoint_config_matches,
    load_resolved_config_identity,
)
from web_agent.memory.manifest import (
    canonical_sha256,
    validate_threshold_calibration_payload,
)
from web_agent.memory.index import deterministic_cosine_top_k
from web_agent.models.model import WebAgentModel
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    ExecutionResult,
    ExecutionStatus,
    MemoryQuery,
    PolicyObservation,
    RecoveryDecision,
    RecoveryStrategy,
    SystemID,
    TransitionInput,
)
from web_agent.runtime.memory_adapter import (
    CallablePostFailureEmbeddingProvider,
    FrozenStoreMemoryReader,
    InMemoryFrozenReader,
    MemoryAdapter,
    MemoryBoundaryError,
    PostFailureEmbedding,
    PostFailureEmbeddingRequest,
    PostFailureEmbeddingProvider,
)
from web_agent.runtime.protocol import switches_for


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
JOINT_DUPLICATE_NAMESPACE = {
    "schema_version": "table2-joint-duplicate-cluster-namespace-v1",
    "namespace_id": "fixture-gold-v2.8-webarena-joint-v1",
    "audit_tool_id": "fixture-joint-duplicate-audit",
    "audit_tool_version": "v1",
    "audit_tool_config_sha256": SHA_A,
    "audit_tool_source_sha256": SHA_C,
}


def _transition(
    *,
    source_sample_id: str = "sample-1",
    recovery_sample_id: str = "recovery-1",
    task_id: str = "train-task-1",
    episode_id: str = "train-episode-1",
    strategy: str = "RETRY",
    action: str = "CLICK",
) -> dict:
    return {
        "source_sample_id": source_sample_id,
        "recovery_sample_id": recovery_sample_id,
        "task_id": task_id,
        "episode_id": episode_id,
        "source_step_index": 1,
        "recovery_strategy": strategy,
        "recovery_success": True,
        "executed_recovery_action": action,
        "failure_state": "before-recovery",
        "post_recovery_state": "after-recovery",
    }


def _verification_evidence(
    *,
    source_sample_id: str = "sample-1",
    recovery_sample_id: str = "recovery-1",
    canonical_task_id: str = "train-task-1",
    episode_id: str = "train-episode-1",
    strategy: str = "RETRY",
    action: str = "CLICK",
) -> dict:
    transition = _transition(
        source_sample_id=source_sample_id,
        recovery_sample_id=recovery_sample_id,
        task_id=canonical_task_id,
        episode_id=episode_id,
        strategy=strategy,
        action=action,
    )
    recovery = {
        "schema_version": RECOVERY_VERIFICATION_SCHEMA_VERSION,
        "authority_type": "verifier",
        "authority_id": "fixture-recovery-verifier",
        "authority_version": "v1",
        "independent_verification": True,
        "source_sample_id": source_sample_id,
        "recovery_sample_id": recovery_sample_id,
        "canonical_task_id": canonical_task_id,
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
        "authority_id": "fixture-final-task-reviewer",
        "authority_version": "rubric-v1",
        "independent_verification": True,
        "source_sample_id": source_sample_id,
        "canonical_task_id": canonical_task_id,
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
            source_sample_id=source_sample_id,
            recovery_sample_id=recovery_sample_id,
            canonical_task_id=canonical_task_id,
            episode_id=episode_id,
            recovery_evidence_sha256=recovery["evidence_sha256"],
            final_task_evidence_sha256=final_task["evidence_sha256"],
        ),
    }


def _source_record() -> dict:
    return {
        "inputs": {"website_domain": "fixture.test"},
        "labels": {
            "outcome_label": "FAILURE",
            "failure_type_4": "NO_EFFECT",
            "action_type": "CLICK",
            "recovery_strategy": "RETRY",
            "recovery_success": True,
            "memory_update_flag": True,
            "reflection_text": "retry after the state settles",
        },
        "meta": {
            "sample_id": "sample-1",
            "task_id": "train-task-1",
            "trajectory_id": "train-episode-1",
            "review_status": "approved",
            "step_index": 1,
        },
    }


def _provenance_payload(
    *, final_success: bool = True, valid: bool = True
) -> dict:
    return {
        "schema_version": "table2-memory-provenance-v1",
        "source_split": "train",
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "dataset_id": "gold-v2.8",
        "dataset_version": "fixture",
        "dataset_artifacts_sha256": SHA_C,
        "records_sha256": SHA_A,
        "duplicate_cluster_namespace": dict(JOINT_DUPLICATE_NAMESPACE),
        "records": {
            "sample-1": {
                "source_split": "train",
                "source_sample_id": "sample-1",
                "provenance_valid": valid,
                "final_task_success": final_success,
                "canonical_task_id": "train-task-1",
                "episode_id": "train-episode-1",
                "exact_duplicate_key": SHA_B,
                "near_duplicate_cluster_id": "near-1",
                "duplicate_cluster_namespace_id": JOINT_DUPLICATE_NAMESPACE[
                    "namespace_id"
                ],
                **_verification_evidence(),
            }
        },
    }


def _provenance(*, final_success: bool = True, valid: bool = True) -> ProvenanceManifest:
    return ProvenanceManifest.from_mapping(
        _provenance_payload(final_success=final_success, valid=valid)
    )


def _selection():
    return select_eligible_candidates(
        [_source_record()],
        source_split="train",
        transitions={"sample-1": _transition()},
        provenance=_provenance(),
    )


def _threshold_calibration() -> dict:
    return build_threshold_calibration_payload(
        [
            {
                "row_id": "train-calibration-1",
                "source_split": "train",
                "cosine_similarity": 0.90,
                "relevant": True,
            },
            {
                "row_id": "train-calibration-2",
                "source_split": "train",
                "cosine_similarity": 0.70,
                "relevant": False,
            },
            {
                "row_id": "train-calibration-3",
                "source_split": "train",
                "cosine_similarity": 0.42,
                "relevant": True,
            },
            {
                "row_id": "train-calibration-4",
                "source_split": "train",
                "cosine_similarity": 0.30,
                "relevant": False,
            },
        ],
        checkpoint_sha256=SHA_A,
        records_sha256=SHA_B,
        dataset_artifacts_sha256=SHA_C,
        protocol_sha256=SHA_A,
    )


def _store_calibration_fixture():
    base = _selection().candidates[0]
    candidates = [base]
    for index, (strategy, action) in enumerate(
        (("RETRY", "CLICK"), ("REPLAN", "TYPE")),
        start=2,
    ):
        source_id = f"sample-{index}"
        recovery_id = f"recovery-{index}"
        task_id = f"train-task-{index}"
        episode_id = f"train-episode-{index}"
        verification = _verification_evidence(
            source_sample_id=source_id,
            recovery_sample_id=recovery_id,
            canonical_task_id=task_id,
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
        candidate = replace(
            base,
            record_index=index - 1,
            memory_id=f"mem_zz{index}",
            source_sample_id=source_id,
            recovery_sample_id=recovery_id,
            canonical_task_id=task_id,
            episode_id=episode_id,
            step_index=index,
            exact_duplicate_key=f"{index + 2:x}" * 64,
            near_duplicate_cluster_id=f"near-{index}",
            recovery_verification_evidence_sha256=recovery_verification_sha,
            final_task_verification_evidence_sha256=final_task_verification_sha,
            verification_evidence_sha256=verification[
                "verification_evidence_sha256"
            ],
            verification_evidence=verification,
            item={
                **base.item,
                "memory_id": f"mem_zz{index}",
                "source_sample_id": source_id,
                "recovery_sample_id": recovery_id,
                "source_task_id": task_id,
                "source_episode_id": episode_id,
                "step_index": index,
                "exact_duplicate_key": f"{index + 2:x}" * 64,
                "duplicate_cluster_id": f"near-{index}",
                "strategy": strategy,
                "executed_recovery_action": action,
                "recovery_verification_evidence_sha256": (
                    recovery_verification_sha
                ),
                "final_task_verification_evidence_sha256": (
                    final_task_verification_sha
                ),
                "verification_evidence_sha256": verification[
                    "verification_evidence_sha256"
                ],
            },
        )
        candidates.append(candidate)
    selection = replace(_selection(), candidates=tuple(candidates))
    provenance_payload = _provenance_payload()
    provenance_payload["records_sha256"] = SHA_B
    provenance_payload["records"] = {
        candidate.source_sample_id: {
            "source_split": "train",
            "source_sample_id": candidate.source_sample_id,
            "provenance_valid": True,
            "final_task_success": True,
            "canonical_task_id": candidate.canonical_task_id,
            "episode_id": candidate.episode_id,
            "exact_duplicate_key": candidate.exact_duplicate_key,
            "near_duplicate_cluster_id": candidate.near_duplicate_cluster_id,
            "duplicate_cluster_namespace_id": JOINT_DUPLICATE_NAMESPACE[
                "namespace_id"
            ],
            **_verification_evidence(
                source_sample_id=candidate.source_sample_id,
                recovery_sample_id=candidate.recovery_sample_id,
                canonical_task_id=candidate.canonical_task_id,
                episode_id=candidate.episode_id,
                strategy=str(candidate.item["strategy"]),
                action=str(candidate.item["executed_recovery_action"]),
            ),
        }
        for candidate in candidates
    }
    provenance = ProvenanceManifest.from_mapping(provenance_payload)
    embeddings = np.zeros((3, 768), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    embeddings[2, 2] = 1.0
    evidence = build_calibration_evidence(
        selection=selection,
        embeddings=embeddings,
        provenance=provenance,
        provenance_manifest_sha256=SHA_B,
        model_seed=42,
        checkpoint_sha256=SHA_A,
        records_sha256=SHA_B,
        dataset_artifacts_sha256=SHA_C,
        resolved_config_sha256=SHA_C,
        resolved_config_record_sha256=SHA_B,
        protocol_sha256=SHA_A,
    )
    return selection, embeddings, evidence


def test_threshold_calibration_is_train_only_hash_closed_and_exactly_replayed():
    payload = _threshold_calibration()
    replay = validate_threshold_calibration_payload(
        payload,
        checkpoint_sha256=SHA_A,
        records_sha256=SHA_B,
        dataset_artifacts_sha256=SHA_C,
        protocol_sha256=SHA_A,
    )

    assert replay.admission_threshold == 0.42
    assert replay.operating_point() == payload["selected_operating_point"]
    assert payload["calibration_rows_sha256"] == canonical_sha256(
        payload["calibration_rows"]
    )
    assert payload["validation_rows_read"] == 0
    assert payload["test_rows_read"] == 0
    assert payload["locked_test_rows_read"] == 0

    arbitrary = deepcopy(payload)
    arbitrary["admission_threshold"] = 0.4200000000001
    with pytest.raises(ManifestError, match="exactly match"):
        validate_threshold_calibration_payload(
            arbitrary,
            checkpoint_sha256=SHA_A,
            records_sha256=SHA_B,
            dataset_artifacts_sha256=SHA_C,
            protocol_sha256=SHA_A,
        )

    tampered_rows = deepcopy(payload)
    tampered_rows["calibration_rows"][0]["relevant"] = False
    with pytest.raises(ManifestError, match="SHA-256"):
        validate_threshold_calibration_payload(
            tampered_rows,
            checkpoint_sha256=SHA_A,
            records_sha256=SHA_B,
            dataset_artifacts_sha256=SHA_C,
            protocol_sha256=SHA_A,
        )

    replayed_tamper = deepcopy(payload)
    replayed_tamper["calibration_rows"][2]["relevant"] = False
    replayed_tamper["calibration_rows_sha256"] = canonical_sha256(
        replayed_tamper["calibration_rows"]
    )
    with pytest.raises(ManifestError, match="exactly match"):
        validate_threshold_calibration_payload(
            replayed_tamper,
            checkpoint_sha256=SHA_A,
            records_sha256=SHA_B,
            dataset_artifacts_sha256=SHA_C,
            protocol_sha256=SHA_A,
        )

    for field, unregistered in (
        ("calibration_objective", "maximize_accuracy"),
        ("calibration_tie_break", "lowest_threshold"),
    ):
        changed_registration = deepcopy(payload)
        changed_registration[field] = unregistered
        with pytest.raises(ManifestError, match=field):
            validate_threshold_calibration_payload(
                changed_registration,
                checkpoint_sha256=SHA_A,
                records_sha256=SHA_B,
                dataset_artifacts_sha256=SHA_C,
                protocol_sha256=SHA_A,
            )


def test_threshold_calibration_uses_registered_highest_threshold_tie_break():
    payload = build_threshold_calibration_payload(
        [
            {
                "row_id": "tie-1",
                "source_split": "train",
                "cosine_similarity": 0.9,
                "relevant": True,
            },
            {
                "row_id": "tie-2",
                "source_split": "train",
                "cosine_similarity": 0.8,
                "relevant": False,
            },
            {
                "row_id": "tie-3",
                "source_split": "train",
                "cosine_similarity": 0.7,
                "relevant": False,
            },
            {
                "row_id": "tie-4",
                "source_split": "train",
                "cosine_similarity": 0.6,
                "relevant": True,
            },
        ],
        checkpoint_sha256=SHA_A,
        records_sha256=SHA_B,
        dataset_artifacts_sha256=SHA_C,
        protocol_sha256=SHA_A,
    )

    assert payload["admission_threshold"] == 0.9
    assert payload["calibration_tie_break"] == "highest_threshold"
    assert payload["selected_operating_point"]["objective_numerator"] == 2
    assert payload["selected_operating_point"]["objective_denominator"] == 3


@pytest.mark.parametrize(
    "field",
    ["validation_rows_read", "test_rows_read", "locked_test_rows_read"],
)
def test_threshold_calibration_rejects_every_nontraining_read(field: str):
    payload = _threshold_calibration()
    payload[field] = 1
    with pytest.raises(ManifestError, match=field):
        validate_threshold_calibration_payload(
            payload,
            checkpoint_sha256=SHA_A,
            records_sha256=SHA_B,
            dataset_artifacts_sha256=SHA_C,
            protocol_sha256=SHA_A,
        )


def test_threshold_calibration_requires_exact_checkpoint_data_protocol_bindings():
    payload = _threshold_calibration()
    binding_cases = {
        "checkpoint_sha256": (SHA_B, SHA_B, SHA_C, SHA_A),
        "records_sha256": (SHA_A, SHA_A, SHA_C, SHA_A),
        "dataset_artifacts_sha256": (SHA_A, SHA_B, SHA_B, SHA_A),
        "protocol_sha256": (SHA_A, SHA_B, SHA_C, SHA_B),
    }
    for field, bindings in binding_cases.items():
        with pytest.raises(ManifestError, match=field):
            validate_threshold_calibration_payload(
                payload,
                checkpoint_sha256=bindings[0],
                records_sha256=bindings[1],
                dataset_artifacts_sha256=bindings[2],
                protocol_sha256=bindings[3],
            )


def test_memory_build_cli_derives_threshold_and_protocol_registers_calibration():
    protocol = Path(__file__).parents[2] / "configs/eval/table2/protocol.yaml"
    _validate_protocol(protocol)
    base_args = [
        "--config", "config.yaml",
        "--checkpoint", "checkpoint.pt",
        "--data-root", "data",
        "--provenance-manifest", "provenance.json",
        "--model-seed", "42",
        "--output-dir", "memory",
    ]
    parsed = parse_args(base_args)
    assert not hasattr(parsed, "admission_threshold")
    with pytest.raises(SystemExit):
        parse_args([*base_args, "--admission-threshold", "0.42"])
    with pytest.raises(SystemExit):
        parse_args([*base_args, "--calibration-manifest", "calibration.json"])


def test_memory_eligibility_fails_closed_on_split_flags_and_provenance():
    with pytest.raises(MemoryEligibilityError):
        select_eligible_candidates(
            [_source_record()],
            source_split="validation",
            transitions={"sample-1": _transition()},
            provenance=_provenance(),
        )

    for forbidden_count in (
        "validation_rows_read",
        "test_rows_read",
        "locked_test_rows_read",
    ):
        payload = {
            "schema_version": "table2-memory-provenance-v1",
            "source_split": "train",
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "dataset_id": "gold-v2.8",
            "dataset_version": "fixture",
            "dataset_artifacts_sha256": SHA_C,
            "records_sha256": SHA_A,
            "duplicate_cluster_namespace": JOINT_DUPLICATE_NAMESPACE,
            "records": {},
        }
        payload[forbidden_count] = 1
        with pytest.raises(ValueError, match=forbidden_count):
            ProvenanceManifest.from_mapping(payload)

    wrong_namespace = _provenance_payload()
    wrong_namespace["records"]["sample-1"][
        "duplicate_cluster_namespace_id"
    ] = "different-joint-namespace"
    with pytest.raises(ValueError, match="different duplicate-cluster namespace"):
        ProvenanceManifest.from_mapping(wrong_namespace)

    unhashed_namespace = _provenance_payload()
    unhashed_namespace["duplicate_cluster_namespace"][
        "audit_tool_source_sha256"
    ] = None
    with pytest.raises(ValueError, match="supplied together"):
        ProvenanceManifest.from_mapping(unhashed_namespace)

    unflagged = _source_record()
    unflagged["labels"]["memory_update_flag"] = False
    with pytest.raises(MemoryEligibilityError):
        select_eligible_candidates(
            [unflagged],
            source_split="train",
            transitions={"sample-1": _transition()},
            provenance=_provenance(),
        )

    with pytest.raises(MemoryEligibilityError):
        select_eligible_candidates(
            [_source_record()],
            source_split="train",
            transitions={"sample-1": _transition()},
            provenance=_provenance(final_success=False),
        )

    missing_transition = deepcopy(_transition())
    missing_transition.pop("post_recovery_state")
    with pytest.raises(MemoryEligibilityError):
        select_eligible_candidates(
            [_source_record()],
            source_split="train",
            transitions={"sample-1": missing_transition},
            provenance=_provenance(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("final_review_decision", "quarantine_policy"),
        ("quarantined", True),
        ("memory_admission_allowed", False),
    ),
)
def test_memory_eligibility_excludes_explicit_source_non_admission_without_overlay(
    field: str, value: object
):
    blocked = _source_record()
    blocked["meta"][field] = value

    selection = select_eligible_candidates(
        [blocked, _source_record()],
        source_split="train",
        transitions={"sample-1": _transition()},
        provenance=_provenance(),
    )

    assert len(selection.candidates) == 1
    assert selection.exclusion_counts["quarantined_or_non_admitted"] == 1


def test_memory_eligibility_excludes_nested_or_provenance_quarantine_markers():
    nested = _source_record()
    nested["meta"]["targeted_review_overlay"] = {
        "status": "quarantine_policy"
    }
    with pytest.raises(MemoryEligibilityError, match="no eligible"):
        select_eligible_candidates(
            [nested],
            source_split="train",
            transitions={"sample-1": _transition()},
            provenance=_provenance(),
        )

    payload = _provenance_payload()
    payload["records"]["sample-1"]["memory_admission_allowed"] = False
    with pytest.raises(MemoryEligibilityError, match="no eligible"):
        select_eligible_candidates(
            [_source_record()],
            source_split="train",
            transitions={"sample-1": _transition()},
            provenance=ProvenanceManifest.from_mapping(payload),
        )


def test_memory_eligibility_rejects_malformed_non_admission_boolean():
    malformed = _source_record()
    malformed["meta"]["quarantined"] = "false"
    with pytest.raises(MemoryEligibilityError, match="quarantined must be a JSON boolean"):
        select_eligible_candidates(
            [malformed],
            source_split="train",
            transitions={"sample-1": _transition()},
            provenance=_provenance(),
        )


def test_memory_eligibility_requires_independent_hashed_verification_evidence():
    missing = _provenance_payload()
    missing["records"]["sample-1"].pop("recovery_verification")
    with pytest.raises(MemoryEligibilityError, match="recovery_verification"):
        select_eligible_candidates(
            [_source_record()],
            source_split="train",
            transitions={"sample-1": _transition()},
            provenance=ProvenanceManifest.from_mapping(missing),
        )

    tampered = _provenance_payload()
    tampered["records"]["sample-1"]["final_task_verification"][
        "terminal_state_sha256"
    ] = SHA_C
    with pytest.raises(MemoryEligibilityError, match="evidence_sha256"):
        select_eligible_candidates(
            [_source_record()],
            source_split="train",
            transitions={"sample-1": _transition()},
            provenance=ProvenanceManifest.from_mapping(tampered),
        )

    transition_mismatch = _provenance_payload()
    record = transition_mismatch["records"]["sample-1"]
    recovery = record["recovery_verification"]
    recovery["pre_recovery_state_sha256"] = SHA_C
    recovery["evidence_sha256"] = canonical_sha256(
        {key: value for key, value in recovery.items() if key != "evidence_sha256"}
    )
    record["verification_evidence_sha256"] = verification_bundle_sha256(
        source_sample_id="sample-1",
        recovery_sample_id="recovery-1",
        canonical_task_id="train-task-1",
        episode_id="train-episode-1",
        recovery_evidence_sha256=recovery["evidence_sha256"],
        final_task_evidence_sha256=record["final_task_verification"][
            "evidence_sha256"
        ],
    )
    with pytest.raises(MemoryEligibilityError, match="differs from transition"):
        select_eligible_candidates(
            [_source_record()],
            source_split="train",
            transitions={"sample-1": _transition()},
            provenance=ProvenanceManifest.from_mapping(transition_mismatch),
        )

    wrong_task = _provenance_payload()
    wrong_task_record = wrong_task["records"]["sample-1"]
    wrong_task_record["canonical_task_id"] = "reattached-task"
    wrong_task_record.update(
        _verification_evidence(canonical_task_id="reattached-task")
    )
    with pytest.raises(MemoryEligibilityError, match="task identity differs"):
        select_eligible_candidates(
            [_source_record()],
            source_split="train",
            transitions={"sample-1": _transition()},
            provenance=ProvenanceManifest.from_mapping(wrong_task),
        )

    wrong_episode = _provenance_payload()
    wrong_episode_record = wrong_episode["records"]["sample-1"]
    wrong_episode_record["episode_id"] = "reattached-episode"
    wrong_episode_record.update(
        _verification_evidence(episode_id="reattached-episode")
    )
    with pytest.raises(MemoryEligibilityError, match="episode identity differs"):
        select_eligible_candidates(
            [_source_record()],
            source_split="train",
            transitions={"sample-1": _transition()},
            provenance=ProvenanceManifest.from_mapping(wrong_episode),
        )


def test_memory_verification_hashes_state_artifact_bytes_not_only_paths(tmp_path):
    (tmp_path / "failure.png").write_bytes(b"failure-state-v1")
    (tmp_path / "recovered.png").write_bytes(b"recovered-state-v1")
    transition = _transition()
    transition.update(
        {
            "failure_state": "failure.png",
            "post_recovery_state": "recovered.png",
            "_data_root": str(tmp_path),
        }
    )
    payload = _provenance_payload()
    record = payload["records"]["sample-1"]
    recovery = record["recovery_verification"]
    recovery["pre_recovery_state_sha256"] = recovery_state_evidence_sha256(
        transition,
        field="failure_state",
    )
    recovery["post_recovery_state_sha256"] = recovery_state_evidence_sha256(
        transition,
        field="post_recovery_state",
    )
    recovery["evidence_sha256"] = canonical_sha256(
        {key: value for key, value in recovery.items() if key != "evidence_sha256"}
    )
    record["verification_evidence_sha256"] = verification_bundle_sha256(
        source_sample_id="sample-1",
        recovery_sample_id="recovery-1",
        canonical_task_id="train-task-1",
        episode_id="train-episode-1",
        recovery_evidence_sha256=recovery["evidence_sha256"],
        final_task_evidence_sha256=record["final_task_verification"][
            "evidence_sha256"
        ],
    )
    provenance = ProvenanceManifest.from_mapping(payload)
    selection = select_eligible_candidates(
        [_source_record()],
        source_split="train",
        transitions={"sample-1": transition},
        provenance=provenance,
    )
    assert len(selection.candidates) == 1

    (tmp_path / "failure.png").write_bytes(b"failure-state-v2")
    with pytest.raises(MemoryEligibilityError, match="differs from transition"):
        select_eligible_candidates(
            [_source_record()],
            source_split="train",
            transitions={"sample-1": transition},
            provenance=provenance,
        )


def test_frozen_store_is_hash_verified_read_only_and_query_excludes_same_task(tmp_path):
    selection, embedding, calibration_evidence = _store_calibration_fixture()
    store = build_frozen_store(
        tmp_path / "seed_42",
        selection=selection,
        embeddings=embedding,
        model_seed=42,
        checkpoint_sha256=SHA_A,
        records_sha256=SHA_B,
        dataset_id="gold-v2.8",
        dataset_version="fixture",
        dataset_artifacts_sha256=SHA_C,
        resolved_config_sha256=SHA_C,
        resolved_config_record_sha256=SHA_B,
        protocol_sha256=SHA_A,
        provenance_manifest_sha256=SHA_B,
        threshold_calibration=calibration_evidence["threshold_calibration"],
        calibration_evidence=calibration_evidence,
        transition_report={"status": "PASS"},
    )

    assert len(store) == 3
    assert store.manifest["schema_version"] == "table2-frozen-memory-v2"
    assert store.manifest["resolved_config_sha256"] == SHA_C
    assert store.manifest["resolved_config_record_sha256"] == SHA_B
    assert store.manifest["verification_evidence"]["item_count"] == 3
    assert len(store.manifest["verification_evidence_sha256"]) == 64
    assert store.manifest["duplicate_cluster_namespace"] == JOINT_DUPLICATE_NAMESPACE
    assert (
        store._items[0]["duplicate_cluster_namespace_id"]  # noqa: SLF001
        == JOINT_DUPLICATE_NAMESPACE["namespace_id"]
    )
    assert store.admission_threshold == pytest.approx(
        calibration_evidence["threshold_calibration"]["admission_threshold"]
    )
    assert not any(hasattr(store, name) for name in ("append", "update", "delete", "save"))
    with pytest.raises(ValueError):
        store._embeddings[0, 0] = 0.0  # noqa: SLF001 - verifies immutable backing data

    query = np.zeros(768, dtype=np.float32)
    query[0] = 1.0
    allowed = store.query(
        query,
        exclusions=QueryExclusions(
            current_task_id="different-task",
            current_episode_id="different-episode",
            duplicate_cluster_ids=frozenset({"different-near-cluster"}),
            additional_memory_ids=frozenset({"mem_zz2", "mem_zz3"}),
        ),
    )
    assert len(allowed.hits) == 1
    assert allowed.hits[0].rank == 1
    assert allowed.hits[0].cosine_similarity == pytest.approx(1.0)

    same_task = store.query(
        query,
        exclusions=QueryExclusions(
            current_task_id="train-task-1",
            current_episode_id="different-episode",
            duplicate_cluster_ids=frozenset({"different-near-cluster"}),
            additional_memory_ids=frozenset({"mem_zz2", "mem_zz3"}),
        ),
    )
    assert same_task.hits == ()
    assert same_task.exclusion_reasons[selection.candidates[0].memory_id] == (
        "same_task"
    )

    bad_candidate = replace(
        selection.candidates[0],
        duplicate_cluster_namespace_id="different-joint-namespace",
    )
    with pytest.raises(ValueError, match="different duplicate-cluster namespace"):
        build_frozen_store(
            tmp_path / "seed_42_namespace_canary",
            selection=replace(
                selection,
                candidates=(bad_candidate, *selection.candidates[1:]),
            ),
            embeddings=embedding,
            model_seed=42,
            checkpoint_sha256=SHA_A,
            records_sha256=SHA_B,
            dataset_id="gold-v2.8",
            dataset_version="fixture",
            dataset_artifacts_sha256=SHA_C,
            resolved_config_sha256=SHA_C,
            resolved_config_record_sha256=SHA_B,
            protocol_sha256=SHA_A,
            provenance_manifest_sha256=SHA_B,
            threshold_calibration=calibration_evidence["threshold_calibration"],
            calibration_evidence=calibration_evidence,
            transition_report={"status": "PASS"},
        )

    missing_records_candidate = replace(
        selection.candidates[0], verification_evidence={}
    )
    with pytest.raises(ValueError, match="does not retain its independent"):
        build_frozen_store(
            tmp_path / "seed_42_missing_verification_records",
            selection=replace(
                selection,
                candidates=(
                    missing_records_candidate,
                    *selection.candidates[1:],
                ),
            ),
            embeddings=embedding,
            model_seed=42,
            checkpoint_sha256=SHA_A,
            records_sha256=SHA_B,
            dataset_id="gold-v2.8",
            dataset_version="fixture",
            dataset_artifacts_sha256=SHA_C,
            resolved_config_sha256=SHA_C,
            resolved_config_record_sha256=SHA_B,
            protocol_sha256=SHA_A,
            provenance_manifest_sha256=SHA_B,
            threshold_calibration=calibration_evidence["threshold_calibration"],
            calibration_evidence=calibration_evidence,
            transition_report={"status": "PASS"},
        )


def test_frozen_store_revalidates_item_verification_evidence_on_load(tmp_path):
    selection, embeddings, evidence = _store_calibration_fixture()
    store = build_frozen_store(
        tmp_path / "seed_42_verification_tamper",
        selection=selection,
        embeddings=embeddings,
        model_seed=42,
        checkpoint_sha256=SHA_A,
        records_sha256=SHA_B,
        dataset_id="gold-v2.8",
        dataset_version="fixture",
        dataset_artifacts_sha256=SHA_C,
        resolved_config_sha256=SHA_C,
        resolved_config_record_sha256=SHA_B,
        protocol_sha256=SHA_A,
        provenance_manifest_sha256=SHA_B,
        threshold_calibration=evidence["threshold_calibration"],
        calibration_evidence=evidence,
        transition_report={"status": "PASS"},
    )

    item_path = store.root / "items.jsonl"
    items = [json.loads(line) for line in item_path.read_text().splitlines()]
    items[0]["recovery_verification_evidence_sha256"] = SHA_C
    item_path.write_text(
        "\n".join(
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in items
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = store.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["items.jsonl"] = {
        "sha256": hashlib.sha256(item_path.read_bytes()).hexdigest(),
        "bytes": item_path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (store.root / "manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="independent verification evidence"):
        FrozenMemoryStore.load(store.root)


def test_frozen_store_rejects_self_consistent_manifest_verification_reattachment(
    tmp_path,
):
    selection, embeddings, evidence = _store_calibration_fixture()
    store = build_frozen_store(
        tmp_path / "seed_42_verification_manifest_tamper",
        selection=selection,
        embeddings=embeddings,
        model_seed=42,
        checkpoint_sha256=SHA_A,
        records_sha256=SHA_B,
        dataset_id="gold-v2.8",
        dataset_version="fixture",
        dataset_artifacts_sha256=SHA_C,
        resolved_config_sha256=SHA_C,
        resolved_config_record_sha256=SHA_B,
        protocol_sha256=SHA_A,
        provenance_manifest_sha256=SHA_B,
        threshold_calibration=evidence["threshold_calibration"],
        calibration_evidence=evidence,
        transition_report={"status": "PASS"},
    )

    manifest_path = store.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    closure = manifest["verification_evidence"]
    closure["items"][0]["final_task_verification_evidence_sha256"] = "d" * 64
    closure["items_sha256"] = canonical_sha256(closure["items"])
    manifest["verification_evidence_sha256"] = canonical_sha256(closure)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (store.root / "manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="differs from manifest closure"):
        FrozenMemoryStore.load(store.root)


def test_frozen_store_rejects_fully_resealed_verification_record_tamper(tmp_path):
    selection, embeddings, evidence = _store_calibration_fixture()
    store = build_frozen_store(
        tmp_path / "seed_42_verification_record_tamper",
        selection=selection,
        embeddings=embeddings,
        model_seed=42,
        checkpoint_sha256=SHA_A,
        records_sha256=SHA_B,
        dataset_id="gold-v2.8",
        dataset_version="fixture",
        dataset_artifacts_sha256=SHA_C,
        resolved_config_sha256=SHA_C,
        resolved_config_record_sha256=SHA_B,
        protocol_sha256=SHA_A,
        provenance_manifest_sha256=SHA_B,
        threshold_calibration=evidence["threshold_calibration"],
        calibration_evidence=evidence,
        transition_report={"status": "PASS"},
    )

    evidence_path = store.root / "verification_evidence.json"
    frozen_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    row = frozen_evidence["records"][0]
    final_task = row["final_task_verification"]
    final_task["terminal_state_sha256"] = "d" * 64
    final_task["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in final_task.items()
            if key != "evidence_sha256"
        }
    )
    row["final_task_verification_evidence_sha256"] = final_task[
        "evidence_sha256"
    ]
    row["verification_evidence_sha256"] = verification_bundle_sha256(
        source_sample_id=row["source_sample_id"],
        recovery_sample_id=row["recovery_sample_id"],
        canonical_task_id=row["canonical_task_id"],
        episode_id=row["episode_id"],
        recovery_evidence_sha256=row[
            "recovery_verification_evidence_sha256"
        ],
        final_task_evidence_sha256=row[
            "final_task_verification_evidence_sha256"
        ],
    )
    row["record_sha256"] = canonical_sha256(
        {key: value for key, value in row.items() if key != "record_sha256"}
    )
    frozen_evidence["records_sha256"] = canonical_sha256(
        frozen_evidence["records"]
    )
    evidence_path.write_text(
        json.dumps(frozen_evidence, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = store.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verification_records_sha256"] = canonical_sha256(frozen_evidence)
    manifest["files"]["verification_evidence.json"] = {
        "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "bytes": evidence_path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (store.root / "manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="differs from verification record"):
        FrozenMemoryStore.load(store.root)


def test_resolved_config_identity_precedes_runtime_path_overrides(tmp_path: Path):
    """Regression: the old builder hashed the runtime-mutated config mapping."""

    selected_full_config = {
        "name": "fixture_FULL_SEED42",
        "seeds": [42],
        "fused_dim": 768,
        "data": {
            "root": "/training/original-gold",
            "causal_routing": True,
            "use_state_after": True,
        },
        "model": {"task_adapters": {"enabled": True}},
        "backbone": {
            "family": "fixture",
            "vlm_model": "fixture/model",
            "revision": "pinned",
            "path": "/training/model",
        },
    }
    config_path = tmp_path / "selected-resolved-config.json"
    config_path.write_text(
        json.dumps(selected_full_config, indent=2) + "\n",
        encoding="utf-8",
    )

    identity = load_resolved_config_identity(config_path)
    runtime_config = deepcopy(identity.mapping)
    runtime_config["data"]["root"] = "/memory-build/mounted-gold"
    runtime_config["data"]["review_overlay_dir"] = "/memory-build/review"

    assert identity.payload_sha256 == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    assert identity.record_sha256 == canonical_sha256(selected_full_config)
    assert identity.payload_sha256 != identity.record_sha256
    assert canonical_sha256(runtime_config) != identity.record_sha256
    assert_checkpoint_config_matches(selected_full_config, identity)
    with pytest.raises(
        ResolvedConfigIdentityError,
        match="checkpoint-saved full configuration differs",
    ):
        assert_checkpoint_config_matches(runtime_config, identity)


def test_frozen_store_replays_calibration_rows_after_hash_consistent_tamper(
    tmp_path: Path,
):
    selection, embeddings, evidence = _store_calibration_fixture()
    store = build_frozen_store(
        tmp_path / "seed_42_calibration_tamper",
        selection=selection,
        embeddings=embeddings,
        model_seed=42,
        checkpoint_sha256=SHA_A,
        records_sha256=SHA_B,
        dataset_id="gold-v2.8",
        dataset_version="fixture",
        dataset_artifacts_sha256=SHA_C,
        resolved_config_sha256=SHA_C,
        resolved_config_record_sha256=SHA_B,
        protocol_sha256=SHA_A,
        provenance_manifest_sha256=SHA_B,
        threshold_calibration=evidence["threshold_calibration"],
        calibration_evidence=evidence,
        transition_report={"status": "PASS"},
    )
    evidence_path = store.root / "calibration_evidence.json"
    tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
    tampered["calibration_rows"][0]["cosine_similarity"] = 0.123
    tampered["calibration_rows_sha256"] = canonical_sha256(
        tampered["calibration_rows"]
    )
    evidence_path.write_text(
        json.dumps(tampered, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = store.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["calibration_evidence_sha256"] = canonical_sha256(tampered)
    manifest["files"]["calibration_evidence.json"] = {
        "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "bytes": evidence_path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (store.root / "manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="top-3 replay"):
        FrozenMemoryStore.load(store.root)


class _MemoryEmbeddingProbe(WebAgentModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.causal_routing = True
        self.task_adapters = nn.ModuleDict({"memory": nn.Identity()})

    def encode(self, batch: dict, prefix: str = ""):
        assert prefix == "post_"
        return {"fused": batch["post_fused"]}


def test_model_memory_embedding_returns_exact_post_action_task_adapter_tensor():
    model = _MemoryEmbeddingProbe()
    batch = {"post_fused": torch.arange(768, dtype=torch.float32).reshape(1, 768)}
    with pytest.raises(RuntimeError):
        model.memory_embedding(batch)

    model.eval()
    result = model.memory_embedding(batch)
    assert torch.equal(result, batch["post_fused"])
    assert result.shape == (1, 768)
    assert result.requires_grad is False


def test_model_memory_embedding_rejects_noncausal_route():
    model = _MemoryEmbeddingProbe()
    model.causal_routing = False
    model.eval()
    batch = {"post_fused": torch.zeros((1, 768), dtype=torch.float32)}

    with pytest.raises(RuntimeError, match="causal post-action"):
        model.memory_embedding(batch)


def test_shared_cosine_search_uses_memory_id_tie_break():
    embeddings = np.zeros((2, 768), dtype=np.float32)
    embeddings[:, 0] = 1.0
    query = np.zeros(768, dtype=np.float32)
    query[0] = 1.0

    ranked = deterministic_cosine_top_k(
        embeddings,
        query,
        ("mem-z", "mem-a"),
        k=2,
    )

    assert [memory_id for _, memory_id, _ in ranked] == ["mem-a", "mem-z"]


def _post_failure_embedding_request(query: MemoryQuery) -> PostFailureEmbeddingRequest:
    pre = PolicyObservation(
        task_id=query.task_id,
        goal="complete the task",
        observation_id="pre-observation",
        screenshot_sha256="1" * 64,
        screenshot_path=None,
        width=1280,
        height=720,
        url="https://fixture.invalid/before",
        title="before",
        current_page_state={"state": "before"},
    )
    post = PolicyObservation(
        task_id=query.task_id,
        goal="complete the task",
        observation_id=query.post_failure_observation_id,
        screenshot_sha256="2" * 64,
        screenshot_path=None,
        width=1280,
        height=720,
        url="https://fixture.invalid/after",
        title="after",
        current_page_state={"state": "after"},
    )
    failed_action = ConcreteAction(
        action_id=query.failed_action_id,
        source_decision_id="decision-1",
        action_type=ActionType.CLICK,
        parameters={"x": 10, "y": 20},
    )
    transition = TransitionInput(
        task_id=query.task_id,
        pre_observation=pre,
        executed_action=failed_action,
        execution_result=ExecutionResult(
            action_id=failed_action.action_id,
            status=ExecutionStatus.EXECUTED,
            executor_step=1,
            state_changed=False,
        ),
        post_observation=post,
    )
    return PostFailureEmbeddingRequest(
        query_id=query.query_id,
        post_failure_observation_id=query.post_failure_observation_id,
        failed_action_id=query.failed_action_id,
        post_action_input=transition,
        post_failure_observation_sha256=str(
            query.post_failure_observation_sha256
        ),
        post_action_input_sha256=transition.record_sha256,
        processor_contract_sha256=str(query.processor_contract_sha256),
        checkpoint_sha256=str(query.checkpoint_sha256),
    )


def _embedding_receipt(
    request: PostFailureEmbeddingRequest,
    values: np.ndarray,
    *,
    processed_batch_sha256: str = SHA_B,
    post_failure_observation_sha256: str | None = None,
) -> PostFailureEmbedding:
    vector = np.asarray(values, dtype=np.float32)
    return PostFailureEmbedding(
        query_id=request.query_id,
        post_failure_observation_id=request.post_failure_observation_id,
        values=tuple(float(value) for value in vector),
        request_sha256=request.record_sha256,
        post_failure_observation_sha256=(
            post_failure_observation_sha256
            if post_failure_observation_sha256 is not None
            else request.post_failure_observation_sha256
        ),
        post_action_input_sha256=request.post_action_input_sha256,
        processor_contract_sha256=request.processor_contract_sha256,
        checkpoint_sha256=request.checkpoint_sha256,
        processed_batch_sha256=processed_batch_sha256,
        embedding_sha256=hashlib.sha256(vector.tobytes(order="C")).hexdigest(),
    )


def test_e3_runtime_bridge_queries_real_frozen_store_with_exact_embedding(tmp_path):
    selection, embedding, calibration_evidence = _store_calibration_fixture()
    store = build_frozen_store(
        tmp_path / "seed_42_bridge",
        selection=selection,
        embeddings=embedding,
        model_seed=42,
        checkpoint_sha256=SHA_A,
        records_sha256=SHA_B,
        dataset_id="gold-v2.8",
        dataset_version="fixture",
        dataset_artifacts_sha256=SHA_C,
        resolved_config_sha256=SHA_C,
        resolved_config_record_sha256=SHA_B,
        protocol_sha256=SHA_A,
        provenance_manifest_sha256=SHA_B,
        threshold_calibration=calibration_evidence["threshold_calibration"],
        calibration_evidence=calibration_evidence,
        transition_report={"status": "PASS"},
    )
    provider = CallablePostFailureEmbeddingProvider(
        embedder=lambda request: _embedding_receipt(request, embedding[0]),
        provider_id="exact-model-memory-embedding",
        provider_version="checkpoint-a",
        checkpoint_sha256=SHA_A,
        processor_contract_sha256=SHA_C,
    )
    reader = FrozenStoreMemoryReader(
        store,
        embedding_provider=provider,
        expected_model_seed=42,
    )
    adapter = MemoryAdapter(
        switches_for(SystemID.E3),
        reader=reader,
        admission_threshold=store.admission_threshold,
        evaluation_mode=True,
        expected_store_manifest_sha256=store.manifest_sha256,
    )
    query = MemoryQuery(
        query_id="query-1",
        task_id="evaluation-task",
        episode_id="evaluation-episode",
        incident_id="incident-1",
        post_failure_observation_id="post-failure-observation",
        failed_action_id="failed-action",
        diagnosis="NO_EFFECT",
        duplicate_cluster_ids=("evaluation-near-cluster",),
        post_failure_observation_sha256=SHA_A,
        post_action_input_sha256=SHA_B,
        processor_contract_sha256=SHA_C,
        checkpoint_sha256=SHA_A,
    )
    embedding_request = _post_failure_embedding_request(query)
    query = replace(
        query,
        post_action_input_sha256=embedding_request.post_action_input_sha256,
    )
    shadow = RecoveryDecision(
        decision_id="shadow-1",
        incident_id=query.incident_id,
        strategy=RecoveryStrategy.BACKTRACK,
        trigger_sources=("policy",),
        diagnosis=query.diagnosis,
    )

    result = adapter.apply_post_failure(
        query=query,
        shadow_decision=shadow,
        rng=random.Random(42),
        embedding_request=embedding_request,
    )
    assert result.query_result.candidate_ids
    assert result.query_result.query_embedding_sha256 is not None
    assert result.query_result.embedding_binding_sha256 is not None
    assert result.query_result.reader_considered_count == 3
    assert result.query_result.reader_eligible_count == 3
    assert result.query_result.reader_id == "frozen-memory-store-v1"
    assert result.query_result.store_manifest_sha256 == store.manifest_sha256
    assert result.query_result.admitted is True
    assert result.final_decision.strategy is RecoveryStrategy.RETRY
    with pytest.raises(MemoryBoundaryError, match="seed"):
        adapter.assert_model_seed(43)

    with pytest.raises(MemoryBoundaryError, match="attested memory store"):
        MemoryAdapter(
            switches_for(SystemID.E3),
            reader=reader,
            admission_threshold=store.admission_threshold,
            evaluation_mode=True,
            expected_store_manifest_sha256="f" * 64,
        )

    unregistered_reader = FrozenStoreMemoryReader(
        store,
        embedding_provider=provider,
        reader_id="lookalike-reader",
        expected_model_seed=42,
    )
    with pytest.raises(MemoryBoundaryError, match="reader identity"):
        MemoryAdapter(
            switches_for(SystemID.E3),
            reader=unregistered_reader,
            admission_threshold=store.admission_threshold,
            evaluation_mode=True,
            expected_store_manifest_sha256=store.manifest_sha256,
        )

    with pytest.raises(MemoryBoundaryError, match="exact boolean"):
        MemoryAdapter(
            switches_for(SystemID.E3),
            reader=reader,
            admission_threshold=store.admission_threshold,
            evaluation_mode=1,  # type: ignore[arg-type]
            expected_store_manifest_sha256=store.manifest_sha256,
        )

    class WrongObservationReceipt(PostFailureEmbeddingProvider):
        def embed(
            self, current: PostFailureEmbeddingRequest
        ) -> PostFailureEmbedding:
            return _embedding_receipt(
                current,
                embedding[0],
                post_failure_observation_sha256=SHA_B,
            )

    hostile_reader = FrozenStoreMemoryReader(
        store,
        embedding_provider=WrongObservationReceipt(),
        expected_model_seed=42,
    )
    with pytest.raises(MemoryBoundaryError, match="causal model-input binding"):
        hostile_reader.query(
            query,
            k=3,
            rng=random.Random(42),
            embedding_request=embedding_request,
        )


def test_fixture_reader_is_explicitly_forbidden_from_evaluation_p4():
    reader = InMemoryFrozenReader(
        reader_id="fixture-reader",
        index_sha256=SHA_A,
        candidates=(),
        model_seed=42,
        registered_admission_threshold=0.5,
    )
    assert reader.evidence_scope == "ENGINEERING_SMOKE_ONLY"
    # Fixture/smoke use remains available when evaluation_mode is explicit false.
    MemoryAdapter(
        switches_for(SystemID.E3),
        reader=reader,
        admission_threshold=0.5,
        evaluation_mode=False,
    )
    with pytest.raises(MemoryBoundaryError, match="exact attested"):
        MemoryAdapter(
            switches_for(SystemID.E3),
            reader=reader,
            admission_threshold=0.5,
            evaluation_mode=True,
            expected_store_manifest_sha256=SHA_A,
        )
