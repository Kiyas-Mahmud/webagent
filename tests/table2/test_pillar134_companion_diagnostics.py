from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any
import threading

import pytest

from web_agent.eval.table2.common import SchemaError, sha256_bytes, sha256_file
from web_agent.eval.table2.companion_diagnostics import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    BOOTSTRAP_CONFIDENCE,
    CANONICAL_INPUT_PROVENANCE_STATUS,
    CompanionBackendIdentity,
    CompanionDiagnosticError,
    ENGINEERING_EVIDENCE_ROLE,
    INPUT_PROVENANCE_BLOCKED_PROMOTION,
    ImageArtifact,
    ObservableTextState,
    PC01_CHECKPOINT_SHA256,
    PC01_CONFIG_SHA256,
    PC01_MODEL_ID,
    PC01_MODEL_REVISION,
    PC01_PROCESSOR_SHA256,
    run_attestation,
)
from web_agent.eval.table2.pc01_companion_diagnostics import (
    PC01_P1_PREDICTOR_ID,
    PC01_P1_PREDICTOR_VERSION,
    PC01_P3_PREDICTOR_ID,
    PC01_P3_PREDICTOR_VERSION,
    PC01_P4_RETRIEVER_ID,
    PC01_P4_RETRIEVER_VERSION,
    PC01_P4_EMBEDDING_PROVIDER_ID,
    PC01_P4_EMBEDDING_PROVIDER_MODULE,
    PC01_P4_EMBEDDING_PROVIDER_QUALNAME,
    PC01_P4_EMBEDDING_PROVIDER_VERSION,
    PC01Pillar1DiagnosticPredictor,
    PC01Pillar3DiagnosticPredictor,
    PC01Pillar4DiagnosticRetriever,
    pc01_companion_predictor_source_sha256,
    pc01_memory_embedding_source_sha256,
    pc01_parameter_provider_source_sha256,
)
from web_agent.runtime.action_parameters import (
    ParameterResolutionError,
    build_registered_hybrid_parameter_provider,
)
from web_agent.runtime.contracts import ActionType, PreActionDecision
from scripts.table2_companion_diagnostic_bootstrap import (
    _verify_clean_repository,
    _verify_committed_source,
)
from web_agent.eval.table2.pillar1_diagnostics import (
    EVIDENCE_ROLE as P1_ROLE,
    P1Example,
    P1RecoveryEvidence,
    P1RecoveryPrediction,
    P1RecoveryView,
    P1Targets,
    P1TransitionPrediction,
    P1TransitionView,
    Pillar1DiagnosticInput,
    canonical_p1_backend_error,
    run_pillar1_diagnostics,
    validate_pillar1_diagnostic_package,
    validate_pillar1_diagnostic_report,
    write_pillar1_diagnostic_package,
)
from web_agent.eval.table2.pillar3_diagnostics import (
    EVIDENCE_ROLE as P3_ROLE,
    PC01_PARAMETER_PROVIDER_ID,
    PC01_PARAMETER_PROVIDER_PROMPT_SHA256,
    PC01_PARAMETER_PROVIDER_VERSION,
    P3DiagnosticView,
    P3Example,
    P3Prediction,
    P3Targets,
    Pillar3DiagnosticInput,
    ProviderAttemptEvidence,
    canonical_p3_backend_error,
    run_pillar3_diagnostics,
    validate_pillar3_diagnostic_package,
    validate_pillar3_diagnostic_report,
    write_pillar3_diagnostic_package,
)
from web_agent.eval.table2.pillar4_diagnostics import (
    EVIDENCE_ROLE as P4_ROLE,
    P4Candidate,
    P4Example,
    P4QueryView,
    P4RetrievalPrediction,
    Pillar4DiagnosticInput,
    canonical_p4_backend_error,
    float32_vector_sha256,
    run_pillar4_diagnostics,
    validate_pillar4_diagnostic_package,
    validate_pillar4_diagnostic_report,
    write_pillar4_diagnostic_package,
)


SHA = "a" * 64
COMMIT = "b" * 40
ACTION_LABELS = ("CLICK", "TYPE", "SELECT", "SCROLL", "NAVIGATE", "PRESS_KEY")
FAILURE_LABELS = ("NONE", "PERCEPTION_ERROR", "ACTION_MISMATCH", "LOOP_DETECTED")
RECOVERY_LABELS = ("NONE", "RETRY", "REPLAN", "BACKTRACK", "ALTERNATIVE_TARGET", "ABORT")


def create_test_companion():  # pragma: no cover - factory identity fixture only
    raise AssertionError("test-only companion factory must not run")


def _source_hash() -> str:
    return sha256_file(Path(__file__))


def _identity(pillar: str, qualname: str, *, memory_sha: str | None = None) -> CompanionBackendIdentity:
    provider = pillar == "P3"
    return CompanionBackendIdentity(
        pillar=pillar,
        predictor_id=f"fixture-{pillar.lower()}",
        predictor_version="v1",
        predictor_module=__name__,
        predictor_qualname=qualname,
        factory_entrypoint=f"{__name__}:create_test_companion",
        model_id="fixture-model",
        model_revision="fixture-revision",
        checkpoint_sha256="1" * 64,
        resolved_config_sha256="2" * 64,
        resolved_config_record_sha256="3" * 64,
        processor_contract_sha256="4" * 64,
        factory_source_sha256=_source_hash(),
        predictor_source_sha256=_source_hash(),
        repository_commit=COMMIT,
        model_seed=42,
        provider_module="fixture.provider" if provider else None,
        provider_qualname="FixtureProvider" if provider else None,
        provider_id="fixture-provider" if provider else None,
        provider_version="v1" if provider else None,
        provider_policy_source="fixture" if provider else None,
        provider_source_sha256="5" * 64 if provider else None,
        provider_prompt_sha256="6" * 64 if provider else None,
        memory_manifest_sha256=memory_sha,
        embedding_provider_id="fixture-embedding" if pillar == "P4" else None,
        embedding_provider_version="v1" if pillar == "P4" else None,
        embedding_provider_module="fixture.embedding" if pillar == "P4" else None,
        embedding_provider_qualname="FixtureEmbedding" if pillar == "P4" else None,
        embedding_provider_source_sha256="7" * 64 if pillar == "P4" else None,
    )


def _image(root: Path, name: str, payload: bytes) -> ImageArtifact:
    path = root / name
    path.write_bytes(payload)
    path.chmod(0o444)
    return ImageArtifact(name, name, sha256_bytes(payload), 16, 12)


def _text(*, page_state: dict[str, Any] | None = None) -> ObservableTextState:
    return ObservableTextState(
        task_text="Submit the public fixture form",
        domain="example.test",
        current_url="https://example.test/form",
        title="Fixture",
        page_state={"visible_text": "form"} if page_state is None else page_state,
    )


def _distribution(labels: tuple[str, ...], selected: str) -> dict[str, float]:
    base = 0.1 / (len(labels) - 1)
    values = {label: base for label in labels}
    values[selected] = 0.9
    return values


class DeterministicP1Predictor:
    def __init__(self, identity: CompanionBackendIdentity) -> None:
        self._identity = identity
        self.transition_views: list[P1TransitionView] = []
        self.recovery_views: list[P1RecoveryView] = []

    @property
    def diagnostic_identity(self) -> CompanionBackendIdentity:
        return self._identity

    def predict_transition(self, view: P1TransitionView, *, seed: int) -> P1TransitionPrediction:
        del seed
        self.transition_views.append(view)
        failed = view.task_id.endswith("failure")
        return P1TransitionPrediction(
            failure_probability=0.9 if failed else 0.1,
            failure_type_probabilities=_distribution(
                FAILURE_LABELS, "ACTION_MISMATCH" if failed else "NONE"
            ),
            needs_recovery_probability=0.9 if failed else 0.1,
            recovery_probabilities=_distribution(
                RECOVERY_LABELS, "RETRY" if failed else "NONE"
            ),
        )

    def predict_recovery(self, view: P1RecoveryView, *, seed: int) -> P1RecoveryPrediction:
        del seed
        self.recovery_views.append(view)
        return P1RecoveryPrediction(0.8, 0.8)


def _p1_input(tmp_path: Path) -> tuple[Pillar1DiagnosticInput, Path]:
    root = tmp_path / "p1-images"
    root.mkdir()
    pre1 = _image(root, "pre1.bin", b"pre-one")
    post1 = _image(root, "post1.bin", b"post-one")
    recovery_pre = _image(root, "recovery-pre.bin", b"recovery-pre")
    recovery_post = _image(root, "recovery-post.bin", b"recovery-post")
    pre2 = _image(root, "pre2.bin", b"pre-two")
    post2 = _image(root, "post2.bin", b"post-two")
    examples = (
        P1Example(
            "p1-failure", "task-failure", pre1, post1, _text(), "CLICK",
            ("policy",), P1Targets(True, "ACTION_MISMATCH", True, "RETRY"),
            P1RecoveryEvidence(
                "attempt-1", "RETRY", recovery_pre, recovery_post, ("CLICK",),
                True, True, "7" * 64,
            ),
            "8" * 64,
        ),
        P1Example(
            "p1-success", "task-success", pre2, post2, _text(), "PRESS_KEY",
            (), P1Targets(False, "NONE", False, "NONE"), None, "9" * 64,
        ),
    )
    return Pillar1DiagnosticInput(
        diagnostic_id="p1-fixture-v1",
        campaign_id="public-pilot-fixture",
        source_partition="completed_public_pilot_campaign",
        registered_seed=42,
        bootstrap_samples=100,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
        bootstrap_seed=123,
        examples=examples,
        backend_identity=_identity("P1", "DeterministicP1Predictor"),
        actions_frozen_before_diagnostics=True,
        labels_attached_after_execution=True,
        oracle_labels_visible_to_predictor=False,
        runtime_outputs_consumed=False,
        affects_primary_table2=False,
        locked_test_rows_read=0,
    ), root


def test_p1_scores_frozen_post_action_and_recovery_evidence_without_runtime_effects(tmp_path: Path) -> None:
    diagnostic, root = _p1_input(tmp_path)
    predictor = DeterministicP1Predictor(diagnostic.backend_identity)
    report = run_pillar1_diagnostics(
        diagnostic,
        evidence_root=root,
        predictor=predictor,
        allow_uncommitted_engineering=True,
    )
    assert report["evidence_role"] == ENGINEERING_EVIDENCE_ROLE
    assert report["promotion_status"] == "UNPROMOTABLE"
    assert report["paper_table_status"] == "N/R"
    assert report["browser_actions_executed"] == 0
    assert report["recovery_triggers_emitted"] == 0
    assert report["summary"]["overall"]["failure_detection_accuracy"]["numerator"] == 2
    assert report["summary"]["overall"]["failure_detection_accuracy"]["denominator"] == 2
    assert report["summary"]["overall"]["failure_detection_accuracy"]["confidence"] == 0.95
    assert not hasattr(predictor.transition_views[0], "targets")
    assert not hasattr(predictor.recovery_views[0], "failure_resolved")
    validate_pillar1_diagnostic_report(report)

    output = tmp_path / "p1-package"
    write_pillar1_diagnostic_package(output, diagnostic=diagnostic, report=report)
    assert validate_pillar1_diagnostic_package(output)["status"] == "PASS"


def test_p1_oracle_targets_cannot_change_predictions_and_oracle_page_state_is_rejected(tmp_path: Path) -> None:
    diagnostic, root = _p1_input(tmp_path)
    first = run_pillar1_diagnostics(
        diagnostic,
        evidence_root=root,
        predictor=DeterministicP1Predictor(diagnostic.backend_identity),
        allow_uncommitted_engineering=True,
    )
    changed_examples = tuple(
        replace(example, targets=P1Targets(False, "NONE", False, "NONE"), recovery=None)
        for example in diagnostic.examples
    )
    changed = replace(diagnostic, examples=changed_examples)
    second = run_pillar1_diagnostics(
        changed,
        evidence_root=root,
        predictor=DeterministicP1Predictor(changed.backend_identity),
        allow_uncommitted_engineering=True,
    )
    first_hashes = [row["prediction_sha256"] for row in first["records"] if row["record_type"] == "transition"]
    second_hashes = [row["prediction_sha256"] for row in second["records"]]
    assert first_hashes == second_hashes
    with pytest.raises(SchemaError, match="sealed/oracle"):
        _text(page_state={"oracle_success": True})


def _parameters(action: str) -> tuple[tuple[float, float, float, float] | None, dict[str, Any]]:
    bbox = (0.1, 0.2, 0.2, 0.2) if action in {"CLICK", "TYPE", "SELECT"} else None
    target = {"target_x": 0.2, "target_y": 0.3, "target_bbox": [0.1, 0.2, 0.2, 0.2]}
    values = {
        "CLICK": {**target, "button": "left", "click_count": 1},
        "TYPE": {**target, "text": "fixture"},
        "SELECT": {**target, "option": "one", "candidate_options": ["one", "two"]},
        "SCROLL": {"direction": "down", "amount": 0.75, "container": "viewport"},
        "NAVIGATE": {"url": "https://example.test/next"},
        "PRESS_KEY": {"key": "ENTER"},
    }[action]
    return bbox, values


class DeterministicP3Predictor:
    def __init__(self, identity: CompanionBackendIdentity) -> None:
        self._identity = identity
        self.views: list[P3DiagnosticView] = []

    @property
    def diagnostic_identity(self) -> CompanionBackendIdentity:
        return self._identity

    def predict(self, view: P3DiagnosticView, *, seed: int) -> P3Prediction:
        del seed
        self.views.append(view)
        action = ACTION_LABELS[int(view.example_id.rsplit("-", 1)[1])]
        bbox, parameters = _parameters(action)
        return P3Prediction(
            status="RESOLVED",
            action_probabilities=_distribution(ACTION_LABELS, action),
            action_type=action,
            bbox=bbox,
            grounding_confidence=0.9,
            parameter_status="RESOLVED",
            provider_action_type=action,
            parameters=parameters,
            provider_attempts=(ProviderAttemptEvidence("deterministic", "RESOLVED", None),),
            error_sha256=None,
            accounted_rejected_executor_requests=0,
        )


class FirstRejectedP3Predictor(DeterministicP3Predictor):
    def predict(self, view: P3DiagnosticView, *, seed: int) -> P3Prediction:
        if view.example_id == "p3-0":
            self.views.append(view)
            return P3Prediction(
                status="POLICY_REJECTED",
                action_probabilities=None,
                action_type=None,
                bbox=None,
                grounding_confidence=None,
                parameter_status="NOT_ATTEMPTED",
                provider_action_type=None,
                parameters=None,
                provider_attempts=(),
                error_sha256=SHA,
                accounted_rejected_executor_requests=1,
            )
        return super().predict(view, seed=seed)


def _p3_input(tmp_path: Path) -> tuple[Pillar3DiagnosticInput, Path]:
    root = tmp_path / "p3-images"
    root.mkdir()
    examples = []
    for index, action in enumerate(ACTION_LABELS):
        bbox, parameters = _parameters(action)
        examples.append(
            P3Example(
                f"p3-{index}",
                f"p3-task-{index % 2}",
                _image(root, f"p3-{index}.bin", f"image-{index}".encode()),
                _text(
                    page_state=(
                        {
                            "observable_select_controls": [
                                {"bbox": list(bbox), "options": ["one", "two"]}
                            ]
                        }
                        if action == "SELECT"
                        else {"visible_text": action}
                    )
                ),
                P3Targets(action, bbox, parameters),
                f"{index + 1:x}" * 64,
            )
        )
    return Pillar3DiagnosticInput(
        diagnostic_id="p3-fixture-v1",
        campaign_id="public-development-fixture",
        source_partition="public_development",
        registered_seed=42,
        bootstrap_samples=100,
        bootstrap_confidence=0.95,
        bootstrap_seed=456,
        examples=tuple(examples),
        backend_identity=_identity("P3", "DeterministicP3Predictor"),
        actions_frozen_before_diagnostics=True,
        targets_visible_to_predictor=False,
        invalid_output_policy="one_rejected_executor_request_no_repair",
        runtime_outputs_consumed=False,
        affects_primary_table2=False,
        locked_test_rows_read=0,
    ), root


def test_p3_reports_all_six_classes_grounding_parameters_and_no_side_effects(tmp_path: Path) -> None:
    diagnostic, root = _p3_input(tmp_path)
    predictor = DeterministicP3Predictor(diagnostic.backend_identity)
    report = run_pillar3_diagnostics(
        diagnostic,
        evidence_root=root,
        predictor=predictor,
        allow_uncommitted_engineering=True,
    )
    assert set(report["summary"]["by_action_class"]) == set(ACTION_LABELS)
    assert report["summary"]["overall"]["resolved_output_rate"]["numerator"] == 6
    assert report["summary"]["overall"]["bbox_recall_iou50"]["denominator"] == 3
    assert report["summary"]["invalid_output_count"] == 0
    assert report["browser_actions_executed"] == 0
    assert all(not hasattr(view, "targets") for view in predictor.views)
    validate_pillar3_diagnostic_report(report)
    output = tmp_path / "p3-package"
    write_pillar3_diagnostic_package(output, diagnostic=diagnostic, report=report)
    assert validate_pillar3_diagnostic_package(output)["status"] == "PASS"


def test_p3_invalid_output_accounting_forbids_silent_repair() -> None:
    with pytest.raises(SchemaError, match="exactly one request"):
        P3Prediction(
            status="POLICY_REJECTED",
            action_probabilities=None,
            action_type=None,
            bbox=None,
            grounding_confidence=None,
            parameter_status="NOT_ATTEMPTED",
            provider_action_type=None,
            parameters=None,
            provider_attempts=(),
            error_sha256=SHA,
            accounted_rejected_executor_requests=0,
        )
    with pytest.raises(SchemaError, match="repair loop"):
        P3Prediction(
            status="PARAMETER_REJECTED",
            action_probabilities=_distribution(ACTION_LABELS, "CLICK"),
            action_type="CLICK",
            bbox=(0.1, 0.2, 0.2, 0.2),
            grounding_confidence=0.9,
            parameter_status="REJECTED",
            provider_action_type=None,
            parameters=None,
            provider_attempts=(
                ProviderAttemptEvidence("deterministic", "REJECTED", "1" * 64),
                ProviderAttemptEvidence("frozen_base_fallback", "REJECTED", "2" * 64),
                ProviderAttemptEvidence("deterministic", "REJECTED", "3" * 64),
            ),
            error_sha256=SHA,
            accounted_rejected_executor_requests=1,
        )


def test_p3_grounding_denominator_counts_a_rejected_grounded_target_as_zero(
    tmp_path: Path,
) -> None:
    diagnostic, root = _p3_input(tmp_path)
    identity = _identity("P3", "FirstRejectedP3Predictor")
    diagnostic = replace(diagnostic, backend_identity=identity)
    report = run_pillar3_diagnostics(
        diagnostic,
        evidence_root=root,
        predictor=FirstRejectedP3Predictor(identity),
        allow_uncommitted_engineering=True,
    )
    recall = report["summary"]["overall"]["bbox_recall_iou50"]
    mean_iou = report["summary"]["overall"]["bbox_iou_mean"]
    assert recall["denominator"] == 3
    assert recall["numerator"] == 2
    assert mean_iou["count"] == 3
    assert mean_iou["estimate"] == pytest.approx(2 / 3)


def test_p3_targets_are_structurally_blind_to_pre_action_predictor(tmp_path: Path) -> None:
    diagnostic, root = _p3_input(tmp_path)
    baseline = run_pillar3_diagnostics(
        diagnostic,
        evidence_root=root,
        predictor=DeterministicP3Predictor(diagnostic.backend_identity),
        allow_uncommitted_engineering=True,
    )
    changed_examples = list(diagnostic.examples)
    changed_examples[0] = replace(
        changed_examples[0],
        targets=P3Targets("PRESS_KEY", None, {"key": "ENTER"}),
    )
    changed = replace(diagnostic, examples=tuple(changed_examples))
    altered = run_pillar3_diagnostics(
        changed,
        evidence_root=root,
        predictor=DeterministicP3Predictor(changed.backend_identity),
        allow_uncommitted_engineering=True,
    )
    assert [row["prediction_sha256"] for row in baseline["records"]] == [
        row["prediction_sha256"] for row in altered["records"]
    ]


class DeterministicP4Retriever:
    def __init__(
        self,
        identity: CompanionBackendIdentity,
        store_root: Path,
        *,
        leak: bool = False,
        same_strategy_for_last: bool = False,
    ) -> None:
        self._identity = identity
        self._store_root = store_root
        self.leak = leak
        self.same_strategy_for_last = same_strategy_for_last
        self.views: list[P4QueryView] = []

    @property
    def diagnostic_identity(self) -> CompanionBackendIdentity:
        return self._identity

    @property
    def store_root(self) -> Path:
        return self._store_root

    def retrieve(self, view: P4QueryView, *, seed: int) -> P4RetrievalPrediction:
        del seed
        self.views.append(view)
        candidate = P4Candidate(
            rank=1,
            memory_id="memory-retry",
            similarity=0.8,
            source_split="validation" if self.leak else "train",
            source_task_id="other-task",
            source_episode_id="other-episode",
            duplicate_cluster_id="other-cluster",
            strategy=(
                "REPLAN"
                if self.same_strategy_for_last and view.task_id.endswith("1")
                else "RETRY"
            ),
            memory_update_flag=True,
            verified_recovery_success=True,
            final_task_success=True,
            provenance_valid=True,
        )
        return P4RetrievalPrediction(
            query_embedding_sha256=view.query_embedding_sha256,
            store_manifest_sha256=self._identity.memory_manifest_sha256 or "",
            admission_threshold=0.5,
            candidates=(candidate,),
            exclusion_reasons={"excluded-same-task": "same_task"},
            considered_count=2,
            eligible_count=1,
            write_enabled=False,
        )


def _p4_input(tmp_path: Path) -> tuple[Pillar4DiagnosticInput, Path]:
    root = tmp_path / "frozen-memory"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text('{"fixture":"frozen-train-only"}\n', encoding="utf-8")
    evidence = root / "items.jsonl"
    evidence.write_text('{"memory_id":"memory-retry"}\n', encoding="utf-8")
    manifest.chmod(0o444)
    evidence.chmod(0o444)
    root.chmod(0o555)
    identity = _identity(
        "P4", "DeterministicP4Retriever", memory_sha=sha256_file(manifest)
    )
    vector = tuple([1.0] + [0.0] * 767)
    examples = tuple(
        P4Example(
            example_id=f"p4-{index}",
            task_id=f"task-{index}",
            episode_id=f"episode-{index}",
            incident_id=f"incident-{index}",
            query_stage="post_action_failure",
            post_failure_observation_id=f"post-{index}",
            failed_action_id=f"action-{index}",
            duplicate_cluster_ids=(f"query-cluster-{index}",),
            query_embedding=vector,
            query_embedding_sha256=float32_vector_sha256(vector),
            embedding_request_sha256=f"{index + 1:x}" * 64,
            processed_batch_sha256=f"{index + 3:x}" * 64,
            checkpoint_sha256=identity.checkpoint_sha256,
            processor_contract_sha256=identity.processor_contract_sha256,
            shadow_strategy="REPLAN",
            target_strategy="RETRY",
            relevant_memory_ids=("memory-retry",),
            source_record_sha256=f"{index + 5:x}" * 64,
        )
        for index in range(2)
    )
    return Pillar4DiagnosticInput(
        diagnostic_id="p4-fixture-v1",
        campaign_id="train-fixture",
        source_partition="train_diagnostic",
        registered_seed=42,
        bootstrap_samples=100,
        bootstrap_confidence=0.95,
        bootstrap_seed=789,
        diagnostic_mode="PRIMARY_INTERVENTION",
        admission_threshold=0.5,
        examples=examples,
        backend_identity=identity,
        actions_frozen_before_diagnostics=True,
        store_frozen_read_only=True,
        relevance_labels_visible_to_retriever=False,
        evaluation_memory_write_enabled=False,
        runtime_outputs_consumed=False,
        affects_primary_table2=False,
        locked_test_rows_read=0,
    ), root


def test_p4_retrieval_admission_intervention_and_read_only_store_are_separate_from_table2(tmp_path: Path) -> None:
    diagnostic, root = _p4_input(tmp_path)
    retriever = DeterministicP4Retriever(diagnostic.backend_identity, root)
    report = run_pillar4_diagnostics(
        diagnostic,
        evidence_root=root,
        predictor=retriever,
        allow_uncommitted_engineering=True,
    )
    assert report["evidence_role"] == ENGINEERING_EVIDENCE_ROLE
    assert report["claim_scope"]["end_to_end_memory_benefit"].startswith("NOT_MEASURED")
    assert report["summary"]["strategy_hit_at_1"]["numerator"] == 2
    assert report["summary"]["intervention_coverage"]["numerator"] == 2
    assert report["summary"]["no_write_verified_rate"]["numerator"] == 2
    assert report["memory_writes_executed"] == 0
    assert all(not hasattr(view, "relevant_memory_ids") for view in retriever.views)
    validate_pillar4_diagnostic_report(report)

    output = tmp_path / "p4-package"
    write_pillar4_diagnostic_package(output, diagnostic=diagnostic, report=report)
    assert validate_pillar4_diagnostic_package(output)["status"] == "PASS"


def test_p4_relevance_labels_cannot_change_retrieval_and_leakage_fails(tmp_path: Path) -> None:
    diagnostic, root = _p4_input(tmp_path)
    first = run_pillar4_diagnostics(
        diagnostic,
        evidence_root=root,
        predictor=DeterministicP4Retriever(diagnostic.backend_identity, root),
        allow_uncommitted_engineering=True,
    )
    changed = replace(
        diagnostic,
        examples=tuple(replace(row, relevant_memory_ids=()) for row in diagnostic.examples),
    )
    second = run_pillar4_diagnostics(
        changed,
        evidence_root=root,
        predictor=DeterministicP4Retriever(changed.backend_identity, root),
        allow_uncommitted_engineering=True,
    )
    assert [row["prediction_sha256"] for row in first["records"]] == [
        row["prediction_sha256"] for row in second["records"]
    ]
    with pytest.raises(CompanionDiagnosticError, match="non-training"):
        run_pillar4_diagnostics(
            diagnostic,
            evidence_root=root,
            predictor=DeterministicP4Retriever(diagnostic.backend_identity, root, leak=True),
            allow_uncommitted_engineering=True,
        )


def test_p4_intervention_coverage_is_distinct_from_changed_strategy_rate(
    tmp_path: Path,
) -> None:
    diagnostic, root = _p4_input(tmp_path)
    report = run_pillar4_diagnostics(
        diagnostic,
        evidence_root=root,
        predictor=DeterministicP4Retriever(
            diagnostic.backend_identity, root, same_strategy_for_last=True
        ),
        allow_uncommitted_engineering=True,
    )
    assert report["summary"]["intervention_coverage"]["numerator"] == 2
    assert report["summary"]["changed_strategy_rate"]["numerator"] == 1


def _canonical_identity(pillar: str) -> CompanionBackendIdentity:
    classes = {
        "P1": (PC01_P1_PREDICTOR_ID, PC01_P1_PREDICTOR_VERSION, PC01Pillar1DiagnosticPredictor),
        "P3": (PC01_P3_PREDICTOR_ID, PC01_P3_PREDICTOR_VERSION, PC01Pillar3DiagnosticPredictor),
        "P4": (PC01_P4_RETRIEVER_ID, PC01_P4_RETRIEVER_VERSION, PC01Pillar4DiagnosticRetriever),
    }
    predictor_id, predictor_version, predictor_class = classes[pillar]
    provider = pillar == "P3"
    return CompanionBackendIdentity(
        pillar=pillar,
        predictor_id=predictor_id,
        predictor_version=predictor_version,
        predictor_module=predictor_class.__module__,
        predictor_qualname=predictor_class.__qualname__,
        factory_entrypoint="deployment.pc01_companion:create_predictor",
        model_id=PC01_MODEL_ID,
        model_revision=PC01_MODEL_REVISION,
        checkpoint_sha256=PC01_CHECKPOINT_SHA256,
        resolved_config_sha256=PC01_CONFIG_SHA256,
        resolved_config_record_sha256=PC01_CONFIG_SHA256,
        processor_contract_sha256=PC01_PROCESSOR_SHA256,
        factory_source_sha256="d" * 64,
        predictor_source_sha256=pc01_companion_predictor_source_sha256(),
        repository_commit=COMMIT,
        model_seed=42,
        provider_module="web_agent.runtime.action_parameters" if provider else None,
        provider_qualname="HybridParameterProvider" if provider else None,
        provider_id=PC01_PARAMETER_PROVIDER_ID if provider else None,
        provider_version=PC01_PARAMETER_PROVIDER_VERSION if provider else None,
        provider_policy_source=PC01_PARAMETER_PROVIDER_ID if provider else None,
        provider_source_sha256=pc01_parameter_provider_source_sha256() if provider else None,
        provider_prompt_sha256=PC01_PARAMETER_PROVIDER_PROMPT_SHA256 if provider else None,
        memory_manifest_sha256="e" * 64 if pillar == "P4" else None,
        embedding_provider_id=PC01_P4_EMBEDDING_PROVIDER_ID if pillar == "P4" else None,
        embedding_provider_version=PC01_P4_EMBEDDING_PROVIDER_VERSION if pillar == "P4" else None,
        embedding_provider_module=PC01_P4_EMBEDDING_PROVIDER_MODULE if pillar == "P4" else None,
        embedding_provider_qualname=PC01_P4_EMBEDDING_PROVIDER_QUALNAME if pillar == "P4" else None,
        embedding_provider_source_sha256=(
            pc01_memory_embedding_source_sha256() if pillar == "P4" else None
        ),
    )


@pytest.mark.parametrize(
    ("pillar", "role", "check"),
    (("P1", P1_ROLE, canonical_p1_backend_error),
     ("P3", P3_ROLE, canonical_p3_backend_error),
     ("P4", P4_ROLE, canonical_p4_backend_error)),
)
def test_only_exact_pc01_identities_are_canonical_and_generic_fixtures_stay_unpromotable(
    pillar: str, role: str, check: Any
) -> None:
    canonical = _canonical_identity(pillar)
    assert check(canonical, require_input_provenance=False) is None
    assert "BLOCKED_AUTHORITATIVE_INPUT_PROVENANCE_REQUIRED" in str(
        check(canonical)
    )
    assert check(replace(canonical, model_revision="0" * 40)) is not None
    assert role != ENGINEERING_EVIDENCE_ROLE
    generic: Any
    if pillar == "P1":
        generic = DeterministicP1Predictor(canonical)
    elif pillar == "P3":
        generic = DeterministicP3Predictor(canonical)
    else:
        generic = object()
    assert "exact" in str(check(canonical, predictor=generic))


def test_exact_backend_with_unproven_inputs_uses_clean_but_blocked_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity("P1", "DeterministicP1Predictor")
    predictor = DeterministicP1Predictor(identity)
    clean_receipt = {
        "repository_commit": identity.repository_commit,
        "git_clean": True,
        "git_status_porcelain_sha256": sha256_bytes(b""),
        "predictor_source_path": "tests/table2/test_pillar134_companion_diagnostics.py",
        "predictor_source_sha256": identity.predictor_source_sha256,
        "factory_source_path": "tests/table2/test_pillar134_companion_diagnostics.py",
        "factory_source_sha256": identity.factory_source_sha256,
    }
    monkeypatch.setattr(
        "web_agent.eval.table2.companion_diagnostics.repository_attestation",
        lambda *args, **kwargs: dict(clean_receipt),
    )
    diagnostic = SimpleNamespace(
        backend_identity=identity,
        evidence_role=ENGINEERING_EVIDENCE_ROLE,
        registered_seed=42,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    blocker = (
        f"{CANONICAL_INPUT_PROVENANCE_STATUS}: fixture source replay is absent"
    )

    role, promotion, receipt, _, _ = run_attestation(
        diagnostic=diagnostic,
        predictor=predictor,
        canonical_error=blocker,
        canonical_role=P1_ROLE,
        repository_root=tmp_path,
        allow_uncommitted_engineering=False,
    )
    assert role == ENGINEERING_EVIDENCE_ROLE
    assert promotion == INPUT_PROVENANCE_BLOCKED_PROMOTION
    assert receipt == clean_receipt

    with pytest.raises(
        CompanionDiagnosticError, match="require clean repository verification"
    ):
        run_attestation(
            diagnostic=diagnostic,
            predictor=predictor,
            canonical_error=blocker,
            canonical_role=P1_ROLE,
            repository_root=None,
            allow_uncommitted_engineering=True,
        )


class _FakeBatchBuilder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> dict[str, Any]:
        import torch

        self.calls.append(kwargs)
        return {"input_ids": torch.tensor([[len(self.calls)]])}


class _FakePC01Runtime:
    def __init__(self, identity: CompanionBackendIdentity, *, action: str = "CLICK") -> None:
        import torch

        self.torch = torch
        self.checkpoint_sha256 = identity.checkpoint_sha256
        self.processor_contract = SimpleNamespace(
            record_sha256=identity.processor_contract_sha256
        )
        self.batch = _FakeBatchBuilder()
        self.lock = threading.Lock()
        self.model = self
        self.action = action

    def _assert_eval(self) -> None:
        return None

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        del batch
        torch = self.torch
        return {
            "outcome": torch.tensor([[0.2, 0.8]]).log(),
            "failure_type": torch.tensor([[0.1, 0.1, 0.7, 0.1]]).log(),
            "recovery": torch.tensor([[0.05, 0.65, 0.1, 0.05, 0.1, 0.05]]).log(),
            "needs_recovery": torch.tensor([[1.0]]),
            "recovery_outcome": torch.tensor([[1.25]]),
        }

    def predict_action(self, task: Any, observation: Any, rng: Any) -> PreActionDecision:
        del task, rng
        probabilities = _distribution(ACTION_LABELS, self.action)
        return PreActionDecision(
            decision_id=f"decision-{self.action.lower()}",
            observation_id=observation.observation_id,
            action_type=ActionType(self.action),
            action_probabilities=probabilities,
            bbox=(0.1, 0.2, 0.2, 0.2),
            grounding_confidence=0.9,
            confidence_before=0.8,
            input_observation_ids=(observation.observation_id,),
            policy_id="pc01-fixture-policy",
            policy_version="v1",
            parameter_hints={},
        )


def test_exact_pc01_p1_adapter_uses_action_conditioned_post_and_executed_recovery_streams(
    tmp_path: Path,
) -> None:
    diagnostic, image_root = _p1_input(tmp_path)
    identity = _canonical_identity("P1")
    runtime = _FakePC01Runtime(identity)
    predictor = PC01Pillar1DiagnosticPredictor(runtime=runtime, identity=identity)
    example = diagnostic.examples[0]
    transition = P1TransitionView(
        example.example_id,
        example.task_id,
        example.pre_image.resolve(image_root),
        example.post_image.resolve(image_root),
        example.text_state,
        example.executed_action,
    )
    prediction = predictor.predict_transition(transition, seed=42)
    assert prediction.failure_probability == pytest.approx(0.8)
    assert [call["phase"] for call in runtime.batch.calls] == ["pre", "post"]
    assert runtime.batch.calls[-1]["executed_action"] == "CLICK"
    assert runtime.batch.calls[-1]["action_value"] == ""

    recovery = example.recovery
    assert recovery is not None
    view = P1RecoveryView(
        example.example_id,
        example.task_id,
        recovery.attempt_id,
        recovery.pre_recovery_image.resolve(image_root),
        recovery.post_recovery_image.resolve(image_root),
        example.text_state,
        recovery.strategy,
        recovery.executed_recovery_actions,
    )
    assessment = predictor.predict_recovery(view, seed=42)
    assert assessment.resolution_probability == assessment.progress_probability
    assert [call["phase"] for call in runtime.batch.calls[-3:]] == [
        "pre", "post", "recovery"
    ]
    assert runtime.batch.calls[-1]["executed_action"] == "CLICK"


def _rejecting_fallback(task: Any, observation: Any, decision: Any, rng: Any) -> dict[str, Any]:
    del task, observation, decision, rng
    raise ParameterResolutionError("fixture frozen base rejection")


def _canonical_provider() -> Any:
    prompt = Path("configs/eval/table2/prompts/parameter_provider_v1.txt").read_bytes()
    return build_registered_hybrid_parameter_provider(
        fallback_resolver=_rejecting_fallback,
        fallback_policy_id="strict-e0-json-action-parser",
        fallback_policy_version="v1",
        prompt_bytes=prompt,
        decoding_parameters={
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 128,
        },
    )


def test_exact_pc01_p3_adapter_preserves_provider_trace_and_one_rejected_request(
    tmp_path: Path,
) -> None:
    diagnostic, image_root = _p3_input(tmp_path)
    identity = _canonical_identity("P3")
    provider = _canonical_provider()
    click_runtime = _FakePC01Runtime(identity, action="CLICK")
    predictor = PC01Pillar3DiagnosticPredictor(
        runtime=click_runtime, parameter_provider=provider, identity=identity
    )
    example = diagnostic.examples[0]
    view = P3DiagnosticView(
        example.example_id,
        example.task_id,
        example.image.resolve(image_root),
        example.text_state,
    )
    resolved = predictor.predict(view, seed=42)
    assert resolved.status == "RESOLVED"
    assert [attempt.source for attempt in resolved.provider_attempts] == ["deterministic"]

    type_runtime = _FakePC01Runtime(identity, action="TYPE")
    rejected_predictor = PC01Pillar3DiagnosticPredictor(
        runtime=type_runtime, parameter_provider=_canonical_provider(), identity=identity
    )
    rejected = rejected_predictor.predict(view, seed=42)
    assert rejected.status == "PARAMETER_REJECTED"
    assert rejected.accounted_rejected_executor_requests == 1
    assert [attempt.source for attempt in rejected.provider_attempts] == [
        "deterministic", "frozen_base_fallback"
    ]
    assert all(attempt.status == "REJECTED" for attempt in rejected.provider_attempts)


def test_companion_clis_have_no_project_import_before_standard_library_git_gate() -> None:
    bootstrap = Path("scripts/table2_companion_diagnostic_bootstrap.py")
    tree = ast.parse(bootstrap.read_text(encoding="utf-8"))
    imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not (
            isinstance(node, ast.ImportFrom) and (node.module or "").startswith("web_agent")
        )
        and not (
            isinstance(node, ast.Import)
            and any(alias.name.startswith("web_agent") for alias in node.names)
        )
        for node in imports
    )
    source = bootstrap.read_text(encoding="utf-8")
    assert source.index("starting_git = _verify_clean_repository(") < source.index(
        "diagnostics = importlib.import_module("
    ) < source.index("factory = _load_factory(")
    assert source.index("report = getattr(diagnostics, registration[\"run\"])(") < source.index(
        "ending_git = _verify_clean_repository("
    )


def test_registry_contains_schemas_and_no_inputs_results_or_locked_partition() -> None:
    registry = json.loads(
        Path("configs/eval/table2/companion_diagnostics_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert registry["artifact_role"] == "SCHEMA_ONLY_NO_INPUTS_NO_RESULTS"
    assert registry["paper_table_status"] == "N/R"
    assert registry["affects_primary_table2"] is False
    assert registry["locked_test_partitions_allowed"] is False
    assert "completed_frozen_campaign" not in registry["registered_source_partitions"]
    assert set(registry["pillars"]) == {"P1", "P3", "P4"}


def test_companion_preimport_gate_requires_exact_clean_commit_and_committed_source(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "factory.py"
    source.write_text("def create():\n    return None\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "diagnostic@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Diagnostic Test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", "factory.py"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"], cwd=repository, check=True
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    assert _verify_clean_repository(repository, expected_commit=commit)[
        "repository_commit"
    ] == commit
    assert _verify_committed_source(
        repository, source, expected_commit=commit
    ) == "factory.py"
    with pytest.raises(RuntimeError, match="exact clean"):
        _verify_clean_repository(repository, expected_commit="0" * 40)
    source.write_text("def create():\n    return 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact clean"):
        _verify_clean_repository(repository, expected_commit=commit)
    with pytest.raises(RuntimeError, match="differs"):
        _verify_committed_source(repository, source, expected_commit=commit)


def test_report_tampering_cannot_invent_canonical_status_or_metric_evidence(tmp_path: Path) -> None:
    diagnostic, root = _p1_input(tmp_path)
    report = run_pillar1_diagnostics(
        diagnostic,
        evidence_root=root,
        predictor=DeterministicP1Predictor(diagnostic.backend_identity),
        allow_uncommitted_engineering=True,
    )
    promoted = deepcopy(report)
    promoted["evidence_role"] = P1_ROLE
    promoted["promotion_status"] = "CANONICAL_PC01_COMPANION_EVIDENCE"
    with pytest.raises(SchemaError, match="not canonical"):
        validate_pillar1_diagnostic_report(promoted)
    fabricated = deepcopy(report)
    fabricated["summary"]["overall"]["failure_detection_accuracy"]["numerator"] = 1
    with pytest.raises(SchemaError, match="not reproducible"):
        validate_pillar1_diagnostic_report(fabricated)
