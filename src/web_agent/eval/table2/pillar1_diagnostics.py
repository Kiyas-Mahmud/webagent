"""Offline Pillar-1 detection, diagnosis, and recovery-assessment diagnostics.

This harness scores frozen post-action and executed-recovery evidence.  It does
not execute a recovery, emit a trigger, read a verifier during inference, or
alter the primary E0--E3 campaign.  Oracle labels live only in runner-owned
targets and are structurally absent from predictor views.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from web_agent.labels import ACTION_TYPE, FAILURE_TYPE, RECOVERY_STRATEGY

from .common import SchemaError, sha256_json, stable_int_seed
from .companion_diagnostics import (
    ENGINEERING_EVIDENCE_ROLE,
    CANONICAL_INPUT_PROVENANCE_STATUS,
    PAPER_TABLE_STATUS,
    CompanionBackendIdentity,
    CompanionDiagnosticError,
    ImageArtifact,
    ObservableTextState,
    ResolvedImage,
    canonical_pc01_base_error,
    load_unique_json,
    metric,
    probability_distribution,
    rehash_images,
    repository_attestation,
    require_bool,
    require_exact_keys,
    require_sha256,
    require_text,
    run_attestation,
    selected_label,
    validate_input_envelope,
    validate_metric_tree,
    validate_package,
    validate_report_envelope,
    verify_predictor,
    write_package,
)


PILLAR = "P1"
INPUT_SCHEMA = "table2.pillar1-diagnostic-input.v1"
REPORT_SCHEMA = "table2.pillar1-diagnostic-report.v1"
EVIDENCE_ROLE = "P1_COMPANION_MECHANISM_EVIDENCE"
INPUT_FILENAME = "pillar1_diagnostic_input.json"
REPORT_FILENAME = "pillar1_diagnostic_report.json"
CLAIM_SCOPE = {
    "failure_detection": "OFFLINE_POST_ACTION_COMPONENT_QUALITY",
    "failure_diagnosis": "OFFLINE_POST_ACTION_COMPONENT_QUALITY",
    "recovery_assessment": "OFFLINE_EXECUTED_TRANSITION_COMPONENT_QUALITY",
    "operational_recovery": "NOT_MEASURED",
    "learned_trigger_effect": "NOT_ISOLATED_BY_THIS_PACKAGE",
    "table2_row_or_contrast": "NONE",
}


@dataclass(frozen=True, slots=True)
class P1Targets:
    failure: bool
    failure_type: str
    needs_recovery: bool
    recovery_strategy: str

    def __post_init__(self) -> None:
        require_bool(self.failure, context="P1 target.failure")
        require_bool(self.needs_recovery, context="P1 target.needs_recovery")
        if self.failure_type not in FAILURE_TYPE:
            raise SchemaError("P1 target failure_type is unregistered")
        if self.recovery_strategy not in RECOVERY_STRATEGY:
            raise SchemaError("P1 target recovery_strategy is unregistered")
        if (self.failure_type == "NONE") is not (not self.failure):
            raise SchemaError("P1 failure target and failure_type are inconsistent")
        if (self.recovery_strategy == "NONE") is not (not self.needs_recovery):
            raise SchemaError("P1 recovery target and strategy are inconsistent")
        if self.needs_recovery and not self.failure:
            raise SchemaError("P1 success target cannot require recovery")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "P1Targets":
        require_exact_keys(value, set(cls.__dataclass_fields__), context="P1 targets")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class P1RecoveryEvidence:
    attempt_id: str
    strategy: str
    pre_recovery_image: ImageArtifact
    post_recovery_image: ImageArtifact
    executed_recovery_actions: tuple[str, ...]
    failure_resolved: bool
    progress: bool
    source_record_sha256: str

    def __post_init__(self) -> None:
        require_text(self.attempt_id, context="P1 recovery attempt_id")
        if self.strategy not in RECOVERY_STRATEGY or self.strategy in {
            "NONE",
            "ABORT",
        }:
            raise SchemaError("P1 recovery evidence requires an executed strategy")
        if (
            type(self.executed_recovery_actions) is not tuple
            or not self.executed_recovery_actions
            or any(
                action not in ACTION_TYPE
                for action in self.executed_recovery_actions
            )
        ):
            raise SchemaError("P1 recovery evidence requires registered executed actions")
        require_bool(self.failure_resolved, context="P1 recovery.failure_resolved")
        require_bool(self.progress, context="P1 recovery.progress")
        require_sha256(self.source_record_sha256, context="P1 recovery source record")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "P1RecoveryEvidence":
        require_exact_keys(value, set(cls.__dataclass_fields__), context="P1 recovery evidence")
        mapped = dict(value)
        mapped["pre_recovery_image"] = ImageArtifact.from_mapping(
            mapped["pre_recovery_image"]
        )
        mapped["post_recovery_image"] = ImageArtifact.from_mapping(
            mapped["post_recovery_image"]
        )
        mapped["executed_recovery_actions"] = tuple(mapped["executed_recovery_actions"])
        return cls(**mapped)


@dataclass(frozen=True, slots=True)
class P1Example:
    example_id: str
    task_id: str
    pre_image: ImageArtifact
    post_image: ImageArtifact
    text_state: ObservableTextState
    executed_action: str
    trigger_sources: tuple[str, ...]
    targets: P1Targets
    recovery: P1RecoveryEvidence | None
    source_record_sha256: str

    def __post_init__(self) -> None:
        require_text(self.example_id, context="P1 example_id")
        require_text(self.task_id, context="P1 task_id")
        if self.executed_action not in ACTION_TYPE:
            raise SchemaError("P1 executed action is unregistered")
        if type(self.trigger_sources) is not tuple or set(self.trigger_sources) - {
            "policy",
            "executor",
            "loop_guard",
        }:
            raise SchemaError("P1 trigger sources are oracle-derived or unregistered")
        if len(self.trigger_sources) != len(set(self.trigger_sources)):
            raise SchemaError("P1 trigger sources contain duplicates")
        require_sha256(self.source_record_sha256, context="P1 source record")
        if self.recovery is not None and not self.targets.needs_recovery:
            raise SchemaError("P1 recovery evidence contradicts no-recovery target")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "P1Example":
        require_exact_keys(value, set(cls.__dataclass_fields__), context="P1 example")
        mapped = dict(value)
        mapped["pre_image"] = ImageArtifact.from_mapping(mapped["pre_image"])
        mapped["post_image"] = ImageArtifact.from_mapping(mapped["post_image"])
        mapped["text_state"] = ObservableTextState.from_mapping(mapped["text_state"])
        mapped["trigger_sources"] = tuple(mapped["trigger_sources"])
        mapped["targets"] = P1Targets.from_mapping(mapped["targets"])
        mapped["recovery"] = (
            None
            if mapped["recovery"] is None
            else P1RecoveryEvidence.from_mapping(mapped["recovery"])
        )
        return cls(**mapped)


@dataclass(frozen=True, slots=True)
class Pillar1DiagnosticInput:
    diagnostic_id: str
    campaign_id: str
    source_partition: str
    registered_seed: int
    bootstrap_samples: int
    bootstrap_confidence: float
    bootstrap_seed: int
    examples: tuple[P1Example, ...]
    backend_identity: CompanionBackendIdentity
    actions_frozen_before_diagnostics: bool
    labels_attached_after_execution: bool
    oracle_labels_visible_to_predictor: bool
    runtime_outputs_consumed: bool
    affects_primary_table2: bool
    locked_test_rows_read: int
    evidence_role: str = ENGINEERING_EVIDENCE_ROLE
    paper_table_status: str = PAPER_TABLE_STATUS
    schema_version: str = INPUT_SCHEMA

    def __post_init__(self) -> None:
        require_text(self.diagnostic_id, context="P1 diagnostic_id")
        require_text(self.campaign_id, context="P1 campaign_id")
        if type(self.examples) is not tuple or not self.examples or any(
            type(example) is not P1Example for example in self.examples
        ):
            raise SchemaError("P1 diagnostic requires frozen examples")
        if type(self.backend_identity) is not CompanionBackendIdentity:
            raise SchemaError("P1 diagnostic backend identity is invalid")
        ids = [example.example_id for example in self.examples]
        if len(ids) != len(set(ids)):
            raise SchemaError("P1 diagnostic example IDs must be unique")
        if self.schema_version != INPUT_SCHEMA:
            raise SchemaError("unregistered P1 diagnostic input schema")
        if self.labels_attached_after_execution is not True:
            raise SchemaError("P1 labels must be attached after action execution")
        if self.oracle_labels_visible_to_predictor is not False:
            raise SchemaError("P1 predictor cannot receive oracle labels")
        validate_input_envelope(
            pillar=PILLAR,
            source_partition=self.source_partition,
            registered_seed=self.registered_seed,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_confidence=self.bootstrap_confidence,
            bootstrap_seed=self.bootstrap_seed,
            actions_frozen_before_diagnostics=self.actions_frozen_before_diagnostics,
            runtime_outputs_consumed=self.runtime_outputs_consumed,
            affects_primary_table2=self.affects_primary_table2,
            locked_test_rows_read=self.locked_test_rows_read,
            evidence_role=self.evidence_role,
            canonical_role=EVIDENCE_ROLE,
            paper_table_status=self.paper_table_status,
            backend_identity=self.backend_identity,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Pillar1DiagnosticInput":
        require_exact_keys(value, set(cls.__dataclass_fields__), context="P1 diagnostic input")
        mapped = dict(value)
        mapped["examples"] = tuple(P1Example.from_mapping(item) for item in mapped["examples"])
        mapped["backend_identity"] = CompanionBackendIdentity.from_mapping(
            mapped["backend_identity"]
        )
        return cls(**mapped)

    @classmethod
    def load(cls, path: str | Path) -> "Pillar1DiagnosticInput":
        return cls.from_mapping(load_unique_json(path))


@dataclass(frozen=True, slots=True)
class P1TransitionView:
    example_id: str
    task_id: str
    pre_image: ResolvedImage
    post_image: ResolvedImage
    text_state: ObservableTextState
    executed_action: str


@dataclass(frozen=True, slots=True)
class P1RecoveryView:
    example_id: str
    task_id: str
    attempt_id: str
    pre_recovery_image: ResolvedImage
    post_recovery_image: ResolvedImage
    text_state: ObservableTextState
    strategy: str
    executed_recovery_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class P1TransitionPrediction:
    failure_probability: float
    failure_type_probabilities: Mapping[str, float]
    needs_recovery_probability: float
    recovery_probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        from .companion_diagnostics import require_probability

        require_probability(self.failure_probability, context="P1 failure probability")
        require_probability(
            self.needs_recovery_probability, context="P1 needs-recovery probability"
        )
        probability_distribution(
            self.failure_type_probabilities,
            labels=tuple(FAILURE_TYPE),
            context="P1 failure-type probabilities",
        )
        probability_distribution(
            self.recovery_probabilities,
            labels=tuple(RECOVERY_STRATEGY),
            context="P1 recovery probabilities",
        )


@dataclass(frozen=True, slots=True)
class P1RecoveryPrediction:
    resolution_probability: float
    progress_probability: float

    def __post_init__(self) -> None:
        from .companion_diagnostics import require_probability

        require_probability(self.resolution_probability, context="P1 resolution probability")
        require_probability(self.progress_probability, context="P1 progress probability")


class Pillar1DiagnosticPredictor(Protocol):
    @property
    def diagnostic_identity(self) -> CompanionBackendIdentity: ...

    def predict_transition(
        self, view: P1TransitionView, *, seed: int
    ) -> P1TransitionPrediction: ...

    def predict_recovery(
        self, view: P1RecoveryView, *, seed: int
    ) -> P1RecoveryPrediction: ...


def canonical_p1_backend_error(
    identity: CompanionBackendIdentity,
    *,
    predictor: Any | None = None,
    require_input_provenance: bool = True,
) -> str | None:
    from .pc01_companion_diagnostics import (
        PC01_P1_PREDICTOR_ID,
        PC01_P1_PREDICTOR_QUALNAME,
        PC01_P1_PREDICTOR_VERSION,
        PC01Pillar1DiagnosticPredictor,
        pc01_companion_predictor_source_sha256,
    )

    error = canonical_pc01_base_error(identity)
    if error is not None:
        return error
    expected = {
        "pillar": PILLAR,
        "predictor_id": PC01_P1_PREDICTOR_ID,
        "predictor_version": PC01_P1_PREDICTOR_VERSION,
        "predictor_module": PC01Pillar1DiagnosticPredictor.__module__,
        "predictor_qualname": PC01_P1_PREDICTOR_QUALNAME,
        "predictor_source_sha256": pc01_companion_predictor_source_sha256(),
    }
    for field, expected_value in expected.items():
        if getattr(identity, field) != expected_value:
            return f"canonical PC-01 P1 backend changed {field}"
    if predictor is not None and type(predictor) is not PC01Pillar1DiagnosticPredictor:
        return "canonical PC-01 P1 evidence requires the exact predictor class"
    if require_input_provenance:
        return (
            f"{CANONICAL_INPUT_PROVENANCE_STATUS}: P1 requires replay of an "
            "authoritative frozen source manifest, access ledger, and source-record bytes"
        )
    return None


def _transition_record(
    example: P1Example,
    prediction: P1TransitionPrediction,
) -> dict[str, Any]:
    failure_type = selected_label(
        prediction.failure_type_probabilities, tuple(FAILURE_TYPE)
    )
    strategy = selected_label(
        prediction.recovery_probabilities, tuple(RECOVERY_STRATEGY)
    )
    return {
        "record_type": "transition",
        "example_id": example.example_id,
        "task_id": example.task_id,
        "source_record_sha256": example.source_record_sha256,
        "trigger_sources": list(example.trigger_sources),
        "target": asdict(example.targets),
        "prediction": {
            **asdict(prediction),
            "predicted_failure": prediction.failure_probability >= 0.5,
            "failure_type": failure_type,
            "predicted_needs_recovery": prediction.needs_recovery_probability >= 0.5,
            "recovery_strategy": strategy,
        },
        "prediction_sha256": sha256_json(asdict(prediction)),
    }


def _recovery_record(
    example: P1Example,
    recovery: P1RecoveryEvidence,
    prediction: P1RecoveryPrediction,
) -> dict[str, Any]:
    return {
        "record_type": "recovery",
        "example_id": example.example_id,
        "task_id": example.task_id,
        "attempt_id": recovery.attempt_id,
        "source_record_sha256": recovery.source_record_sha256,
        "strategy": recovery.strategy,
        "target": {
            "failure_resolved": recovery.failure_resolved,
            "progress": recovery.progress,
        },
        "prediction": {
            **asdict(prediction),
            "predicted_failure_resolved": prediction.resolution_probability >= 0.5,
            "predicted_progress": prediction.progress_probability >= 0.5,
        },
        "prediction_sha256": sha256_json(asdict(prediction)),
    }


def _summary(
    records: list[dict[str, Any]], *, bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, Any]:
    transitions = [row for row in records if row["record_type"] == "transition"]
    recoveries = [row for row in records if row["record_type"] == "recovery"]

    def stat(name: str, rows, kind: str, function):
        return metric(
            rows,
            name=f"P1.{name}",
            metric_kind=kind,
            value=function,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            contribution_unit="frozen_post_action_or_recovery_example",
        )

    overall = {
        "failure_detection_accuracy": stat(
            "failure_accuracy",
            transitions,
            "rate",
            lambda row: float(
                row["prediction"]["predicted_failure"] == row["target"]["failure"]
            ),
        ),
        "failure_precision": stat(
            "failure_precision",
            transitions,
            "rate",
            lambda row: (
                float(row["target"]["failure"])
                if row["prediction"]["predicted_failure"]
                else None
            ),
        ),
        "failure_recall": stat(
            "failure_recall",
            transitions,
            "rate",
            lambda row: (
                float(row["prediction"]["predicted_failure"])
                if row["target"]["failure"]
                else None
            ),
        ),
        "failure_probability_brier": stat(
            "failure_brier",
            transitions,
            "mean",
            lambda row: (
                float(row["prediction"]["failure_probability"])
                - float(row["target"]["failure"])
            )
            ** 2,
        ),
        "failure_type_accuracy_on_failures": stat(
            "failure_type_accuracy",
            transitions,
            "rate",
            lambda row: (
                float(row["prediction"]["failure_type"] == row["target"]["failure_type"])
                if row["target"]["failure"]
                else None
            ),
        ),
        "needs_recovery_accuracy": stat(
            "needs_recovery_accuracy",
            transitions,
            "rate",
            lambda row: float(
                row["prediction"]["predicted_needs_recovery"]
                == row["target"]["needs_recovery"]
            ),
        ),
        "needs_recovery_recall": stat(
            "needs_recovery_recall",
            transitions,
            "rate",
            lambda row: (
                float(row["prediction"]["predicted_needs_recovery"])
                if row["target"]["needs_recovery"]
                else None
            ),
        ),
        "needs_recovery_probability_brier": stat(
            "needs_recovery_brier",
            transitions,
            "mean",
            lambda row: (
                float(row["prediction"]["needs_recovery_probability"])
                - float(row["target"]["needs_recovery"])
            )
            ** 2,
        ),
        "strategy_hit_at_1": stat(
            "strategy_hit1",
            transitions,
            "rate",
            lambda row: (
                float(
                    row["prediction"]["recovery_strategy"]
                    == row["target"]["recovery_strategy"]
                )
                if row["target"]["needs_recovery"]
                else None
            ),
        ),
        "recovery_resolution_accuracy": stat(
            "recovery_resolution_accuracy",
            recoveries,
            "rate",
            lambda row: float(
                row["prediction"]["predicted_failure_resolved"]
                == row["target"]["failure_resolved"]
            ),
        ),
        "recovery_resolution_probability_brier": stat(
            "recovery_resolution_brier",
            recoveries,
            "mean",
            lambda row: (
                float(row["prediction"]["resolution_probability"])
                - float(row["target"]["failure_resolved"])
            )
            ** 2,
        ),
        "recovery_progress_accuracy": stat(
            "recovery_progress_accuracy",
            recoveries,
            "rate",
            lambda row: float(
                row["prediction"]["predicted_progress"] == row["target"]["progress"]
            ),
        ),
    }
    failure_type = {}
    for label in sorted({row["target"]["failure_type"] for row in transitions}):
        selected = [row for row in transitions if row["target"]["failure_type"] == label]
        failure_type[label] = {
            "example_count": len(selected),
            "detection_accuracy": stat(
                f"failure_type.{label}.detection", selected, "rate", lambda row: float(
                    row["prediction"]["predicted_failure"] == row["target"]["failure"]
                )
            ),
            "diagnosis_accuracy": stat(
                f"failure_type.{label}.diagnosis",
                selected,
                "rate",
                lambda row: float(
                    row["prediction"]["failure_type"] == row["target"]["failure_type"]
                ),
            ),
        }
    strategy = {}
    for label in sorted({row["strategy"] for row in recoveries}):
        selected = [row for row in recoveries if row["strategy"] == label]
        strategy[label] = {
            "example_count": len(selected),
            "resolution_accuracy": stat(
                f"strategy.{label}.resolution",
                selected,
                "rate",
                lambda row: float(
                    row["prediction"]["predicted_failure_resolved"]
                    == row["target"]["failure_resolved"]
                ),
            ),
        }
    trigger_source = {}
    sources = sorted({source for row in transitions for source in row["trigger_sources"]})
    for source in sources:
        selected = [row for row in transitions if source in row["trigger_sources"]]
        trigger_source[source] = {
            "example_count": len(selected),
            "needs_recovery_accuracy": stat(
                f"trigger.{source}.needs_recovery",
                selected,
                "rate",
                lambda row: float(
                    row["prediction"]["predicted_needs_recovery"]
                    == row["target"]["needs_recovery"]
                ),
            ),
        }
    return {
        "transition_example_count": len(transitions),
        "recovery_example_count": len(recoveries),
        "overall": overall,
        "by_failure_type": failure_type,
        "by_recovery_strategy": strategy,
        "by_trigger_source": trigger_source,
    }


def run_pillar1_diagnostics(
    diagnostic: Pillar1DiagnosticInput,
    *,
    evidence_root: str | Path,
    predictor: Pillar1DiagnosticPredictor,
    repository_root: str | Path | None = None,
    allow_uncommitted_engineering: bool = False,
) -> dict[str, Any]:
    if diagnostic.evidence_role == EVIDENCE_ROLE and repository_root is None:
        raise CompanionDiagnosticError(
            "canonical P1 evidence requires clean repository verification"
        )
    pre_run_attestation = None
    if repository_root is not None:
        pre_source, _ = verify_predictor(predictor, diagnostic.backend_identity)
        pre_run_attestation = repository_attestation(
            repository_root,
            expected_commit=diagnostic.backend_identity.repository_commit,
            source_path=pre_source,
            factory_entrypoint=diagnostic.backend_identity.factory_entrypoint,
            factory_source_sha256=diagnostic.backend_identity.factory_source_sha256,
        )
    canonical_error = canonical_p1_backend_error(
        diagnostic.backend_identity, predictor=predictor
    )
    role, promotion, source_attestation, source_path, source_hash = run_attestation(
        diagnostic=diagnostic,
        predictor=predictor,
        canonical_error=canonical_error,
        canonical_role=EVIDENCE_ROLE,
        repository_root=repository_root,
        allow_uncommitted_engineering=allow_uncommitted_engineering,
    )
    if pre_run_attestation is not None and source_attestation != pre_run_attestation:
        raise CompanionDiagnosticError("P1 repository identity changed before inference")
    raw_evidence_root = Path(evidence_root).absolute()
    if raw_evidence_root.is_symlink() or not raw_evidence_root.is_dir():
        raise CompanionDiagnosticError(
            "P1 evidence root must be a non-symlink directory"
        )
    resolved_evidence_root = raw_evidence_root.resolve()
    records: list[dict[str, Any]] = []
    resolved_images: list[ResolvedImage] = []
    for example in diagnostic.examples:
        pre = example.pre_image.resolve(resolved_evidence_root)
        post = example.post_image.resolve(resolved_evidence_root)
        resolved_images.extend((pre, post))
        transition_view = P1TransitionView(
            example.example_id,
            example.task_id,
            pre,
            post,
            example.text_state,
            example.executed_action,
        )
        before = sha256_json(asdict(transition_view))
        prediction = predictor.predict_transition(
            transition_view,
            seed=stable_int_seed(
                diagnostic.registered_seed,
                diagnostic.diagnostic_id,
                example.example_id,
                "P1.transition",
            ),
        )
        if type(prediction) is not P1TransitionPrediction:
            raise CompanionDiagnosticError("P1 predictor returned an invalid transition contract")
        if sha256_json(asdict(transition_view)) != before:
            raise CompanionDiagnosticError("P1 predictor mutated its causal transition input")
        records.append(_transition_record(example, prediction))
        if example.recovery is not None:
            recovery = example.recovery
            recovery_pre = recovery.pre_recovery_image.resolve(resolved_evidence_root)
            recovery_post = recovery.post_recovery_image.resolve(resolved_evidence_root)
            resolved_images.extend((recovery_pre, recovery_post))
            recovery_view = P1RecoveryView(
                example.example_id,
                example.task_id,
                recovery.attempt_id,
                recovery_pre,
                recovery_post,
                example.text_state,
                recovery.strategy,
                recovery.executed_recovery_actions,
            )
            before = sha256_json(asdict(recovery_view))
            recovery_prediction = predictor.predict_recovery(
                recovery_view,
                seed=stable_int_seed(
                    diagnostic.registered_seed,
                    diagnostic.diagnostic_id,
                    example.example_id,
                    recovery.attempt_id,
                    "P1.recovery",
                ),
            )
            if type(recovery_prediction) is not P1RecoveryPrediction:
                raise CompanionDiagnosticError("P1 predictor returned an invalid recovery contract")
            if sha256_json(asdict(recovery_view)) != before:
                raise CompanionDiagnosticError("P1 predictor mutated its recovery input")
            records.append(_recovery_record(example, recovery, recovery_prediction))
    rehash_images(resolved_images)
    if source_hash != verify_predictor(predictor, diagnostic.backend_identity)[1]:
        raise CompanionDiagnosticError("P1 predictor source changed during inference")
    if repository_root is not None:
        ending = repository_attestation(
            repository_root,
            expected_commit=diagnostic.backend_identity.repository_commit,
            source_path=source_path,
            factory_entrypoint=diagnostic.backend_identity.factory_entrypoint,
            factory_source_sha256=diagnostic.backend_identity.factory_source_sha256,
        )
        if ending != source_attestation:
            raise CompanionDiagnosticError("P1 repository identity changed during inference")
    summary = _summary(
        records,
        bootstrap_samples=diagnostic.bootstrap_samples,
        bootstrap_seed=diagnostic.bootstrap_seed,
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "pillar": PILLAR,
        "diagnostic_id": diagnostic.diagnostic_id,
        "campaign_id": diagnostic.campaign_id,
        "source_partition": diagnostic.source_partition,
        "evidence_role": role,
        "promotion_status": promotion,
        "paper_table_status": PAPER_TABLE_STATUS,
        "claim_scope": dict(CLAIM_SCOPE),
        "affects_primary_table2": False,
        "input_manifest_sha256": sha256_json(asdict(diagnostic)),
        "backend_identity": diagnostic.backend_identity.to_dict(),
        "source_attestation": source_attestation,
        "statistics": {
            "bootstrap_samples": diagnostic.bootstrap_samples,
            "bootstrap_confidence": diagnostic.bootstrap_confidence,
            "bootstrap_seed": diagnostic.bootstrap_seed,
            "cluster_unit": "task_id",
        },
        "records": records,
        "summary": summary,
        "browser_actions_executed": 0,
        "recovery_triggers_emitted": 0,
        "memory_queries_executed": 0,
        "memory_writes_executed": 0,
        "locked_test_rows_read": 0,
    }
    validate_pillar1_diagnostic_report(report)
    return report


def validate_pillar1_diagnostic_report(report: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "pillar",
        "diagnostic_id",
        "campaign_id",
        "source_partition",
        "evidence_role",
        "promotion_status",
        "paper_table_status",
        "claim_scope",
        "affects_primary_table2",
        "input_manifest_sha256",
        "backend_identity",
        "source_attestation",
        "statistics",
        "records",
        "summary",
        "browser_actions_executed",
        "recovery_triggers_emitted",
        "memory_queries_executed",
        "memory_writes_executed",
        "locked_test_rows_read",
    }
    require_exact_keys(report, required, context="P1 diagnostic report")
    backend = CompanionBackendIdentity.from_mapping(report["backend_identity"])
    validate_report_envelope(
        report,
        pillar=PILLAR,
        report_schema=REPORT_SCHEMA,
        canonical_role=EVIDENCE_ROLE,
        canonical_error=canonical_p1_backend_error(backend),
        backend_identity=backend,
    )
    if (
        report["claim_scope"] != CLAIM_SCOPE
        or type(report["locked_test_rows_read"]) is not int
        or report["locked_test_rows_read"] != 0
    ):
        raise SchemaError("P1 report scientific boundary changed")
    if (
        type(report["recovery_triggers_emitted"]) is not int
        or report["recovery_triggers_emitted"] != 0
        or type(report["memory_queries_executed"]) is not int
        or report["memory_queries_executed"] != 0
    ):
        raise SchemaError("P1 offline report claims a runtime side effect")
    statistics = report["statistics"]
    if not isinstance(statistics, Mapping) or set(statistics) != {
        "bootstrap_samples",
        "bootstrap_confidence",
        "bootstrap_seed",
        "cluster_unit",
    }:
        raise SchemaError("P1 report statistics registration changed")
    if (
        statistics["bootstrap_confidence"] != 0.95
        or statistics["cluster_unit"] != "task_id"
        or type(statistics["bootstrap_samples"]) is not int
        or statistics["bootstrap_samples"] <= 0
        or type(statistics["bootstrap_seed"]) is not int
        or statistics["bootstrap_seed"] < 0
    ):
        raise SchemaError("P1 report statistics registration is invalid")
    records = report["records"]
    if not isinstance(records, list) or not records:
        raise SchemaError("P1 report requires records")
    transition_fields = {
        "record_type", "example_id", "task_id", "source_record_sha256",
        "trigger_sources", "target", "prediction", "prediction_sha256",
    }
    recovery_fields = {
        "record_type", "example_id", "task_id", "attempt_id",
        "source_record_sha256", "strategy", "target", "prediction",
        "prediction_sha256",
    }
    for row in records:
        if not isinstance(row, Mapping):
            raise SchemaError("P1 report record must be an object")
        require_text(row.get("example_id"), context="P1 report example_id")
        require_text(row.get("task_id"), context="P1 report task_id")
        require_sha256(row.get("source_record_sha256"), context="P1 report source")
        prediction = row.get("prediction")
        if not isinstance(prediction, Mapping):
            raise SchemaError("P1 report prediction must be an object")
        if row.get("record_type") == "transition":
            require_exact_keys(row, transition_fields, context="P1 transition record")
            P1Targets.from_mapping(row["target"])
            expected_prediction_fields = {
                "failure_probability", "failure_type_probabilities",
                "needs_recovery_probability", "recovery_probabilities",
                "predicted_failure", "failure_type", "predicted_needs_recovery",
                "recovery_strategy",
            }
            require_exact_keys(
                prediction, expected_prediction_fields, context="P1 transition prediction"
            )
            base = P1TransitionPrediction(
                failure_probability=prediction["failure_probability"],
                failure_type_probabilities=prediction["failure_type_probabilities"],
                needs_recovery_probability=prediction["needs_recovery_probability"],
                recovery_probabilities=prediction["recovery_probabilities"],
            )
            if (
                prediction["predicted_failure"] is not (base.failure_probability >= 0.5)
                or prediction["predicted_needs_recovery"]
                is not (base.needs_recovery_probability >= 0.5)
                or prediction["failure_type"] != selected_label(
                    base.failure_type_probabilities, tuple(FAILURE_TYPE)
                )
                or prediction["recovery_strategy"] != selected_label(
                    base.recovery_probabilities, tuple(RECOVERY_STRATEGY)
                )
            ):
                raise SchemaError("P1 transition derived prediction fields changed")
            base_payload = asdict(base)
        elif row.get("record_type") == "recovery":
            require_exact_keys(row, recovery_fields, context="P1 recovery record")
            if row["strategy"] not in RECOVERY_STRATEGY or row["strategy"] == "NONE":
                raise SchemaError("P1 recovery record strategy is invalid")
            require_exact_keys(
                row["target"], {"failure_resolved", "progress"}, context="P1 recovery target"
            )
            require_bool(row["target"]["failure_resolved"], context="P1 recovery target")
            require_bool(row["target"]["progress"], context="P1 recovery progress")
            expected_prediction_fields = {
                "resolution_probability", "progress_probability",
                "predicted_failure_resolved", "predicted_progress",
            }
            require_exact_keys(
                prediction, expected_prediction_fields, context="P1 recovery prediction"
            )
            base = P1RecoveryPrediction(
                resolution_probability=prediction["resolution_probability"],
                progress_probability=prediction["progress_probability"],
            )
            if (
                prediction["predicted_failure_resolved"]
                is not (base.resolution_probability >= 0.5)
                or prediction["predicted_progress"]
                is not (base.progress_probability >= 0.5)
            ):
                raise SchemaError("P1 recovery derived prediction fields changed")
            base_payload = asdict(base)
        else:
            raise SchemaError("P1 report record type is unregistered")
        if row["prediction_sha256"] != sha256_json(base_payload):
            raise SchemaError("P1 prediction hash changed")
    recomputed = _summary(
        records,
        bootstrap_samples=statistics["bootstrap_samples"],
        bootstrap_seed=statistics["bootstrap_seed"],
    )
    if report["summary"] != recomputed:
        raise SchemaError("P1 diagnostic summary is not reproducible from records")
    validate_metric_tree(report["summary"])


def write_pillar1_diagnostic_package(
    output_dir: str | Path,
    *,
    diagnostic: Pillar1DiagnosticInput,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    validate_pillar1_diagnostic_report(report)
    _validate_p1_report_input_binding(diagnostic, report)
    return write_package(
        output_dir,
        pillar=PILLAR,
        diagnostic=diagnostic,
        report=report,
        input_filename=INPUT_FILENAME,
        report_filename=REPORT_FILENAME,
    )


def validate_pillar1_diagnostic_package(output_dir: str | Path) -> dict[str, Any]:
    result = validate_package(
        output_dir,
        pillar=PILLAR,
        input_filename=INPUT_FILENAME,
        report_filename=REPORT_FILENAME,
        load_input=Pillar1DiagnosticInput.from_mapping,
        validate_report=validate_pillar1_diagnostic_report,
    )
    root = Path(output_dir)
    diagnostic = Pillar1DiagnosticInput.from_mapping(
        load_unique_json(root / INPUT_FILENAME)
    )
    _validate_p1_report_input_binding(diagnostic, load_unique_json(root / REPORT_FILENAME))
    return result


def _validate_p1_report_input_binding(
    diagnostic: Pillar1DiagnosticInput, report: Mapping[str, Any]
) -> None:
    transitions = {
        row["example_id"]: row for row in report["records"]
        if row["record_type"] == "transition"
    }
    recoveries = {
        row["example_id"]: row for row in report["records"]
        if row["record_type"] == "recovery"
    }
    if len(transitions) != len(diagnostic.examples) or set(transitions) != {
        row.example_id for row in diagnostic.examples
    }:
        raise CompanionDiagnosticError("P1 report/input transition inventory changed")
    expected_recovery_ids = {
        row.example_id for row in diagnostic.examples if row.recovery is not None
    }
    if len(recoveries) != len(expected_recovery_ids) or set(recoveries) != expected_recovery_ids:
        raise CompanionDiagnosticError("P1 report/input recovery inventory changed")
    for example in diagnostic.examples:
        transition = transitions[example.example_id]
        expected = {
            "task_id": example.task_id,
            "source_record_sha256": example.source_record_sha256,
            "trigger_sources": list(example.trigger_sources),
            "target": asdict(example.targets),
        }
        if any(transition[field] != value for field, value in expected.items()):
            raise CompanionDiagnosticError("P1 report/input transition evidence changed")
        if example.recovery is not None:
            recovery = recoveries[example.example_id]
            expected_recovery = {
                "task_id": example.task_id,
                "attempt_id": example.recovery.attempt_id,
                "source_record_sha256": example.recovery.source_record_sha256,
                "strategy": example.recovery.strategy,
                "target": {
                    "failure_resolved": example.recovery.failure_resolved,
                    "progress": example.recovery.progress,
                },
            }
            if any(recovery[field] != value for field, value in expected_recovery.items()):
                raise CompanionDiagnosticError("P1 report/input recovery evidence changed")
