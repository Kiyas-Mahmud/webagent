"""Offline Pillar-3 six-action, grounding, and parameter diagnostics.

Predictions use only a frozen pre-action observation.  Invalid policy or
parameter-provider output is preserved as one rejected executor request; the
harness has no repair loop and executes no browser action.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from web_agent.labels import ACTION_TYPE
from web_agent.runtime.action_parameters import validate_action_parameters
from web_agent.runtime.contracts import ActionType

from .common import SchemaError, canonical_json_bytes, sha256_json, stable_int_seed
from .companion_diagnostics import (
    ENGINEERING_EVIDENCE_ROLE,
    CANONICAL_INPUT_PROVENANCE_STATUS,
    PAPER_TABLE_STATUS,
    CompanionBackendIdentity,
    CompanionDiagnosticError,
    ImageArtifact,
    ObservableTextState,
    ResolvedImage,
    bbox_iou,
    canonical_pc01_base_error,
    json_mapping,
    load_unique_json,
    metric,
    normalized_bbox,
    probability_distribution,
    rehash_images,
    repository_attestation,
    require_exact_keys,
    require_probability,
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


PILLAR = "P3"
INPUT_SCHEMA = "table2.pillar3-diagnostic-input.v1"
REPORT_SCHEMA = "table2.pillar3-diagnostic-report.v1"
EVIDENCE_ROLE = "P3_COMPANION_MECHANISM_EVIDENCE"
INPUT_FILENAME = "pillar3_diagnostic_input.json"
REPORT_FILENAME = "pillar3_diagnostic_report.json"
CLAIM_SCOPE = {
    "action_class": "OFFLINE_SIX_CLASS_COMPONENT_QUALITY",
    "grounding": "OFFLINE_NORMALIZED_TARGET_DIAGNOSTIC",
    "parameter_resolution": "OFFLINE_PROVIDER_VALIDITY_AND_AGREEMENT",
    "browser_execution": "NOT_MEASURED",
    "silent_repair": "FORBIDDEN",
    "table2_row_or_contrast": "NONE",
}
GROUNDED_ACTIONS = frozenset({"CLICK", "TYPE", "SELECT"})
PREDICTION_STATUSES = frozenset(
    {"RESOLVED", "POLICY_REJECTED", "PARAMETER_REJECTED"}
)
PC01_PARAMETER_PROVIDER_ID = "deterministic_then_frozen_base_fallback"
PC01_PARAMETER_PROVIDER_VERSION = "v1"
PC01_PARAMETER_PROVIDER_PROMPT_SHA256 = (
    "ef91fbf3300e0a7bbfa3ecca7f54b7deca686dbf9177ec5a662ae3af9b6a8bd3"
)


@dataclass(frozen=True, slots=True)
class P3Targets:
    action_type: str
    bbox: tuple[float, float, float, float] | None
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.action_type not in ACTION_TYPE:
            raise SchemaError("P3 target action_type is unregistered")
        object.__setattr__(
            self, "bbox", normalized_bbox(self.bbox, context="P3 target bbox", nullable=True)
        )
        if self.action_type in GROUNDED_ACTIONS and self.bbox is None:
            raise SchemaError("P3 grounded target requires a bbox")
        if self.action_type not in GROUNDED_ACTIONS and self.bbox is not None:
            raise SchemaError("P3 non-grounded target must not invent a bbox")
        object.__setattr__(
            self,
            "parameters",
            json_mapping(self.parameters, context="P3 target parameters"),
        )
        try:
            validate_action_parameters(
                ActionType(self.action_type), self.parameters, self.bbox
            )
        except (TypeError, ValueError) as exc:
            raise SchemaError("P3 target parameters violate the six-action contract") from exc

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "P3Targets":
        require_exact_keys(value, set(cls.__dataclass_fields__), context="P3 targets")
        mapped = dict(value)
        mapped["bbox"] = normalized_bbox(mapped["bbox"], context="P3 target bbox")
        return cls(**mapped)


@dataclass(frozen=True, slots=True)
class P3Example:
    example_id: str
    task_id: str
    image: ImageArtifact
    text_state: ObservableTextState
    targets: P3Targets
    source_record_sha256: str

    def __post_init__(self) -> None:
        require_text(self.example_id, context="P3 example_id")
        require_text(self.task_id, context="P3 task_id")
        require_sha256(self.source_record_sha256, context="P3 source record")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "P3Example":
        require_exact_keys(value, set(cls.__dataclass_fields__), context="P3 example")
        mapped = dict(value)
        mapped["image"] = ImageArtifact.from_mapping(mapped["image"])
        mapped["text_state"] = ObservableTextState.from_mapping(mapped["text_state"])
        mapped["targets"] = P3Targets.from_mapping(mapped["targets"])
        return cls(**mapped)


@dataclass(frozen=True, slots=True)
class Pillar3DiagnosticInput:
    diagnostic_id: str
    campaign_id: str
    source_partition: str
    registered_seed: int
    bootstrap_samples: int
    bootstrap_confidence: float
    bootstrap_seed: int
    examples: tuple[P3Example, ...]
    backend_identity: CompanionBackendIdentity
    actions_frozen_before_diagnostics: bool
    targets_visible_to_predictor: bool
    invalid_output_policy: str
    runtime_outputs_consumed: bool
    affects_primary_table2: bool
    locked_test_rows_read: int
    evidence_role: str = ENGINEERING_EVIDENCE_ROLE
    paper_table_status: str = PAPER_TABLE_STATUS
    schema_version: str = INPUT_SCHEMA

    def __post_init__(self) -> None:
        require_text(self.diagnostic_id, context="P3 diagnostic_id")
        require_text(self.campaign_id, context="P3 campaign_id")
        if type(self.examples) is not tuple or not self.examples or any(
            type(example) is not P3Example for example in self.examples
        ):
            raise SchemaError("P3 diagnostic requires frozen examples")
        if type(self.backend_identity) is not CompanionBackendIdentity:
            raise SchemaError("P3 diagnostic backend identity is invalid")
        if len({row.example_id for row in self.examples}) != len(self.examples):
            raise SchemaError("P3 example IDs must be unique")
        if self.schema_version != INPUT_SCHEMA:
            raise SchemaError("unregistered P3 diagnostic input schema")
        if self.targets_visible_to_predictor is not False:
            raise SchemaError("P3 predictor cannot receive target/oracle fields")
        if self.invalid_output_policy != "one_rejected_executor_request_no_repair":
            raise SchemaError("P3 invalid-output accounting policy changed")
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
    def from_mapping(cls, value: Mapping[str, Any]) -> "Pillar3DiagnosticInput":
        require_exact_keys(value, set(cls.__dataclass_fields__), context="P3 diagnostic input")
        mapped = dict(value)
        mapped["examples"] = tuple(P3Example.from_mapping(row) for row in mapped["examples"])
        mapped["backend_identity"] = CompanionBackendIdentity.from_mapping(
            mapped["backend_identity"]
        )
        return cls(**mapped)

    @classmethod
    def load(cls, path: str | Path) -> "Pillar3DiagnosticInput":
        return cls.from_mapping(load_unique_json(path))


@dataclass(frozen=True, slots=True)
class P3DiagnosticView:
    example_id: str
    task_id: str
    image: ResolvedImage
    text_state: ObservableTextState


@dataclass(frozen=True, slots=True)
class ProviderAttemptEvidence:
    source: str
    status: str
    error_sha256: str | None

    def __post_init__(self) -> None:
        if self.source not in {"deterministic", "frozen_base_fallback"}:
            raise SchemaError("P3 provider attempt source is unregistered")
        if self.status not in {"RESOLVED", "REJECTED"}:
            raise SchemaError("P3 provider attempt status is unregistered")
        if self.status == "REJECTED":
            require_sha256(self.error_sha256, context="P3 provider rejection")
        elif self.error_sha256 is not None:
            raise SchemaError("resolved P3 provider attempt cannot carry an error")


@dataclass(frozen=True, slots=True)
class P3Prediction:
    status: str
    action_probabilities: Mapping[str, float] | None
    action_type: str | None
    bbox: tuple[float, float, float, float] | None
    grounding_confidence: float | None
    parameter_status: str
    provider_action_type: str | None
    parameters: Mapping[str, Any] | None
    provider_attempts: tuple[ProviderAttemptEvidence, ...]
    error_sha256: str | None
    accounted_rejected_executor_requests: int

    def __post_init__(self) -> None:
        if self.status not in PREDICTION_STATUSES:
            raise SchemaError("P3 prediction status is unregistered")
        if type(self.accounted_rejected_executor_requests) is not int:
            raise SchemaError("P3 rejected-request accounting must be an exact integer")
        if type(self.provider_attempts) is not tuple or any(
            type(attempt) is not ProviderAttemptEvidence for attempt in self.provider_attempts
        ):
            raise SchemaError("P3 provider attempts violate the typed contract")
        if len(self.provider_attempts) > 2:
            raise SchemaError("P3 predictor attempted an unregistered repair loop")
        sources = tuple(item.source for item in self.provider_attempts)
        if len(sources) != len(set(sources)):
            raise SchemaError("P3 predictor repeated a provider stage")
        if sources not in {
            (),
            ("deterministic",),
            ("deterministic", "frozen_base_fallback"),
        }:
            raise SchemaError("P3 provider attempt order changed")
        if self.status == "POLICY_REJECTED":
            if any(
                value is not None
                for value in (
                    self.action_probabilities,
                    self.action_type,
                    self.bbox,
                    self.grounding_confidence,
                    self.provider_action_type,
                    self.parameters,
                )
            ) or self.parameter_status != "NOT_ATTEMPTED" or self.provider_attempts:
                raise SchemaError("P3 policy rejection contains repaired/derived output")
            require_sha256(self.error_sha256, context="P3 policy rejection")
            if self.accounted_rejected_executor_requests != 1:
                raise SchemaError("P3 policy rejection must account for exactly one request")
            return
        if self.action_probabilities is None or self.action_type not in ACTION_TYPE:
            raise SchemaError("P3 action prediction is incomplete")
        probability_distribution(
            self.action_probabilities,
            labels=tuple(ACTION_TYPE),
            context="P3 action probabilities",
        )
        if selected_label(self.action_probabilities, tuple(ACTION_TYPE)) != self.action_type:
            raise SchemaError("P3 selected action disagrees with its probabilities")
        object.__setattr__(
            self, "bbox", normalized_bbox(self.bbox, context="P3 predicted bbox", nullable=True)
        )
        if self.action_type in GROUNDED_ACTIONS and self.bbox is None:
            raise SchemaError("P3 grounded prediction lacks a bbox")
        if self.action_type not in GROUNDED_ACTIONS and self.bbox is not None:
            raise SchemaError("P3 non-grounded prediction invented a bbox")
        if self.grounding_confidence is None:
            raise SchemaError("P3 action prediction lacks grounding confidence")
        require_probability(self.grounding_confidence, context="P3 grounding confidence")
        if self.status == "PARAMETER_REJECTED":
            if (
                self.parameter_status != "REJECTED"
                or self.provider_action_type is not None
                or self.parameters is not None
                or not self.provider_attempts
                or self.provider_attempts[-1].status != "REJECTED"
                or sources != ("deterministic", "frozen_base_fallback")
                or any(attempt.status != "REJECTED" for attempt in self.provider_attempts)
            ):
                raise SchemaError("P3 parameter rejection contains a silent repair")
            require_sha256(self.error_sha256, context="P3 parameter rejection")
            if self.accounted_rejected_executor_requests != 1:
                raise SchemaError("P3 parameter rejection must account for exactly one request")
            return
        if (
            self.parameter_status != "RESOLVED"
            or self.provider_action_type != self.action_type
            or self.parameters is None
            or not self.provider_attempts
            or self.provider_attempts[-1].status != "RESOLVED"
            or self.error_sha256 is not None
            or self.accounted_rejected_executor_requests != 0
        ):
            raise SchemaError("P3 resolved output violates no-repair accounting")
        parameters = json_mapping(self.parameters, context="P3 predicted parameters")
        object.__setattr__(self, "parameters", parameters)
        try:
            validate_action_parameters(ActionType(self.action_type), parameters, self.bbox)
        except (TypeError, ValueError) as exc:
            raise SchemaError("P3 resolved parameters violate six-action contract") from exc


class Pillar3DiagnosticPredictor(Protocol):
    @property
    def diagnostic_identity(self) -> CompanionBackendIdentity: ...

    def predict(self, view: P3DiagnosticView, *, seed: int) -> P3Prediction: ...


def canonical_p3_backend_error(
    identity: CompanionBackendIdentity,
    *,
    predictor: Any | None = None,
    require_input_provenance: bool = True,
) -> str | None:
    from .pc01_companion_diagnostics import (
        PC01_P3_PREDICTOR_ID,
        PC01_P3_PREDICTOR_QUALNAME,
        PC01_P3_PREDICTOR_VERSION,
        PC01Pillar3DiagnosticPredictor,
        pc01_companion_predictor_source_sha256,
        pc01_parameter_provider_source_sha256,
    )

    error = canonical_pc01_base_error(identity)
    if error is not None:
        return error
    expected = {
        "pillar": PILLAR,
        "predictor_id": PC01_P3_PREDICTOR_ID,
        "predictor_version": PC01_P3_PREDICTOR_VERSION,
        "predictor_module": PC01Pillar3DiagnosticPredictor.__module__,
        "predictor_qualname": PC01_P3_PREDICTOR_QUALNAME,
        "predictor_source_sha256": pc01_companion_predictor_source_sha256(),
        "provider_module": "web_agent.runtime.action_parameters",
        "provider_qualname": "HybridParameterProvider",
        "provider_id": PC01_PARAMETER_PROVIDER_ID,
        "provider_version": PC01_PARAMETER_PROVIDER_VERSION,
        "provider_policy_source": PC01_PARAMETER_PROVIDER_ID,
        "provider_source_sha256": pc01_parameter_provider_source_sha256(),
        "provider_prompt_sha256": PC01_PARAMETER_PROVIDER_PROMPT_SHA256,
    }
    for field, expected_value in expected.items():
        if getattr(identity, field) != expected_value:
            return f"canonical PC-01 P3 backend changed {field}"
    if predictor is not None and type(predictor) is not PC01Pillar3DiagnosticPredictor:
        return "canonical PC-01 P3 evidence requires the exact predictor class"
    if require_input_provenance:
        return (
            f"{CANONICAL_INPUT_PROVENANCE_STATUS}: P3 requires replay of an "
            "authoritative frozen source manifest, access ledger, and source-record bytes"
        )
    return None


def _target_size(bbox: tuple[float, float, float, float] | None) -> str:
    if bbox is None:
        return "not_grounded"
    area = bbox[2] * bbox[3]
    if area < 0.01:
        return "small"
    if area < 0.1:
        return "medium"
    return "large"


def _record(example: P3Example, prediction: P3Prediction) -> dict[str, Any]:
    predicted = asdict(prediction)
    iou = None
    if example.targets.bbox is not None:
        # Grounding recall must keep every grounded target in its denominator.
        # A rejected/missing predicted box is a miss, not an unavailable row.
        iou = (
            0.0
            if prediction.bbox is None
            else bbox_iou(prediction.bbox, example.targets.bbox)
        )
    parameter_exact = None
    if prediction.parameters is not None:
        parameter_exact = canonical_json_bytes(prediction.parameters) == canonical_json_bytes(
            example.targets.parameters
        )
    return {
        "record_type": "pre_action",
        "example_id": example.example_id,
        "task_id": example.task_id,
        "source_record_sha256": example.source_record_sha256,
        "target": {**asdict(example.targets), "target_size": _target_size(example.targets.bbox)},
        "prediction": predicted,
        "prediction_sha256": sha256_json(predicted),
        "bbox_iou": iou,
        "parameter_exact_match": parameter_exact,
    }


def _summary(
    records: list[dict[str, Any]], *, bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, Any]:
    def stat(name: str, rows, kind: str, function):
        return metric(
            rows,
            name=f"P3.{name}",
            metric_kind=kind,
            value=function,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            contribution_unit="frozen_pre_action_example",
        )

    overall = {
        "resolved_output_rate": stat(
            "resolved",
            records,
            "rate",
            lambda row: float(row["prediction"]["status"] == "RESOLVED"),
        ),
        "policy_rejection_rate": stat(
            "policy_rejection",
            records,
            "rate",
            lambda row: float(row["prediction"]["status"] == "POLICY_REJECTED"),
        ),
        "parameter_rejection_rate": stat(
            "parameter_rejection",
            records,
            "rate",
            lambda row: float(row["prediction"]["status"] == "PARAMETER_REJECTED"),
        ),
        "action_accuracy_including_rejections": stat(
            "action_accuracy_including_rejections",
            records,
            "rate",
            lambda row: float(
                row["prediction"]["action_type"] == row["target"]["action_type"]
            ),
        ),
        "action_accuracy_when_predicted": stat(
            "action_accuracy",
            records,
            "rate",
            lambda row: (
                float(row["prediction"]["action_type"] == row["target"]["action_type"])
                if row["prediction"]["action_type"] is not None
                else None
            ),
        ),
        "bbox_iou_mean": stat(
            "bbox_iou", records, "mean", lambda row: row["bbox_iou"]
        ),
        "bbox_recall_iou50": stat(
            "bbox_iou50",
            records,
            "rate",
            lambda row: None if row["bbox_iou"] is None else float(row["bbox_iou"] >= 0.5),
        ),
        "parameter_resolution_rate": stat(
            "parameter_resolution",
            records,
            "rate",
            lambda row: float(row["prediction"]["parameter_status"] == "RESOLVED"),
        ),
        "parameter_exact_match_when_resolved": stat(
            "parameter_exact",
            records,
            "rate",
            lambda row: (
                None
                if row["parameter_exact_match"] is None
                else float(row["parameter_exact_match"])
            ),
        ),
        "provider_attempts_mean": stat(
            "provider_attempts",
            records,
            "mean",
            lambda row: float(len(row["prediction"]["provider_attempts"])),
        ),
        "accounted_rejected_executor_requests_mean": stat(
            "rejected_executor_requests",
            records,
            "mean",
            lambda row: float(row["prediction"]["accounted_rejected_executor_requests"]),
        ),
    }
    by_action = {}
    for label in ACTION_TYPE:
        selected = [row for row in records if row["target"]["action_type"] == label]
        by_action[label] = {
            "example_count": len(selected),
            "action_recall_including_rejections": stat(
                f"action.{label}.recall_including_rejections",
                selected,
                "rate",
                lambda row: float(
                    row["prediction"]["action_type"]
                    == row["target"]["action_type"]
                ),
            ),
            "parameter_resolution_rate": stat(
                f"action.{label}.parameter",
                selected,
                "rate",
                lambda row: float(row["prediction"]["parameter_status"] == "RESOLVED"),
            ),
        }
    by_target_size = {}
    for size in ("small", "medium", "large"):
        selected = [row for row in records if row["target"]["target_size"] == size]
        by_target_size[size] = {
            "example_count": len(selected),
            "bbox_iou_mean": stat(
                f"target_size.{size}.iou", selected, "mean", lambda row: row["bbox_iou"]
            ),
            "bbox_recall_iou50": stat(
                f"target_size.{size}.iou50",
                selected,
                "rate",
                lambda row: None if row["bbox_iou"] is None else float(row["bbox_iou"] >= 0.5),
            ),
        }
    return {
        "example_count": len(records),
        "invalid_output_count": sum(row["prediction"]["status"] != "RESOLVED" for row in records),
        "accounted_rejected_executor_requests": sum(
            row["prediction"]["accounted_rejected_executor_requests"] for row in records
        ),
        "overall": overall,
        "by_action_class": by_action,
        "by_target_size": by_target_size,
    }


def run_pillar3_diagnostics(
    diagnostic: Pillar3DiagnosticInput,
    *,
    evidence_root: str | Path,
    predictor: Pillar3DiagnosticPredictor,
    repository_root: str | Path | None = None,
    allow_uncommitted_engineering: bool = False,
) -> dict[str, Any]:
    if diagnostic.evidence_role == EVIDENCE_ROLE and repository_root is None:
        raise CompanionDiagnosticError(
            "canonical P3 evidence requires clean repository verification"
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
    canonical_error = canonical_p3_backend_error(
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
        raise CompanionDiagnosticError("P3 repository identity changed before inference")
    raw_evidence_root = Path(evidence_root).absolute()
    if raw_evidence_root.is_symlink() or not raw_evidence_root.is_dir():
        raise CompanionDiagnosticError(
            "P3 evidence root must be a non-symlink directory"
        )
    resolved_evidence_root = raw_evidence_root.resolve()
    records: list[dict[str, Any]] = []
    images: list[ResolvedImage] = []
    for example in diagnostic.examples:
        image = example.image.resolve(resolved_evidence_root)
        images.append(image)
        view = P3DiagnosticView(
            example.example_id, example.task_id, image, example.text_state
        )
        before = sha256_json(asdict(view))
        prediction = predictor.predict(
            view,
            seed=stable_int_seed(
                diagnostic.registered_seed,
                diagnostic.diagnostic_id,
                example.example_id,
                "P3.pre_action",
            ),
        )
        if type(prediction) is not P3Prediction:
            raise CompanionDiagnosticError("P3 predictor returned an invalid output contract")
        if sha256_json(asdict(view)) != before:
            raise CompanionDiagnosticError("P3 predictor mutated its pre-action input")
        records.append(_record(example, prediction))
    rehash_images(images)
    if source_hash != verify_predictor(predictor, diagnostic.backend_identity)[1]:
        raise CompanionDiagnosticError("P3 predictor source changed during inference")
    if repository_root is not None:
        ending = repository_attestation(
            repository_root,
            expected_commit=diagnostic.backend_identity.repository_commit,
            source_path=source_path,
            factory_entrypoint=diagnostic.backend_identity.factory_entrypoint,
            factory_source_sha256=diagnostic.backend_identity.factory_source_sha256,
        )
        if ending != source_attestation:
            raise CompanionDiagnosticError("P3 repository identity changed during inference")
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
        "invalid_output_policy": diagnostic.invalid_output_policy,
        "records": records,
        "summary": summary,
        "browser_actions_executed": 0,
        "recovery_triggers_emitted": 0,
        "memory_queries_executed": 0,
        "memory_writes_executed": 0,
        "locked_test_rows_read": 0,
    }
    validate_pillar3_diagnostic_report(report)
    return report


def validate_pillar3_diagnostic_report(report: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "pillar", "diagnostic_id", "campaign_id", "source_partition",
        "evidence_role", "promotion_status", "paper_table_status", "claim_scope",
        "affects_primary_table2", "input_manifest_sha256", "backend_identity",
        "source_attestation", "statistics", "invalid_output_policy", "records", "summary",
        "browser_actions_executed", "recovery_triggers_emitted", "memory_queries_executed",
        "memory_writes_executed", "locked_test_rows_read",
    }
    require_exact_keys(report, required, context="P3 diagnostic report")
    backend = CompanionBackendIdentity.from_mapping(report["backend_identity"])
    validate_report_envelope(
        report,
        pillar=PILLAR,
        report_schema=REPORT_SCHEMA,
        canonical_role=EVIDENCE_ROLE,
        canonical_error=canonical_p3_backend_error(backend),
        backend_identity=backend,
    )
    if (
        report["claim_scope"] != CLAIM_SCOPE
        or report["invalid_output_policy"] != "one_rejected_executor_request_no_repair"
        or type(report["locked_test_rows_read"]) is not int
        or report["locked_test_rows_read"] != 0
        or type(report["recovery_triggers_emitted"]) is not int
        or report["recovery_triggers_emitted"] != 0
        or type(report["memory_queries_executed"]) is not int
        or report["memory_queries_executed"] != 0
    ):
        raise SchemaError("P3 report scientific boundary changed")
    statistics = report["statistics"]
    if not isinstance(statistics, Mapping) or set(statistics) != {
        "bootstrap_samples", "bootstrap_confidence", "bootstrap_seed", "cluster_unit"
    }:
        raise SchemaError("P3 report statistics registration changed")
    if (
        statistics["bootstrap_confidence"] != 0.95
        or statistics["cluster_unit"] != "task_id"
        or type(statistics["bootstrap_samples"]) is not int
        or statistics["bootstrap_samples"] <= 0
        or type(statistics["bootstrap_seed"]) is not int
        or statistics["bootstrap_seed"] < 0
    ):
        raise SchemaError("P3 report interval registration changed")
    records = report["records"]
    if not isinstance(records, list) or not records:
        raise SchemaError("P3 report requires records")
    # Recreate each typed prediction so status/no-repair accounting cannot be
    # changed while retaining a recomputable-looking summary.
    for row in records:
        require_exact_keys(
            row,
            {
                "record_type", "example_id", "task_id", "source_record_sha256",
                "target", "prediction", "prediction_sha256", "bbox_iou",
                "parameter_exact_match",
            },
            context="P3 report record",
        )
        if row["record_type"] != "pre_action":
            raise SchemaError("P3 report record type changed")
        require_text(row["example_id"], context="P3 report example_id")
        require_text(row["task_id"], context="P3 report task_id")
        require_sha256(row["source_record_sha256"], context="P3 report source")
        target = dict(row["target"])
        target_size = target.pop("target_size", None)
        typed_target = P3Targets.from_mapping(target)
        if target_size != _target_size(typed_target.bbox):
            raise SchemaError("P3 report target-size bin changed")
        prediction = dict(row["prediction"])
        prediction["provider_attempts"] = tuple(
            ProviderAttemptEvidence(**attempt) for attempt in prediction["provider_attempts"]
        )
        typed_prediction = P3Prediction(**prediction)
        if row["prediction_sha256"] != sha256_json(asdict(typed_prediction)):
            raise SchemaError("P3 report prediction hash changed")
        expected_iou = (
            None
            if typed_target.bbox is None
            else (
                0.0
                if typed_prediction.bbox is None
                else bbox_iou(typed_prediction.bbox, typed_target.bbox)
            )
        )
        expected_parameter = (
            None
            if typed_prediction.parameters is None
            else canonical_json_bytes(typed_prediction.parameters)
            == canonical_json_bytes(typed_target.parameters)
        )
        if row["bbox_iou"] != expected_iou or (
            row["parameter_exact_match"] is not expected_parameter
        ):
            raise SchemaError("P3 report derived grounding/parameter evidence changed")
    recomputed = _summary(
        records,
        bootstrap_samples=statistics["bootstrap_samples"],
        bootstrap_seed=statistics["bootstrap_seed"],
    )
    if report["summary"] != recomputed:
        raise SchemaError("P3 diagnostic summary is not reproducible from records")
    validate_metric_tree(report["summary"])


def write_pillar3_diagnostic_package(
    output_dir: str | Path,
    *,
    diagnostic: Pillar3DiagnosticInput,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    validate_pillar3_diagnostic_report(report)
    _validate_p3_report_input_binding(diagnostic, report)
    return write_package(
        output_dir,
        pillar=PILLAR,
        diagnostic=diagnostic,
        report=report,
        input_filename=INPUT_FILENAME,
        report_filename=REPORT_FILENAME,
    )


def validate_pillar3_diagnostic_package(output_dir: str | Path) -> dict[str, Any]:
    result = validate_package(
        output_dir,
        pillar=PILLAR,
        input_filename=INPUT_FILENAME,
        report_filename=REPORT_FILENAME,
        load_input=Pillar3DiagnosticInput.from_mapping,
        validate_report=validate_pillar3_diagnostic_report,
    )
    root = Path(output_dir)
    diagnostic = Pillar3DiagnosticInput.from_mapping(
        load_unique_json(root / INPUT_FILENAME)
    )
    _validate_p3_report_input_binding(diagnostic, load_unique_json(root / REPORT_FILENAME))
    return result


def _validate_p3_report_input_binding(
    diagnostic: Pillar3DiagnosticInput, report: Mapping[str, Any]
) -> None:
    records = report["records"]
    by_id = {row["example_id"]: row for row in records}
    if len(by_id) != len(diagnostic.examples) or set(by_id) != {
        row.example_id for row in diagnostic.examples
    }:
        raise CompanionDiagnosticError("P3 report/input example inventory changed")
    for example in diagnostic.examples:
        row = by_id[example.example_id]
        expected_target = {
            **asdict(example.targets),
            "target_size": _target_size(example.targets.bbox),
        }
        if (
            row["task_id"] != example.task_id
            or row["source_record_sha256"] != example.source_record_sha256
            or sha256_json(row["target"]) != sha256_json(expected_target)
        ):
            raise CompanionDiagnosticError("P3 report/input target evidence changed")
