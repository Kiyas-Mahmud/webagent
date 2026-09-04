"""Offline Pillar-4 retrieval, admission, and intervention diagnostics.

The harness queries one immutable train-only store after a frozen post-failure
embedding.  Relevance labels are withheld from the retriever, store bytes are
hashed before and after every run, and any provenance/leakage violation fails
instead of becoming a favourable metric.  No result is an E2-vs-E3 task-
success claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import struct
from typing import Any, Protocol

from web_agent.labels import RECOVERY_STRATEGY

from .common import SchemaError, sha256_bytes, sha256_file, sha256_json, stable_int_seed
from .companion_diagnostics import (
    ENGINEERING_EVIDENCE_ROLE,
    CANONICAL_INPUT_PROVENANCE_STATUS,
    PAPER_TABLE_STATUS,
    CompanionBackendIdentity,
    CompanionDiagnosticError,
    canonical_pc01_base_error,
    hash_directory_read_only,
    load_unique_json,
    metric,
    require_bool,
    require_exact_keys,
    require_sha256,
    require_text,
    repository_attestation,
    run_attestation,
    validate_input_envelope,
    validate_metric_tree,
    validate_package,
    validate_report_envelope,
    verify_predictor,
    write_package,
)


PILLAR = "P4"
INPUT_SCHEMA = "table2.pillar4-diagnostic-input.v1"
REPORT_SCHEMA = "table2.pillar4-diagnostic-report.v1"
EVIDENCE_ROLE = "P4_COMPANION_MECHANISM_EVIDENCE"
INPUT_FILENAME = "pillar4_diagnostic_input.json"
REPORT_FILENAME = "pillar4_diagnostic_report.json"
EMBEDDING_DIMENSION = 768
RUNTIME_TOP_K = 3
DIAGNOSTIC_MODES = frozenset({"PRIMARY_INTERVENTION", "RETRIEVAL_ONLY"})
CLAIM_SCOPE = {
    "retrieval": "OFFLINE_FROZEN_TRAIN_ONLY_COMPONENT_QUALITY",
    "admission": "OFFLINE_FROZEN_THRESHOLD_BEHAVIOUR",
    "intervention": "OFFLINE_STRATEGY_CHANGE_ONLY",
    "end_to_end_memory_benefit": "NOT_MEASURED_REQUIRES_PAIRED_E2_VS_E3",
    "memory_write": "FORBIDDEN",
    "table2_row_or_contrast": "NONE",
}


def float32_vector_sha256(values: Sequence[float]) -> str:
    return sha256_bytes(b"".join(struct.pack("<f", float(value)) for value in values))


def _embedding(value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != EMBEDDING_DIMENSION:
        raise SchemaError(f"P4 query embedding must contain {EMBEDDING_DIMENSION} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise SchemaError("P4 query embedding contains non-finite values")
    norm = math.sqrt(sum(item * item for item in result))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise SchemaError("P4 diagnostic query embedding must be L2 normalized")
    return result


@dataclass(frozen=True, slots=True)
class P4Example:
    example_id: str
    task_id: str
    episode_id: str
    incident_id: str
    query_stage: str
    post_failure_observation_id: str
    failed_action_id: str
    duplicate_cluster_ids: tuple[str, ...]
    query_embedding: tuple[float, ...]
    query_embedding_sha256: str
    embedding_request_sha256: str
    processed_batch_sha256: str
    checkpoint_sha256: str
    processor_contract_sha256: str
    shadow_strategy: str
    target_strategy: str
    relevant_memory_ids: tuple[str, ...]
    source_record_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "example_id", "task_id", "episode_id", "incident_id",
            "post_failure_observation_id", "failed_action_id",
        ):
            require_text(getattr(self, name), context=f"P4 {name}")
        if self.query_stage != "post_action_failure":
            raise SchemaError("P4 diagnostics forbid pre-action/non-failure retrieval")
        if type(self.duplicate_cluster_ids) is not tuple or not self.duplicate_cluster_ids or any(
            not isinstance(item, str) or not item.strip() for item in self.duplicate_cluster_ids
        ):
            raise SchemaError("P4 query requires audited duplicate-cluster IDs")
        if len(self.duplicate_cluster_ids) != len(set(self.duplicate_cluster_ids)):
            raise SchemaError("P4 duplicate-cluster IDs must be unique")
        embedding = _embedding(self.query_embedding)
        if embedding != self.query_embedding:
            object.__setattr__(self, "query_embedding", embedding)
        require_sha256(self.query_embedding_sha256, context="P4 query embedding")
        if float32_vector_sha256(embedding) != self.query_embedding_sha256:
            raise SchemaError("P4 query embedding bytes differ from their SHA-256")
        for name in (
            "embedding_request_sha256", "processed_batch_sha256", "checkpoint_sha256",
            "processor_contract_sha256", "source_record_sha256",
        ):
            require_sha256(getattr(self, name), context=f"P4 {name}")
        if self.shadow_strategy not in RECOVERY_STRATEGY:
            raise SchemaError("P4 shadow strategy is unregistered")
        if self.target_strategy not in RECOVERY_STRATEGY:
            raise SchemaError("P4 target strategy is unregistered")
        if type(self.relevant_memory_ids) is not tuple or any(
            not isinstance(item, str) or not item.strip() for item in self.relevant_memory_ids
        ):
            raise SchemaError("P4 relevant-memory IDs must be non-empty")
        if len(self.relevant_memory_ids) != len(set(self.relevant_memory_ids)):
            raise SchemaError("P4 relevant-memory IDs must be unique")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "P4Example":
        require_exact_keys(value, set(cls.__dataclass_fields__), context="P4 example")
        mapped = dict(value)
        mapped["duplicate_cluster_ids"] = tuple(mapped["duplicate_cluster_ids"])
        mapped["query_embedding"] = _embedding(mapped["query_embedding"])
        mapped["relevant_memory_ids"] = tuple(mapped["relevant_memory_ids"])
        return cls(**mapped)


@dataclass(frozen=True, slots=True)
class Pillar4DiagnosticInput:
    diagnostic_id: str
    campaign_id: str
    source_partition: str
    registered_seed: int
    bootstrap_samples: int
    bootstrap_confidence: float
    bootstrap_seed: int
    diagnostic_mode: str
    admission_threshold: float
    examples: tuple[P4Example, ...]
    backend_identity: CompanionBackendIdentity
    actions_frozen_before_diagnostics: bool
    store_frozen_read_only: bool
    relevance_labels_visible_to_retriever: bool
    evaluation_memory_write_enabled: bool
    runtime_outputs_consumed: bool
    affects_primary_table2: bool
    locked_test_rows_read: int
    evidence_role: str = ENGINEERING_EVIDENCE_ROLE
    paper_table_status: str = PAPER_TABLE_STATUS
    schema_version: str = INPUT_SCHEMA

    def __post_init__(self) -> None:
        require_text(self.diagnostic_id, context="P4 diagnostic_id")
        require_text(self.campaign_id, context="P4 campaign_id")
        if self.diagnostic_mode not in DIAGNOSTIC_MODES:
            raise SchemaError("P4 diagnostic mode is unregistered")
        if isinstance(self.admission_threshold, bool) or not isinstance(
            self.admission_threshold, (int, float)
        ) or not math.isfinite(float(self.admission_threshold)) or not -1 <= float(
            self.admission_threshold
        ) <= 1:
            raise SchemaError("P4 admission threshold must be finite in [-1, 1]")
        if type(self.examples) is not tuple or any(
            type(row) is not P4Example for row in self.examples
        ) or not self.examples or len({row.example_id for row in self.examples}) != len(
            self.examples
        ):
            raise SchemaError("P4 diagnostic requires unique frozen examples")
        if type(self.backend_identity) is not CompanionBackendIdentity:
            raise SchemaError("P4 diagnostic backend identity is invalid")
        if self.schema_version != INPUT_SCHEMA:
            raise SchemaError("unregistered P4 diagnostic input schema")
        if self.store_frozen_read_only is not True:
            raise SchemaError("P4 diagnostic store must be frozen/read-only")
        if self.relevance_labels_visible_to_retriever is not False:
            raise SchemaError("P4 retriever cannot receive relevance labels")
        if self.evaluation_memory_write_enabled is not False:
            raise SchemaError("P4 evaluation memory writes are forbidden")
        if any(
            row.checkpoint_sha256 != self.backend_identity.checkpoint_sha256
            for row in self.examples
        ):
            raise SchemaError("P4 query embedding cites another checkpoint")
        if any(
            row.processor_contract_sha256 != self.backend_identity.processor_contract_sha256
            for row in self.examples
        ):
            raise SchemaError("P4 query embedding cites another processor")
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
    def from_mapping(cls, value: Mapping[str, Any]) -> "Pillar4DiagnosticInput":
        require_exact_keys(value, set(cls.__dataclass_fields__), context="P4 diagnostic input")
        mapped = dict(value)
        mapped["examples"] = tuple(P4Example.from_mapping(row) for row in mapped["examples"])
        mapped["backend_identity"] = CompanionBackendIdentity.from_mapping(
            mapped["backend_identity"]
        )
        return cls(**mapped)

    @classmethod
    def load(cls, path: str | Path) -> "Pillar4DiagnosticInput":
        return cls.from_mapping(load_unique_json(path))


@dataclass(frozen=True, slots=True)
class P4QueryView:
    example_id: str
    task_id: str
    episode_id: str
    incident_id: str
    post_failure_observation_id: str
    failed_action_id: str
    duplicate_cluster_ids: tuple[str, ...]
    query_embedding: tuple[float, ...]
    query_embedding_sha256: str
    embedding_request_sha256: str
    processed_batch_sha256: str
    checkpoint_sha256: str
    processor_contract_sha256: str
    shadow_strategy: str


@dataclass(frozen=True, slots=True)
class P4Candidate:
    rank: int
    memory_id: str
    similarity: float
    source_split: str
    source_task_id: str
    source_episode_id: str
    duplicate_cluster_id: str
    strategy: str
    memory_update_flag: bool
    verified_recovery_success: bool
    final_task_success: bool
    provenance_valid: bool

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise SchemaError("P4 candidate rank must be a positive exact integer")
        for name in (
            "memory_id", "source_split", "source_task_id", "source_episode_id",
            "duplicate_cluster_id",
        ):
            require_text(getattr(self, name), context=f"P4 candidate {name}")
        if isinstance(self.similarity, bool) or not isinstance(self.similarity, (int, float)):
            raise SchemaError("P4 candidate similarity must be numeric")
        if not math.isfinite(float(self.similarity)) or not -1 <= float(self.similarity) <= 1:
            raise SchemaError("P4 candidate similarity must be finite in [-1, 1]")
        if self.strategy not in RECOVERY_STRATEGY or self.strategy == "NONE":
            raise SchemaError("P4 candidate strategy is unregistered or non-corrective")
        for name in (
            "memory_update_flag", "verified_recovery_success", "final_task_success",
            "provenance_valid",
        ):
            require_bool(getattr(self, name), context=f"P4 candidate {name}")


@dataclass(frozen=True, slots=True)
class P4RetrievalPrediction:
    query_embedding_sha256: str
    store_manifest_sha256: str
    admission_threshold: float
    candidates: tuple[P4Candidate, ...]
    exclusion_reasons: Mapping[str, str]
    considered_count: int
    eligible_count: int
    write_enabled: bool

    def __post_init__(self) -> None:
        require_sha256(self.query_embedding_sha256, context="P4 prediction query embedding")
        require_sha256(self.store_manifest_sha256, context="P4 prediction store manifest")
        if isinstance(self.admission_threshold, bool) or not isinstance(
            self.admission_threshold, (int, float)
        ) or not math.isfinite(float(self.admission_threshold)) or not -1 <= float(
            self.admission_threshold
        ) <= 1:
            raise SchemaError("P4 prediction admission threshold is invalid")
        if type(self.candidates) is not tuple or any(
            type(candidate) is not P4Candidate for candidate in self.candidates
        ):
            raise SchemaError("P4 candidates violate the typed contract")
        if len(self.candidates) > RUNTIME_TOP_K:
            raise SchemaError("P4 primary retrieval returned more than top-3")
        expected_ranks = tuple(range(1, len(self.candidates) + 1))
        if tuple(candidate.rank for candidate in self.candidates) != expected_ranks:
            raise SchemaError("P4 candidate ranks are not contiguous")
        expected_order = tuple(
            sorted(self.candidates, key=lambda row: (-float(row.similarity), row.memory_id))
        )
        if self.candidates != expected_order:
            raise SchemaError("P4 candidates violate deterministic similarity/ID ordering")
        if len({row.memory_id for row in self.candidates}) != len(self.candidates):
            raise SchemaError("P4 retrieval returned duplicate memory IDs")
        if type(self.considered_count) is not int or type(self.eligible_count) is not int:
            raise SchemaError("P4 reader counts must be exact integers")
        if (
            self.considered_count < 0
            or self.eligible_count < 0
            or self.eligible_count > self.considered_count
            or self.eligible_count < len(self.candidates)
        ):
            raise SchemaError("P4 reader counts are inconsistent")
        if not isinstance(self.exclusion_reasons, Mapping) or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in self.exclusion_reasons.items()
        ):
            raise SchemaError("P4 exclusion evidence is malformed")
        if set(self.exclusion_reasons.values()) - {
            "same_task", "same_episode", "duplicate_cluster", "explicit_memory_id"
        }:
            raise SchemaError("P4 exclusion reason is unregistered")
        if set(self.exclusion_reasons) & {candidate.memory_id for candidate in self.candidates}:
            raise SchemaError("P4 memory cannot be both excluded and returned")
        if self.considered_count != self.eligible_count + len(self.exclusion_reasons):
            raise SchemaError("P4 exclusion evidence does not cover considered memories")
        if self.write_enabled is not False:
            raise SchemaError("P4 diagnostic retriever exposes a write capability")


class Pillar4DiagnosticRetriever(Protocol):
    @property
    def diagnostic_identity(self) -> CompanionBackendIdentity: ...

    @property
    def store_root(self) -> Path: ...

    def retrieve(self, view: P4QueryView, *, seed: int) -> P4RetrievalPrediction: ...


def canonical_p4_backend_error(
    identity: CompanionBackendIdentity,
    *,
    predictor: Any | None = None,
    require_input_provenance: bool = True,
) -> str | None:
    from .pc01_companion_diagnostics import (
        PC01_P4_RETRIEVER_ID,
        PC01_P4_RETRIEVER_QUALNAME,
        PC01_P4_RETRIEVER_VERSION,
        PC01_P4_EMBEDDING_PROVIDER_ID,
        PC01_P4_EMBEDDING_PROVIDER_MODULE,
        PC01_P4_EMBEDDING_PROVIDER_QUALNAME,
        PC01_P4_EMBEDDING_PROVIDER_VERSION,
        PC01Pillar4DiagnosticRetriever,
        pc01_companion_predictor_source_sha256,
        pc01_memory_embedding_source_sha256,
    )

    error = canonical_pc01_base_error(identity)
    if error is not None:
        return error
    expected = {
        "pillar": PILLAR,
        "predictor_id": PC01_P4_RETRIEVER_ID,
        "predictor_version": PC01_P4_RETRIEVER_VERSION,
        "predictor_module": PC01Pillar4DiagnosticRetriever.__module__,
        "predictor_qualname": PC01_P4_RETRIEVER_QUALNAME,
        "predictor_source_sha256": pc01_companion_predictor_source_sha256(),
        "embedding_provider_id": PC01_P4_EMBEDDING_PROVIDER_ID,
        "embedding_provider_version": PC01_P4_EMBEDDING_PROVIDER_VERSION,
        "embedding_provider_module": PC01_P4_EMBEDDING_PROVIDER_MODULE,
        "embedding_provider_qualname": PC01_P4_EMBEDDING_PROVIDER_QUALNAME,
        "embedding_provider_source_sha256": pc01_memory_embedding_source_sha256(),
    }
    for field, expected_value in expected.items():
        if getattr(identity, field) != expected_value:
            return f"canonical PC-01 P4 backend changed {field}"
    if predictor is not None and type(predictor) is not PC01Pillar4DiagnosticRetriever:
        return "canonical PC-01 P4 evidence requires the exact retriever class"
    if require_input_provenance:
        return (
            f"{CANONICAL_INPUT_PROVENANCE_STATUS}: P4 requires authoritative "
            "source-record replay and exact checkpoint-backed memory-embedding "
            "execution evidence"
        )
    return None


def _validate_no_leakage(view: P4QueryView, prediction: P4RetrievalPrediction) -> None:
    for candidate in prediction.candidates:
        if candidate.source_split != "train":
            raise CompanionDiagnosticError("P4 retrieval returned a non-training memory")
        if not (
            candidate.memory_update_flag
            and candidate.verified_recovery_success
            and candidate.final_task_success
            and candidate.provenance_valid
        ):
            raise CompanionDiagnosticError("P4 retrieval returned provenance-ineligible memory")
        if candidate.source_task_id == view.task_id:
            raise CompanionDiagnosticError("P4 retrieval returned same-task memory")
        if candidate.source_episode_id == view.episode_id:
            raise CompanionDiagnosticError("P4 retrieval returned same-episode memory")
        if candidate.duplicate_cluster_id in view.duplicate_cluster_ids:
            raise CompanionDiagnosticError("P4 retrieval returned duplicate-cluster memory")


def _record(
    example: P4Example,
    prediction: P4RetrievalPrediction,
    *,
    diagnostic_mode: str,
) -> dict[str, Any]:
    admitted = next(
        (
            candidate
            for candidate in prediction.candidates
            if float(candidate.similarity) >= float(prediction.admission_threshold)
        ),
        None,
    )
    intervention_enabled = diagnostic_mode == "PRIMARY_INTERVENTION"
    final_strategy = example.shadow_strategy
    if intervention_enabled and admitted is not None and admitted.strategy != "NONE":
        final_strategy = admitted.strategy
    relevant = set(example.relevant_memory_ids)
    reciprocal_rank = 0.0
    for candidate in prediction.candidates:
        if candidate.memory_id in relevant:
            reciprocal_rank = 1.0 / candidate.rank
            break
    return {
        "record_type": "post_failure_memory_query",
        "example_id": example.example_id,
        "task_id": example.task_id,
        "episode_id": example.episode_id,
        "incident_id": example.incident_id,
        "query_stage": example.query_stage,
        "query_embedding_sha256": example.query_embedding_sha256,
        "embedding_request_sha256": example.embedding_request_sha256,
        "processed_batch_sha256": example.processed_batch_sha256,
        "post_failure_observation_id": example.post_failure_observation_id,
        "failed_action_id": example.failed_action_id,
        "duplicate_cluster_ids": list(example.duplicate_cluster_ids),
        "source_record_sha256": example.source_record_sha256,
        "target": {
            "target_strategy": example.target_strategy,
            "relevant_memory_ids": list(example.relevant_memory_ids),
        },
        "shadow_strategy": example.shadow_strategy,
        "prediction": asdict(prediction),
        "prediction_sha256": sha256_json(asdict(prediction)),
        "admitted_candidate_id": None if admitted is None else admitted.memory_id,
        "admitted_strategy": None if admitted is None else admitted.strategy,
        "admitted_relevant": (
            None if admitted is None else admitted.memory_id in relevant
        ),
        "intervention_enabled": intervention_enabled,
        "final_strategy": final_strategy,
        "changed_strategy": final_strategy != example.shadow_strategy,
        "strategy_hit_at_1": bool(
            prediction.candidates
            and prediction.candidates[0].strategy == example.target_strategy
        ),
        "strategy_hit_at_3": any(
            candidate.strategy == example.target_strategy for candidate in prediction.candidates
        ),
        "retrieval_reciprocal_rank": reciprocal_rank,
        "irrelevant_fraction": (
            None
            if not prediction.candidates
            else sum(candidate.memory_id not in relevant for candidate in prediction.candidates)
            / len(prediction.candidates)
        ),
    }


def _summary(
    records: list[dict[str, Any]], *, bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, Any]:
    def stat(name: str, kind: str, function):
        return metric(
            records,
            name=f"P4.{name}",
            metric_kind=kind,
            value=function,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            contribution_unit="frozen_post_failure_query",
        )

    return {
        "query_count": len(records),
        "retrieval_coverage": stat(
            "retrieval_coverage", "rate", lambda row: float(bool(row["prediction"]["candidates"]))
        ),
        "admission_rate": stat(
            "admission", "rate", lambda row: float(row["admitted_candidate_id"] is not None)
        ),
        "admitted_relevance_rate_when_admitted": stat(
            "admitted_relevance",
            "rate",
            lambda row: (
                None
                if row["admitted_relevant"] is None
                else float(row["admitted_relevant"])
            ),
        ),
        "abstention_rate": stat(
            "abstention", "rate", lambda row: float(row["admitted_candidate_id"] is None)
        ),
        "strategy_hit_at_1": stat(
            "strategy_hit1", "rate", lambda row: float(row["strategy_hit_at_1"])
        ),
        "strategy_hit_at_3": stat(
            "strategy_hit3", "rate", lambda row: float(row["strategy_hit_at_3"])
        ),
        "mean_reciprocal_rank": stat(
            "mrr", "mean", lambda row: float(row["retrieval_reciprocal_rank"])
        ),
        "irrelevant_memory_fraction": stat(
            "irrelevant_fraction", "mean", lambda row: row["irrelevant_fraction"]
        ),
        "intervention_coverage": stat(
            "intervention",
            "rate",
            lambda row: float(
                row["intervention_enabled"]
                and row["admitted_candidate_id"] is not None
            ),
        ),
        "changed_strategy_rate": stat(
            "changed_strategy", "rate", lambda row: float(row["changed_strategy"])
        ),
        "final_strategy_agreement": stat(
            "final_strategy_agreement",
            "rate",
            lambda row: float(row["final_strategy"] == row["target"]["target_strategy"]),
        ),
        "shadow_strategy_agreement": stat(
            "shadow_strategy_agreement",
            "rate",
            lambda row: float(
                row["shadow_strategy"] == row["target"]["target_strategy"]
            ),
        ),
        "changed_strategy_target_agreement": stat(
            "changed_strategy_target_agreement",
            "rate",
            lambda row: (
                float(row["final_strategy"] == row["target"]["target_strategy"])
                if row["changed_strategy"]
                else None
            ),
        ),
        "no_write_verified_rate": stat(
            "no_write", "rate", lambda row: float(row["prediction"]["write_enabled"] is False)
        ),
    }


def run_pillar4_diagnostics(
    diagnostic: Pillar4DiagnosticInput,
    *,
    evidence_root: str | Path,
    predictor: Pillar4DiagnosticRetriever,
    repository_root: str | Path | None = None,
    allow_uncommitted_engineering: bool = False,
) -> dict[str, Any]:
    if diagnostic.evidence_role == EVIDENCE_ROLE and repository_root is None:
        raise CompanionDiagnosticError(
            "canonical P4 evidence requires clean repository verification"
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
    canonical_error = canonical_p4_backend_error(
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
        raise CompanionDiagnosticError("P4 repository identity changed before inference")
    raw_root = Path(evidence_root).absolute()
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise CompanionDiagnosticError(
            "P4 evidence root must be a non-symlink directory"
        )
    root = raw_root.resolve()
    if predictor.store_root.resolve() != root:
        raise CompanionDiagnosticError("P4 retriever store differs from supplied evidence root")
    before_sha256, before_files = hash_directory_read_only(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != (
        diagnostic.backend_identity.memory_manifest_sha256
    ):
        raise CompanionDiagnosticError("P4 store manifest differs from frozen backend identity")
    records: list[dict[str, Any]] = []
    for example in diagnostic.examples:
        view = P4QueryView(
            example.example_id,
            example.task_id,
            example.episode_id,
            example.incident_id,
            example.post_failure_observation_id,
            example.failed_action_id,
            example.duplicate_cluster_ids,
            example.query_embedding,
            example.query_embedding_sha256,
            example.embedding_request_sha256,
            example.processed_batch_sha256,
            example.checkpoint_sha256,
            example.processor_contract_sha256,
            example.shadow_strategy,
        )
        before_view = sha256_json(asdict(view))
        prediction = predictor.retrieve(
            view,
            seed=stable_int_seed(
                diagnostic.registered_seed, diagnostic.diagnostic_id, example.example_id, "P4.query"
            ),
        )
        if type(prediction) is not P4RetrievalPrediction:
            raise CompanionDiagnosticError("P4 retriever returned an invalid contract")
        if sha256_json(asdict(view)) != before_view:
            raise CompanionDiagnosticError("P4 retriever mutated its query input")
        if prediction.query_embedding_sha256 != example.query_embedding_sha256:
            raise CompanionDiagnosticError("P4 retriever cited another query embedding")
        if prediction.store_manifest_sha256 != diagnostic.backend_identity.memory_manifest_sha256:
            raise CompanionDiagnosticError("P4 retriever cited another frozen store")
        if not math.isclose(
            float(prediction.admission_threshold),
            float(diagnostic.admission_threshold),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise CompanionDiagnosticError("P4 retriever changed the frozen admission threshold")
        _validate_no_leakage(view, prediction)
        records.append(
            _record(example, prediction, diagnostic_mode=diagnostic.diagnostic_mode)
        )
    after_sha256, after_files = hash_directory_read_only(root)
    if before_sha256 != after_sha256 or before_files != after_files:
        raise CompanionDiagnosticError("P4 frozen store changed during diagnostics")
    if source_hash != verify_predictor(predictor, diagnostic.backend_identity)[1]:
        raise CompanionDiagnosticError("P4 retriever source changed during inference")
    if repository_root is not None:
        ending = repository_attestation(
            repository_root,
            expected_commit=diagnostic.backend_identity.repository_commit,
            source_path=source_path,
            factory_entrypoint=diagnostic.backend_identity.factory_entrypoint,
            factory_source_sha256=diagnostic.backend_identity.factory_source_sha256,
        )
        if ending != source_attestation:
            raise CompanionDiagnosticError("P4 repository identity changed during inference")
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
        "diagnostic_mode": diagnostic.diagnostic_mode,
        "admission_threshold": diagnostic.admission_threshold,
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
        "store_integrity": {
            "directory_sha256_before": before_sha256,
            "directory_sha256_after": after_sha256,
            "file_count": len(before_files),
            "unchanged": True,
            "write_enabled": False,
        },
        "records": records,
        "summary": summary,
        "browser_actions_executed": 0,
        "recovery_triggers_emitted": 0,
        "memory_queries_executed": len(records),
        "memory_writes_executed": 0,
        "locked_test_rows_read": 0,
    }
    validate_pillar4_diagnostic_report(report)
    return report


def validate_pillar4_diagnostic_report(report: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "pillar", "diagnostic_id", "campaign_id", "source_partition",
        "diagnostic_mode", "admission_threshold", "evidence_role", "promotion_status",
        "paper_table_status",
        "claim_scope", "affects_primary_table2", "input_manifest_sha256",
        "backend_identity", "source_attestation", "statistics", "store_integrity",
        "records", "summary", "browser_actions_executed", "recovery_triggers_emitted",
        "memory_queries_executed", "memory_writes_executed", "locked_test_rows_read",
    }
    require_exact_keys(report, required, context="P4 diagnostic report")
    backend = CompanionBackendIdentity.from_mapping(report["backend_identity"])
    validate_report_envelope(
        report,
        pillar=PILLAR,
        report_schema=REPORT_SCHEMA,
        canonical_role=EVIDENCE_ROLE,
        canonical_error=canonical_p4_backend_error(backend),
        backend_identity=backend,
    )
    if (
        report["claim_scope"] != CLAIM_SCOPE
        or report["diagnostic_mode"] not in DIAGNOSTIC_MODES
        or isinstance(report["admission_threshold"], bool)
        or not isinstance(report["admission_threshold"], (int, float))
        or not math.isfinite(float(report["admission_threshold"]))
        or not -1 <= float(report["admission_threshold"]) <= 1
        or type(report["locked_test_rows_read"]) is not int
        or report["locked_test_rows_read"] != 0
        or type(report["recovery_triggers_emitted"]) is not int
        or report["recovery_triggers_emitted"] != 0
        or type(report["memory_queries_executed"]) is not int
        or report["memory_queries_executed"] != len(report["records"])
    ):
        raise SchemaError("P4 report scientific boundary changed")
    integrity = report["store_integrity"]
    if not isinstance(integrity, Mapping) or set(integrity) != {
        "directory_sha256_before", "directory_sha256_after", "file_count", "unchanged",
        "write_enabled",
    }:
        raise SchemaError("P4 store-integrity evidence changed")
    if (
        integrity["unchanged"] is not True
        or integrity["write_enabled"] is not False
        or integrity["directory_sha256_before"] != integrity["directory_sha256_after"]
    ):
        raise SchemaError("P4 report does not prove an unchanged read-only store")
    require_sha256(integrity["directory_sha256_before"], context="P4 store closure")
    if type(integrity["file_count"]) is not int or integrity["file_count"] <= 0:
        raise SchemaError("P4 store-integrity file count is invalid")
    statistics = report["statistics"]
    if not isinstance(statistics, Mapping) or set(statistics) != {
        "bootstrap_samples", "bootstrap_confidence", "bootstrap_seed", "cluster_unit"
    } or (
        statistics["bootstrap_confidence"] != 0.95
        or statistics["cluster_unit"] != "task_id"
        or type(statistics["bootstrap_samples"]) is not int
        or statistics["bootstrap_samples"] <= 0
        or type(statistics["bootstrap_seed"]) is not int
        or statistics["bootstrap_seed"] < 0
    ):
        raise SchemaError("P4 report statistics registration changed")
    records = report["records"]
    if not isinstance(records, list) or not records:
        raise SchemaError("P4 report requires query records")
    for row in records:
        require_exact_keys(
            row,
            {
                "record_type", "example_id", "task_id", "episode_id", "incident_id",
                "query_stage", "query_embedding_sha256", "embedding_request_sha256",
                "processed_batch_sha256", "post_failure_observation_id",
                "failed_action_id", "duplicate_cluster_ids", "source_record_sha256",
                "target", "shadow_strategy", "prediction", "prediction_sha256",
                "admitted_candidate_id", "admitted_strategy", "intervention_enabled",
                "admitted_relevant",
                "final_strategy", "changed_strategy", "strategy_hit_at_1",
                "strategy_hit_at_3", "retrieval_reciprocal_rank", "irrelevant_fraction",
            },
            context="P4 query record",
        )
        if row["record_type"] != "post_failure_memory_query" or (
            row["query_stage"] != "post_action_failure"
        ):
            raise SchemaError("P4 query record stage changed")
        for field in (
            "example_id", "task_id", "episode_id", "incident_id",
            "post_failure_observation_id", "failed_action_id",
        ):
            require_text(row[field], context=f"P4 report {field}")
        for field in (
            "query_embedding_sha256", "embedding_request_sha256",
            "processed_batch_sha256", "source_record_sha256",
        ):
            require_sha256(row[field], context=f"P4 report {field}")
        if not isinstance(row["duplicate_cluster_ids"], list) or not row["duplicate_cluster_ids"]:
            raise SchemaError("P4 report duplicate clusters changed")
        require_exact_keys(
            row["target"], {"target_strategy", "relevant_memory_ids"}, context="P4 target"
        )
        if row["target"]["target_strategy"] not in RECOVERY_STRATEGY:
            raise SchemaError("P4 target strategy changed")
        relevant = row["target"]["relevant_memory_ids"]
        if not isinstance(relevant, list) or len(relevant) != len(set(relevant)):
            raise SchemaError("P4 relevance evidence changed")
        prediction = dict(row["prediction"])
        prediction["candidates"] = tuple(
            P4Candidate(**candidate) for candidate in prediction["candidates"]
        )
        typed = P4RetrievalPrediction(**prediction)
        if (
            typed.query_embedding_sha256 != row["query_embedding_sha256"]
            or typed.store_manifest_sha256 != backend.memory_manifest_sha256
            or not math.isclose(
                float(typed.admission_threshold),
                float(report["admission_threshold"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or (
            row["prediction_sha256"] != sha256_json(asdict(typed))
            )
        ):
            raise SchemaError("P4 prediction/query binding changed")
        view = P4QueryView(
            row["example_id"], row["task_id"], row["episode_id"], row["incident_id"],
            row["post_failure_observation_id"], row["failed_action_id"],
            tuple(row["duplicate_cluster_ids"]),
            tuple([1.0] + [0.0] * 767), float32_vector_sha256([1.0] + [0.0] * 767),
            "1" * 64, "2" * 64, backend.checkpoint_sha256,
            backend.processor_contract_sha256, row["shadow_strategy"],
        )
        _validate_no_leakage(view, typed)
        admitted = next(
            (
                candidate
                for candidate in typed.candidates
                if candidate.similarity >= typed.admission_threshold
            ),
            None,
        )
        expected_enabled = report["diagnostic_mode"] == "PRIMARY_INTERVENTION"
        expected_final = row["shadow_strategy"]
        if expected_enabled and admitted is not None and admitted.strategy != "NONE":
            expected_final = admitted.strategy
        relevant_ids = set(relevant)
        expected_rr = next(
            (
                1.0 / candidate.rank
                for candidate in typed.candidates
                if candidate.memory_id in relevant_ids
            ),
            0.0,
        )
        expected_irrelevant = (
            None if not typed.candidates else
            sum(candidate.memory_id not in relevant_ids for candidate in typed.candidates)
            / len(typed.candidates)
        )
        expected = {
            "admitted_candidate_id": None if admitted is None else admitted.memory_id,
            "admitted_strategy": None if admitted is None else admitted.strategy,
            "admitted_relevant": (
                None if admitted is None else admitted.memory_id in relevant_ids
            ),
            "intervention_enabled": expected_enabled,
            "final_strategy": expected_final,
            "changed_strategy": expected_final != row["shadow_strategy"],
            "strategy_hit_at_1": bool(
                typed.candidates
                and typed.candidates[0].strategy
                == row["target"]["target_strategy"]
            ),
            "strategy_hit_at_3": any(
                candidate.strategy == row["target"]["target_strategy"]
                for candidate in typed.candidates
            ),
            "retrieval_reciprocal_rank": expected_rr,
            "irrelevant_fraction": expected_irrelevant,
        }
        if any(row[field] != value for field, value in expected.items()):
            raise SchemaError("P4 derived admission/intervention evidence changed")
    recomputed = _summary(
        records,
        bootstrap_samples=statistics["bootstrap_samples"],
        bootstrap_seed=statistics["bootstrap_seed"],
    )
    if report["summary"] != recomputed:
        raise SchemaError("P4 diagnostic summary is not reproducible from records")
    validate_metric_tree(report["summary"])


def write_pillar4_diagnostic_package(
    output_dir: str | Path,
    *, diagnostic: Pillar4DiagnosticInput,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    validate_pillar4_diagnostic_report(report)
    _validate_p4_report_input_binding(diagnostic, report)
    return write_package(
        output_dir,
        pillar=PILLAR,
        diagnostic=diagnostic,
        report=report,
        input_filename=INPUT_FILENAME,
        report_filename=REPORT_FILENAME,
    )


def validate_pillar4_diagnostic_package(output_dir: str | Path) -> dict[str, Any]:
    result = validate_package(
        output_dir,
        pillar=PILLAR,
        input_filename=INPUT_FILENAME,
        report_filename=REPORT_FILENAME,
        load_input=Pillar4DiagnosticInput.from_mapping,
        validate_report=validate_pillar4_diagnostic_report,
    )
    root = Path(output_dir)
    diagnostic = Pillar4DiagnosticInput.from_mapping(
        load_unique_json(root / INPUT_FILENAME)
    )
    _validate_p4_report_input_binding(diagnostic, load_unique_json(root / REPORT_FILENAME))
    return result


def _validate_p4_report_input_binding(
    diagnostic: Pillar4DiagnosticInput, report: Mapping[str, Any]
) -> None:
    records = report["records"]
    by_id = {row["example_id"]: row for row in records}
    if len(by_id) != len(diagnostic.examples) or set(by_id) != {
        row.example_id for row in diagnostic.examples
    }:
        raise CompanionDiagnosticError("P4 report/input query inventory changed")
    if not math.isclose(
        float(report["admission_threshold"]),
        float(diagnostic.admission_threshold),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or report["diagnostic_mode"] != diagnostic.diagnostic_mode:
        raise CompanionDiagnosticError("P4 report/input retrieval registration changed")
    for example in diagnostic.examples:
        row = by_id[example.example_id]
        expected = {
            "task_id": example.task_id,
            "episode_id": example.episode_id,
            "incident_id": example.incident_id,
            "query_stage": example.query_stage,
            "query_embedding_sha256": example.query_embedding_sha256,
            "embedding_request_sha256": example.embedding_request_sha256,
            "processed_batch_sha256": example.processed_batch_sha256,
            "post_failure_observation_id": example.post_failure_observation_id,
            "failed_action_id": example.failed_action_id,
            "duplicate_cluster_ids": list(example.duplicate_cluster_ids),
            "source_record_sha256": example.source_record_sha256,
            "shadow_strategy": example.shadow_strategy,
            "target": {
                "target_strategy": example.target_strategy,
                "relevant_memory_ids": list(example.relevant_memory_ids),
            },
        }
        if any(row[field] != value for field, value in expected.items()):
            raise CompanionDiagnosticError("P4 report/input query evidence changed")
