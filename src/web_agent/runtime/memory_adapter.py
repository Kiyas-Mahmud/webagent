"""Post-failure-only, immutable P4 retrieval and E3 intervention.

The runtime owns only the causal query contract and an immutable read bridge.
Building, calibrating, or mutating a memory store is deliberately outside this
module and cannot be reached through any object defined here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
import random
from threading import Lock
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from web_agent.runtime.contracts import (
    MEMORY_QUERY_VECTOR_DIMENSION,
    MemoryCandidate,
    MemoryQuery,
    MemoryQueryResult,
    RecoveryDecision,
    RecoveryStrategy,
    TransitionInput,
    VersionedRecord,
    canonical_sha256,
    float32_vector_sha256,
)
from web_agent.runtime.protocol import (
    REGISTERED_MEMORY_EMBEDDING_DIMENSION,
    REGISTERED_MEMORY_EMBEDDING_STAGE,
    REGISTERED_MEMORY_TOP_K,
    SystemSwitches,
)
from web_agent.runtime.model_calls import record_model_call

if TYPE_CHECKING:
    from web_agent.memory.frozen_store import FrozenMemoryStore


class MemoryBoundaryError(RuntimeError):
    """A P4 provenance, causality, or immutability boundary was violated."""


REGISTERED_FROZEN_STORE_READER_ID = "frozen-memory-store-v1"


def _float32_sha256(values: Sequence[float]) -> str:
    """Hash the exact little-endian float32 bytes used by cosine retrieval."""

    return float32_vector_sha256(values)


def _freeze_json(value: Any) -> Any:
    """Copy JSON-like model input into recursively immutable containers."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _immutable_transition(value: TransitionInput) -> TransitionInput:
    return replace(
        value,
        pre_observation=replace(
            value.pre_observation,
            current_page_state=_freeze_json(value.pre_observation.current_page_state),
        ),
        executed_action=replace(
            value.executed_action,
            parameters=_freeze_json(value.executed_action.parameters),
        ),
        post_observation=replace(
            value.post_observation,
            current_page_state=_freeze_json(value.post_observation.current_page_state),
        ),
    )


@dataclass(frozen=True, slots=True)
class PostFailureEmbeddingRequest(VersionedRecord):
    """Exact immutable P4 input handed to the selected-checkpoint backend."""

    query_id: str
    post_failure_observation_id: str
    failed_action_id: str
    post_action_input: TransitionInput
    post_failure_observation_sha256: str
    post_action_input_sha256: str
    processor_contract_sha256: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "query_id",
            "post_failure_observation_id",
            "failed_action_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"post-failure embedding request requires {name}")
        if not isinstance(self.post_action_input, TransitionInput):
            raise TypeError("post-failure embedding request requires TransitionInput")
        immutable = _immutable_transition(self.post_action_input)
        object.__setattr__(self, "post_action_input", immutable)
        if immutable.post_observation.observation_id != self.post_failure_observation_id:
            raise ValueError("embedding request cites another post-failure observation")
        if immutable.executed_action.action_id != self.failed_action_id:
            raise ValueError("embedding request cites another failed action")
        for name in (
            "post_failure_observation_sha256",
            "post_action_input_sha256",
            "processor_contract_sha256",
            "checkpoint_sha256",
        ):
            if not _lowercase_sha256(str(getattr(self, name))):
                raise ValueError(f"post-failure embedding request {name} must be SHA-256")
        if self.post_action_input_sha256 != immutable.record_sha256:
            raise ValueError(
                "embedding request hash differs from the exact post-action input"
            )


@dataclass(frozen=True, slots=True)
class PostFailureEmbedding(VersionedRecord):
    """Backend-created evidence for one exact P4 forward call.

    The selected-checkpoint backend, not the runtime wrapper, must create this
    receipt. Runtime independently verifies every identity and both byte hashes.
    """

    query_id: str
    post_failure_observation_id: str
    values: tuple[float, ...]
    request_sha256: str
    post_failure_observation_sha256: str
    post_action_input_sha256: str
    processor_contract_sha256: str
    checkpoint_sha256: str
    processed_batch_sha256: str
    embedding_sha256: str
    embedding_stage: str = REGISTERED_MEMORY_EMBEDDING_STAGE

    def __post_init__(self) -> None:
        if not self.query_id or not self.post_failure_observation_id:
            raise ValueError("post-failure embedding requires query/observation IDs")
        if self.embedding_stage != REGISTERED_MEMORY_EMBEDDING_STAGE:
            raise ValueError(
                "memory embedding must come from the exact post-action memory "
                "task-adapter stage"
            )
        values = tuple(float(value) for value in self.values)
        object.__setattr__(self, "values", values)
        if len(values) != REGISTERED_MEMORY_EMBEDDING_DIMENSION:
            raise ValueError(
                "memory embedding must contain exactly "
                f"{REGISTERED_MEMORY_EMBEDDING_DIMENSION} values"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("memory embedding contains non-finite values")
        norm = math.hypot(*values)
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("memory embedding must have non-zero finite norm")
        for name in (
            "post_failure_observation_sha256",
            "post_action_input_sha256",
            "processor_contract_sha256",
            "checkpoint_sha256",
            "request_sha256",
            "processed_batch_sha256",
            "embedding_sha256",
        ):
            value = str(getattr(self, name))
            if not _lowercase_sha256(value):
                raise ValueError(f"post-failure embedding {name} must be SHA-256")
        if self.embedding_sha256 != _float32_sha256(values):
            raise ValueError("post-failure embedding byte hash differs from values")


class PostFailureEmbeddingProvider(ABC):
    """Injected adapter to the frozen model's exact post-action P4 tensor."""

    embedding_stage: str = REGISTERED_MEMORY_EMBEDDING_STAGE
    embedding_dimension: int = REGISTERED_MEMORY_EMBEDDING_DIMENSION
    frozen: bool = True
    write_enabled: bool = False

    @abstractmethod
    def embed(self, request: PostFailureEmbeddingRequest) -> PostFailureEmbedding:
        raise NotImplementedError


EmbeddingCallable = Callable[[PostFailureEmbeddingRequest], PostFailureEmbedding]


@dataclass(frozen=True, slots=True)
class CallablePostFailureEmbeddingProvider(PostFailureEmbeddingProvider):
    """Validate evidence returned by the selected-checkpoint backend itself."""

    embedder: EmbeddingCallable
    provider_id: str
    provider_version: str
    checkpoint_sha256: str
    processor_contract_sha256: str
    embedding_stage: str = REGISTERED_MEMORY_EMBEDDING_STAGE
    embedding_dimension: int = REGISTERED_MEMORY_EMBEDDING_DIMENSION
    frozen: bool = True
    write_enabled: bool = False
    _batch_by_embedding: dict[str, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _embedding_by_batch: dict[str, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _audit_lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.provider_id or not self.provider_version:
            raise ValueError("embedding provider identity/version are required")
        if self.embedding_stage != REGISTERED_MEMORY_EMBEDDING_STAGE:
            raise ValueError("embedding provider uses an unregistered stage")
        if self.embedding_dimension != REGISTERED_MEMORY_EMBEDDING_DIMENSION:
            raise ValueError("embedding provider uses an unregistered dimension")
        if not self.frozen or self.write_enabled:
            raise ValueError("embedding provider must be frozen and read-only")
        for name in ("checkpoint_sha256", "processor_contract_sha256"):
            if not _lowercase_sha256(str(getattr(self, name))):
                raise ValueError(f"embedding provider {name} must be SHA-256")

    def embed(self, request: PostFailureEmbeddingRequest) -> PostFailureEmbedding:
        if not isinstance(request, PostFailureEmbeddingRequest):
            raise MemoryBoundaryError("embedding provider requires exact post-action input")
        if request.checkpoint_sha256 != self.checkpoint_sha256:
            raise MemoryBoundaryError("embedding request cites another checkpoint")
        if request.processor_contract_sha256 != self.processor_contract_sha256:
            raise MemoryBoundaryError("embedding request cites another processor")
        request_sha256 = request.record_sha256
        record_model_call(
            stage="memory_embedding",
            component_id=f"{self.provider_id}@{self.provider_version}",
        )
        receipt = self.embedder(request)
        if request.record_sha256 != request_sha256:
            raise MemoryBoundaryError("embedding backend mutated its immutable request")
        if type(receipt) is not PostFailureEmbedding:
            raise MemoryBoundaryError(
                "embedding backend must return its own typed forward-call evidence"
            )
        expected = {
            "query_id": request.query_id,
            "post_failure_observation_id": request.post_failure_observation_id,
            "request_sha256": request_sha256,
            "post_failure_observation_sha256": (
                request.post_failure_observation_sha256
            ),
            "post_action_input_sha256": request.post_action_input_sha256,
            "processor_contract_sha256": self.processor_contract_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "embedding_stage": self.embedding_stage,
        }
        if any(getattr(receipt, name) != value for name, value in expected.items()):
            raise MemoryBoundaryError(
                "embedding backend evidence differs from the exact selected-model request"
            )

        # Exact float32 reuse across different processed batches is a stale or
        # constant-output canary. Conversely, one exact batch must be deterministic.
        with self._audit_lock:
            prior_batch = self._batch_by_embedding.get(receipt.embedding_sha256)
            if prior_batch is not None and prior_batch != receipt.processed_batch_sha256:
                raise MemoryBoundaryError(
                    "embedding bytes were reused across different processed batches"
                )
            prior_embedding = self._embedding_by_batch.get(
                receipt.processed_batch_sha256
            )
            if prior_embedding is not None and prior_embedding != receipt.embedding_sha256:
                raise MemoryBoundaryError(
                    "identical processed batch produced nondeterministic embedding bytes"
                )
            self._batch_by_embedding[receipt.embedding_sha256] = (
                receipt.processed_batch_sha256
            )
            self._embedding_by_batch[receipt.processed_batch_sha256] = (
                receipt.embedding_sha256
            )
        return receipt


@dataclass(frozen=True, slots=True)
class ReaderQueryResult(VersionedRecord):
    """Immutable reader output, including the complete leakage-filter audit."""

    candidates: tuple[MemoryCandidate, ...]
    exclusion_reasons: Mapping[str, str]
    considered_count: int
    eligible_count: int
    query_embedding_sha256: str | None = None
    embedding_binding_sha256: str | None = None
    embedding_request_sha256: str | None = None
    processed_batch_sha256: str | None = None
    normalized_query_embedding: tuple[float, ...] | None = None
    normalized_query_embedding_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.considered_count) is not int or type(self.eligible_count) is not int:
            raise ValueError("memory reader counts must be exact integers")
        if self.considered_count < 0 or self.eligible_count < 0:
            raise ValueError("memory reader counts cannot be negative")
        if self.eligible_count > self.considered_count:
            raise ValueError("eligible memory count exceeds considered count")
        if len(self.candidates) > self.eligible_count:
            raise ValueError("reader returned more candidates than eligible items")
        if self.query_embedding_sha256 is not None and (
            len(self.query_embedding_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.query_embedding_sha256
            )
        ):
            raise ValueError(
                "query embedding hash must be lowercase hexadecimal SHA-256"
            )
        if self.embedding_binding_sha256 is not None and not _lowercase_sha256(
            self.embedding_binding_sha256
        ):
            raise ValueError("embedding binding hash must be lowercase SHA-256")
        for name in ("embedding_request_sha256", "processed_batch_sha256"):
            value = getattr(self, name)
            if value is not None and not _lowercase_sha256(value):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if (self.embedding_request_sha256 is None) != (
            self.processed_batch_sha256 is None
        ):
            raise ValueError("embedding request/batch hashes must be supplied together")
        if (self.normalized_query_embedding is None) != (
            self.normalized_query_embedding_sha256 is None
        ):
            raise ValueError(
                "normalized query embedding/vector hash must be supplied together"
            )
        if self.normalized_query_embedding is not None:
            values = tuple(float(value) for value in self.normalized_query_embedding)
            object.__setattr__(self, "normalized_query_embedding", values)
            if len(values) != MEMORY_QUERY_VECTOR_DIMENSION:
                raise ValueError(
                    "normalized query embedding must contain exactly "
                    f"{MEMORY_QUERY_VECTOR_DIMENSION} values"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("normalized query embedding contains non-finite values")
            if not math.isclose(
                math.hypot(*values),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-5,
            ):
                raise ValueError("normalized query embedding is not L2-normalized")
            assert self.normalized_query_embedding_sha256 is not None
            if not _lowercase_sha256(self.normalized_query_embedding_sha256):
                raise ValueError(
                    "normalized query embedding hash must be lowercase SHA-256"
                )
            if (
                _float32_sha256(values)
                != self.normalized_query_embedding_sha256
            ):
                raise ValueError(
                    "normalized query embedding hash differs from float32 values"
                )


class FrozenMemoryReader(ABC):
    reader_id: str
    index_sha256: str
    model_seed: int | None = None
    registered_admission_threshold: float | None = None
    store_manifest_sha256: str | None = None
    evidence_scope: str = "UNREGISTERED"
    frozen: bool = True
    write_enabled: bool = False

    @abstractmethod
    def query(
        self,
        query: MemoryQuery,
        *,
        k: int,
        rng: random.Random,
        embedding_request: PostFailureEmbeddingRequest | None = None,
    ) -> ReaderQueryResult:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class InMemoryFrozenReader(FrozenMemoryReader):
    """Small deterministic reader for fixtures and already-scored candidates."""

    evidence_scope: ClassVar[str] = "ENGINEERING_SMOKE_ONLY"
    store_manifest_sha256: ClassVar[str | None] = None

    reader_id: str
    index_sha256: str
    candidates: tuple[MemoryCandidate, ...]
    model_seed: int | None = None
    registered_admission_threshold: float | None = None
    frozen: bool = True
    write_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.reader_id or len(self.index_sha256) != 64:
            raise ValueError("frozen reader identity/hash is invalid")
        if not self.frozen or self.write_enabled:
            raise ValueError("primary memory reader must be frozen and read-only")
        if self.registered_admission_threshold is not None and not (
            -1.0 <= self.registered_admission_threshold <= 1.0
        ):
            raise ValueError("registered admission threshold must be in [-1, 1]")

    def query(
        self,
        query: MemoryQuery,
        *,
        k: int,
        rng: random.Random,
        embedding_request: PostFailureEmbeddingRequest | None = None,
    ) -> ReaderQueryResult:
        del query, rng, embedding_request
        ordered = tuple(
            sorted(
                self.candidates,
                key=lambda item: (-item.similarity, item.memory_id),
            )[:k]
        )
        return ReaderQueryResult(
            candidates=ordered,
            exclusion_reasons={},
            considered_count=len(self.candidates),
            eligible_count=len(self.candidates),
        )


class FrozenStoreMemoryReader(FrozenMemoryReader):
    """Read-only bridge from runtime queries to ``FrozenMemoryStore.query``.

    It requests exactly one embedding for the cited post-failure observation,
    validates the 768-dimensional float32 vector, constructs every registered
    leakage exclusion, and maps store hits into runtime contracts. It has no
    build/save/update method and never exposes the store's backing arrays.
    """

    __slots__ = (
        "_store",
        "_embedding_provider",
        "reader_id",
        "index_sha256",
        "store_manifest_sha256",
        "model_seed",
        "registered_admission_threshold",
    )

    evidence_scope = "EVALUATION_ATTESTED"
    frozen = True
    write_enabled = False

    def __init__(
        self,
        store: FrozenMemoryStore,
        *,
        embedding_provider: PostFailureEmbeddingProvider,
        reader_id: str = REGISTERED_FROZEN_STORE_READER_ID,
        expected_model_seed: int | None = None,
    ) -> None:
        # Lazy import: E0-E2 and fixture-only runs need no numerical stack.
        from web_agent.memory.frozen_store import FrozenMemoryStore as StoreType

        if not isinstance(store, StoreType):
            raise TypeError("FrozenStoreMemoryReader requires FrozenMemoryStore")
        if not reader_id:
            raise ValueError("frozen store reader_id is required")
        if embedding_provider.embedding_stage != REGISTERED_MEMORY_EMBEDDING_STAGE:
            raise MemoryBoundaryError("embedding provider stage is not registered")
        if embedding_provider.embedding_dimension != REGISTERED_MEMORY_EMBEDDING_DIMENSION:
            raise MemoryBoundaryError("embedding provider dimension is not registered")
        if not embedding_provider.frozen or embedding_provider.write_enabled:
            raise MemoryBoundaryError("embedding provider must be frozen/read-only")
        if expected_model_seed is not None and store.model_seed != expected_model_seed:
            raise MemoryBoundaryError(
                "frozen memory store seed differs from the matched model seed"
            )
        self._store = store
        self._embedding_provider = embedding_provider
        self.reader_id = reader_id
        self.index_sha256 = store.manifest_sha256
        self.store_manifest_sha256 = store.manifest_sha256
        self.model_seed = store.model_seed
        self.registered_admission_threshold = store.admission_threshold

    def query(
        self,
        query: MemoryQuery,
        *,
        k: int,
        rng: random.Random,
        embedding_request: PostFailureEmbeddingRequest | None = None,
    ) -> ReaderQueryResult:
        del rng  # Frozen cosine retrieval has a deterministic tie-break.
        if k != REGISTERED_MEMORY_TOP_K:
            raise MemoryBoundaryError(
                f"frozen store top-k is registered as {REGISTERED_MEMORY_TOP_K}"
            )
        if not query.duplicate_cluster_ids or any(
            not str(cluster_id).strip() for cluster_id in query.duplicate_cluster_ids
        ):
            raise MemoryBoundaryError(
                "frozen-store query needs non-empty audited duplicate clusters"
            )

        if embedding_request is None:
            raise MemoryBoundaryError(
                "frozen-store query lacks its exact post-action embedding request"
            )
        request_bindings = {
            "query_id": query.query_id,
            "post_failure_observation_id": query.post_failure_observation_id,
            "failed_action_id": query.failed_action_id,
            "post_failure_observation_sha256": (
                query.post_failure_observation_sha256
            ),
            "post_action_input_sha256": query.post_action_input_sha256,
            "processor_contract_sha256": query.processor_contract_sha256,
            "checkpoint_sha256": query.checkpoint_sha256,
        }
        if any(
            getattr(embedding_request, name) != value
            for name, value in request_bindings.items()
        ):
            raise MemoryBoundaryError(
                "embedding request differs from its current post-failure query"
            )
        if embedding_request.post_action_input.task_id != query.task_id:
            raise MemoryBoundaryError("embedding request transition belongs to another task")

        embedding = self._embedding_provider.embed(embedding_request)
        if not isinstance(embedding, PostFailureEmbedding):
            raise MemoryBoundaryError(
                "embedding provider returned an invalid post-failure contract"
            )
        if (
            embedding.query_id != query.query_id
            or embedding.post_failure_observation_id
            != query.post_failure_observation_id
        ):
            raise MemoryBoundaryError(
                "embedding does not cite the current post-failure query/observation"
            )
        binding_fields = (
            "post_failure_observation_sha256",
            "post_action_input_sha256",
            "processor_contract_sha256",
            "checkpoint_sha256",
        )
        if any(getattr(query, name) is None for name in binding_fields):
            raise MemoryBoundaryError(
                "frozen-store query lacks observation/input/processor/checkpoint binding"
            )
        if any(
            getattr(embedding, name) != getattr(query, name)
            for name in binding_fields
        ):
            raise MemoryBoundaryError(
                "embedding receipt differs from its causal model-input binding"
            )
        store_checkpoint = str(self._store.manifest.get("checkpoint_sha256") or "")
        if query.checkpoint_sha256 != store_checkpoint:
            raise MemoryBoundaryError(
                "embedding query checkpoint differs from the frozen memory store"
            )

        import numpy as np
        from web_agent.memory.frozen_store import QueryExclusions

        vector = np.asarray(embedding.values, dtype=np.float32)
        if vector.shape != (REGISTERED_MEMORY_EMBEDDING_DIMENSION,):
            raise MemoryBoundaryError("post-failure embedding shape changed after casting")
        if not np.isfinite(vector).all():
            raise MemoryBoundaryError("post-failure embedding contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise MemoryBoundaryError("post-failure embedding has zero/non-finite norm")
        embedding_sha256 = _float32_sha256(vector.tolist())
        if embedding.embedding_sha256 != embedding_sha256:
            raise MemoryBoundaryError(
                "embedding backend receipt differs from retrieval float32 bytes"
            )
        normalized = np.asarray(vector / norm, dtype=np.float32)
        normalized_sha256 = _float32_sha256(normalized.tolist())
        embedding_binding_sha256 = canonical_sha256(
            {
                "query_id": query.query_id,
                "post_failure_observation_id": query.post_failure_observation_id,
                **{name: getattr(query, name) for name in binding_fields},
                "embedding_request_sha256": embedding.request_sha256,
                "processed_batch_sha256": embedding.processed_batch_sha256,
                "embedding_stage": embedding.embedding_stage,
                "embedding_sha256": embedding_sha256,
                "normalized_embedding_sha256": normalized_sha256,
            }
        )

        raw = self._store.query(
            normalized,
            exclusions=QueryExclusions(
                current_task_id=query.task_id,
                current_episode_id=query.episode_id,
                duplicate_cluster_ids=frozenset(query.duplicate_cluster_ids),
            ),
        )
        candidates = tuple(self._candidate_from_hit(hit) for hit in raw.hits)
        return ReaderQueryResult(
            candidates=candidates,
            exclusion_reasons=dict(raw.exclusion_reasons),
            considered_count=raw.considered_count,
            eligible_count=raw.eligible_count,
            query_embedding_sha256=embedding_sha256,
            embedding_binding_sha256=embedding_binding_sha256,
            embedding_request_sha256=embedding.request_sha256,
            processed_batch_sha256=embedding.processed_batch_sha256,
            normalized_query_embedding=tuple(float(value) for value in normalized),
            normalized_query_embedding_sha256=normalized_sha256,
        )

    @staticmethod
    def _candidate_from_hit(hit: Any) -> MemoryCandidate:
        item = hit.item

        def required_text(name: str) -> str:
            value = str(item.get(name) or "").strip()
            if not value:
                raise MemoryBoundaryError(
                    f"frozen memory hit {hit.memory_id!r} lacks {name}"
                )
            return value

        try:
            strategy = RecoveryStrategy(required_text("strategy"))
        except ValueError as exc:
            raise MemoryBoundaryError(
                f"frozen memory hit {hit.memory_id!r} has unknown strategy"
            ) from exc
        reflection = str(item.get("reflection_text") or "").strip()
        executed = required_text("executed_recovery_action")
        memory_id = required_text("memory_id")
        if memory_id != str(hit.memory_id):
            raise MemoryBoundaryError("frozen hit ID differs from its item provenance")
        return MemoryCandidate(
            memory_id=memory_id,
            source_split=required_text("source_split"),
            source_task_id=required_text("source_task_id"),
            source_episode_id=required_text("source_episode_id"),
            duplicate_cluster_id=required_text("duplicate_cluster_id"),
            strategy=strategy,
            similarity=float(hit.cosine_similarity),
            memory_update_flag=item.get("memory_update_flag") is True,
            verified_recovery_success=item.get("verified_recovery_success") is True,
            final_task_success=item.get("final_task_success") is True,
            advice=reflection or f"registered recovery action: {executed}",
        )


@dataclass(frozen=True, slots=True)
class MemoryDecision(VersionedRecord):
    shadow_decision: RecoveryDecision
    final_decision: RecoveryDecision
    query_result: MemoryQueryResult


class MemoryAdapter:
    def __init__(
        self,
        switches: SystemSwitches,
        *,
        reader: FrozenMemoryReader | None,
        top_k: int = REGISTERED_MEMORY_TOP_K,
        admission_threshold: float,
        evaluation_mode: bool = False,
        expected_store_manifest_sha256: str | None = None,
    ) -> None:
        if type(evaluation_mode) is not bool:
            raise MemoryBoundaryError("memory evaluation_mode must be an exact boolean")
        if top_k != REGISTERED_MEMORY_TOP_K:
            raise ValueError(
                f"primary memory top_k is registered as {REGISTERED_MEMORY_TOP_K}"
            )
        if not -1.0 <= admission_threshold <= 1.0:
            raise ValueError("cosine admission threshold must be in [-1, 1]")
        if switches.memory_query:
            if reader is None:
                raise MemoryBoundaryError("E3 requires a frozen memory reader")
            if not reader.frozen or reader.write_enabled:
                raise MemoryBoundaryError("E3 memory reader must be immutable/read-only")
            registered_threshold = reader.registered_admission_threshold
            if registered_threshold is not None and not math.isclose(
                admission_threshold,
                registered_threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise MemoryBoundaryError(
                    "runtime admission threshold differs from frozen store manifest"
                )
        elif reader is not None:
            raise MemoryBoundaryError(
                f"{switches.system_id.value} must not receive a memory reader"
            )
        if evaluation_mode:
            if not switches.memory_query or reader is None:
                raise MemoryBoundaryError(
                    "evaluation memory binding is valid only for E3"
                )
            if type(reader) is not FrozenStoreMemoryReader:
                raise MemoryBoundaryError(
                    "evaluation E3 requires the exact attested FrozenStoreMemoryReader; "
                    "fixture readers are ENGINEERING_SMOKE_ONLY"
                )
            if reader.reader_id != REGISTERED_FROZEN_STORE_READER_ID:
                raise MemoryBoundaryError(
                    "evaluation E3 reader identity differs from the registered reader"
                )
            if (
                expected_store_manifest_sha256 is None
                or len(expected_store_manifest_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_store_manifest_sha256
                )
            ):
                raise MemoryBoundaryError(
                    "evaluation E3 requires an attested memory manifest SHA-256"
                )
            if reader.store_manifest_sha256 != expected_store_manifest_sha256:
                raise MemoryBoundaryError(
                    "evaluation E3 reader differs from the attested memory store"
                )
        elif expected_store_manifest_sha256 is not None:
            raise MemoryBoundaryError(
                "fixture/smoke memory cannot claim an evaluation store attestation"
            )
        self.switches = switches
        self.reader = reader
        self.top_k = top_k
        self.admission_threshold = admission_threshold
        self.evaluation_mode = evaluation_mode
        self.expected_store_manifest_sha256 = expected_store_manifest_sha256

    def assert_model_seed(self, model_seed: int) -> None:
        """Fail closed if an E3 reader belongs to another matched seed."""
        if self.reader is None:
            raise MemoryBoundaryError("memory reader is unavailable")
        if self.reader.model_seed is not None and self.reader.model_seed != model_seed:
            raise MemoryBoundaryError(
                "frozen memory reader seed differs from the episode model seed"
            )

    def apply_post_failure(
        self,
        *,
        query: MemoryQuery,
        shadow_decision: RecoveryDecision,
        rng: random.Random,
        embedding_request: PostFailureEmbeddingRequest | None = None,
    ) -> MemoryDecision:
        """Apply memory only after the E2-equivalent shadow has been fixed."""
        if not self.switches.memory_query or not self.switches.memory_intervention:
            raise MemoryBoundaryError(
                f"memory query is forbidden for {self.switches.system_id.value}"
            )
        if query.incident_id != shadow_decision.incident_id:
            raise MemoryBoundaryError("memory query and shadow decision incidents differ")
        if not query.post_failure_observation_id or not query.failed_action_id:
            raise MemoryBoundaryError("memory query is not a post-failure query")
        if not query.duplicate_cluster_ids or any(
            not str(cluster_id).strip() for cluster_id in query.duplicate_cluster_ids
        ):
            raise MemoryBoundaryError(
                "memory query requires non-empty audited duplicate-cluster evidence"
            )
        shadow_sha256 = shadow_decision.sha256
        assert self.reader is not None
        start = perf_counter()
        if embedding_request is None:
            # Engineering-smoke readers predate the production-only model-input
            # capability and never perform a model embedding call.
            reader_result = self.reader.query(query, k=self.top_k, rng=rng)
        else:
            reader_result = self.reader.query(
                query,
                k=self.top_k,
                rng=rng,
                embedding_request=embedding_request,
            )
        if not isinstance(reader_result, ReaderQueryResult):
            raise MemoryBoundaryError("memory reader returned an invalid query contract")
        if self.evaluation_mode and reader_result.query_embedding_sha256 is None:
            raise MemoryBoundaryError(
                "evaluation E3 query lacks the exact post-failure embedding digest"
            )
        if self.evaluation_mode and reader_result.embedding_binding_sha256 is None:
            raise MemoryBoundaryError(
                "evaluation E3 query lacks a causal embedding-binding digest"
            )
        if self.evaluation_mode and (
            reader_result.normalized_query_embedding is None
            or reader_result.normalized_query_embedding_sha256 is None
        ):
            raise MemoryBoundaryError(
                "evaluation E3 query lacks the replayable normalized embedding"
            )
        returned = reader_result.candidates
        if len(returned) > self.top_k:
            raise MemoryBoundaryError("memory reader returned more than top_k")
        returned_ids = tuple(item.memory_id for item in returned)
        if len(returned_ids) != len(set(returned_ids)):
            raise MemoryBoundaryError("memory reader returned duplicate candidate IDs")
        registered_order = tuple(
            sorted(returned, key=lambda item: (-float(item.similarity), item.memory_id))
        )
        if tuple(returned) != registered_order:
            raise MemoryBoundaryError(
                "memory reader violated deterministic (-similarity, memory_id) ordering"
            )
        exclusions: dict[str, str] = dict(reader_result.exclusion_reasons)
        eligible: list[MemoryCandidate] = []
        for candidate in returned:
            # Invalid provenance is not a soft filtering choice: the frozen
            # index itself is untrustworthy, so primary evaluation stops.
            try:
                candidate.assert_primary_eligible()
            except ValueError as exc:
                raise MemoryBoundaryError(str(exc)) from exc
            if candidate.source_task_id == query.task_id:
                exclusions[candidate.memory_id] = "same_task"
            elif candidate.source_episode_id == query.episode_id:
                exclusions[candidate.memory_id] = "same_episode"
            elif candidate.duplicate_cluster_id in query.duplicate_cluster_ids:
                exclusions[candidate.memory_id] = "duplicate_cluster"
            else:
                eligible.append(candidate)
        admitted = next(
            (
                candidate
                for candidate in eligible
                if candidate.similarity >= self.admission_threshold
            ),
            None,
        )
        final = shadow_decision
        if admitted is not None and admitted.strategy is not RecoveryStrategy.NONE:
            final = replace(
                shadow_decision,
                decision_id=f"{shadow_decision.decision_id}:memory:{admitted.memory_id}",
                strategy=admitted.strategy,
            )
        result = MemoryQueryResult(
            query_id=query.query_id,
            shadow_decision_sha256=shadow_sha256,
            candidate_ids=tuple(item.memory_id for item in returned),
            scores=tuple(float(item.similarity) for item in returned),
            exclusion_reasons=exclusions,
            admitted_candidate_id=admitted.memory_id if admitted else None,
            admitted=admitted is not None,
            changed_strategy=final.strategy is not shadow_decision.strategy,
            changed_target_or_parameters=False,
            final_strategy=final.strategy,
            latency_ms=(perf_counter() - start) * 1000.0,
            query_embedding_sha256=reader_result.query_embedding_sha256,
            reader_considered_count=reader_result.considered_count,
            reader_eligible_count=reader_result.eligible_count,
            reader_id=self.reader.reader_id,
            store_manifest_sha256=self.reader.store_manifest_sha256,
            embedding_binding_sha256=reader_result.embedding_binding_sha256,
            embedding_request_sha256=reader_result.embedding_request_sha256,
            processed_batch_sha256=reader_result.processed_batch_sha256,
            normalized_query_embedding=reader_result.normalized_query_embedding,
            normalized_query_embedding_sha256=(
                reader_result.normalized_query_embedding_sha256
            ),
        )
        return MemoryDecision(
            shadow_decision=shadow_decision,
            final_decision=final,
            query_result=result,
        )


def _lowercase_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
