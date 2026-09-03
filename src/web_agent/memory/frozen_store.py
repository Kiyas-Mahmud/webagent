"""Read-only, hash-verified Table 2 corrective-memory retrieval.

The runtime class intentionally exposes only ``load`` and ``query``.  Store
creation belongs to :mod:`web_agent.memory.builder`; there is no append, update,
delete, rebuild, or save path available to an evaluation process.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from web_agent.memory.index import deterministic_cosine_top_k
from web_agent.memory.manifest import (
    EMBEDDING_DIMENSION,
    ELIGIBILITY_POLICY_VERSION,
    RUNTIME_TOP_K,
    ManifestError,
    VerifiedStoreManifest,
    canonical_sha256,
    require_sha256,
    verify_store_manifest,
)


def _nonempty(value: object, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ManifestError(f"missing frozen-memory field: {field}")
    return normalized


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class QueryExclusions:
    """Required leakage filters for one post-failure memory query.

    The frozen duplicate audit must map every exact/near duplicate key into one
    or more cluster IDs. Empty/sentinel values are rejected instead of silently
    running a less-protected query.
    """

    current_task_id: str
    current_episode_id: str
    duplicate_cluster_ids: frozenset[str]
    additional_memory_ids: frozenset[str] = frozenset()

    def validated(self) -> "QueryExclusions":
        task = _nonempty(self.current_task_id, field="current_task_id")
        episode = _nonempty(self.current_episode_id, field="current_episode_id")
        clusters = frozenset(
            _nonempty(value, field="query.duplicate_cluster_ids")
            for value in self.duplicate_cluster_ids
        )
        if not clusters:
            raise ManifestError("query.duplicate_cluster_ids cannot be empty")
        ids = frozenset(
            _nonempty(value, field="additional_memory_ids")
            for value in self.additional_memory_ids
        )
        return QueryExclusions(task, episode, clusters, ids)


@dataclass(frozen=True)
class MemoryHit:
    rank: int
    memory_id: str
    cosine_similarity: float
    item: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenQueryResult:
    """Raw top-3 neighbours plus a complete leakage-exclusion audit."""

    hits: tuple[MemoryHit, ...]
    exclusion_reasons: Mapping[str, str]
    considered_count: int
    eligible_count: int


class FrozenMemoryStore:
    """Immutable, per-seed cosine store with deterministic top-3 retrieval."""

    __slots__ = (
        "_root",
        "_embeddings",
        "_items",
        "_manifest",
        "_manifest_sha256",
    )

    def __init__(
        self,
        *,
        root: Path,
        embeddings: np.ndarray,
        items: tuple[Mapping[str, Any], ...],
        verified: VerifiedStoreManifest,
    ) -> None:
        self._root = root
        embeddings.setflags(write=False)
        self._embeddings = embeddings
        self._items = items
        self._manifest = _freeze(dict(verified.payload))
        self._manifest_sha256 = verified.sha256

    @classmethod
    def load(cls, root: str | Path) -> "FrozenMemoryStore":
        directory = Path(root).resolve()
        verified = verify_store_manifest(directory)

        try:
            embeddings = np.load(
                directory / "embeddings.npy",
                allow_pickle=False,
            )
        except (OSError, ValueError) as error:
            raise ManifestError(f"cannot load frozen embeddings: {error}") from error
        if embeddings.dtype != np.float32:
            raise ManifestError("frozen embeddings must use float32")
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ManifestError(
                "frozen embeddings must have shape "
                f"[N, {EMBEDDING_DIMENSION}], got {embeddings.shape}"
            )
        if not np.isfinite(embeddings).all():
            raise ManifestError("frozen embeddings contain non-finite values")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
            raise ManifestError("frozen embeddings are not L2 normalized")

        items: list[Mapping[str, Any]] = []
        duplicate_namespace = verified.payload["duplicate_cluster_namespace"]
        if not isinstance(duplicate_namespace, Mapping):
            raise ManifestError("verified duplicate-cluster namespace is malformed")
        duplicate_namespace_id = _nonempty(
            duplicate_namespace.get("namespace_id"),
            field="manifest.duplicate_cluster_namespace.namespace_id",
        )
        item_path = directory / "items.jsonl"
        for line_number, raw in enumerate(
            item_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw.strip():
                raise ManifestError(f"blank items.jsonl line {line_number}")
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ManifestError(
                    f"invalid items.jsonl line {line_number}: {error}"
                ) from error
            if not isinstance(item, dict):
                raise ManifestError(f"items.jsonl line {line_number} is not an object")
            cls._validate_item(
                item,
                line_number=line_number,
                duplicate_cluster_namespace_id=duplicate_namespace_id,
            )
            items.append(_freeze(item))

        expected_count = int(verified.payload["item_count"])
        if len(items) != expected_count or embeddings.shape[0] != expected_count:
            raise ManifestError(
                "memory item/embedding count mismatch: "
                f"manifest={expected_count}, items={len(items)}, "
                f"embeddings={embeddings.shape[0]}"
            )
        memory_ids = [str(item["memory_id"]) for item in items]
        if memory_ids != sorted(memory_ids) or len(memory_ids) != len(set(memory_ids)):
            raise ManifestError("memory IDs must be unique and lexicographically sorted")

        from web_agent.memory.verification import (
            VerificationEvidenceError,
            verification_manifest_payload,
        )

        try:
            recomputed_verification = verification_manifest_payload(tuple(items))
        except (VerificationEvidenceError, ManifestError) as error:
            raise ManifestError(
                f"frozen item verification-evidence replay failed: {error}"
            ) from error
        if canonical_sha256(recomputed_verification) != verified.payload.get(
            "verification_evidence_sha256"
        ) or recomputed_verification != verified.payload.get("verification_evidence"):
            raise ManifestError(
                "frozen item verification evidence differs from manifest closure"
            )

        from web_agent.memory.verification import (
            validate_verification_records_payload,
        )

        try:
            verification_records = json.loads(
                (directory / "verification_evidence.json").read_text(
                    encoding="utf-8"
                )
            )
            validate_verification_records_payload(
                verification_records,
                expected_item_count=expected_count,
                items=tuple(items),
            )
        except (
            OSError,
            json.JSONDecodeError,
            VerificationEvidenceError,
            ManifestError,
        ) as error:
            raise ManifestError(
                f"frozen independent verification-record replay failed: {error}"
            ) from error

        from web_agent.memory.calibration_builder import (
            CalibrationEvidenceError,
            validate_frozen_calibration_bindings,
        )

        try:
            calibration_evidence = json.loads(
                (directory / "calibration_evidence.json").read_text(
                    encoding="utf-8"
                )
            )
            validate_frozen_calibration_bindings(
                calibration_evidence,
                items=tuple(items),
                embeddings=embeddings,
            )
        except (OSError, json.JSONDecodeError, CalibrationEvidenceError) as error:
            raise ManifestError(
                f"frozen calibration evidence replay failed: {error}"
            ) from error

        return cls(
            root=directory,
            embeddings=embeddings,
            items=tuple(items),
            verified=verified,
        )

    @staticmethod
    def _validate_item(
        item: Mapping[str, Any],
        *,
        line_number: int,
        duplicate_cluster_namespace_id: str,
    ) -> None:
        prefix = f"items.jsonl line {line_number}"
        for field in (
            "memory_id",
            "source_dataset_id",
            "source_dataset_version",
            "source_sample_id",
            "recovery_sample_id",
            "source_task_id",
            "source_episode_id",
            "duplicate_cluster_id",
            "duplicate_cluster_namespace_id",
            "failure_type",
            "strategy",
            "executed_recovery_action",
        ):
            _nonempty(item.get(field), field=f"{prefix}.{field}")
        if item.get("schema_version") != ELIGIBILITY_POLICY_VERSION:
            raise ManifestError(f"{prefix}.schema_version is not supported")
        if item.get("strategy") == "NONE":
            raise ManifestError(f"{prefix}.strategy cannot be NONE")
        if item.get("failure_type") == "NONE":
            raise ManifestError(f"{prefix}.failure_type cannot be NONE")
        step_index = item.get("step_index")
        if type(step_index) is not int or step_index < 0:
            raise ManifestError(f"{prefix}.step_index must be a non-negative integer")
        require_sha256(
            item.get("exact_duplicate_key"),
            field=f"{prefix}.exact_duplicate_key",
        )
        if item.get("source_split") != "train":
            raise ManifestError(f"{prefix}.source_split must be train")
        if item.get("duplicate_cluster_namespace_id") != duplicate_cluster_namespace_id:
            raise ManifestError(
                f"{prefix}.duplicate_cluster_namespace_id differs from manifest"
            )
        for field in (
            "memory_update_flag",
            "verified_recovery_success",
            "final_task_success",
            "provenance_valid",
        ):
            if item.get(field) is not True:
                raise ManifestError(f"{prefix}.{field} must be true")
        from web_agent.memory.verification import (
            VerificationEvidenceError,
            validate_frozen_item_verification,
        )

        try:
            validate_frozen_item_verification(item)
        except (VerificationEvidenceError, ManifestError) as error:
            raise ManifestError(
                f"{prefix} has invalid independent verification evidence: {error}"
            ) from error

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self._manifest

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @property
    def model_seed(self) -> int:
        return int(self._manifest["model_seed"])

    @property
    def admission_threshold(self) -> float:
        return float(self._manifest["admission_threshold"])

    def __len__(self) -> int:
        return len(self._items)

    def query(
        self,
        embedding: np.ndarray | list[float] | tuple[float, ...],
        *,
        exclusions: QueryExclusions,
    ) -> FrozenQueryResult:
        """Return at most three eligible neighbours with deterministic ties."""
        checked = exclusions.validated()
        query = np.asarray(embedding, dtype=np.float32)
        if query.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(
                f"query embedding must have shape ({EMBEDDING_DIMENSION},), "
                f"got {query.shape}"
            )
        if not np.isfinite(query).all():
            raise ValueError("query embedding contains non-finite values")
        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("query embedding must have non-zero finite norm")
        eligible_mask = np.zeros(len(self._items), dtype=np.bool_)
        exclusion_reasons: dict[str, str] = {}
        for index, item in enumerate(self._items):
            memory_id = str(item["memory_id"])
            if memory_id in checked.additional_memory_ids:
                exclusion_reasons[memory_id] = "explicit_memory_id"
                continue
            if str(item["source_task_id"]) == checked.current_task_id:
                exclusion_reasons[memory_id] = "same_task"
                continue
            if str(item["source_episode_id"]) == checked.current_episode_id:
                exclusion_reasons[memory_id] = "same_episode"
                continue
            if str(item["duplicate_cluster_id"]) in checked.duplicate_cluster_ids:
                exclusion_reasons[memory_id] = "duplicate_cluster"
                continue
            eligible_mask[index] = True

        eligible_count = int(np.count_nonzero(eligible_mask))
        memory_ids = tuple(str(item["memory_id"]) for item in self._items)
        ranked = deterministic_cosine_top_k(
            self._embeddings,
            query,
            memory_ids,
            k=RUNTIME_TOP_K,
            eligible_mask=eligible_mask,
        ) if eligible_count else ()
        hits = tuple(
            MemoryHit(
                rank=rank,
                memory_id=memory_id,
                cosine_similarity=score,
                item=self._items[eligible_index],
            )
            for rank, (score, memory_id, eligible_index) in enumerate(
                ranked, start=1
            )
        )
        return FrozenQueryResult(
            hits=hits,
            exclusion_reasons=MappingProxyType(exclusion_reasons),
            considered_count=len(self._items),
            eligible_count=eligible_count,
        )
