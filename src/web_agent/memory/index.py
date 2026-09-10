"""Pillar 4 retrieval index (SPEC 4.4 / ARCH 5.4). Inference-time, NOT trained.

BUILD: over TRAIN rows where memory_update_flag=True AND recovery_success=True,
  store {fused_embedding, failure_type, recovery_strategy, reflection_text, domain}.
QUERY: a fused embedding -> top-3 cosine neighbours -> their recovery_strategy.

Simple numpy cosine top-k; upgrade to FAISS for production scale.
"""

from __future__ import annotations

import numpy as np


def deterministic_cosine_top_k(
    embeddings: np.ndarray,
    embedding: np.ndarray,
    item_ids: list[str] | tuple[str, ...],
    *,
    k: int = 3,
    eligible_mask: np.ndarray | list[bool] | tuple[bool, ...] | None = None,
) -> tuple[tuple[float, str, int], ...]:
    """Return ``(score, item_id, row_index)`` in the registered stable order.

    This is the shared low-level search primitive.  Evaluation-time provenance,
    exclusions, admission, and immutability remain the responsibility of the
    frozen-store wrapper; this function only performs validated cosine ranking.
    """

    matrix = np.asarray(embeddings, dtype=np.float32)
    query = np.asarray(embedding, dtype=np.float32).reshape(-1)
    ids = tuple(str(item_id) for item_id in item_ids)
    if matrix.ndim != 2:
        raise ValueError("cosine index embeddings must be a matrix")
    if query.shape != (matrix.shape[1],):
        raise ValueError(
            "cosine query dimension differs from index dimension: "
            f"{query.shape} vs {(matrix.shape[1],)}"
        )
    if matrix.shape[0] != len(ids) or len(ids) != len(set(ids)):
        raise ValueError("cosine ranking requires one unique ID per row")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("cosine top-k must be a positive integer")
    if not np.isfinite(matrix).all() or not np.isfinite(query).all():
        raise ValueError("cosine inputs contain non-finite values")
    norm = float(np.linalg.norm(query))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("cosine query must have a non-zero finite norm")
    if eligible_mask is None:
        eligible_indices = np.arange(matrix.shape[0], dtype=np.int64)
    else:
        mask = np.asarray(eligible_mask)
        if mask.shape != (matrix.shape[0],) or mask.dtype.kind != "b":
            raise ValueError(
                "cosine eligible_mask must be one boolean per index row"
            )
        eligible_indices = np.flatnonzero(mask)
    if eligible_indices.size == 0:
        return ()

    normalized = query / norm
    # Score the existing matrix once.  Filtering after scoring avoids an O(ND)
    # candidate-matrix copy for every leave-one-out calibration query.
    similarities = np.clip(matrix @ normalized, -1.0, 1.0)
    if eligible_indices.size <= k:
        selected = eligible_indices.tolist()
    else:
        eligible_scores = similarities[eligible_indices]
        kth_position = int(eligible_scores.size - k)
        kth_score = np.partition(eligible_scores, kth_position)[kth_position]
        higher = eligible_indices[eligible_scores > kth_score].tolist()
        boundary = eligible_indices[eligible_scores == kth_score].tolist()
        # At most k-1 rows are strictly above the boundary.  Only the exact-tie
        # boundary needs memory-ID ordering; this preserves the registered
        # (-score, memory_id) result without sorting every eligible row.
        boundary.sort(key=lambda index: ids[index])
        selected = higher + boundary[: k - len(higher)]
    ranked = [
        (float(similarities[index]), ids[index], int(index))
        for index in selected
    ]
    ranked.sort(key=lambda value: (-value[0], value[1]))
    return tuple(ranked)


class MemoryIndex:
    def __init__(self):
        self.emb: np.ndarray | None = None     # [N, 768] L2-normalized
        self.meta: list[dict] = []

    def build(self, embeddings, metadata) -> None:
        e = np.asarray(embeddings, dtype=np.float32)
        norm = np.linalg.norm(e, axis=1, keepdims=True)
        self.emb = e / np.clip(norm, 1e-8, None)
        self.meta = list(metadata)

    def query(self, embedding, k: int = 3):
        if self.emb is None or len(self.meta) == 0:
            return []
        item_ids = tuple(
            str(item.get("memory_id", f"row-{index:020d}"))
            for index, item in enumerate(self.meta)
        )
        ranked = deterministic_cosine_top_k(
            self.emb,
            np.asarray(embedding, dtype=np.float32),
            item_ids,
            k=k,
        )
        return [
            {**self.meta[index], "score": score}
            for score, _, index in ranked
        ]

    def __len__(self) -> int:
        return len(self.meta)
