"""Pillar 4 retrieval index (SPEC 4.4 / ARCH 5.4). Inference-time, NOT trained.

BUILD: over TRAIN rows where memory_update_flag=True AND recovery_success=True,
  store {fused_embedding, failure_type, recovery_strategy, reflection_text, domain}.
QUERY: a fused embedding -> top-3 cosine neighbours -> their recovery_strategy.

Simple numpy cosine top-k; upgrade to FAISS for production scale.
"""

from __future__ import annotations

import numpy as np


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
        q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        q = q / max(np.linalg.norm(q), 1e-8)
        sims = self.emb @ q                     # cosine (both normalized)
        top = np.argsort(-sims)[:k]
        return [{**self.meta[i], "score": float(sims[i])} for i in top]

    def __len__(self) -> int:
        return len(self.meta)
