"""Pillar 4 retrieval index (SPEC 4.4 / ARCH 5.4). Inference-time, NOT trained.

BUILD: over TRAIN rows where memory_update_flag AND recovery_success, store
  {fused_embedding, failure_type, recovery_strategy, reflection_text, website_domain}.
QUERY: current fused_embedding -> top-3 cosine -> return their recovery_strategy.

Start simple (numpy/sklearn cosine); upgrade to FAISS for production.
The 5 passes of one original_task_id form natural episodic pairs for retrieval eval.
"""

from __future__ import annotations


class MemoryIndex:
    def __init__(self):
        # TODO(Phase 6): hold embeddings matrix + parallel metadata list.
        raise NotImplementedError("Build in Phase 6 (after Y1; see docs/IMPLEMENTATION_PLAN.md §6).")

    def build(self, embeddings, metadata):
        raise NotImplementedError

    def query(self, embedding, k: int = 3):
        """Return top-k similar past failures + their recovery_strategy."""
        raise NotImplementedError
