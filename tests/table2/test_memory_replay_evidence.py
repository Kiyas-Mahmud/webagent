from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from web_agent.eval.table2.common import SchemaError
from web_agent.eval.table2.package_validator import (
    _validate_and_replay_evaluation_memory_query,
)
from web_agent.memory.frozen_store import (
    FrozenMemoryStore,
    QueryExclusions,
)
from web_agent.memory.manifest import VerifiedStoreManifest
from web_agent.runtime.contracts import float32_vector_sha256


def _store(tmp_path) -> FrozenMemoryStore:
    embeddings = np.zeros((4, 768), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    embeddings[2, 2] = 1.0
    embeddings[3, 0] = 1.0
    items = (
        {
            "memory_id": "mem-a",
            "source_task_id": "train-task-a",
            "source_episode_id": "train-episode-a",
            "duplicate_cluster_id": "train-cluster-a",
            "strategy": "RETRY",
        },
        {
            "memory_id": "mem-b",
            "source_task_id": "train-task-b",
            "source_episode_id": "train-episode-b",
            "duplicate_cluster_id": "train-cluster-b",
            "strategy": "REPLAN",
        },
        {
            "memory_id": "mem-c",
            "source_task_id": "train-task-c",
            "source_episode_id": "train-episode-c",
            "duplicate_cluster_id": "train-cluster-c",
            "strategy": "ALTERNATIVE_TARGET",
        },
        {
            "memory_id": "mem-excluded",
            "source_task_id": "train-task-d",
            "source_episode_id": "train-episode-d",
            "duplicate_cluster_id": "evaluation-cluster",
            "strategy": "ABORT",
        },
    )
    return FrozenMemoryStore(
        root=tmp_path,
        embeddings=embeddings,
        items=items,
        verified=VerifiedStoreManifest(
            payload={"model_seed": 42, "admission_threshold": 0.5},
            sha256="a" * 64,
        ),
    )


def _evidence(store: FrozenMemoryStore) -> tuple[dict, dict, dict]:
    vector = np.zeros(768, dtype=np.float32)
    vector[0] = 1.0
    query_contract = {
        "query_id": "query-1",
        "task_id": "evaluation-task",
        "episode_id": "evaluation-episode",
        "duplicate_cluster_ids": ["evaluation-cluster"],
    }
    replayed = store.query(
        vector,
        exclusions=QueryExclusions(
            current_task_id=query_contract["task_id"],
            current_episode_id=query_contract["episode_id"],
            duplicate_cluster_ids=frozenset(
                query_contract["duplicate_cluster_ids"]
            ),
        ),
    )
    shadow = {
        "schema_version": "table2.runtime.v1",
        "record_type": "RecoveryDecision",
        "decision_id": "shadow-1",
        "incident_id": "incident-1",
        "strategy": "BACKTRACK",
        "trigger_sources": ["policy"],
        "diagnosis": "NO_EFFECT",
    }
    final = {
        **shadow,
        "decision_id": "shadow-1:memory:mem-a",
        "strategy": "RETRY",
    }
    result = {
        "query_id": "query-1",
        "candidate_ids": [hit.memory_id for hit in replayed.hits],
        "scores": [float(hit.cosine_similarity) for hit in replayed.hits],
        "exclusion_reasons": dict(replayed.exclusion_reasons),
        "reader_considered_count": replayed.considered_count,
        "reader_eligible_count": replayed.eligible_count,
        "admitted_candidate_id": "mem-a",
        "admitted": True,
        "final_strategy": "RETRY",
        "changed_strategy": True,
        "changed_target_or_parameters": False,
        "normalized_query_embedding": vector.tolist(),
        "normalized_query_embedding_sha256": float32_vector_sha256(
            vector.tolist()
        ),
    }
    return query_contract, shadow, {
        "shadow_decision": shadow,
        "final_decision": final,
        "query_result": result,
    }


def _validate(store: FrozenMemoryStore, query_contract: dict, payload: dict) -> None:
    _validate_and_replay_evaluation_memory_query(
        schedule_row={"matched_model_seed": 42},
        payload=payload,
        query_contract=query_contract,
        query_result=payload["query_result"],
        shadow_decision=payload["shadow_decision"],
        store=store,
    )


def test_evaluation_memory_evidence_replays_exactly(tmp_path) -> None:
    store = _store(tmp_path)
    query_contract, _, payload = _evidence(store)
    _validate(store, query_contract, payload)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        (
            lambda payload: payload["query_result"]["scores"].__setitem__(0, 0.9),
            "scores differ",
        ),
        (
            lambda payload: payload["query_result"]["candidate_ids"].reverse(),
            "candidate IDs differ",
        ),
        (
            lambda payload: payload["query_result"]["exclusion_reasons"].clear(),
            "exclusion reasons differ",
        ),
        (
            lambda payload: payload["query_result"].update(
                {"admitted": False, "admitted_candidate_id": None}
            ),
            "admission differs",
        ),
        (
            lambda payload: payload["query_result"].update(
                {"final_strategy": "ABORT"}
            ),
            "final strategy differs",
        ),
        (
            lambda payload: payload["final_decision"].update(
                {"strategy": "ABORT"}
            ),
            "final decision differs",
        ),
    ),
)
def test_evaluation_memory_replay_rejects_hostile_result_tampering(
    tmp_path,
    tamper,
    message,
) -> None:
    store = _store(tmp_path)
    query_contract, _, base = _evidence(store)
    payload = deepcopy(base)
    tamper(payload)
    with pytest.raises(SchemaError, match=message):
        _validate(store, query_contract, payload)


def test_evaluation_memory_replay_rejects_resealed_wrong_vector(tmp_path) -> None:
    store = _store(tmp_path)
    query_contract, _, payload = _evidence(store)
    hostile = deepcopy(payload)
    vector = np.zeros(768, dtype=np.float32)
    vector[1] = 1.0
    hostile["query_result"]["normalized_query_embedding"] = vector.tolist()
    hostile["query_result"]["normalized_query_embedding_sha256"] = (
        float32_vector_sha256(vector.tolist())
    )

    with pytest.raises(SchemaError, match="candidate IDs differ"):
        _validate(store, query_contract, hostile)
