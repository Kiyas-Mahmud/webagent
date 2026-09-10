from __future__ import annotations

from dataclasses import fields
import hashlib
import struct

import pytest

from web_agent.eval.table2.production_runner import SeedRuntimeBinding
from web_agent.runtime.checkpoint_inference import LoadedSelectedCheckpointBackend
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    ExecutionResult,
    ExecutionStatus,
    PolicyObservation,
    TransitionInput,
)
from web_agent.runtime.memory_adapter import (
    CallablePostFailureEmbeddingProvider,
    MemoryBoundaryError,
    PostFailureEmbedding,
    PostFailureEmbeddingRequest,
)


CHECKPOINT_SHA256 = "a" * 64
PROCESSOR_SHA256 = "b" * 64


def _request(query_id: str, state: str) -> PostFailureEmbeddingRequest:
    pre = PolicyObservation(
        task_id="task-1",
        goal="complete the task",
        observation_id=f"{query_id}:pre",
        screenshot_sha256="1" * 64,
        screenshot_path=None,
        width=1280,
        height=720,
        url="https://fixture.invalid/before",
        title="before",
        current_page_state={"state": "before"},
    )
    post = PolicyObservation(
        task_id="task-1",
        goal="complete the task",
        observation_id=f"{query_id}:post",
        screenshot_sha256="2" * 64,
        screenshot_path=None,
        width=1280,
        height=720,
        url="https://fixture.invalid/after",
        title="after",
        current_page_state={"state": state},
    )
    action = ConcreteAction(
        action_id=f"{query_id}:action",
        source_decision_id=f"{query_id}:decision",
        action_type=ActionType.CLICK,
        parameters={"x": 10, "y": 20},
    )
    transition = TransitionInput(
        task_id="task-1",
        pre_observation=pre,
        executed_action=action,
        execution_result=ExecutionResult(
            action_id=action.action_id,
            status=ExecutionStatus.EXECUTED,
            executor_step=1,
            state_changed=False,
        ),
        post_observation=post,
    )
    return PostFailureEmbeddingRequest(
        query_id=query_id,
        post_failure_observation_id=post.observation_id,
        failed_action_id=action.action_id,
        post_action_input=transition,
        post_failure_observation_sha256=hashlib.sha256(
            f"raw:{query_id}:{state}".encode()
        ).hexdigest(),
        post_action_input_sha256=transition.record_sha256,
        processor_contract_sha256=PROCESSOR_SHA256,
        checkpoint_sha256=CHECKPOINT_SHA256,
    )


def _embedding_sha256(values: tuple[float, ...]) -> str:
    return hashlib.sha256(
        struct.pack(f"<{len(values)}f", *(float(value) for value in values))
    ).hexdigest()


def _receipt(
    request: PostFailureEmbeddingRequest,
    values: tuple[float, ...],
    processed_batch_sha256: str,
) -> PostFailureEmbedding:
    return PostFailureEmbedding(
        query_id=request.query_id,
        post_failure_observation_id=request.post_failure_observation_id,
        values=values,
        request_sha256=request.record_sha256,
        post_failure_observation_sha256=(
            request.post_failure_observation_sha256
        ),
        post_action_input_sha256=request.post_action_input_sha256,
        processor_contract_sha256=request.processor_contract_sha256,
        checkpoint_sha256=request.checkpoint_sha256,
        processed_batch_sha256=processed_batch_sha256,
        embedding_sha256=_embedding_sha256(values),
    )


def _provider(callback):
    return CallablePostFailureEmbeddingProvider(
        embedder=callback,
        provider_id="selected-checkpoint-memory-embedding",
        provider_version="v1",
        checkpoint_sha256=CHECKPOINT_SHA256,
        processor_contract_sha256=PROCESSOR_SHA256,
    )


def test_embedding_request_is_recursively_immutable() -> None:
    request = _request("query-1", "first")

    with pytest.raises(TypeError):
        request.post_action_input.post_observation.current_page_state["state"] = (
            "tampered"
        )
    with pytest.raises(TypeError):
        request.post_action_input.executed_action.parameters["x"] = 999


def test_backend_stale_receipt_for_another_exact_input_is_rejected() -> None:
    first = _request("query-1", "first")
    second = _request("query-2", "second")
    values = (1.0,) + (0.0,) * 767
    stale = _receipt(first, values, "c" * 64)
    provider = _provider(lambda _request: stale)

    with pytest.raises(MemoryBoundaryError, match="evidence differs"):
        provider.embed(second)


def test_constant_embedding_across_different_processed_batches_is_rejected() -> None:
    first = _request("query-1", "first")
    second = _request("query-2", "second")
    values = (1.0,) + (0.0,) * 767

    def constant_backend(request: PostFailureEmbeddingRequest) -> PostFailureEmbedding:
        batch_sha256 = "c" * 64 if request.query_id == "query-1" else "d" * 64
        return _receipt(request, values, batch_sha256)

    provider = _provider(constant_backend)
    provider.embed(first)
    with pytest.raises(MemoryBoundaryError, match="reused across different"):
        provider.embed(second)


def test_production_binding_cannot_inject_a_separate_p4_provider() -> None:
    backend_fields = {item.name for item in fields(LoadedSelectedCheckpointBackend)}
    seed_fields = {item.name for item in fields(SeedRuntimeBinding)}

    assert "memory_embedding" in backend_fields
    assert "post_failure_embedding_provider" not in seed_fields
    assert "post_failure_embedding_checkpoint_sha256" not in seed_fields
