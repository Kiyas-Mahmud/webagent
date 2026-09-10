from __future__ import annotations

import secrets

import pytest

from web_agent.eval.table2.process_broker_protocol import (
    PROCESS_BROKER_CHILD_CLEANUP_RECORD_TYPE,
    PROCESS_BROKER_CHILD_CLEANUP_SCHEMA_VERSION,
    PROCESS_BROKER_PROTOCOL_VERSION,
    ProcessBrokerProtocolError,
    ProcessBrokerTranscript,
    authenticated_envelope,
    validate_child_cleanup_result,
)


def _request_frame(key: bytes) -> tuple[dict, dict]:
    payload = {"episode_id": "episode-1", "task_id": "task-1"}
    body = {
        "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
        "role": "runtime",
        "session_id": "session-1",
        "sequence": 0,
        "nonce": "a" * 64,
        "operation": "runtime_close",
        "payload": payload,
    }
    return authenticated_envelope(body, authentication_key=key), payload


def _response_frame(key: bytes) -> tuple[dict, dict]:
    result = {"closed": True}
    body = {
        "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
        "role": "sealed_runtime_response",
        "session_id": "session-1",
        "sequence": 0,
        "nonce": "a" * 64,
        "status": "PASS",
        "result": result,
    }
    return authenticated_envelope(body, authentication_key=key), result


def _commit_exchange(
    transcript: ProcessBrokerTranscript,
    *,
    key: bytes,
    operation: str = "runtime_close",
) -> dict:
    request, payload = _request_frame(key)
    response, result = _response_frame(key)
    transcript.commit_authenticated_frame(
        direction="request",
        role="runtime",
        sequence=0,
        operation=operation,
        frame=request,
        payload=payload,
    )
    return transcript.commit_authenticated_frame(
        direction="response",
        role="runtime",
        sequence=0,
        operation=operation,
        frame=response,
        payload=result,
    )


def test_authenticated_transcript_is_deterministic_and_context_bound() -> None:
    key = secrets.token_bytes(32)
    first = ProcessBrokerTranscript(session_id="session-1")
    second = ProcessBrokerTranscript(session_id="session-1")
    assert _commit_exchange(first, key=key) == _commit_exchange(second, key=key)
    assert first.snapshot()["entry_count"] == 2

    another_session = ProcessBrokerTranscript(session_id="session-2")
    assert _commit_exchange(another_session, key=key) != first.snapshot()

    another_operation = ProcessBrokerTranscript(session_id="session-1")
    assert _commit_exchange(
        another_operation, key=key, operation="runtime_terminal"
    ) != first.snapshot()

    another_authentication = ProcessBrokerTranscript(session_id="session-1")
    assert _commit_exchange(
        another_authentication, key=secrets.token_bytes(32)
    ) != first.snapshot()


def test_child_cleanup_result_requires_truthful_browser_close_and_sidecar() -> None:
    base = {
        "schema_version": PROCESS_BROKER_CHILD_CLEANUP_SCHEMA_VERSION,
        "record_type": PROCESS_BROKER_CHILD_CLEANUP_RECORD_TYPE,
        "cleanup_disposition": "CONTROL_ABORT_COMPLETED",
        "browser_close_attempted": True,
        "browser_close_completed": True,
        "manual_rescue_check_count": 1,
        "manual_rescue_tail_sha256": "b" * 64,
    }
    assert validate_child_cleanup_result(base) == base

    for changed in (
        {**base, "browser_close_completed": False},
        {**base, "manual_rescue_tail_sha256": None},
        {**base, "extra": True},
        {**base, "cleanup_disposition": "PROCESS_EXIT_INFERRED_CLOSE"},
    ):
        with pytest.raises(ProcessBrokerProtocolError):
            validate_child_cleanup_result(changed)

    no_browser = {
        **base,
        "cleanup_disposition": "NO_BROWSER_CREATED",
        "browser_close_attempted": False,
        "browser_close_completed": False,
        "manual_rescue_check_count": 0,
        "manual_rescue_tail_sha256": None,
    }
    assert validate_child_cleanup_result(no_browser) == no_browser
