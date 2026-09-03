from __future__ import annotations

from dataclasses import asdict

import pytest

from web_agent.eval.table2.sealed_page_broker import (
    BrowserGymStartStateApplicationReceipt,
    BrowserGymValidationDisabledBoundary,
    PageSessionAbortReceipt,
    PageSessionCloseReceipt,
    SealedBoundEvaluation,
    SealedBoundPageEvaluationRequest,
    SealedFinalEvaluation,
    SealedFinalPageEvaluationRequest,
    create_one_way_sealed_page_broker,
)
from web_agent.eval.table2.sealed_verifier import (
    SealedVerifierSink,
    SealedVerifierWriter,
    verify_sealed_stream,
)
from web_agent.runtime.contracts import (
    EpisodeSummary,
    OpaqueTerminalSignal,
    RuntimeStartState,
    SystemID,
    TerminalReason,
    VerifierReceiptBinding,
    canonical_sha256,
)


SHA = "a" * 64
EPISODE = "broker:E2:webarena.0:repeat-0:seed-42"
TASK = "webarena.0"
SESSION = "page-session-1"


def _boundary(**changes) -> BrowserGymValidationDisabledBoundary:
    values = {
        "boundary_id": "pc01-browsergym-validation-disabled",
        "boundary_version": "v1",
        "source_sha256": SHA,
    }
    values.update(changes)
    return BrowserGymValidationDisabledBoundary(**values)


def _start_state() -> RuntimeStartState:
    return RuntimeStartState(
        sites=("reddit",),
        start_url="http://reddit.local/task/0",
        require_login=True,
        storage_state=".auth/reddit_state.json",
        geolocation=None,
        require_reset=False,
    )


def _start_state_receipt(**changes) -> BrowserGymStartStateApplicationReceipt:
    start_state = _start_state()
    values = {
        "episode_id": EPISODE,
        "task_id": TASK,
        "runtime_start_state": start_state,
        "applied_start_state_sha256": start_state.start_state_sha256,
        "applier_id": "pc01-six-field-start-state-applier",
        "applier_version": "v1",
        "applier_source_sha256": SHA,
    }
    values.update(changes)
    return BrowserGymStartStateApplicationReceipt(**values)


def _sink(tmp_path) -> SealedVerifierSink:
    return SealedVerifierSink(
        tmp_path,
        block_id="task_webarena-0__seed_42__repeat_0",
        attempt_id=0,
        system_id="E2",
        episode_id=EPISODE,
        matched_seed=42,
        task_id=TASK,
        repeat_id=0,
    )


def _binding() -> VerifierReceiptBinding:
    return VerifierReceiptBinding(
        receipt_kind="after_reset",
        observation_id="observation-1",
        observation_sha256="b" * 64,
    )


def _summary() -> EpisodeSummary:
    return EpisodeSummary(
        episode_id=EPISODE,
        protocol_id="table2-pc01-pilot-v1",
        system_id=SystemID.E2,
        task_id=TASK,
        repeat_id=0,
        model_seed=42,
        valid_for_primary=True,
        terminal_reason=TerminalReason.ACTION_BUDGET_EXHAUSTED,
        executor_steps=0,
        normal_actions=0,
        recovery_actions=0,
        recovery_attempts=0,
        failure_incidents=0,
        memory_queries=0,
        memory_interventions=0,
        elapsed_seconds=1.0,
    )


def _final_evidence() -> dict:
    return {
        "task_success": False,
        "terminal_reason": "ACTION_BUDGET_EXHAUSTED",
        "loop_detected": False,
        "environment_failure": False,
        "failure_incidents": [],
        "recovery_verifications": [],
        "verified_failure_event_count": 0,
        "repeated_error_event_count": 0,
        "memory_relevance": {},
    }


def _publish(runtime, page, cleanup_calls=None):
    calls = [] if cleanup_calls is None else cleanup_calls
    return runtime.publish(
        session_id=SESSION,
        episode_id=EPISODE,
        task_id=TASK,
        page=page,
        page_state_digest=lambda current: canonical_sha256(current),
        close_live_page=lambda: calls.append("closed"),
        validation_boundary=_boundary(),
        start_state_application=_start_state_receipt(),
    )


def test_runtime_and_evaluator_receive_disjoint_one_way_capabilities(tmp_path) -> None:
    runtime, sealed = create_one_way_sealed_page_broker()
    assert not hasattr(runtime, "evaluate_bound")
    assert not hasattr(runtime, "evaluate_final")
    assert not hasattr(sealed, "publish")
    assert not hasattr(sealed, "close")

    page = {"url": "https://example.test", "title": "visible"}
    cleanup_calls = []
    registration = _publish(runtime, page, cleanup_calls)
    assert registration.live_page_returned is False
    assert "page" not in asdict(registration)

    sink = _sink(tmp_path)
    request = SealedBoundPageEvaluationRequest(
        session_id=SESSION,
        episode_id=EPISODE,
        task_id=TASK,
        binding=_binding(),
    )
    signal = sealed.evaluate_bound(
        request,
        writer=SealedVerifierWriter(sink),
        evaluator=lambda live_page, current: SealedBoundEvaluation(
            verification={
                "task_success": live_page["title"] == "visible",
                "request_sha256": current.record_sha256,
            },
            should_terminate=True,
        ),
    )
    assert type(signal) is OpaqueTerminalSignal
    assert set(signal.to_dict()) == {
        "schema_version",
        "record_type",
        "event_id",
        "token_sha256",
        "terminate",
    }
    assert signal.terminate is True
    close = runtime.close(session_id=SESSION, episode_id=EPISODE, task_id=TASK)
    assert type(close) is PageSessionCloseReceipt
    assert close.live_page_returned is False
    assert close.underlying_browser_close_called is False
    assert close.deferred_until_sealed_finalization is True
    assert "page" not in asdict(close)
    assert cleanup_calls == []

    final_signal = sealed.evaluate_final(
        SealedFinalPageEvaluationRequest(
            session_id=SESSION,
            episode_id=EPISODE,
            task_id=TASK,
            summary=_summary(),
        ),
        writer=SealedVerifierWriter(sink),
        evaluator=lambda live_page, current: SealedFinalEvaluation(
            verification=_final_evidence()
        ),
    )
    assert type(final_signal) is OpaqueTerminalSignal
    assert cleanup_calls == ["closed"]
    records = verify_sealed_stream(sink.path)
    assert records[0]["evidence"]["task_success"] is True
    assert records[-1]["event_kind"] == "episode_final"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("default_step_called", True),
        ("task_validation_called", True),
        ("reward_slot_read", True),
        ("termination_slots_read", True),
        ("info_slot_read", True),
        ("oracle_output_exposed_to_runtime", True),
        ("generic_webarena_task_used", True),
        ("adapter_close_deferred_until_sealed_finalization", False),
        ("cleanup_on_final_success", False),
        ("cleanup_on_final_error", False),
        ("cleanup_on_runtime_abort", False),
    ),
)
def test_page_publication_requires_validation_disabled_browsergym_execution(
    field: str,
    value: bool,
) -> None:
    with pytest.raises(ValueError, match="bypass task validation"):
        _boundary(**{field: value})


def test_evaluator_page_mutation_fails_before_any_oracle_result_crosses_boundary(
    tmp_path,
) -> None:
    runtime, sealed = create_one_way_sealed_page_broker()
    page = {"url": "https://example.test", "counter": 0}
    cleanup_calls = []
    _publish(runtime, page, cleanup_calls)
    sink = _sink(tmp_path)

    def mutate(live_page, request):
        live_page["counter"] += 1
        return SealedBoundEvaluation(
            verification={"task_success": True},
            should_terminate=True,
        )

    with pytest.raises(RuntimeError, match="mutated"):
        sealed.evaluate_bound(
            SealedBoundPageEvaluationRequest(
                session_id=SESSION,
                episode_id=EPISODE,
                task_id=TASK,
                binding=_binding(),
            ),
            writer=SealedVerifierWriter(sink),
            evaluator=mutate,
        )
    assert not sink.path.exists()
    assert cleanup_calls == ["closed"]
    close = runtime.close(session_id=SESSION, episode_id=EPISODE, task_id=TASK)
    assert type(close) is PageSessionAbortReceipt
    assert close.cleanup_reason == "bound_evaluation_error"


def test_untyped_evaluator_output_is_never_forwarded(tmp_path) -> None:
    runtime, sealed = create_one_way_sealed_page_broker()
    cleanup_calls = []
    _publish(runtime, {"url": "https://example.test"}, cleanup_calls)
    sink = _sink(tmp_path)
    with pytest.raises(TypeError, match="unsealed bound result"):
        sealed.evaluate_bound(
            SealedBoundPageEvaluationRequest(
                session_id=SESSION,
                episode_id=EPISODE,
                task_id=TASK,
                binding=_binding(),
            ),
            writer=SealedVerifierWriter(sink),
            evaluator=lambda page, request: {
                "oracle_success": True,
                "reward": 1.0,
            },
        )
    assert not sink.path.exists()
    assert cleanup_calls == ["closed"]


def test_final_evidence_is_written_once_and_only_opaque_signal_returns(tmp_path) -> None:
    runtime, sealed = create_one_way_sealed_page_broker()
    cleanup_calls = []
    _publish(runtime, {"url": "https://example.test"}, cleanup_calls)
    runtime.close(session_id=SESSION, episode_id=EPISODE, task_id=TASK)
    sink = _sink(tmp_path)
    request = SealedFinalPageEvaluationRequest(
        session_id=SESSION,
        episode_id=EPISODE,
        task_id=TASK,
        summary=_summary(),
    )
    evidence = _final_evidence()
    signal = sealed.evaluate_final(
        request,
        writer=SealedVerifierWriter(sink),
        evaluator=lambda page, current: SealedFinalEvaluation(
            verification=evidence
        ),
    )
    assert type(signal) is OpaqueTerminalSignal
    assert signal.terminate is True
    assert cleanup_calls == ["closed"]
    with pytest.raises(ValueError, match="exactly once"):
        sealed.evaluate_final(
            request,
            writer=SealedVerifierWriter(sink),
            evaluator=lambda page, current: SealedFinalEvaluation(
                verification=evidence
            ),
        )


def test_final_evaluator_requires_prior_adapter_close_and_still_cleans_on_error(
    tmp_path,
) -> None:
    runtime, sealed = create_one_way_sealed_page_broker()
    cleanup_calls = []
    _publish(runtime, {"url": "https://example.test"}, cleanup_calls)
    with pytest.raises(ValueError, match="prior deferred adapter close"):
        sealed.evaluate_final(
            SealedFinalPageEvaluationRequest(
                session_id=SESSION,
                episode_id=EPISODE,
                task_id=TASK,
                summary=_summary(),
            ),
            writer=SealedVerifierWriter(_sink(tmp_path)),
            evaluator=lambda page, request: SealedFinalEvaluation(
                verification=_final_evidence()
            ),
        )
    assert cleanup_calls == ["closed"]


def test_final_evaluator_failure_closes_underlying_browser_exactly_once(
    tmp_path,
) -> None:
    runtime, sealed = create_one_way_sealed_page_broker()
    cleanup_calls = []
    _publish(runtime, {"url": "https://example.test"}, cleanup_calls)
    runtime.close(session_id=SESSION, episode_id=EPISODE, task_id=TASK)

    def fail(page, request):
        raise RuntimeError("evaluator unavailable")

    with pytest.raises(RuntimeError, match="evaluator unavailable"):
        sealed.evaluate_final(
            SealedFinalPageEvaluationRequest(
                session_id=SESSION,
                episode_id=EPISODE,
                task_id=TASK,
                summary=_summary(),
            ),
            writer=SealedVerifierWriter(_sink(tmp_path)),
            evaluator=fail,
        )
    assert cleanup_calls == ["closed"]
    with pytest.raises(ValueError, match="not registered"):
        runtime.abort(session_id=SESSION, episode_id=EPISODE, task_id=TASK)


def test_final_html_evaluator_may_navigate_only_after_deferred_close(tmp_path) -> None:
    runtime, sealed = create_one_way_sealed_page_broker()
    cleanup_calls = []
    page = {"url": "https://example.test/original"}
    _publish(runtime, page, cleanup_calls)
    runtime.close(session_id=SESSION, episode_id=EPISODE, task_id=TASK)

    def final_html_evaluator(live_page, request):
        live_page["url"] = "https://example.test/evaluator-target"
        return SealedFinalEvaluation(verification=_final_evidence())

    signal = sealed.evaluate_final(
        SealedFinalPageEvaluationRequest(
            session_id=SESSION,
            episode_id=EPISODE,
            task_id=TASK,
            summary=_summary(),
        ),
        writer=SealedVerifierWriter(_sink(tmp_path)),
        evaluator=final_html_evaluator,
    )
    assert type(signal) is OpaqueTerminalSignal
    assert cleanup_calls == ["closed"]


def test_runtime_abort_is_the_only_nonfinal_cleanup_path() -> None:
    runtime, sealed = create_one_way_sealed_page_broker()
    cleanup_calls = []
    _publish(runtime, {"url": "https://example.test"}, cleanup_calls)
    receipt = runtime.abort(session_id=SESSION, episode_id=EPISODE, task_id=TASK)
    assert type(receipt) is PageSessionAbortReceipt
    assert receipt.cleanup_reason == "runtime_abort"
    assert cleanup_calls == ["closed"]
    assert not hasattr(sealed, "abort")
    with pytest.raises(ValueError, match="not registered"):
        runtime.abort(session_id=SESSION, episode_id=EPISODE, task_id=TASK)


def test_six_field_start_state_receipt_rejects_generic_webarena_task() -> None:
    with pytest.raises(ValueError, match="GenericWebArenaTask"):
        _start_state_receipt(generic_webarena_task_used=True)


def test_page_publication_rejects_partial_or_mismatched_start_state() -> None:
    with pytest.raises(ValueError, match="all six"):
        _start_state_receipt(applied_fields=("sites", "start_url"))

    runtime, _ = create_one_way_sealed_page_broker()
    other = _start_state_receipt(task_id="webarena.1")
    cleanup_calls = []
    with pytest.raises(ValueError, match="identity mismatch"):
        runtime.publish(
            session_id=SESSION,
            episode_id=EPISODE,
            task_id=TASK,
            page={"url": "https://example.test"},
            page_state_digest=lambda current: canonical_sha256(current),
            close_live_page=lambda: cleanup_calls.append("closed"),
            validation_boundary=_boundary(),
            start_state_application=other,
        )
    assert cleanup_calls == ["closed"]
