from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import hashlib
import hmac
import json
from pathlib import Path
import socket
from typing import Any

import pytest

from web_agent.benchmarks.base import EnvironmentAdapter
from web_agent.eval.table2 import process_broker_webarena_backend as backend_module
from web_agent.eval.table2.common import (
    SchemaError,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from web_agent.eval.table2.process_broker import ProcessIsolatedBroker
from web_agent.eval.table2.process_broker_finalization import (
    PROCESS_BROKER_FINALIZATION_RECEIPT_SCHEMA_VERSION,
    ProcessIsolatedEpisodeFinalizationBinding,
    SEALED_CALLBACK_GUARD_RELATIVE_PATH,
    SEALED_FINALIZATION_OPERATION,
)
from web_agent.eval.table2.process_broker_protocol import (
    RUNTIME_BROKER_OPERATIONS,
    ProcessBrokerProtocolError,
    expected_verifier_receipt_binding,
)
from web_agent.eval.table2.process_broker_webarena_backend import (
    PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
    PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
    PROCESS_BROKER_WEBARENA_FINALIZING_BACKEND_SCHEMA_VERSION,
    ProcessBrokerWebArenaEnvironmentAdapter,
    create_backend,
)
from web_agent.eval.table2.sealed_verifier import (
    SealedVerifierSink,
    SealedVerifierStreamTarget,
    SealedVerifierWriter,
    verify_sealed_stream,
)
from web_agent.eval.table2.production_runner import (
    ProcessIsolatedWebArenaEpisodeBinding,
    ProductionTable2Runner,
    _ActiveProcessWebArenaSession,
)
from web_agent.benchmarks.browsergym_webarena import (
    BrowserGymEpisodeAbortReceipt,
)
from web_agent.runtime.event_log import EpisodeEventLogs
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    EpisodeSummary,
    ObservationStage,
    RuntimeStartState,
    SystemID,
    TaskSpecification,
    TerminalReason,
    VerifierReceiptBinding,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SOURCE = "src/web_agent/eval/table2/process_broker_fixture_backend.py"
FIXTURE_FACTORY = (
    "web_agent.eval.table2.process_broker_fixture_backend:"
    "create_environment_adapter"
)
FIXTURE_FINALIZER = (
    "web_agent.eval.table2.process_broker_fixture_backend:"
    "finalize_environment_episode"
)
FIXTURE_TRANSITION_EVALUATOR = (
    "web_agent.eval.table2.process_broker_fixture_backend:"
    "evaluate_environment_transition"
)


def _task(*, goal: str = "complete the visible fixture task") -> TaskSpecification:
    start = RuntimeStartState(
        sites=("shopping",),
        start_url="https://fixture.invalid/start",
        require_login=False,
        storage_state=None,
        geolocation=None,
        require_reset=True,
    )
    return TaskSpecification(
        task_id="task-1",
        goal=goal,
        benchmark_id="webarena",
        benchmark_version="webarena-fixture-v1",
        start_state_id=start.start_state_sha256,
        site="shopping",
        start_url=start.start_url,
        development_partition=True,
        destructive_actions_allowed=False,
        metadata={
            "task_partition": "normal",
            "upstream_index": 1,
            "benchmark_task_id": "webarena.1",
            "source_content_sha256": "b" * 64,
        },
        runtime_start_state=start,
    )


def _backend_config(
    task: TaskSpecification,
    runtime_dir: Path,
    target: SealedVerifierStreamTarget,
) -> dict:
    source_hash = sha256_file(ROOT / FIXTURE_SOURCE)
    return {
        "schema_version": (
            PROCESS_BROKER_WEBARENA_FINALIZING_BACKEND_SCHEMA_VERSION
        ),
        "adapter_factory_entrypoint": FIXTURE_FACTORY,
        "adapter_factory_source_relative_path": FIXTURE_SOURCE,
        "adapter_factory_source_sha256": source_hash,
        "sealed_finalizer_entrypoint": FIXTURE_FINALIZER,
        "sealed_finalizer_source_relative_path": FIXTURE_SOURCE,
        "sealed_finalizer_source_sha256": source_hash,
        "sealed_transition_callback_entrypoint": (
            FIXTURE_TRANSITION_EVALUATOR
        ),
        "sealed_transition_callback_source_relative_path": FIXTURE_SOURCE,
        "sealed_transition_callback_source_sha256": source_hash,
        "task_specification": task.to_dict(),
        "episode_runtime_dir": str(runtime_dir),
        "sealed_stream_target": target.to_dict(),
    }


def _binding(observation, action=None) -> VerifierReceiptBinding:
    return VerifierReceiptBinding.from_dict(
        expected_verifier_receipt_binding(
            observation=observation.to_dict(),
            action=action.to_dict() if action is not None else None,
        )
    )


def _summary(
    terminal_reason: TerminalReason,
    *,
    executor_steps: int = 0,
    verifier_signal=None,
) -> EpisodeSummary:
    return EpisodeSummary(
        episode_id="episode-1",
        protocol_id="table2-pc01-pilot-v1",
        system_id=SystemID.E0,
        task_id="task-1",
        repeat_id=0,
        model_seed=42,
        valid_for_primary=True,
        terminal_reason=terminal_reason,
        executor_steps=executor_steps,
        normal_actions=executor_steps,
        recovery_actions=0,
        recovery_attempts=0,
        failure_incidents=0,
        memory_queries=0,
        memory_interventions=0,
        elapsed_seconds=0.25,
        verifier_event_id=(
            verifier_signal.event_id if verifier_signal is not None else None
        ),
        verifier_token_sha256=(
            verifier_signal.token_sha256 if verifier_signal is not None else None
        ),
        event_log_sha256="d" * 64,
    )


def _sink(tmp_path: Path) -> SealedVerifierSink:
    return SealedVerifierSink(
        tmp_path / "campaign",
        block_id="task_task-1__seed_42__repeat_000",
        attempt_id=0,
        system_id="E0",
        episode_id="episode-1",
        matched_seed=42,
        task_id="task-1",
        repeat_id=0,
    )


def _broker(tmp_path: Path, task: TaskSpecification):
    sink = _sink(tmp_path)
    runtime_dir = sink.system_root / "runtime"
    runtime_dir.mkdir(parents=True)
    target = SealedVerifierStreamTarget.from_sink(sink)
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
        backend_source_relative_path=PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
        backend_dependency_source_relative_paths=(FIXTURE_SOURCE,),
        sealed_backend_config=_backend_config(task, runtime_dir, target),
        require_sealed_finalization=True,
    )
    return broker, runtime_dir, sink, target


def _reachable_values(root: object) -> list[object]:
    """Inspect retained instance state without invoking properties/callbacks."""

    pending = [root]
    seen: set[int] = set()
    values: list[object] = []
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        values.append(value)
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
            continue
        if is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, item.name) for item in fields(value))
        instance_dict = getattr(value, "__dict__", None)
        if isinstance(instance_dict, dict):
            pending.extend(instance_dict.values())
        for owner in type(value).__mro__:
            slots = owner.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot in {"__dict__", "__weakref__"}:
                    continue
                name = (
                    f"_{owner.__name__.lstrip('_')}{slot}"
                    if slot.startswith("__") and not slot.endswith("__")
                    else slot
                )
                try:
                    pending.append(object.__getattribute__(value, name))
                except (AttributeError, TypeError):
                    pass
    return values


def _sealed_records(sink: SealedVerifierSink) -> list[dict[str, Any]]:
    return sink.verified_records()


def _rewrite_authenticated_record(
    sink: SealedVerifierSink,
    *,
    field: str,
    value: object,
) -> None:
    record = json.loads(sink.path.read_text(encoding="utf-8"))
    if field == "<extra>":
        record["unexpected"] = value
    else:
        record[field] = value
    body = {
        name: item
        for name, item in record.items()
        if name not in {"record_hash", "opaque_token_sha256"}
    }
    record_hash = sha256_json(body)
    record["record_hash"] = record_hash
    key = object.__getattribute__(sink, "_key")
    record["opaque_token_sha256"] = hmac.new(
        key,
        canonical_json_bytes([record["event_id"], record_hash]),
        hashlib.sha256,
    ).hexdigest()
    sink.path.write_bytes(canonical_json_bytes(record) + b"\n")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("<extra>", True),
        ("event_index", None),
        ("event_index", "0"),
        ("event_index", 0.0),
        ("event_index", True),
        ("event_index", -1),
        ("event_kind", {"kind": "transition"}),
        ("evidence", ["not", "an", "object"]),
        ("attempt_id", "0"),
        ("system_id", {"value": "E0"}),
    ),
)
def test_authenticated_sealed_record_requires_exact_closed_top_level_schema(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    sink = _sink(tmp_path)
    sink.record({"fixture": True}, should_terminate=False)
    _rewrite_authenticated_record(sink, field=field, value=value)

    with pytest.raises(SchemaError):
        verify_sealed_stream(sink.path)


@pytest.mark.parametrize(
    "malformed_record",
    (
        b'{"evidence":NaN}\n',
        b'{"evidence":"\\ud800"}\n',
    ),
)
def test_malformed_sealed_json_always_fails_as_schema_error(
    tmp_path: Path,
    malformed_record: bytes,
) -> None:
    sink = _sink(tmp_path)
    sink.record({"fixture": True}, should_terminate=False)
    sink.path.write_bytes(malformed_record)

    with pytest.raises(SchemaError):
        verify_sealed_stream(sink.path)


class _ConstructionTrackingAdapter(EnvironmentAdapter):
    benchmark_id = "webarena"

    def __init__(self, benchmark_version: str, *, close_raises: bool) -> None:
        self.benchmark_version = benchmark_version
        self.close_calls = 0
        self.close_raises = close_raises

    def reset(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("construction-only adapter must not reset")

    def observe(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("construction-only adapter must not observe")

    def execute(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("construction-only adapter must not execute")

    def terminal_signal(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("construction-only adapter must not finalize")

    def close(self) -> None:
        self.close_calls += 1
        if self.close_raises:
            raise RuntimeError("fixture close failure")


@pytest.mark.parametrize("failure", ("task_mutation", "backend_construction"))
def test_post_factory_failure_closes_adapter_once_and_releases_stream_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    task = _task()
    sink = _sink(tmp_path)
    runtime_dir = sink.system_root / "runtime"
    runtime_dir.mkdir(parents=True)
    target = SealedVerifierStreamTarget.from_sink(sink)
    adapter = _ConstructionTrackingAdapter(
        task.benchmark_version if failure == "task_mutation" else "wrong-version",
        close_raises=failure == "task_mutation",
    )

    def factory(*, task: TaskSpecification, episode_runtime_dir: Path) -> Any:
        assert episode_runtime_dir == runtime_dir
        if failure == "task_mutation":
            object.__setattr__(task, "goal", "mutated by fixture factory")
        return adapter

    monkeypatch.setattr(
        backend_module,
        "_load_source_attested_adapter_factory",
        lambda _config: factory,
    )
    monkeypatch.setattr(
        backend_module,
        "_load_source_attested_finalizer",
        lambda _config: (lambda **_kwargs: None),
    )
    monkeypatch.setattr(
        backend_module,
        "_load_source_attested_transition_evaluator",
        lambda _config: (lambda **_kwargs: None),
    )

    expected_error = (
        ProcessBrokerProtocolError if failure == "task_mutation" else ValueError
    )
    expected_message = "mutated protected task|adapter version differs"
    with pytest.raises(expected_error, match=expected_message):
        create_backend(_backend_config(task, runtime_dir, target))

    assert adapter.close_calls == 1
    replacement_owner = target.open_child_sink(episode_runtime_dir=runtime_dir)
    replacement_owner.release_child_ownership()


def test_runtime_adapter_recursively_retains_no_sealed_capability_or_endpoint(
    tmp_path: Path,
) -> None:
    task = _task()
    broker, _runtime_dir, sink, _target = _broker(tmp_path, task)
    broker.start()
    try:
        adapter = ProcessBrokerWebArenaEnvironmentAdapter(
            client=broker.runtime_client(),
            benchmark_version=task.benchmark_version,
        )
        reachable = _reachable_values(adapter)
        assert not any(
            isinstance(
                value,
                (SealedVerifierSink, SealedVerifierWriter, SealedVerifierStreamTarget),
            )
            for value in reachable
        )
        assert not any(isinstance(value, socket.socket) for value in reachable)
        seal_key = object.__getattribute__(sink, "_key")
        assert not any(value == seal_key for value in reachable if isinstance(value, bytes))
        forbidden_paths = {str(sink.path), str(sink.sealed_root / ".seal_key")}
        assert not any(value in forbidden_paths for value in reachable if isinstance(value, str))
    finally:
        broker.stop()
    assert broker.cleaned


def test_stream_target_rejects_drift_nonempty_legacy_writer_and_concurrent_owner(
    tmp_path: Path,
) -> None:
    task = _task()
    sink = _sink(tmp_path)
    runtime_dir = sink.system_root / "runtime"
    runtime_dir.mkdir(parents=True)
    target = SealedVerifierStreamTarget.from_sink(sink)

    with pytest.raises(SchemaError, match="transferred"):
        sink.record({"legacy": True}, should_terminate=False)
    with pytest.raises(SchemaError, match="runtime directory"):
        replace(target, system_id="E1").open_child_sink(
            episode_runtime_dir=runtime_dir
        )
    with pytest.raises(SchemaError, match="digest"):
        replace(target, seal_key_sha256="f" * 64).open_child_sink(
            episode_runtime_dir=runtime_dir
        )

    child_sink = target.open_child_sink(episode_runtime_dir=runtime_dir)
    try:
        with pytest.raises(SchemaError, match="already has an owner"):
            target.open_child_sink(episode_runtime_dir=runtime_dir)
    finally:
        child_sink.release_child_ownership()

    occupied = _sink(tmp_path / "occupied")
    occupied.record({"fixture": "existing"}, should_terminate=False)
    with pytest.raises(SchemaError, match="not empty"):
        SealedVerifierStreamTarget.from_sink(occupied)

    legacy_config = _backend_config(task, runtime_dir, target)
    legacy_config["sealed_stream_target"] = SealedVerifierWriter(sink)
    with pytest.raises(ProcessBrokerProtocolError, match="canonical JSON"):
        create_backend(legacy_config)


def test_child_owned_stream_never_follows_symlinked_episode_directory(
    tmp_path: Path,
) -> None:
    sink = _sink(tmp_path)
    runtime_dir = sink.system_root / "runtime"
    runtime_dir.mkdir(parents=True)
    target = SealedVerifierStreamTarget.from_sink(sink)
    child_sink = target.open_child_sink(episode_runtime_dir=runtime_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    child_sink.path.parent.symlink_to(outside, target_is_directory=True)
    try:
        with pytest.raises(SchemaError, match="directory tree"):
            child_sink.record({"must_not_escape": True}, should_terminate=False)
    finally:
        child_sink.release_child_ownership()

    assert list(outside.iterdir()) == []
    assert not (outside / "verifier_events.jsonl").exists()


def test_sink_construction_rejects_preexisting_symlinked_episode_directory(
    tmp_path: Path,
) -> None:
    episode_dir = (
        tmp_path
        / "campaign"
        / "paired_blocks"
        / "seed_42"
        / "task-1"
        / "repeat_0"
        / "rerun_0"
        / "E0"
        / "sealed"
        / "episode-1"
    )
    episode_dir.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    episode_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SchemaError, match="episode directory"):
        _sink(tmp_path)

    assert list(outside.iterdir()) == []
    assert not (episode_dir.parent / ".seal_key").exists()


def test_child_owner_lock_unlink_recreate_cannot_create_second_owner(
    tmp_path: Path,
) -> None:
    sink = _sink(tmp_path)
    runtime_dir = sink.system_root / "runtime"
    runtime_dir.mkdir(parents=True)
    target = SealedVerifierStreamTarget.from_sink(sink)
    first_owner = target.open_child_sink(episode_runtime_dir=runtime_dir)
    lock_path = sink.sealed_root / ".child_owner.lock"

    lock_path.unlink()
    lock_path.write_bytes(b"replacement marker")
    lock_path.chmod(0o600)
    replacement = lock_path.stat()
    replacement_identity = (replacement.st_dev, replacement.st_ino)
    try:
        with pytest.raises(SchemaError, match="already has an owner"):
            target.open_child_sink(episode_runtime_dir=runtime_dir)
        assert (lock_path.stat().st_dev, lock_path.stat().st_ino) == replacement_identity
    finally:
        first_owner.release_child_ownership()

    # Cleanup closes only the held descriptors; it never removes a pathname
    # that may now designate somebody else's replacement inode.
    assert lock_path.read_bytes() == b"replacement marker"
    assert (lock_path.stat().st_dev, lock_path.stat().st_ino) == replacement_identity
    next_owner = target.open_child_sink(episode_runtime_dir=runtime_dir)
    next_owner.release_child_ownership()


def test_finalization_is_separate_single_use_capability_and_browser_survives_close(
    tmp_path: Path,
) -> None:
    task = _task()
    broker, _runtime_dir, sink, _target = _broker(tmp_path, task)
    broker.start()
    try:
        runtime = broker.runtime_client()
        assert SEALED_FINALIZATION_OPERATION not in RUNTIME_BROKER_OPERATIONS
        assert not hasattr(runtime, "finalize_episode")
        adapter = ProcessBrokerWebArenaEnvironmentAdapter(
            client=runtime,
            benchmark_version=task.benchmark_version,
        )
        reset = adapter.reset(task, episode_id="episode-1", seed=42)
        reset_signal = adapter.terminal_signal(task, _binding(reset))
        assert reset_signal.terminate is False
        action = ConcreteAction(
            action_id="action-1",
            source_decision_id="decision-1",
            action_type=ActionType.NAVIGATE,
            parameters={"url": "https://fixture.invalid/done"},
        )
        adapter.execute(action)
        post = adapter.observe(
            stage=ObservationStage.POST_ACTION,
            prior_action_id=action.action_id,
        )
        terminal = adapter.terminal_signal(task, _binding(post, action))
        assert terminal.terminate is True
        adapter.close()

        receipt = broker.receipt
        assert receipt.sealed_finalization_capability_available is True
        assert receipt.sealed_finalization_operation_in_runtime_allowlist is False
        assert receipt.child_owned_sealed_sink is True
        assert receipt.runtime_adapter_sealed_capability_free is True
        assert receipt.separate_evidence_transport_present is False
        assert receipt.runtime_terminal_returns_outer_sealed_signal is True
        assert receipt.sealed_finalization_returns_outer_sealed_signal is True
        finalizer = broker.finalization_client()
        signal = finalizer.finalize_episode(
            episode_summary=_summary(
                TerminalReason.OPAQUE_VERIFIER_TERMINAL,
                executor_steps=1,
                verifier_signal=terminal,
            ),
        )
        assert signal.terminate is True
        finalization_receipt = finalizer.finalization_receipt
        assert finalization_receipt.schema_version == (
            PROCESS_BROKER_FINALIZATION_RECEIPT_SCHEMA_VERSION
        )
        assert finalization_receipt.transcript_entry_count == 14
        assert finalization_receipt.sealed_callback_guard_identity[
            "relative_path"
        ] == SEALED_CALLBACK_GUARD_RELATIVE_PATH
        assert finalization_receipt.sealed_callback_guard_identity[
            "record_count"
        ] >= 1
        with pytest.raises(TypeError):
            finalization_receipt.sealed_callback_guard_identity[
                "record_count"
            ] = 0
        sealed = verify_sealed_stream(sink.path)
        assert [row["event_kind"] for row in sealed] == [
            "after_reset",
            "after_normal_action",
            "episode_final",
        ]
        assert sealed[0]["event_id"] == reset_signal.event_id
        assert sealed[0]["opaque_token_sha256"] == reset_signal.token_sha256
        assert sealed[1]["event_id"] == terminal.event_id
        assert sealed[1]["opaque_token_sha256"] == terminal.token_sha256
        assert sealed[2]["event_id"] == signal.event_id
        assert sealed[2]["opaque_token_sha256"] == signal.token_sha256
        assert sealed[-1]["event_kind"] == "episode_final"
        # The child finalizer observed the action after runtime_close, proving
        # that decision-plane close did not destroy the browser state first.
        assert sealed[-1]["evidence"]["task_success"] is True
        with pytest.raises(ProcessBrokerProtocolError, match="consumed"):
            finalizer.finalize_episode(
                episode_summary=_summary(TerminalReason.ABORT),
            )
        with pytest.raises(RuntimeError, match="issued once"):
            broker.finalization_client()
    finally:
        broker.stop()
    assert broker.cleaned
    cleanup = broker.cleanup_receipt
    assert cleanup.transcript_entry_count == 16
    assert cleanup.shutdown_request_previous_transcript_entry_count == (
        finalization_receipt.transcript_entry_count
    )
    assert cleanup.shutdown_request_previous_transcript_root_sha256 == (
        finalization_receipt.transcript_root_sha256
    )
    assert cleanup.transcript_entry_count == (
        cleanup.shutdown_request_previous_transcript_entry_count + 2
    )
    assert cleanup.child_cleanup_result["cleanup_disposition"] == (
        "RUNTIME_CLOSE_ACKNOWLEDGED"
    )
    assert cleanup.child_cleanup_result["browser_close_completed"] is True
    assert cleanup.child_cleanup_result["manual_rescue_check_count"] == 0
    assert cleanup.child_cleanup_result["manual_rescue_tail_sha256"] is None


@pytest.mark.parametrize("reason", [TerminalReason.TIMEOUT, TerminalReason.ABORT])
def test_finalization_atomically_ends_runtime_when_timeout_skips_runtime_close(
    tmp_path: Path,
    reason: TerminalReason,
) -> None:
    task = _task()
    broker, _runtime_dir, sink, _target = _broker(tmp_path, task)
    broker.start()
    runtime = broker.runtime_client()
    adapter = ProcessBrokerWebArenaEnvironmentAdapter(
        client=runtime,
        benchmark_version=task.benchmark_version,
    )
    reset = adapter.reset(task, episode_id="episode-1", seed=42)
    assert adapter.terminal_signal(task, _binding(reset)).terminate is False
    # Deliberately omit adapter.close: Executor does this when its absolute
    # episode deadline has already expired.  Orchestration must still finalize.
    signal = broker.finalization_client().finalize_episode(
        episode_summary=_summary(reason),
    )
    assert signal.terminate is True
    sealed = verify_sealed_stream(sink.path)
    assert sealed[-1]["evidence"]["task_success"] is False
    assert sealed[-1]["evidence"]["terminal_reason"] == reason.value
    with pytest.raises(ProcessBrokerProtocolError, match="rejected"):
        runtime.close(episode_id="episode-1", task_id="task-1")
    broker.stop()
    assert broker.cleaned


def test_finalization_rejects_reset_observation_without_terminal_receipt(
    tmp_path: Path,
) -> None:
    task = _task()
    broker, _runtime_dir, sink, _target = _broker(tmp_path, task)
    broker.start()
    adapter = ProcessBrokerWebArenaEnvironmentAdapter(
        client=broker.runtime_client(),
        benchmark_version=task.benchmark_version,
    )
    adapter.reset(task, episode_id="episode-1", seed=42)
    try:
        with pytest.raises(ProcessBrokerProtocolError, match="rejected"):
            broker.finalization_client().finalize_episode(
                episode_summary=_summary(TerminalReason.TIMEOUT),
            )
        assert not sink.has_episode_final
    finally:
        broker.stop()
    assert broker.cleaned


def test_finalization_rejects_executed_action_without_observation_or_terminal(
    tmp_path: Path,
) -> None:
    task = _task()
    broker, _runtime_dir, sink, _target = _broker(tmp_path, task)
    broker.start()
    adapter = ProcessBrokerWebArenaEnvironmentAdapter(
        client=broker.runtime_client(),
        benchmark_version=task.benchmark_version,
    )
    reset = adapter.reset(task, episode_id="episode-1", seed=42)
    adapter.terminal_signal(task, _binding(reset))
    adapter.execute(
        ConcreteAction(
            action_id="pending-action",
            source_decision_id="pending-decision",
            action_type=ActionType.NAVIGATE,
            parameters={"url": "https://fixture.invalid/pending"},
        )
    )
    try:
        with pytest.raises(ProcessBrokerProtocolError, match="rejected"):
            broker.finalization_client().finalize_episode(
                episode_summary=_summary(
                    TerminalReason.TIMEOUT,
                    executor_steps=1,
                ),
            )
        assert not sink.has_episode_final
    finally:
        broker.stop()
    assert broker.cleaned


def test_finalizer_exception_writes_no_evidence_and_control_cleanup_still_reaps(
    tmp_path: Path,
) -> None:
    task = _task(goal="fixture finalizer-error")
    broker, _runtime_dir, sink, _target = _broker(tmp_path, task)
    broker.start()
    runtime = broker.runtime_client()
    adapter = ProcessBrokerWebArenaEnvironmentAdapter(
        client=runtime,
        benchmark_version=task.benchmark_version,
    )
    reset = adapter.reset(task, episode_id="episode-1", seed=42)
    adapter.terminal_signal(task, _binding(reset))
    adapter.close()
    with pytest.raises(ProcessBrokerProtocolError):
        broker.finalization_client().finalize_episode(
            episode_summary=_summary(TerminalReason.ABORT),
        )
    assert not sink.has_episode_final
    broker.stop()
    assert broker.cleaned


@pytest.mark.parametrize(
    "goal,expected_records",
    [
        ("fixture transition no write", 0),
        ("fixture transition double write", 2),
        ("fixture transition write then raise", 1),
        ("fixture transition forged signal", 1),
    ],
)
def test_transition_callback_failures_preserve_prefix_and_block_finalization(
    tmp_path: Path,
    goal: str,
    expected_records: int,
) -> None:
    task = _task(goal=goal)
    broker, _runtime_dir, sink, _target = _broker(tmp_path, task)
    broker.start()
    runtime = broker.runtime_client()
    adapter = ProcessBrokerWebArenaEnvironmentAdapter(
        client=runtime,
        benchmark_version=task.benchmark_version,
    )
    reset = adapter.reset(task, episode_id="episode-1", seed=42)
    try:
        with pytest.raises(ProcessBrokerProtocolError):
            adapter.terminal_signal(task, _binding(reset))
        records = _sealed_records(sink)
        assert len(records) == expected_records
        assert [row["event_index"] for row in records] == list(
            range(expected_records)
        )
        with pytest.raises(ProcessBrokerProtocolError, match="rejected"):
            broker.finalization_client().finalize_episode(
                episode_summary=_summary(TerminalReason.ABORT)
            )
        # No rollback, retry, or final append may rewrite the durable prefix.
        assert _sealed_records(sink) == records
    finally:
        broker.stop()
    assert broker.cleaned


@pytest.mark.parametrize(
    "goal,expected_records",
    [
        ("fixture finalizer no write", 1),
        ("fixture finalizer double write", 2),
        ("fixture finalizer write then raise", 2),
        ("fixture finalizer forged signal", 2),
    ],
)
def test_final_callback_failures_preserve_prefix_and_cleanup(
    tmp_path: Path,
    goal: str,
    expected_records: int,
) -> None:
    task = _task(goal=goal)
    broker, _runtime_dir, sink, _target = _broker(tmp_path, task)
    broker.start()
    adapter = ProcessBrokerWebArenaEnvironmentAdapter(
        client=broker.runtime_client(),
        benchmark_version=task.benchmark_version,
    )
    reset = adapter.reset(task, episode_id="episode-1", seed=42)
    adapter.terminal_signal(task, _binding(reset))
    adapter.close()
    try:
        with pytest.raises(ProcessBrokerProtocolError):
            broker.finalization_client().finalize_episode(
                episode_summary=_summary(TerminalReason.ABORT)
            )
        records = _sealed_records(sink)
        assert len(records) == expected_records
        assert [row["event_index"] for row in records] == list(
            range(expected_records)
        )
        assert records[0]["event_kind"] == "after_reset"
        if expected_records == 2:
            assert records[1]["event_kind"] == "episode_final"
    finally:
        broker.stop()
    assert broker.cleaned


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("episode_id", "another-episode"),
        ("task_id", "another-task"),
        ("system_id", SystemID.E1),
        ("repeat_id", 1),
        ("model_seed", 43),
    ),
)
def test_runtime_summary_identity_is_bound_before_child_finalizer(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    task = _task()
    broker, _runtime_dir, sink, _target = _broker(tmp_path, task)
    broker.start()
    runtime = broker.runtime_client()
    adapter = ProcessBrokerWebArenaEnvironmentAdapter(
        client=runtime,
        benchmark_version=task.benchmark_version,
    )
    reset = adapter.reset(task, episode_id="episode-1", seed=42)
    adapter.terminal_signal(task, _binding(reset))
    adapter.close()
    before = _sealed_records(sink)
    wrong = replace(_summary(TerminalReason.ABORT), **{field: value})
    with pytest.raises(ProcessBrokerProtocolError, match="rejected"):
        broker.finalization_client().finalize_episode(
            episode_summary=wrong,
        )
    assert _sealed_records(sink) == before
    assert not sink.has_episode_final
    broker.stop()


def test_backend_sealed_reads_reject_replaced_ancestor_before_final_callback(
    tmp_path: Path,
) -> None:
    task = _task()
    broker, _runtime_dir, sink, _target = _broker(tmp_path, task)
    broker.start()
    adapter = ProcessBrokerWebArenaEnvironmentAdapter(
        client=broker.runtime_client(),
        benchmark_version=task.benchmark_version,
    )
    reset = adapter.reset(task, episode_id="episode-1", seed=42)
    adapter.terminal_signal(task, _binding(reset))
    adapter.close()
    before = _sealed_records(sink)
    relative_stream = sink.path.relative_to(sink.system_root)
    relocated = tmp_path / "relocated-system-root"
    sink.system_root.rename(relocated)
    sink.system_root.symlink_to(relocated, target_is_directory=True)

    try:
        with pytest.raises(ProcessBrokerProtocolError, match="rejected"):
            broker.finalization_client().finalize_episode(
                episode_summary=_summary(TerminalReason.ABORT),
            )
        assert verify_sealed_stream(relocated / relative_stream) == before
    finally:
        broker.stop()
    assert broker.cleaned


@pytest.mark.parametrize(
    "reason,execute_action",
    [
        (TerminalReason.OPAQUE_VERIFIER_TERMINAL, True),
        (TerminalReason.TIMEOUT, False),
        (TerminalReason.ABORT, False),
    ],
)
def test_production_runner_finalizes_then_cleans_process_binding(
    tmp_path: Path,
    reason: TerminalReason,
    execute_action: bool,
) -> None:
    task = _task()
    broker, runtime_dir, sink, _target = _broker(tmp_path, task)
    broker.start()
    runtime = broker.runtime_client()
    adapter = ProcessBrokerWebArenaEnvironmentAdapter(
        client=runtime,
        benchmark_version=task.benchmark_version,
    )
    reset = adapter.reset(task, episode_id="episode-1", seed=42)
    reset_signal = adapter.terminal_signal(task, _binding(reset))
    terminal = None
    if execute_action:
        action = ConcreteAction(
            action_id="action-1",
            source_decision_id="decision-1",
            action_type=ActionType.NAVIGATE,
            parameters={"url": "https://fixture.invalid/done"},
        )
        adapter.execute(action)
        post = adapter.observe(
            stage=ObservationStage.POST_ACTION,
            prior_action_id=action.action_id,
        )
        terminal = adapter.terminal_signal(task, _binding(post, action))
        adapter.close()
    finalization = ProcessIsolatedEpisodeFinalizationBinding(
        client=broker.finalization_client(),
        episode_runtime_dir=runtime_dir,
    )

    def cleanup_episode() -> dict[str, Any]:
        receipt = broker.stop()
        if receipt is None:
            raise RuntimeError("authenticated fixture cleanup receipt is absent")
        return receipt.to_dict()

    def abort_episode(episode_id: str, task_id: str) -> BrowserGymEpisodeAbortReceipt:
        cleanup = cleanup_episode()
        return BrowserGymEpisodeAbortReceipt(
            episode_id=episode_id,
            task_id=task_id,
            outcome="BROKER_ABORTED",
            underlying_browser_created=True,
            underlying_browser_close_called=True,
            broker_abort_receipt_sha256=sha256_json(cleanup),
        )

    factory_identity = {"schema_version": "fixture-process-episode-factory-v1"}
    process_binding = ProcessIsolatedWebArenaEpisodeBinding(
        benchmark_version=task.benchmark_version,
        environment_adapter=adapter,
        finalize_episode_evidence=finalization.finalize_episode_evidence,
        finalization_receipt=lambda: finalization.finalization_receipt.to_dict(),
        abort_episode=abort_episode,
        cleanup_episode=cleanup_episode,
        broker_receipt=broker.receipt.to_dict(),
        live_deployment_binding_sha256="a" * 64,
        live_deployment_manifest_sha256="b" * 64,
        measured_timeout_binding_sha256=(
            broker.receipt.ipc_timeout_binding_sha256
        ),
        episode_factory_public_identity=factory_identity,
        episode_factory_descriptor_sha256=sha256_json(factory_identity),
    )
    logs = EpisodeEventLogs(
        tmp_path / "events",
        episode_id="episode-1",
        include_memory=False,
        system_id="E0",
        task_id="task-1",
        repeat_id=0,
        matched_seed=42,
    )
    runner = object.__new__(ProductionTable2Runner)
    runner._active_webarena = {
        "episode-1": _ActiveProcessWebArenaSession(
            finalizer=process_binding.finalize_episode_evidence,
            finalization_receipt=process_binding.finalization_receipt,
            cleanup_episode=process_binding.cleanup_episode,
            abort_episode=process_binding.abort_episode,
            event_logs=logs,
            task_id="task-1",
            launch_receipt_sha256=sha256_json(broker.receipt.to_dict()),
            launch_source_set_sha256=broker.receipt.source_set_sha256,
            sealed_child_pid=broker.receipt.sealed_evaluator_pid,
        )
    }
    summary = _summary(
        reason,
        executor_steps=1 if execute_action else 0,
        verifier_signal=terminal if execute_action else None,
    )
    signal = runner.finalize_episode(
        result=summary,
        verifier_writer=SealedVerifierWriter(sink),
        episode_id="episode-1",
        runtime_dir=runtime_dir,
    )
    assert signal.terminate is True
    assert broker.cleaned
    sealed = verify_sealed_stream(sink.path)
    assert sealed[0]["event_id"] == reset_signal.event_id
    assert sealed[-1]["event_id"] == signal.event_id
    assert sealed[-1]["evidence"]["terminal_reason"] == reason.value


def test_production_runner_finalizer_error_aborts_and_reaps_process(
    tmp_path: Path,
) -> None:
    task = _task(goal="fixture finalizer-error")
    broker, runtime_dir, sink, _target = _broker(tmp_path, task)
    broker.start()
    adapter = ProcessBrokerWebArenaEnvironmentAdapter(
        client=broker.runtime_client(),
        benchmark_version=task.benchmark_version,
    )
    reset = adapter.reset(task, episode_id="episode-1", seed=42)
    adapter.terminal_signal(task, _binding(reset))
    adapter.close()
    finalization = ProcessIsolatedEpisodeFinalizationBinding(
        client=broker.finalization_client(), episode_runtime_dir=runtime_dir
    )
    aborts: list[str] = []

    def cleanup_episode() -> dict[str, Any]:
        receipt = broker.stop()
        if receipt is None:
            raise RuntimeError("authenticated fixture cleanup receipt is absent")
        return receipt.to_dict()

    def abort_episode(episode_id: str, task_id: str) -> BrowserGymEpisodeAbortReceipt:
        aborts.append(episode_id)
        cleanup = cleanup_episode()
        return BrowserGymEpisodeAbortReceipt(
            episode_id=episode_id,
            task_id=task_id,
            outcome="BROKER_ABORTED",
            underlying_browser_created=True,
            underlying_browser_close_called=True,
            broker_abort_receipt_sha256=sha256_json(cleanup),
        )

    logs = EpisodeEventLogs(
        tmp_path / "events",
        episode_id="episode-1",
        include_memory=False,
        system_id="E0",
        task_id="task-1",
        repeat_id=0,
        matched_seed=42,
    )
    runner = object.__new__(ProductionTable2Runner)
    runner._active_webarena = {
        "episode-1": _ActiveProcessWebArenaSession(
            finalizer=finalization.finalize_episode_evidence,
            finalization_receipt=lambda: (
                finalization.finalization_receipt.to_dict()
            ),
            cleanup_episode=cleanup_episode,
            abort_episode=abort_episode,
            event_logs=logs,
            task_id="task-1",
            launch_receipt_sha256=sha256_json(broker.receipt.to_dict()),
            launch_source_set_sha256=broker.receipt.source_set_sha256,
            sealed_child_pid=broker.receipt.sealed_evaluator_pid,
        )
    }
    with pytest.raises(ProcessBrokerProtocolError):
        runner.finalize_episode(
            result=_summary(TerminalReason.ABORT),
            verifier_writer=SealedVerifierWriter(sink),
            episode_id="episode-1",
            runtime_dir=runtime_dir,
        )
    assert aborts == ["episode-1"]
    assert broker.cleaned
    assert not sink.has_episode_final
