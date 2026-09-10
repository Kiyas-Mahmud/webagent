"""Typed same-process page-flow fixture for the sealed WebArena evaluator.

Runtime code may publish a page handle but has no evaluation method.  Sealed
evaluator code may evaluate a published handle but has no publication method
and receives only a write-only :class:`SealedVerifierWriter`.  Every evaluator
result is written before an opaque terminal signal crosses back to runtime.

``EpisodeRunner`` closes its adapter before campaign-level finalization.  The
runtime close capability therefore records a deferred close request and keeps
the page alive only for the sealed final evaluator.  Final success, final
failure, and explicit orchestration abort all invoke the registered underlying
close callback exactly once. Publication also requires a receipt proving that
all six upstream WebArena start-state fields were applied by a custom wrapper;
``GenericWebArenaTask`` is explicitly noncompliant because it ignores the
registered ``storage_state`` and ``require_reset`` semantics.

This is a reviewed-code API/dataflow separation, not a confidentiality or
process boundary.  A Python participant in the same interpreter can import the
other capability or introspect private state.  Consequently this implementation
is unpromotable engineering evidence and the canonical live runner fails before
provider import.  A future production broker must use a separately authenticated
process-isolated transport and evidence schema.  Bound checks still require the
page digest to remain unchanged because the policy may continue; the final
fixture evaluator may perform its registered HTML navigation only after the
adapter-close request, immediately before unconditional cleanup.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any

from web_agent.runtime.contracts import (
    EpisodeSummary,
    JsonValue,
    OpaqueTerminalSignal,
    RuntimeStartState,
    VerifierReceiptBinding,
    VersionedRecord,
)

from .live_deployment import (
    REGISTERED_BROWSERGYM_VERSION,
    REGISTERED_VALIDATION_DISABLED_EXECUTION_PATH,
    UPSTREAM_WEBARENA_START_STATE_FIELDS,
)
from .sealed_verifier import SealedVerifierWriter


PageStateDigest = Callable[[Any], str]
LivePageCleanup = Callable[[], None]


def _sha256(value: object, *, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class BrowserGymValidationDisabledBoundary(VersionedRecord):
    """Attestation required before a live page can enter the broker."""

    boundary_id: str
    boundary_version: str
    source_sha256: str
    browsergym_version: str = REGISTERED_BROWSERGYM_VERSION
    execution_path: str = REGISTERED_VALIDATION_DISABLED_EXECUTION_PATH
    default_step_called: bool = False
    task_validation_called: bool = False
    reward_slot_read: bool = False
    termination_slots_read: bool = False
    info_slot_read: bool = False
    oracle_output_exposed_to_runtime: bool = False
    upstream_start_state_fields: tuple[str, ...] = (
        UPSTREAM_WEBARENA_START_STATE_FIELDS
    )
    generic_webarena_task_used: bool = False
    adapter_close_deferred_until_sealed_finalization: bool = True
    cleanup_on_final_success: bool = True
    cleanup_on_final_error: bool = True
    cleanup_on_runtime_abort: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.boundary_id) is not str
            or not self.boundary_id.strip()
            or type(self.boundary_version) is not str
            or not self.boundary_version.strip()
        ):
            raise ValueError("validation-disabled boundary identity/version are required")
        _sha256(self.source_sha256, field="validation-disabled boundary source")
        expected = {
            "browsergym_version": REGISTERED_BROWSERGYM_VERSION,
            "execution_path": REGISTERED_VALIDATION_DISABLED_EXECUTION_PATH,
            "default_step_called": False,
            "task_validation_called": False,
            "reward_slot_read": False,
            "termination_slots_read": False,
            "info_slot_read": False,
            "oracle_output_exposed_to_runtime": False,
            "upstream_start_state_fields": (
                UPSTREAM_WEBARENA_START_STATE_FIELDS
            ),
            "generic_webarena_task_used": False,
            "adapter_close_deferred_until_sealed_finalization": True,
            "cleanup_on_final_success": True,
            "cleanup_on_final_error": True,
            "cleanup_on_runtime_abort": True,
        }
        for field, value in expected.items():
            actual = getattr(self, field)
            if type(actual) is not type(value) or actual != value:
                raise ValueError(
                    "BrowserGym runtime execution must bypass task validation and "
                    f"oracle-bearing result slots; invalid {field}"
                )


@dataclass(frozen=True, slots=True)
class BrowserGymStartStateApplicationReceipt(VersionedRecord):
    """Bind the live page to all six upstream WebArena start-state fields."""

    episode_id: str
    task_id: str
    runtime_start_state: RuntimeStartState
    applied_start_state_sha256: str
    applier_id: str
    applier_version: str
    applier_source_sha256: str
    applied_fields: tuple[str, ...] = UPSTREAM_WEBARENA_START_STATE_FIELDS
    generic_webarena_task_used: bool = False
    credential_material_returned_to_runtime: bool = False

    def __post_init__(self) -> None:
        for field in ("episode_id", "task_id", "applier_id", "applier_version"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"start-state application requires {field}")
        if type(self.runtime_start_state) is not RuntimeStartState:
            raise TypeError("start-state application requires RuntimeStartState")
        _sha256(self.applier_source_sha256, field="start-state applier source")
        _sha256(self.applied_start_state_sha256, field="applied start state")
        if self.applied_start_state_sha256 != self.runtime_start_state.start_state_sha256:
            raise ValueError("applied start-state hash differs from all six frozen fields")
        if (
            type(self.applied_fields) is not tuple
            or self.applied_fields != UPSTREAM_WEBARENA_START_STATE_FIELDS
        ):
            raise ValueError("all six upstream WebArena start-state fields must be applied")
        if self.generic_webarena_task_used is not False:
            raise ValueError(
                "GenericWebArenaTask ignores registered storage_state/require_reset semantics"
            )
        if self.credential_material_returned_to_runtime is not False:
            raise ValueError("start-state application returned credential material to runtime")


@dataclass(frozen=True, slots=True)
class PageRegistrationReceipt(VersionedRecord):
    session_id: str
    episode_id: str
    task_id: str
    validation_boundary_sha256: str
    start_state_application_sha256: str
    initial_page_state_sha256: str
    adapter_close_deferred_until_sealed_finalization: bool = True
    live_page_returned: bool = False

    def __post_init__(self) -> None:
        for field in ("session_id", "episode_id", "task_id"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"page registration requires {field}")
        _sha256(self.validation_boundary_sha256, field="validation boundary")
        _sha256(self.start_state_application_sha256, field="start-state application")
        _sha256(self.initial_page_state_sha256, field="initial page state")
        if self.adapter_close_deferred_until_sealed_finalization is not True:
            raise ValueError("page registration must defer the underlying browser close")
        if self.live_page_returned is not False:
            raise ValueError("runtime page registration cannot return the live page")


@dataclass(frozen=True, slots=True)
class PageSessionCloseReceipt(VersionedRecord):
    session_id: str
    episode_id: str
    task_id: str
    final_page_state_sha256: str
    close_requested: bool = True
    underlying_browser_close_called: bool = False
    deferred_until_sealed_finalization: bool = True
    live_page_returned: bool = False

    def __post_init__(self) -> None:
        for field in ("session_id", "episode_id", "task_id"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"page close receipt requires {field}")
        _sha256(self.final_page_state_sha256, field="final page state")
        if self.close_requested is not True:
            raise ValueError("adapter close receipt must record the close request")
        if self.underlying_browser_close_called is not False:
            raise ValueError("ordinary adapter close cannot close the live browser")
        if self.deferred_until_sealed_finalization is not True:
            raise ValueError("adapter close must be deferred until sealed finalization")
        if self.live_page_returned is not False:
            raise ValueError("runtime page close cannot return the live page")


@dataclass(frozen=True, slots=True)
class PageSessionAbortReceipt(VersionedRecord):
    session_id: str
    episode_id: str
    task_id: str
    final_page_state_sha256: str
    cleanup_reason: str
    underlying_browser_close_called: bool = True
    live_page_returned: bool = False

    def __post_init__(self) -> None:
        for field in ("session_id", "episode_id", "task_id"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"page abort receipt requires {field}")
        _sha256(self.final_page_state_sha256, field="aborted page state")
        if self.cleanup_reason not in {
            "runtime_abort",
            "bound_evaluation_error",
        }:
            raise ValueError("page abort receipt has an unregistered cleanup reason")
        if self.underlying_browser_close_called is not True:
            raise ValueError("page abort must invoke the underlying browser close")
        if self.live_page_returned is not False:
            raise ValueError("runtime page abort cannot return the live page")


@dataclass(frozen=True, slots=True)
class SealedBoundPageEvaluationRequest(VersionedRecord):
    session_id: str
    episode_id: str
    task_id: str
    binding: VerifierReceiptBinding

    def __post_init__(self) -> None:
        for field in ("session_id", "episode_id", "task_id"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"bound page evaluation requires {field}")
        if type(self.binding) is not VerifierReceiptBinding:
            raise TypeError("bound page evaluation requires VerifierReceiptBinding")


@dataclass(frozen=True, slots=True)
class SealedFinalPageEvaluationRequest(VersionedRecord):
    session_id: str
    episode_id: str
    task_id: str
    summary: EpisodeSummary

    def __post_init__(self) -> None:
        for field in ("session_id", "episode_id", "task_id"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"final page evaluation requires {field}")
        if type(self.summary) is not EpisodeSummary:
            raise TypeError("final page evaluation requires EpisodeSummary")
        if self.summary.episode_id != self.episode_id or self.summary.task_id != self.task_id:
            raise ValueError("final page evaluation summary identity mismatch")


@dataclass(frozen=True, slots=True)
class SealedBoundEvaluation(VersionedRecord):
    verification: Mapping[str, JsonValue]
    should_terminate: bool

    def __post_init__(self) -> None:
        if not isinstance(self.verification, Mapping) or not self.verification:
            raise ValueError("sealed bound evaluation requires verifier evidence")
        if type(self.should_terminate) is not bool:
            raise TypeError("sealed bound evaluation termination must be boolean")
        # Canonical record hashing validates recursive JSON/finiteness now.
        _ = self.record_sha256


@dataclass(frozen=True, slots=True)
class SealedFinalEvaluation(VersionedRecord):
    verification: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if not isinstance(self.verification, Mapping) or not self.verification:
            raise ValueError("sealed final evaluation requires verifier evidence")
        _ = self.record_sha256


BoundWebArenaEvaluator = Callable[
    [Any, SealedBoundPageEvaluationRequest], SealedBoundEvaluation
]
FinalWebArenaEvaluator = Callable[
    [Any, SealedFinalPageEvaluationRequest], SealedFinalEvaluation
]


@dataclass(slots=True)
class _PageSession:
    episode_id: str
    task_id: str
    page: Any
    digest: PageStateDigest
    close_live_page: LivePageCleanup
    last_page_state_sha256: str
    close_requested: bool = False
    cleanup_called: bool = False
    finalized: bool = False


class _OneWayPageBrokerState:
    def __init__(self) -> None:
        self._sessions: dict[str, _PageSession] = {}
        self._finalized_sessions: set[tuple[str, str, str]] = set()
        self._lock = RLock()

    @staticmethod
    def _digest(session: _PageSession) -> str:
        return _sha256(session.digest(session.page), field="live page state")

    def publish(
        self,
        *,
        session_id: str,
        episode_id: str,
        task_id: str,
        page: Any,
        page_state_digest: PageStateDigest,
        close_live_page: LivePageCleanup,
        validation_boundary: BrowserGymValidationDisabledBoundary,
        start_state_application: BrowserGymStartStateApplicationReceipt,
    ) -> PageRegistrationReceipt:
        if page is None or not callable(page_state_digest):
            raise TypeError("page publication requires a live page and state digester")
        if not callable(close_live_page):
            raise TypeError(
                "page publication requires an underlying browser close callback"
            )
        session = _PageSession(
            episode_id=episode_id,
            task_id=task_id,
            page=page,
            digest=page_state_digest,
            close_live_page=close_live_page,
            last_page_state_sha256="0" * 64,
        )
        try:
            if type(validation_boundary) is not BrowserGymValidationDisabledBoundary:
                raise TypeError("page publication requires the validation-disabled boundary")
            if type(start_state_application) is not BrowserGymStartStateApplicationReceipt:
                raise TypeError("page publication requires the six-field start-state receipt")
            if (
                start_state_application.episode_id != episode_id
                or start_state_application.task_id != task_id
            ):
                raise ValueError("page/start-state application identity mismatch")
            with self._lock:
                identity = (session_id, episode_id, task_id)
                if session_id in self._sessions or identity in self._finalized_sessions:
                    raise ValueError("sealed page session cannot be overwritten")
                if any(
                    item.episode_id == episode_id or item.task_id == task_id
                    for item in self._sessions.values()
                ):
                    raise ValueError("episode/task already owns a sealed live-page session")
                digest = self._digest(session)
                session.last_page_state_sha256 = digest
                receipt = PageRegistrationReceipt(
                    session_id=session_id,
                    episode_id=episode_id,
                    task_id=task_id,
                    validation_boundary_sha256=validation_boundary.record_sha256,
                    start_state_application_sha256=(
                        start_state_application.record_sha256
                    ),
                    initial_page_state_sha256=digest,
                )
                self._sessions[session_id] = session
                return receipt
        except BaseException as primary_error:
            try:
                self._cleanup(session)
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "page publication failed and underlying browser cleanup also failed"
                ) from cleanup_error
            raise primary_error

    def _session(self, session_id: str, episode_id: str, task_id: str) -> _PageSession:
        try:
            session = self._sessions[session_id]
        except KeyError as exc:
            if (session_id, episode_id, task_id) in self._finalized_sessions:
                raise ValueError(
                    "sealed page session may be finalized exactly once"
                ) from exc
            raise ValueError("sealed page session is not registered") from exc
        if session.episode_id != episode_id or session.task_id != task_id:
            raise ValueError("sealed page session identity mismatch")
        return session

    @staticmethod
    def _cleanup(session: _PageSession) -> None:
        if session.cleanup_called:
            raise RuntimeError("underlying live-page close may be invoked exactly once")
        session.cleanup_called = True
        callback = session.close_live_page
        try:
            result = callback()
            if result is not None:
                raise TypeError("underlying live-page close callback must return None")
        finally:
            # Drop every broker-owned live capability even when the external
            # close reports failure. The raised error still invalidates the run.
            session.page = None
            session.close_live_page = lambda: None

    def _cleanup_after_failure(
        self,
        *,
        session_id: str,
        session: _PageSession,
        primary_error: BaseException,
        retain_for_adapter_close: bool,
    ) -> None:
        try:
            self._cleanup(session)
        except BaseException as cleanup_error:
            self._sessions.pop(session_id, None)
            raise RuntimeError(
                "sealed page operation failed and underlying browser cleanup also failed"
            ) from cleanup_error
        if not retain_for_adapter_close:
            self._sessions.pop(session_id, None)
        raise primary_error

    def evaluate_bound(
        self,
        request: SealedBoundPageEvaluationRequest,
        *,
        writer: SealedVerifierWriter,
        evaluator: BoundWebArenaEvaluator,
    ) -> OpaqueTerminalSignal:
        if type(request) is not SealedBoundPageEvaluationRequest:
            raise TypeError("sealed broker requires a bound evaluation request")
        if type(writer) is not SealedVerifierWriter or not callable(evaluator):
            raise TypeError("sealed broker requires a write-only writer and evaluator")
        with self._lock:
            session = self._session(request.session_id, request.episode_id, request.task_id)
            if session.finalized:
                raise ValueError("sealed page session was already finalized")
            try:
                if session.close_requested:
                    raise ValueError(
                        "bound evaluation is forbidden after deferred adapter close"
                    )
                if session.cleanup_called:
                    raise RuntimeError("sealed page session was already cleaned")
                before = self._digest(session)
                session.last_page_state_sha256 = before
                result = evaluator(session.page, request)
                after = self._digest(session)
                session.last_page_state_sha256 = after
                if after != before:
                    raise RuntimeError(
                        "WebArena evaluator mutated the registered live page"
                    )
                if type(result) is not SealedBoundEvaluation:
                    raise TypeError(
                        "WebArena evaluator returned an unsealed bound result"
                    )
                signal = writer.record_bound_receipt(
                    request.binding,
                    result.verification,
                    should_terminate=result.should_terminate,
                )
                if type(signal) is not OpaqueTerminalSignal:
                    raise TypeError("sealed writer returned a non-opaque runtime result")
                return signal
            except BaseException as exc:
                self._cleanup_after_failure(
                    session_id=request.session_id,
                    session=session,
                    primary_error=exc,
                    retain_for_adapter_close=True,
                )
                raise AssertionError("unreachable")

    def evaluate_final(
        self,
        request: SealedFinalPageEvaluationRequest,
        *,
        writer: SealedVerifierWriter,
        evaluator: FinalWebArenaEvaluator,
    ) -> OpaqueTerminalSignal:
        if type(request) is not SealedFinalPageEvaluationRequest:
            raise TypeError("sealed broker requires a final evaluation request")
        if type(writer) is not SealedVerifierWriter or not callable(evaluator):
            raise TypeError("sealed broker requires a write-only writer and evaluator")
        with self._lock:
            session = self._session(request.session_id, request.episode_id, request.task_id)
            if session.finalized:
                raise ValueError("sealed page session may be finalized exactly once")
            try:
                if not session.close_requested:
                    raise ValueError(
                        "sealed final evaluation requires the prior deferred adapter close"
                    )
                if session.cleanup_called:
                    raise RuntimeError("sealed page session was already cleaned")
                before = self._digest(session)
                session.last_page_state_sha256 = before
                result = evaluator(session.page, request)
                after = self._digest(session)
                session.last_page_state_sha256 = after
                if type(result) is not SealedFinalEvaluation:
                    raise TypeError(
                        "WebArena evaluator returned an unsealed final result"
                    )
                signal = writer.record_episode_final(result.verification)
                if type(signal) is not OpaqueTerminalSignal:
                    raise TypeError("sealed writer returned a non-opaque runtime result")
                session.finalized = True
                self._finalized_sessions.add(
                    (request.session_id, request.episode_id, request.task_id)
                )
                self._cleanup(session)
                del self._sessions[request.session_id]
                return signal
            except BaseException as exc:
                if session.cleanup_called:
                    self._sessions.pop(request.session_id, None)
                    raise
                self._cleanup_after_failure(
                    session_id=request.session_id,
                    session=session,
                    primary_error=exc,
                    retain_for_adapter_close=False,
                )
                raise AssertionError("unreachable")

    def close(
        self,
        *,
        session_id: str,
        episode_id: str,
        task_id: str,
    ) -> PageSessionCloseReceipt | PageSessionAbortReceipt:
        with self._lock:
            session = self._session(session_id, episode_id, task_id)
            if session.close_requested:
                raise ValueError("runtime adapter close may be requested exactly once")
            if session.cleanup_called:
                del self._sessions[session_id]
                return PageSessionAbortReceipt(
                    session_id=session_id,
                    episode_id=episode_id,
                    task_id=task_id,
                    final_page_state_sha256=session.last_page_state_sha256,
                    cleanup_reason="bound_evaluation_error",
                )
            try:
                digest = self._digest(session)
            except BaseException as exc:
                self._cleanup_after_failure(
                    session_id=session_id,
                    session=session,
                    primary_error=exc,
                    retain_for_adapter_close=False,
                )
                raise AssertionError("unreachable")
            session.last_page_state_sha256 = digest
            session.close_requested = True
            return PageSessionCloseReceipt(
                session_id=session_id,
                episode_id=episode_id,
                task_id=task_id,
                final_page_state_sha256=digest,
            )

    def abort(
        self,
        *,
        session_id: str,
        episode_id: str,
        task_id: str,
    ) -> PageSessionAbortReceipt:
        """Clean a live page when orchestration cannot reach sealed finalization."""

        with self._lock:
            session = self._session(session_id, episode_id, task_id)
            if session.cleanup_called:
                raise RuntimeError("underlying live-page close may be invoked exactly once")
            try:
                digest = self._digest(session)
                session.last_page_state_sha256 = digest
            except BaseException as exc:
                self._cleanup_after_failure(
                    session_id=session_id,
                    session=session,
                    primary_error=exc,
                    retain_for_adapter_close=False,
                )
                raise AssertionError("unreachable")
            try:
                self._cleanup(session)
            finally:
                self._sessions.pop(session_id, None)
            return PageSessionAbortReceipt(
                session_id=session_id,
                episode_id=episode_id,
                task_id=task_id,
                final_page_state_sha256=digest,
                cleanup_reason="runtime_abort",
            )


class RuntimePagePublisher:
    """Runtime-only capability: publish/defer-close/abort, never evaluate."""

    __slots__ = ("__publish", "__close", "__abort")

    def __init__(self, state: _OneWayPageBrokerState) -> None:
        object.__setattr__(self, "_RuntimePagePublisher__publish", state.publish)
        object.__setattr__(self, "_RuntimePagePublisher__close", state.close)
        object.__setattr__(self, "_RuntimePagePublisher__abort", state.abort)

    def __getattribute__(self, name: str) -> Any:
        if name in {"publish", "close", "abort", "__class__", "__doc__"}:
            return object.__getattribute__(self, name)
        raise AttributeError(
            "runtime page publisher exposes publish/deferred-close/abort only"
        )

    def publish(self, **kwargs: Any) -> PageRegistrationReceipt:
        callback = object.__getattribute__(self, "_RuntimePagePublisher__publish")
        return callback(**kwargs)

    def close(
        self, **kwargs: Any
    ) -> PageSessionCloseReceipt | PageSessionAbortReceipt:
        callback = object.__getattribute__(self, "_RuntimePagePublisher__close")
        return callback(**kwargs)

    def abort(self, **kwargs: Any) -> PageSessionAbortReceipt:
        callback = object.__getattribute__(self, "_RuntimePagePublisher__abort")
        return callback(**kwargs)


class SealedPageEvaluatorCapability:
    """Evaluator-only capability: sealed evaluation, with no page publication."""

    __slots__ = ("__bound", "__final")

    def __init__(self, state: _OneWayPageBrokerState) -> None:
        object.__setattr__(self, "_SealedPageEvaluatorCapability__bound", state.evaluate_bound)
        object.__setattr__(self, "_SealedPageEvaluatorCapability__final", state.evaluate_final)

    def __getattribute__(self, name: str) -> Any:
        if name in {"evaluate_bound", "evaluate_final", "__class__", "__doc__"}:
            return object.__getattribute__(self, name)
        raise AttributeError("sealed page evaluator exposes evaluation methods only")

    def evaluate_bound(
        self,
        request: SealedBoundPageEvaluationRequest,
        **kwargs: Any,
    ) -> OpaqueTerminalSignal:
        callback = object.__getattribute__(
            self, "_SealedPageEvaluatorCapability__bound"
        )
        return callback(request, **kwargs)

    def evaluate_final(
        self,
        request: SealedFinalPageEvaluationRequest,
        **kwargs: Any,
    ) -> OpaqueTerminalSignal:
        callback = object.__getattribute__(
            self, "_SealedPageEvaluatorCapability__final"
        )
        return callback(request, **kwargs)


def create_one_way_sealed_page_broker() -> tuple[
    RuntimePagePublisher,
    SealedPageEvaluatorCapability,
]:
    """Create typed fixture capabilities over one same-process private state."""

    state = _OneWayPageBrokerState()
    return RuntimePagePublisher(state), SealedPageEvaluatorCapability(state)
