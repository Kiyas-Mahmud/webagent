"""Typed, oracle-blind reset boundary for shared production model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from time import monotonic
from typing import Any

from web_agent.runtime.contracts import (
    SystemID,
    VersionedRecord,
    canonical_sha256,
)
from web_agent.runtime.deadline import interrupt_after


PROCESS_BROKER_IMPORT_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).resolve().read_bytes()
).hexdigest()

REGISTERED_STATEFUL_BACKEND_ROLES = (
    "selected_checkpoint_policy",
    "selected_backbone_policy",
    "parameter_provider",
    "recovery_planner",
    "post_failure_embedding",
)

REGISTERED_WEBARENA_RESET_COMMITMENTS = (
    "service",
    "account",
    "database",
    "start_state",
)


class PreBrowserSetupTimeout(RuntimeError):
    """The frozen pre-browser setup boundary was reached or exceeded."""


def _require_sha256(name: str, value: object) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


@dataclass(frozen=True, slots=True)
class WebArenaResetStateRequest(VersionedRecord):
    """Oracle-blind identity presented to the reset-state attester.

    The capability deliberately receives neither evaluator configuration nor
    success, progress, failure, relevance, reward, or reference-trajectory
    data.  ``start_state_id`` is already the canonical hash of the immutable
    :class:`~web_agent.runtime.contracts.RuntimeStartState` carried by the
    environment-facing :class:`TaskSpecification`.  The receipt therefore
    binds the reset attestation to every registered public reset input without
    receiving credential contents.
    """

    episode_id: str
    task_id: str
    benchmark_version: str
    start_state_id: str
    reset_stage_seed: int

    def __post_init__(self) -> None:
        for name in ("episode_id", "task_id", "benchmark_version"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"WebArena reset-state request requires {name}")
        _require_sha256("start_state_id", self.start_state_id)
        if isinstance(self.reset_stage_seed, bool) or not isinstance(
            self.reset_stage_seed, int
        ):
            raise TypeError("WebArena reset-state seed must be an exact integer")
        if self.reset_stage_seed < 0:
            raise ValueError("WebArena reset-state seed cannot be negative")


def webarena_reset_state_digest(
    *,
    attester_id: str,
    attester_version: str,
    attester_source_sha256: str,
    task_id: str,
    benchmark_version: str,
    start_state_id: str,
    reset_stage_seed: int,
    commitments_sha256: Mapping[str, str],
) -> str:
    """Return the commitment that must be identical in a matched E0--E3 block."""

    return canonical_sha256(
        {
            "attester_id": attester_id,
            "attester_version": attester_version,
            "attester_source_sha256": attester_source_sha256,
            "task_id": task_id,
            "benchmark_version": benchmark_version,
            "start_state_id": start_state_id,
            "reset_stage_seed": reset_stage_seed,
            "commitments_sha256": {
                role: commitments_sha256[role]
                for role in REGISTERED_WEBARENA_RESET_COMMITMENTS
            },
        }
    )


@dataclass(frozen=True, slots=True)
class WebArenaResetStateReceipt(VersionedRecord):
    """Source-produced, hashes-only receipt for the hidden WebArena baseline."""

    episode_id: str
    task_id: str
    benchmark_version: str
    start_state_id: str
    reset_stage_seed: int
    attester_id: str
    attester_version: str
    attester_source_sha256: str
    commitments_sha256: Mapping[str, str]
    reset_state_sha256: str
    reset_applied: bool = True
    oracle_labels_observed: bool = False

    def __post_init__(self) -> None:
        request = WebArenaResetStateRequest(
            episode_id=self.episode_id,
            task_id=self.task_id,
            benchmark_version=self.benchmark_version,
            start_state_id=self.start_state_id,
            reset_stage_seed=self.reset_stage_seed,
        )
        del request
        if not self.attester_id.strip() or not self.attester_version.strip():
            raise ValueError("WebArena reset-state attester identity/version are required")
        _require_sha256("attester_source_sha256", self.attester_source_sha256)
        if self.reset_applied is not True or type(self.reset_applied) is not bool:
            raise ValueError("WebArena reset-state receipt must attest reset_applied=true")
        if (
            self.oracle_labels_observed is not False
            or type(self.oracle_labels_observed) is not bool
        ):
            raise ValueError(
                "WebArena reset-state receipt must attest oracle_labels_observed=false"
            )
        if not isinstance(self.commitments_sha256, Mapping) or set(
            self.commitments_sha256
        ) != set(REGISTERED_WEBARENA_RESET_COMMITMENTS):
            raise ValueError(
                "WebArena reset-state receipt must cover service/account/database/"
                "start_state commitments exactly"
            )
        for role in REGISTERED_WEBARENA_RESET_COMMITMENTS:
            _require_sha256(
                f"commitments_sha256[{role!r}]",
                self.commitments_sha256[role],
            )
        _require_sha256("reset_state_sha256", self.reset_state_sha256)
        expected = webarena_reset_state_digest(
            attester_id=self.attester_id,
            attester_version=self.attester_version,
            attester_source_sha256=self.attester_source_sha256,
            task_id=self.task_id,
            benchmark_version=self.benchmark_version,
            start_state_id=self.start_state_id,
            reset_stage_seed=self.reset_stage_seed,
            commitments_sha256=self.commitments_sha256,
        )
        if self.reset_state_sha256 != expected:
            raise ValueError(
                "WebArena reset-state digest does not bind the registered commitments"
            )


WebArenaResetAttestCallable = Callable[
    [WebArenaResetStateRequest], WebArenaResetStateReceipt
]


@dataclass(frozen=True, slots=True)
class FrozenWebArenaResetStateAttester:
    """Strict wrapper around the source-attested reset-service snapshot callback."""

    attester_id: str
    attester_version: str
    source_sha256: str
    callback: WebArenaResetAttestCallable
    frozen: bool = True
    oracle_labels_exposed: bool = False

    def __post_init__(self) -> None:
        if not self.attester_id.strip() or not self.attester_version.strip():
            raise ValueError("WebArena reset-state attester identity/version are required")
        _require_sha256("WebArena reset-state attester source_sha256", self.source_sha256)
        if not callable(self.callback):
            raise TypeError("WebArena reset-state attester callback must be callable")
        if self.frozen is not True or type(self.frozen) is not bool:
            raise ValueError("WebArena reset-state attester must be frozen")
        if self.oracle_labels_exposed is not False or type(
            self.oracle_labels_exposed
        ) is not bool:
            raise ValueError("WebArena reset-state attester must be oracle-blind")

    def attest(self, request: WebArenaResetStateRequest) -> WebArenaResetStateReceipt:
        if type(request) is not WebArenaResetStateRequest:
            raise TypeError("WebArena reset-state attester requires the exact request")
        receipt = self.callback(request)
        if type(receipt) is not WebArenaResetStateReceipt:
            raise TypeError("WebArena reset-state attester returned the wrong receipt")
        request_identity = (
            request.episode_id,
            request.task_id,
            request.benchmark_version,
            request.start_state_id,
            request.reset_stage_seed,
        )
        receipt_identity = (
            receipt.episode_id,
            receipt.task_id,
            receipt.benchmark_version,
            receipt.start_state_id,
            receipt.reset_stage_seed,
        )
        if receipt_identity != request_identity:
            raise ValueError("WebArena reset-state receipt cites another request")
        if (
            receipt.attester_id != self.attester_id
            or receipt.attester_version != self.attester_version
            or receipt.attester_source_sha256 != self.source_sha256
        ):
            raise ValueError("WebArena reset-state receipt identity is not source-bound")
        return receipt


def webarena_reset_receipt_from_commitments(
    request: WebArenaResetStateRequest,
    *,
    attester_id: str,
    attester_version: str,
    attester_source_sha256: str,
    commitments_sha256: Mapping[str, str],
) -> WebArenaResetStateReceipt:
    """Construct a typed receipt after the integration snapshots every hidden backend."""

    digest = webarena_reset_state_digest(
        attester_id=attester_id,
        attester_version=attester_version,
        attester_source_sha256=attester_source_sha256,
        task_id=request.task_id,
        benchmark_version=request.benchmark_version,
        start_state_id=request.start_state_id,
        reset_stage_seed=request.reset_stage_seed,
        commitments_sha256=commitments_sha256,
    )
    return WebArenaResetStateReceipt(
        episode_id=request.episode_id,
        task_id=request.task_id,
        benchmark_version=request.benchmark_version,
        start_state_id=request.start_state_id,
        reset_stage_seed=request.reset_stage_seed,
        attester_id=attester_id,
        attester_version=attester_version,
        attester_source_sha256=attester_source_sha256,
        commitments_sha256=dict(commitments_sha256),
        reset_state_sha256=digest,
    )


@dataclass(frozen=True, slots=True)
class PreBrowserSetupEvidence(VersionedRecord):
    """Measured completion of all per-episode work before browser reset begins."""

    episode_id: str
    system_id: SystemID
    timeout_seconds: float
    elapsed_seconds: float
    boundary: str = "immediately_before_webarena_environment_reset"
    completed_within_budget: bool = True

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("pre-browser setup evidence requires an episode identity")
        for name in ("timeout_seconds", "elapsed_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"pre-browser setup {name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"pre-browser setup {name} must be finite/nonnegative")
        if float(self.timeout_seconds) <= 0:
            raise ValueError("pre-browser setup timeout must be positive")
        if self.elapsed_seconds >= self.timeout_seconds:
            raise ValueError("pre-browser setup evidence exceeds its frozen budget")
        if self.boundary != "immediately_before_webarena_environment_reset":
            raise ValueError("pre-browser setup evidence has the wrong clock boundary")
        if (
            self.completed_within_budget is not True
            or type(self.completed_within_budget) is not bool
        ):
            raise ValueError("pre-browser setup must attest completion within budget")


@dataclass(slots=True)
class PreBrowserSetupDeadline:
    """Post-call deadline guard covering every per-episode pre-browser stage."""

    timeout_seconds: float
    clock: Callable[[], float] = monotonic
    _started_at: float | None = None
    _completed: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise TypeError("pre-browser setup timeout must be numeric")
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise ValueError("pre-browser setup timeout must be finite and positive")
        if not callable(self.clock):
            raise TypeError("pre-browser setup deadline requires a clock")
        self._started_at = self.clock()

    @property
    def elapsed_seconds(self) -> float:
        assert self._started_at is not None
        return max(0.0, float(self.clock() - self._started_at))

    def require_time_remaining(self, *, operation: str) -> None:
        if self._completed:
            raise RuntimeError("pre-browser setup deadline is already completed")
        if self.elapsed_seconds >= float(self.timeout_seconds):
            raise PreBrowserSetupTimeout(
                f"pre-browser setup reached {self.timeout_seconds:.0f}s during {operation}"
            )

    def run_blocking(self, operation: str, callback: Callable[[], Any]) -> Any:
        if not operation.strip():
            raise ValueError("pre-browser setup operation cannot be empty")
        self.require_time_remaining(operation=operation)
        remaining = max(0.0, float(self.timeout_seconds) - self.elapsed_seconds)

        def timeout_error() -> PreBrowserSetupTimeout:
            return PreBrowserSetupTimeout(
                f"pre-browser setup reached {self.timeout_seconds:.0f}s during {operation}"
            )

        try:
            with interrupt_after(remaining, exception_factory=timeout_error):
                result = callback()
        except Exception as exc:
            try:
                self.require_time_remaining(operation=operation)
            except PreBrowserSetupTimeout as timeout:
                raise timeout from exc
            raise
        self.require_time_remaining(operation=operation)
        return result

    def complete(
        self,
        *,
        episode_id: str,
        system_id: SystemID,
    ) -> PreBrowserSetupEvidence:
        self.require_time_remaining(operation="browser reset boundary")
        elapsed = self.elapsed_seconds
        self._completed = True
        return PreBrowserSetupEvidence(
            episode_id=episode_id,
            system_id=system_id,
            timeout_seconds=float(self.timeout_seconds),
            elapsed_seconds=elapsed,
        )


@dataclass(frozen=True, slots=True)
class EpisodeStateResetRequest(VersionedRecord):
    """The complete capability input; deliberately contains no task or oracle."""

    episode_id: str
    system_id: SystemID
    reset_stage_seed: int

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode state reset requires an episode identity")
        if isinstance(self.reset_stage_seed, bool) or not isinstance(
            self.reset_stage_seed, int
        ):
            raise TypeError("episode state reset stage seed must be an exact integer")
        if self.reset_stage_seed < 0:
            raise ValueError("episode state reset stage seed cannot be negative")


@dataclass(frozen=True, slots=True)
class EpisodeStateResetEvidence(VersionedRecord):
    """Source-produced commitment to a fresh baseline for every shared backend."""

    episode_id: str
    system_id: SystemID
    reset_stage_seed: int
    resetter_id: str
    resetter_version: str
    resetter_source_sha256: str
    backend_state_sha256: Mapping[str, str]
    initial_state_sha256: str
    reset_applied: bool = True
    prior_state_discarded: bool = True

    def __post_init__(self) -> None:
        request = EpisodeStateResetRequest(
            episode_id=self.episode_id,
            system_id=self.system_id,
            reset_stage_seed=self.reset_stage_seed,
        )
        del request
        if not self.resetter_id.strip() or not self.resetter_version.strip():
            raise ValueError("episode state resetter identity/version are required")
        _require_sha256("resetter_source_sha256", self.resetter_source_sha256)
        if self.reset_applied is not True or type(self.reset_applied) is not bool:
            raise ValueError("episode state reset evidence must attest reset_applied=true")
        if (
            self.prior_state_discarded is not True
            or type(self.prior_state_discarded) is not bool
        ):
            raise ValueError(
                "episode state reset evidence must attest prior_state_discarded=true"
            )
        if not isinstance(self.backend_state_sha256, Mapping) or set(
            self.backend_state_sha256
        ) != set(REGISTERED_STATEFUL_BACKEND_ROLES):
            raise ValueError(
                "episode state reset evidence must cover every registered backend role"
            )
        for role in REGISTERED_STATEFUL_BACKEND_ROLES:
            _require_sha256(
                f"backend_state_sha256[{role!r}]",
                self.backend_state_sha256[role],
            )
        _require_sha256("initial_state_sha256", self.initial_state_sha256)
        expected = initial_state_digest(
            resetter_id=self.resetter_id,
            resetter_version=self.resetter_version,
            resetter_source_sha256=self.resetter_source_sha256,
            reset_stage_seed=self.reset_stage_seed,
            backend_state_sha256=self.backend_state_sha256,
        )
        if self.initial_state_sha256 != expected:
            raise ValueError(
                "initial state digest does not bind the registered backend reset state"
            )


def initial_state_digest(
    *,
    resetter_id: str,
    resetter_version: str,
    resetter_source_sha256: str,
    reset_stage_seed: int,
    backend_state_sha256: Mapping[str, str],
) -> str:
    """Digest shared across E0--E3 when their reset baselines truly match."""

    return canonical_sha256(
        {
            "resetter_id": resetter_id,
            "resetter_version": resetter_version,
            "resetter_source_sha256": resetter_source_sha256,
            "reset_stage_seed": reset_stage_seed,
            "backend_state_sha256": {
                role: backend_state_sha256[role]
                for role in REGISTERED_STATEFUL_BACKEND_ROLES
            },
        }
    )


class EpisodeStateResetter(ABC):
    """Frozen integration capability that erases shared state between episodes."""

    resetter_id: str
    resetter_version: str
    source_sha256: str
    frozen: bool = True

    @abstractmethod
    def reset(self, request: EpisodeStateResetRequest) -> EpisodeStateResetEvidence:
        raise NotImplementedError

    @abstractmethod
    def digest_backend_state(self) -> Mapping[str, str]:
        """Hash every registered backend without changing its state."""

        raise NotImplementedError


ResetCallable = Callable[[EpisodeStateResetRequest], EpisodeStateResetEvidence]
DigestCallable = Callable[[], Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class CallableEpisodeStateResetter(EpisodeStateResetter):
    """Strict wrapper for a source-attested integration reset callback."""

    resetter_id: str
    resetter_version: str
    source_sha256: str
    callback: ResetCallable
    digest_callback: DigestCallable
    frozen: bool = True

    def __post_init__(self) -> None:
        if not self.resetter_id.strip() or not self.resetter_version.strip():
            raise ValueError("episode state resetter identity/version are required")
        _require_sha256("episode state resetter source_sha256", self.source_sha256)
        if not callable(self.callback):
            raise TypeError("episode state reset callback must be callable")
        if not callable(self.digest_callback):
            raise TypeError("episode backend-state digest callback must be callable")
        if self.frozen is not True or type(self.frozen) is not bool:
            raise ValueError("production episode state resetter must be frozen")

    def reset(self, request: EpisodeStateResetRequest) -> EpisodeStateResetEvidence:
        if type(request) is not EpisodeStateResetRequest:
            raise TypeError("episode state reset requires the exact typed request")
        evidence = self.callback(request)
        if type(evidence) is not EpisodeStateResetEvidence:
            raise TypeError("episode state resetter returned the wrong evidence contract")
        if (
            evidence.episode_id != request.episode_id
            or evidence.system_id is not request.system_id
            or evidence.reset_stage_seed != request.reset_stage_seed
        ):
            raise ValueError("episode state reset evidence cites another reset request")
        if (
            evidence.resetter_id != self.resetter_id
            or evidence.resetter_version != self.resetter_version
            or evidence.resetter_source_sha256 != self.source_sha256
        ):
            raise ValueError("episode state reset evidence identity is not source-bound")
        return evidence

    def digest_backend_state(self) -> Mapping[str, str]:
        return validate_backend_state_digests(self.digest_callback())


def validate_backend_state_digests(value: object) -> dict[str, str]:
    """Return the exact five-role lowercase SHA-256 mapping or fail closed."""

    if not isinstance(value, Mapping) or set(value) != set(
        REGISTERED_STATEFUL_BACKEND_ROLES
    ):
        raise ValueError(
            "backend-state digest must cover every registered backend role"
        )
    result: dict[str, str] = {}
    for role in REGISTERED_STATEFUL_BACKEND_ROLES:
        digest = str(value[role])
        _require_sha256(f"backend_state_sha256[{role!r}]", digest)
        result[role] = digest
    return result


def evidence_from_backend_digests(
    request: EpisodeStateResetRequest,
    *,
    resetter_id: str,
    resetter_version: str,
    resetter_source_sha256: str,
    backend_state_sha256: Mapping[str, str],
) -> EpisodeStateResetEvidence:
    """Construct evidence only after the integration has reset and hashed all roles."""

    digest = initial_state_digest(
        resetter_id=resetter_id,
        resetter_version=resetter_version,
        resetter_source_sha256=resetter_source_sha256,
        reset_stage_seed=request.reset_stage_seed,
        backend_state_sha256=backend_state_sha256,
    )
    return EpisodeStateResetEvidence(
        episode_id=request.episode_id,
        system_id=request.system_id,
        reset_stage_seed=request.reset_stage_seed,
        resetter_id=resetter_id,
        resetter_version=resetter_version,
        resetter_source_sha256=resetter_source_sha256,
        backend_state_sha256=dict(backend_state_sha256),
        initial_state_sha256=digest,
    )
