"""Dependency-free benchmark interface used by the runtime executor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

from web_agent.runtime.contracts import (
    ConcreteAction,
    ExecutionEvidence,
    ExecutionStatus,
    Observation,
    ObservationStage,
    OpaqueTerminalSignal,
    TaskSpecification,
    VerifierReceiptBinding,
    VersionedRecord,
)


class BenchmarkUnavailableError(RuntimeError):
    pass


class BenchmarkStateError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AdapterExecution(VersionedRecord):
    """Raw adapter result before the shared executor attaches budget state."""

    status: ExecutionStatus
    state_changed: bool
    environment_error: bool = False
    error_kind: str | None = None
    message: str = ""
    internal_retry_count: int = 0
    latency_ms: float = 0.0
    evidence: ExecutionEvidence | None = None

    def __post_init__(self) -> None:
        if self.internal_retry_count < 0:
            raise ValueError("internal retry count cannot be negative")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(float(self.latency_ms))
            or self.latency_ms < 0
        ):
            raise ValueError("adapter execution latency must be nonnegative")
        if self.evidence is not None:
            if type(self.evidence) is not ExecutionEvidence:
                raise TypeError("adapter execution evidence has the wrong type")
            if self.evidence.status is not self.status:
                raise ValueError("adapter execution contradicts its evidence")


class BenchmarkAdapter(ABC):
    """Minimal controlled browser/task environment contract."""

    benchmark_id: str
    benchmark_version: str

    @abstractmethod
    def reset(
        self,
        task: TaskSpecification,
        *,
        episode_id: str,
        seed: int,
    ) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def observe(
        self,
        *,
        stage: ObservationStage,
        prior_action_id: str | None = None,
    ) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def execute(self, action: ConcreteAction) -> AdapterExecution:
        raise NotImplementedError

    @abstractmethod
    def terminal_signal(
        self,
        task: TaskSpecification,
        binding: VerifierReceiptBinding | None = None,
    ) -> OpaqueTerminalSignal:
        """Return only an opaque result for the supplied causal receipt binding.

        ``None`` remains available to small adapter unit tests, but canonical
        ``EpisodeRunner`` execution always supplies an exact reset/action
        binding and campaign validation rejects unbound/final-only evidence.
        """
        raise NotImplementedError

    def reset_state_receipt(self) -> VersionedRecord | None:
        """Return typed hashes-only environment-reset evidence when available.

        Generic fixtures are not required to implement the WebArena-specific
        receipt.  Evaluation-mode ordinary WebArena execution fails closed if
        its adapter returns ``None``.
        """

        return None

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "BenchmarkAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# Public plan-level name retained as an exact compatibility alias.  Existing
# integrations may continue to import ``BenchmarkAdapter`` while Table 2 code
# and external adapters can use the architecture's ``EnvironmentAdapter``
# contract without creating a second, divergent interface.
EnvironmentAdapter = BenchmarkAdapter
