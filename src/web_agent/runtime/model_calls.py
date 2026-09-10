"""Runner-owned model-call accounting and the frozen logical-call cap."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any


# A 30-request episode may dispatch one pre-action policy call, one parameter
# fallback call, and one post-action assessment per browser request (90 calls),
# plus up to four registered recovery incidents.  Each incident may dispatch a
# planner, recovery assessment, and memory-embedding call even when its
# high-level strategy produces no browser request.  The fail-closed ceiling is
# therefore 90 + (4 * 3) = 102 calls.
REGISTERED_MAX_MODEL_CALLS_PER_EPISODE = 102
REGISTERED_MODEL_CALL_STAGES = frozenset(
    {
        "pre_action_policy",
        "action_parameter_fallback",
        "post_action_assessment",
        "recovery_planner",
        "recovery_assessment",
        "memory_embedding",
    }
)


class ModelCallBudgetExceeded(RuntimeError):
    """A logical model dispatch would exceed the frozen episode cap."""


@dataclass(slots=True)
class ModelCallLedger:
    """Count model-boundary dispatches before backend code is invoked."""

    episode_id: str
    sink: Callable[[dict[str, Any]], None] | None = None
    maximum: int = REGISTERED_MAX_MODEL_CALLS_PER_EPISODE
    count: int = 0

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("model-call ledger requires an episode ID")
        if type(self.maximum) is not int or self.maximum <= 0:
            raise ValueError("model-call maximum must be a positive integer")

    def record(self, *, stage: str, component_id: str) -> None:
        if stage not in REGISTERED_MODEL_CALL_STAGES:
            raise ValueError(f"unregistered model-call stage: {stage!r}")
        if not component_id.strip():
            raise ValueError("model-call dispatch requires a component identity")
        next_index = self.count + 1
        if next_index > self.maximum:
            raise ModelCallBudgetExceeded(
                f"episode exceeded {self.maximum} logical model calls"
            )
        payload = {
            "episode_id": self.episode_id,
            "model_call_index": next_index,
            "stage": stage,
            "component_id": component_id,
            "maximum_model_calls": self.maximum,
        }
        if self.sink is not None:
            self.sink(payload)
        self.count = next_index


_ACTIVE_LEDGER: ContextVar[ModelCallLedger | None] = ContextVar(
    "table2_active_model_call_ledger",
    default=None,
)


def activate_model_call_ledger(ledger: ModelCallLedger) -> Token:
    if not isinstance(ledger, ModelCallLedger):
        raise TypeError("active model-call ledger has the wrong type")
    return _ACTIVE_LEDGER.set(ledger)


def deactivate_model_call_ledger(token: Token) -> None:
    _ACTIVE_LEDGER.reset(token)


def record_model_call(*, stage: str, component_id: str) -> None:
    """Record one model dispatch when executing inside an EpisodeRunner."""

    ledger = _ACTIVE_LEDGER.get()
    if ledger is not None:
        ledger.record(stage=stage, component_id=component_id)
