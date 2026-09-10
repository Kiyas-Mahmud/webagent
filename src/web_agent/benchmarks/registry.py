"""Explicit benchmark factory registry; importing it has no external effects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from web_agent.benchmarks.base import BenchmarkAdapter


BenchmarkFactory = Callable[..., BenchmarkAdapter]


@dataclass(slots=True)
class BenchmarkRegistry:
    _factories: dict[str, BenchmarkFactory] = field(default_factory=dict)

    def register(
        self,
        name: str,
        factory: BenchmarkFactory,
        *,
        replace: bool = False,
    ) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("benchmark registry name cannot be empty")
        if key in self._factories and not replace:
            raise ValueError(f"benchmark is already registered: {key}")
        self._factories[key] = factory

    def create(self, name: str, **kwargs: Any) -> BenchmarkAdapter:
        key = name.strip().lower()
        try:
            factory = self._factories[key]
        except KeyError as exc:
            raise KeyError(
                f"unknown benchmark {key!r}; registered={sorted(self._factories)}"
            ) from exc
        adapter = factory(**kwargs)
        if not isinstance(adapter, BenchmarkAdapter):
            raise TypeError(f"benchmark factory {key!r} returned {type(adapter).__name__}")
        return adapter

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def default_registry() -> BenchmarkRegistry:
    """Create the built-in registry without constructing/importing WebArena."""
    from web_agent.benchmarks.fixture import DeterministicFixtureAdapter
    from web_agent.benchmarks.webarena import WebArenaAdapter

    registry = BenchmarkRegistry()
    registry.register("fixture", DeterministicFixtureAdapter)
    registry.register("webarena", WebArenaAdapter)
    return registry
