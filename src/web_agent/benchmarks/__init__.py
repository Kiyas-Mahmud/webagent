"""Benchmark adapters for the Table 2 browser runtime.

Public names are resolved lazily.  This keeps importing a narrow benchmark
contract (notably the process-broker wire schema) from executing fixture,
recovery, and registry modules that are outside that contract's attested source
closure.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AdapterExecution",
    "BenchmarkAdapter",
    "BenchmarkRegistry",
    "BenchmarkStateError",
    "BenchmarkUnavailableError",
    "EnvironmentAdapter",
    "DeterministicFixtureAdapter",
    "FixtureAdapter",
    "FixtureScenario",
    "RecoveryDiagnosticAdapter",
    "default_fixture_scenario",
    "default_registry",
    "run_all_systems_fixture_smoke",
    "load_registered_recovery_diagnostics",
    "run_failure_memory_intervention_smoke",
    "run_recovery_diagnostic_fixture_campaign",
]


_EXPORT_MODULE = {
    "AdapterExecution": "web_agent.benchmarks.base",
    "BenchmarkAdapter": "web_agent.benchmarks.base",
    "BenchmarkStateError": "web_agent.benchmarks.base",
    "BenchmarkUnavailableError": "web_agent.benchmarks.base",
    "EnvironmentAdapter": "web_agent.benchmarks.base",
    "DeterministicFixtureAdapter": "web_agent.benchmarks.fixture",
    "FixtureAdapter": "web_agent.benchmarks.fixture",
    "FixtureScenario": "web_agent.benchmarks.fixture",
    "default_fixture_scenario": "web_agent.benchmarks.fixture",
    "run_all_systems_fixture_smoke": "web_agent.benchmarks.fixture",
    "BenchmarkRegistry": "web_agent.benchmarks.registry",
    "default_registry": "web_agent.benchmarks.registry",
    "RecoveryDiagnosticAdapter": "web_agent.benchmarks.recovery_fixture",
    "load_registered_recovery_diagnostics": "web_agent.benchmarks.recovery_fixture",
    "run_failure_memory_intervention_smoke": "web_agent.benchmarks.recovery_fixture",
    "run_recovery_diagnostic_fixture_campaign": (
        "web_agent.benchmarks.recovery_fixture"
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULE[name]
    except KeyError as exc:  # pragma: no cover - standard module behavior
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
