"""Benchmark adapters for the Table 2 browser runtime."""

from web_agent.benchmarks.base import (
    AdapterExecution,
    BenchmarkAdapter,
    BenchmarkStateError,
    BenchmarkUnavailableError,
    EnvironmentAdapter,
)
from web_agent.benchmarks.fixture import (
    DeterministicFixtureAdapter,
    FixtureAdapter,
    FixtureScenario,
    default_fixture_scenario,
    run_all_systems_fixture_smoke,
)
from web_agent.benchmarks.registry import BenchmarkRegistry, default_registry
from web_agent.benchmarks.recovery_fixture import (
    RecoveryDiagnosticAdapter,
    load_registered_recovery_diagnostics,
    run_failure_memory_intervention_smoke,
    run_recovery_diagnostic_fixture_campaign,
)

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
