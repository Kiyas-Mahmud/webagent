"""Leakage-safe end-to-end evaluation for the registered Table 2 campaign.

The public conveniences are loaded lazily.  This is a scientific boundary as
well as an import optimization: runtime modules may use dependency-free Table 2
contracts without importing the package validator, sealed evaluator, or the
DGX checkpoint gate back into the browser policy during module initialization.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "OpaqueTerminalSignal",
    "CampaignRunner",
    "SYSTEM_IDS",
    "SealedVerifier",
    "SealedVerifierSink",
    "ValidationReport",
    "build_paired_schedule",
    "compute_paired_contrasts",
    "compute_retrieval_diagnostics",
    "compute_table2_metrics",
    "validate_campaign",
]


_PUBLIC_EXPORTS = {
    "OpaqueTerminalSignal": ("sealed_verifier", "OpaqueTerminalSignal"),
    "CampaignRunner": ("campaign", "CampaignRunner"),
    "SYSTEM_IDS": ("schedule", "SYSTEM_IDS"),
    "SealedVerifier": ("sealed_verifier", "SealedVerifier"),
    "SealedVerifierSink": ("sealed_verifier", "SealedVerifierSink"),
    "ValidationReport": ("package_validator", "ValidationReport"),
    "build_paired_schedule": ("schedule", "build_paired_schedule"),
    "compute_paired_contrasts": ("statistics", "compute_paired_contrasts"),
    "compute_retrieval_diagnostics": (
        "retrieval_metrics",
        "compute_retrieval_diagnostics",
    ),
    "compute_table2_metrics": ("metrics", "compute_table2_metrics"),
    "validate_campaign": ("package_validator", "validate_campaign"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
