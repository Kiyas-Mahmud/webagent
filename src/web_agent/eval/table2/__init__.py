"""Leakage-safe end-to-end evaluation for the registered Table 2 campaign.

The package deliberately accepts plain mappings at its public boundaries.  The
runtime package is developed independently, and its dataclasses are converted
with :func:`web_agent.eval.table2.common.as_mapping` when they are available.
Evaluation code is the only code allowed to read the sealed verifier stream.
"""

from .package_validator import ValidationReport, validate_campaign
from .campaign import CampaignRunner
from .metrics import compute_table2_metrics
from .retrieval_metrics import compute_retrieval_diagnostics
from .schedule import SYSTEM_IDS, build_paired_schedule
from .sealed_verifier import OpaqueTerminalSignal, SealedVerifier, SealedVerifierSink
from .statistics import compute_paired_contrasts

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
