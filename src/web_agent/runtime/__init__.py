"""Table 2 end-to-end runtime.

The runtime is deliberately separate from :mod:`web_agent.train`: it consumes
frozen policies and immutable manifests but never mutates training state.
Protocol exports are lazy so importing a data-only contract does not execute
the wider runtime graph outside a narrow source-attestation boundary.
"""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path
from typing import Any


PROCESS_BROKER_IMPORT_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).resolve().read_bytes()
).hexdigest()

__all__ = [
    "REGISTERED_BUDGETS",
    "REGISTERED_RECOVERY_TRIGGER_CONFIG",
    "RecoveryTriggerConfig",
    "RuntimeBudgets",
    "RuntimeProtocol",
    "StageRNGFactory",
    "SystemSwitches",
    "validate_frozen_protocol_mapping",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module("web_agent.runtime.protocol"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
