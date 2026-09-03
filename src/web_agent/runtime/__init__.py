"""Table 2 end-to-end runtime.

The runtime is deliberately separate from :mod:`web_agent.train`: it consumes
frozen policies and immutable manifests but never mutates training state.
"""

from web_agent.runtime.protocol import (
    REGISTERED_BUDGETS,
    REGISTERED_RECOVERY_TRIGGER_CONFIG,
    RecoveryTriggerConfig,
    RuntimeBudgets,
    RuntimeProtocol,
    StageRNGFactory,
    SystemSwitches,
    validate_frozen_protocol_mapping,
)

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
