"""Failure-Aware Resilient Autonomous Web Agent.

Backbone-agnostic 4-pillar architecture: one shared fused embedding feeds four
task heads (failure, action, memory) trained with one combined weighted loss.
Only the front-end encoder swaps across the 19 models; heads + loss are fixed.

See docs/AGENT.md for project rules and docs/PROJECT_SPECIFICATION.md for specs.
"""

import hashlib
from pathlib import Path


# Bind parent-executed package initialization to the process-broker receipt.
PROCESS_BROKER_IMPORT_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).resolve().read_bytes()
).hexdigest()

__version__ = "0.1.0"
