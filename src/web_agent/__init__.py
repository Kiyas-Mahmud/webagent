"""Failure-Aware Resilient Autonomous Web Agent.

Backbone-agnostic 4-pillar architecture: one shared fused embedding feeds four
task heads (failure, action, memory) trained with one combined weighted loss.
Only the front-end encoder swaps across the 19 models; heads + loss are fixed.

See docs/AGENT.md for project rules and docs/PROJECT_SPECIFICATION.md for specs.
"""

__version__ = "0.1.0"
