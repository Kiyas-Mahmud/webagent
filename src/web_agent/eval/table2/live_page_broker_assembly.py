"""Same-process live-page broker fixture assembly.

This module creates typed runtime and evaluator capability objects over one
private in-memory state.  It is useful for deterministic engineering tests of
the message shapes, cleanup rules and opaque return contract.

It is deliberately **not production authority**.  Both accessors are importable
in the same Python interpreter, so reviewed runtime/provider code could obtain
the evaluator capability and introspect same-process state.  Source attestation
does not prevent that.  The current runner attestation records this limitation
and canonical live launch fails before provider import.  Production requires a
separately authenticated process-isolated transport and receipt under a future
registered schema.
"""

from __future__ import annotations

from .sealed_page_broker import (
    RuntimePagePublisher,
    SealedPageEvaluatorCapability,
    create_one_way_sealed_page_broker,
)


_RUNTIME_PAGE_PUBLISHER, _SEALED_PAGE_EVALUATOR_CAPABILITY = (
    create_one_way_sealed_page_broker()
)


def process_runtime_page_publisher() -> RuntimePagePublisher:
    """Return the runtime-side publisher for engineering fixtures only."""

    return _RUNTIME_PAGE_PUBLISHER


def process_sealed_page_evaluator_capability() -> SealedPageEvaluatorCapability:
    """Return the evaluator-side capability for engineering fixtures only."""

    return _SEALED_PAGE_EVALUATOR_CAPABILITY


def assert_process_wide_broker_assembly() -> None:
    """Check typed API separation without claiming process isolation."""

    if type(_RUNTIME_PAGE_PUBLISHER) is not RuntimePagePublisher:
        raise RuntimeError("process-wide live-page publisher has the wrong type")
    if type(_SEALED_PAGE_EVALUATOR_CAPABILITY) is not SealedPageEvaluatorCapability:
        raise RuntimeError("process-wide sealed page evaluator has the wrong type")
    if hasattr(_RUNTIME_PAGE_PUBLISHER, "evaluate_bound") or hasattr(
        _RUNTIME_PAGE_PUBLISHER, "evaluate_final"
    ):
        raise RuntimeError("runtime publisher unexpectedly exposes evaluation")
    if hasattr(_SEALED_PAGE_EVALUATOR_CAPABILITY, "publish") or hasattr(
        _SEALED_PAGE_EVALUATOR_CAPABILITY, "close"
    ):
        raise RuntimeError("sealed evaluator unexpectedly exposes page publication")


assert_process_wide_broker_assembly()
