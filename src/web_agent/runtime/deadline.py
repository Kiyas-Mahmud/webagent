"""POSIX wall-clock interruption for Table 2 blocking runtime calls.

Post-call clock checks cannot stop a browser, model, or evaluator callback that
never returns. Evaluation runs use this process-local guard so the registered
deadline can interrupt a blocking main-thread call.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import math
import signal
from threading import current_thread, main_thread
from time import monotonic
from typing import TypeVar


_E = TypeVar("_E", bound=BaseException)


class HardDeadlineUnavailable(RuntimeError):
    """The host cannot provide the registered interruptible deadline."""


def hard_deadline_supported() -> bool:
    """Return whether this process can arm a real-time signal deadline."""

    return (
        current_thread() is main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "ITIMER_REAL")
        and hasattr(signal, "setitimer")
        and hasattr(signal, "getitimer")
    )


@contextmanager
def interrupt_after(
    seconds: float,
    *,
    exception_factory: Callable[[], _E],
) -> Iterator[None]:
    """Interrupt the current main-thread call after ``seconds`` wall time.

    An existing earlier real-time timer remains the effective deadline and is
    restored on exit. Runtime fails closed when the host/thread cannot provide
    this interrupt mechanism.
    """

    duration = float(seconds)
    if not math.isfinite(duration) or duration <= 0.0:
        raise exception_factory()
    if not hard_deadline_supported():
        raise HardDeadlineUnavailable(
            "interruptible wall-clock deadlines require POSIX SIGALRM on the main thread"
        )

    prior_handler = signal.getsignal(signal.SIGALRM)
    prior_delay, prior_interval = signal.getitimer(signal.ITIMER_REAL)
    armed_delay = min(duration, prior_delay) if prior_delay > 0.0 else duration
    started = monotonic()

    def expire(_signum: int, _frame: object) -> None:
        raise exception_factory()

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, armed_delay)
    try:
        yield
    finally:
        elapsed = max(0.0, monotonic() - started)
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prior_handler)
        if prior_delay > 0.0:
            remaining = max(1e-9, prior_delay - elapsed)
            signal.setitimer(signal.ITIMER_REAL, remaining, prior_interval)
