"""Watchdog that detects machine suspend/resume mid-benchmark and aborts.

A machine suspend (e.g. laptop lid close / WSL2 host sleep) freezes the whole
benchmark process. On resume, external resources — notably the Ladybug buffer
pool and its native heap state — can be left degraded while the process
happily keeps executing: the 2026-09-18 overnight MMLongBench run degraded
from 0% to 100% tool-error rate after a ~2h suspend and silently burned 356
questions at ~19% accuracy.

Wall-clock time (``time.time()``) jumps forward across a suspend while
monotonic time (``time.monotonic()``) does not advance. A background thread
samples the divergence between the two clocks; a sudden growth of the gap
means the machine was suspended, and every subsequent question run is aborted
fail-fast instead of being recorded against a degraded database.

Usage::

    watchdog = SuspendWatchdog()
    watchdog.start()
    try:
        ...  # submit / await question tasks
    finally:
        watchdog.stop()
"""

from __future__ import annotations

import threading
import time

from loguru import logger

_DEFAULT_INTERVAL_SECONDS = 15.0
_DEFAULT_THRESHOLD_SECONDS = 120.0

_ACTIVE: SuspendWatchdog | None = None
_ACTIVE_LOCK = threading.Lock()


class SuspendWatchdog:
    """Detect suspend/resume via wall-clock vs monotonic clock divergence.

    Args:
        interval: Sampling period of the watchdog thread in seconds.
        threshold: Minimum growth of the ``(time.time() - time.monotonic())``
            gap between two samples that is classified as a suspend. Generous
            by default so NTP adjustments cannot trip it.
    """

    def __init__(
        self,
        interval: float = _DEFAULT_INTERVAL_SECONDS,
        threshold: float = _DEFAULT_THRESHOLD_SECONDS,
    ) -> None:
        self._interval = interval
        self._threshold = threshold
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def triggered(self) -> bool:
        """True once a suspend/resume has been detected."""
        return self._event.is_set()

    def start(self) -> None:
        """Start the watchdog thread and register it as the process-active one."""
        global _ACTIVE
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="suspend-watchdog", daemon=True)
        with _ACTIVE_LOCK:
            _ACTIVE = self
        self._thread.start()

    def stop(self) -> None:
        """Stop the watchdog thread and clear the process-active registration."""
        global _ACTIVE
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
            self._thread = None
        with _ACTIVE_LOCK:
            if _ACTIVE is self:
                _ACTIVE = None

    def raise_if_triggered(self) -> None:
        """Raise immediately when a suspend/resume was detected."""
        if self._event.is_set():
            raise RuntimeError(
                "Machine suspend/resume detected mid-run — aborting to avoid recording runs "
                "against a degraded database. Restart the machine fresh and re-launch; "
                "already-completed run records are kept (append-only, last-wins)."
            )

    def _loop(self) -> None:
        baseline = time.time() - time.monotonic()
        while not self._stop.wait(self._interval):
            gap = time.time() - time.monotonic()
            if gap - baseline > self._threshold:
                self._event.set()
                logger.critical(
                    "[SuspendWatchdog] Wall-clock/monotonic divergence grew by {:.0f}s (> {:.0f}s) — "
                    "machine suspend/resume detected. New question runs will fail fast.",
                    gap - baseline,
                    self._threshold,
                )
                return
            # Track the max gap so NTP step adjustments can never trip the
            # threshold in the reverse direction.
            baseline = max(baseline, gap)


def check_suspend() -> None:
    """Fail fast when the process-active watchdog detected a suspend/resume.

    Called before each question run; raises ``RuntimeError`` when aborting.
    """
    with _ACTIVE_LOCK:
        watchdog = _ACTIVE
    if watchdog is not None:
        watchdog.raise_if_triggered()
