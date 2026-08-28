"""Explicit rate limiting for source adapters.

FBref blocks aggressively on burst traffic (rule of Fase 1: "respeta
pausas"). This is a plain minimum-delay-between-calls limiter — no token
bucket, no burst allowance — because that is exactly what a scraper being
polite to one host needs, and anything fancier would just be a knob nobody
tunes. The clock and sleep function are injectable so tests never actually
sleep.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class RateLimiter:
    """Blocks in ``wait()`` until at least ``min_delay_seconds`` have passed
    since the previous call to ``wait()`` on this instance."""

    def __init__(
        self,
        min_delay_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_delay_seconds < 0:
            raise ValueError("min_delay_seconds no puede ser negativo")
        self._min_delay = min_delay_seconds
        self._clock = clock
        self._sleep = sleep
        self._last_call_at: float | None = None

    def wait(self) -> float:
        """Sleeps just long enough to respect the minimum delay, then
        records this moment as the new "last call". Returns how long it slept."""
        now = self._clock()
        if self._last_call_at is None:
            slept = 0.0
        else:
            elapsed = now - self._last_call_at
            remaining = self._min_delay - elapsed
            slept = max(remaining, 0.0)
            if slept > 0:
                self._sleep(slept)
                now = self._clock()
        self._last_call_at = now
        return slept

    def reset(self) -> None:
        self._last_call_at = None
