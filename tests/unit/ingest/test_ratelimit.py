"""RateLimiter tests use a fake clock/sleep — nothing here ever really sleeps."""

from __future__ import annotations

import pytest

from deportivas.ingest.ratelimit import RateLimiter


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_first_call_never_waits() -> None:
    clock = _FakeClock()
    slept: list[float] = []
    limiter = RateLimiter(4.0, clock=clock, sleep=slept.append)

    waited = limiter.wait()

    assert waited == 0.0
    assert slept == []


def test_second_call_waits_remaining_delay() -> None:
    clock = _FakeClock()

    def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    limiter = RateLimiter(4.0, clock=clock, sleep=fake_sleep)
    limiter.wait()
    clock.advance(1.0)  # solo paso 1s de los 4s requeridos

    waited = limiter.wait()

    assert waited == pytest.approx(3.0)


def test_call_after_full_delay_does_not_wait() -> None:
    clock = _FakeClock()
    slept: list[float] = []
    limiter = RateLimiter(4.0, clock=clock, sleep=slept.append)
    limiter.wait()
    clock.advance(5.0)  # ya paso mas que el delay minimo

    waited = limiter.wait()

    assert waited == 0.0
    assert slept == []


def test_zero_delay_never_waits() -> None:
    clock = _FakeClock()
    slept: list[float] = []
    limiter = RateLimiter(0.0, clock=clock, sleep=slept.append)
    limiter.wait()
    waited = limiter.wait()
    assert waited == 0.0
    assert slept == []


def test_negative_delay_rejected() -> None:
    with pytest.raises(ValueError, match="negativo"):
        RateLimiter(-1.0)


def test_reset_clears_history() -> None:
    clock = _FakeClock()
    slept: list[float] = []
    limiter = RateLimiter(4.0, clock=clock, sleep=slept.append)
    limiter.wait()
    limiter.reset()

    waited = limiter.wait()

    assert waited == 0.0
    assert slept == []


def test_default_clock_and_sleep_are_real_time() -> None:
    """Not exercised for timing (that would make tests slow/flaky) — just
    proves the public defaults are the real time module, not a stub left
    behind by mistake."""
    import time

    limiter = RateLimiter(0.0)
    assert limiter._clock is time.monotonic
    assert limiter._sleep is time.sleep
