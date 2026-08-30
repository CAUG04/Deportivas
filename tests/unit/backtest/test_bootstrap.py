"""bootstrap.py: percentile bootstrap confidence interval on the mean."""

from __future__ import annotations

import pytest

from deportivas.backtest.bootstrap import bootstrap_confidence_interval


def test_raises_on_fewer_than_two_values() -> None:
    with pytest.raises(ValueError, match="al menos 2 valores"):
        bootstrap_confidence_interval([1.0], iterations=100, confidence_level=0.95)


def test_raises_on_no_values() -> None:
    with pytest.raises(ValueError, match="al menos 2 valores"):
        bootstrap_confidence_interval([], iterations=100, confidence_level=0.95)


def test_constant_values_give_a_degenerate_interval() -> None:
    lower, upper = bootstrap_confidence_interval(
        [1.0, 1.0, 1.0, 1.0], iterations=200, confidence_level=0.95, seed=1
    )
    assert lower == pytest.approx(1.0)
    assert upper == pytest.approx(1.0)


def test_same_seed_is_deterministic() -> None:
    values = [0.1, -0.05, 0.2, 0.0, -0.1, 0.15]
    first = bootstrap_confidence_interval(values, iterations=500, confidence_level=0.95, seed=42)
    second = bootstrap_confidence_interval(values, iterations=500, confidence_level=0.95, seed=42)
    assert first == second


def test_interval_is_ordered_and_brackets_a_reasonable_range() -> None:
    values = [0.1, -0.05, 0.2, 0.0, -0.1, 0.15, 0.05, -0.02]
    lower, upper = bootstrap_confidence_interval(
        values, iterations=2000, confidence_level=0.95, seed=7
    )
    assert lower < upper
    mean = sum(values) / len(values)
    assert lower <= mean <= upper


def test_wider_confidence_level_gives_a_wider_interval() -> None:
    values = [0.1, -0.05, 0.2, 0.0, -0.1, 0.15, 0.05, -0.02, 0.3, -0.2]
    narrow = bootstrap_confidence_interval(values, iterations=2000, confidence_level=0.5, seed=3)
    wide = bootstrap_confidence_interval(values, iterations=2000, confidence_level=0.95, seed=3)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])
