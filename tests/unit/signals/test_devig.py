"""devig.py: three margin-removal methods, all reducing to the same
zero-margin behaviour and all producing a fair distribution that sums to 1."""

from __future__ import annotations

import pytest

from deportivas.signals.devig import (
    devig,
    devig_multiplicative,
    devig_power,
    devig_shin,
    implied_probabilities,
)

# Cuotas 1x2 con un margen realista de ~5%: 1/2.0 + 1/3.4 + 1/4.2 ~= 1.048
_THREE_WAY_PRICES = [2.0, 3.4, 4.2]
_TWO_WAY_PRICES = [1.87, 1.95]  # over/under con margen tipico


def test_implied_probabilities_is_the_reciprocal_of_price() -> None:
    assert implied_probabilities([2.0, 4.0]) == pytest.approx([0.5, 0.25])


@pytest.mark.parametrize("devig_fn", [devig_multiplicative, devig_power, devig_shin])
def test_fair_probabilities_sum_to_one(devig_fn: object) -> None:
    fair = devig_fn(_THREE_WAY_PRICES)  # type: ignore[operator]
    assert sum(fair) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("devig_fn", [devig_multiplicative, devig_power, devig_shin])
def test_fair_probabilities_preserve_favourite_ordering(devig_fn: object) -> None:
    fair = devig_fn(_THREE_WAY_PRICES)  # type: ignore[operator]
    assert fair[0] > fair[1] > fair[2]  # 2.0 es el favorito, 4.2 el menos probable


@pytest.mark.parametrize("devig_fn", [devig_multiplicative, devig_power, devig_shin])
def test_zero_margin_prices_are_returned_unchanged(devig_fn: object) -> None:
    """Cuotas sin margen (1/2 + 1/2 = 1): no hay nada que quitar."""
    fair = devig_fn([2.0, 2.0])  # type: ignore[operator]
    assert fair == pytest.approx([0.5, 0.5])


def test_multiplicative_just_normalises() -> None:
    raw = implied_probabilities(_THREE_WAY_PRICES)
    total = sum(raw)
    expected = [p / total for p in raw]
    assert devig_multiplicative(_THREE_WAY_PRICES) == pytest.approx(expected)


def test_power_shrinks_the_favourite_less_than_the_longshot_in_relative_terms() -> None:
    """El sesgo favorito-longshot: el metodo power corrige devolviendole al
    favorito una probabilidad relativamente mas alta que la normalizacion
    simple, y al menos probable una mas baja."""
    naive = devig_multiplicative(_THREE_WAY_PRICES)
    power = devig_power(_THREE_WAY_PRICES)
    assert power[0] > naive[0]  # favorito: power lo sube
    assert power[-1] < naive[-1]  # menos probable: power lo baja


def test_shin_shrinks_the_favourite_less_than_the_longshot_in_relative_terms() -> None:
    naive = devig_multiplicative(_THREE_WAY_PRICES)
    shin = devig_shin(_THREE_WAY_PRICES)
    assert shin[0] > naive[0]
    assert shin[-1] < naive[-1]


def test_shin_and_power_agree_qualitatively_but_not_exactly() -> None:
    power = devig_power(_THREE_WAY_PRICES)
    shin = devig_shin(_THREE_WAY_PRICES)
    assert power != pytest.approx(shin, abs=1e-9)


@pytest.mark.parametrize("devig_fn", [devig_multiplicative, devig_power, devig_shin])
def test_two_way_market_also_works(devig_fn: object) -> None:
    fair = devig_fn(_TWO_WAY_PRICES)  # type: ignore[operator]
    assert sum(fair) == pytest.approx(1.0, abs=1e-6)
    assert fair[0] > fair[1]  # cuota mas corta (1.87 < 1.95): mas probable


def test_devig_dispatches_by_method_name() -> None:
    assert devig(_THREE_WAY_PRICES, method="multiplicative") == pytest.approx(
        devig_multiplicative(_THREE_WAY_PRICES)
    )
    assert devig(_THREE_WAY_PRICES, method="power") == pytest.approx(devig_power(_THREE_WAY_PRICES))
    assert devig(_THREE_WAY_PRICES, method="shin") == pytest.approx(devig_shin(_THREE_WAY_PRICES))


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="metodo de devig desconocido"):
        devig(_THREE_WAY_PRICES, method="bogus")
