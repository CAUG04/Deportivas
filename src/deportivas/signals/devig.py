"""Removing a bookmaker's margin (overround) from decimal odds to recover a
fair, no-vig probability for each selection in a mutually exclusive market
(1x2's three selections, over/under's two, ...).

Three methods, matching ``config/thresholds.yaml``'s ``devig.method``:

* ``multiplicative``: the simplest, oldest approach — normalise the raw
  implied probabilities so they sum to 1. Assumes the bookmaker's margin is
  spread evenly across every selection, which is rarely exactly true.
* ``power``: finds a single exponent ``k`` such that raising every raw
  implied probability to the ``k``-th power sums to 1. Shrinks large
  probabilities more than small ones in relative terms — a mild correction
  toward the well-documented favourite-longshot bias (bettors overvalue
  longshots, so a book's quoted longshot probability overstates the true
  one more than its quoted favourite probability does).
* ``shin``: Shin's (1992) insider-trading model — the most theoretically
  grounded of the three, and this project's default (``reference_bookmaker:
  pinnacle`` in the same config section, because Pinnacle's low, stable
  margin makes it the best available proxy for a "fair" market to devig in
  the first place). Solves for a single market-wide "informed money"
  parameter ``z``; the same qualitative favourite-longshot correction as
  ``power``, on firmer theoretical footing.

All three reduce to the same, exact result when the input prices already
have zero margin (``sum(raw) == 1``): there is nothing to remove.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy.optimize import brentq

MULTIPLICATIVE = "multiplicative"
POWER = "power"
SHIN = "shin"

# Cualquier libro real tiene un margen modesto (unos pocos puntos porcentuales
# a un maximo de ~20-30% en casas muy blandas); z de Shin se mantiene muy por
# debajo de esta cota en la practica. No es un limite arbitrario: es hasta
# donde (1 - z) en el denominador de fair_at() sigue siendo numericamente sano.
_SHIN_Z_UPPER_BOUND = 0.45
_POWER_K_UPPER_BOUND = 50.0


def implied_probabilities(prices: Sequence[float]) -> list[float]:
    """Raw, margin-included implied probability for each decimal price."""
    return [1.0 / price for price in prices]


def devig_multiplicative(prices: Sequence[float]) -> list[float]:
    raw = implied_probabilities(prices)
    total = sum(raw)
    return [p / total for p in raw]


def devig_power(prices: Sequence[float]) -> list[float]:
    raw = implied_probabilities(prices)
    total = sum(raw)
    if total <= 1.0:
        return raw  # sin margen (o cuotas ya inconsistentes): nada que quitar

    def residual(k: float) -> float:
        return float(sum(p**k for p in raw) - 1.0)

    k = brentq(residual, 1.0, _POWER_K_UPPER_BOUND)
    return [p**k for p in raw]


def devig_shin(prices: Sequence[float]) -> list[float]:
    raw = implied_probabilities(prices)
    total = sum(raw)
    if total <= 1.0:
        return raw

    def fair_at(z: float) -> list[float]:
        return [
            (math.sqrt(z**2 + 4.0 * (1.0 - z) * (p**2) / total) - z) / (2.0 * (1.0 - z))
            for p in raw
        ]

    def residual(z: float) -> float:
        return sum(fair_at(z)) - 1.0

    z = brentq(residual, 0.0, _SHIN_Z_UPPER_BOUND)
    return fair_at(z)


_METHODS = {
    MULTIPLICATIVE: devig_multiplicative,
    POWER: devig_power,
    SHIN: devig_shin,
}


def devig(prices: Sequence[float], *, method: str) -> list[float]:
    if method not in _METHODS:
        raise ValueError(
            f"metodo de devig desconocido: {method!r} (usar uno de {sorted(_METHODS)})"
        )
    return _METHODS[method](prices)
