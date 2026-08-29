"""Bivariate Poisson match model for football.

Turns a fitted GLM (attack/defense/home-advantage coefficients, from
``features/football/dixon_coles_glm.py``) into a full goal-scoring
probability matrix for one matchup, then reads every score-matrix-derived
football market off that *same* matrix — ``config/markets.yaml``'s own
``derived_from: score_matrix`` contract for ``1x2``, ``over_under`` and
``btts`` — rather than predicting each market separately from scratch.

Independent Poisson, not the full Dixon-Coles paper: no low-score
correlation adjustment (tau) for 0-0/1-0/0-1/1-1 — the same simplification
``dixon_coles_glm.py``'s own docstring already names.

``asian_handicap`` is also ``derived_from: score_matrix`` in the catalog but
out of scope here: push/half-win settlement on a handicap line belongs with
signal settlement (Fase 5+), not model training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import poisson as _poisson

from deportivas.config.catalog import MarketSpec
from deportivas.domain.enums import Market, Selection
from deportivas.features.football.dixon_coles_glm import DixonColesGlmFit

MAX_GOALS = 10  # truncacion de la matriz; P(>=11 goles de un lado) es despreciable


def expected_goals(
    model: DixonColesGlmFit, *, attacking_team: str, defending_team: str, is_home: bool
) -> float:
    """A team or opponent unseen by this fit reads as "average" (0.0), same
    default ``strength.py`` uses for its rating feature."""
    log_rate = (
        model.intercept
        + model.attack.get(attacking_team, 0.0)
        - model.defense.get(defending_team, 0.0)
    )
    if is_home:
        log_rate += model.home_advantage
    return math.exp(log_rate)


def score_matrix(
    model: DixonColesGlmFit, home_team: str, away_team: str, *, max_goals: int = MAX_GOALS
) -> list[list[float]]:
    """``matrix[i][j]`` = P(home scores i, away scores j), renormalised so
    the truncated grid sums to exactly 1.0 rather than leaking the tail
    probability nowhere."""
    lambda_home = expected_goals(
        model, attacking_team=home_team, defending_team=away_team, is_home=True
    )
    lambda_away = expected_goals(
        model, attacking_team=away_team, defending_team=home_team, is_home=False
    )
    home_pmf = [float(_poisson.pmf(i, lambda_home)) for i in range(max_goals + 1)]
    away_pmf = [float(_poisson.pmf(j, lambda_away)) for j in range(max_goals + 1)]
    raw = [[h * a for a in away_pmf] for h in home_pmf]
    total = sum(sum(row) for row in raw)
    return [[cell / total for cell in row] for row in raw]


@dataclass(frozen=True, slots=True)
class MarketProbability:
    selection: str
    line: float | None
    prob: float


def market_probabilities(matrix: list[list[float]], market: MarketSpec) -> list[MarketProbability]:
    if market.id == Market.ONE_X_TWO.value:
        return _one_x_two(matrix)
    if market.id == Market.BTTS.value:
        return _btts(matrix)
    if market.id == Market.OVER_UNDER.value:
        return _over_under(matrix, market.default_lines)
    raise ValueError(f"mercado no soportado por el modelo Poisson de futbol: {market.id!r}")


def _one_x_two(matrix: list[list[float]]) -> list[MarketProbability]:
    n = len(matrix)
    home = sum(matrix[i][j] for i in range(n) for j in range(n) if i > j)
    draw = sum(matrix[i][i] for i in range(n))
    away = sum(matrix[i][j] for i in range(n) for j in range(n) if i < j)
    return [
        MarketProbability(Selection.HOME.value, None, home),
        MarketProbability(Selection.DRAW.value, None, draw),
        MarketProbability(Selection.AWAY.value, None, away),
    ]


def _btts(matrix: list[list[float]]) -> list[MarketProbability]:
    n = len(matrix)
    p_home_zero = sum(matrix[0][j] for j in range(n))
    p_away_zero = sum(matrix[i][0] for i in range(n))
    p_no = p_home_zero + p_away_zero - matrix[0][0]
    return [
        MarketProbability(Selection.YES.value, None, 1.0 - p_no),
        MarketProbability(Selection.NO.value, None, p_no),
    ]


def _over_under(matrix: list[list[float]], lines: tuple[float, ...]) -> list[MarketProbability]:
    n = len(matrix)
    result = []
    for line in lines:
        over = sum(matrix[i][j] for i in range(n) for j in range(n) if i + j > line)
        result.append(MarketProbability(Selection.OVER.value, line, over))
        result.append(MarketProbability(Selection.UNDER.value, line, 1.0 - over))
    return result
