"""poisson.py: bivariate Poisson score matrix and score-matrix-derived
football markets (1x2, over_under, btts)."""

from __future__ import annotations

import math

import pytest

from deportivas.config.catalog import load_markets
from deportivas.features.football.dixon_coles_glm import DixonColesGlmFit
from deportivas.models.football.poisson import (
    expected_goals,
    market_probabilities,
    score_matrix,
)


def _model(**overrides: object) -> DixonColesGlmFit:
    base: dict[str, object] = {
        "attack": {"strong": 0.5, "weak": -0.5},
        "defense": {"strong": -0.3, "weak": 0.3},
        "home_advantage": 0.2,
        "intercept": 0.1,
    }
    base.update(overrides)
    return DixonColesGlmFit(**base)  # type: ignore[arg-type]


def test_expected_goals_home_advantage_increases_the_rate() -> None:
    model = _model()
    home_rate = expected_goals(model, attacking_team="strong", defending_team="weak", is_home=True)
    away_rate = expected_goals(model, attacking_team="strong", defending_team="weak", is_home=False)
    assert home_rate > away_rate


def test_expected_goals_unseen_team_defaults_to_average() -> None:
    model = _model()
    rate = expected_goals(model, attacking_team="unknown", defending_team="unknown", is_home=False)
    assert rate == pytest.approx(math.exp(model.intercept))


def test_score_matrix_sums_to_one() -> None:
    matrix = score_matrix(_model(), "strong", "weak")
    total = sum(sum(row) for row in matrix)
    assert total == pytest.approx(1.0)


def test_score_matrix_shape_matches_max_goals() -> None:
    matrix = score_matrix(_model(), "strong", "weak", max_goals=5)
    assert len(matrix) == 6
    assert all(len(row) == 6 for row in matrix)


def test_score_matrix_favours_the_stronger_home_side() -> None:
    matrix = score_matrix(_model(), "strong", "weak")
    n = len(matrix)
    home_win = sum(matrix[i][j] for i in range(n) for j in range(n) if i > j)
    away_win = sum(matrix[i][j] for i in range(n) for j in range(n) if i < j)
    assert home_win > away_win


def test_one_x_two_probabilities_sum_to_one() -> None:
    matrix = score_matrix(_model(), "strong", "weak")
    market = load_markets().get("1x2")
    probs = market_probabilities(matrix, market)
    assert {p.selection for p in probs} == {"home", "draw", "away"}
    assert sum(p.prob for p in probs) == pytest.approx(1.0)
    assert all(p.line is None for p in probs)


def test_btts_probabilities_sum_to_one() -> None:
    matrix = score_matrix(_model(), "strong", "weak")
    market = load_markets().get("btts")
    probs = market_probabilities(matrix, market)
    assert {p.selection for p in probs} == {"yes", "no"}
    assert sum(p.prob for p in probs) == pytest.approx(1.0)


def test_btts_yes_matches_direct_inclusion_exclusion() -> None:
    matrix = score_matrix(_model(), "strong", "weak")
    n = len(matrix)
    p_home_zero = sum(matrix[0][j] for j in range(n))
    p_away_zero = sum(matrix[i][0] for i in range(n))
    expected_yes = 1.0 - (p_home_zero + p_away_zero - matrix[0][0])

    market = load_markets().get("btts")
    probs = {p.selection: p.prob for p in market_probabilities(matrix, market)}
    assert probs["yes"] == pytest.approx(expected_yes)


def test_over_under_covers_every_default_line_with_a_matching_pair() -> None:
    matrix = score_matrix(_model(), "strong", "weak")
    market = load_markets().get("over_under")
    probs = market_probabilities(matrix, market)
    lines = {p.line for p in probs}
    assert lines == set(market.default_lines)
    for line in market.default_lines:
        over = next(p.prob for p in probs if p.selection == "over" and p.line == line)
        under = next(p.prob for p in probs if p.selection == "under" and p.line == line)
        assert over + under == pytest.approx(1.0)


def test_over_under_probability_decreases_as_line_increases() -> None:
    matrix = score_matrix(_model(), "strong", "weak")
    market = load_markets().get("over_under")
    probs = {
        p.line: p.prob
        for p in market_probabilities(matrix, market)
        if p.selection == "over" and p.line is not None
    }
    ordered_lines = sorted(probs)
    ordered_probs = [probs[line] for line in ordered_lines]
    assert ordered_probs == sorted(ordered_probs, reverse=True)


def test_unsupported_market_raises() -> None:
    matrix = score_matrix(_model(), "strong", "weak")
    market = load_markets().get("asian_handicap")
    with pytest.raises(ValueError, match="no soportado"):
        market_probabilities(matrix, market)
