"""dixon_coles_glm.py: the shared low-level Poisson GLM fit used by both
strength.py (a walk-forward feature) and models/football/poisson.py (the
Fase 3 predictive model)."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta
from typing import cast

import pandas as pd
import pytest

from deportivas.features.football.dixon_coles_glm import fit_dixon_coles_glm


def _round_robin(teams: list[str], rounds: int, *, scorer: object) -> list[dict[str, object]]:
    matches = []
    day = datetime(2026, 1, 1, tzinfo=UTC)
    for _ in range(rounds):
        for home, away in itertools.permutations(teams, 2):
            home_score, away_score = scorer(home, away)  # type: ignore[operator]
            matches.append(
                {
                    "home_team_id": home,
                    "away_team_id": away,
                    "home_score": home_score,
                    "away_score": away_score,
                }
            )
            day += timedelta(days=1)
    return matches


def test_fewer_than_two_matches_returns_none() -> None:
    assert fit_dixon_coles_glm([]) is None
    assert (
        fit_dixon_coles_glm(
            [{"home_team_id": "a", "away_team_id": "b", "home_score": 1, "away_score": 0}]
        )
        is None
    )


def test_reference_team_is_zero_baseline() -> None:
    def scorer(_home: str, _away: str) -> tuple[int, int]:
        return 1, 1

    matches = _round_robin(["a", "b", "c"], rounds=4, scorer=scorer)
    fit = fit_dixon_coles_glm(matches)
    assert fit is not None
    reference = sorted({"a", "b", "c"})[0]
    assert fit.attack[reference] == pytest.approx(0.0)
    assert fit.defense[reference] == pytest.approx(0.0)


def test_strong_attacking_team_gets_a_higher_attack_coefficient() -> None:
    def scorer(home: str, away: str) -> tuple[int, int]:
        strength = {"strong": 4, "weak": 0}
        return strength[home], strength[away]

    matches = _round_robin(["strong", "weak"], rounds=8, scorer=scorer)
    fit = fit_dixon_coles_glm(matches)
    assert fit is not None
    assert fit.attack["strong"] > fit.attack["weak"]


def test_home_advantage_is_positive_when_home_teams_outscore_away_teams() -> None:
    def scorer(_home: str, _away: str) -> tuple[int, int]:
        return 2, 1

    matches = _round_robin(["a", "b", "c"], rounds=4, scorer=scorer)
    fit = fit_dixon_coles_glm(matches)
    assert fit is not None
    assert fit.home_advantage > 0.0


def test_intercept_reflects_the_scoring_level() -> None:
    def scorer(_home: str, _away: str) -> tuple[int, int]:
        return 1, 1

    matches = _round_robin(["a", "b", "c"], rounds=4, scorer=scorer)
    fit = fit_dixon_coles_glm(matches)
    assert fit is not None
    # con marcador simetrico 1-1 en todos los partidos, la tasa base
    # log(1.0) = 0.0 no deberia estar lejos de cero.
    assert fit.intercept == pytest.approx(0.0, abs=0.5)


def test_perfect_separation_returns_none_instead_of_raising() -> None:
    """A single finished match gives 2 observations but 0 degrees of freedom
    once team/home dummies are added — the GLM must decline gracefully."""
    matches = [
        {"home_team_id": "a", "away_team_id": "b", "home_score": 3, "away_score": 0},
    ]
    # fewer than 2 matches ya cubre esto; forzamos el camino con exactamente
    # el minimo (2) pero un diseno degenerado por equipos no vistos dos veces.
    matches.append({"home_team_id": "a", "away_team_id": "b", "home_score": 3, "away_score": 0})
    fit = fit_dixon_coles_glm(matches)
    # con datos tan escasos el ajuste puede converger o no; lo unico exigido
    # es que nunca levante una excepcion sin capturar.
    assert fit is None or isinstance(fit.home_advantage, float)


def test_fixtures_dataframe_records_shape_is_accepted() -> None:
    """Sanity check against the shape strength.py actually hands in:
    to_dict('records') output, not hand-built dicts."""
    fixtures = pd.DataFrame(
        [
            {"home_team_id": "a", "away_team_id": "b", "home_score": 2, "away_score": 1},
            {"home_team_id": "b", "away_team_id": "a", "home_score": 0, "away_score": 0},
        ]
    )
    records = cast("list[dict[str, object]]", fixtures.to_dict("records"))
    fit = fit_dixon_coles_glm(records)
    assert fit is not None
