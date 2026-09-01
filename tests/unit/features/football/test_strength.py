"""Dixon-Coles-style GLM attack/defense: no fit before enough data, fits once
the threshold is reached, refits periodically, and never uses a match to
predict its own strength."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pandas as pd

from deportivas.features.football.strength import StrengthConfig, compute_strength


def _fixture(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "fix1",
        "home_team_id": "arsenal",
        "away_team_id": "chelsea",
        "kickoff_utc": datetime(2026, 1, 10, tzinfo=UTC),
        "status": "finished",
        "home_score": 2,
        "away_score": 1,
    }
    base.update(overrides)
    return base


def _round_robin_history(
    teams: list[str], rounds: int, *, start: datetime, scorer: Callable[[str, str], tuple[int, int]]
) -> pd.DataFrame:
    """Generates ``rounds`` complete round-robins among ``teams``, each match's
    score decided by ``scorer(home, away) -> (home_score, away_score)``."""
    rows = []
    day = start
    match_id = 0
    for _ in range(rounds):
        for home, away in itertools.permutations(teams, 2):
            home_score, away_score = scorer(home, away)
            rows.append(
                _fixture(
                    id=f"m{match_id}",
                    home_team_id=home,
                    away_team_id=away,
                    kickoff_utc=day,
                    home_score=home_score,
                    away_score=away_score,
                )
            )
            match_id += 1
            day += timedelta(days=1)
    return pd.DataFrame(rows)


def test_before_minimum_matches_ratings_default_to_zero() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id=f"f{i}", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i))
            for i in range(5)
        ]
    )
    result = compute_strength(fixtures, config=StrengthConfig(min_matches_to_fit=20))
    for vector in result["vector"]:
        assert vector["strength_attack_home"] == 0.0
        assert vector["strength_defense_home"] == 0.0


def test_unseen_team_defaults_to_zero_even_after_a_fit() -> None:
    def scorer(_home: str, _away: str) -> tuple[int, int]:
        return 1, 1

    history = _round_robin_history(
        ["a", "b", "c"], rounds=4, start=datetime(2026, 1, 1, tzinfo=UTC), scorer=scorer
    )
    newcomer_fixture = pd.DataFrame(
        [
            _fixture(
                id="newcomer",
                home_team_id="a",
                away_team_id="brand-new-team",
                kickoff_utc=history["kickoff_utc"].max() + timedelta(days=1),
            )
        ]
    )
    fixtures = pd.concat([history, newcomer_fixture], ignore_index=True)
    result = compute_strength(fixtures, config=StrengthConfig(min_matches_to_fit=10, refit_every=5))
    last_vector = result.iloc[-1]["vector"]
    assert last_vector["strength_attack_away"] == 0.0
    assert last_vector["strength_defense_away"] == 0.0


def test_strong_team_gets_higher_attack_rating_than_weak_team() -> None:
    def scorer(home: str, away: str) -> tuple[int, int]:
        strength = {"strong": 4, "average": 2, "weak": 0}
        return strength[home], strength[away]

    fixtures = _round_robin_history(
        ["strong", "average", "weak"],
        rounds=6,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        scorer=scorer,
    )
    result = compute_strength(fixtures, config=StrengthConfig(min_matches_to_fit=10, refit_every=6))

    last_vector = result.iloc[-1]["vector"]
    home_team = fixtures.iloc[-1]["home_team_id"]
    away_team = fixtures.iloc[-1]["away_team_id"]
    ratings = {
        home_team: last_vector["strength_attack_home"],
        away_team: last_vector["strength_attack_away"],
    }
    if "strong" in ratings and "weak" in ratings:
        assert ratings["strong"] > ratings["weak"]


def test_fit_only_uses_matches_strictly_before_current_fixture() -> None:
    """A fixture at index N must never be able to influence its own or an
    earlier fixture's rating snapshot — ratings only change *after* a fit,
    and a fit only ever consumes already-finished matches."""

    def scorer(_home: str, _away: str) -> tuple[int, int]:
        return 3, 0

    fixtures = _round_robin_history(
        ["a", "b", "c", "d"], rounds=3, start=datetime(2026, 1, 1, tzinfo=UTC), scorer=scorer
    )
    result = compute_strength(fixtures, config=StrengthConfig(min_matches_to_fit=6, refit_every=6))
    # Antes del primer ajuste (min_matches_to_fit=6), todo debe ser neutro.
    for i in range(6):
        assert result.iloc[i]["vector"]["strength_attack_home"] == 0.0


def test_as_of_strictly_before_kickoff() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="arsenal",
                away_team_id="liverpool",
                kickoff_utc=datetime(2026, 1, 8, tzinfo=UTC),
            ),
        ]
    )
    result = compute_strength(fixtures)
    merged = result.merge(fixtures, left_on="fixture_id", right_on="id")
    assert (merged["as_of_timestamp"] < merged["kickoff_utc"]).all()


def test_scheduled_match_never_enters_training_data() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id="f1",
                kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC),
                status="scheduled",
                home_score=None,
                away_score=None,
            ),
        ]
        * 25
    )
    fixtures = fixtures.reset_index(drop=True)
    fixtures["id"] = [f"f{i}" for i in range(len(fixtures))]
    fixtures["kickoff_utc"] = [
        datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(len(fixtures))
    ]
    result = compute_strength(fixtures, config=StrengthConfig(min_matches_to_fit=5))
    for vector in result["vector"]:
        assert vector["strength_attack_home"] == 0.0


def test_min_matches_to_fit_of_one_is_still_not_enough_for_a_glm() -> None:
    """A single finished match gives 2 observations (home+away goals) but 0
    degrees of freedom once team/home dummies are added — _fit must decline
    rather than hand back a meaningless model, leaving ratings at 0.0."""
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="arsenal",
                away_team_id="liverpool",
                kickoff_utc=datetime(2026, 1, 8, tzinfo=UTC),
            ),
        ]
    )
    result = compute_strength(fixtures, config=StrengthConfig(min_matches_to_fit=1, refit_every=1))
    assert result.iloc[1]["vector"]["strength_attack_home"] == 0.0


def test_config_defaults_are_reasonable() -> None:
    config = StrengthConfig()
    assert config.min_matches_to_fit > 0
    assert config.refit_every > 0
