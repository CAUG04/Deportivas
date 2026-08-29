"""Elo walk-forward: ratings only move on finished matches, home advantage
shifts the expected result, and as_of always precedes kickoff."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from deportivas.features.football.elo import EloConfig, compute_elo, expected_home_win_prob


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


def test_expected_home_win_prob_equal_ratings_with_home_advantage_favours_home() -> None:
    prob = expected_home_win_prob(1500, 1500, home_advantage=60)
    assert prob > 0.5


def test_expected_home_win_prob_no_advantage_equal_ratings_is_half() -> None:
    assert expected_home_win_prob(1500, 1500, home_advantage=0) == pytest.approx(0.5)


def test_first_ever_match_uses_initial_ratings() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_elo(fixtures)
    vector = result.iloc[0]["vector"]
    assert vector["elo_home"] == 1500.0
    assert vector["elo_away"] == 1500.0


def test_first_ever_match_as_of_is_before_kickoff() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_elo(fixtures)
    assert result.iloc[0]["as_of_timestamp"] < fixtures.iloc[0]["kickoff_utc"]


def test_home_win_increases_home_rating_for_next_match() -> None:
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
    result = compute_elo(fixtures)
    second_vector = result.iloc[1]["vector"]
    assert second_vector["elo_home"] > 1500.0  # arsenal gano el primer partido


def test_draw_moves_both_ratings_toward_each_other_when_unequal() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC), home_score=1, away_score=1
            ),
        ]
    )
    # Favorito en casa (por ventaja de localia) empata: su rating deberia bajar un poco.
    result = compute_elo(fixtures, config=EloConfig(home_advantage=100))
    fixtures2 = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="arsenal",
                away_team_id="liverpool",
                kickoff_utc=datetime(2026, 1, 8, tzinfo=UTC),
                home_score=1,
                away_score=1,
            ),
        ]
    )
    result2 = compute_elo(fixtures2, config=EloConfig(home_advantage=100))
    assert result2.iloc[1]["vector"]["elo_home"] < result.iloc[0]["vector"]["elo_home"] + 100


def test_away_win_decreases_home_rating_for_next_match() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC), home_score=0, away_score=3
            ),
            _fixture(
                id="f2",
                home_team_id="arsenal",
                away_team_id="liverpool",
                kickoff_utc=datetime(2026, 1, 8, tzinfo=UTC),
            ),
        ]
    )
    result = compute_elo(fixtures)
    assert result.iloc[1]["vector"]["elo_home"] < 1500.0


def test_scheduled_unplayed_match_does_not_update_ratings() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id="f1",
                kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC),
                status="scheduled",
                home_score=None,
                away_score=None,
            ),
            _fixture(
                id="f2",
                home_team_id="arsenal",
                away_team_id="liverpool",
                kickoff_utc=datetime(2026, 1, 8, tzinfo=UTC),
            ),
        ]
    )
    result = compute_elo(fixtures)
    assert result.iloc[1]["vector"]["elo_home"] == 1500.0


def test_as_of_is_kickoff_of_most_recent_prior_match_for_either_team() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id="f1",
                home_team_id="arsenal",
                away_team_id="chelsea",
                kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            _fixture(
                id="f2",
                home_team_id="liverpool",
                away_team_id="arsenal",
                kickoff_utc=datetime(2026, 1, 8, tzinfo=UTC),
            ),
        ]
    )
    result = compute_elo(fixtures)
    assert result.iloc[1]["as_of_timestamp"] == datetime(2026, 1, 1, tzinfo=UTC)


def test_as_of_uses_the_later_of_both_teams_prior_matches() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id="f1",
                home_team_id="arsenal",
                away_team_id="chelsea",
                kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            _fixture(
                id="f2",
                home_team_id="liverpool",
                away_team_id="chelsea",
                kickoff_utc=datetime(2026, 1, 5, tzinfo=UTC),
            ),
            _fixture(
                id="f3",
                home_team_id="arsenal",
                away_team_id="liverpool",
                kickoff_utc=datetime(2026, 1, 10, tzinfo=UTC),
            ),
        ]
    )
    result = compute_elo(fixtures)
    # arsenal jugo el 1, liverpool jugo el 5: el mas reciente es el 5
    assert result.iloc[2]["as_of_timestamp"] == datetime(2026, 1, 5, tzinfo=UTC)


def test_all_as_of_strictly_before_kickoff_across_a_longer_sequence() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id=f"f{i}", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i))
            for i in range(10)
        ]
    )
    result = compute_elo(fixtures)
    merged = result.merge(fixtures, left_on="fixture_id", right_on="id")
    assert (merged["as_of_timestamp"] < merged["kickoff_utc"]).all()


def test_home_win_probability_in_vector_between_zero_and_one() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_elo(fixtures)
    assert 0.0 < result.iloc[0]["vector"]["elo_home_win_prob"] < 1.0
