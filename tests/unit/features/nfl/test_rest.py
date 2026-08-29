"""rest.py: days since each side's previous game, walk-forward."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from deportivas.features.nfl.rest import compute_rest_days


def _fixture(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "fix1",
        "home_team_id": "buf",
        "away_team_id": "kc",
        "kickoff_utc": datetime(2026, 1, 10, tzinfo=UTC),
    }
    base.update(overrides)
    return base


def test_first_game_has_no_rest_days() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_rest_days(fixtures)
    vector = result.iloc[0]["vector"]
    assert vector["rest_days_home"] is None
    assert vector["rest_days_away"] is None


def test_rest_days_computed_from_previous_game() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="buf",
                away_team_id="mia",
                kickoff_utc=datetime(2026, 1, 8, tzinfo=UTC),
            ),
        ]
    )
    result = compute_rest_days(fixtures)
    assert result.iloc[1]["vector"]["rest_days_home"] == 7


def test_short_week_shows_up_as_few_rest_days() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="buf",
                away_team_id="mia",
                kickoff_utc=datetime(2026, 1, 5, tzinfo=UTC),
            ),
        ]
    )
    result = compute_rest_days(fixtures)
    assert result.iloc[1]["vector"]["rest_days_home"] == 4


def test_bye_week_shows_up_as_many_rest_days() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="buf",
                away_team_id="mia",
                kickoff_utc=datetime(2026, 1, 15, tzinfo=UTC),
            ),
        ]
    )
    result = compute_rest_days(fixtures)
    assert result.iloc[1]["vector"]["rest_days_home"] == 14


def test_as_of_strictly_before_kickoff_over_a_sequence() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id=f"f{i}", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=7 * i)
            )
            for i in range(6)
        ]
    )
    result = compute_rest_days(fixtures)
    merged = result.merge(fixtures, left_on="fixture_id", right_on="id")
    assert (merged["as_of_timestamp"] < merged["kickoff_utc"]).all()


def test_as_of_uses_the_later_of_both_teams_prior_games() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id="f1",
                home_team_id="buf",
                away_team_id="mia",
                kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            _fixture(
                id="f2",
                home_team_id="kc",
                away_team_id="mia",
                kickoff_utc=datetime(2026, 1, 5, tzinfo=UTC),
            ),
            _fixture(
                id="f3",
                home_team_id="buf",
                away_team_id="kc",
                kickoff_utc=datetime(2026, 1, 10, tzinfo=UTC),
            ),
        ]
    )
    result = compute_rest_days(fixtures)
    # buf jugo el 1, kc jugo el 5: el mas reciente es el 5
    assert result.iloc[2]["as_of_timestamp"] == datetime(2026, 1, 5, tzinfo=UTC)
