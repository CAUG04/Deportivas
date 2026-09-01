from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from deportivas.features.football.rest_congestion import compute_rest_and_congestion


def _fixture(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "fix1",
        "home_team_id": "arsenal",
        "away_team_id": "chelsea",
        "kickoff_utc": datetime(2026, 1, 10, tzinfo=UTC),
    }
    base.update(overrides)
    return base


def test_first_match_has_no_rest_days() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_rest_and_congestion(fixtures)
    vector = result.iloc[0]["vector"]
    assert vector["rest_days_home"] is None
    assert vector["rest_days_away"] is None


def test_rest_days_computed_from_previous_match() -> None:
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
    result = compute_rest_and_congestion(fixtures)
    assert result.iloc[1]["vector"]["rest_days_home"] == 7


def test_congestion_counts_matches_in_trailing_window() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id="f1",
                home_team_id="arsenal",
                away_team_id="a",
                kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            _fixture(
                id="f2",
                home_team_id="arsenal",
                away_team_id="b",
                kickoff_utc=datetime(2026, 1, 3, tzinfo=UTC),
            ),
            _fixture(
                id="f3",
                home_team_id="arsenal",
                away_team_id="c",
                kickoff_utc=datetime(2026, 1, 5, tzinfo=UTC),
            ),
            _fixture(
                id="f4",
                home_team_id="arsenal",
                away_team_id="d",
                kickoff_utc=datetime(2026, 1, 12, tzinfo=UTC),
            ),
        ]
    )
    result = compute_rest_and_congestion(fixtures, congestion_window_days=14)
    # Antes del 4o partido, arsenal jugo el 1, 3 y 5 de enero: los tres caen en la ventana de 14 dias.
    assert result.iloc[3]["vector"]["matches_last_window_home"] == 3


def test_congestion_excludes_matches_outside_window() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id="f1",
                home_team_id="arsenal",
                away_team_id="a",
                kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            _fixture(
                id="f2",
                home_team_id="arsenal",
                away_team_id="b",
                kickoff_utc=datetime(2026, 2, 1, tzinfo=UTC),
            ),
        ]
    )
    result = compute_rest_and_congestion(fixtures, congestion_window_days=14)
    assert result.iloc[1]["vector"]["matches_last_window_home"] == 0


def test_as_of_strictly_before_kickoff_over_a_sequence() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id=f"f{i}", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i))
            for i in range(6)
        ]
    )
    result = compute_rest_and_congestion(fixtures)
    merged = result.merge(fixtures, left_on="fixture_id", right_on="id")
    assert (merged["as_of_timestamp"] < merged["kickoff_utc"]).all()


def test_current_fixture_never_counted_in_its_own_congestion() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_rest_and_congestion(fixtures)
    assert result.iloc[0]["vector"]["matches_last_window_home"] == 0
