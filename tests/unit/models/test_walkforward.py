"""walkforward.py: season-level walk-forward windows, ordered by earliest
kickoff rather than by season-label sort order."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from deportivas.models.walkforward import season_order, walk_forward_windows


def _fixtures(*rows: tuple[str, datetime]) -> pd.DataFrame:
    return pd.DataFrame([{"season": season, "kickoff_utc": kickoff} for season, kickoff in rows])


def test_season_order_sorts_by_earliest_kickoff() -> None:
    fixtures = _fixtures(
        ("2223", datetime(2022, 8, 1, tzinfo=UTC)),
        ("2324", datetime(2023, 8, 1, tzinfo=UTC)),
        ("2425", datetime(2024, 8, 1, tzinfo=UTC)),
    )
    assert season_order(fixtures) == ("2223", "2324", "2425")


def test_season_order_ignores_label_format_and_uses_actual_dates() -> None:
    """A season labelled oddly (out of the usual lexical order) still sorts
    correctly because ordering comes from kickoff dates, not the label."""
    fixtures = _fixtures(
        ("season-b", datetime(2022, 8, 1, tzinfo=UTC)),
        ("season-a", datetime(2023, 8, 1, tzinfo=UTC)),
    )
    assert season_order(fixtures) == ("season-b", "season-a")


def test_walk_forward_windows_train_on_every_prior_season() -> None:
    fixtures = _fixtures(
        ("2223", datetime(2022, 8, 1, tzinfo=UTC)),
        ("2324", datetime(2023, 8, 1, tzinfo=UTC)),
        ("2425", datetime(2024, 8, 1, tzinfo=UTC)),
    )
    windows = walk_forward_windows(fixtures)
    assert len(windows) == 2
    assert windows[0].train_seasons == ("2223",)
    assert windows[0].validate_season == "2324"
    assert windows[1].train_seasons == ("2223", "2324")
    assert windows[1].validate_season == "2425"


def test_single_season_produces_no_windows() -> None:
    fixtures = _fixtures(("2223", datetime(2022, 8, 1, tzinfo=UTC)))
    assert walk_forward_windows(fixtures) == []


def test_empty_fixtures_produces_no_windows() -> None:
    fixtures = pd.DataFrame(columns=["season", "kickoff_utc"])
    assert walk_forward_windows(fixtures) == []
