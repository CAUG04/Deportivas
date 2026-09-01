"""epa_rolling.py: rolling offensive/defensive EPA-per-play and success rate,
walk-forward, mirroring football's xg_rolling.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from deportivas.features.nfl.epa_rolling import EpaRollingConfig, compute_epa_rolling


def _fixture(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "fix1",
        "home_team_id": "buf",
        "away_team_id": "kc",
        "kickoff_utc": datetime(2026, 1, 10, tzinfo=UTC),
    }
    base.update(overrides)
    return base


def _stats(fixture_id: str, team_id: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "fixture_id": fixture_id,
        "team_id": team_id,
        "offensive_plays": 60,
        "offensive_epa_per_play": 0.1,
        "offensive_success_rate": 0.45,
        "defensive_epa_per_play_allowed": -0.05,
        "defensive_success_rate_allowed": 0.4,
    }
    base.update(overrides)
    return base


def test_first_game_has_no_rolling_history() -> None:
    fixtures = pd.DataFrame([_fixture()])
    stats = pd.DataFrame(columns=["fixture_id", "team_id", "offensive_plays"])
    result = compute_epa_rolling(fixtures, stats)
    vector = result.iloc[0]["vector"]
    assert vector["off_epa_rolling_4_home"] is None
    assert vector["def_epa_allowed_rolling_4_home"] is None


def test_rolling_value_reflects_prior_game() -> None:
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
    stats = pd.DataFrame(
        [
            _stats("f1", "buf", offensive_epa_per_play=0.3),
            _stats("f1", "kc", offensive_epa_per_play=-0.1),
        ]
    )
    result = compute_epa_rolling(fixtures, stats)
    second = result.iloc[1]["vector"]
    assert second["off_epa_rolling_4_home"] == pytest.approx(0.3)


def test_team_without_offensive_plays_never_enters_history() -> None:
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
    stats = pd.DataFrame([_stats("f1", "buf", offensive_plays=0, offensive_epa_per_play=None)])
    result = compute_epa_rolling(fixtures, stats)
    second = result.iloc[1]["vector"]
    assert second["off_epa_rolling_4_home"] is None


def test_all_window_sizes_present() -> None:
    fixtures = pd.DataFrame([_fixture()])
    stats = pd.DataFrame(columns=["fixture_id", "team_id", "offensive_plays"])
    result = compute_epa_rolling(fixtures, stats, config=EpaRollingConfig(windows=(4, 8, 16)))
    vector = result.iloc[0]["vector"]
    for window in (4, 8, 16):
        for side in ("home", "away"):
            assert f"off_epa_rolling_{window}_{side}" in vector
            assert f"off_success_rolling_{window}_{side}" in vector
            assert f"def_epa_allowed_rolling_{window}_{side}" in vector
            assert f"def_success_allowed_rolling_{window}_{side}" in vector


def test_more_recent_game_weighted_more_with_decay() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id="f1",
                home_team_id="buf",
                away_team_id="a",
                kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            _fixture(
                id="f2",
                home_team_id="buf",
                away_team_id="b",
                kickoff_utc=datetime(2026, 1, 8, tzinfo=UTC),
            ),
            _fixture(
                id="f3",
                home_team_id="buf",
                away_team_id="c",
                kickoff_utc=datetime(2026, 1, 15, tzinfo=UTC),
            ),
        ]
    )
    stats = pd.DataFrame(
        [
            _stats("f1", "buf", offensive_epa_per_play=0.0),
            _stats("f1", "a", offensive_epa_per_play=0.0),
            _stats("f2", "buf", offensive_epa_per_play=1.0),
            _stats("f2", "b", offensive_epa_per_play=0.0),
        ]
    )
    result = compute_epa_rolling(fixtures, stats, config=EpaRollingConfig(windows=(4,), decay=0.5))
    third = result.iloc[2]["vector"]
    # media simple seria 0.5; con decay=0.5 el partido mas reciente (1.0) pesa mas
    assert third["off_epa_rolling_4_home"] > 0.5


def test_as_of_strictly_before_kickoff_across_a_sequence() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id=f"f{i}", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=7 * i)
            )
            for i in range(6)
        ]
    )
    stats = pd.DataFrame(columns=["fixture_id", "team_id", "offensive_plays"])
    result = compute_epa_rolling(fixtures, stats)
    merged = result.merge(fixtures, left_on="fixture_id", right_on="id")
    assert (merged["as_of_timestamp"] < merged["kickoff_utc"]).all()


def test_config_defaults_are_reasonable() -> None:
    config = EpaRollingConfig()
    assert config.windows == (4, 8, 16)
    assert 0.0 < config.decay <= 1.0
