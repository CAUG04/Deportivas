from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from deportivas.features.football.xg_rolling import XgRollingConfig, compute_xg_rolling


def _fixture(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "fix1",
        "home_team_id": "arsenal",
        "away_team_id": "chelsea",
        "kickoff_utc": datetime(2026, 1, 10, tzinfo=UTC),
    }
    base.update(overrides)
    return base


def _stats(fixture_id: str, team_id: str, xg: float) -> dict[str, object]:
    return {"fixture_id": fixture_id, "team_id": team_id, "xg": xg}


def test_first_match_has_no_rolling_history() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_xg_rolling(fixtures, pd.DataFrame(columns=["fixture_id", "team_id", "xg"]))
    vector = result.iloc[0]["vector"]
    assert vector["xg_for_rolling_5_home"] is None
    assert vector["xg_against_rolling_5_home"] is None


def test_rolling_average_uses_prior_matches_only() -> None:
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
    stats = pd.DataFrame(
        [
            _stats("f1", "arsenal", 2.0),
            _stats("f1", "chelsea", 1.0),
        ]
    )
    result = compute_xg_rolling(fixtures, stats, config=XgRollingConfig(windows=(5,), decay=1.0))
    vector = result.iloc[1]["vector"]
    assert vector["xg_for_rolling_5_home"] == pytest.approx(2.0)
    assert vector["xg_against_rolling_5_home"] == pytest.approx(1.0)


def test_decay_weights_recent_match_more() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="arsenal",
                away_team_id="everton",
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
    stats = pd.DataFrame(
        [
            _stats("f1", "arsenal", 1.0),
            _stats("f1", "chelsea", 0.0),
            _stats("f2", "arsenal", 3.0),
            _stats("f2", "everton", 0.0),
        ]
    )
    # decay < 1: el partido mas reciente (f2, xg=3.0) pesa mas que f1 (xg=1.0),
    # asi que el promedio ponderado debe quedar mas cerca de 3.0 que del simple 2.0.
    result = compute_xg_rolling(fixtures, stats, config=XgRollingConfig(windows=(5,), decay=0.5))
    vector = result.iloc[2]["vector"]
    assert vector["xg_for_rolling_5_home"] > 2.0


def test_window_size_limits_how_far_back_it_looks() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="arsenal",
                away_team_id="everton",
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
    stats = pd.DataFrame(
        [
            _stats("f1", "arsenal", 10.0),
            _stats("f1", "chelsea", 0.0),
            _stats("f2", "arsenal", 2.0),
            _stats("f2", "everton", 0.0),
        ]
    )
    result = compute_xg_rolling(fixtures, stats, config=XgRollingConfig(windows=(1,), decay=1.0))
    # ventana=1: solo debe ver el partido inmediatamente anterior (f2, xg=2.0), no f1 (xg=10.0)
    assert result.iloc[2]["vector"]["xg_for_rolling_1_home"] == pytest.approx(2.0)


def test_match_without_any_xg_data_is_skipped_from_history() -> None:
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
    # f1 no tiene xg de ninguna fuente
    result = compute_xg_rolling(fixtures, pd.DataFrame(columns=["fixture_id", "team_id", "xg"]))
    assert result.iloc[1]["vector"]["xg_for_rolling_5_home"] is None


def test_match_with_only_one_side_xg_is_skipped_from_history() -> None:
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
    stats = pd.DataFrame([_stats("f1", "arsenal", 2.0)])  # falta el lado de chelsea
    result = compute_xg_rolling(fixtures, stats)
    assert result.iloc[1]["vector"]["xg_for_rolling_5_home"] is None


def test_multiple_window_sizes_all_present_in_vector() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_xg_rolling(
        fixtures, pd.DataFrame(columns=["fixture_id", "team_id", "xg"]), config=XgRollingConfig()
    )
    vector = result.iloc[0]["vector"]
    for window in (5, 10, 20):
        for side in ("home", "away"):
            assert f"xg_for_rolling_{window}_{side}" in vector
            assert f"xg_against_rolling_{window}_{side}" in vector


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
    result = compute_xg_rolling(fixtures, pd.DataFrame(columns=["fixture_id", "team_id", "xg"]))
    merged = result.merge(fixtures, left_on="fixture_id", right_on="id")
    assert (merged["as_of_timestamp"] < merged["kickoff_utc"]).all()
