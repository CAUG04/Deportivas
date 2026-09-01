"""rest_and_margin.py: rest days, back-to-back flag, and rolling scoring
margin, shared by every schedule-only sport (NBA, NHL, MLB)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from deportivas.features.rest_and_margin import RestAndMarginConfig, compute_rest_and_scoring_margin


def _fixture(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "fix1",
        "home_team_id": "lakers",
        "away_team_id": "celtics",
        "kickoff_utc": datetime(2026, 1, 10, tzinfo=UTC),
        "status": "finished",
        "home_score": 110,
        "away_score": 100,
    }
    base.update(overrides)
    return base


def test_first_game_has_no_rest_days_and_is_not_back_to_back() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_rest_and_scoring_margin(fixtures)
    vector = result.iloc[0]["vector"]
    assert vector["rest_days_home"] is None
    assert vector["back_to_back_home"] is False


def test_rest_days_computed_from_previous_game() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="lakers",
                away_team_id="warriors",
                kickoff_utc=datetime(2026, 1, 4, tzinfo=UTC),
            ),
        ]
    )
    result = compute_rest_and_scoring_margin(fixtures)
    assert result.iloc[1]["vector"]["rest_days_home"] == 3


def test_back_to_back_when_rest_days_at_or_below_threshold() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="lakers",
                away_team_id="warriors",
                kickoff_utc=datetime(2026, 1, 2, tzinfo=UTC),
            ),
        ]
    )
    result = compute_rest_and_scoring_margin(fixtures)
    assert result.iloc[1]["vector"]["back_to_back_home"] is True


def test_not_back_to_back_when_rest_days_above_threshold() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="lakers",
                away_team_id="warriors",
                kickoff_utc=datetime(2026, 1, 5, tzinfo=UTC),
            ),
        ]
    )
    result = compute_rest_and_scoring_margin(fixtures)
    assert result.iloc[1]["vector"]["back_to_back_home"] is False


def test_back_to_back_threshold_is_configurable() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="lakers",
                away_team_id="warriors",
                kickoff_utc=datetime(2026, 1, 3, tzinfo=UTC),
            ),
        ]
    )
    result = compute_rest_and_scoring_margin(
        fixtures, config=RestAndMarginConfig(back_to_back_max_rest_days=2)
    )
    assert result.iloc[1]["vector"]["back_to_back_home"] is True


def test_scoring_margin_reflects_prior_result() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id="f1",
                home_score=120,
                away_score=100,
                kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            _fixture(
                id="f2",
                home_team_id="lakers",
                away_team_id="warriors",
                kickoff_utc=datetime(2026, 1, 4, tzinfo=UTC),
            ),
        ]
    )
    result = compute_rest_and_scoring_margin(fixtures, config=RestAndMarginConfig(windows=(5,)))
    assert result.iloc[1]["vector"]["scoring_margin_rolling_5_home"] == pytest.approx(20.0)


def test_scheduled_unplayed_game_does_not_update_margin_history() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", status="scheduled", home_score=None, away_score=None),
            _fixture(
                id="f2",
                home_team_id="lakers",
                away_team_id="warriors",
                kickoff_utc=datetime(2026, 1, 15, tzinfo=UTC),
            ),
        ]
    )
    result = compute_rest_and_scoring_margin(fixtures, config=RestAndMarginConfig(windows=(5,)))
    assert result.iloc[1]["vector"]["scoring_margin_rolling_5_home"] is None


def test_away_team_margin_reflects_its_own_prior_result() -> None:
    fixtures = pd.DataFrame(
        [
            # celtics jugo de visita en f1 y gano 100-90: su margen fue +10.
            _fixture(
                id="f1", home_score=90, away_score=100, kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)
            ),
            _fixture(
                id="f2",
                home_team_id="warriors",
                away_team_id="celtics",
                kickoff_utc=datetime(2026, 1, 4, tzinfo=UTC),
            ),
        ]
    )
    result = compute_rest_and_scoring_margin(fixtures, config=RestAndMarginConfig(windows=(5,)))
    assert result.iloc[1]["vector"]["scoring_margin_rolling_5_away"] == pytest.approx(10.0)


def test_all_window_sizes_present() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_rest_and_scoring_margin(
        fixtures, config=RestAndMarginConfig(windows=(5, 10, 20))
    )
    vector = result.iloc[0]["vector"]
    for window in (5, 10, 20):
        for side in ("home", "away"):
            assert f"scoring_margin_rolling_{window}_{side}" in vector


def test_as_of_strictly_before_kickoff_across_a_sequence() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id=f"f{i}", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i))
            for i in range(10)
        ]
    )
    result = compute_rest_and_scoring_margin(fixtures)
    merged = result.merge(fixtures, left_on="fixture_id", right_on="id")
    assert (merged["as_of_timestamp"] < merged["kickoff_utc"]).all()


def test_config_defaults_are_reasonable() -> None:
    config = RestAndMarginConfig()
    assert config.windows == (5, 10, 20)
    assert 0.0 < config.decay <= 1.0
    assert config.back_to_back_max_rest_days >= 0
