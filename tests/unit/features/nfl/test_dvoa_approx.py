"""dvoa_approx.py: a pure, opponent-adjusted transform of epa_rolling.py's
own output — no new walk-forward pass, no new as_of tracking."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from deportivas.features.nfl.dvoa_approx import compute_dvoa_approx


def _epa_row(**vector_overrides: object) -> dict[str, object]:
    vector: dict[str, object] = {
        "off_epa_rolling_8_home": 0.2,
        "off_epa_rolling_8_away": -0.1,
        "def_epa_allowed_rolling_8_home": -0.05,
        "def_epa_allowed_rolling_8_away": 0.1,
    }
    vector.update(vector_overrides)
    return {
        "fixture_id": "fix1",
        "as_of_timestamp": datetime(2026, 1, 5, tzinfo=UTC),
        "vector": vector,
    }


def test_dvoa_off_is_offense_minus_opponent_defense() -> None:
    epa_rolling = pd.DataFrame([_epa_row()])
    result = compute_dvoa_approx(epa_rolling)
    vector = result.iloc[0]["vector"]
    # home off (0.2) - away def allowed (0.1) = 0.1
    assert vector["dvoa_off_home"] == pytest.approx(0.1)
    # away off (-0.1) - home def allowed (-0.05) = -0.05
    assert vector["dvoa_off_away"] == pytest.approx(-0.05)


def test_dvoa_def_is_opponent_offense_minus_own_defense() -> None:
    epa_rolling = pd.DataFrame([_epa_row()])
    result = compute_dvoa_approx(epa_rolling)
    vector = result.iloc[0]["vector"]
    # away off (-0.1) - home def allowed (-0.05) = -0.05
    assert vector["dvoa_def_home"] == pytest.approx(-0.05)
    # home off (0.2) - away def allowed (0.1) = 0.1
    assert vector["dvoa_def_away"] == pytest.approx(0.1)


def test_missing_input_value_propagates_as_none() -> None:
    epa_rolling = pd.DataFrame([_epa_row(off_epa_rolling_8_home=None)])
    result = compute_dvoa_approx(epa_rolling)
    vector = result.iloc[0]["vector"]
    assert vector["dvoa_off_home"] is None
    assert vector["dvoa_def_away"] is None


def test_as_of_timestamp_passes_through_unchanged() -> None:
    epa_rolling = pd.DataFrame([_epa_row()])
    result = compute_dvoa_approx(epa_rolling)
    assert result.iloc[0]["as_of_timestamp"] == datetime(2026, 1, 5, tzinfo=UTC)


def test_custom_window_reads_the_matching_keys() -> None:
    epa_rolling = pd.DataFrame(
        [
            {
                "fixture_id": "fix1",
                "as_of_timestamp": datetime(2026, 1, 5, tzinfo=UTC),
                "vector": {
                    "off_epa_rolling_4_home": 0.5,
                    "off_epa_rolling_4_away": 0.0,
                    "def_epa_allowed_rolling_4_home": 0.0,
                    "def_epa_allowed_rolling_4_away": 0.0,
                },
            }
        ]
    )
    result = compute_dvoa_approx(epa_rolling, window=4)
    assert result.iloc[0]["vector"]["dvoa_off_home"] == pytest.approx(0.5)


def test_fixture_id_preserved() -> None:
    epa_rolling = pd.DataFrame([_epa_row()])
    result = compute_dvoa_approx(epa_rolling)
    assert result.iloc[0]["fixture_id"] == "fix1"
