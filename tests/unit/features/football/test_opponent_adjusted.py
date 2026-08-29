from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from deportivas.features.football.opponent_adjusted import compute_opponent_adjusted_defense


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


def _strength(fixture_id: str, attack_home: float, attack_away: float) -> dict[str, object]:
    return {
        "fixture_id": fixture_id,
        "vector": {"strength_attack_home": attack_home, "strength_attack_away": attack_away},
    }


def test_first_match_has_no_history() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_opponent_adjusted_defense(
        fixtures,
        pd.DataFrame(columns=["fixture_id", "team_id", "xg"]),
        pd.DataFrame(columns=["fixture_id", "vector"]),
    )
    assert result.iloc[0]["vector"]["defense_adjusted_5_home"] is None


def test_conceding_against_a_strong_attack_is_less_penalised() -> None:
    """Two teams concede the same 2.0 xG in their one prior match, but team A's
    opponent had a much higher attack rating — A's adjusted number should be
    better (lower) than B's."""
    fixtures = pd.DataFrame(
        [
            _fixture(
                id="f1",
                home_team_id="a",
                away_team_id="strong_opp",
                kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            _fixture(
                id="f2",
                home_team_id="b",
                away_team_id="weak_opp",
                kickoff_utc=datetime(2026, 1, 2, tzinfo=UTC),
            ),
            _fixture(
                id="f3",
                home_team_id="a",
                away_team_id="b",
                kickoff_utc=datetime(2026, 1, 10, tzinfo=UTC),
            ),
        ]
    )
    stats = pd.DataFrame(
        [
            _stats("f1", "a", 0.5),
            _stats("f1", "strong_opp", 2.0),  # a conceded 2.0
            _stats("f2", "b", 0.5),
            _stats("f2", "weak_opp", 2.0),  # b conceded 2.0 too
        ]
    )
    strength = pd.DataFrame(
        [
            _strength(
                "f1", attack_home=0.0, attack_away=1.5
            ),  # strong_opp (away) tiene attack alto
            _strength("f2", attack_home=0.0, attack_away=0.1),  # weak_opp (away) tiene attack bajo
        ]
    )
    result = compute_opponent_adjusted_defense(fixtures, stats, strength, windows=(5,))
    vector = result.iloc[2]["vector"]
    assert vector["defense_adjusted_5_home"] < vector["defense_adjusted_5_away"]


def test_match_without_both_sides_xg_does_not_enter_history() -> None:
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
    stats = pd.DataFrame([_stats("f1", "arsenal", 1.0)])  # falta el lado de chelsea
    result = compute_opponent_adjusted_defense(
        fixtures, stats, pd.DataFrame(columns=["fixture_id", "vector"])
    )
    assert result.iloc[1]["vector"]["defense_adjusted_5_home"] is None


def test_missing_strength_defaults_opponent_attack_to_zero() -> None:
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
    stats = pd.DataFrame([_stats("f1", "arsenal", 1.0), _stats("f1", "chelsea", 1.0)])
    result = compute_opponent_adjusted_defense(
        fixtures, stats, pd.DataFrame(columns=["fixture_id", "vector"])
    )
    # sin dato de fuerza para f1, se asume 0.0: adjusted = conceded(1.0) - 0.0
    assert result.iloc[1]["vector"]["defense_adjusted_5_home"] == pytest.approx(1.0)


def test_all_window_sizes_present() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_opponent_adjusted_defense(
        fixtures,
        pd.DataFrame(columns=["fixture_id", "team_id", "xg"]),
        pd.DataFrame(columns=["fixture_id", "vector"]),
        windows=(5, 10, 20),
    )
    vector = result.iloc[0]["vector"]
    for window in (5, 10, 20):
        for side in ("home", "away"):
            assert f"defense_adjusted_{window}_{side}" in vector
