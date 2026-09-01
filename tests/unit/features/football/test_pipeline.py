"""pipeline.py: merges every football feature module into one vector per
fixture, then hands it to write_features."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FEATURES, FIXTURES, TEAM_MATCH_STATS
from deportivas.features.football.pipeline import (
    compute_and_write_football_features,
    compute_football_features,
)
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository


@pytest.fixture(autouse=True)
def _settings_in_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DEPORTIVAS_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fixture(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "fix1",
        "competition_id": "eng-premier-league",
        "season": "2526",
        "home_team_id": "arsenal",
        "away_team_id": "chelsea",
        "kickoff_utc": datetime(2026, 1, 10, tzinfo=UTC),
        "status": "finished",
        "home_score": 2,
        "away_score": 1,
    }
    base.update(overrides)
    return base


def _stats(fixture_id: str, team_id: str, xg: float) -> dict[str, object]:
    return {"fixture_id": fixture_id, "team_id": team_id, "xg": xg}


def test_merged_vector_contains_keys_from_every_module() -> None:
    fixtures = pd.DataFrame([_fixture()])
    stats = pd.DataFrame(columns=["fixture_id", "team_id", "xg"])

    result = compute_football_features(fixtures, stats)

    vector = result.iloc[0]["vector"]
    assert "elo_home" in vector  # elo.py
    assert "rest_days_home" in vector  # rest_congestion.py
    assert "xg_for_rolling_5_home" in vector  # xg_rolling.py
    assert "strength_attack_home" in vector  # strength.py
    assert "defense_adjusted_5_home" in vector  # opponent_adjusted.py


def test_as_of_timestamp_matches_the_shared_walk_forward_bound() -> None:
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
    stats = pd.DataFrame(columns=["fixture_id", "team_id", "xg"])

    result = compute_football_features(fixtures, stats)

    second = result[result["fixture_id"] == "f2"].iloc[0]
    assert second["as_of_timestamp"] == datetime(2026, 1, 1, tzinfo=UTC)


def test_output_carries_competition_id_and_season() -> None:
    fixtures = pd.DataFrame([_fixture()])
    stats = pd.DataFrame(columns=["fixture_id", "team_id", "xg"])

    result = compute_football_features(fixtures, stats)

    assert result.iloc[0]["competition_id"] == "eng-premier-league"
    assert result.iloc[0]["season"] == "2526"


def test_one_row_per_fixture() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id=f"f{i}", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i))
            for i in range(8)
        ]
    )
    stats = pd.DataFrame(columns=["fixture_id", "team_id", "xg"])

    result = compute_football_features(fixtures, stats)

    assert len(result) == 8
    assert set(result["fixture_id"]) == {f"f{i}" for i in range(8)}


def test_all_as_of_strictly_before_kickoff_across_a_sequence() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id=f"f{i}", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i))
            for i in range(8)
        ]
    )
    stats = pd.DataFrame(columns=["fixture_id", "team_id", "xg"])

    result = compute_football_features(fixtures, stats)

    merged = result.merge(fixtures, left_on="fixture_id", right_on="id")
    assert (merged["as_of_timestamp"] < merged["kickoff_utc"]).all()


def test_xg_rolling_uses_the_stats_passed_in() -> None:
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
    stats = pd.DataFrame([_stats("f1", "arsenal", 2.0), _stats("f1", "chelsea", 0.5)])

    result = compute_football_features(fixtures, stats)

    second = result[result["fixture_id"] == "f2"].iloc[0]
    assert second["vector"]["xg_for_rolling_5_home"] == pytest.approx(2.0)


def test_compute_and_write_football_features_persists_the_merged_vector() -> None:
    fixtures_repo = ParquetTableRepository(FIXTURES, get_settings().parquet_dir)
    fixtures_repo.write(
        pd.DataFrame(
            [
                {
                    **_fixture(),
                    "stage": None,
                    "matchday": 1,
                    "home_score_ht": None,
                    "away_score_ht": None,
                    "kickoff_is_estimated": False,
                    "source": "test",
                    "ingested_at": datetime.now(UTC),
                }
            ]
        )
    )
    stats_repo = ParquetTableRepository(TEAM_MATCH_STATS, get_settings().parquet_dir)
    stats_repo.write(
        pd.DataFrame(
            [
                {
                    "fixture_id": "fix1",
                    "team_id": "arsenal",
                    "source": "fbref",
                    "competition_id": "eng-premier-league",
                    "season": "2526",
                    "is_home": True,
                    "goals": 2,
                    "shots": None,
                    "shots_on_target": None,
                    "xg": 1.5,
                    "corners": None,
                    "yellow_cards": None,
                    "red_cards": None,
                    "possession": None,
                    "ingested_at": datetime.now(UTC),
                }
            ]
        )
    )

    written = compute_and_write_football_features("eng-premier-league")

    assert written == 1
    features = ParquetTableRepository(FEATURES, get_settings().parquet_dir).read()
    assert len(features) == 1
    assert features.iloc[0]["feature_set"] == "football_v1"
