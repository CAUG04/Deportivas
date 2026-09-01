"""pipeline.py: merges every NFL feature module into one vector per fixture,
then hands it to write_features."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FEATURES, FIXTURES, NFL_TEAM_GAME_STATS
from deportivas.features.nfl.pipeline import compute_and_write_nfl_features, compute_nfl_features
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
        "competition_id": "usa-nfl",
        "season": "2025",
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


def test_merged_vector_contains_keys_from_every_module() -> None:
    fixtures = pd.DataFrame([_fixture()])
    stats = pd.DataFrame(columns=["fixture_id", "team_id", "offensive_plays"])

    result = compute_nfl_features(fixtures, stats)

    vector = result.iloc[0]["vector"]
    assert "off_epa_rolling_4_home" in vector  # epa_rolling.py
    assert "rest_days_home" in vector  # rest.py
    assert "dvoa_off_home" in vector  # dvoa_approx.py


def test_output_carries_competition_id_and_season() -> None:
    fixtures = pd.DataFrame([_fixture()])
    stats = pd.DataFrame(columns=["fixture_id", "team_id", "offensive_plays"])

    result = compute_nfl_features(fixtures, stats)

    assert result.iloc[0]["competition_id"] == "usa-nfl"
    assert result.iloc[0]["season"] == "2025"


def test_one_row_per_fixture() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id=f"f{i}", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=7 * i)
            )
            for i in range(6)
        ]
    )
    stats = pd.DataFrame(columns=["fixture_id", "team_id", "offensive_plays"])

    result = compute_nfl_features(fixtures, stats)

    assert len(result) == 6
    assert set(result["fixture_id"]) == {f"f{i}" for i in range(6)}


def test_all_as_of_strictly_before_kickoff_across_a_sequence() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(
                id=f"f{i}", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=7 * i)
            )
            for i in range(6)
        ]
    )
    stats = pd.DataFrame(columns=["fixture_id", "team_id", "offensive_plays"])

    result = compute_nfl_features(fixtures, stats)

    merged = result.merge(fixtures, left_on="fixture_id", right_on="id")
    assert (merged["as_of_timestamp"] < merged["kickoff_utc"]).all()


def test_epa_rolling_uses_the_stats_passed_in() -> None:
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

    result = compute_nfl_features(fixtures, stats)

    second = result[result["fixture_id"] == "f2"].iloc[0]
    assert second["vector"]["off_epa_rolling_4_home"] == pytest.approx(0.3)


def test_compute_and_write_nfl_features_persists_the_merged_vector() -> None:
    fixtures_repo = ParquetTableRepository(FIXTURES, get_settings().parquet_dir)
    fixtures_repo.write(
        pd.DataFrame(
            [
                {
                    **_fixture(),
                    "status": "finished",
                    "stage": "REG",
                    "matchday": 10,
                    "home_score": 27,
                    "away_score": 24,
                    "home_score_ht": None,
                    "away_score_ht": None,
                    "kickoff_is_estimated": False,
                    "source": "test",
                    "ingested_at": datetime.now(UTC),
                }
            ]
        )
    )
    stats_repo = ParquetTableRepository(NFL_TEAM_GAME_STATS, get_settings().parquet_dir)
    stats_repo.write(
        pd.DataFrame(
            [
                {
                    "fixture_id": "fix1",
                    "team_id": "buf",
                    "source": "nfl",
                    "competition_id": "usa-nfl",
                    "season": "2025",
                    "is_home": True,
                    "offensive_plays": 60,
                    "offensive_epa_per_play": 0.1,
                    "offensive_success_rate": 0.45,
                    "defensive_plays": 58,
                    "defensive_epa_per_play_allowed": -0.05,
                    "defensive_success_rate_allowed": 0.4,
                    "ingested_at": datetime.now(UTC),
                }
            ]
        )
    )

    written = compute_and_write_nfl_features("usa-nfl")

    assert written == 1
    features = ParquetTableRepository(FEATURES, get_settings().parquet_dir).read()
    assert len(features) == 1
    assert features.iloc[0]["feature_set"] == "nfl_v1"
