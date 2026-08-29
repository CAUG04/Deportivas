"""writer.py: persists feature vectors, refusing anything the leakage guard
would flag."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FEATURES
from deportivas.domain.leakage import LeakageError
from deportivas.features.writer import write_features
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository


@pytest.fixture(autouse=True)
def _settings_in_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DEPORTIVAS_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fixtures_df(**overrides: object) -> pd.DataFrame:
    base: dict[str, object] = {
        "fixture_id": "fix1",
        "kickoff_utc": datetime(2026, 1, 10, 17, 30, tzinfo=UTC),
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_write_features_round_trips_vector_as_json() -> None:
    rows = pd.DataFrame(
        [
            {
                "fixture_id": "fix1",
                "competition_id": "eng-premier-league",
                "season": "2526",
                "as_of_timestamp": datetime(2026, 1, 5, tzinfo=UTC),
                "vector": {"elo_home": 1550.0, "elo_away": 1490.0},
            }
        ]
    )
    written = write_features("football_v1", rows, _fixtures_df())
    assert written == 1

    result = ParquetTableRepository(FEATURES, get_settings().parquet_dir).read()
    assert len(result) == 1
    vector = json.loads(result.iloc[0]["vector"])
    assert vector == {"elo_home": 1550.0, "elo_away": 1490.0}
    assert result.iloc[0]["feature_set"] == "football_v1"
    assert result.iloc[0]["source"] == "football_v1"


def test_write_features_empty_dataframe_is_a_noop() -> None:
    written = write_features(
        "football_v1",
        pd.DataFrame(columns=["fixture_id", "as_of_timestamp", "vector"]),
        _fixtures_df(),
    )
    assert written == 0


def test_write_features_rejects_leakage() -> None:
    rows = pd.DataFrame(
        [
            {
                "fixture_id": "fix1",
                "competition_id": "eng-premier-league",
                "season": "2526",
                "as_of_timestamp": datetime(2026, 1, 10, 17, 30, tzinfo=UTC),  # == kickoff: leakage
                "vector": {"elo_home": 1550.0},
            }
        ]
    )
    with pytest.raises(LeakageError):
        write_features("football_v1", rows, _fixtures_df())

    assert ParquetTableRepository(FEATURES, get_settings().parquet_dir).read().empty


def test_write_features_different_feature_sets_coexist() -> None:
    rows_v1 = pd.DataFrame(
        [
            {
                "fixture_id": "fix1",
                "competition_id": "eng-premier-league",
                "season": "2526",
                "as_of_timestamp": datetime(2026, 1, 5, tzinfo=UTC),
                "vector": {"elo_home": 1550.0},
            }
        ]
    )
    rows_v2 = pd.DataFrame(
        [
            {
                "fixture_id": "fix1",
                "competition_id": "eng-premier-league",
                "season": "2526",
                "as_of_timestamp": datetime(2026, 1, 5, tzinfo=UTC),
                "vector": {"elo_home": 1600.0},
            }
        ]
    )
    write_features("football_v1", rows_v1, _fixtures_df())
    write_features("football_v2", rows_v2, _fixtures_df())

    result = ParquetTableRepository(FEATURES, get_settings().parquet_dir).read()
    assert len(result) == 2
    assert set(result["feature_set"]) == {"football_v1", "football_v2"}


def test_write_features_rerun_is_idempotent() -> None:
    rows = pd.DataFrame(
        [
            {
                "fixture_id": "fix1",
                "competition_id": "eng-premier-league",
                "season": "2526",
                "as_of_timestamp": datetime(2026, 1, 5, tzinfo=UTC),
                "vector": {"elo_home": 1550.0},
            }
        ]
    )
    write_features("football_v1", rows, _fixtures_df())
    write_features("football_v1", rows, _fixtures_df())

    result = ParquetTableRepository(FEATURES, get_settings().parquet_dir).read()
    assert len(result) == 1
