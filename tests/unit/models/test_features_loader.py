"""features_loader.py: joins a sport's computed feature vectors with each
fixture's actual result — the shared starting point for feature-based
models (moneyline classifiers)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FEATURES, FIXTURES
from deportivas.models.features_loader import load_feature_vectors
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository


@pytest.fixture(autouse=True)
def _settings_in_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DEPORTIVAS_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fixture_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "fix1",
        "competition_id": "usa-nfl",
        "season": "2025",
        "kickoff_utc": datetime(2026, 1, 10, tzinfo=UTC),
        "home_team_id": "buf",
        "away_team_id": "kc",
        "status": "finished",
        "stage": None,
        "matchday": None,
        "home_score": 27,
        "away_score": 24,
        "home_score_ht": None,
        "away_score_ht": None,
        "kickoff_is_estimated": False,
        "source": "test",
        "ingested_at": datetime.now(UTC),
    }
    base.update(overrides)
    return base


def _feature_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "fixture_id": "fix1",
        "feature_set": "nfl_v1",
        "competition_id": "usa-nfl",
        "season": "2025",
        "as_of_timestamp": datetime(2026, 1, 5, tzinfo=UTC),
        "vector": '{"off_epa_rolling_4_home": 0.1}',
        "computed_at": datetime.now(UTC),
        "source": "nfl_v1",
        "ingested_at": datetime.now(UTC),
    }
    base.update(overrides)
    return base


def test_joins_feature_vector_with_fixture_result() -> None:
    ParquetTableRepository(FIXTURES, get_settings().parquet_dir).write(
        pd.DataFrame([_fixture_row()])
    )
    ParquetTableRepository(FEATURES, get_settings().parquet_dir).write(
        pd.DataFrame([_feature_row()])
    )

    result = load_feature_vectors("usa-nfl", "nfl_v1")

    assert len(result) == 1
    row = result.iloc[0]
    assert row["home_score"] == 27
    assert row["away_score"] == 24
    assert row["vector"] == {"off_epa_rolling_4_home": 0.1}
    assert row["as_of_timestamp"] == datetime(2026, 1, 5, tzinfo=UTC)


def test_fixture_without_a_computed_vector_is_dropped() -> None:
    ParquetTableRepository(FIXTURES, get_settings().parquet_dir).write(
        pd.DataFrame(
            [
                _fixture_row(id="fix1"),
                _fixture_row(id="fix2", kickoff_utc=datetime(2026, 1, 11, tzinfo=UTC)),
            ]
        )
    )
    ParquetTableRepository(FEATURES, get_settings().parquet_dir).write(
        pd.DataFrame([_feature_row(fixture_id="fix1")])
    )

    result = load_feature_vectors("usa-nfl", "nfl_v1")

    assert list(result["id"]) == ["fix1"]


def test_filters_by_feature_set() -> None:
    ParquetTableRepository(FIXTURES, get_settings().parquet_dir).write(
        pd.DataFrame([_fixture_row()])
    )
    ParquetTableRepository(FEATURES, get_settings().parquet_dir).write(
        pd.DataFrame(
            [_feature_row(feature_set="nfl_v1"), _feature_row(feature_set="nfl_v2", vector="{}")]
        )
    )

    result = load_feature_vectors("usa-nfl", "nfl_v1")

    assert len(result) == 1
    assert result.iloc[0]["vector"] == {"off_epa_rolling_4_home": 0.1}


def test_sorted_by_kickoff() -> None:
    ParquetTableRepository(FIXTURES, get_settings().parquet_dir).write(
        pd.DataFrame(
            [
                _fixture_row(id="later", kickoff_utc=datetime(2026, 2, 1, tzinfo=UTC)),
                _fixture_row(id="earlier", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            ]
        )
    )
    ParquetTableRepository(FEATURES, get_settings().parquet_dir).write(
        pd.DataFrame([_feature_row(fixture_id="later"), _feature_row(fixture_id="earlier")])
    )

    result = load_feature_vectors("usa-nfl", "nfl_v1")

    assert list(result["id"]) == ["earlier", "later"]
