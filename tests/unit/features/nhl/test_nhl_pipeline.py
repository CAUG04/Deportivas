"""pipeline.py: NHL feature vector (rest/back-to-back/scoring margin), then
hands it to write_features."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FEATURES, FIXTURES
from deportivas.features.nhl.pipeline import compute_and_write_nhl_features, compute_nhl_features
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
        "competition_id": "usa-nhl",
        "season": "2025",
        "home_team_id": "bruins",
        "away_team_id": "rangers",
        "kickoff_utc": datetime(2026, 1, 10, tzinfo=UTC),
        "status": "finished",
        "home_score": 4,
        "away_score": 2,
    }
    base.update(overrides)
    return base


def test_vector_contains_rest_back_to_back_and_margin_keys() -> None:
    fixtures = pd.DataFrame([_fixture()])

    result = compute_nhl_features(fixtures)

    vector = result.iloc[0]["vector"]
    assert "rest_days_home" in vector
    assert "back_to_back_home" in vector
    assert "scoring_margin_rolling_5_home" in vector


def test_output_carries_competition_id_and_season() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_nhl_features(fixtures)
    assert result.iloc[0]["competition_id"] == "usa-nhl"
    assert result.iloc[0]["season"] == "2025"


def test_one_row_per_fixture() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id=f"f{i}", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i))
            for i in range(6)
        ]
    )
    result = compute_nhl_features(fixtures)
    assert len(result) == 6


def test_all_as_of_strictly_before_kickoff() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id=f"f{i}", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i))
            for i in range(6)
        ]
    )
    result = compute_nhl_features(fixtures)
    merged = result.merge(fixtures, left_on="fixture_id", right_on="id")
    assert (merged["as_of_timestamp"] < merged["kickoff_utc"]).all()


def test_back_to_back_flagged_for_games_one_day_apart() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="bruins",
                away_team_id="rangers",
                kickoff_utc=datetime(2026, 1, 2, tzinfo=UTC),
            ),
        ]
    )
    result = compute_nhl_features(fixtures)
    assert result.iloc[1]["vector"]["back_to_back_home"] is True


def test_compute_and_write_nhl_features_persists_the_vector() -> None:
    ParquetTableRepository(FIXTURES, get_settings().parquet_dir).write(
        pd.DataFrame(
            [
                {
                    **_fixture(),
                    "stage": None,
                    "matchday": None,
                    "home_score_ht": None,
                    "away_score_ht": None,
                    "kickoff_is_estimated": True,
                    "source": "test",
                    "ingested_at": datetime.now(UTC),
                }
            ]
        )
    )

    written = compute_and_write_nhl_features("usa-nhl")

    assert written == 1
    features = ParquetTableRepository(FEATURES, get_settings().parquet_dir).read()
    assert len(features) == 1
    assert features.iloc[0]["feature_set"] == "nhl_v1"
