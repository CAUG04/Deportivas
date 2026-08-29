"""pipeline.py: MLB feature vector (rest/back-to-back/scoring margin), then
hands it to write_features. MLB tunes rest_and_margin.py's defaults: longer
windows for a 162-game season, and a same-day-only back-to-back threshold
since day-to-day games are the norm rather than the exception."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FEATURES, FIXTURES
from deportivas.features.mlb.pipeline import compute_and_write_mlb_features, compute_mlb_features
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
        "competition_id": "usa-mlb",
        "season": "2025",
        "home_team_id": "yankees",
        "away_team_id": "redsox",
        "kickoff_utc": datetime(2026, 6, 10, tzinfo=UTC),
        "status": "finished",
        "home_score": 5,
        "away_score": 3,
    }
    base.update(overrides)
    return base


def test_vector_contains_rest_back_to_back_and_margin_keys() -> None:
    fixtures = pd.DataFrame([_fixture()])

    result = compute_mlb_features(fixtures)

    vector = result.iloc[0]["vector"]
    assert "rest_days_home" in vector
    assert "back_to_back_home" in vector
    assert "scoring_margin_rolling_10_home" in vector


def test_uses_longer_windows_than_the_nba_nhl_default() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_mlb_features(fixtures)
    vector = result.iloc[0]["vector"]
    assert "scoring_margin_rolling_40_home" in vector
    assert "scoring_margin_rolling_5_home" not in vector


def test_playing_the_next_day_is_not_flagged_back_to_back() -> None:
    """Consecutive-day games are routine in MLB; the tighter threshold should
    only flag a same-day doubleheader, not the ordinary daily schedule."""
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 6, 1, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="yankees",
                away_team_id="redsox",
                kickoff_utc=datetime(2026, 6, 2, tzinfo=UTC),
            ),
        ]
    )
    result = compute_mlb_features(fixtures)
    assert result.iloc[1]["vector"]["back_to_back_home"] is False


def test_same_day_doubleheader_is_flagged_back_to_back() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id="f1", kickoff_utc=datetime(2026, 6, 1, 17, 0, tzinfo=UTC)),
            _fixture(
                id="f2",
                home_team_id="yankees",
                away_team_id="redsox",
                kickoff_utc=datetime(2026, 6, 1, 20, 0, tzinfo=UTC),
            ),
        ]
    )
    result = compute_mlb_features(fixtures)
    assert result.iloc[1]["vector"]["back_to_back_home"] is True


def test_output_carries_competition_id_and_season() -> None:
    fixtures = pd.DataFrame([_fixture()])
    result = compute_mlb_features(fixtures)
    assert result.iloc[0]["competition_id"] == "usa-mlb"
    assert result.iloc[0]["season"] == "2025"


def test_all_as_of_strictly_before_kickoff() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture(id=f"f{i}", kickoff_utc=datetime(2026, 6, 1, tzinfo=UTC) + timedelta(days=i))
            for i in range(6)
        ]
    )
    result = compute_mlb_features(fixtures)
    merged = result.merge(fixtures, left_on="fixture_id", right_on="id")
    assert (merged["as_of_timestamp"] < merged["kickoff_utc"]).all()


def test_compute_and_write_mlb_features_persists_the_vector() -> None:
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

    written = compute_and_write_mlb_features("usa-mlb")

    assert written == 1
    features = ParquetTableRepository(FEATURES, get_settings().parquet_dir).read()
    assert len(features) == 1
    assert features.iloc[0]["feature_set"] == "mlb_v1"
