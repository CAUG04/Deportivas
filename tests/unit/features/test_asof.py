"""asof.py: point-in-time loaders used by every feature pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FIXTURES, NFL_TEAM_GAME_STATS, TEAM_MATCH_STATS
from deportivas.features.asof import load_fixtures, load_nfl_team_game_stats, load_team_match_stats
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
        "competition_id": "eng-premier-league",
        "season": "2526",
        "kickoff_utc": datetime(2026, 1, 10, 17, 30, tzinfo=UTC),
        "home_team_id": "football:arsenal",
        "away_team_id": "football:chelsea",
        "status": "finished",
        "stage": None,
        "matchday": 1,
        "home_score": 2,
        "away_score": 1,
        "home_score_ht": None,
        "away_score_ht": None,
        "kickoff_is_estimated": False,
        "source": "test",
        "ingested_at": datetime.now(UTC),
    }
    base.update(overrides)
    return base


def test_load_fixtures_sorts_by_kickoff() -> None:
    repo = ParquetTableRepository(FIXTURES, get_settings().parquet_dir)
    repo.write(
        pd.DataFrame(
            [
                _fixture_row(id="later", kickoff_utc=datetime(2026, 2, 1, tzinfo=UTC)),
                _fixture_row(id="earlier", kickoff_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            ]
        )
    )

    result = load_fixtures("eng-premier-league")

    assert list(result["id"]) == ["earlier", "later"]


def test_load_fixtures_filters_by_competition() -> None:
    repo = ParquetTableRepository(FIXTURES, get_settings().parquet_dir)
    repo.write(
        pd.DataFrame(
            [
                _fixture_row(id="epl", competition_id="eng-premier-league"),
                _fixture_row(id="laliga", competition_id="esp-la-liga", season="2526"),
            ]
        )
    )

    result = load_fixtures("eng-premier-league")

    assert list(result["id"]) == ["epl"]


def _stats_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "fixture_id": "fix1",
        "team_id": "football:arsenal",
        "source": "fbref",
        "competition_id": "eng-premier-league",
        "season": "2526",
        "is_home": True,
        "goals": 2,
        "shots": 15,
        "shots_on_target": 6,
        "xg": 1.8,
        "corners": 7,
        "yellow_cards": 1,
        "red_cards": 0,
        "possession": 0.55,
        "ingested_at": datetime.now(UTC),
    }
    base.update(overrides)
    return base


def test_load_team_match_stats_prefers_fbref_over_understat() -> None:
    repo = ParquetTableRepository(TEAM_MATCH_STATS, get_settings().parquet_dir)
    repo.write(
        pd.DataFrame(
            [
                _stats_row(source="understat", xg=1.5),
                _stats_row(source="fbref", xg=1.8),
            ]
        )
    )

    result = load_team_match_stats("eng-premier-league")

    assert len(result) == 1
    assert result.iloc[0]["source"] == "fbref"
    assert result.iloc[0]["xg"] == pytest.approx(1.8)


def test_load_team_match_stats_falls_back_when_preferred_source_absent() -> None:
    repo = ParquetTableRepository(TEAM_MATCH_STATS, get_settings().parquet_dir)
    repo.write(pd.DataFrame([_stats_row(source="understat", xg=1.5)]))

    result = load_team_match_stats("eng-premier-league")

    assert len(result) == 1
    assert result.iloc[0]["source"] == "understat"


def test_load_team_match_stats_unknown_source_still_included_last() -> None:
    repo = ParquetTableRepository(TEAM_MATCH_STATS, get_settings().parquet_dir)
    repo.write(
        pd.DataFrame(
            [
                _stats_row(source="some_future_source", xg=9.9),
                _stats_row(source="understat", xg=1.5),
            ]
        )
    )

    result = load_team_match_stats("eng-premier-league")

    assert len(result) == 1
    assert result.iloc[0]["source"] == "understat"


def test_load_team_match_stats_empty_competition_returns_empty() -> None:
    result = load_team_match_stats("does-not-exist")
    assert result.empty


def test_load_team_match_stats_keeps_both_teams_of_same_fixture() -> None:
    repo = ParquetTableRepository(TEAM_MATCH_STATS, get_settings().parquet_dir)
    repo.write(
        pd.DataFrame(
            [
                _stats_row(team_id="football:arsenal", is_home=True),
                _stats_row(team_id="football:chelsea", is_home=False),
            ]
        )
    )

    result = load_team_match_stats("eng-premier-league")

    assert len(result) == 2
    assert set(result["team_id"]) == {"football:arsenal", "football:chelsea"}


def _nfl_stats_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "fixture_id": "fix1",
        "team_id": "american_football:buf",
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
    base.update(overrides)
    return base


def test_load_nfl_team_game_stats_filters_by_competition() -> None:
    repo = ParquetTableRepository(NFL_TEAM_GAME_STATS, get_settings().parquet_dir)
    repo.write(
        pd.DataFrame(
            [
                _nfl_stats_row(fixture_id="usa", competition_id="usa-nfl"),
                _nfl_stats_row(fixture_id="other", competition_id="other-league", season="2025"),
            ]
        )
    )

    result = load_nfl_team_game_stats("usa-nfl")

    assert list(result["fixture_id"]) == ["usa"]


def test_load_nfl_team_game_stats_dedupes_by_source_priority() -> None:
    repo = ParquetTableRepository(NFL_TEAM_GAME_STATS, get_settings().parquet_dir)
    repo.write(
        pd.DataFrame(
            [
                _nfl_stats_row(source="nfl", offensive_epa_per_play=0.1),
                _nfl_stats_row(source="some_future_source", offensive_epa_per_play=0.9),
            ]
        )
    )

    result = load_nfl_team_game_stats("usa-nfl")

    assert len(result) == 1
    assert result.iloc[0]["source"] == "nfl"


def test_load_nfl_team_game_stats_empty_competition_returns_empty() -> None:
    result = load_nfl_team_game_stats("does-not-exist")
    assert result.empty
