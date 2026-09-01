"""UnderstatSource mapping logic, tested against a small DataFrame shaped like
soccerdata.Understat's real output (verified by reading soccerdata's source,
which builds one wide row per match with home_*/away_* columns) — never
against the live network."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from deportivas.contracts.tables import TEAM_ALIASES, TEAMS
from deportivas.ingest.aliases import TeamAliasResolver
from deportivas.ingest.ratelimit import RateLimiter
from deportivas.ingest.sources.understat import UnderstatSource
from deportivas.storage.duckdb_repo.raw_store import ParquetRawDocumentRepository
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository


def _source(tmp_path: Path) -> UnderstatSource:
    raw_repo = ParquetRawDocumentRepository(tmp_path / "raw", tmp_path / "parquet")
    teams_repo = ParquetTableRepository(TEAMS, tmp_path / "parquet")
    aliases_repo = ParquetTableRepository(TEAM_ALIASES, tmp_path / "parquet")
    resolver = TeamAliasResolver(teams_repo, aliases_repo, sport="football")
    return UnderstatSource(
        raw_repo=raw_repo,
        rate_limiter=RateLimiter(0.0),
        data_dir=tmp_path / "cache",
        aliases=resolver,
    )


def _match_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "league": "ENG-Premier League",
        "season": "2526",
        "game": "2026-01-10 arsenal-chelsea",
        "league_id": 1,
        "season_id": 1,
        "game_id": 999,
        "date": pd.Timestamp("2026-01-10 17:30:00"),
        "home_team_id": 1,
        "away_team_id": 2,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_team_code": "ARS",
        "away_team_code": "CHE",
        "home_points": 3,
        "away_points": 0,
        "home_expected_points": 2.1,
        "away_expected_points": 0.9,
        "home_goals": 2,
        "away_goals": 1,
        "home_xg": 1.8,
        "away_xg": 1.1,
        "home_np_xg": 1.6,
        "away_np_xg": 1.1,
        "home_np_xg_difference": 0.5,
        "away_np_xg_difference": -0.5,
        "home_ppda": 8.2,
        "away_ppda": 11.4,
        "home_deep_completions": 6,
        "away_deep_completions": 4,
    }
    base.update(overrides)
    return base


def test_to_team_match_stats_splits_one_row_into_two(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_match_row()])

    stats = source._to_team_match_stats(raw, competition_id="eng-premier-league")

    assert len(stats) == 2
    assert set(stats["is_home"]) == {True, False}


def test_to_team_match_stats_home_row_values(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_match_row()])

    stats = source._to_team_match_stats(raw, competition_id="eng-premier-league")

    home = stats[stats["is_home"]].iloc[0]
    assert home["team_id"] == "football:arsenal"
    assert home["goals"] == 2
    assert home["xg"] == pytest.approx(1.8)
    assert home["shots"] is None
    assert home["source"] == "understat"


def test_to_team_match_stats_away_row_values(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_match_row()])

    stats = source._to_team_match_stats(raw, competition_id="eng-premier-league")

    away = stats[~stats["is_home"]].iloc[0]
    assert away["team_id"] == "football:chelsea"
    assert away["goals"] == 1
    assert away["xg"] == pytest.approx(1.1)


def test_to_team_match_stats_both_rows_share_fixture_id(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_match_row()])

    stats = source._to_team_match_stats(raw, competition_id="eng-premier-league")

    assert stats.iloc[0]["fixture_id"] == stats.iloc[1]["fixture_id"]


def test_to_team_match_stats_fixture_id_matches_fbref_style_hash(tmp_path: Path) -> None:
    from deportivas.domain.ids import fixture_id

    source = _source(tmp_path)
    raw = pd.DataFrame([_match_row()])

    stats = source._to_team_match_stats(raw, competition_id="eng-premier-league")

    expected = fixture_id(
        "eng-premier-league",
        "2526",
        "football:arsenal",
        "football:chelsea",
        datetime(2026, 1, 10, 17, 30, tzinfo=UTC),
    )
    assert stats.iloc[0]["fixture_id"] == expected


def test_to_team_match_stats_row_without_date_is_dropped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_match_row(date=None)])

    stats = source._to_team_match_stats(raw, competition_id="eng-premier-league")

    assert stats.empty


class _FakeUnderstatReader:
    last_instance: _FakeUnderstatReader | None = None

    def __init__(self, *, leagues: list[str], seasons: list[str], data_dir: Path) -> None:
        self.leagues = leagues
        self.seasons = seasons
        self.data_dir = data_dir
        type(self).last_instance = self

    def read_team_match_stats(self) -> pd.DataFrame:
        df = pd.DataFrame([_match_row()])
        return df.set_index(["league", "season", "game"])


def test_fetch_team_match_stats_waits_archives_and_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deportivas.ingest.sources.understat.sd.Understat", _FakeUnderstatReader)
    source = _source(tmp_path)
    waits: list[float] = []

    def fake_wait() -> float:
        waits.append(1.0)
        return 0.0

    monkeypatch.setattr(source, "_wait", fake_wait)

    stats = source.fetch_team_match_stats(
        competition_id="eng-premier-league", understat_league="EPL", seasons=["2526"]
    )

    assert len(waits) == 1
    assert len(stats) == 2
    assert _FakeUnderstatReader.last_instance is not None
    assert _FakeUnderstatReader.last_instance.leagues == ["EPL"]
