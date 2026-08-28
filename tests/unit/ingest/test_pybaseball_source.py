"""PybaseballSource mapping logic, tested against small DataFrames shaped
like the long-documented baseball-reference schedule format.

Unlike the other adapters, these column names are NOT verified against the
installed library's source (schedule_and_record scrapes HTML directly — see
the module docstring for why). These tests protect the mapping logic itself
against regressions; they do not prove the assumed column names are correct."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from deportivas.contracts.tables import TEAM_ALIASES, TEAMS
from deportivas.ingest.aliases import TeamAliasResolver
from deportivas.ingest.ratelimit import RateLimiter
from deportivas.ingest.sources.pybaseball_source import PybaseballSource
from deportivas.storage.duckdb_repo.raw_store import ParquetRawDocumentRepository
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository


def _source(tmp_path: Path) -> PybaseballSource:
    raw_repo = ParquetRawDocumentRepository(tmp_path / "raw", tmp_path / "parquet")
    teams_repo = ParquetTableRepository(TEAMS, tmp_path / "parquet")
    aliases_repo = ParquetTableRepository(TEAM_ALIASES, tmp_path / "parquet")
    resolver = TeamAliasResolver(teams_repo, aliases_repo, sport="baseball")
    return PybaseballSource(raw_repo=raw_repo, rate_limiter=RateLimiter(0.0), aliases=resolver)


def _home_game_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "Date": "Monday, Apr 7",
        "Tm": "PHI",
        "Home_Away": float("nan"),
        "Opp": "ATL",
        "W/L": "W",
        "R": 5,
        "RA": 2,
    }
    base.update(overrides)
    return base


def _away_game_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "Date": "Tuesday, Apr 8",
        "Tm": "PHI",
        "Home_Away": "@",
        "Opp": "ATL",
        "W/L": "L",
        "R": 3,
        "RA": 6,
    }
    base.update(overrides)
    return base


def test_home_game_orientation_and_score(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_home_game_row()])

    fixtures = source._to_fixtures(raw, competition_id="usa-mlb", season=2025)

    row = fixtures.iloc[0]
    assert row["home_team_id"] == "baseball:phi"
    assert row["away_team_id"] == "baseball:atl"
    assert row["home_score"] == 5
    assert row["away_score"] == 2
    assert row["kickoff_utc"] == datetime(2025, 4, 7, 19, 0, tzinfo=UTC)


def test_away_game_orientation_and_score(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_away_game_row()])

    fixtures = source._to_fixtures(raw, competition_id="usa-mlb", season=2025)

    row = fixtures.iloc[0]
    assert row["home_team_id"] == "baseball:atl"
    assert row["away_team_id"] == "baseball:phi"
    assert row["home_score"] == 6
    assert row["away_score"] == 3


def test_home_and_away_rows_for_same_game_produce_same_fixture_id(tmp_path: Path) -> None:
    """Both teams' schedule_and_record calls report the same real game; the
    deterministic hash must collapse them onto one fixture row."""
    source = _source(tmp_path)
    home_side = source._to_fixtures(
        pd.DataFrame([_home_game_row()]), competition_id="usa-mlb", season=2025
    )
    # Mismo partido real desde el angulo de ATL: ATL jugo de visitante (R/RA
    # invertidos respecto a la fila de PHI, que jugo de local).
    away_side = source._to_fixtures(
        pd.DataFrame([_away_game_row(Tm="ATL", Opp="PHI", R=2, RA=5, Date="Monday, Apr 7")]),
        competition_id="usa-mlb",
        season=2025,
    )

    assert home_side.iloc[0]["id"] == away_side.iloc[0]["id"]


def test_doubleheader_suffix_does_not_break_date_parsing(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_home_game_row(Date="Sunday, Sep 28 (1)")])

    fixtures = source._to_fixtures(raw, competition_id="usa-mlb", season=2025)

    assert len(fixtures) == 1
    assert fixtures.iloc[0]["kickoff_utc"] == datetime(2025, 9, 28, 19, 0, tzinfo=UTC)


def test_missing_date_row_is_skipped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_home_game_row(Date=None)])

    fixtures = source._to_fixtures(raw, competition_id="usa-mlb", season=2025)

    assert fixtures.empty


def test_unparseable_date_row_is_skipped_not_crashed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_home_game_row(Date="not a real date at all !!")])

    fixtures = source._to_fixtures(raw, competition_id="usa-mlb", season=2025)

    assert fixtures.empty


def test_unplayed_game_is_scheduled(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_home_game_row(R=None, RA=None)])

    fixtures = source._to_fixtures(raw, competition_id="usa-mlb", season=2025)

    assert fixtures.iloc[0]["status"] == "scheduled"


def test_row_missing_team_or_opponent_is_dropped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_home_game_row(Opp=None)])

    fixtures = source._to_fixtures(raw, competition_id="usa-mlb", season=2025)

    assert fixtures.empty


def test_fetch_schedule_iterates_teams_waits_and_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    waits: list[float] = []
    calls: list[tuple[int, str]] = []

    def fake_wait() -> float:
        waits.append(1.0)
        return 0.0

    def fake_schedule_and_record(season: int, team: str) -> pd.DataFrame:
        calls.append((season, team))
        return pd.DataFrame([_home_game_row(Tm=team)])

    monkeypatch.setattr(source, "_wait", fake_wait)
    monkeypatch.setattr(
        "deportivas.ingest.sources.pybaseball_source.pb.schedule_and_record",
        fake_schedule_and_record,
    )

    fixtures = source.fetch_schedule(
        competition_id="usa-mlb", season=2025, team_abbreviations=["PHI", "ATL"]
    )

    assert len(waits) == 2
    assert calls == [(2025, "PHI"), (2025, "ATL")]
    assert len(fixtures) == 2


def test_fetch_schedule_empty_team_list_returns_empty_dataframe(tmp_path: Path) -> None:
    source = _source(tmp_path)
    fixtures = source.fetch_schedule(competition_id="usa-mlb", season=2025, team_abbreviations=[])
    assert fixtures.empty
