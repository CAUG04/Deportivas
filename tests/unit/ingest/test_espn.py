"""EspnSource mapping logic, tested against a small DataFrame shaped like
soccerdata.ESPN's real output (verified by reading soccerdata's source) —
never against the live network."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from deportivas.contracts.tables import TEAM_ALIASES, TEAMS
from deportivas.ingest.aliases import TeamAliasResolver
from deportivas.ingest.ratelimit import RateLimiter
from deportivas.ingest.sources.espn import EspnSource
from deportivas.storage.duckdb_repo.raw_store import ParquetRawDocumentRepository
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository


def _source(tmp_path: Path) -> EspnSource:
    raw_repo = ParquetRawDocumentRepository(tmp_path / "raw", tmp_path / "parquet")
    teams_repo = ParquetTableRepository(TEAMS, tmp_path / "parquet")
    aliases_repo = ParquetTableRepository(TEAM_ALIASES, tmp_path / "parquet")
    resolver = TeamAliasResolver(teams_repo, aliases_repo, sport="football")
    return EspnSource(
        raw_repo=raw_repo,
        rate_limiter=RateLimiter(0.0),
        data_dir=tmp_path / "cache",
        aliases=resolver,
    )


def _schedule_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "league": "col.1",
        "season": "2026",
        "game": "2026-01-10 millonarios-santafe",
        "date": pd.Timestamp("2026-01-10 20:00:00"),
        "home_team": "Millonarios",
        "away_team": "Santa Fe",
        "game_id": 123456,
        "league_id": "col.1",
    }
    base.update(overrides)
    return base


def test_to_fixtures_always_scheduled_no_scores(tmp_path: Path) -> None:
    """ESPN.read_schedule() carries no final score column — see module
    docstring — so every mapped fixture is "scheduled", even played ones."""
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row()])

    fixtures = source._to_fixtures(raw, competition_id="col-primera-a")

    row = fixtures.iloc[0]
    assert row["status"] == "scheduled"
    assert row["home_score"] is None
    assert row["away_score"] is None
    assert row["kickoff_utc"] == datetime(2026, 1, 10, 20, 0, tzinfo=UTC)


def test_to_fixtures_resolves_team_names(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row()])

    fixtures = source._to_fixtures(raw, competition_id="col-primera-a")

    assert fixtures.iloc[0]["home_team_id"] == "football:millonarios"
    assert fixtures.iloc[0]["away_team_id"] == "football:santa-fe"


def test_to_fixtures_row_without_date_is_dropped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row(date=None)])

    fixtures = source._to_fixtures(raw, competition_id="col-primera-a")

    assert fixtures.empty


class _FakeEspnReader:
    last_instance: _FakeEspnReader | None = None

    def __init__(self, *, leagues: list[str], seasons: list[str], data_dir: Path) -> None:
        self.leagues = leagues
        self.seasons = seasons
        self.data_dir = data_dir
        type(self).last_instance = self

    def read_schedule(self) -> pd.DataFrame:
        df = pd.DataFrame([_schedule_row()])
        return df.set_index(["league", "season", "game"])


def test_fetch_schedule_waits_archives_and_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deportivas.ingest.sources.espn.sd.ESPN", _FakeEspnReader)
    source = _source(tmp_path)
    waits: list[float] = []

    def fake_wait() -> float:
        waits.append(1.0)
        return 0.0

    monkeypatch.setattr(source, "_wait", fake_wait)

    fixtures = source.fetch_schedule(
        competition_id="col-primera-a", espn_league="col.1", seasons=["2026"]
    )

    assert len(waits) == 1
    assert len(fixtures) == 1
    assert _FakeEspnReader.last_instance is not None
    assert _FakeEspnReader.last_instance.leagues == ["col.1"]
