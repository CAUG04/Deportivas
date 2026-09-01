"""NbaSource / NhlSource mapping logic, tested against small DataFrames
shaped like sportsdataverse's real (documented) output — never against the
live network."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from deportivas.contracts.tables import TEAM_ALIASES, TEAMS
from deportivas.ingest.aliases import TeamAliasResolver
from deportivas.ingest.ratelimit import RateLimiter
from deportivas.ingest.sources.sportsdataverse_source import NbaSource, NhlSource
from deportivas.storage.duckdb_repo.raw_store import ParquetRawDocumentRepository
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository


def _resolver(tmp_path: Path, sport: str) -> TeamAliasResolver:
    teams_repo = ParquetTableRepository(TEAMS, tmp_path / "parquet")
    aliases_repo = ParquetTableRepository(TEAM_ALIASES, tmp_path / "parquet")
    return TeamAliasResolver(teams_repo, aliases_repo, sport=sport)


def _nba_source(tmp_path: Path) -> NbaSource:
    raw_repo = ParquetRawDocumentRepository(tmp_path / "raw", tmp_path / "parquet")
    return NbaSource(
        raw_repo=raw_repo, rate_limiter=RateLimiter(0.0), aliases=_resolver(tmp_path, "basketball")
    )


def _nhl_source(tmp_path: Path) -> NhlSource:
    raw_repo = ParquetRawDocumentRepository(tmp_path / "raw", tmp_path / "parquet")
    return NhlSource(
        raw_repo=raw_repo, rate_limiter=RateLimiter(0.0), aliases=_resolver(tmp_path, "ice_hockey")
    )


# -- NBA ------------------------------------------------------------------


def _nba_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "game_id": 401585123,
        "season": 2025,
        "season_type": 2,
        "game_date_time": pd.Timestamp("2025-11-16 19:30:00", tz="America/New_York"),
        "game_date": pd.Timestamp("2025-11-16"),
        "home_abbreviation": "BOS",
        "away_abbreviation": "LAL",
        "home_score": 112,
        "away_score": 108,
        "status_type_completed": True,
    }
    base.update(overrides)
    return base


def test_nba_to_fixtures_converts_eastern_to_utc(tmp_path: Path) -> None:
    source = _nba_source(tmp_path)
    raw = pd.DataFrame([_nba_row()])

    fixtures = source._to_fixtures(raw, competition_id="usa-nba")

    row = fixtures.iloc[0]
    assert row["kickoff_utc"] == datetime(2025, 11, 17, 0, 30, tzinfo=UTC)  # EST = UTC-5
    assert row["status"] == "finished"
    assert bool(row["kickoff_is_estimated"]) is False


def test_nba_to_fixtures_unfinished_game_is_scheduled(tmp_path: Path) -> None:
    source = _nba_source(tmp_path)
    raw = pd.DataFrame([_nba_row(status_type_completed=False)])

    fixtures = source._to_fixtures(raw, competition_id="usa-nba")

    assert fixtures.iloc[0]["status"] == "scheduled"


def test_nba_to_fixtures_row_without_datetime_is_dropped(tmp_path: Path) -> None:
    source = _nba_source(tmp_path)
    raw = pd.DataFrame([_nba_row(game_date_time=None)])

    fixtures = source._to_fixtures(raw, competition_id="usa-nba")

    assert fixtures.empty


def test_nba_to_fixtures_resolves_team_abbreviations(tmp_path: Path) -> None:
    source = _nba_source(tmp_path)
    raw = pd.DataFrame([_nba_row()])

    fixtures = source._to_fixtures(raw, competition_id="usa-nba")

    assert fixtures.iloc[0]["home_team_id"] == "basketball:bos"
    assert fixtures.iloc[0]["away_team_id"] == "basketball:lal"


class _FakeNbaLoader:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, *, seasons: list[int], return_as_pandas: bool) -> pd.DataFrame:
        self.calls.append({"seasons": seasons, "return_as_pandas": return_as_pandas})
        return pd.DataFrame([_nba_row()])


def test_nba_fetch_schedule_waits_archives_and_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _nba_source(tmp_path)
    waits: list[float] = []

    def fake_wait() -> float:
        waits.append(1.0)
        return 0.0

    monkeypatch.setattr(source, "_wait", fake_wait)
    fake_loader = _FakeNbaLoader()
    monkeypatch.setattr(
        "deportivas.ingest.sources.sportsdataverse_source.sdv_nba.load_nba_schedule", fake_loader
    )

    fixtures = source.fetch_schedule(competition_id="usa-nba", seasons=[2025])

    assert len(waits) == 1
    assert len(fixtures) == 1
    assert fake_loader.calls == [{"seasons": [2025], "return_as_pandas": True}]


# -- NHL ------------------------------------------------------------------


def _nhl_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "game_id": 2025020123,
        "season": 20252026,
        "game_type": "2",
        "game_date": pd.Timestamp("2025-11-16"),
        "game_time": "19:00",
        "home_team_abbr": "TOR",
        "away_team_abbr": "MTL",
        "home_score": 4,
        "away_score": 2,
        "game_state": "OFF",
    }
    base.update(overrides)
    return base


def test_nhl_to_fixtures_uses_nominal_hour_and_flags_estimated(tmp_path: Path) -> None:
    source = _nhl_source(tmp_path)
    raw = pd.DataFrame([_nhl_row()])

    fixtures = source._to_fixtures(raw, competition_id="usa-nhl")

    row = fixtures.iloc[0]
    assert row["kickoff_utc"] == datetime(2025, 11, 16, 19, 0, tzinfo=UTC)
    assert bool(row["kickoff_is_estimated"]) is True


def test_nhl_to_fixtures_final_game_state_is_finished(tmp_path: Path) -> None:
    source = _nhl_source(tmp_path)
    raw = pd.DataFrame([_nhl_row(game_state="FINAL")])

    fixtures = source._to_fixtures(raw, competition_id="usa-nhl")

    assert fixtures.iloc[0]["status"] == "finished"


def test_nhl_to_fixtures_scheduled_game_state_is_scheduled(tmp_path: Path) -> None:
    source = _nhl_source(tmp_path)
    raw = pd.DataFrame([_nhl_row(game_state="SCHEDULED")])

    fixtures = source._to_fixtures(raw, competition_id="usa-nhl")

    assert fixtures.iloc[0]["status"] == "scheduled"


def test_nhl_to_fixtures_nan_game_type_is_none_stage(tmp_path: Path) -> None:
    source = _nhl_source(tmp_path)
    raw = pd.DataFrame([_nhl_row(game_type=float("nan"))])

    fixtures = source._to_fixtures(raw, competition_id="usa-nhl")

    assert fixtures.iloc[0]["stage"] is None


def test_nhl_to_fixtures_row_without_date_is_dropped(tmp_path: Path) -> None:
    source = _nhl_source(tmp_path)
    raw = pd.DataFrame([_nhl_row(game_date=None)])

    fixtures = source._to_fixtures(raw, competition_id="usa-nhl")

    assert fixtures.empty


def test_nhl_to_fixtures_resolves_team_abbreviations(tmp_path: Path) -> None:
    source = _nhl_source(tmp_path)
    raw = pd.DataFrame([_nhl_row()])

    fixtures = source._to_fixtures(raw, competition_id="usa-nhl")

    assert fixtures.iloc[0]["home_team_id"] == "ice_hockey:tor"
    assert fixtures.iloc[0]["away_team_id"] == "ice_hockey:mtl"


class _FakeNhlLoader:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, *, seasons: list[int], return_as_pandas: bool) -> pd.DataFrame:
        self.calls.append({"seasons": seasons, "return_as_pandas": return_as_pandas})
        return pd.DataFrame([_nhl_row()])


def test_nhl_fetch_schedule_waits_archives_and_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _nhl_source(tmp_path)
    waits: list[float] = []

    def fake_wait() -> float:
        waits.append(1.0)
        return 0.0

    monkeypatch.setattr(source, "_wait", fake_wait)
    fake_loader = _FakeNhlLoader()
    monkeypatch.setattr(
        "deportivas.ingest.sources.sportsdataverse_source.sdv_nhl.load_nhl_schedule", fake_loader
    )

    fixtures = source.fetch_schedule(competition_id="usa-nhl", seasons=[2025])

    assert len(waits) == 1
    assert len(fixtures) == 1
    assert fake_loader.calls == [{"seasons": [2025], "return_as_pandas": True}]
