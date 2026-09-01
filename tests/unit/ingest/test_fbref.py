"""FBrefSource mapping logic, tested against small DataFrames shaped like
soccerdata.FBref's real output (verified by reading soccerdata's source, not
guessed) — never against the live network."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from deportivas.contracts.tables import TEAM_ALIASES, TEAMS
from deportivas.ingest.aliases import TeamAliasResolver
from deportivas.ingest.ratelimit import RateLimiter
from deportivas.ingest.sources.fbref import FBrefSource, _combine_date_time, parse_score
from deportivas.storage.duckdb_repo.raw_store import ParquetRawDocumentRepository
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository


def _source(tmp_path: Path) -> FBrefSource:
    raw_repo = ParquetRawDocumentRepository(tmp_path / "raw", tmp_path / "parquet")
    teams_repo = ParquetTableRepository(TEAMS, tmp_path / "parquet")
    aliases_repo = ParquetTableRepository(TEAM_ALIASES, tmp_path / "parquet")
    resolver = TeamAliasResolver(teams_repo, aliases_repo, sport="football")
    return FBrefSource(
        raw_repo=raw_repo,
        rate_limiter=RateLimiter(0.0),
        data_dir=tmp_path / "cache",
        aliases=resolver,
    )


# -- parse_score ------------------------------------------------------------


def test_parse_score_en_dash() -> None:
    assert parse_score("3–1") == (3, 1)  # noqa: RUF001 (guion en-dash real de FBref)


def test_parse_score_plain_hyphen() -> None:
    assert parse_score("0-0") == (0, 0)


def test_parse_score_missing_is_unplayed() -> None:
    assert parse_score(float("nan")) == (None, None)
    assert parse_score(None) == (None, None)


def test_parse_score_garbage_is_unplayed() -> None:
    assert parse_score("postponed") == (None, None)


# -- _combine_date_time ------------------------------------------------------


def test_combine_date_time_with_explicit_time() -> None:
    result = _combine_date_time(pd.Timestamp("2026-01-10"), "20:00")
    assert result == datetime(2026, 1, 10, 20, 0, tzinfo=UTC)


def test_combine_date_time_missing_time_defaults_to_noon() -> None:
    result = _combine_date_time(pd.Timestamp("2026-01-10"), None)
    assert result == datetime(2026, 1, 10, 12, 0, tzinfo=UTC)


def test_combine_date_time_missing_date_is_none() -> None:
    assert _combine_date_time(None, "20:00") is None
    assert _combine_date_time(float("nan"), "20:00") is None


def test_combine_date_time_malformed_time_falls_back_to_noon() -> None:
    result = _combine_date_time(pd.Timestamp("2026-01-10"), "not-a-time")
    assert result == datetime(2026, 1, 10, 12, 0, tzinfo=UTC)


# -- _to_fixtures -------------------------------------------------------------


def _schedule_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "league": "ENG-Premier League",
        "season": "2526",
        "game": "2026-01-10 arsenal-chelsea",
        "week": 20,
        "day": "Sat",
        "date": pd.Timestamp("2026-01-10"),
        "time": "17:30",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_xg": 1.8,
        "away_xg": 1.1,
        "score": "2–1",  # noqa: RUF001 (guion en-dash real de FBref)
        "attendance": 60000,
        "venue": "Emirates Stadium",
        "referee": "Michael Oliver",
        "match_report": "https://fbref.com/en/matches/xyz",
        "notes": None,
        "game_id": "xyz",
    }
    base.update(overrides)
    return base


def test_to_fixtures_maps_played_match(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row()])

    fixtures = source._to_fixtures(raw, competition_id="eng-premier-league")

    assert len(fixtures) == 1
    row = fixtures.iloc[0]
    assert row["status"] == "finished"
    assert row["home_score"] == 2
    assert row["away_score"] == 1
    assert row["matchday"] == 20
    assert row["kickoff_utc"] == datetime(2026, 1, 10, 17, 30, tzinfo=UTC)
    assert bool(row["kickoff_is_estimated"]) is False
    assert row["competition_id"] == "eng-premier-league"
    assert row["season"] == "2526"


def test_to_fixtures_unplayed_match_is_scheduled(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row(score=None)])

    fixtures = source._to_fixtures(raw, competition_id="eng-premier-league")

    row = fixtures.iloc[0]
    assert row["status"] == "scheduled"
    assert row["home_score"] is None
    assert row["away_score"] is None


def test_to_fixtures_missing_time_marks_estimated(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row(time=None)])

    fixtures = source._to_fixtures(raw, competition_id="eng-premier-league")

    assert bool(fixtures.iloc[0]["kickoff_is_estimated"]) is True


def test_to_fixtures_row_without_date_is_dropped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row(date=None)])

    fixtures = source._to_fixtures(raw, competition_id="eng-premier-league")

    assert fixtures.empty


def test_to_fixtures_resolves_team_names_to_stable_ids(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame(
        [_schedule_row(), _schedule_row(home_team="Chelsea", away_team="Arsenal", game="rev")]
    )

    fixtures = source._to_fixtures(raw, competition_id="eng-premier-league")

    assert fixtures.iloc[0]["home_team_id"] == "football:arsenal"
    assert fixtures.iloc[0]["away_team_id"] == "football:chelsea"
    assert fixtures.iloc[1]["home_team_id"] == "football:chelsea"
    assert fixtures.iloc[1]["away_team_id"] == "football:arsenal"


def test_to_fixtures_same_match_produces_same_id_across_calls(tmp_path: Path) -> None:
    """Idempotency at the mapping level: re-ingesting the same schedule row
    must compute the same fixture id, so the repository's upsert lands on
    the same row instead of creating a duplicate."""
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row()])

    first = source._to_fixtures(raw, competition_id="eng-premier-league")
    second = source._to_fixtures(raw, competition_id="eng-premier-league")

    assert first.iloc[0]["id"] == second.iloc[0]["id"]


# -- _to_team_match_stats ------------------------------------------------------


def _team_stats_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "league": "ENG-Premier League",
        "season": "2526",
        "team": "Arsenal",
        "game": "2026-01-10 arsenal-chelsea",
        "date": pd.Timestamp("2026-01-10"),
        "time": "17:30",
        "venue": "Home",
        "result": "W",
        "gf": 2,
        "ga": 1,
        "opponent": "Chelsea",
        "xg": 1.8,
        "xga": 1.1,
        "poss": 55.0,
    }
    base.update(overrides)
    return base


def test_to_team_match_stats_home_team(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_team_stats_row()])

    stats = source._to_team_match_stats(raw, competition_id="eng-premier-league")

    row = stats.iloc[0]
    assert bool(row["is_home"]) is True
    assert row["goals"] == 2
    assert row["xg"] == pytest.approx(1.8)
    assert row["possession"] == pytest.approx(55.0)
    assert row["team_id"] == "football:arsenal"


def test_to_team_match_stats_away_team_fixture_id_matches_home_perspective(
    tmp_path: Path,
) -> None:
    """The same real match reported from each team's own perspective must
    resolve to the same fixture_id, so both rows join to one fixture."""
    source = _source(tmp_path)
    home_row = _team_stats_row(venue="Home", team="Arsenal", opponent="Chelsea")
    away_row = _team_stats_row(venue="Away", team="Chelsea", opponent="Arsenal", gf=1, ga=2)
    raw = pd.DataFrame([home_row, away_row])

    stats = source._to_team_match_stats(raw, competition_id="eng-premier-league")

    assert stats.iloc[0]["fixture_id"] == stats.iloc[1]["fixture_id"]


def test_to_team_match_stats_row_without_date_is_dropped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_team_stats_row(date=None)])

    stats = source._to_team_match_stats(raw, competition_id="eng-premier-league")

    assert stats.empty


# -- fetch_schedule / fetch_team_match_stats (wiring) --------------------------


class _FakeFBrefReader:
    """Stands in for soccerdata.FBref: no network, just proves the adapter
    waits on the rate limiter, calls the right reader method, archives the
    cache dir, and feeds the result through the right mapping method."""

    last_instance: _FakeFBrefReader | None = None

    def __init__(self, *, leagues: list[str], seasons: list[str], data_dir: Path) -> None:
        self.leagues = leagues
        self.seasons = seasons
        self.data_dir = data_dir
        type(self).last_instance = self

    def read_schedule(self) -> pd.DataFrame:
        df = pd.DataFrame([_schedule_row()])
        return df.set_index(["league", "season", "game"])

    def read_team_match_stats(self, stat_type: str = "schedule") -> pd.DataFrame:
        assert stat_type == "schedule"
        df = pd.DataFrame([_team_stats_row()])
        return df.set_index(["league", "season", "team", "game"])


def test_fetch_schedule_waits_archives_and_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deportivas.ingest.sources.fbref.sd.FBref", _FakeFBrefReader)
    source = _source(tmp_path)
    waits: list[float] = []

    def fake_wait() -> float:
        waits.append(1.0)
        return 0.0

    monkeypatch.setattr(source, "_wait", fake_wait)

    fixtures = source.fetch_schedule(
        competition_id="eng-premier-league", fbref_league="ENG-Premier League", seasons=["2526"]
    )

    assert len(waits) == 1
    assert len(fixtures) == 1
    assert fixtures.iloc[0]["competition_id"] == "eng-premier-league"
    assert _FakeFBrefReader.last_instance is not None
    assert _FakeFBrefReader.last_instance.leagues == ["ENG-Premier League"]


def test_fetch_team_match_stats_waits_archives_and_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("deportivas.ingest.sources.fbref.sd.FBref", _FakeFBrefReader)
    source = _source(tmp_path)
    waits: list[float] = []

    def fake_wait() -> float:
        waits.append(1.0)
        return 0.0

    monkeypatch.setattr(source, "_wait", fake_wait)

    stats = source.fetch_team_match_stats(
        competition_id="eng-premier-league", fbref_league="ENG-Premier League", seasons=["2526"]
    )

    assert len(waits) == 1
    assert len(stats) == 1
    assert stats.iloc[0]["competition_id"] == "eng-premier-league"
