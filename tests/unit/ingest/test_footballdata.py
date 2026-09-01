"""FootballDataSource mapping logic, tested against small DataFrames shaped
like soccerdata.MatchHistory's real output (verified by reading soccerdata's
source) — never against the live network."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from deportivas.contracts.tables import TEAM_ALIASES, TEAMS
from deportivas.ingest.aliases import TeamAliasResolver
from deportivas.ingest.ratelimit import RateLimiter
from deportivas.ingest.sources.footballdata import FootballDataSource
from deportivas.storage.duckdb_repo.raw_store import ParquetRawDocumentRepository
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository


def _source(tmp_path: Path) -> FootballDataSource:
    raw_repo = ParquetRawDocumentRepository(tmp_path / "raw", tmp_path / "parquet")
    teams_repo = ParquetTableRepository(TEAMS, tmp_path / "parquet")
    aliases_repo = ParquetTableRepository(TEAM_ALIASES, tmp_path / "parquet")
    resolver = TeamAliasResolver(teams_repo, aliases_repo, sport="football")
    return FootballDataSource(
        raw_repo=raw_repo,
        rate_limiter=RateLimiter(0.0),
        data_dir=tmp_path / "cache",
        aliases=resolver,
    )


def _game_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "league": "ENG-Premier League",
        "season": "2526",
        "game": "2026-01-10 arsenal-chelsea",
        "date": pd.Timestamp("2026-01-10 17:30:00"),
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "referee": "Michael Oliver",
        "FTHG": 2,
        "FTAG": 1,
        "FTR": "H",
        "HTHG": 1,
        "HTAG": 0,
        "HTR": "H",
        "B365H": 1.80,
        "B365D": 3.60,
        "B365A": 4.20,
        "B365CH": 1.75,
        "B365CD": 3.70,
        "B365CA": 4.50,
        "PSH": 1.85,
        "PSD": 3.55,
        "PSA": 4.10,
        "PSCH": 1.78,
        "PSCD": 3.65,
        "PSCA": 4.40,
    }
    base.update(overrides)
    return base


# -- _to_fixtures -------------------------------------------------------------


def test_to_fixtures_maps_final_score(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_game_row()])

    fixtures = source._to_fixtures(raw, competition_id="eng-premier-league")

    assert len(fixtures) == 1
    row = fixtures.iloc[0]
    assert row["status"] == "finished"
    assert row["home_score"] == 2
    assert row["away_score"] == 1
    assert row["home_score_ht"] == 1
    assert row["away_score_ht"] == 0
    assert row["kickoff_utc"] == datetime(2026, 1, 10, 17, 30, tzinfo=UTC)
    assert bool(row["kickoff_is_estimated"]) is True


def test_to_fixtures_row_without_date_is_dropped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_game_row(date=None)])

    fixtures = source._to_fixtures(raw, competition_id="eng-premier-league")

    assert fixtures.empty


def test_to_fixtures_unplayed_match_is_scheduled(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_game_row(FTHG=None, FTAG=None)])

    fixtures = source._to_fixtures(raw, competition_id="eng-premier-league")

    assert fixtures.iloc[0]["status"] == "scheduled"


def test_to_fixtures_id_matches_fbref_style_hash(tmp_path: Path) -> None:
    """Same real match, same competition/season/teams/kickoff: this source
    and FBref must compute the same fixture id so their rows join."""
    from deportivas.domain.ids import fixture_id

    source = _source(tmp_path)
    raw = pd.DataFrame([_game_row()])

    fixtures = source._to_fixtures(raw, competition_id="eng-premier-league")

    expected = fixture_id(
        "eng-premier-league",
        "2526",
        "football:arsenal",
        "football:chelsea",
        datetime(2026, 1, 10, 17, 30, tzinfo=UTC),
    )
    assert fixtures.iloc[0]["id"] == expected


# -- _to_odds -------------------------------------------------------------------


def test_to_odds_maps_pinnacle_opening_and_closing(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_game_row()])

    odds = source._to_odds(raw, competition_id="eng-premier-league")

    pinnacle = odds[odds["bookmaker"] == "pinnacle"]
    assert set(pinnacle["is_closing"]) == {True, False}
    opening_home = pinnacle[(~pinnacle["is_closing"]) & (pinnacle["selection"] == "home")]
    assert opening_home.iloc[0]["price"] == pytest.approx(1.85)
    closing_home = pinnacle[(pinnacle["is_closing"]) & (pinnacle["selection"] == "home")]
    assert closing_home.iloc[0]["price"] == pytest.approx(1.78)


def test_to_odds_closing_captured_at_is_kickoff(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_game_row()])

    odds = source._to_odds(raw, competition_id="eng-premier-league")

    closing = odds[odds["is_closing"]]
    assert (closing["captured_at"] == datetime(2026, 1, 10, 17, 30, tzinfo=UTC)).all()


def test_to_odds_opening_captured_at_is_one_day_before_kickoff(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_game_row()])

    odds = source._to_odds(raw, competition_id="eng-premier-league")

    opening = odds[~odds["is_closing"]]
    assert (opening["captured_at"] == datetime(2026, 1, 9, 17, 30, tzinfo=UTC)).all()


def test_to_odds_missing_bookmaker_columns_are_skipped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    row = _game_row()
    for key in list(row):
        if (
            key.startswith("WH")
            or key.startswith("BW")
            or key.startswith("IW")
            or key.startswith("VC")
        ):
            del row[key]
    raw = pd.DataFrame([row])

    odds = source._to_odds(raw, competition_id="eng-premier-league")

    assert set(odds["bookmaker"]) == {"pinnacle", "bet365"}


def test_to_odds_all_rows_have_unique_ids(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_game_row()])

    odds = source._to_odds(raw, competition_id="eng-premier-league")

    assert odds["id"].is_unique
    assert (odds["id"].str.len() <= 64).all()


def test_to_odds_row_without_date_is_dropped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_game_row(date=None)])

    odds = source._to_odds(raw, competition_id="eng-premier-league")

    assert odds.empty


# -- fetch_games / fetch_1x2_odds (wiring) -------------------------------------


class _FakeMatchHistoryReader:
    last_instance: _FakeMatchHistoryReader | None = None

    def __init__(self, *, leagues: list[str], seasons: list[str], data_dir: Path) -> None:
        self.leagues = leagues
        self.seasons = seasons
        self.data_dir = data_dir
        type(self).last_instance = self

    def read_games(self) -> pd.DataFrame:
        df = pd.DataFrame([_game_row()])
        return df.set_index(["league", "season", "game"])


def test_fetch_games_waits_archives_and_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deportivas.ingest.sources.footballdata.sd.MatchHistory", _FakeMatchHistoryReader
    )
    source = _source(tmp_path)
    waits: list[float] = []

    def fake_wait() -> float:
        waits.append(1.0)
        return 0.0

    monkeypatch.setattr(source, "_wait", fake_wait)

    fixtures = source.fetch_games(
        competition_id="eng-premier-league", match_history_league="E0", seasons=["2526"]
    )

    assert len(waits) == 1
    assert len(fixtures) == 1
    assert _FakeMatchHistoryReader.last_instance is not None
    assert _FakeMatchHistoryReader.last_instance.leagues == ["E0"]


def test_fetch_1x2_odds_waits_archives_and_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deportivas.ingest.sources.footballdata.sd.MatchHistory", _FakeMatchHistoryReader
    )
    source = _source(tmp_path)
    waits: list[float] = []

    def fake_wait() -> float:
        waits.append(1.0)
        return 0.0

    monkeypatch.setattr(source, "_wait", fake_wait)

    odds = source.fetch_1x2_odds(
        competition_id="eng-premier-league", match_history_league="E0", seasons=["2526"]
    )

    assert len(waits) == 1
    assert not odds.empty
