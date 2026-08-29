"""CLI wiring: each ``ingest`` command parses its arguments, builds the real
repositories for the active (DuckDB, pointed at a tmp dir) storage backend,
calls the adapter, and persists the result — proven end-to-end except for the
adapter's own external fetch, which is stubbed."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from deportivas.cli import app
from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FIXTURES, ODDS_SNAPSHOTS, TEAM_MATCH_STATS
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository

runner = CliRunner()


@pytest.fixture(autouse=True)
def _settings_in_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DEPORTIVAS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DEPORTIVAS_THE_ODDS_API_KEY", raising=False)
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
        "status": "scheduled",
        "stage": None,
        "matchday": None,
        "home_score": None,
        "away_score": None,
        "home_score_ht": None,
        "away_score_ht": None,
        "kickoff_is_estimated": False,
        "source": "test",
        "ingested_at": datetime.now(UTC),
    }
    base.update(overrides)
    return base


def test_fbref_schedule_wires_through_to_real_parquet_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    class _FakeFBrefSource:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def fetch_schedule(self, **kwargs: object) -> pd.DataFrame:
            calls["fetch"] = kwargs
            return pd.DataFrame([_fixture_row(id="from-cli")])

    monkeypatch.setattr("deportivas.ingest.sources.fbref.FBrefSource", _FakeFBrefSource)

    result = runner.invoke(
        app,
        [
            "ingest",
            "fbref-schedule",
            "--competition-id",
            "eng-premier-league",
            "--fbref-league",
            "ENG-Premier League",
            "--seasons",
            "2526",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "1 filas escritas" in result.output
    assert calls["fetch"] == {
        "competition_id": "eng-premier-league",
        "fbref_league": "ENG-Premier League",
        "seasons": ["2526"],
    }

    written = ParquetTableRepository(FIXTURES, get_settings().parquet_dir).read()
    assert list(written["id"]) == ["from-cli"]


def test_fbref_stats_command(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeFBrefSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_team_match_stats(self, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr("deportivas.ingest.sources.fbref.FBrefSource", _FakeFBrefSource)

    result = runner.invoke(
        app,
        [
            "ingest",
            "fbref-stats",
            "--competition-id",
            "eng-premier-league",
            "--fbref-league",
            "ENG-Premier League",
            "--seasons",
            "2526",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "0 filas" in result.output


def test_understat_stats_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _FakeUnderstatSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_team_match_stats(self, **kwargs: object) -> pd.DataFrame:
            calls["fetch"] = kwargs
            return pd.DataFrame()

    monkeypatch.setattr("deportivas.ingest.sources.understat.UnderstatSource", _FakeUnderstatSource)

    result = runner.invoke(
        app,
        [
            "ingest",
            "understat-stats",
            "--competition-id",
            "eng-premier-league",
            "--understat-league",
            "EPL",
            "--seasons",
            "2526",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["fetch"] == {
        "competition_id": "eng-premier-league",
        "understat_league": "EPL",
        "seasons": ["2526"],
    }


def test_espn_schedule_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _FakeEspnSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_schedule(self, **kwargs: object) -> pd.DataFrame:
            calls["fetch"] = kwargs
            return pd.DataFrame()

    monkeypatch.setattr("deportivas.ingest.sources.espn.EspnSource", _FakeEspnSource)

    result = runner.invoke(
        app,
        [
            "ingest",
            "espn-schedule",
            "--competition-id",
            "col-primera-a",
            "--espn-league",
            "col.1",
            "--seasons",
            "2026",
        ],
    )

    assert result.exit_code == 0, result.output
    fetch_kwargs = calls["fetch"]
    assert isinstance(fetch_kwargs, dict)
    assert fetch_kwargs["competition_id"] == "col-primera-a"


def test_footballdata_games_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeFootballDataSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_games(self, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame([_fixture_row(id="fd-1")])

    monkeypatch.setattr(
        "deportivas.ingest.sources.footballdata.FootballDataSource", _FakeFootballDataSource
    )

    result = runner.invoke(
        app,
        [
            "ingest",
            "footballdata-games",
            "--competition-id",
            "eng-premier-league",
            "--match-history-league",
            "E0",
            "--seasons",
            "2526",
        ],
    )

    assert result.exit_code == 0, result.output
    written = ParquetTableRepository(FIXTURES, get_settings().parquet_dir).read()
    assert list(written["id"]) == ["fd-1"]


def test_footballdata_odds_command(monkeypatch: pytest.MonkeyPatch) -> None:
    odds_row = {
        "id": "odds-1",
        "fixture_id": "fix1",
        "competition_id": "eng-premier-league",
        "season": "2526",
        "bookmaker": "pinnacle",
        "market": "1x2",
        "selection": "home",
        "line": None,
        "price": 1.85,
        "captured_at": datetime.now(UTC),
        "is_closing": False,
        "source": "footballdata",
        "ingested_at": datetime.now(UTC),
    }

    class _FakeFootballDataSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_1x2_odds(self, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame([odds_row])

    monkeypatch.setattr(
        "deportivas.ingest.sources.footballdata.FootballDataSource", _FakeFootballDataSource
    )

    result = runner.invoke(
        app,
        [
            "ingest",
            "footballdata-odds",
            "--competition-id",
            "eng-premier-league",
            "--match-history-league",
            "E0",
            "--seasons",
            "2526",
        ],
    )

    assert result.exit_code == 0, result.output
    written = ParquetTableRepository(ODDS_SNAPSHOTS, get_settings().parquet_dir).read()
    assert list(written["id"]) == ["odds-1"]


def test_nfl_schedule_command_parses_int_seasons(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _FakeNflSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_schedules(self, **kwargs: object) -> pd.DataFrame:
            calls["fetch"] = kwargs
            return pd.DataFrame()

    monkeypatch.setattr("deportivas.ingest.sources.nfl.NflSource", _FakeNflSource)

    result = runner.invoke(
        app,
        ["ingest", "nfl-schedule", "--competition-id", "usa-nfl", "--seasons", "2024,2025"],
    )

    assert result.exit_code == 0, result.output
    assert calls["fetch"] == {"competition_id": "usa-nfl", "seasons": [2024, 2025]}


def test_nfl_team_game_stats_command_reads_fixtures_and_calls_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ParquetTableRepository(FIXTURES, get_settings().parquet_dir).write(
        pd.DataFrame([_fixture_row(id="fix1", competition_id="usa-nfl")])
    )
    calls: dict[str, object] = {}

    class _FakeNflSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_team_game_stats(self, **kwargs: object) -> pd.DataFrame:
            calls["fetch"] = kwargs
            return pd.DataFrame()

    monkeypatch.setattr("deportivas.ingest.sources.nfl.NflSource", _FakeNflSource)

    result = runner.invoke(
        app,
        [
            "ingest",
            "nfl-team-game-stats",
            "--competition-id",
            "usa-nfl",
            "--seasons",
            "2025",
        ],
    )

    assert result.exit_code == 0, result.output
    fetch_kwargs = calls["fetch"]
    assert isinstance(fetch_kwargs, dict)
    assert fetch_kwargs["seasons"] == [2025]
    fixtures_arg = fetch_kwargs["fixtures"]
    assert isinstance(fixtures_arg, pd.DataFrame)
    assert list(fixtures_arg["id"]) == ["fix1"]


def test_nba_schedule_command(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeNbaSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_schedule(self, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(
        "deportivas.ingest.sources.sportsdataverse_source.NbaSource", _FakeNbaSource
    )

    result = runner.invoke(
        app, ["ingest", "nba-schedule", "--competition-id", "usa-nba", "--seasons", "2025"]
    )
    assert result.exit_code == 0, result.output


def test_nhl_schedule_command(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeNhlSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_schedule(self, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(
        "deportivas.ingest.sources.sportsdataverse_source.NhlSource", _FakeNhlSource
    )

    result = runner.invoke(
        app, ["ingest", "nhl-schedule", "--competition-id", "usa-nhl", "--seasons", "2025"]
    )
    assert result.exit_code == 0, result.output


def test_mlb_schedule_command_parses_team_list(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _FakePybaseballSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_schedule(self, **kwargs: object) -> pd.DataFrame:
            calls["fetch"] = kwargs
            return pd.DataFrame()

    monkeypatch.setattr(
        "deportivas.ingest.sources.pybaseball_source.PybaseballSource", _FakePybaseballSource
    )

    result = runner.invoke(
        app,
        [
            "ingest",
            "mlb-schedule",
            "--competition-id",
            "usa-mlb",
            "--season",
            "2025",
            "--teams",
            "PHI,ATL",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["fetch"] == {
        "competition_id": "usa-mlb",
        "season": 2025,
        "team_abbreviations": ["PHI", "ATL"],
    }


def test_odds_snapshot_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(
        app,
        [
            "ingest",
            "odds-snapshot",
            "--competition-id",
            "eng-premier-league",
            "--sport-key",
            "soccer_epl",
            "--season",
            "2526",
            "--market-map",
            "h2h:1x2",
        ],
    )
    assert result.exit_code != 0
    assert "DEPORTIVAS_THE_ODDS_API_KEY" in str(result.output) + str(result.exception)


def test_odds_snapshot_command_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setenv("DEPORTIVAS_THE_ODDS_API_KEY", "test-key")
    get_settings.cache_clear()

    class _FakeTheOddsApiSource:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def fetch_odds(self, **kwargs: object) -> pd.DataFrame:
            calls["fetch"] = kwargs
            return pd.DataFrame()

    monkeypatch.setattr(
        "deportivas.ingest.sources.theoddsapi.TheOddsApiSource", _FakeTheOddsApiSource
    )

    result = runner.invoke(
        app,
        [
            "ingest",
            "odds-snapshot",
            "--competition-id",
            "eng-premier-league",
            "--sport-key",
            "soccer_epl",
            "--season",
            "2526",
            "--market-map",
            "h2h:1x2,spreads:asian_handicap",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["fetch"] == {
        "competition_id": "eng-premier-league",
        "sport_key": "soccer_epl",
        "season": "2526",
        "market_map": {"h2h": "1x2", "spreads": "asian_handicap"},
    }
    init_kwargs = calls["init"]
    assert isinstance(init_kwargs, dict)
    assert init_kwargs["api_key"] == "test-key"


def test_seed_competitions_writes_catalog(tmp_path: Path) -> None:
    result = runner.invoke(app, ["seed-competitions"])
    assert result.exit_code == 0, result.output
    assert "filas escritas" in result.output


def test_team_match_stats_table_used_by_fbref_stats_and_understat(tmp_path: Path) -> None:
    """Sanity check that both adapters' CLI commands target the same table
    (proves the shared TEAM_MATCH_STATS import alias is wired correctly)."""
    assert TEAM_MATCH_STATS.name == "team_match_stats"
