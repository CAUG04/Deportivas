"""CLI wiring: each ``ingest`` command parses its arguments, builds the real
repositories for the active (DuckDB, pointed at a tmp dir) storage backend,
calls the adapter, and persists the result — proven end-to-end except for the
adapter's own external fetch, which is stubbed."""

from __future__ import annotations

import json
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
    # Los comandos fbref-*/understat-*/espn-*/footballdata-* llaman a
    # ensure_custom_league_dict() antes de importar su adaptador (ver
    # ingest/soccerdata_config.py); sin esto escribirian de verdad en el
    # ~/soccerdata real de la maquina que corre los tests.
    monkeypatch.setenv("SOCCERDATA_DIR", str(tmp_path / "soccerdata"))
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
        "regions": "uk,eu,us",
    }
    init_kwargs = calls["init"]
    assert isinstance(init_kwargs, dict)
    assert init_kwargs["api_key"] == "test-key"


def test_odds_snapshot_command_custom_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setenv("DEPORTIVAS_THE_ODDS_API_KEY", "test-key")
    get_settings.cache_clear()

    class _FakeTheOddsApiSource:
        def __init__(self, **kwargs: object) -> None:
            pass

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
            "h2h:1x2",
            "--regions",
            "eu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["fetch"]["regions"] == "eu"  # type: ignore[index]


def test_seed_competitions_writes_catalog(tmp_path: Path) -> None:
    result = runner.invoke(app, ["seed-competitions"])
    assert result.exit_code == 0, result.output
    assert "filas escritas" in result.output


def test_team_match_stats_table_used_by_fbref_stats_and_understat(tmp_path: Path) -> None:
    """Sanity check that both adapters' CLI commands target the same table
    (proves the shared TEAM_MATCH_STATS import alias is wired correctly)."""
    assert TEAM_MATCH_STATS.name == "team_match_stats"


def test_compute_football_features_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str) -> int:
        calls["competition_id"] = competition_id
        return 3

    monkeypatch.setattr(
        "deportivas.features.football.pipeline.compute_and_write_football_features", _fake
    )

    result = runner.invoke(
        app, ["features", "compute-football", "--competition-id", "eng-premier-league"]
    )

    assert result.exit_code == 0, result.output
    assert calls["competition_id"] == "eng-premier-league"
    assert "football_v1: 3 filas escritas" in result.output


def test_compute_nfl_features_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str) -> int:
        calls["competition_id"] = competition_id
        return 5

    monkeypatch.setattr("deportivas.features.nfl.pipeline.compute_and_write_nfl_features", _fake)

    result = runner.invoke(app, ["features", "compute-nfl", "--competition-id", "usa-nfl"])

    assert result.exit_code == 0, result.output
    assert calls["competition_id"] == "usa-nfl"
    assert "nfl_v1: 5 filas escritas" in result.output


def test_compute_nba_features_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str) -> int:
        calls["competition_id"] = competition_id
        return 7

    monkeypatch.setattr("deportivas.features.nba.pipeline.compute_and_write_nba_features", _fake)

    result = runner.invoke(app, ["features", "compute-nba", "--competition-id", "usa-nba"])

    assert result.exit_code == 0, result.output
    assert calls["competition_id"] == "usa-nba"
    assert "nba_v1: 7 filas escritas" in result.output


def test_compute_nhl_features_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str) -> int:
        calls["competition_id"] = competition_id
        return 9

    monkeypatch.setattr("deportivas.features.nhl.pipeline.compute_and_write_nhl_features", _fake)

    result = runner.invoke(app, ["features", "compute-nhl", "--competition-id", "usa-nhl"])

    assert result.exit_code == 0, result.output
    assert calls["competition_id"] == "usa-nhl"
    assert "nhl_v1: 9 filas escritas" in result.output


def test_compute_mlb_features_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str) -> int:
        calls["competition_id"] = competition_id
        return 11

    monkeypatch.setattr("deportivas.features.mlb.pipeline.compute_and_write_mlb_features", _fake)

    result = runner.invoke(app, ["features", "compute-mlb", "--competition-id", "usa-mlb"])

    assert result.exit_code == 0, result.output
    assert calls["competition_id"] == "usa-mlb"
    assert "mlb_v1: 11 filas escritas" in result.output


def test_train_football_model_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str, *, calibration_method: str | None = None) -> list[int]:
        calls["competition_id"] = competition_id
        calls["calibration_method"] = calibration_method
        return [180, 190]

    monkeypatch.setattr("deportivas.models.football.train.compute_and_write_football_models", _fake)

    result = runner.invoke(
        app,
        [
            "models",
            "train-football",
            "--competition-id",
            "eng-premier-league",
            "--calibration-method",
            "platt",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["competition_id"] == "eng-premier-league"
    assert calls["calibration_method"] == "platt"
    assert "football poisson: 2 ventana(s), 370 filas de predicciones" in result.output


def test_train_football_model_command_defaults_calibration_method_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str, *, calibration_method: str | None = None) -> list[int]:
        calls["calibration_method"] = calibration_method
        return []

    monkeypatch.setattr("deportivas.models.football.train.compute_and_write_football_models", _fake)

    result = runner.invoke(
        app, ["models", "train-football", "--competition-id", "eng-premier-league"]
    )

    assert result.exit_code == 0, result.output
    assert calls["calibration_method"] is None
    assert "0 ventanas entrenadas" in result.output


@pytest.mark.parametrize(
    ("command", "module", "function", "competition_id", "label"),
    [
        (
            "train-nfl",
            "deportivas.models.nfl.train",
            "compute_and_write_nfl_moneyline_model",
            "usa-nfl",
            "nfl moneyline",
        ),
        (
            "train-nba",
            "deportivas.models.nba.train",
            "compute_and_write_nba_moneyline_model",
            "usa-nba",
            "nba moneyline",
        ),
        (
            "train-nhl",
            "deportivas.models.nhl.train",
            "compute_and_write_nhl_moneyline_model",
            "usa-nhl",
            "nhl moneyline",
        ),
        (
            "train-mlb",
            "deportivas.models.mlb.train",
            "compute_and_write_mlb_moneyline_model",
            "usa-mlb",
            "mlb moneyline",
        ),
    ],
)
def test_train_moneyline_model_command(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    module: str,
    function: str,
    competition_id: str,
    label: str,
) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str, *, calibration_method: str | None = None) -> list[int]:
        calls["competition_id"] = competition_id
        calls["calibration_method"] = calibration_method
        return [180, 190]

    monkeypatch.setattr(f"{module}.{function}", _fake)

    result = runner.invoke(
        app,
        [
            "models",
            command,
            "--competition-id",
            competition_id,
            "--calibration-method",
            "platt",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["competition_id"] == competition_id
    assert calls["calibration_method"] == "platt"
    assert f"{label}: 2 ventana(s), 370 filas de predicciones" in result.output


def test_train_nfl_model_command_defaults_calibration_method_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str, *, calibration_method: str | None = None) -> list[int]:
        calls["calibration_method"] = calibration_method
        return []

    monkeypatch.setattr("deportivas.models.nfl.train.compute_and_write_nfl_moneyline_model", _fake)

    result = runner.invoke(app, ["models", "train-nfl", "--competition-id", "usa-nfl"])

    assert result.exit_code == 0, result.output
    assert calls["calibration_method"] is None
    assert "0 ventanas entrenadas" in result.output


def test_generate_signals_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str) -> int:
        calls["competition_id"] = competition_id
        return 42

    monkeypatch.setattr("deportivas.signals.generate.compute_and_write_signals", _fake)

    result = runner.invoke(app, ["signals", "generate", "--competition-id", "eng-premier-league"])

    assert result.exit_code == 0, result.output
    assert calls["competition_id"] == "eng-premier-league"
    assert "signals: 42 fila(s) escritas" in result.output


def test_settle_backtest_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str) -> int:
        calls["competition_id"] = competition_id
        return 7

    monkeypatch.setattr("deportivas.backtest.settlement.compute_and_write_results", _fake)

    result = runner.invoke(app, ["backtest", "settle", "--competition-id", "eng-premier-league"])

    assert result.exit_code == 0, result.output
    assert calls["competition_id"] == "eng-premier-league"
    assert "results: 7 fila(s) escritas" in result.output


def test_backtest_report_command(monkeypatch: pytest.MonkeyPatch) -> None:
    from deportivas.backtest.report import BacktestReport, MetricSummary

    fake_report = BacktestReport(
        overall=MetricSummary(n=10, mean_clv=0.032, clv_ci=(0.01, 0.05), mean_pnl=0.004, roi=0.021),
        by_tier={"alta": MetricSummary(n=4, mean_clv=0.05, clv_ci=None, mean_pnl=0.01, roi=0.05)},
        by_market={
            "1x2": MetricSummary(n=10, mean_clv=0.032, clv_ci=None, mean_pnl=0.004, roi=0.021)
        },
        baselines={
            "always_favourite": MetricSummary(
                n=10, mean_clv=-0.01, clv_ci=None, mean_pnl=-0.02, roi=-0.02
            ),
            "random": MetricSummary(n=0, mean_clv=None, clv_ci=None, mean_pnl=0.0, roi=None),
        },
    )

    def _fake(competition_id: str) -> BacktestReport:
        return fake_report

    monkeypatch.setattr("deportivas.backtest.report.build_backtest_report", _fake)

    result = runner.invoke(app, ["backtest", "report", "--competition-id", "eng-premier-league"])

    assert result.exit_code == 0, result.output
    assert "-- global --" in result.output
    assert "n=10" in result.output
    assert "CLV medio=+3.20%" in result.output
    assert "IC [+1.00%, +5.00%]" in result.output
    assert "-- por tier --" in result.output
    assert "alta:" in result.output
    assert "-- por mercado --" in result.output
    assert "1x2:" in result.output
    assert "-- baselines" in result.output
    assert "always_favourite:" in result.output
    assert "random: sin datos liquidados" in result.output


def test_backtest_report_command_omits_empty_breakdowns(monkeypatch: pytest.MonkeyPatch) -> None:
    from deportivas.backtest.report import BacktestReport, MetricSummary

    empty_report = BacktestReport(
        overall=MetricSummary(n=0, mean_clv=None, clv_ci=None, mean_pnl=0.0, roi=None),
        by_tier={},
        by_market={},
        baselines={},
    )

    def _fake(competition_id: str) -> BacktestReport:
        return empty_report

    monkeypatch.setattr("deportivas.backtest.report.build_backtest_report", _fake)

    result = runner.invoke(app, ["backtest", "report", "--competition-id", "eng-premier-league"])

    assert result.exit_code == 0, result.output
    assert "-- global --" in result.output
    assert "sin datos liquidados" in result.output
    assert "-- por tier --" not in result.output
    assert "-- por mercado --" not in result.output


def test_export_run_command_for_one_competition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str) -> dict[str, Path]:
        calls["competition_id"] = competition_id
        return {"signals": tmp_path / "signals.json", "backtest": tmp_path / "backtest.json"}

    monkeypatch.setattr("deportivas.export.json_export.export_competition", _fake)

    result = runner.invoke(app, ["export", "run", "--competition-id", "eng-premier-league"])

    assert result.exit_code == 0, result.output
    assert calls["competition_id"] == "eng-premier-league"
    assert "signals: " in result.output
    assert "backtest: " in result.output


def test_export_run_command_defaults_to_every_competition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_ids: list[str] | None = None) -> dict[str, Path]:
        calls["competition_ids"] = competition_ids
        return {"competitions": tmp_path / "competitions.json"}

    monkeypatch.setattr("deportivas.export.json_export.export_all", _fake)

    result = runner.invoke(app, ["export", "run"])

    assert result.exit_code == 0, result.output
    assert calls["competition_ids"] is None
    assert "competitions: " in result.output


def test_mark_closing_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str) -> int:
        calls["competition_id"] = competition_id
        return 5

    monkeypatch.setattr("deportivas.ingest.closing.mark_closing_lines", _fake)

    result = runner.invoke(
        app, ["ingest", "mark-closing", "--competition-id", "eng-premier-league"]
    )

    assert result.exit_code == 0, result.output
    assert calls["competition_id"] == "eng-premier-league"
    assert "odds_snapshots: 5 fila(s) marcadas is_closing" in result.output


def test_list_competitions_command() -> None:
    result = runner.invoke(app, ["list-competitions"])

    assert result.exit_code == 0, result.output
    competitions = json.loads(result.output)
    assert isinstance(competitions, list)
    assert all(c["enabled"] for c in competitions)
    ids = {c["id"] for c in competitions}
    assert "eng-premier-league" in ids
    assert "usa-nfl" in ids
    epl = next(c for c in competitions if c["id"] == "eng-premier-league")
    assert epl["sport"] == "football"
    assert epl["sources"]["fbref"] == "Premier League"
    assert epl["odds"]["the_odds_api"] == "soccer_epl"


def test_current_seasons_command() -> None:
    result = runner.invoke(
        app,
        [
            "current-seasons",
            "--competition-id",
            "eng-premier-league",
            "--count",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    # temporada de futbol, formato "YYZZ" de 4 digitos -- el valor exacto
    # depende de la fecha en que corra el test, el formato no.
    assert result.output.strip().isdigit()
    assert len(result.output.strip()) == 4


def test_current_seasons_command_defaults_to_two() -> None:
    result = runner.invoke(app, ["current-seasons", "--competition-id", "usa-nfl"])

    assert result.exit_code == 0, result.output
    assert len(result.output.strip().split(",")) == 2
