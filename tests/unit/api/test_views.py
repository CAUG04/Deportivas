"""views.py: framework-agnostic read layer shared by api/app.py and
export/json_export.py."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from deportivas.api.views import (
    BacktestReportOut,
    backtest_report,
    list_competitions,
    list_signals,
)
from deportivas.backtest.report import BacktestReport, MetricSummary
from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FIXTURES, SIGNALS, TEAMS
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository

_COMPETITION = "eng-premier-league"  # real, de config/competitions.yaml
_FIXTURE_ID = "fx1"
_KICKOFF = datetime(2024, 1, 10, tzinfo=UTC)
_ENTRY_AT = _KICKOFF - timedelta(days=1)


@pytest.fixture(autouse=True)
def _settings_in_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DEPORTIVAS_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fixture_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": _FIXTURE_ID,
        "competition_id": _COMPETITION,
        "season": "s1",
        "kickoff_utc": _KICKOFF,
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
        "ingested_at": _ENTRY_AT,
    }
    base.update(overrides)
    return base


def _team_row(team_id: str, name: str) -> dict[str, object]:
    return {
        "id": team_id,
        "canonical_name": name,
        "sport": "football",
        "country": "England",
        "source": "test",
        "ingested_at": _ENTRY_AT,
    }


def _signal_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "sig1",
        "fixture_id": _FIXTURE_ID,
        "competition_id": _COMPETITION,
        "season": "s1",
        "market": "1x2",
        "selection": "home",
        "line": None,
        "model_name": "m1",
        "model_version": "v1",
        "prob_model": 0.6,
        "prob_fair": 0.5,
        "fair_price": 2.0,
        "entry_price": 1.8,
        "entry_bookmaker": "pinnacle",
        "entry_captured_at": _ENTRY_AT,
        "devig_method": "multiplicative",
        "edge": 0.1,
        "tier": "alta",
        "tier_reasons": '{"min_edge": true}',
        "stake_fraction": 0.02,
        "created_at": _ENTRY_AT,
        "source": "test",
        "ingested_at": _ENTRY_AT,
    }
    base.update(overrides)
    return base


# --- list_competitions ------------------------------------------------


def test_list_competitions_includes_a_known_enabled_competition() -> None:
    competitions = list_competitions()
    assert any(c.id == _COMPETITION for c in competitions)


def test_list_competitions_excludes_disabled_ones() -> None:
    competitions = list_competitions()
    assert all(c.id for c in competitions)  # no vacios; enabled ya filtra en config.catalog


# --- list_signals --------------------------------------------------------


def test_list_signals_raises_for_unknown_competition() -> None:
    with pytest.raises(KeyError):
        list_signals("no-existe-esta-competicion")


def test_list_signals_empty_without_data() -> None:
    assert list_signals(_COMPETITION) == []


def test_list_signals_enriches_team_names_and_kickoff() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(TEAMS, settings.parquet_dir).write(
        pd.DataFrame(
            [
                _team_row("football:arsenal", "Arsenal"),
                _team_row("football:chelsea", "Chelsea"),
            ]
        )
    )
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(pd.DataFrame([_signal_row()]))

    signals = list_signals(_COMPETITION)

    assert len(signals) == 1
    assert signals[0].home_team == "Arsenal"
    assert signals[0].away_team == "Chelsea"
    assert signals[0].kickoff_utc == _KICKOFF
    assert signals[0].tier_reasons == {"min_edge": True}


def test_list_signals_falls_back_to_raw_id_when_team_unknown() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(pd.DataFrame([_signal_row()]))
    # sin escribir teams: no hay nombre canonico que resolver

    signals = list_signals(_COMPETITION)

    assert signals[0].home_team == "football:arsenal"
    assert signals[0].away_team == "football:chelsea"


def test_list_signals_skips_orphaned_fixture_reference() -> None:
    settings = get_settings()
    # sin fixtures en absoluto: la senal no puede resolverse
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(pd.DataFrame([_signal_row()]))
    assert list_signals(_COMPETITION) == []


def test_list_signals_only_actionable_excludes_baja_and_descartar_by_default() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(
        pd.DataFrame(
            [
                _signal_row(id="sig-alta", tier="alta"),
                _signal_row(id="sig-baja", tier="baja"),
                _signal_row(id="sig-descartar", tier="descartar"),
            ]
        )
    )

    signals = list_signals(_COMPETITION)

    assert {s.id for s in signals} == {"sig-alta"}


def test_list_signals_only_actionable_false_returns_every_tier() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(
        pd.DataFrame(
            [
                _signal_row(id="sig-alta", tier="alta"),
                _signal_row(id="sig-descartar", tier="descartar"),
            ]
        )
    )

    signals = list_signals(_COMPETITION, only_actionable=False)

    assert {s.id for s in signals} == {"sig-alta", "sig-descartar"}


def test_list_signals_explicit_tier_overrides_only_actionable() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(
        pd.DataFrame([_signal_row(id="sig-descartar", tier="descartar")])
    )

    signals = list_signals(_COMPETITION, tier="descartar")

    assert {s.id for s in signals} == {"sig-descartar"}


def test_list_signals_filters_by_market() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(
        pd.DataFrame(
            [
                _signal_row(id="sig-1x2", market="1x2"),
                _signal_row(id="sig-btts", market="btts", selection="yes"),
            ]
        )
    )

    signals = list_signals(_COMPETITION, market="btts")

    assert {s.id for s in signals} == {"sig-btts"}


def test_list_signals_respects_limit_newest_first() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(
        pd.DataFrame(
            [
                _signal_row(id="old", entry_captured_at=_ENTRY_AT - timedelta(days=1)),
                _signal_row(id="new", entry_captured_at=_ENTRY_AT),
            ]
        )
    )

    signals = list_signals(_COMPETITION, limit=1)

    assert len(signals) == 1
    assert signals[0].id == "new"


# --- backtest_report --------------------------------------------------


def test_backtest_report_raises_for_unknown_competition() -> None:
    with pytest.raises(KeyError):
        backtest_report("no-existe-esta-competicion")


def test_backtest_report_wraps_build_backtest_report(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_report = BacktestReport(
        overall=MetricSummary(n=5, mean_clv=0.02, clv_ci=(0.0, 0.04), mean_pnl=0.01, roi=0.03),
        by_tier={"alta": MetricSummary(n=5, mean_clv=0.02, clv_ci=None, mean_pnl=0.01, roi=0.03)},
        by_market={},
        baselines={
            "random": MetricSummary(n=0, mean_clv=None, clv_ci=None, mean_pnl=0.0, roi=None)
        },
    )
    monkeypatch.setattr(
        "deportivas.api.views.build_backtest_report", lambda competition_id: fake_report
    )

    report = backtest_report(_COMPETITION)

    assert isinstance(report, BacktestReportOut)
    assert report.overall.n == 5
    assert report.overall.clv_ci_low == 0.0
    assert report.overall.clv_ci_high == 0.04
    assert report.by_tier["alta"].n == 5
    assert report.by_market == {}
    assert report.baselines["random"].n == 0
    assert report.baselines["random"].clv_ci_low is None
