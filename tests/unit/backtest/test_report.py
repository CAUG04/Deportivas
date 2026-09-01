"""report.py: aggregates results into CLV (the project's primary metric)
and ROI, by tier and by market, compared against the baselines."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from deportivas.backtest.report import (
    MetricSummary,
    _settled_signals,
    _summarize,
    _summarize_by,
    build_backtest_report,
)
from deportivas.config.catalog import BacktestConfig
from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FIXTURES, ODDS_SNAPSHOTS, RESULTS, SIGNALS
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository

_COMPETITION = "test-comp"
_KICKOFF = datetime(2024, 1, 10, tzinfo=UTC)
_ENTRY_AT = _KICKOFF - timedelta(days=1)

_id_counter = itertools.count()


@pytest.fixture(autouse=True)
def _settings_in_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DEPORTIVAS_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _config(**overrides: object) -> BacktestConfig:
    base: dict[str, object] = {
        "bootstrap_iterations": 200,
        "confidence_level": 0.95,
        "min_matches_per_window": 2,
        "baselines": ("always_favourite", "random"),
    }
    base.update(overrides)
    return BacktestConfig.model_validate(base)


def _results_df(clv_values: list[float | None], pnl_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"clv": clv_values, "pnl": pnl_values, "stake_fraction": [0.02] * len(pnl_values)}
    )


# --- _summarize ----------------------------------------------------------


def test_summarize_empty_dataframe() -> None:
    summary = _summarize(
        pd.DataFrame(columns=["clv", "pnl", "stake_fraction"]),
        stake_col="stake_fraction",
        config=_config(),
    )
    assert summary == MetricSummary(n=0, mean_clv=None, clv_ci=None, mean_pnl=0.0, roi=None)


def test_summarize_computes_mean_clv_and_roi_weighted_by_stake() -> None:
    df = _results_df([0.1, 0.2, -0.05], [0.02, -0.02, 0.01])
    summary = _summarize(df, stake_col="stake_fraction", config=_config())
    assert summary.n == 3
    assert summary.mean_clv == pytest.approx((0.1 + 0.2 - 0.05) / 3)
    assert summary.roi == pytest.approx((0.02 - 0.02 + 0.01) / (0.02 * 3))


def test_summarize_ignores_none_clv_values() -> None:
    df = _results_df([0.1, None, 0.3], [0.01, 0.0, 0.02])
    summary = _summarize(df, stake_col="stake_fraction", config=_config())
    assert summary.mean_clv == pytest.approx(0.2)


def test_summarize_withholds_ci_below_min_matches_per_window() -> None:
    df = _results_df([0.1, 0.2, 0.3], [0.01, 0.02, 0.03])
    summary = _summarize(df, stake_col="stake_fraction", config=_config(min_matches_per_window=10))
    assert summary.mean_clv is not None
    assert summary.clv_ci is None


def test_summarize_computes_ci_when_enough_samples() -> None:
    df = _results_df([0.1, 0.2, 0.3, -0.1], [0.01, 0.02, 0.03, -0.01])
    summary = _summarize(df, stake_col="stake_fraction", config=_config(min_matches_per_window=2))
    assert summary.clv_ci is not None
    assert summary.clv_ci[0] <= summary.clv_ci[1]


def test_summarize_flat_stake_roi_equals_mean_pnl() -> None:
    df = _results_df([0.1, 0.2], [1.0, -1.0])
    summary = _summarize(df, stake_col=None, config=_config())
    assert summary.roi == pytest.approx(summary.mean_pnl)


def test_summarize_roi_is_none_when_total_stake_is_zero() -> None:
    df = pd.DataFrame({"clv": [0.1], "pnl": [0.0], "stake_fraction": [0.0]})
    summary = _summarize(df, stake_col="stake_fraction", config=_config())
    assert summary.roi is None


# --- _summarize_by ---------------------------------------------------------


def test_summarize_by_empty_dataframe_returns_empty_dict() -> None:
    assert (
        _summarize_by(
            pd.DataFrame(columns=["tier", "clv", "pnl", "stake_fraction"]),
            "tier",
            stake_col="stake_fraction",
            config=_config(),
        )
        == {}
    )


def test_summarize_by_groups_correctly() -> None:
    df = pd.DataFrame(
        {
            "tier": ["alta", "alta", "baja"],
            "clv": [0.1, 0.2, -0.1],
            "pnl": [0.02, 0.02, 0.0],
            "stake_fraction": [0.02, 0.02, 0.0],
        }
    )
    grouped = _summarize_by(df, "tier", stake_col="stake_fraction", config=_config())
    assert set(grouped) == {"alta", "baja"}
    assert grouped["alta"].n == 2
    assert grouped["baja"].n == 1


# --- _settled_signals -------------------------------------------------------


def test_settled_signals_empty_without_results() -> None:
    assert _settled_signals(_COMPETITION).empty


def _fixture_row(fixture_id: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": fixture_id,
        "competition_id": _COMPETITION,
        "season": "s1",
        "kickoff_utc": _KICKOFF,
        "home_team_id": "home",
        "away_team_id": "away",
        "status": "finished",
        "stage": None,
        "matchday": None,
        "home_score": 2,
        "away_score": 1,
        "home_score_ht": None,
        "away_score_ht": None,
        "kickoff_is_estimated": False,
        "source": "test",
        "ingested_at": _ENTRY_AT,
    }
    base.update(overrides)
    return base


def _signal_row(signal_id: str, fixture_id: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": signal_id,
        "fixture_id": fixture_id,
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
        "tier_reasons": "{}",
        "stake_fraction": 0.02,
        "created_at": _ENTRY_AT,
        "source": "test",
        "ingested_at": _ENTRY_AT,
    }
    base.update(overrides)
    return base


def _result_row(
    result_id: str, signal_id: str, fixture_id: str, **overrides: object
) -> dict[str, object]:
    base: dict[str, object] = {
        "id": result_id,
        "signal_id": signal_id,
        "fixture_id": fixture_id,
        "competition_id": _COMPETITION,
        "season": "s1",
        "outcome": "win",
        "pnl": 0.016,
        "closing_price": 1.7,
        "closing_captured_at": _KICKOFF - timedelta(minutes=5),
        "clv": 1.8 / 1.7 - 1.0,
        "settled_at": _KICKOFF,
        "source": "test",
        "ingested_at": _KICKOFF,
    }
    base.update(overrides)
    return base


def test_settled_signals_joins_tier_market_and_stake_fraction() -> None:
    settings = get_settings()
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(
        pd.DataFrame([_signal_row("sig1", "fx1", tier="media", market="btts")])
    )
    ParquetTableRepository(RESULTS, settings.parquet_dir).write(
        pd.DataFrame([_result_row("res1", "sig1", "fx1")])
    )

    merged = _settled_signals(_COMPETITION)

    assert len(merged) == 1
    assert merged.iloc[0]["tier"] == "media"
    assert merged.iloc[0]["market"] == "btts"
    assert merged.iloc[0]["stake_fraction"] == pytest.approx(0.02)


# --- build_backtest_report (integracion) ------------------------------


def test_build_backtest_report_empty_competition() -> None:
    report = build_backtest_report(_COMPETITION)
    assert report.overall.n == 0
    assert report.by_tier == {}
    assert report.by_market == {}
    assert set(report.baselines) == {"always_favourite", "random"}


def test_build_backtest_report_breaks_down_by_tier_and_market_and_includes_baselines() -> None:
    settings = get_settings()
    fixtures = [
        _fixture_row("fx1"),
        _fixture_row("fx2", home_team_id="home2", away_team_id="away2"),
    ]
    signals = [
        _signal_row("sig1", "fx1", tier="alta", market="1x2", selection="home"),
        _signal_row(
            "sig2", "fx2", tier="descartar", market="btts", selection="yes", stake_fraction=0.0
        ),
    ]
    results = [
        _result_row("res1", "sig1", "fx1", pnl=0.016, clv=0.05),
        _result_row("res2", "sig2", "fx2", pnl=0.0, clv=-0.02),
    ]
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame(fixtures))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(pd.DataFrame(signals))
    ParquetTableRepository(RESULTS, settings.parquet_dir).write(pd.DataFrame(results))
    ParquetTableRepository(ODDS_SNAPSHOTS, settings.parquet_dir).write(
        pd.DataFrame(
            [
                {
                    "id": f"odds{next(_id_counter)}",
                    "fixture_id": fixture_id,
                    "competition_id": _COMPETITION,
                    "season": "s1",
                    "bookmaker": "pinnacle",
                    "market": market,
                    "selection": selection,
                    "line": None,
                    "price": price,
                    "captured_at": _ENTRY_AT,
                    "is_closing": False,
                    "source": "test",
                    "ingested_at": _ENTRY_AT,
                }
                for fixture_id, market, selections in [
                    ("fx1", "1x2", [("home", 1.8), ("draw", 3.5), ("away", 4.5)]),
                    ("fx2", "btts", [("yes", 1.9), ("no", 1.95)]),
                ]
                for selection, price in selections
            ]
        )
    )

    report = build_backtest_report(_COMPETITION)

    assert report.overall.n == 2
    assert set(report.by_tier) == {"alta", "descartar"}
    assert set(report.by_market) == {"1x2", "btts"}
    assert report.by_tier["alta"].n == 1
    assert report.by_tier["descartar"].mean_pnl == 0.0
    assert set(report.baselines) == {"always_favourite", "random"}
    assert report.baselines["always_favourite"].n == 2
