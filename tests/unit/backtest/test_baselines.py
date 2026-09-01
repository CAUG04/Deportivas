"""baselines.py: always_favourite / random shadow strategies over the exact
same entry snapshot a real signal used, settled with a flat stake."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from deportivas.backtest.baselines import (
    ALWAYS_FAVOURITE,
    RANDOM,
    _baseline_selection,
    compute_baseline_results,
)
from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FIXTURES, ODDS_SNAPSHOTS, SIGNALS
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository

_COMPETITION = "test-comp"
_FIXTURE_ID = "fx1"
_KICKOFF = datetime(2024, 1, 10, tzinfo=UTC)
_ENTRY_AT = _KICKOFF - timedelta(days=1)

_odds_id_counter = itertools.count()


@pytest.fixture(autouse=True)
def _settings_in_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DEPORTIVAS_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _odds_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": f"odds{next(_odds_id_counter)}",
        "fixture_id": _FIXTURE_ID,
        "competition_id": _COMPETITION,
        "season": "s1",
        "bookmaker": "pinnacle",
        "market": "1x2",
        "selection": "home",
        "line": None,
        "price": 1.8,
        "captured_at": _ENTRY_AT,
        "is_closing": False,
        "source": "test",
        "ingested_at": _ENTRY_AT,
    }
    base.update(overrides)
    return base


def _fixture_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": _FIXTURE_ID,
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
        "tier_reasons": "{}",
        "stake_fraction": 0.02,
        "created_at": _ENTRY_AT,
        "source": "test",
        "ingested_at": _ENTRY_AT,
    }
    base.update(overrides)
    return base


# --- _baseline_selection -----------------------------------------------


def test_always_favourite_picks_the_lowest_price() -> None:
    selection = _baseline_selection(
        {"home": 1.8, "draw": 3.5, "away": 4.5}, baseline=ALWAYS_FAVOURITE, seed="x"
    )
    assert selection == "home"


def test_random_is_deterministic_for_the_same_seed() -> None:
    prices = {"home": 1.8, "draw": 3.5, "away": 4.5}
    first = _baseline_selection(prices, baseline=RANDOM, seed="sig1")
    second = _baseline_selection(prices, baseline=RANDOM, seed="sig1")
    assert first == second
    assert first in prices


def test_unknown_baseline_raises() -> None:
    with pytest.raises(ValueError, match="baseline desconocida"):
        _baseline_selection({"home": 1.8, "away": 2.0}, baseline="martingale", seed="x")


# --- compute_baseline_results (integracion) --------------------------------


def test_compute_baseline_results_empty_without_signals() -> None:
    result = compute_baseline_results(_COMPETITION, ALWAYS_FAVOURITE)
    assert result.empty
    assert list(result.columns) == [
        "signal_id",
        "fixture_id",
        "market",
        "selection",
        "entry_price",
        "outcome",
        "pnl",
        "clv",
    ]


def test_compute_baseline_results_skips_unfinished_fixtures() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(
        pd.DataFrame([_fixture_row(status="scheduled", home_score=None, away_score=None)])
    )
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(pd.DataFrame([_signal_row()]))
    result = compute_baseline_results(_COMPETITION, ALWAYS_FAVOURITE)
    assert result.empty


def test_compute_baseline_results_skips_finished_fixtures_missing_a_score() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(
        pd.DataFrame([_fixture_row(home_score=None, away_score=None)])
    )
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(pd.DataFrame([_signal_row()]))
    result = compute_baseline_results(_COMPETITION, ALWAYS_FAVOURITE)
    assert result.empty


def test_compute_baseline_results_skips_signals_without_a_matching_entry_snapshot() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(pd.DataFrame([_signal_row()]))
    # sin odds_snapshots en absoluto: no hay como reconstruir el snapshot de entrada
    result = compute_baseline_results(_COMPETITION, ALWAYS_FAVOURITE)
    assert result.empty


def test_compute_baseline_results_always_favourite_settles_against_the_entry_snapshot() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(pd.DataFrame([_signal_row()]))
    ParquetTableRepository(ODDS_SNAPSHOTS, settings.parquet_dir).write(
        pd.DataFrame(
            [
                _odds_row(selection="home", price=1.8, captured_at=_ENTRY_AT),
                _odds_row(selection="draw", price=3.5, captured_at=_ENTRY_AT),
                _odds_row(selection="away", price=4.5, captured_at=_ENTRY_AT),
                _odds_row(
                    selection="home",
                    price=1.7,
                    captured_at=_KICKOFF - timedelta(minutes=5),
                    is_closing=True,
                ),
                _odds_row(
                    selection="draw",
                    price=3.6,
                    captured_at=_KICKOFF - timedelta(minutes=5),
                    is_closing=True,
                ),
                _odds_row(
                    selection="away",
                    price=5.0,
                    captured_at=_KICKOFF - timedelta(minutes=5),
                    is_closing=True,
                ),
            ]
        )
    )

    result = compute_baseline_results(_COMPETITION, ALWAYS_FAVOURITE)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["signal_id"] == "sig1"
    assert row["selection"] == "home"  # precio mas bajo de las tres
    assert row["entry_price"] == pytest.approx(1.8)
    assert row["outcome"] == "win"  # home 2-1
    assert row["pnl"] == pytest.approx(1.0 * (1.8 - 1.0))  # stake plano de 1 unidad
    assert row["clv"] == pytest.approx(1.8 / 1.7 - 1.0)


def test_compute_baseline_results_random_picks_one_of_the_priced_selections() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(pd.DataFrame([_signal_row()]))
    ParquetTableRepository(ODDS_SNAPSHOTS, settings.parquet_dir).write(
        pd.DataFrame(
            [
                _odds_row(selection="home", price=1.8, captured_at=_ENTRY_AT),
                _odds_row(selection="draw", price=3.5, captured_at=_ENTRY_AT),
                _odds_row(selection="away", price=4.5, captured_at=_ENTRY_AT),
            ]
        )
    )

    result = compute_baseline_results(_COMPETITION, RANDOM)

    assert len(result) == 1
    assert result.iloc[0]["selection"] in {"home", "draw", "away"}
