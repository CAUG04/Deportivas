"""settlement.py: grades signals against the fixture's final score and
writes results, with CLV as the metric that matters (rule of the project)."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from deportivas.backtest.settlement import (
    closing_price,
    compute_and_write_results,
    pnl_for_outcome,
    signal_outcome,
)
from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FIXTURES, ODDS_SNAPSHOTS, RESULTS, SIGNALS
from deportivas.domain.enums import BetOutcome
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository

_COMPETITION = "test-comp"
_FIXTURE_ID = "fx1"
_KICKOFF = datetime(2024, 1, 10, tzinfo=UTC)

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
        "market": "over_under",
        "selection": "over",
        "line": 2.5,
        "price": 1.9,
        "captured_at": _KICKOFF - timedelta(hours=1),
        "is_closing": False,
        "source": "test",
        "ingested_at": _KICKOFF,
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
        "ingested_at": _KICKOFF,
    }
    base.update(overrides)
    return base


def _signal_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "sig1",
        "fixture_id": _FIXTURE_ID,
        "competition_id": _COMPETITION,
        "season": "s1",
        "market": "over_under",
        "selection": "over",
        "line": 2.5,
        "model_name": "m1",
        "model_version": "v1",
        "prob_model": 0.6,
        "prob_fair": 0.5,
        "fair_price": 2.0,
        "entry_price": 2.0,
        "entry_bookmaker": "pinnacle",
        "entry_captured_at": _KICKOFF - timedelta(days=1),
        "devig_method": "multiplicative",
        "edge": 0.1,
        "tier": "alta",
        "tier_reasons": "{}",
        "stake_fraction": 0.02,
        "created_at": _KICKOFF - timedelta(days=1),
        "source": "test",
        "ingested_at": _KICKOFF - timedelta(days=1),
    }
    base.update(overrides)
    return base


# --- signal_outcome ---------------------------------------------------


@pytest.mark.parametrize(
    ("selection", "home_score", "away_score", "expected"),
    [
        ("home", 2, 1, BetOutcome.WIN),
        ("home", 1, 2, BetOutcome.LOSS),
        ("draw", 1, 1, BetOutcome.WIN),
        ("draw", 2, 1, BetOutcome.LOSS),
        ("away", 0, 3, BetOutcome.WIN),
        ("away", 3, 0, BetOutcome.LOSS),
    ],
)
def testsignal_outcome_one_x_two(
    selection: str, home_score: float, away_score: float, expected: BetOutcome
) -> None:
    outcome = signal_outcome("1x2", selection, None, home_score=home_score, away_score=away_score)
    assert outcome == expected


@pytest.mark.parametrize(
    ("selection", "home_score", "away_score", "expected"),
    [
        ("yes", 1, 1, BetOutcome.WIN),
        ("yes", 1, 0, BetOutcome.LOSS),
        ("no", 0, 0, BetOutcome.WIN),
        ("no", 2, 1, BetOutcome.LOSS),
    ],
)
def testsignal_outcome_btts(
    selection: str, home_score: float, away_score: float, expected: BetOutcome
) -> None:
    outcome = signal_outcome("btts", selection, None, home_score=home_score, away_score=away_score)
    assert outcome == expected


@pytest.mark.parametrize(
    ("selection", "home_score", "away_score", "expected"),
    [
        ("over", 2, 1, BetOutcome.WIN),
        ("over", 1, 1, BetOutcome.LOSS),
        ("under", 1, 1, BetOutcome.WIN),
        ("under", 2, 1, BetOutcome.LOSS),
        ("over", 1, 1.5, BetOutcome.PUSH),
    ],
)
def testsignal_outcome_over_under(
    selection: str, home_score: float, away_score: float, expected: BetOutcome
) -> None:
    outcome = signal_outcome(
        "over_under", selection, 2.5, home_score=home_score, away_score=away_score
    )
    assert outcome == expected


def testsignal_outcome_over_under_requires_a_line() -> None:
    with pytest.raises(ValueError, match="requiere una linea"):
        signal_outcome("over_under", "over", None, home_score=2, away_score=1)


@pytest.mark.parametrize(
    ("selection", "home_score", "away_score", "expected"),
    [
        ("home", 20, 10, BetOutcome.WIN),
        ("home", 10, 20, BetOutcome.LOSS),
        ("away", 10, 20, BetOutcome.WIN),
        ("home", 14, 14, BetOutcome.PUSH),
    ],
)
def testsignal_outcome_moneyline(
    selection: str, home_score: float, away_score: float, expected: BetOutcome
) -> None:
    outcome = signal_outcome(
        "moneyline", selection, None, home_score=home_score, away_score=away_score
    )
    assert outcome == expected


def testsignal_outcome_unsupported_market_raises() -> None:
    with pytest.raises(ValueError, match="no soportado"):
        signal_outcome("asian_handicap", "home", -1.5, home_score=2, away_score=1)


# --- pnl_for_outcome --------------------------------------------------


def test_pnl_for_win() -> None:
    assert pnl_for_outcome(BetOutcome.WIN, stake_fraction=0.02, entry_price=2.5) == pytest.approx(
        0.03
    )


def test_pnl_for_loss() -> None:
    assert pnl_for_outcome(BetOutcome.LOSS, stake_fraction=0.02, entry_price=2.5) == pytest.approx(
        -0.02
    )


def test_pnl_for_half_win() -> None:
    assert pnl_for_outcome(
        BetOutcome.HALF_WIN, stake_fraction=0.02, entry_price=2.5
    ) == pytest.approx(0.015)


def test_pnl_for_half_loss() -> None:
    assert pnl_for_outcome(
        BetOutcome.HALF_LOSS, stake_fraction=0.02, entry_price=2.5
    ) == pytest.approx(-0.01)


@pytest.mark.parametrize("outcome", [BetOutcome.PUSH, BetOutcome.VOID])
def test_pnl_for_push_and_void_is_zero(outcome: BetOutcome) -> None:
    assert pnl_for_outcome(outcome, stake_fraction=0.02, entry_price=2.5) == 0.0


# --- closing_price ----------------------------------------------------


def testclosing_price_prefers_the_is_closing_flag() -> None:
    odds = pd.DataFrame(
        [
            _odds_row(price=1.8, captured_at=_KICKOFF - timedelta(hours=2), is_closing=False),
            _odds_row(price=1.95, captured_at=_KICKOFF - timedelta(minutes=5), is_closing=True),
        ]
    )
    result = closing_price(
        odds,
        fixture_id=_FIXTURE_ID,
        bookmaker="pinnacle",
        market="over_under",
        selection="over",
        line=2.5,
        kickoff_utc=_KICKOFF,
    )
    assert result is not None
    price, captured_at = result
    assert price == pytest.approx(1.95)
    assert captured_at == _KICKOFF - timedelta(minutes=5)


def testclosing_price_falls_back_to_latest_pre_kickoff_snapshot() -> None:
    odds = pd.DataFrame(
        [
            _odds_row(price=1.8, captured_at=_KICKOFF - timedelta(hours=2), is_closing=False),
            _odds_row(price=1.85, captured_at=_KICKOFF - timedelta(minutes=5), is_closing=False),
        ]
    )
    result = closing_price(
        odds,
        fixture_id=_FIXTURE_ID,
        bookmaker="pinnacle",
        market="over_under",
        selection="over",
        line=2.5,
        kickoff_utc=_KICKOFF,
    )
    assert result is not None
    price, _ = result
    assert price == pytest.approx(1.85)


def testclosing_price_excludes_post_kickoff_snapshots() -> None:
    odds = pd.DataFrame(
        [_odds_row(price=2.5, captured_at=_KICKOFF + timedelta(minutes=10), is_closing=True)]
    )
    result = closing_price(
        odds,
        fixture_id=_FIXTURE_ID,
        bookmaker="pinnacle",
        market="over_under",
        selection="over",
        line=2.5,
        kickoff_utc=_KICKOFF,
    )
    assert result is None


def testclosing_price_none_without_any_matching_odds() -> None:
    odds = pd.DataFrame(
        [_odds_row(bookmaker="bet365")]  # otro bookmaker: no cuenta para este cierre
    )
    result = closing_price(
        odds,
        fixture_id=_FIXTURE_ID,
        bookmaker="pinnacle",
        market="over_under",
        selection="over",
        line=2.5,
        kickoff_utc=_KICKOFF,
    )
    assert result is None


# --- compute_and_write_results (integracion) -------------------------------


def test_compute_and_write_results_returns_zero_without_signals() -> None:
    assert compute_and_write_results(_COMPETITION) == 0


def test_compute_and_write_results_skips_unfinished_fixtures() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(
        pd.DataFrame([_fixture_row(status="scheduled", home_score=None, away_score=None)])
    )
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(pd.DataFrame([_signal_row()]))
    assert compute_and_write_results(_COMPETITION) == 0


def test_compute_and_write_results_skips_finished_fixtures_missing_a_score() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(
        pd.DataFrame([_fixture_row(home_score=None, away_score=None)])
    )
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(pd.DataFrame([_signal_row()]))
    assert compute_and_write_results(_COMPETITION) == 0


def test_compute_and_write_results_settles_a_winning_signal_with_clv() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(
        pd.DataFrame([_signal_row(entry_price=2.0, stake_fraction=0.02)])
    )
    ParquetTableRepository(ODDS_SNAPSHOTS, settings.parquet_dir).write(
        pd.DataFrame(
            [_odds_row(price=1.8, captured_at=_KICKOFF - timedelta(minutes=5), is_closing=True)]
        )
    )

    written = compute_and_write_results(_COMPETITION)

    assert written == 1
    results = ParquetTableRepository(RESULTS, settings.parquet_dir).read()
    row = results.iloc[0]
    assert row["signal_id"] == "sig1"
    assert row["outcome"] == "win"  # 2+1=3 > 2.5
    assert row["pnl"] == pytest.approx(0.02 * (2.0 - 1.0))
    assert row["closing_price"] == pytest.approx(1.8)
    assert row["clv"] == pytest.approx(2.0 / 1.8 - 1.0)


def test_compute_and_write_results_clv_is_none_without_closing_data() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(pd.DataFrame([_signal_row()]))

    written = compute_and_write_results(_COMPETITION)

    assert written == 1
    results = ParquetTableRepository(RESULTS, settings.parquet_dir).read()
    row = results.iloc[0]
    assert pd.isna(row["closing_price"])
    assert pd.isna(row["clv"])


def test_compute_and_write_results_is_idempotent_on_rerun() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(pd.DataFrame([_signal_row()]))

    compute_and_write_results(_COMPETITION)
    compute_and_write_results(_COMPETITION)

    results = ParquetTableRepository(RESULTS, settings.parquet_dir).read()
    assert len(results) == 1


def test_compute_and_write_results_settles_descartar_signals_too() -> None:
    # stake_fraction=0 (tier descartar): pnl siempre 0, pero el CLV se calcula
    # igual -- es la verificacion honesta de si el tier hizo bien en descartar
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(
        pd.DataFrame([_signal_row(tier="descartar", stake_fraction=0.0)])
    )
    ParquetTableRepository(ODDS_SNAPSHOTS, settings.parquet_dir).write(
        pd.DataFrame(
            [_odds_row(price=1.8, captured_at=_KICKOFF - timedelta(minutes=5), is_closing=True)]
        )
    )

    written = compute_and_write_results(_COMPETITION)

    assert written == 1
    results = ParquetTableRepository(RESULTS, settings.parquet_dir).read()
    row = results.iloc[0]
    assert row["pnl"] == 0.0
    assert row["clv"] == pytest.approx(2.0 / 1.8 - 1.0)
