"""generate.py: joins predictions against observed odds, devigs, computes
edge, classifies a tier and, for actionable tiers, a Kelly stake."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from deportivas.config.catalog import LineMoveConfig, load_thresholds
from deportivas.config.settings import get_settings
from deportivas.contracts.tables import (
    FIXTURES,
    MODEL_REGISTRY,
    ODDS_SNAPSHOTS,
    PREDICTIONS,
    SIGNALS,
)
from deportivas.domain.ids import deterministic_id
from deportivas.signals.generate import (
    _build_signal_row,
    _fair_probabilities,
    _has_favourable_line_move,
    _line_mask,
    _market_snapshots,
    _MarketSnapshot,
    _resolve_market,
    _ResolvedMarket,
    _sample_matches_by_model,
    compute_and_write_signals,
)
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository

_COMPETITION = "test-comp"
_FIXTURE_ID = "fx1"
_AS_OF = datetime(2024, 1, 1, tzinfo=UTC)
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
        "price": 2.0,
        "captured_at": _AS_OF + timedelta(days=1),
        "is_closing": False,
        "source": "test",
        "ingested_at": _AS_OF,
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
        "ingested_at": _AS_OF,
    }
    base.update(overrides)
    return base


def _prediction_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "pred1",
        "fixture_id": _FIXTURE_ID,
        "competition_id": _COMPETITION,
        "season": "s1",
        "model_name": "m1",
        "model_version": "v1",
        "market": "over_under",
        "selection": "over",
        "line": 2.5,
        "prob_raw": 0.6,
        "prob_calibrated": 0.65,
        "as_of_timestamp": _AS_OF,
        "predicted_at": _AS_OF,
        "source": "m1",
        "ingested_at": _AS_OF,
    }
    base.update(overrides)
    return base


def _model_registry_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "reg1",
        "model_name": "m1",
        "model_version": "v1",
        "sport": "football",
        "feature_set": "fs1",
        "trained_at": _AS_OF,
        "train_window_start": _AS_OF,
        "train_window_end": _AS_OF,
        "hyperparameters": json.dumps({}),
        "metrics": json.dumps({"n_train_matches": 500}),
        "calibration_method": "isotonic",
        "git_sha": None,
        "source": "test",
        "ingested_at": _AS_OF,
    }
    base.update(overrides)
    return base


# --- _line_mask -------------------------------------------------------------


def test_line_mask_none_matches_nan_lines() -> None:
    lines = pd.Series([1.5, None, 2.5])
    assert list(_line_mask(lines, None)) == [False, True, False]


def test_line_mask_value_matches_exact_line() -> None:
    lines = pd.Series([1.5, 2.5, 2.5])
    assert list(_line_mask(lines, 2.5)) == [False, True, True]


# --- _market_snapshots --------------------------------------------------


def test_market_snapshots_groups_by_captured_at_and_sorts_chronologically() -> None:
    early = _AS_OF
    late = _AS_OF + timedelta(days=1)
    df = pd.DataFrame(
        [
            _odds_row(selection="over", price=2.0, captured_at=late),
            _odds_row(selection="under", price=2.0, captured_at=late),
            _odds_row(selection="over", price=1.9, captured_at=early),
            _odds_row(selection="under", price=2.2, captured_at=early),
        ]
    )
    snapshots = _market_snapshots(df)
    assert [s.captured_at for s in snapshots] == [early, late]
    assert snapshots[0].prices == {"over": 1.9, "under": 2.2}
    assert snapshots[1].prices == {"over": 2.0, "under": 2.0}


def test_market_snapshots_drops_moments_with_a_single_priced_selection() -> None:
    df = pd.DataFrame([_odds_row(selection="over", price=2.0)])
    assert _market_snapshots(df) == []


# --- _resolve_market ------------------------------------------------------


def test_resolve_market_prefers_reference_bookmaker() -> None:
    df = pd.DataFrame(
        [
            _odds_row(bookmaker="pinnacle", selection="over", price=2.0),
            _odds_row(bookmaker="pinnacle", selection="under", price=2.0),
            _odds_row(bookmaker="bet365", selection="over", price=1.8),
            _odds_row(bookmaker="bet365", selection="under", price=2.1),
        ]
    )
    resolved = _resolve_market(
        df,
        fixture_id=_FIXTURE_ID,
        market="over_under",
        selection="over",
        line=2.5,
        as_of_timestamp=_AS_OF,
        kickoff_utc=_KICKOFF,
        reference_bookmaker="pinnacle",
        fallback_bookmakers=("bet365",),
    )
    assert resolved is not None
    assert resolved.bookmaker == "pinnacle"


def test_resolve_market_falls_back_in_order() -> None:
    df = pd.DataFrame(
        [
            _odds_row(bookmaker="bet365", selection="over", price=1.8),
            _odds_row(bookmaker="bet365", selection="under", price=2.1),
        ]
    )
    resolved = _resolve_market(
        df,
        fixture_id=_FIXTURE_ID,
        market="over_under",
        selection="over",
        line=2.5,
        as_of_timestamp=_AS_OF,
        kickoff_utc=_KICKOFF,
        reference_bookmaker="pinnacle",
        fallback_bookmakers=("betfair_ex", "bet365"),
    )
    assert resolved is not None
    assert resolved.bookmaker == "bet365"


def test_resolve_market_returns_none_without_any_usable_bookmaker() -> None:
    # una sola seleccion capturada: no hay margen que quitar, se descarta
    df = pd.DataFrame([_odds_row(bookmaker="pinnacle", selection="over", price=2.0)])
    resolved = _resolve_market(
        df,
        fixture_id=_FIXTURE_ID,
        market="over_under",
        selection="over",
        line=2.5,
        as_of_timestamp=_AS_OF,
        kickoff_utc=_KICKOFF,
        reference_bookmaker="pinnacle",
        fallback_bookmakers=(),
    )
    assert resolved is None


def test_resolve_market_excludes_odds_captured_before_as_of_timestamp() -> None:
    too_early = _AS_OF - timedelta(days=1)
    df = pd.DataFrame(
        [
            _odds_row(bookmaker="pinnacle", selection="over", price=2.0, captured_at=too_early),
            _odds_row(bookmaker="pinnacle", selection="under", price=2.0, captured_at=too_early),
        ]
    )
    resolved = _resolve_market(
        df,
        fixture_id=_FIXTURE_ID,
        market="over_under",
        selection="over",
        line=2.5,
        as_of_timestamp=_AS_OF,
        kickoff_utc=_KICKOFF,
        reference_bookmaker="pinnacle",
        fallback_bookmakers=(),
    )
    assert resolved is None


def test_resolve_market_excludes_odds_captured_after_kickoff() -> None:
    too_late = _KICKOFF + timedelta(days=1)
    df = pd.DataFrame(
        [
            _odds_row(bookmaker="pinnacle", selection="over", price=2.0, captured_at=too_late),
            _odds_row(bookmaker="pinnacle", selection="under", price=2.0, captured_at=too_late),
        ]
    )
    resolved = _resolve_market(
        df,
        fixture_id=_FIXTURE_ID,
        market="over_under",
        selection="over",
        line=2.5,
        as_of_timestamp=_AS_OF,
        kickoff_utc=_KICKOFF,
        reference_bookmaker="pinnacle",
        fallback_bookmakers=(),
    )
    assert resolved is None


def test_resolve_market_matches_lineless_markets_via_none() -> None:
    df = pd.DataFrame(
        [
            _odds_row(market="btts", selection="yes", line=None, price=2.0),
            _odds_row(market="btts", selection="no", line=None, price=2.0),
        ]
    )
    resolved = _resolve_market(
        df,
        fixture_id=_FIXTURE_ID,
        market="btts",
        selection="yes",
        line=None,
        as_of_timestamp=_AS_OF,
        kickoff_utc=_KICKOFF,
        reference_bookmaker="pinnacle",
        fallback_bookmakers=(),
    )
    assert resolved is not None
    assert resolved.snapshots[0].prices == {"yes": 2.0, "no": 2.0}


# --- _fair_probabilities --------------------------------------------------


def test_fair_probabilities_normalizes_by_selection() -> None:
    fair = _fair_probabilities({"over": 2.0, "under": 2.0}, method="multiplicative")
    assert fair == {"over": pytest.approx(0.5), "under": pytest.approx(0.5)}


# --- _has_favourable_line_move --------------------------------------------


def test_favourable_line_move_requires_min_snapshots() -> None:
    resolved = _ResolvedMarket(
        bookmaker="pinnacle",
        snapshots=[_MarketSnapshot(captured_at=_AS_OF, prices={"over": 2.0, "under": 2.0})],
    )
    config = LineMoveConfig(favourable_drop=0.01, min_snapshots=2)
    assert (
        _has_favourable_line_move(
            resolved, selection="over", method="multiplicative", config=config
        )
        is False
    )


def test_favourable_line_move_true_when_fair_price_drops_enough() -> None:
    resolved = _ResolvedMarket(
        bookmaker="pinnacle",
        snapshots=[
            _MarketSnapshot(captured_at=_AS_OF, prices={"over": 2.0, "under": 2.0}),
            _MarketSnapshot(captured_at=_KICKOFF, prices={"over": 1.8, "under": 2.25}),
        ],
    )
    config = LineMoveConfig(favourable_drop=0.01, min_snapshots=2)
    assert (
        _has_favourable_line_move(
            resolved, selection="over", method="multiplicative", config=config
        )
        is True
    )


def test_favourable_line_move_false_when_price_moves_the_wrong_way() -> None:
    resolved = _ResolvedMarket(
        bookmaker="pinnacle",
        snapshots=[
            _MarketSnapshot(captured_at=_AS_OF, prices={"over": 1.8, "under": 2.25}),
            _MarketSnapshot(captured_at=_KICKOFF, prices={"over": 2.0, "under": 2.0}),
        ],
    )
    config = LineMoveConfig(favourable_drop=0.01, min_snapshots=2)
    assert (
        _has_favourable_line_move(
            resolved, selection="over", method="multiplicative", config=config
        )
        is False
    )


# --- _sample_matches_by_model ----------------------------------------------


def test_sample_matches_by_model_reads_n_train_matches_from_metrics_json() -> None:
    registry = pd.DataFrame(
        [
            {
                "model_name": "m1",
                "model_version": "v1",
                "metrics": json.dumps({"n_train_matches": 500}),
            }
        ]
    )
    assert _sample_matches_by_model(registry) == {("m1", "v1"): 500}


def test_sample_matches_by_model_skips_rows_missing_the_key() -> None:
    registry = pd.DataFrame(
        [{"model_name": "m1", "model_version": "v1", "metrics": json.dumps({"brier": 0.2})}]
    )
    assert _sample_matches_by_model(registry) == {}


# --- _build_signal_row -----------------------------------------------------


def test_build_signal_row_returns_none_without_resolved_odds() -> None:
    thresholds = load_thresholds()
    odds = pd.DataFrame(
        columns=["fixture_id", "bookmaker", "market", "selection", "line", "price", "captured_at"]
    )
    row = _build_signal_row(
        fixture_id=_FIXTURE_ID,
        competition_id=_COMPETITION,
        season="s1",
        market="over_under",
        selection="over",
        line=2.5,
        model_name="m1",
        model_version="v1",
        prob_raw=0.6,
        prob_calibrated=None,
        as_of_timestamp=_AS_OF,
        kickoff_utc=_KICKOFF,
        odds=odds,
        sample_matches=0,
        thresholds=thresholds,
        now=_AS_OF,
    )
    assert row is None


def test_build_signal_row_computes_edge_prob_model_and_uses_calibrated_when_present() -> None:
    thresholds = load_thresholds()
    odds = pd.DataFrame(
        [
            _odds_row(bookmaker="pinnacle", selection="over", price=2.0),
            _odds_row(bookmaker="pinnacle", selection="under", price=2.0),
        ]
    )
    row = _build_signal_row(
        fixture_id=_FIXTURE_ID,
        competition_id=_COMPETITION,
        season="s1",
        market="over_under",
        selection="over",
        line=2.5,
        model_name="m1",
        model_version="v1",
        prob_raw=0.55,
        prob_calibrated=0.65,
        as_of_timestamp=_AS_OF,
        kickoff_utc=_KICKOFF,
        odds=odds,
        sample_matches=10,
        thresholds=thresholds,
        now=_AS_OF,
    )
    assert row is not None
    assert row["id"] == deterministic_id(_FIXTURE_ID, "over_under", "over", "2.5", "m1", "v1")
    assert row["prob_model"] == pytest.approx(0.65)
    assert row["prob_fair"] == pytest.approx(0.5)
    assert row["fair_price"] == pytest.approx(2.0)
    assert row["edge"] == pytest.approx(0.15)
    assert row["entry_price"] == 2.0
    assert row["entry_bookmaker"] == "pinnacle"
    assert row["tier"] == "baja"  # sample_matches=10 < 300: no alcanza alta ni media
    assert row["stake_fraction"] == 0.0  # baja no es accionable
    # el desglose es el de "baja" (el tier obtenido), no el de "alta": por
    # eso las condiciones que "baja" no exige salen en True aunque no se
    # cumplan realmente para alta/media
    reasons = json.loads(row["tier_reasons"])  # type: ignore[arg-type]
    assert reasons == {
        "min_edge": True,
        "min_sample_matches": True,
        "requires_calibration": True,
        "requires_sharp_price": True,
        "requires_favourable_line_move": True,
    }


def test_build_signal_row_negative_edge_gives_descartar_tier() -> None:
    thresholds = load_thresholds()
    odds = pd.DataFrame(
        [
            _odds_row(bookmaker="pinnacle", selection="over", price=2.0),
            _odds_row(bookmaker="pinnacle", selection="under", price=2.0),
        ]
    )
    row = _build_signal_row(
        fixture_id=_FIXTURE_ID,
        competition_id=_COMPETITION,
        season="s1",
        market="over_under",
        selection="over",
        line=2.5,
        model_name="m1",
        model_version="v1",
        prob_raw=0.3,
        prob_calibrated=None,
        as_of_timestamp=_AS_OF,
        kickoff_utc=_KICKOFF,
        odds=odds,
        sample_matches=0,
        thresholds=thresholds,
        now=_AS_OF,
    )
    assert row is not None
    assert row["prob_model"] == pytest.approx(0.3)  # no calibrada: usa prob_raw
    assert row["tier"] == "descartar"
    assert row["stake_fraction"] == 0.0


# --- compute_and_write_signals (integracion) -------------------------------


def test_compute_and_write_signals_returns_zero_without_predictions() -> None:
    assert compute_and_write_signals(_COMPETITION) == 0


def test_compute_and_write_signals_returns_zero_when_odds_are_empty() -> None:
    settings = get_settings()
    ParquetTableRepository(PREDICTIONS, settings.parquet_dir).write(
        pd.DataFrame([_prediction_row()])
    )
    assert compute_and_write_signals(_COMPETITION) == 0


def test_compute_and_write_signals_persists_descartar_as_a_valid_outcome() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(PREDICTIONS, settings.parquet_dir).write(
        pd.DataFrame([_prediction_row(prob_raw=0.3, prob_calibrated=None)])
    )
    ParquetTableRepository(ODDS_SNAPSHOTS, settings.parquet_dir).write(
        pd.DataFrame(
            [
                _odds_row(bookmaker="pinnacle", selection="over", price=2.0),
                _odds_row(bookmaker="pinnacle", selection="under", price=2.0),
            ]
        )
    )

    written = compute_and_write_signals(_COMPETITION)

    assert written == 1
    signals = ParquetTableRepository(SIGNALS, settings.parquet_dir).read()
    assert len(signals) == 1
    assert signals.iloc[0]["tier"] == "descartar"
    assert signals.iloc[0]["stake_fraction"] == 0.0


def test_compute_and_write_signals_skips_predictions_without_a_known_fixture() -> None:
    settings = get_settings()
    # sin fixtures en absoluto: load_fixtures() da un dataframe vacio
    ParquetTableRepository(PREDICTIONS, settings.parquet_dir).write(
        pd.DataFrame([_prediction_row()])
    )
    ParquetTableRepository(ODDS_SNAPSHOTS, settings.parquet_dir).write(
        pd.DataFrame(
            [
                _odds_row(bookmaker="pinnacle", selection="over", price=2.0),
                _odds_row(bookmaker="pinnacle", selection="under", price=2.0),
            ]
        )
    )
    assert compute_and_write_signals(_COMPETITION) == 0


def test_compute_and_write_signals_skips_predictions_without_resolvable_odds() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(PREDICTIONS, settings.parquet_dir).write(
        pd.DataFrame(
            [
                _prediction_row(),
                _prediction_row(id="pred2", selection="under", line=99.5),
            ]
        )
    )
    ParquetTableRepository(ODDS_SNAPSHOTS, settings.parquet_dir).write(
        pd.DataFrame(
            [
                _odds_row(bookmaker="pinnacle", selection="over", line=2.5, price=2.0),
                _odds_row(bookmaker="pinnacle", selection="under", line=2.5, price=2.0),
            ]
        )
    )

    written = compute_and_write_signals(_COMPETITION)

    assert written == 1
    signals = ParquetTableRepository(SIGNALS, settings.parquet_dir).read()
    assert len(signals) == 1
    assert signals.iloc[0]["line"] == 2.5


def test_compute_and_write_signals_reaches_alta_tier_with_full_conditions_met() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(PREDICTIONS, settings.parquet_dir).write(
        pd.DataFrame([_prediction_row(prob_raw=0.6, prob_calibrated=0.7)])
    )
    ParquetTableRepository(ODDS_SNAPSHOTS, settings.parquet_dir).write(
        pd.DataFrame(
            [
                _odds_row(
                    bookmaker="pinnacle",
                    selection="over",
                    price=2.0,
                    captured_at=_AS_OF + timedelta(days=1),
                ),
                _odds_row(
                    bookmaker="pinnacle",
                    selection="under",
                    price=2.0,
                    captured_at=_AS_OF + timedelta(days=1),
                ),
                _odds_row(
                    bookmaker="pinnacle",
                    selection="over",
                    price=1.8,
                    captured_at=_AS_OF + timedelta(days=2),
                ),
                _odds_row(
                    bookmaker="pinnacle",
                    selection="under",
                    price=2.25,
                    captured_at=_AS_OF + timedelta(days=2),
                ),
            ]
        )
    )
    ParquetTableRepository(MODEL_REGISTRY, settings.parquet_dir).write(
        pd.DataFrame([_model_registry_row()])
    )

    written = compute_and_write_signals(_COMPETITION)

    assert written == 1
    signals = ParquetTableRepository(SIGNALS, settings.parquet_dir).read()
    row = signals.iloc[0]
    assert row["tier"] == "alta"
    assert row["stake_fraction"] > 0.0
    assert row["entry_price"] == 2.0
    assert row["entry_bookmaker"] == "pinnacle"
    assert row["devig_method"] == load_thresholds().devig.method.value
