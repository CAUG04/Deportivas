"""resolve.py: point-in-time market resolution shared by signal generation
and the backtest's baseline strategies."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from deportivas.odds.resolve import fair_probabilities, line_mask, market_snapshots, resolve_market

_FIXTURE_ID = "fx1"
_AS_OF = datetime(2024, 1, 1, tzinfo=UTC)
_KICKOFF = datetime(2024, 1, 10, tzinfo=UTC)

_odds_id_counter = itertools.count()


def _odds_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": f"odds{next(_odds_id_counter)}",
        "fixture_id": _FIXTURE_ID,
        "competition_id": "test-comp",
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


# --- line_mask --------------------------------------------------------------


def test_line_mask_none_matches_nan_lines() -> None:
    lines = pd.Series([1.5, None, 2.5])
    assert list(line_mask(lines, None)) == [False, True, False]


def test_line_mask_value_matches_exact_line() -> None:
    lines = pd.Series([1.5, 2.5, 2.5])
    assert list(line_mask(lines, 2.5)) == [False, True, True]


# --- market_snapshots ---------------------------------------------------


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
    snapshots = market_snapshots(df)
    assert [s.captured_at for s in snapshots] == [early, late]
    assert snapshots[0].prices == {"over": 1.9, "under": 2.2}
    assert snapshots[1].prices == {"over": 2.0, "under": 2.0}


def test_market_snapshots_drops_moments_with_a_single_priced_selection() -> None:
    df = pd.DataFrame([_odds_row(selection="over", price=2.0)])
    assert market_snapshots(df) == []


# --- resolve_market ---------------------------------------------------------


def test_resolve_market_prefers_reference_bookmaker() -> None:
    df = pd.DataFrame(
        [
            _odds_row(bookmaker="pinnacle", selection="over", price=2.0),
            _odds_row(bookmaker="pinnacle", selection="under", price=2.0),
            _odds_row(bookmaker="bet365", selection="over", price=1.8),
            _odds_row(bookmaker="bet365", selection="under", price=2.1),
        ]
    )
    resolved = resolve_market(
        df,
        fixture_id=_FIXTURE_ID,
        market="over_under",
        line=2.5,
        as_of_timestamp=_AS_OF,
        kickoff_utc=_KICKOFF,
        reference_bookmaker="pinnacle",
        fallback_bookmakers=("bet365",),
        required_selection="over",
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
    resolved = resolve_market(
        df,
        fixture_id=_FIXTURE_ID,
        market="over_under",
        line=2.5,
        as_of_timestamp=_AS_OF,
        kickoff_utc=_KICKOFF,
        reference_bookmaker="pinnacle",
        fallback_bookmakers=("betfair_ex", "bet365"),
        required_selection="over",
    )
    assert resolved is not None
    assert resolved.bookmaker == "bet365"


def test_resolve_market_returns_none_without_any_usable_bookmaker() -> None:
    # una sola seleccion capturada: no hay margen que quitar, se descarta
    df = pd.DataFrame([_odds_row(bookmaker="pinnacle", selection="over", price=2.0)])
    resolved = resolve_market(
        df,
        fixture_id=_FIXTURE_ID,
        market="over_under",
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
    resolved = resolve_market(
        df,
        fixture_id=_FIXTURE_ID,
        market="over_under",
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
    resolved = resolve_market(
        df,
        fixture_id=_FIXTURE_ID,
        market="over_under",
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
    resolved = resolve_market(
        df,
        fixture_id=_FIXTURE_ID,
        market="btts",
        line=None,
        as_of_timestamp=_AS_OF,
        kickoff_utc=_KICKOFF,
        reference_bookmaker="pinnacle",
        fallback_bookmakers=(),
        required_selection="yes",
    )
    assert resolved is not None
    assert resolved.snapshots[0].prices == {"yes": 2.0, "no": 2.0}


def test_resolve_market_without_required_selection_returns_every_priced_selection() -> None:
    # sin required_selection (uso de backtest/baselines.py): no filtra por
    # ninguna seleccion en particular, sirve para elegir el favorito despues
    df = pd.DataFrame(
        [
            _odds_row(market="1x2", selection="home", line=None, price=1.8),
            _odds_row(market="1x2", selection="draw", line=None, price=3.5),
            _odds_row(market="1x2", selection="away", line=None, price=4.5),
        ]
    )
    resolved = resolve_market(
        df,
        fixture_id=_FIXTURE_ID,
        market="1x2",
        line=None,
        as_of_timestamp=_AS_OF,
        kickoff_utc=_KICKOFF,
        reference_bookmaker="pinnacle",
        fallback_bookmakers=(),
    )
    assert resolved is not None
    assert resolved.snapshots[0].prices == {"home": 1.8, "draw": 3.5, "away": 4.5}


def test_resolve_market_required_selection_skips_bookmaker_missing_it() -> None:
    # pinnacle solo tiene "over"; bet365 tiene ambas: debe caer a bet365
    df = pd.DataFrame(
        [
            _odds_row(bookmaker="pinnacle", selection="over", price=2.0),
            _odds_row(bookmaker="bet365", selection="over", price=1.9),
            _odds_row(bookmaker="bet365", selection="under", price=2.0),
        ]
    )
    resolved = resolve_market(
        df,
        fixture_id=_FIXTURE_ID,
        market="over_under",
        line=2.5,
        as_of_timestamp=_AS_OF,
        kickoff_utc=_KICKOFF,
        reference_bookmaker="pinnacle",
        fallback_bookmakers=("bet365",),
        required_selection="under",
    )
    assert resolved is not None
    assert resolved.bookmaker == "bet365"


# --- fair_probabilities ------------------------------------------------


def test_fair_probabilities_normalizes_by_selection() -> None:
    fair = fair_probabilities({"over": 2.0, "under": 2.0}, method="multiplicative")
    assert fair == {"over": pytest.approx(0.5), "under": pytest.approx(0.5)}
