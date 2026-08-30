"""Baseline strategies — ``config/thresholds.yaml``'s ``backtest.baselines``
list (``always_favourite``, ``random``). These answer the question a real
edge estimate has to beat to be worth anything: what would a naive strategy,
staking a flat unit on the exact same markets at the exact same entry
snapshot as a real signal, have returned?

"Exact same entry snapshot" matters: a baseline computed from a different
moment's prices would make any difference in performance an artefact of
which odds got compared, not of the strategies themselves. So this reads
the same ``(fixture, bookmaker, market, line, captured_at)`` the real signal
already resolved and picks a different selection from that identical
snapshot, rather than re-resolving the market independently.

The stake is a flat unit (``stake_fraction`` is meaningless here — a
baseline has no probability estimate to size a Kelly fraction against), so
``pnl`` is directly a per-bet return, not a bankroll fraction; CLV is
unaffected by staking and stays comparable as-is.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import cast

import pandas as pd

from deportivas.backtest.settlement import closing_price, pnl_for_outcome, signal_outcome
from deportivas.contracts.tables import ODDS_SNAPSHOTS, SIGNALS
from deportivas.domain.enums import FixtureStatus
from deportivas.features.asof import load_fixtures
from deportivas.odds.resolve import line_mask, market_snapshots
from deportivas.storage.factory import get_table_repository

ALWAYS_FAVOURITE = "always_favourite"
RANDOM = "random"
_FLAT_STAKE = 1.0


def _baseline_selection(prices: dict[str, float], *, baseline: str, seed: str) -> str:
    if baseline == ALWAYS_FAVOURITE:
        # el precio decimal mas bajo es la seleccion con mayor probabilidad
        # implicita; un empate a precio lo resuelve el orden de insercion
        return min(prices, key=lambda selection: prices[selection])
    if baseline == RANDOM:
        return random.Random(seed).choice(sorted(prices))
    raise ValueError(f"baseline desconocida: {baseline!r} (usar {ALWAYS_FAVOURITE!r} o {RANDOM!r})")


_RESULT_COLUMNS = [
    "signal_id",
    "fixture_id",
    "market",
    "selection",
    "entry_price",
    "outcome",
    "pnl",
    "clv",
]


def compute_baseline_results(competition_id: str, baseline: str) -> pd.DataFrame:
    """For every signal that would be settleable (same criterion as
    ``settlement.compute_and_write_results``: a finished fixture with a known
    score), looks at the exact entry snapshot the real signal used, picks a
    selection per ``baseline``'s rule, and settles it with a flat stake.
    Returns a DataFrame with one row per settled baseline bet — never
    persisted to a table, it exists only for ``backtest/report.py`` to
    compare against the real results."""
    signals = get_table_repository(SIGNALS).read(filters={"competition_id": competition_id})
    if signals.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    fixtures = load_fixtures(competition_id)
    fixtures_by_id = {str(row["id"]): row for row in fixtures.to_dict("records")}
    odds = get_table_repository(ODDS_SNAPSHOTS).read(filters={"competition_id": competition_id})

    rows: list[dict[str, object]] = []
    for record in signals.to_dict("records"):
        fixture_id = str(record["fixture_id"])
        fixture = fixtures_by_id.get(fixture_id)
        if fixture is None or fixture["status"] != FixtureStatus.FINISHED.value:
            continue
        home_score, away_score = fixture["home_score"], fixture["away_score"]
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        market = str(record["market"])
        line = float(record["line"]) if pd.notna(record["line"]) else None
        bookmaker = str(record["entry_bookmaker"])
        entry_captured_at = cast(datetime, record["entry_captured_at"])

        entry_slice = odds[
            (odds["fixture_id"] == fixture_id)
            & (odds["bookmaker"] == bookmaker)
            & (odds["market"] == market)
            & line_mask(odds["line"], line)
            & (odds["captured_at"] == entry_captured_at)
        ]
        snapshots = market_snapshots(entry_slice)
        if not snapshots:
            continue
        prices = snapshots[0].prices

        signal_id = str(record["id"])
        selection = _baseline_selection(prices, baseline=baseline, seed=f"{baseline}:{signal_id}")
        entry_price = prices[selection]

        outcome = signal_outcome(
            market, selection, line, home_score=float(home_score), away_score=float(away_score)
        )
        pnl = pnl_for_outcome(outcome, stake_fraction=_FLAT_STAKE, entry_price=entry_price)
        closing = closing_price(
            odds,
            fixture_id=fixture_id,
            bookmaker=bookmaker,
            market=market,
            selection=selection,
            line=line,
            kickoff_utc=cast(datetime, fixture["kickoff_utc"]),
        )
        closing_price_value = closing[0] if closing is not None else None
        clv = (entry_price / closing_price_value - 1.0) if closing_price_value is not None else None

        rows.append(
            {
                "signal_id": signal_id,
                "fixture_id": fixture_id,
                "market": market,
                "selection": selection,
                "entry_price": entry_price,
                "outcome": outcome.value,
                "pnl": pnl,
                "clv": clv,
            }
        )

    if not rows:
        return pd.DataFrame(columns=_RESULT_COLUMNS)
    return pd.DataFrame(rows, columns=_RESULT_COLUMNS)
