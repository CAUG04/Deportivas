"""Shared point-in-time odds resolution.

Used by signal generation (:mod:`deportivas.signals.generate`) and by the
backtest's baseline strategies (:mod:`deportivas.backtest.baselines`), so
both look at exactly the same market snapshots when comparing a real signal
against what a naive strategy would have done — otherwise a difference in
performance could just be an artefact of resolving odds differently, not of
the strategies themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import cast

import pandas as pd

from deportivas.signals.devig import devig


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    captured_at: datetime
    prices: dict[str, float]  # selection -> cuota decimal, todas capturadas juntas


@dataclass(frozen=True, slots=True)
class ResolvedMarket:
    bookmaker: str
    snapshots: list[MarketSnapshot]  # cronologico, mas antiguo primero


def market_snapshots(odds: pd.DataFrame) -> list[MarketSnapshot]:
    """Groups one (fixture, bookmaker, market, line) slice of ``odds_snapshots``
    by ``captured_at`` — every selection of a market is scraped together in
    one pass, so rows sharing a timestamp are one snapshot. A moment with
    fewer than two priced selections carries no margin to remove and is not
    real market data, so it is dropped rather than devigged into a trivial
    (and wrong) probability of 1.0."""
    snapshots = [
        MarketSnapshot(
            captured_at=cast(datetime, captured_at),
            prices=dict(zip(group["selection"], group["price"], strict=True)),
        )
        for captured_at, group in odds.groupby("captured_at")
    ]
    return sorted((s for s in snapshots if len(s.prices) >= 2), key=lambda s: s.captured_at)


def line_mask(lines: pd.Series, line: float | None) -> pd.Series:
    if line is None:
        return lines.isna()
    return lines == line


def resolve_market(
    odds: pd.DataFrame,
    *,
    fixture_id: str,
    market: str,
    line: float | None,
    as_of_timestamp: datetime,
    kickoff_utc: datetime,
    reference_bookmaker: str,
    fallback_bookmakers: tuple[str, ...],
    required_selection: str | None = None,
) -> ResolvedMarket | None:
    """Tries ``reference_bookmaker`` first, then each fallback in order,
    returning the first one with at least one in-window snapshot — one that
    also prices ``required_selection``, when given. ``None`` when no
    bookmaker has anything usable — there is nothing honest to build on."""
    same_market = odds[
        (odds["fixture_id"] == fixture_id)
        & (odds["market"] == market)
        & line_mask(odds["line"], line)
        & (odds["captured_at"] >= as_of_timestamp)
        & (odds["captured_at"] <= kickoff_utc)
    ]
    for bookmaker in (reference_bookmaker, *fallback_bookmakers):
        snapshots = market_snapshots(same_market[same_market["bookmaker"] == bookmaker])
        if required_selection is not None:
            snapshots = [s for s in snapshots if required_selection in s.prices]
        if snapshots:
            return ResolvedMarket(bookmaker=bookmaker, snapshots=snapshots)
    return None


def fair_probabilities(prices: dict[str, float], *, method: str) -> dict[str, float]:
    selections = list(prices)
    fair = devig([prices[s] for s in selections], method=method)
    return dict(zip(selections, fair, strict=True))
