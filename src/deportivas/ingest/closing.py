"""Flags the true closing line on ``odds_snapshots`` — Fase 8's settlement
job, referenced since Fase 1 (see ``theoddsapi.py`` and
``storage/protocols.py``'s own docstring on ``mark_closing``) but only now
implemented, once a fixture has actually kicked off and no further
pre-kickoff snapshot can ever arrive for it.

This is an optimisation, not a correctness dependency: ``backtest/settlement.py``'s
``closing_price`` already falls back to the latest pre-kickoff snapshot at
read time when no row is flagged, so CLV is correct with or without this
job ever running. What this buys is a persisted, auditable answer to "what
was the closing line" that doesn't have to be recomputed on every read, and
that a live ``is_closing=True`` bookmaker line in the raw data means what it
says.
"""

from __future__ import annotations

from datetime import UTC, datetime

from deportivas.contracts.tables import ODDS_SNAPSHOTS
from deportivas.features.asof import load_fixtures
from deportivas.storage.factory import get_table_repository

_GROUP_COLUMNS = ["bookmaker", "market", "selection", "line"]


def mark_closing_lines(competition_id: str, *, now: datetime | None = None) -> int:
    """For every fixture in ``competition_id`` that has already kicked off,
    flags the single latest odds snapshot captured at or before kickoff, per
    (bookmaker, market, selection, line), as ``is_closing=True``.

    Skips a fixture that already carries at least one ``is_closing=True``
    row among its own snapshots — cheap and safe to run daily against every
    fixture in the competition without re-scanning history each time; once a
    fixture's closing line is marked, it never needs revisiting (the
    snapshots that decided it are immutable, and nothing pre-kickoff can
    ever arrive after the fact). Returns the number of rows newly flagged.
    """
    current_time = now if now is not None else datetime.now(UTC)
    fixtures = load_fixtures(competition_id)
    started = fixtures[fixtures["kickoff_utc"] <= current_time]
    if started.empty:
        return 0

    odds_repo = get_table_repository(ODDS_SNAPSHOTS)
    odds = odds_repo.read(filters={"competition_id": competition_id})
    if odds.empty:
        return 0

    ids_to_mark: list[str] = []
    for fixture_id, kickoff in zip(started["id"], started["kickoff_utc"], strict=True):
        fixture_odds = odds[odds["fixture_id"] == fixture_id]
        if fixture_odds.empty or fixture_odds["is_closing"].any():
            continue  # sin cuotas capturadas, o ya marcado en una corrida anterior

        pre_kickoff = fixture_odds[fixture_odds["captured_at"] <= kickoff]
        if pre_kickoff.empty:
            continue

        latest_idx = pre_kickoff.groupby(_GROUP_COLUMNS, dropna=False)["captured_at"].idxmax()
        ids_to_mark.extend(pre_kickoff.loc[latest_idx, "id"].tolist())

    if not ids_to_mark:
        return 0
    return odds_repo.mark_closing(ids_to_mark)
