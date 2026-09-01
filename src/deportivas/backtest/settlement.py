"""Settlement — grades every ``signals`` row against its fixture's final
score and writes one ``results`` row per settled signal, including
``tier == descartar`` and ``baja`` ones: their CLV and outcome are exactly
the honest check on whether the tier system itself is calling it right,
not something to discard just because no money was staked on them (their
``stake_fraction`` is 0, so their ``pnl`` is always 0 regardless of outcome
— only CLV is informative there).

CLV (``entry_price / closing_price - 1``) is this project's primary metric,
not pnl: pnl depends on stake sizing and on which side of variance a single
bet landed on, while CLV asks the question that actually validates the
edge — did the market itself move to agree with the price we got, before
the game even started.

The closing price is looked up on the *same* bookmaker recorded as
``entry_bookmaker`` on the signal, preferring a row flagged ``is_closing``
when one exists; until the job that flags it (Fase 8) exists, this falls
back to the latest snapshot captured at or before kickoff for that same
bookmaker — the best available proxy for "the price the market settled on"
without that flag.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pandas as pd

from deportivas.contracts.tables import ODDS_SNAPSHOTS, RESULTS, SIGNALS
from deportivas.domain.enums import BetOutcome, FixtureStatus, Market, Selection
from deportivas.domain.ids import deterministic_id
from deportivas.features.asof import load_fixtures
from deportivas.odds.resolve import line_mask
from deportivas.storage.factory import get_table_repository


def signal_outcome(
    market: str, selection: str, line: float | None, *, home_score: float, away_score: float
) -> BetOutcome:
    if market == Market.ONE_X_TWO.value:
        if home_score > away_score:
            result = Selection.HOME.value
        elif home_score < away_score:
            result = Selection.AWAY.value
        else:
            result = Selection.DRAW.value
        return BetOutcome.WIN if selection == result else BetOutcome.LOSS
    if market == Market.BTTS.value:
        both_scored = home_score >= 1 and away_score >= 1
        wants_yes = selection == Selection.YES.value
        return BetOutcome.WIN if wants_yes == both_scored else BetOutcome.LOSS
    if market == Market.OVER_UNDER.value:
        if line is None:  # pragma: no cover - defensivo, over_under siempre trae linea
            raise ValueError("over_under requiere una linea")
        total = home_score + away_score
        if total == line:
            return BetOutcome.PUSH
        wants_over = selection == Selection.OVER.value
        return BetOutcome.WIN if wants_over == (total > line) else BetOutcome.LOSS
    if market == Market.MONEYLINE.value:
        if home_score == away_score:
            return BetOutcome.PUSH
        result = Selection.HOME.value if home_score > away_score else Selection.AWAY.value
        return BetOutcome.WIN if selection == result else BetOutcome.LOSS
    raise ValueError(f"mercado no soportado para liquidacion: {market!r}")  # pragma: no cover


def pnl_for_outcome(outcome: BetOutcome, *, stake_fraction: float, entry_price: float) -> float:
    if outcome == BetOutcome.WIN:
        return stake_fraction * (entry_price - 1.0)
    if outcome == BetOutcome.LOSS:
        return -stake_fraction
    if outcome == BetOutcome.HALF_WIN:
        return stake_fraction * (entry_price - 1.0) / 2.0
    if outcome == BetOutcome.HALF_LOSS:
        return -stake_fraction / 2.0
    return 0.0  # PUSH, VOID: el stake se devuelve intacto


def closing_price(
    odds: pd.DataFrame,
    *,
    fixture_id: str,
    bookmaker: str,
    market: str,
    selection: str,
    line: float | None,
    kickoff_utc: datetime,
) -> tuple[float, datetime] | None:
    same = odds[
        (odds["fixture_id"] == fixture_id)
        & (odds["bookmaker"] == bookmaker)
        & (odds["market"] == market)
        & (odds["selection"] == selection)
        & line_mask(odds["line"], line)
        & (odds["captured_at"] <= kickoff_utc)
    ]
    if same.empty:
        return None
    closing = same[same["is_closing"]]
    candidates = closing if not closing.empty else same
    row = candidates.loc[candidates["captured_at"].idxmax()]
    return float(row["price"]), cast(datetime, row["captured_at"])  # type: ignore[arg-type]


def compute_and_write_results(competition_id: str) -> int:
    """Settles every ``signals`` row for ``competition_id`` whose fixture has
    finished with a known score, writing one row to ``results`` each.
    Idempotent: ``results.write`` upserts on ``signal_id``, so re-running
    after new closing odds arrive just updates ``closing_price``/``clv``
    rather than duplicating the row. Returns the number of rows written (0
    when there are no signals, or none of them have a settled fixture yet)."""
    signals = get_table_repository(SIGNALS).read(filters={"competition_id": competition_id})
    if signals.empty:
        return 0

    fixtures = load_fixtures(competition_id)
    fixtures_by_id = {str(row["id"]): row for row in fixtures.to_dict("records")}
    odds = get_table_repository(ODDS_SNAPSHOTS).read(filters={"competition_id": competition_id})

    now = datetime.now(UTC)
    rows: list[dict[str, object]] = []
    for record in signals.to_dict("records"):
        fixture_id = str(record["fixture_id"])
        fixture = fixtures_by_id.get(fixture_id)
        if fixture is None or fixture["status"] != FixtureStatus.FINISHED.value:
            continue
        home_score, away_score = fixture["home_score"], fixture["away_score"]
        if pd.isna(home_score) or pd.isna(away_score):
            continue  # datos incompletos: "finished" sin marcador, no se puede liquidar

        market = str(record["market"])
        selection = str(record["selection"])
        line = float(record["line"]) if pd.notna(record["line"]) else None
        outcome = signal_outcome(
            market, selection, line, home_score=float(home_score), away_score=float(away_score)
        )
        entry_price = float(record["entry_price"])
        pnl = pnl_for_outcome(
            outcome, stake_fraction=float(record["stake_fraction"]), entry_price=entry_price
        )
        closing = closing_price(
            odds,
            fixture_id=fixture_id,
            bookmaker=str(record["entry_bookmaker"]),
            market=market,
            selection=selection,
            line=line,
            kickoff_utc=cast(datetime, fixture["kickoff_utc"]),
        )
        closing_price_value, closing_captured_at = closing if closing is not None else (None, None)
        clv = (entry_price / closing_price_value - 1.0) if closing_price_value is not None else None

        signal_id = str(record["id"])
        rows.append(
            {
                "id": deterministic_id(signal_id),
                "signal_id": signal_id,
                "fixture_id": fixture_id,
                "competition_id": str(record["competition_id"]),
                "season": str(record["season"]),
                "outcome": outcome.value,
                "pnl": pnl,
                "closing_price": closing_price_value,
                "closing_captured_at": closing_captured_at,
                "clv": clv,
                "settled_at": now,
                "source": "backtest_settlement",
                "ingested_at": now,
            }
        )

    if not rows:
        return 0
    results_repo = get_table_repository(RESULTS, temporal_column="settled_at")
    return results_repo.write(pd.DataFrame(rows))
