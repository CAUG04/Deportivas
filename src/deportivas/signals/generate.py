"""Signal generation — joins ``predictions`` against observed ``odds_snapshots``,
removes the bookmaker's margin (:mod:`deportivas.signals.devig`), computes the
edge against the model's own probability, classifies a confidence tier
(:mod:`deportivas.signals.tiers`) and, for actionable tiers, a fractional
Kelly stake (:mod:`deportivas.signals.staking`). One row is written to
``signals`` per prediction with a resolved odds market — including
``tier == descartar`` rows, a deliberately-persisted outcome (see
``contracts/tables.py``'s ``SIGNALS`` docstring).

Point-in-time correctness bounds every price this module reads: a signal's
``entry_price`` is drawn only from snapshots captured at or after the
prediction's own ``as_of_timestamp`` — the information cutoff the model
actually used (rule 2), not ``predicted_at``, which is only the wall-clock
moment the training job happened to run and, for a historical walk-forward
window, is always "now" regardless of which season is being validated. Using
``predicted_at`` as the lower bound would make every historical prediction's
odds window empty and no backtest signal could ever be built. The upper bound
is the fixture's own kickoff, so no in-play price ever leaks in. Within that
window the earliest snapshot is the entry price; the earliest and the latest
together decide whether the market moved favourably before kickoff.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import pandas as pd

from deportivas.config.catalog import LineMoveConfig, ThresholdsCatalog, load_thresholds
from deportivas.contracts.tables import MODEL_REGISTRY, ODDS_SNAPSHOTS, PREDICTIONS, SIGNALS
from deportivas.domain.ids import deterministic_id
from deportivas.features.asof import load_fixtures
from deportivas.signals.devig import devig
from deportivas.signals.staking import kelly_stake_fraction
from deportivas.signals.tiers import TierInputs, classify_tier
from deportivas.storage.factory import get_table_repository


@dataclass(frozen=True, slots=True)
class _MarketSnapshot:
    captured_at: datetime
    prices: dict[str, float]  # selection -> cuota decimal, todas capturadas juntas


@dataclass(frozen=True, slots=True)
class _ResolvedMarket:
    bookmaker: str
    snapshots: list[_MarketSnapshot]  # cronologico, mas antiguo primero


def _market_snapshots(odds: pd.DataFrame) -> list[_MarketSnapshot]:
    """Groups one (fixture, bookmaker, market, line) slice of ``odds_snapshots``
    by ``captured_at`` — every selection of a market is scraped together in
    one pass, so rows sharing a timestamp are one snapshot. A moment with
    fewer than two priced selections carries no margin to remove and is not
    real market data, so it is dropped rather than devigged into a trivial
    (and wrong) probability of 1.0."""
    snapshots = [
        _MarketSnapshot(
            captured_at=cast(datetime, captured_at),
            prices=dict(zip(group["selection"], group["price"], strict=True)),
        )
        for captured_at, group in odds.groupby("captured_at")
    ]
    return sorted((s for s in snapshots if len(s.prices) >= 2), key=lambda s: s.captured_at)


def _line_mask(lines: pd.Series, line: float | None) -> pd.Series:
    if line is None:
        return lines.isna()
    return lines == line


def _resolve_market(
    odds: pd.DataFrame,
    *,
    fixture_id: str,
    market: str,
    selection: str,
    line: float | None,
    as_of_timestamp: datetime,
    kickoff_utc: datetime,
    reference_bookmaker: str,
    fallback_bookmakers: tuple[str, ...],
) -> _ResolvedMarket | None:
    """Tries ``reference_bookmaker`` first, then each fallback in order,
    returning the first one with at least one in-window snapshot that
    actually prices ``selection``. ``None`` when no bookmaker has anything
    usable — there is nothing honest to build a signal from."""
    same_market = odds[
        (odds["fixture_id"] == fixture_id)
        & (odds["market"] == market)
        & _line_mask(odds["line"], line)
        & (odds["captured_at"] >= as_of_timestamp)
        & (odds["captured_at"] <= kickoff_utc)
    ]
    for bookmaker in (reference_bookmaker, *fallback_bookmakers):
        snapshots = [
            snapshot
            for snapshot in _market_snapshots(same_market[same_market["bookmaker"] == bookmaker])
            if selection in snapshot.prices
        ]
        if snapshots:
            return _ResolvedMarket(bookmaker=bookmaker, snapshots=snapshots)
    return None


def _fair_probabilities(prices: dict[str, float], *, method: str) -> dict[str, float]:
    selections = list(prices)
    fair = devig([prices[s] for s in selections], method=method)
    return dict(zip(selections, fair, strict=True))


def _has_favourable_line_move(
    resolved: _ResolvedMarket, *, selection: str, method: str, config: LineMoveConfig
) -> bool:
    """The market's own fair price for ``selection`` must have dropped by at
    least ``favourable_drop`` between the entry and the latest snapshot —
    ``config/thresholds.yaml``'s own wording ("la cuota justa... ha bajado").
    Requires at least ``min_snapshots`` distinct moments; a single snapshot
    has no movement to measure."""
    if len(resolved.snapshots) < config.min_snapshots:
        return False
    entry_fair = _fair_probabilities(resolved.snapshots[0].prices, method=method)[selection]
    latest_fair = _fair_probabilities(resolved.snapshots[-1].prices, method=method)[selection]
    entry_fair_price = 1.0 / entry_fair
    latest_fair_price = 1.0 / latest_fair
    relative_drop = (entry_fair_price - latest_fair_price) / entry_fair_price
    return relative_drop >= config.favourable_drop


def _sample_matches_by_model(registry: pd.DataFrame) -> dict[tuple[str, str], int]:
    """``model_registry.metrics``'s ``n_train_matches``, keyed by (model_name,
    model_version) — the tier's ``min_sample_matches`` condition. There is no
    ``competition_id`` column on ``model_registry`` to filter by, so this
    reads the whole table once per run."""
    result: dict[tuple[str, str], int] = {}
    for record in registry.to_dict("records"):
        metrics = json.loads(str(record["metrics"]))
        n_train_matches = metrics.get("n_train_matches")
        if isinstance(n_train_matches, int):
            result[(str(record["model_name"]), str(record["model_version"]))] = n_train_matches
    return result


def _build_signal_row(
    *,
    fixture_id: str,
    competition_id: str,
    season: str,
    market: str,
    selection: str,
    line: float | None,
    model_name: str,
    model_version: str,
    prob_raw: float,
    prob_calibrated: float | None,
    as_of_timestamp: datetime,
    kickoff_utc: datetime,
    odds: pd.DataFrame,
    sample_matches: int,
    thresholds: ThresholdsCatalog,
    now: datetime,
) -> dict[str, object] | None:
    resolved = _resolve_market(
        odds,
        fixture_id=fixture_id,
        market=market,
        selection=selection,
        line=line,
        as_of_timestamp=as_of_timestamp,
        kickoff_utc=kickoff_utc,
        reference_bookmaker=thresholds.devig.reference_bookmaker,
        fallback_bookmakers=thresholds.devig.fallback_bookmakers,
    )
    if resolved is None:
        return None

    devig_method = thresholds.devig.method.value
    entry = resolved.snapshots[0]
    prob_fair = _fair_probabilities(entry.prices, method=devig_method)[selection]

    is_calibrated = prob_calibrated is not None
    prob_model = prob_calibrated if prob_calibrated is not None else prob_raw
    edge = prob_model - prob_fair

    tier_result = classify_tier(
        TierInputs(
            edge=edge,
            sample_matches=sample_matches,
            is_calibrated=is_calibrated,
            has_sharp_price=resolved.bookmaker == thresholds.devig.reference_bookmaker,
            has_favourable_line_move=_has_favourable_line_move(
                resolved, selection=selection, method=devig_method, config=thresholds.line_move
            ),
        ),
        thresholds.tiers,
    )

    entry_price = entry.prices[selection]
    stake_fraction = (
        kelly_stake_fraction(
            prob_model=prob_model, decimal_price=entry_price, config=thresholds.staking
        )
        if tier_result.tier.is_actionable
        else 0.0
    )

    return {
        "id": deterministic_id(fixture_id, market, selection, str(line), model_name, model_version),
        "fixture_id": fixture_id,
        "competition_id": competition_id,
        "season": season,
        "market": market,
        "selection": selection,
        "line": line,
        "model_name": model_name,
        "model_version": model_version,
        "prob_model": prob_model,
        "prob_fair": prob_fair,
        "fair_price": 1.0 / prob_fair,
        "entry_price": entry_price,
        "entry_bookmaker": resolved.bookmaker,
        "entry_captured_at": entry.captured_at,
        "devig_method": devig_method,
        "edge": edge,
        "tier": tier_result.tier.value,
        "tier_reasons": json.dumps(tier_result.reasons, sort_keys=True),
        "stake_fraction": stake_fraction,
        "created_at": now,
        "source": f"signals_{model_name}",
        "ingested_at": now,
    }


def compute_and_write_signals(competition_id: str) -> int:
    """Builds one signal per ``predictions`` row for ``competition_id`` that
    has a resolvable odds market, and writes them to ``signals``. Returns the
    number of rows written (0 when there are no predictions or no odds at
    all — there is nothing to short-circuit further than that)."""
    thresholds = load_thresholds()
    predictions = get_table_repository(PREDICTIONS).read(filters={"competition_id": competition_id})
    odds = get_table_repository(ODDS_SNAPSHOTS).read(filters={"competition_id": competition_id})
    if predictions.empty or odds.empty:
        return 0

    fixtures = load_fixtures(competition_id)
    kickoff_by_fixture: dict[str, datetime] = dict(
        zip(fixtures["id"], fixtures["kickoff_utc"], strict=True)
    )
    sample_matches_by_model = _sample_matches_by_model(get_table_repository(MODEL_REGISTRY).read())

    now = datetime.now(UTC)
    rows: list[dict[str, object]] = []
    for record in predictions.to_dict("records"):
        fixture_id = str(record["fixture_id"])
        kickoff_utc = kickoff_by_fixture.get(fixture_id)
        if kickoff_utc is None:
            continue  # fixture desconocida; no hay como acotar la ventana pre-kickoff

        model_name = str(record["model_name"])
        model_version = str(record["model_version"])
        row = _build_signal_row(
            fixture_id=fixture_id,
            competition_id=str(record["competition_id"]),
            season=str(record["season"]),
            market=str(record["market"]),
            selection=str(record["selection"]),
            line=float(record["line"]) if pd.notna(record["line"]) else None,
            model_name=model_name,
            model_version=model_version,
            prob_raw=float(record["prob_raw"]),
            prob_calibrated=(
                float(record["prob_calibrated"]) if pd.notna(record["prob_calibrated"]) else None
            ),
            as_of_timestamp=cast(datetime, record["as_of_timestamp"]),
            kickoff_utc=kickoff_utc,
            odds=odds,
            sample_matches=sample_matches_by_model.get((model_name, model_version), 0),
            thresholds=thresholds,
            now=now,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        return 0
    signals_repo = get_table_repository(SIGNALS, temporal_column="created_at")
    return signals_repo.write(pd.DataFrame(rows))
