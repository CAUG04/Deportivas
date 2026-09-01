"""Framework-agnostic view layer — pure functions that build read-only,
JSON-serializable models straight from the repositories. Both the live API
(:mod:`deportivas.api.app`, FastAPI) and the static export
(:mod:`deportivas.export.json_export`, what the $0 static frontend of Fase 7
actually reads — see ``Settings.export_dir``'s own docstring) call these
same functions, so neither surface can drift from the other: whatever the
API would answer live is exactly what the export would have written
offline, because it is the same code building the same models.

An unknown ``competition_id`` raises ``KeyError`` (from
``config.catalog``'s own ``Competition`` lookup) rather than silently
returning empty data — a public-facing read surface should say "that
competition doesn't exist" clearly, not look identical to "that competition
has no signals yet".
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from pydantic import BaseModel

from deportivas.backtest.report import BacktestReport, MetricSummary, build_backtest_report
from deportivas.config.catalog import load_competitions
from deportivas.contracts.tables import FIXTURES, SIGNALS, TEAMS
from deportivas.domain.enums import Tier
from deportivas.storage.factory import get_table_repository


class CompetitionOut(BaseModel):
    id: str
    name: str
    country: str
    sport: str
    tier: int


def list_competitions() -> list[CompetitionOut]:
    return [
        CompetitionOut(id=c.id, name=c.name, country=c.country, sport=c.sport.value, tier=c.tier)
        for c in load_competitions().enabled
    ]


class SignalOut(BaseModel):
    id: str
    fixture_id: str
    home_team: str
    away_team: str
    kickoff_utc: datetime
    market: str
    selection: str
    line: float | None
    model_name: str
    model_version: str
    prob_model: float
    prob_fair: float
    fair_price: float
    entry_price: float
    entry_bookmaker: str
    entry_captured_at: datetime
    edge: float
    tier: str
    tier_reasons: dict[str, bool]
    stake_fraction: float
    created_at: datetime


def _team_names() -> dict[str, str]:
    """Every team, id -> canonical name. Not scoped by competition: a
    competition's fixtures already restrict which team ids come up, and
    ``teams`` itself carries no ``competition_id`` to filter by (a team can
    play in more than one)."""
    teams = get_table_repository(TEAMS).read()
    return dict(zip(teams["id"], teams["canonical_name"], strict=True))


def list_signals(
    competition_id: str,
    *,
    tier: str | None = None,
    market: str | None = None,
    only_actionable: bool = True,
    limit: int = 200,
) -> list[SignalOut]:
    """Signals for ``competition_id``, newest entry first, enriched with the
    fixture's team names and kickoff so a consumer never has to join back to
    ``fixtures``/``teams`` itself. ``only_actionable`` (default ``True``)
    keeps only ``alta``/``media`` — a specific ``tier`` overrides it, since
    asking for ``descartar`` explicitly is itself a meaningful query (the
    project's own "knowing what not to bet on" transparency)."""
    load_competitions().get(competition_id)  # KeyError si no existe

    signals = get_table_repository(SIGNALS).read(filters={"competition_id": competition_id})
    if tier is not None:
        signals = signals[signals["tier"] == tier]
    elif only_actionable:
        signals = signals[signals["tier"].isin([Tier.ALTA.value, Tier.MEDIA.value])]
    if market is not None:
        signals = signals[signals["market"] == market]
    if signals.empty:
        return []

    fixtures = get_table_repository(FIXTURES).read(filters={"competition_id": competition_id})
    fixtures_by_id = {str(row["id"]): row for row in fixtures.to_dict("records")}
    team_names = _team_names()

    signals = signals.sort_values("entry_captured_at", ascending=False).head(limit)

    out = []
    for record in signals.to_dict("records"):
        fixture = fixtures_by_id.get(str(record["fixture_id"]))
        if fixture is None:
            continue  # datos huerfanos: la senal referencia un fixture que ya no esta
        home_team_id = str(fixture["home_team_id"])
        away_team_id = str(fixture["away_team_id"])
        out.append(
            SignalOut(
                id=str(record["id"]),
                fixture_id=str(record["fixture_id"]),
                home_team=team_names.get(home_team_id, home_team_id),
                away_team=team_names.get(away_team_id, away_team_id),
                kickoff_utc=fixture["kickoff_utc"],
                market=str(record["market"]),
                selection=str(record["selection"]),
                line=float(record["line"]) if pd.notna(record["line"]) else None,
                model_name=str(record["model_name"]),
                model_version=str(record["model_version"]),
                prob_model=float(record["prob_model"]),
                prob_fair=float(record["prob_fair"]),
                fair_price=float(record["fair_price"]),
                entry_price=float(record["entry_price"]),
                entry_bookmaker=str(record["entry_bookmaker"]),
                entry_captured_at=record["entry_captured_at"],
                edge=float(record["edge"]),
                tier=str(record["tier"]),
                tier_reasons=json.loads(str(record["tier_reasons"])),
                stake_fraction=float(record["stake_fraction"]),
                created_at=record["created_at"],
            )
        )
    return out


class MetricSummaryOut(BaseModel):
    n: int
    mean_clv: float | None
    clv_ci_low: float | None
    clv_ci_high: float | None
    mean_pnl: float
    roi: float | None


def _metric_summary_out(summary: MetricSummary) -> MetricSummaryOut:
    ci_low, ci_high = summary.clv_ci if summary.clv_ci is not None else (None, None)
    return MetricSummaryOut(
        n=summary.n,
        mean_clv=summary.mean_clv,
        clv_ci_low=ci_low,
        clv_ci_high=ci_high,
        mean_pnl=summary.mean_pnl,
        roi=summary.roi,
    )


class BacktestReportOut(BaseModel):
    overall: MetricSummaryOut
    by_tier: dict[str, MetricSummaryOut]
    by_market: dict[str, MetricSummaryOut]
    baselines: dict[str, MetricSummaryOut]


def backtest_report(competition_id: str) -> BacktestReportOut:
    load_competitions().get(competition_id)  # KeyError si no existe
    report: BacktestReport = build_backtest_report(competition_id)
    return BacktestReportOut(
        overall=_metric_summary_out(report.overall),
        by_tier={k: _metric_summary_out(v) for k, v in report.by_tier.items()},
        by_market={k: _metric_summary_out(v) for k, v in report.by_market.items()},
        baselines={k: _metric_summary_out(v) for k, v in report.baselines.items()},
    )
