"""Backtest report — aggregates ``results`` into the metric this project is
built around: CLV (``entry_price / closing_price - 1``), not pnl. A single
lucky or unlucky bet can swing pnl either way; CLV asks a question that
resolves independently of variance — did the market itself move to agree
with the price we got, before the outcome was ever known. pnl/ROI are
reported alongside it, never in place of it.

``results`` carries neither ``tier`` nor ``market`` — only ``signals`` does
— so every summary here first joins the two on ``signal_id``. Reported
alongside the real strategy are the ``always_favourite``/``random``
baselines from :mod:`deportivas.backtest.baselines`, computed fresh (they
are never persisted): the real edge is only worth something if it beats
what a naive strategy would have done on the exact same bets.

A bucket's confidence interval on CLV is withheld (``None``) below
``backtest.min_matches_per_window`` settled bets — bootstrapping five data
points produces an interval that looks precise and isn't; this project
would rather show nothing than a false sense of confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from deportivas.backtest.baselines import compute_baseline_results
from deportivas.backtest.bootstrap import bootstrap_confidence_interval
from deportivas.config.catalog import BacktestConfig, load_thresholds
from deportivas.contracts.tables import RESULTS, SIGNALS
from deportivas.storage.factory import get_table_repository


@dataclass(frozen=True, slots=True)
class MetricSummary:
    n: int
    mean_clv: float | None
    clv_ci: tuple[float, float] | None
    mean_pnl: float
    roi: float | None


@dataclass(frozen=True, slots=True)
class BacktestReport:
    overall: MetricSummary
    by_tier: dict[str, MetricSummary]
    by_market: dict[str, MetricSummary]
    baselines: dict[str, MetricSummary]


def _settled_signals(competition_id: str) -> pd.DataFrame:
    """``results`` joined back to its originating ``signals`` row for
    ``tier``/``market``/``stake_fraction`` — columns ``results`` itself does
    not carry."""
    results = get_table_repository(RESULTS).read(filters={"competition_id": competition_id})
    if results.empty:
        return results
    signals = get_table_repository(SIGNALS).read(filters={"competition_id": competition_id})
    return results.merge(
        signals[["id", "tier", "market", "stake_fraction"]],
        left_on="signal_id",
        right_on="id",
        suffixes=("", "_signal"),
    )


def _summarize(df: pd.DataFrame, *, stake_col: str | None, config: BacktestConfig) -> MetricSummary:
    """``stake_col`` names the column to weight ROI by (the real strategy's
    ``stake_fraction``); ``None`` means a flat unit stake per bet (the
    baselines), where the mean pnl already *is* the per-bet ROI."""
    n = len(df)
    if n == 0:
        return MetricSummary(n=0, mean_clv=None, clv_ci=None, mean_pnl=0.0, roi=None)

    clv_values = df["clv"].dropna().tolist()
    mean_clv = float(sum(clv_values) / len(clv_values)) if clv_values else None
    clv_ci = None
    if len(clv_values) >= 2 and len(clv_values) >= config.min_matches_per_window:
        clv_ci = bootstrap_confidence_interval(
            clv_values,
            iterations=config.bootstrap_iterations,
            confidence_level=config.confidence_level,
        )

    mean_pnl = float(df["pnl"].mean())
    if stake_col is not None:
        total_stake = float(df[stake_col].sum())
        roi = float(df["pnl"].sum() / total_stake) if total_stake > 0 else None
    else:
        roi = mean_pnl
    return MetricSummary(n=n, mean_clv=mean_clv, clv_ci=clv_ci, mean_pnl=mean_pnl, roi=roi)


def _summarize_by(
    df: pd.DataFrame, column: str, *, stake_col: str, config: BacktestConfig
) -> dict[str, MetricSummary]:
    if df.empty:
        return {}
    return {
        str(key): _summarize(group, stake_col=stake_col, config=config)
        for key, group in df.groupby(column)
    }


def build_backtest_report(competition_id: str) -> BacktestReport:
    """The full report for ``competition_id``: overall CLV/ROI, broken down
    by tier and by market, plus each configured baseline
    (``config/thresholds.yaml``'s ``backtest.baselines``) computed fresh over
    the same settled signals for comparison."""
    config = load_thresholds().backtest
    settled = _settled_signals(competition_id)

    overall = _summarize(settled, stake_col="stake_fraction", config=config)
    by_tier = _summarize_by(settled, "tier", stake_col="stake_fraction", config=config)
    by_market = _summarize_by(settled, "market", stake_col="stake_fraction", config=config)
    baselines = {
        name: _summarize(
            compute_baseline_results(competition_id, name), stake_col=None, config=config
        )
        for name in config.baselines
    }
    return BacktestReport(
        overall=overall, by_tier=by_tier, by_market=by_market, baselines=baselines
    )
