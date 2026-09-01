"""Rolling xG for/against over the last 5/10/20 matches, exponentially
decayed toward the most recent game, walk-forward.

Each team's own match history (kickoff, xg scored, xg conceded) is built up
incrementally as fixtures are processed in order — the same pattern as
``elo.py`` and ``rest_congestion.py``. A match only enters a team's history
once *both* sides have recorded xG for it (``asof.load_team_match_stats``
already picked one source per match); if a fixture never got an xG estimate
from any source, it is silently skipped for this feature rather than
poisoning the rolling average with a missing value treated as zero.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True, slots=True)
class XgRollingConfig:
    windows: tuple[int, ...] = (5, 10, 20)
    decay: float = 0.9  # peso multiplicativo por partido mas antiguo


def _weighted_avg(values_most_recent_first: list[float], decay: float) -> float | None:
    if not values_most_recent_first:
        return None
    weights = [decay**i for i in range(len(values_most_recent_first))]
    return sum(w * v for w, v in zip(weights, values_most_recent_first, strict=True)) / sum(weights)


def compute_xg_rolling(
    fixtures: pd.DataFrame,
    team_match_stats: pd.DataFrame,
    *,
    config: XgRollingConfig | None = None,
) -> pd.DataFrame:
    """``fixtures`` sorted by kickoff ascending. ``team_match_stats`` already
    deduplicated to one row per (fixture_id, team_id) — see ``asof.load_team_match_stats``.
    """
    config = config if config is not None else XgRollingConfig()
    xg_by_fixture_team = {
        (row["fixture_id"], row["team_id"]): row["xg"]
        for row in team_match_stats.to_dict("records")
        if pd.notna(row["xg"])
    }

    # history[team] = [(xg_for, xg_against), ...] en orden cronologico (mas antiguo primero)
    history: dict[str, list[tuple[float, float]]] = defaultdict(list)
    last_played: dict[str, datetime] = {}
    rows: list[dict[str, object]] = []

    for record in fixtures.to_dict("records"):
        home, away = record["home_team_id"], record["away_team_id"]
        kickoff = record["kickoff_utc"]

        candidates = [ts for ts in (last_played.get(home), last_played.get(away)) if ts is not None]
        as_of = max(candidates) if candidates else kickoff - timedelta(seconds=1)

        vector: dict[str, object] = {}
        for side, team in (("home", home), ("away", away)):
            recent_first = list(reversed(history[team]))
            for window in config.windows:
                window_slice = recent_first[:window]
                vector[f"xg_for_rolling_{window}_{side}"] = _weighted_avg(
                    [v[0] for v in window_slice], config.decay
                )
                vector[f"xg_against_rolling_{window}_{side}"] = _weighted_avg(
                    [v[1] for v in window_slice], config.decay
                )

        rows.append({"fixture_id": record["id"], "as_of_timestamp": as_of, "vector": vector})

        home_xg = xg_by_fixture_team.get((record["id"], home))
        away_xg = xg_by_fixture_team.get((record["id"], away))
        if home_xg is not None and away_xg is not None:
            history[home].append((home_xg, away_xg))
            history[away].append((away_xg, home_xg))
        last_played[home] = kickoff
        last_played[away] = kickoff

    return pd.DataFrame(rows)
