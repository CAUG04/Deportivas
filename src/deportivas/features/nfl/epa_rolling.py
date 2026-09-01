"""Rolling offensive/defensive EPA-per-play and success rate, walk-forward.

Same shape as football's ``xg_rolling.py``: each team's own history (EPA per
play and success rate, offense and defense) is built up incrementally as
fixtures are processed in kickoff order, and a game only enters a team's
history once its ``nfl_team_game_stats`` row actually has offensive plays —
a game not yet ingested for that team is silently skipped rather than
poisoning the average with a missing value treated as zero.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from datetime import datetime

_TeamGameRecord = tuple[float, float, float, float]  # off_epa, off_success, def_epa, def_success


@dataclass(frozen=True, slots=True)
class EpaRollingConfig:
    windows: tuple[int, ...] = (4, 8, 16)
    decay: float = 0.9  # peso multiplicativo por partido mas antiguo


def _weighted_avg(values_most_recent_first: list[float], decay: float) -> float | None:
    if not values_most_recent_first:
        return None
    weights = [decay**i for i in range(len(values_most_recent_first))]
    return sum(w * v for w, v in zip(weights, values_most_recent_first, strict=True)) / sum(weights)


def compute_epa_rolling(
    fixtures: pd.DataFrame,
    team_game_stats: pd.DataFrame,
    *,
    config: EpaRollingConfig | None = None,
) -> pd.DataFrame:
    """``fixtures`` sorted by kickoff ascending. ``team_game_stats`` one row
    per (fixture_id, team_id) — see ``nfl_team_game_stats``.
    """
    config = config if config is not None else EpaRollingConfig()
    stats_by_fixture_team = {
        (row["fixture_id"], row["team_id"]): row
        for row in team_game_stats.to_dict("records")
        if row.get("offensive_plays", 0)
    }

    history: dict[str, list[_TeamGameRecord]] = defaultdict(list)
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
                vector[f"off_epa_rolling_{window}_{side}"] = _weighted_avg(
                    [v[0] for v in window_slice], config.decay
                )
                vector[f"off_success_rolling_{window}_{side}"] = _weighted_avg(
                    [v[1] for v in window_slice], config.decay
                )
                vector[f"def_epa_allowed_rolling_{window}_{side}"] = _weighted_avg(
                    [v[2] for v in window_slice], config.decay
                )
                vector[f"def_success_allowed_rolling_{window}_{side}"] = _weighted_avg(
                    [v[3] for v in window_slice], config.decay
                )

        rows.append({"fixture_id": record["id"], "as_of_timestamp": as_of, "vector": vector})

        for team in (home, away):
            game_stats = stats_by_fixture_team.get((record["id"], team))
            if game_stats is not None:
                history[team].append(
                    (
                        game_stats["offensive_epa_per_play"],
                        game_stats["offensive_success_rate"],
                        game_stats["defensive_epa_per_play_allowed"],
                        game_stats["defensive_success_rate_allowed"],
                    )
                )
        last_played[home] = kickoff
        last_played[away] = kickoff

    return pd.DataFrame(rows)
