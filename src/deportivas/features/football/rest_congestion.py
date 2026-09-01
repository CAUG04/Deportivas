"""Days of rest and fixture congestion, walk-forward.

Both come from the same per-team match-history list, built up as fixtures
are processed in kickoff order — a team's own history never includes the
fixture currently being scored.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from datetime import datetime

DEFAULT_CONGESTION_WINDOW_DAYS = 14


def compute_rest_and_congestion(
    fixtures: pd.DataFrame, *, congestion_window_days: int = DEFAULT_CONGESTION_WINDOW_DAYS
) -> pd.DataFrame:
    """``fixtures`` sorted by kickoff ascending, columns id, home_team_id,
    away_team_id, kickoff_utc.

    Returns one row per fixture with rest days and match count in the
    trailing window for each side. ``rest_days`` is ``None`` for a team's
    first match in the dataset — there is nothing to measure it against.
    """
    history: dict[str, list[datetime]] = defaultdict(list)
    rows: list[dict[str, object]] = []

    for record in fixtures.to_dict("records"):
        home, away = record["home_team_id"], record["away_team_id"]
        kickoff = record["kickoff_utc"]
        home_hist = history[home]
        away_hist = history[away]

        last_home = home_hist[-1] if home_hist else None
        last_away = away_hist[-1] if away_hist else None
        candidates = [ts for ts in (last_home, last_away) if ts is not None]
        as_of = max(candidates) if candidates else kickoff - timedelta(seconds=1)

        window_start = kickoff - timedelta(days=congestion_window_days)
        rows.append(
            {
                "fixture_id": record["id"],
                "as_of_timestamp": as_of,
                "vector": {
                    "rest_days_home": (kickoff - home_hist[-1]).days if home_hist else None,
                    "rest_days_away": (kickoff - away_hist[-1]).days if away_hist else None,
                    "matches_last_window_home": sum(1 for ts in home_hist if ts >= window_start),
                    "matches_last_window_away": sum(1 for ts in away_hist if ts >= window_start),
                },
            }
        )

        home_hist.append(kickoff)
        away_hist.append(kickoff)

    return pd.DataFrame(rows)
