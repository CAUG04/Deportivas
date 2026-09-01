"""Days of rest since each side's previous game, walk-forward.

NFL plays roughly one game a week, so the "fixture congestion" signal used
for football (many matches in a trailing window) has no real analogue here —
what matters instead is the single gap since the last game, which already
captures a short week (Thursday game, ~4 days) or a bye (~13-14 days) on its
own. Built from the same per-team kickoff history as ``rest_congestion.py``.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from datetime import datetime


def compute_rest_days(fixtures: pd.DataFrame) -> pd.DataFrame:
    """``fixtures`` sorted by kickoff ascending, columns id, home_team_id,
    away_team_id, kickoff_utc.

    Returns one row per fixture with each side's rest days. ``rest_days`` is
    ``None`` for a team's first game in the dataset — there is nothing to
    measure it against.
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

        rows.append(
            {
                "fixture_id": record["id"],
                "as_of_timestamp": as_of,
                "vector": {
                    "rest_days_home": (kickoff - home_hist[-1]).days if home_hist else None,
                    "rest_days_away": (kickoff - away_hist[-1]).days if away_hist else None,
                },
            }
        )

        home_hist.append(kickoff)
        away_hist.append(kickoff)

    return pd.DataFrame(rows)
