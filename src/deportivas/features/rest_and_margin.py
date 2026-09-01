"""Rest days, back-to-back flag, and rolling scoring margin — shared across
every sport whose only ingested per-game signal is the final score
(``fixtures.home_score``/``away_score``): NBA, NHL and MLB today. Football
and NFL have their own richer, per-sport feature sets built from team-level
match stats and play-by-play, so they don't use this module; this is what
"alcance controlado" looks like for a sport with schedule-only ingestion —
a genuinely computed, walk-forward-safe signal, just a simpler one than a
possession-adjusted net rating would be.

Same walk-forward shape as ``football/rest_congestion.py`` and
``football/xg_rolling.py``: both come from the same per-team history, built
up as fixtures are processed in kickoff order.
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
class RestAndMarginConfig:
    windows: tuple[int, ...] = (5, 10, 20)
    decay: float = 0.9  # peso multiplicativo por partido mas antiguo
    back_to_back_max_rest_days: int = 1  # rest_days <= esto cuenta como back-to-back


def _weighted_avg(values_most_recent_first: list[float], decay: float) -> float | None:
    if not values_most_recent_first:
        return None
    weights = [decay**i for i in range(len(values_most_recent_first))]
    return sum(w * v for w, v in zip(weights, values_most_recent_first, strict=True)) / sum(weights)


def compute_rest_and_scoring_margin(
    fixtures: pd.DataFrame, *, config: RestAndMarginConfig | None = None
) -> pd.DataFrame:
    """``fixtures`` sorted by kickoff ascending, columns id, home_team_id,
    away_team_id, kickoff_utc, status, home_score, away_score.

    Returns one row per fixture: fixture_id, as_of_timestamp, and a vector
    with each side's rest days, whether that counts as a back-to-back, and
    its rolling scoring margin (points/goals/runs for minus against) over
    each configured trailing window.
    """
    config = config if config is not None else RestAndMarginConfig()
    last_played: dict[str, datetime] = {}
    margin_history: dict[str, list[float]] = defaultdict(list)
    rows: list[dict[str, object]] = []

    for record in fixtures.to_dict("records"):
        home, away = record["home_team_id"], record["away_team_id"]
        kickoff = record["kickoff_utc"]
        last_home = last_played.get(home)
        last_away = last_played.get(away)

        candidates = [ts for ts in (last_home, last_away) if ts is not None]
        as_of = max(candidates) if candidates else kickoff - timedelta(seconds=1)

        rest_days_home = (kickoff - last_home).days if last_home is not None else None
        rest_days_away = (kickoff - last_away).days if last_away is not None else None

        vector: dict[str, object] = {
            "rest_days_home": rest_days_home,
            "rest_days_away": rest_days_away,
            "back_to_back_home": rest_days_home is not None
            and rest_days_home <= config.back_to_back_max_rest_days,
            "back_to_back_away": rest_days_away is not None
            and rest_days_away <= config.back_to_back_max_rest_days,
        }
        for side, team in (("home", home), ("away", away)):
            recent_first = list(reversed(margin_history[team]))
            for window in config.windows:
                vector[f"scoring_margin_rolling_{window}_{side}"] = _weighted_avg(
                    recent_first[:window], config.decay
                )

        rows.append({"fixture_id": record["id"], "as_of_timestamp": as_of, "vector": vector})

        home_score, away_score = record.get("home_score"), record.get("away_score")
        is_finished = (
            record.get("status") == "finished" and pd.notna(home_score) and pd.notna(away_score)
        )
        if is_finished:
            home_points, away_points = float(home_score), float(away_score)  # type: ignore[arg-type]
            margin_history[home].append(home_points - away_points)
            margin_history[away].append(away_points - home_points)

        last_played[home] = kickoff
        last_played[away] = kickoff

    return pd.DataFrame(rows)
