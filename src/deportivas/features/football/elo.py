"""Elo ratings with a home-advantage adjustment, walk-forward.

Standard chess-style Elo, adapted for football: home advantage is added to
the home team's effective rating before computing the expected result, and
the K-factor update treats a draw as each side "winning half a point". Only
finished matches update state; a scheduled fixture still gets a snapshot of
both teams' current ratings (useful for pre-match signals) but never moves
the ratings themselves — that would leak the very result being predicted.

Every row's ``as_of`` is the kickoff of the most recent prior match either
team played (or, for a side's tournament debut, its own kickoff minus one
second — the tightest valid bound when literally nothing is known yet).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True, slots=True)
class EloConfig:
    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 60.0


def expected_home_win_prob(
    home_rating: float, away_rating: float, *, home_advantage: float
) -> float:
    diff = (home_rating + home_advantage) - away_rating
    return float(1.0 / (1.0 + 10.0 ** (-diff / 400.0)))


def compute_elo(fixtures: pd.DataFrame, *, config: EloConfig | None = None) -> pd.DataFrame:
    """``fixtures`` must be sorted by kickoff ascending, with columns id,
    home_team_id, away_team_id, kickoff_utc, status, home_score, away_score.

    Returns one row per fixture: fixture_id, as_of_timestamp, and a vector
    with each team's pre-match rating, the (home-advantage-adjusted)
    difference, and the resulting home win probability.
    """
    config = config if config is not None else EloConfig()
    ratings: dict[str, float] = {}
    last_played: dict[str, datetime] = {}
    rows: list[dict[str, object]] = []

    for record in fixtures.to_dict("records"):
        home, away = record["home_team_id"], record["away_team_id"]
        kickoff = record["kickoff_utc"]
        home_rating = ratings.get(home, config.initial_rating)
        away_rating = ratings.get(away, config.initial_rating)
        prob = expected_home_win_prob(
            home_rating, away_rating, home_advantage=config.home_advantage
        )

        candidates = [ts for ts in (last_played.get(home), last_played.get(away)) if ts is not None]
        as_of = max(candidates) if candidates else kickoff - timedelta(seconds=1)

        rows.append(
            {
                "fixture_id": record["id"],
                "as_of_timestamp": as_of,
                "vector": {
                    "elo_home": home_rating,
                    "elo_away": away_rating,
                    "elo_diff_adjusted": (home_rating + config.home_advantage) - away_rating,
                    "elo_home_win_prob": prob,
                },
            }
        )

        home_score, away_score = record.get("home_score"), record.get("away_score")
        is_finished = (
            record.get("status") == "finished" and pd.notna(home_score) and pd.notna(away_score)
        )
        if is_finished:
            home_goals, away_goals = float(home_score), float(away_score)  # type: ignore[arg-type]
            if home_goals > away_goals:
                actual_home = 1.0
            elif home_goals < away_goals:
                actual_home = 0.0
            else:
                actual_home = 0.5
            ratings[home] = home_rating + config.k_factor * (actual_home - prob)
            ratings[away] = away_rating + config.k_factor * ((1.0 - actual_home) - (1.0 - prob))

        last_played[home] = kickoff
        last_played[away] = kickoff

    return pd.DataFrame(rows)
