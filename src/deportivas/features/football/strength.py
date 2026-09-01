"""Dixon-Coles-style attack/defense strength ratings, walk-forward.

The actual GLM fit (the well-known "Dixon-Coles via GLM" trick) lives in
``dixon_coles_glm.py``, shared with Fase 3's predictive Poisson model — this
module is the walk-forward wrapper around it: refit periodically over every
match so far, and expose only attack/defense as a compact per-team rating.
That rating is a *feature* — a summary of each team's current strength —
not itself a prediction; ``models/football/poisson.py`` is what turns a
matchup into goal probabilities.

Refitting a GLM before literally every single fixture would be correct but
wastefully slow for a full competition history, and adds essentially no
signal over refitting every few matches. So the model refits periodically
(``refit_every`` newly finished matches) rather than continuously; every
fixture between refits reads the most recently fitted ratings. Before enough
matches exist to fit at all (``min_matches_to_fit``), or for a team the
current fit has never seen (newly promoted, or simply no data yet), attack
and defense default to 0.0 — "average", in this model's log-scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import pandas as pd

from deportivas.features.football.dixon_coles_glm import DixonColesGlmFit, fit_dixon_coles_glm

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True, slots=True)
class StrengthConfig:
    min_matches_to_fit: int = 20
    refit_every: int = 10


def compute_strength(
    fixtures: pd.DataFrame, *, config: StrengthConfig | None = None
) -> pd.DataFrame:
    """``fixtures`` sorted by kickoff ascending, columns id, home_team_id,
    away_team_id, kickoff_utc, status, home_score, away_score.
    """
    config = config if config is not None else StrengthConfig()
    finished_so_far: list[dict[str, object]] = []
    fitted: DixonColesGlmFit | None = None
    matches_since_fit = 0
    last_played: dict[str, datetime] = {}
    rows: list[dict[str, object]] = []

    for record in fixtures.to_dict("records"):
        home, away = record["home_team_id"], record["away_team_id"]
        kickoff = record["kickoff_utc"]

        candidates = [ts for ts in (last_played.get(home), last_played.get(away)) if ts is not None]
        as_of = max(candidates) if candidates else kickoff - timedelta(seconds=1)

        attack = fitted.attack if fitted is not None else {}
        defense = fitted.defense if fitted is not None else {}
        rows.append(
            {
                "fixture_id": record["id"],
                "as_of_timestamp": as_of,
                "vector": {
                    "strength_attack_home": attack.get(home, 0.0),
                    "strength_defense_home": defense.get(home, 0.0),
                    "strength_attack_away": attack.get(away, 0.0),
                    "strength_defense_away": defense.get(away, 0.0),
                },
            }
        )

        home_score, away_score = record.get("home_score"), record.get("away_score")
        if record.get("status") == "finished" and pd.notna(home_score) and pd.notna(away_score):
            finished_so_far.append(
                {
                    "home_team_id": home,
                    "away_team_id": away,
                    "home_score": home_score,
                    "away_score": away_score,
                }
            )
            matches_since_fit += 1
            should_fit = fitted is None and len(finished_so_far) >= config.min_matches_to_fit
            should_refit = fitted is not None and matches_since_fit >= config.refit_every
            if should_fit or should_refit:
                new_fit = fit_dixon_coles_glm(finished_so_far)
                if new_fit is not None:
                    fitted = new_fit
                    matches_since_fit = 0

        last_played[home] = kickoff
        last_played[away] = kickoff

    return pd.DataFrame(rows)
