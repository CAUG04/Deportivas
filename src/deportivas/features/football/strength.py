"""Dixon-Coles-style attack/defense strength ratings, walk-forward.

This is the well-known "Dixon-Coles via GLM" trick: fit
``log(E[goals]) = home_advantage*is_home + attack[scoring_team] -
defense[conceding_team]`` as a Poisson GLM over every match so far, with each
match contributing two observations (one for each side's goals). The
attack/defense numbers this produces are a *feature* — a compact, fitted
summary of each team's current strength — not a prediction; Fase 3's actual
Poisson model may fit its own version for that purpose.

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
import statsmodels.api as sm

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True, slots=True)
class StrengthConfig:
    min_matches_to_fit: int = 20
    refit_every: int = 10


@dataclass(frozen=True, slots=True)
class _FittedStrength:
    attack: dict[str, float]
    defense: dict[str, float]


def _fit(finished: list[dict[str, object]]) -> _FittedStrength | None:
    # 2 observaciones (local+visitante) por partido: con menos de 2 partidos
    # no hay suficientes grados de libertad para nada util. En la practica el
    # umbral por defecto de StrengthConfig lo evita, pero un min_matches_to_fit
    # bajo (1) puede llegar aqui.
    if len(finished) < 2:
        return None

    teams = sorted(
        {str(m["home_team_id"]) for m in finished} | {str(m["away_team_id"]) for m in finished}
    )
    observations = []
    for match in finished:
        observations.append(
            {
                "goals": match["home_score"],
                "is_home": 1,
                "attack": match["home_team_id"],
                "defense": match["away_team_id"],
            }
        )
        observations.append(
            {
                "goals": match["away_score"],
                "is_home": 0,
                "attack": match["away_team_id"],
                "defense": match["home_team_id"],
            }
        )
    obs = pd.DataFrame(observations)
    obs["attack"] = pd.Categorical(obs["attack"], categories=teams)
    obs["defense"] = pd.Categorical(obs["defense"], categories=teams)

    attack_dummies = pd.get_dummies(obs["attack"], prefix="atk", drop_first=True)
    defense_dummies = pd.get_dummies(obs["defense"], prefix="def", drop_first=True)
    design = pd.concat([obs[["is_home"]], attack_dummies, defense_dummies], axis=1).astype(float)
    design = sm.add_constant(design, has_constant="add")

    model = sm.GLM(obs["goals"].astype(float), design, family=sm.families.Poisson())
    try:
        result = model.fit()
    except (
        ValueError,
        sm.tools.sm_exceptions.PerfectSeparationError,
    ):  # pragma: no cover - defensivo
        return None

    reference_team = teams[0]
    attack = {reference_team: 0.0}
    defense = {reference_team: 0.0}
    for team in teams[1:]:
        attack[team] = float(result.params.get(f"atk_{team}", 0.0))
        defense[team] = float(result.params.get(f"def_{team}", 0.0))
    return _FittedStrength(attack=attack, defense=defense)


def compute_strength(
    fixtures: pd.DataFrame, *, config: StrengthConfig | None = None
) -> pd.DataFrame:
    """``fixtures`` sorted by kickoff ascending, columns id, home_team_id,
    away_team_id, kickoff_utc, status, home_score, away_score.
    """
    config = config if config is not None else StrengthConfig()
    finished_so_far: list[dict[str, object]] = []
    fitted: _FittedStrength | None = None
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
                new_fit = _fit(finished_so_far)
                if new_fit is not None:
                    fitted = new_fit
                    matches_since_fit = 0

        last_played[home] = kickoff
        last_played[away] = kickoff

    return pd.DataFrame(rows)
