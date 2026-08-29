"""Shared low-level fit for the well-known "Dixon-Coles via GLM" trick: fit
``log(E[goals]) = intercept + home_advantage*is_home + attack[scorer] -
defense[concedor]`` as a Poisson GLM over a set of finished matches, two
observations per match (one for each side's goals).

Two different callers need this, at different altitudes:

* ``strength.py``'s walk-forward FEATURE only exposes attack/defense as a
  compact per-team rating, refit periodically — it doesn't need the
  intercept or home_advantage coefficients, since those describe the league
  as a whole rather than a single team.
* ``models/football/poisson.py``'s actual predictive MODEL needs every
  coefficient to reconstruct each side's expected goals (``lambda``) for a
  specific matchup, fit once per walk-forward training window rather than
  refit continuously.

Independent Poisson, not the full Dixon-Coles paper: no low-score
correlation adjustment (tau) for 0-0/1-0/0-1/1-1 — the same simplification
this module's callers already named honestly in their own docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True, slots=True)
class DixonColesGlmFit:
    attack: dict[str, float]
    defense: dict[str, float]
    home_advantage: float
    intercept: float


def fit_dixon_coles_glm(finished: list[dict[str, object]]) -> DixonColesGlmFit | None:
    # 2 observaciones (local+visitante) por partido: con menos de 2 partidos
    # no hay suficientes grados de libertad para nada util.
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
    return DixonColesGlmFit(
        attack=attack,
        defense=defense,
        home_advantage=float(result.params.get("is_home", 0.0)),
        intercept=float(result.params.get("const", 0.0)),
    )
