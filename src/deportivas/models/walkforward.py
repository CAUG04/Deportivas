"""Season-level walk-forward windows — project rule #3: "Validación
walk-forward. Nada de train_test_split aleatorio. Entrena en temporadas
1..N, valida en N+1, avanza. Métricas por ventana."

Deliberately mechanical and sport-agnostic: this only knows how to order
seasons and slice them into windows. Whether a given window actually has
enough finished matches to train on (``config/thresholds.yaml``'s
``calibration.min_training_samples``) is a per-sport model trainer's call,
not this module's — a season with very few matches is still a real season
and belongs in the sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    train_seasons: tuple[str, ...]
    validate_season: str


def season_order(fixtures: pd.DataFrame) -> tuple[str, ...]:
    """Season labels ordered by their earliest kickoff — not lexically, so a
    future change to how seasons are labelled can never silently reorder
    training."""
    return tuple(fixtures.groupby("season")["kickoff_utc"].min().sort_values().index)


def walk_forward_windows(fixtures: pd.DataFrame) -> list[WalkForwardWindow]:
    """One window per season after the first: that season's predecessors are
    the training set, that season itself is validated against a model that
    never saw it."""
    seasons = season_order(fixtures)
    return [
        WalkForwardWindow(train_seasons=seasons[:i], validate_season=seasons[i])
        for i in range(1, len(seasons))
    ]
