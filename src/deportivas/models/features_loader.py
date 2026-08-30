"""Loads a sport's already-computed, point-in-time-safe feature vectors (the
``features`` table) joined with each fixture's actual result — the shared
starting point for every feature-based model (moneyline classifiers, later
margin regressions) in this package.

Football's Poisson model doesn't use this: it fits its own attack/defense
coefficients directly from match scores rather than reading a feature
vector (see ``models/football/train.py``'s own docstring).
"""

from __future__ import annotations

import json

import pandas as pd

from deportivas.contracts.tables import FEATURES
from deportivas.features.asof import load_fixtures
from deportivas.storage.factory import get_table_repository


def load_feature_vectors(competition_id: str, feature_set: str) -> pd.DataFrame:
    """One row per fixture that has a computed ``feature_set`` vector,
    sorted by kickoff: every ``fixtures`` column plus ``as_of_timestamp``
    and ``vector`` (parsed back into a plain dict) from ``features``.

    A fixture with no computed vector for this ``feature_set`` is dropped
    (inner join) — nothing to feed a model with.
    """
    fixtures = load_fixtures(competition_id)
    features_repo = get_table_repository(FEATURES)
    features = features_repo.read(
        filters={"competition_id": competition_id, "feature_set": feature_set}
    )
    merged = fixtures.merge(
        features[["fixture_id", "as_of_timestamp", "vector"]],
        left_on="id",
        right_on="fixture_id",
        how="inner",
    )
    merged["vector"] = merged["vector"].map(json.loads)
    return merged.sort_values("kickoff_utc", kind="stable", ignore_index=True)
