"""Combines every football feature module into one per-fixture vector.

Elo, rest/congestion, xG rolling and Dixon-Coles strength all derive
``as_of_timestamp`` the same way — the kickoff of the most recent prior match
either side played — over the same walk-forward pass across the same
fixtures, so for a given fixture they agree exactly; the merged vector keeps
that single value. ``opponent_adjusted`` reuses ``strength``'s already-fitted
attack ratings and carries no ``as_of_timestamp`` of its own for that reason
(see ``opponent_adjusted.py``), so it only ever contributes vector keys here.
"""

from __future__ import annotations

import pandas as pd

from deportivas.features.asof import load_fixtures, load_team_match_stats
from deportivas.features.football.elo import compute_elo
from deportivas.features.football.opponent_adjusted import compute_opponent_adjusted_defense
from deportivas.features.football.rest_congestion import compute_rest_and_congestion
from deportivas.features.football.strength import compute_strength
from deportivas.features.football.xg_rolling import compute_xg_rolling
from deportivas.features.writer import write_features

FEATURE_SET = "football_v1"


def _merge_vectors(base: pd.DataFrame, *extras: pd.DataFrame) -> pd.DataFrame:
    """Left-to-right dict-merges each frame's ``vector`` column into ``base``'s,
    joined on ``fixture_id``. Every frame must cover exactly the same set of
    fixtures — true here since all five modules walk the same ``fixtures``."""
    merged = base[["fixture_id", "as_of_timestamp", "vector"]].copy()
    for extra in extras:
        merged = merged.merge(
            extra[["fixture_id", "vector"]], on="fixture_id", suffixes=("", "_extra")
        )
        merged["vector"] = pd.Series(
            [
                {**left, **right}
                for left, right in zip(merged["vector"], merged["vector_extra"], strict=True)
            ],
            index=merged.index,
        )
        merged = merged.drop(columns="vector_extra")
    return merged


def compute_football_features(
    fixtures: pd.DataFrame, team_match_stats: pd.DataFrame
) -> pd.DataFrame:
    """``fixtures`` sorted by kickoff ascending, with the columns every module
    in this package needs (see each one's docstring), plus ``competition_id``
    and ``season`` for the output rows. Returns one row per fixture, ready for
    ``write_features``: fixture_id, competition_id, season, as_of_timestamp, vector.
    """
    elo = compute_elo(fixtures)
    rest = compute_rest_and_congestion(fixtures)
    xg = compute_xg_rolling(fixtures, team_match_stats)
    strength = compute_strength(fixtures)
    opponent_adjusted = compute_opponent_adjusted_defense(fixtures, team_match_stats, strength)

    merged = _merge_vectors(elo, rest, xg, strength, opponent_adjusted)
    fixture_meta = fixtures[["id", "competition_id", "season"]].rename(columns={"id": "fixture_id"})
    return merged.merge(fixture_meta, on="fixture_id")


def compute_and_write_football_features(competition_id: str) -> int:
    """Loads a competition's fixtures and stats, computes the combined
    football feature vector, and persists it under ``FEATURE_SET``."""
    fixtures = load_fixtures(competition_id)
    team_match_stats = load_team_match_stats(competition_id)
    rows = compute_football_features(fixtures, team_match_stats)
    return write_features(FEATURE_SET, rows, fixtures.rename(columns={"id": "fixture_id"}))
