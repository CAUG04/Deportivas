"""Combines every NFL feature module into one per-fixture vector.

``epa_rolling`` and ``rest`` both derive ``as_of_timestamp`` the same way —
the kickoff of the most recent prior game either side played — over the same
walk-forward pass across the same fixtures, so for a given fixture they
agree exactly; the merged vector keeps that single value. ``dvoa_approx`` is
a pure transform of ``epa_rolling``'s own output and carries the same
``as_of_timestamp`` through unchanged.
"""

from __future__ import annotations

import pandas as pd

from deportivas.features.asof import load_fixtures, load_nfl_team_game_stats
from deportivas.features.merge import merge_vectors
from deportivas.features.nfl.dvoa_approx import compute_dvoa_approx
from deportivas.features.nfl.epa_rolling import compute_epa_rolling
from deportivas.features.nfl.rest import compute_rest_days
from deportivas.features.writer import write_features

FEATURE_SET = "nfl_v1"


def compute_nfl_features(fixtures: pd.DataFrame, team_game_stats: pd.DataFrame) -> pd.DataFrame:
    """``fixtures`` sorted by kickoff ascending, with the columns every module
    in this package needs (see each one's docstring), plus ``competition_id``
    and ``season`` for the output rows. Returns one row per fixture, ready for
    ``write_features``: fixture_id, competition_id, season, as_of_timestamp, vector.
    """
    epa_rolling = compute_epa_rolling(fixtures, team_game_stats)
    rest = compute_rest_days(fixtures)
    dvoa_approx = compute_dvoa_approx(epa_rolling)

    merged = merge_vectors(epa_rolling, rest, dvoa_approx)
    fixture_meta = fixtures[["id", "competition_id", "season"]].rename(columns={"id": "fixture_id"})
    return merged.merge(fixture_meta, on="fixture_id")


def compute_and_write_nfl_features(competition_id: str) -> int:
    """Loads a competition's fixtures and stats, computes the combined NFL
    feature vector, and persists it under ``FEATURE_SET``."""
    fixtures = load_fixtures(competition_id)
    team_game_stats = load_nfl_team_game_stats(competition_id)
    rows = compute_nfl_features(fixtures, team_game_stats)
    return write_features(FEATURE_SET, rows, fixtures.rename(columns={"id": "fixture_id"}))
