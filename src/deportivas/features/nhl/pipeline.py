"""NHL features: rest days, back-to-backs, and rolling scoring margin.

Same reasoning as ``nba/pipeline.py``: NHL's only ingested per-game signal is
the final score (``fixtures``, via ``sportsdataverse``), so this is the
honest, schedule-only substitute for a possession-adjusted rating. An
82-game season makes the default trailing windows (5/10/20) a reasonable
read on recent form.
"""

from __future__ import annotations

import pandas as pd

from deportivas.features.asof import load_fixtures
from deportivas.features.rest_and_margin import compute_rest_and_scoring_margin
from deportivas.features.writer import write_features

FEATURE_SET = "nhl_v1"


def compute_nhl_features(fixtures: pd.DataFrame) -> pd.DataFrame:
    """``fixtures`` sorted by kickoff ascending, with ``competition_id`` and
    ``season`` for the output rows. Returns one row per fixture, ready for
    ``write_features``: fixture_id, competition_id, season, as_of_timestamp, vector.
    """
    vectors = compute_rest_and_scoring_margin(fixtures)
    fixture_meta = fixtures[["id", "competition_id", "season"]].rename(columns={"id": "fixture_id"})
    return vectors.merge(fixture_meta, on="fixture_id")


def compute_and_write_nhl_features(competition_id: str) -> int:
    """Loads a competition's fixtures, computes the NHL feature vector, and
    persists it under ``FEATURE_SET``."""
    fixtures = load_fixtures(competition_id)
    rows = compute_nhl_features(fixtures)
    return write_features(FEATURE_SET, rows, fixtures.rename(columns={"id": "fixture_id"}))
