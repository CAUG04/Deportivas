"""MLB features: rest days, back-to-backs, and rolling scoring margin.

MLB's only ingested per-game signal is the final score (``fixtures``, via
``pybaseball``), and unlike NBA/NHL there is no announced-starting-pitcher
data ingested yet, so pitcher-level signals are out of scope for this phase.
Two defaults differ from ``rest_and_margin.py``'s NBA/NHL-tuned ones:

* ``windows=(10, 20, 40)`` — a 162-game season makes the default 5/10/20
  window too short to smooth out day-to-day scoring noise.
* ``back_to_back_max_rest_days=0`` — MLB teams play on consecutive days as a
  matter of course, so "rest_days <= 1" would be true for nearly every game
  and say nothing; the meaningfully unusual case is a same-day doubleheader
  (``rest_days == 0``).
"""

from __future__ import annotations

import pandas as pd

from deportivas.features.asof import load_fixtures
from deportivas.features.rest_and_margin import RestAndMarginConfig, compute_rest_and_scoring_margin
from deportivas.features.writer import write_features

FEATURE_SET = "mlb_v1"

_CONFIG = RestAndMarginConfig(windows=(10, 20, 40), back_to_back_max_rest_days=0)


def compute_mlb_features(fixtures: pd.DataFrame) -> pd.DataFrame:
    """``fixtures`` sorted by kickoff ascending, with ``competition_id`` and
    ``season`` for the output rows. Returns one row per fixture, ready for
    ``write_features``: fixture_id, competition_id, season, as_of_timestamp, vector.
    """
    vectors = compute_rest_and_scoring_margin(fixtures, config=_CONFIG)
    fixture_meta = fixtures[["id", "competition_id", "season"]].rename(columns={"id": "fixture_id"})
    return vectors.merge(fixture_meta, on="fixture_id")


def compute_and_write_mlb_features(competition_id: str) -> int:
    """Loads a competition's fixtures, computes the MLB feature vector, and
    persists it under ``FEATURE_SET``."""
    fixtures = load_fixtures(competition_id)
    rows = compute_mlb_features(fixtures)
    return write_features(FEATURE_SET, rows, fixtures.rename(columns={"id": "fixture_id"}))
