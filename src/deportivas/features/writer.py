"""Persists a feature pipeline's output, gated by the leakage guard.

``as_of_timestamp`` is not "now" and not "kickoff" — it is the true upper
bound on the information a vector actually used, tracked by each pipeline's
walk-forward loop (typically the kickoff of the most recent prior match
either team played). A loose bound like "now" would pass the leakage check
trivially without saying anything useful; the tight bound is what lets the
backtest engine (Fase 4) reconstruct exactly what was knowable at the time.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from deportivas.contracts.tables import FEATURES
from deportivas.domain.leakage import assert_no_leakage
from deportivas.storage.factory import get_table_repository


def write_features(feature_set: str, rows: pd.DataFrame, fixtures: pd.DataFrame) -> int:
    """``rows`` needs columns: fixture_id, competition_id, season, as_of_timestamp,
    vector (a plain dict per row, JSON-serialised here). ``fixtures`` needs
    fixture_id and kickoff_utc, for the leakage check."""
    if rows.empty:
        return 0

    now = datetime.now(UTC)
    to_write = rows.copy()
    to_write["feature_set"] = feature_set
    to_write["vector"] = to_write["vector"].map(
        lambda v: json.dumps(v, sort_keys=True, default=str)
    )
    to_write["computed_at"] = now
    to_write["source"] = feature_set
    to_write["ingested_at"] = now

    assert_no_leakage(to_write, fixtures)

    repo = get_table_repository(FEATURES, temporal_column="as_of_timestamp")
    return repo.write(to_write[list(FEATURES.column_names)])
