"""Point-in-time data access for feature computation.

Every feature pipeline in this package processes a competition's fixtures in
kickoff order and, for each one, must see only what happened strictly before
it. These helpers load the raw material (fixtures, team match stats) once per
competition and hand pipelines a clean, sorted, source-deduplicated view —
the walk-forward loop itself lives in each sport's pipeline, not here, since
the state each one accumulates (Elo ratings, rolling xG windows, last-played
dates) differs too much to force into one shape.
"""

from __future__ import annotations

import pandas as pd

from deportivas.contracts.tables import FIXTURES, TEAM_MATCH_STATS
from deportivas.storage.factory import get_table_repository

DEFAULT_SOURCE_PRIORITY: tuple[str, ...] = ("fbref", "footballdata", "understat", "espn")


def load_fixtures(competition_id: str) -> pd.DataFrame:
    """Every fixture for a competition, sorted by kickoff (oldest first)."""
    repo = get_table_repository(FIXTURES)
    df = repo.read(filters={"competition_id": competition_id})
    return df.sort_values("kickoff_utc", kind="stable", ignore_index=True)


def load_team_match_stats(
    competition_id: str, *, source_priority: tuple[str, ...] = DEFAULT_SOURCE_PRIORITY
) -> pd.DataFrame:
    """Team match stats for a competition, one row per (fixture_id, team_id).

    Multiple sources can report the same match (FBref and Understat both
    estimate xG independently — that's why ``team_match_stats``'s key
    includes ``source``, see ``contracts/tables.py``). Here, for feature
    computation, exactly one row per team per match is needed: the highest-
    priority source that actually reported that match wins outright, rather
    than mixing individual columns across sources row by row.
    """
    repo = get_table_repository(TEAM_MATCH_STATS)
    df = repo.read(filters={"competition_id": competition_id})
    if df.empty:
        return df
    priority = {source: rank for rank, source in enumerate(source_priority)}
    fallback_rank = len(source_priority)
    df = df.assign(_priority=df["source"].map(priority).fillna(fallback_rank))
    df = df.sort_values("_priority", kind="stable")
    df = df.drop_duplicates(subset=["fixture_id", "team_id"], keep="first")
    return df.drop(columns="_priority").reset_index(drop=True)
