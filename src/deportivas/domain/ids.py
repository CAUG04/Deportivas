"""Deterministic ids.

A fixture reported by FBref, ESPN and football-data.co.uk must resolve to the
*same* ``fixture_id`` in all three so their rows join on it — a random uuid
per source would make that impossible without a lookup round-trip. Hashing
the natural key gives every adapter the same id independently, with no
coordination needed.
"""

from __future__ import annotations

import hashlib
from datetime import datetime


def fixture_id(
    competition_id: str,
    season: str,
    home_team_id: str,
    away_team_id: str,
    kickoff_utc: datetime,
) -> str:
    if kickoff_utc.tzinfo is None:
        raise ValueError("kickoff_utc debe ser timezone-aware (UTC)")
    parts = "|".join((competition_id, season, home_team_id, away_team_id, kickoff_utc.isoformat()))
    return hashlib.sha1(parts.encode("utf-8"), usedforsecurity=False).hexdigest()


def deterministic_id(*parts: str) -> str:
    """A short, deterministic id for any row whose primary key just needs to
    be stable and unique — not, like ``fixture_id``, independently
    recomputable and joinable across sources. Concatenating raw parts
    (a fixture id, a bookmaker, a captured_at timestamp...) directly would
    blow past every id column's 64-char limit; this always returns 40 hex
    characters regardless of how long or how many the parts are."""
    joined = "|".join(parts)
    return hashlib.sha1(joined.encode("utf-8"), usedforsecurity=False).hexdigest()
