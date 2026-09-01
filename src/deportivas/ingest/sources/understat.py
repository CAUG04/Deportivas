"""Understat via ``soccerdata``: match-level xG (-> ``team_match_stats``).

Understat's own shot-based xG model is independent from FBref's (StatsBomb
data) — having both gives the future Dixon-Coles/xG features (Fase 2) two
uncorrelated estimates of the same thing instead of one.

Shape note: unlike FBref, ``read_team_match_stats()`` returns **one row per
match** with ``home_*``/``away_*`` prefixed columns for both sides, not one
row per team. This adapter splits each match row into the two
``team_match_stats`` rows (home and away) the table actually wants.

Only ``xg`` and ``goals`` are populated here; Understat does not publish
shots, shots on target, corners or cards, so the rest stay ``None`` rather
than guessed at.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import soccerdata as sd

from deportivas.domain.ids import fixture_id
from deportivas.ingest.base import DataSource
from deportivas.ingest.sources._scalars import (
    to_optional_float,
    to_optional_int,
    to_utc_from_timestamp,
)

if TYPE_CHECKING:
    from pathlib import Path

    from deportivas.ingest.aliases import TeamAliasResolver
    from deportivas.ingest.ratelimit import RateLimiter
    from deportivas.storage.protocols import RawDocumentRepository


class UnderstatSource(DataSource):
    name = "understat"

    def __init__(
        self,
        *,
        raw_repo: RawDocumentRepository,
        rate_limiter: RateLimiter,
        data_dir: Path,
        aliases: TeamAliasResolver,
    ) -> None:
        super().__init__(raw_repo=raw_repo, rate_limiter=rate_limiter)
        self._data_dir = data_dir
        self._aliases = aliases

    def fetch_team_match_stats(
        self, *, competition_id: str, understat_league: str, seasons: list[str]
    ) -> pd.DataFrame:
        """Returns rows shaped for the ``team_match_stats`` table."""
        self._wait()
        since = datetime.now(UTC)
        reader = sd.Understat(leagues=[understat_league], seasons=seasons, data_dir=self._data_dir)
        raw = reader.read_team_match_stats().reset_index()
        self._archive_cache_dir(self._data_dir, since=since)
        return self._to_team_match_stats(raw, competition_id=competition_id)

    def _to_team_match_stats(self, raw: pd.DataFrame, *, competition_id: str) -> pd.DataFrame:
        now = datetime.now(UTC)
        rows: list[dict[str, object]] = []
        for data in raw.to_dict("records"):
            ts = to_utc_from_timestamp(data.get("date"))
            if ts is None:
                continue
            kickoff = datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, tzinfo=UTC)
            season = str(data["season"])
            home_team_id = self._aliases.resolve("understat", str(data["home_team"]))
            away_team_id = self._aliases.resolve("understat", str(data["away_team"]))
            fid = fixture_id(competition_id, season, home_team_id, away_team_id, kickoff)

            for is_home, team_id, prefix in (
                (True, home_team_id, "home"),
                (False, away_team_id, "away"),
            ):
                rows.append(
                    {
                        "fixture_id": fid,
                        "team_id": team_id,
                        "competition_id": competition_id,
                        "season": season,
                        "is_home": is_home,
                        "goals": to_optional_int(data.get(f"{prefix}_goals")),
                        "shots": None,
                        "shots_on_target": None,
                        "xg": to_optional_float(data.get(f"{prefix}_xg")),
                        "corners": None,
                        "yellow_cards": None,
                        "red_cards": None,
                        "possession": None,
                        "source": self.name,
                        "ingested_at": now,
                    }
                )
        return pd.DataFrame(rows)
