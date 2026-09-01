"""ESPN via ``soccerdata``: schedule listing (-> ``fixtures``).

This is the only source in ``config/competitions.yaml`` with an ``espn`` key
for **every** competition, including Liga BetPlay Dimayor (Colombia), which
no other source in this project covers at all (rule of Fase 1: documented,
not invented). It is also the intended fallback for the European leagues if
FBref ever blocks the ingestion job.

Real limitation, not a bug: ``ESPN.read_schedule()`` returns fixture listings
(teams, kickoff) but no final score — that lives behind a separate
per-match ``read_matchsheet(match_id)`` call soccerdata exposes, which is one
HTTP request per finished match and therefore not something a bulk schedule
fetch can do cheaply. So every fixture from this adapter is
``status="scheduled"``, even ones already played. Backfilling Colombian
results via ``read_matchsheet`` is a real, scoped-out follow-up, not silently
pretended to be solved here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import soccerdata as sd

from deportivas.domain.ids import fixture_id
from deportivas.ingest.base import DataSource
from deportivas.ingest.sources._scalars import to_utc_from_timestamp

if TYPE_CHECKING:
    from pathlib import Path

    from deportivas.ingest.aliases import TeamAliasResolver
    from deportivas.ingest.ratelimit import RateLimiter
    from deportivas.storage.protocols import RawDocumentRepository


class EspnSource(DataSource):
    name = "espn"

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

    def fetch_schedule(
        self, *, competition_id: str, espn_league: str, seasons: list[str]
    ) -> pd.DataFrame:
        """Returns rows shaped for the ``fixtures`` table (no final scores — see module docstring)."""
        self._wait()
        since = datetime.now(UTC)
        reader = sd.ESPN(leagues=[espn_league], seasons=seasons, data_dir=self._data_dir)
        raw = reader.read_schedule().reset_index()
        self._archive_cache_dir(self._data_dir, since=since)
        return self._to_fixtures(raw, competition_id=competition_id)

    def _to_fixtures(self, raw: pd.DataFrame, *, competition_id: str) -> pd.DataFrame:
        now = datetime.now(UTC)
        rows: list[dict[str, object]] = []
        for data in raw.to_dict("records"):
            ts = to_utc_from_timestamp(data.get("date"))
            if ts is None:
                continue
            kickoff = datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, tzinfo=UTC)
            season = str(data["season"])
            home_team_id = self._aliases.resolve("espn", str(data["home_team"]))
            away_team_id = self._aliases.resolve("espn", str(data["away_team"]))

            rows.append(
                {
                    "id": fixture_id(competition_id, season, home_team_id, away_team_id, kickoff),
                    "competition_id": competition_id,
                    "season": season,
                    "kickoff_utc": kickoff,
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "status": "scheduled",
                    "stage": None,
                    "matchday": None,
                    "home_score": None,
                    "away_score": None,
                    "home_score_ht": None,
                    "away_score_ht": None,
                    "kickoff_is_estimated": False,
                    "source": self.name,
                    "ingested_at": now,
                }
            )
        return pd.DataFrame(rows)
