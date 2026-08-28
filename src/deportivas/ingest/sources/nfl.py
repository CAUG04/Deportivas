"""NFL via ``nfl_data_py.import_schedules`` (-> ``fixtures``).

``nfl_data_py`` fetches parquet/CSV straight from its GitHub releases with no
on-disk cache and no raw bytes exposed to the caller — unlike the
``soccerdata`` adapters, there is no cache directory to archive after the
call. The returned DataFrame itself, serialised to Parquet, is archived as
the raw snapshot instead (documented limitation, see ``ingest/base.py``'s
module docstring for why).

``gameday`` + ``gametime`` are published in US Eastern local time, not UTC.
This adapter converts properly via ``zoneinfo`` (America/New_York, DST-aware)
rather than the "treat as UTC and flag estimated" shortcut used for sources
that don't name their timezone — NFL's is well known and stable.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import nfl_data_py as nfl
import pandas as pd

from deportivas.domain.ids import fixture_id
from deportivas.ingest.base import DataSource
from deportivas.ingest.sources._scalars import to_optional_int, to_utc_from_timestamp

if TYPE_CHECKING:
    from deportivas.ingest.aliases import TeamAliasResolver
    from deportivas.ingest.ratelimit import RateLimiter
    from deportivas.storage.protocols import RawDocumentRepository

_EASTERN = ZoneInfo("America/New_York")


class NflSource(DataSource):
    name = "nfl"

    def __init__(
        self,
        *,
        raw_repo: RawDocumentRepository,
        rate_limiter: RateLimiter,
        aliases: TeamAliasResolver,
    ) -> None:
        super().__init__(raw_repo=raw_repo, rate_limiter=rate_limiter)
        self._aliases = aliases

    def fetch_schedules(self, *, competition_id: str, seasons: list[int]) -> pd.DataFrame:
        """Returns rows shaped for the ``fixtures`` table."""
        self._wait()
        raw = nfl.import_schedules(seasons)
        self._archive_bytes(
            endpoint="import_schedules",
            params={"seasons": seasons},
            content=raw.to_parquet(),
            content_type="application/octet-stream",
            status_code=None,
        )
        return self._to_fixtures(raw, competition_id=competition_id)

    def _to_fixtures(self, raw: pd.DataFrame, *, competition_id: str) -> pd.DataFrame:
        now = datetime.now(UTC)
        rows: list[dict[str, object]] = []
        for data in raw.to_dict("records"):
            kickoff = _combine_eastern(data.get("gameday"), data.get("gametime"))
            if kickoff is None:
                continue
            season = str(data["season"])
            home_team_id = self._aliases.resolve("nfl", str(data["home_team"]))
            away_team_id = self._aliases.resolve("nfl", str(data["away_team"]))
            home_score = to_optional_int(data.get("home_score"))
            away_score = to_optional_int(data.get("away_score"))

            rows.append(
                {
                    "id": fixture_id(competition_id, season, home_team_id, away_team_id, kickoff),
                    "competition_id": competition_id,
                    "season": season,
                    "kickoff_utc": kickoff,
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "status": "finished" if home_score is not None else "scheduled",
                    "stage": _to_optional_str(data.get("game_type")),
                    "matchday": to_optional_int(data.get("week")),
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_score_ht": None,
                    "away_score_ht": None,
                    "kickoff_is_estimated": data.get("gametime") is None,
                    "source": self.name,
                    "ingested_at": now,
                }
            )
        return pd.DataFrame(rows)


def _combine_eastern(gameday: object, gametime: object) -> datetime | None:
    date = to_utc_from_timestamp(gameday)
    if date is None:
        return None
    hour, minute = 13, 0  # sin hora publicada, se asume la franja de 1pm ET
    if isinstance(gametime, str) and gametime.strip():
        with contextlib.suppress(ValueError):
            hour, minute = (int(p) for p in gametime.strip().split(":")[:2])
    local = datetime(date.year, date.month, date.day, hour, minute, tzinfo=_EASTERN)
    return local.astimezone(UTC)


def _to_optional_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)
