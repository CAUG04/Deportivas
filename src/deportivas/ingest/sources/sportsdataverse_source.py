"""NBA and NHL via ``sportsdataverse`` (-> ``fixtures``).

Like ``nfl_data_py``, ``sportsdataverse`` fetches published Parquet releases
directly with no on-disk cache and no raw bytes exposed — the resulting
DataFrame itself is archived as the raw snapshot (see ``ingest/base.py``).

NBA's schedule (``load_nba_schedule``) ships a ready-made tz-aware
``game_date_time`` column (``America/New_York``); this adapter converts it to
UTC directly rather than reconstructing a datetime from separate parts.

NHL's schedule (``load_nhl_schedule``) only publishes separate ``game_date``
and ``game_time`` string columns with an undocumented, unverified time
format/timezone. Rather than guess it, this adapter uses ``game_date`` alone
with a nominal kickoff hour, the same honest fallback FBref's adapter uses
when a source doesn't name a time zone — always flagged
``kickoff_is_estimated=True`` for NHL.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import sportsdataverse.nba as sdv_nba
import sportsdataverse.nhl as sdv_nhl

from deportivas.domain.ids import fixture_id
from deportivas.ingest.base import DataSource
from deportivas.ingest.sources._scalars import to_optional_int, to_utc_from_timestamp

if TYPE_CHECKING:
    from deportivas.ingest.aliases import TeamAliasResolver
    from deportivas.ingest.ratelimit import RateLimiter
    from deportivas.storage.protocols import RawDocumentRepository


class _SportsDataverseSource(DataSource):
    """Shared plumbing for the two sportsdataverse-backed adapters below:
    wait, fetch, archive the DataFrame as the raw snapshot. Only the mapping
    method differs per sport."""

    def __init__(
        self,
        *,
        raw_repo: RawDocumentRepository,
        rate_limiter: RateLimiter,
        aliases: TeamAliasResolver,
    ) -> None:
        super().__init__(raw_repo=raw_repo, rate_limiter=rate_limiter)
        self._aliases = aliases

    def _fetch_and_archive(
        self, loader: Callable[..., pd.DataFrame], seasons: list[int]
    ) -> pd.DataFrame:
        self._wait()
        raw = loader(seasons=seasons, return_as_pandas=True)
        self._archive_bytes(
            endpoint=f"{self.name}_schedule",
            params={"seasons": seasons},
            content=raw.to_parquet(),
            content_type="application/octet-stream",
            status_code=None,
        )
        return raw


class NbaSource(_SportsDataverseSource):
    name = "sportsdataverse_nba"

    def fetch_schedule(self, *, competition_id: str, seasons: list[int]) -> pd.DataFrame:
        raw = self._fetch_and_archive(sdv_nba.load_nba_schedule, seasons)
        return self._to_fixtures(raw, competition_id=competition_id)

    def _to_fixtures(self, raw: pd.DataFrame, *, competition_id: str) -> pd.DataFrame:
        now = datetime.now(UTC)
        rows: list[dict[str, object]] = []
        for data in raw.to_dict("records"):
            ts = to_utc_from_timestamp(data.get("game_date_time"))
            if ts is None:
                continue
            # game_date_time es tz-aware (America/New_York) segun la propia
            # documentacion de sportsdataverse; no hay rama "sin tz" porque
            # tratarla como UTC directamente seria un dato incorrecto, no una
            # aproximacion honesta como en NHL (que sí usa una hora nominal).
            kickoff = ts.tz_convert("UTC").to_pydatetime()
            season = str(data.get("season"))
            home_team_id = self._aliases.resolve(self.name, str(data["home_abbreviation"]))
            away_team_id = self._aliases.resolve(self.name, str(data["away_abbreviation"]))
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
                    "status": "finished"
                    if bool(data.get("status_type_completed"))
                    else "scheduled",
                    "stage": None,
                    "matchday": None,
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_score_ht": None,
                    "away_score_ht": None,
                    "kickoff_is_estimated": False,
                    "source": self.name,
                    "ingested_at": now,
                }
            )
        return pd.DataFrame(rows)


class NhlSource(_SportsDataverseSource):
    name = "sportsdataverse_nhl"

    def fetch_schedule(self, *, competition_id: str, seasons: list[int]) -> pd.DataFrame:
        raw = self._fetch_and_archive(sdv_nhl.load_nhl_schedule, seasons)
        return self._to_fixtures(raw, competition_id=competition_id)

    def _to_fixtures(self, raw: pd.DataFrame, *, competition_id: str) -> pd.DataFrame:
        now = datetime.now(UTC)
        rows: list[dict[str, object]] = []
        for data in raw.to_dict("records"):
            ts = to_utc_from_timestamp(data.get("game_date"))
            if ts is None:
                continue
            kickoff = datetime(ts.year, ts.month, ts.day, 19, 0, tzinfo=UTC)  # ver docstring
            season = str(data.get("season"))
            home_team_id = self._aliases.resolve(self.name, str(data["home_team_abbr"]))
            away_team_id = self._aliases.resolve(self.name, str(data["away_team_abbr"]))
            home_score = to_optional_int(data.get("home_score"))
            away_score = to_optional_int(data.get("away_score"))
            game_state = str(data.get("game_state", ""))

            rows.append(
                {
                    "id": fixture_id(competition_id, season, home_team_id, away_team_id, kickoff),
                    "competition_id": competition_id,
                    "season": season,
                    "kickoff_utc": kickoff,
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "status": "finished" if game_state in {"OFF", "FINAL"} else "scheduled",
                    "stage": _to_optional_str(data.get("game_type")),
                    "matchday": None,
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_score_ht": None,
                    "away_score_ht": None,
                    "kickoff_is_estimated": True,
                    "source": self.name,
                    "ingested_at": now,
                }
            )
        return pd.DataFrame(rows)


def _to_optional_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)
