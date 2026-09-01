"""football-data.co.uk via ``soccerdata.MatchHistory``: the only source in
this project with genuine multi-season historical odds, including Pinnacle
closing lines (columns suffixed ``C``, e.g. ``PSCH``) — this is what makes a
real backtest possible instead of one starting from today.

Scope for Fase 1: only the 1X2 market is mapped. football-data.co.uk's
over/under and Asian handicap column names have changed more than once across
seasons (unlike the 1X2 columns, stable for 20+ years), and getting one wrong
would silently corrupt odds data rather than fail loudly. Those markets come
from The Odds API going forward instead; extending this adapter to them is a
documented follow-up, not a guess made now.

football-data.co.uk publishes no real per-price timestamp. ``captured_at`` is
therefore an approximation, not an observation: pre-closing bookmaker columns
(``B365H``, ``PSH``, ...) get kickoff minus 1 day, closing columns (``PSCH``,
...) get kickoff itself. This is honest, not exact — real timestamped
snapshots start with the daily odds-capture job (``theoddsapi.py``), which is
also the only source with actual Colombian odds (rule of Fase 1: no invented
source, and this one is flagged as approximate directly in the data via
``is_closing`` rather than silently presented as a real capture).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import soccerdata as sd

from deportivas.domain.ids import deterministic_id, fixture_id
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

# (bookmaker key, column prefix). Pinnacle first: it is the reference price
# for the fair line (devig target in odds/), so it must always be present
# when football-data.co.uk publishes it.
_BOOKMAKER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("pinnacle", "PS"),
    ("bet365", "B365"),
    ("williamhill", "WH"),
    ("betandwin", "BW"),
    ("interwetten", "IW"),
    ("vcbet", "VC"),
)
_SELECTIONS = (("H", "home"), ("D", "draw"), ("A", "away"))


class FootballDataSource(DataSource):
    name = "footballdata"

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

    def fetch_games(
        self, *, competition_id: str, match_history_league: str, seasons: list[str]
    ) -> pd.DataFrame:
        """Returns rows shaped for the ``fixtures`` table, with final scores."""
        self._wait()
        since = datetime.now(UTC)
        reader = sd.MatchHistory(
            leagues=[match_history_league], seasons=seasons, data_dir=self._data_dir
        )
        raw = reader.read_games().reset_index()
        self._archive_cache_dir(self._data_dir, since=since)
        return self._to_fixtures(raw, competition_id=competition_id)

    def fetch_1x2_odds(
        self, *, competition_id: str, match_history_league: str, seasons: list[str]
    ) -> pd.DataFrame:
        """Returns rows shaped for the ``odds_snapshots`` table (market="1x2" only)."""
        self._wait()
        since = datetime.now(UTC)
        reader = sd.MatchHistory(
            leagues=[match_history_league], seasons=seasons, data_dir=self._data_dir
        )
        raw = reader.read_games().reset_index()
        self._archive_cache_dir(self._data_dir, since=since)
        return self._to_odds(raw, competition_id=competition_id)

    # -- mapping ----------------------------------------------------------

    def _to_fixtures(self, raw: pd.DataFrame, *, competition_id: str) -> pd.DataFrame:
        now = datetime.now(UTC)
        rows: list[dict[str, object]] = []
        for data in raw.to_dict("records"):
            kickoff = _to_utc(data.get("date"))
            if kickoff is None:
                continue
            season = str(data["season"])
            home_team_id = self._aliases.resolve("footballdata", str(data["home_team"]))
            away_team_id = self._aliases.resolve("footballdata", str(data["away_team"]))
            home_score = to_optional_int(data.get("FTHG"))
            away_score = to_optional_int(data.get("FTAG"))

            rows.append(
                {
                    "id": fixture_id(competition_id, season, home_team_id, away_team_id, kickoff),
                    "competition_id": competition_id,
                    "season": season,
                    "kickoff_utc": kickoff,
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "status": "finished" if home_score is not None else "scheduled",
                    "stage": None,
                    "matchday": None,
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_score_ht": to_optional_int(data.get("HTHG")),
                    "away_score_ht": to_optional_int(data.get("HTAG")),
                    "kickoff_is_estimated": True,  # ver docstring del modulo
                    "source": self.name,
                    "ingested_at": now,
                }
            )
        return pd.DataFrame(rows)

    def _to_odds(self, raw: pd.DataFrame, *, competition_id: str) -> pd.DataFrame:
        now = datetime.now(UTC)
        rows: list[dict[str, object]] = []
        for data in raw.to_dict("records"):
            kickoff = _to_utc(data.get("date"))
            if kickoff is None:
                continue
            season = str(data["season"])
            home_team_id = self._aliases.resolve("footballdata", str(data["home_team"]))
            away_team_id = self._aliases.resolve("footballdata", str(data["away_team"]))
            fid = fixture_id(competition_id, season, home_team_id, away_team_id, kickoff)
            opening_captured_at = kickoff - timedelta(days=1)

            for bookmaker, prefix in _BOOKMAKER_PREFIXES:
                for is_closing, col_prefix, captured_at in (
                    (False, prefix, opening_captured_at),
                    (True, prefix + "C", kickoff),
                ):
                    for suffix, selection in _SELECTIONS:
                        price = to_optional_float(data.get(f"{col_prefix}{suffix}"))
                        if price is None:
                            continue
                        rows.append(
                            {
                                "id": deterministic_id(
                                    fid, bookmaker, "1x2", selection, captured_at.isoformat()
                                ),
                                "fixture_id": fid,
                                "competition_id": competition_id,
                                "season": season,
                                "bookmaker": bookmaker,
                                "market": "1x2",
                                "selection": selection,
                                "line": None,
                                "price": price,
                                "captured_at": captured_at,
                                "is_closing": is_closing,
                                "source": self.name,
                                "ingested_at": now,
                            }
                        )
        return pd.DataFrame(rows)


def _to_utc(value: object) -> datetime | None:
    ts = to_utc_from_timestamp(value)
    if ts is None:
        return None
    return datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, tzinfo=UTC)
