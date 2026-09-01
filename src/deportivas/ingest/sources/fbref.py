"""FBref via ``soccerdata``: schedule (-> ``fixtures``) and per-team match
logs (-> ``team_match_stats``), for the 5 major European leagues that
``soccerdata.LEAGUE_DICT`` covers directly plus the others declared in
``config/competitions.yaml``.

FBref throttles aggressively: every call to ``fetch_schedule`` /
``fetch_team_match_stats`` waits on the shared rate limiter first (rule of
Fase 1). ``soccerdata`` keeps its own on-disk cache of the HTML it downloads
(pointed at ``data_dir`` below); this adapter archives whatever new files
land there into the append-only raw layer right after each call — see
``ingest/base.py`` for why that two-step exists.

FBref also fronts a CAPTCHA that ``soccerdata``'s Selenium-based reader
hits on datacenter IPs like GitHub Actions runners (Fase 10 found this in
production). ``soccerdata``'s own CAPTCHA solver is a no-op in headless
mode — it only attempts to solve one when ``headless=False``, which needs a
virtual display (Xvfb) behind it, or Chrome has nowhere to draw. That's
what ``Settings.fbref_headless`` and the ``headless`` constructor arg here
are for — see the workflows for the ``xvfb-run`` wrapper that makes
``headless=False`` viable in CI at all.
"""

from __future__ import annotations

import contextlib
import re
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

_SCORE_RE = re.compile(r"^\s*(\d+)\s*[–‐—-]\s*(\d+)\s*$")  # noqa: RUF001 (guiones reales de FBref)


def parse_score(text: object) -> tuple[int | None, int | None]:
    """FBref renders an unplayed fixture's score as an empty/NaN cell and a
    played one as e.g. "3-1" (en dash in the real data). Anything else is
    treated as unplayed rather than guessed at."""
    if not isinstance(text, str):
        return None, None
    match = _SCORE_RE.match(text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


class FBrefSource(DataSource):
    name = "fbref"

    def __init__(
        self,
        *,
        raw_repo: RawDocumentRepository,
        rate_limiter: RateLimiter,
        data_dir: Path,
        aliases: TeamAliasResolver,
        headless: bool = True,
    ) -> None:
        super().__init__(raw_repo=raw_repo, rate_limiter=rate_limiter)
        self._data_dir = data_dir
        self._aliases = aliases
        # Ver Settings.fbref_headless: el solver de CAPTCHA de soccerdata
        # (PyAutoGUI) es un no-op cuando headless=True. False solo funciona
        # con una pantalla virtual (Xvfb) detras -- responsabilidad del
        # llamador, no de este adaptador.
        self._headless = headless

    def fetch_schedule(
        self, *, competition_id: str, fbref_league: str, seasons: list[str]
    ) -> pd.DataFrame:
        """Returns rows shaped for the ``fixtures`` table."""
        self._wait()
        since = datetime.now(UTC)
        reader = sd.FBref(
            leagues=[fbref_league],
            seasons=seasons,
            data_dir=self._data_dir,
            headless=self._headless,
        )
        raw = reader.read_schedule().reset_index()
        self._archive_cache_dir(self._data_dir, since=since)
        return self._to_fixtures(raw, competition_id=competition_id)

    def fetch_team_match_stats(
        self, *, competition_id: str, fbref_league: str, seasons: list[str]
    ) -> pd.DataFrame:
        """Returns rows shaped for the ``team_match_stats`` table.

        ``fixture_id`` is recomputed from the same natural key as
        ``fetch_schedule`` uses (same hash function, same inputs), not looked
        up — so this does not require the matching fixtures to already be in
        storage, and both tables' rows join correctly once they are.
        """
        self._wait()
        since = datetime.now(UTC)
        reader = sd.FBref(
            leagues=[fbref_league],
            seasons=seasons,
            data_dir=self._data_dir,
            headless=self._headless,
        )
        raw = reader.read_team_match_stats(stat_type="schedule").reset_index()
        self._archive_cache_dir(self._data_dir, since=since)
        return self._to_team_match_stats(raw, competition_id=competition_id)

    # -- mapping --------------------------------------------------------

    def _to_fixtures(self, raw: pd.DataFrame, *, competition_id: str) -> pd.DataFrame:
        now = datetime.now(UTC)
        rows: list[dict[str, object]] = []
        for data in raw.to_dict("records"):
            season = str(data["season"])
            home_team_id = self._aliases.resolve("fbref", str(data["home_team"]))
            away_team_id = self._aliases.resolve("fbref", str(data["away_team"]))

            date = data.get("date")
            time_str = data.get("time")
            kickoff_is_estimated = not (isinstance(time_str, str) and time_str.strip())
            kickoff = _combine_date_time(date, time_str)
            if kickoff is None:
                continue

            home_score, away_score = parse_score(data.get("score"))
            status = "finished" if home_score is not None else "scheduled"

            rows.append(
                {
                    "id": fixture_id(competition_id, season, home_team_id, away_team_id, kickoff),
                    "competition_id": competition_id,
                    "season": season,
                    "kickoff_utc": kickoff,
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "status": status,
                    "stage": None,
                    "matchday": to_optional_int(data.get("week")),
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_score_ht": None,
                    "away_score_ht": None,
                    "kickoff_is_estimated": kickoff_is_estimated,
                    "source": self.name,
                    "ingested_at": now,
                }
            )
        return pd.DataFrame(rows)

    def _to_team_match_stats(self, raw: pd.DataFrame, *, competition_id: str) -> pd.DataFrame:
        now = datetime.now(UTC)
        rows: list[dict[str, object]] = []
        for data in raw.to_dict("records"):
            season = str(data["season"])
            team_id = self._aliases.resolve("fbref", str(data["team"]))
            opponent_id = self._aliases.resolve("fbref", str(data.get("opponent")))
            is_home = str(data.get("venue")) == "Home"
            home_team_id, away_team_id = (
                (team_id, opponent_id) if is_home else (opponent_id, team_id)
            )

            date = data.get("date")
            kickoff = _combine_date_time(date, data.get("time"))
            if kickoff is None:
                continue

            gf = to_optional_int(data.get("gf"))
            rows.append(
                {
                    "fixture_id": fixture_id(
                        competition_id, season, home_team_id, away_team_id, kickoff
                    ),
                    "team_id": team_id,
                    "competition_id": competition_id,
                    "season": season,
                    "is_home": is_home,
                    "goals": gf,
                    # stat_type="schedule" no trae tiros ni corners (serian
                    # tiros de stat_type="shooting", que este metodo no pide
                    # todavia; corners no lo publica FBref en ningun stat_type).
                    "shots": to_optional_int(data.get("sh")),
                    "shots_on_target": to_optional_int(data.get("sot")),
                    "xg": to_optional_float(data.get("xg")),
                    "corners": None,
                    "yellow_cards": None,
                    "red_cards": None,
                    "possession": to_optional_float(data.get("poss")),
                    "source": self.name,
                    "ingested_at": now,
                }
            )
        return pd.DataFrame(rows)


def _combine_date_time(date: object, time_str: object) -> datetime | None:
    ts = to_utc_from_timestamp(date)
    if ts is None:
        return None
    hour, minute = 12, 0  # sin hora publicada, se asume mediodia y se marca estimado
    if isinstance(time_str, str) and time_str.strip():
        with contextlib.suppress(ValueError):
            hour, minute = (int(p) for p in time_str.strip().split(":")[:2])
    return datetime(ts.year, ts.month, ts.day, hour, minute, tzinfo=UTC)
