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
import structlog

from deportivas.domain.ids import fixture_id
from deportivas.ingest.base import DataSource
from deportivas.ingest.sources._scalars import to_optional_int, to_utc_from_timestamp

if TYPE_CHECKING:
    from deportivas.ingest.aliases import TeamAliasResolver
    from deportivas.ingest.ratelimit import RateLimiter
    from deportivas.storage.protocols import RawDocumentRepository

logger = structlog.get_logger(__name__)

_EASTERN = ZoneInfo("America/New_York")

# Solo las columnas que _to_team_game_stats necesita: nfl_data_py.import_pbp_data
# trae cientos de columnas por defecto y descargarlas todas es lento e innecesario.
PBP_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "play_type",
    "epa",
    "success",
)

# nflfastR ya distingue jugadas "de verdad" (pass/run) de kickoffs, punts,
# penalizaciones sin jugada, etc. EPA y success solo tienen sentido futbolistico
# sobre estas dos.
_SCRIMMAGE_PLAY_TYPES = frozenset({"pass", "run"})


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

    def fetch_team_game_stats(self, *, seasons: list[int], fixtures: pd.DataFrame) -> pd.DataFrame:
        """Returns rows shaped for the ``nfl_team_game_stats`` table.

        ``fixtures`` must already hold this competition's rows for ``seasons``
        (``deportivas.features.asof.load_fixtures``, or an equivalent read of
        the ``fixtures`` table) — play-by-play keys each game by nflfastR's
        own ``game_id``, not our deterministic fixture id, so matching a play
        back to a fixture goes through (season, home_team_id, away_team_id)
        instead, which is unique within a season's regular NFL schedule.

        Fetches **one season per call**, tolerating the ones nflverse has not
        published yet, because ``import_pbp_data`` handles that case badly in
        two separate ways. Its own "data not available for this year" branch
        reads ``except Error as e:`` and ``Error`` is not defined anywhere in
        the package, so the moment a year's parquet is missing the fallback
        itself raises ``NameError`` — that is what the first real ``daily.yml``
        run hit on the 2026 season, days before nflverse published it. And even
        with that fixed upstream, asking for a list of seasons where *none*
        resolve leaves its ``plays`` variable unassigned and raises
        ``UnboundLocalError`` on return. Passing one season at a time and
        skipping what fails keeps a published season's data from being lost to
        an unpublished one sharing the same call.

        The ``except Exception`` this needs is deliberate, not lazy: when the
        library signals "no data" by raising ``NameError`` from its own error
        handler, there is no meaningful exception type left to catch narrowly.
        It wraps only the library call, so it cannot hide a failure of ours.
        """
        frames: list[pd.DataFrame] = []
        fetched: list[int] = []
        for season in seasons:
            self._wait()
            try:
                frames.append(
                    nfl.import_pbp_data([season], columns=list(PBP_COLUMNS), downcast=True)
                )
            except Exception as exc:  # ver el comentario de abajo: no hay tipo fiable que capturar
                logger.warning(
                    "nfl_pbp_temporada_no_disponible",
                    season=season,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            fetched.append(season)

        if not frames:
            return pd.DataFrame()
        raw = pd.concat(frames, ignore_index=True)
        self._archive_bytes(
            endpoint="import_pbp_data",
            params={"seasons": fetched},
            content=raw.to_parquet(),
            content_type="application/octet-stream",
            status_code=None,
        )
        return self._to_team_game_stats(raw, fixtures=fixtures)

    def _to_team_game_stats(self, raw: pd.DataFrame, *, fixtures: pd.DataFrame) -> pd.DataFrame:
        now = datetime.now(UTC)
        plays = raw[raw["play_type"].isin(_SCRIMMAGE_PLAY_TYPES) & raw["epa"].notna()]
        if plays.empty:
            return pd.DataFrame()

        games = plays.drop_duplicates("game_id")[["game_id", "season", "home_team", "away_team"]]
        offense = _aggregate_side(plays, side_col="posteam")
        defense = _aggregate_side(plays, side_col="defteam")
        fixture_lookup = {
            (str(row["season"]), row["home_team_id"], row["away_team_id"]): row
            for row in fixtures.to_dict("records")
        }

        rows: list[dict[str, object]] = []
        for game in games.to_dict("records"):
            game_id = game["game_id"]
            season = str(game["season"])
            home_team_id = self._aliases.resolve("nfl", str(game["home_team"]))
            away_team_id = self._aliases.resolve("nfl", str(game["away_team"]))
            fixture = fixture_lookup.get((season, home_team_id, away_team_id))
            if fixture is None:
                continue  # partido sin fixture ingerido todavia (o de una temporada distinta)

            for team_code, team_id, is_home in (
                (game["home_team"], home_team_id, True),
                (game["away_team"], away_team_id, False),
            ):
                off = offense.get((game_id, team_code))
                deff = defense.get((game_id, team_code))
                rows.append(
                    {
                        "fixture_id": fixture["id"],
                        "team_id": team_id,
                        "source": self.name,
                        "competition_id": fixture["competition_id"],
                        "season": season,
                        "is_home": is_home,
                        "offensive_plays": off["plays"] if off is not None else 0,
                        "offensive_epa_per_play": off["epa"] if off is not None else None,
                        "offensive_success_rate": off["success"] if off is not None else None,
                        "defensive_plays": deff["plays"] if deff is not None else 0,
                        "defensive_epa_per_play_allowed": deff["epa"] if deff is not None else None,
                        "defensive_success_rate_allowed": deff["success"]
                        if deff is not None
                        else None,
                        "ingested_at": now,
                    }
                )
        return pd.DataFrame(rows)


def _aggregate_side(
    plays: pd.DataFrame, *, side_col: str
) -> dict[tuple[object, object], dict[str, object]]:
    grouped = (
        plays.groupby(["game_id", side_col])
        .agg(epa=("epa", "mean"), success=("success", "mean"), plays=("epa", "size"))
        .reset_index()
    )
    return {
        (row["game_id"], row[side_col]): {
            "epa": row["epa"],
            "success": row["success"],
            "plays": row["plays"],
        }
        for row in grouped.to_dict("records")
    }


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
