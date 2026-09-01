"""MLB via ``pybaseball.schedule_and_record`` (-> ``fixtures``).

**Column names here are not verified against a live fetch** (this codebase
was built without network access to baseball-reference.com) — unlike every
other adapter in this project, where columns were checked against the
installed library's actual source. ``schedule_and_record`` scrapes an HTML
table directly, so its column names live in baseball-reference's markup, not
in ``pybaseball``'s Python source, and could not be confirmed offline. The
mapping below uses the format documented and observed for years (``Date``,
``Home_Away``, ``Opp``, ``R``, ``RA``, ``W/L``), defensively (``.get()``
everywhere, nothing raises on an unexpected shape). Fase 10's
``sources-health.yml`` is exactly the mechanism meant to catch a wrong guess
here — verifying this mapping against a real fetch is a flagged follow-up,
not something quietly assumed correct.

No single call returns the whole league's schedule: baseball-reference's table
is per-team, so ``fetch_schedule`` takes a list of the 30 team abbreviations
and fetches once per team. Each real game therefore appears in two teams'
results; since ``fixture_id`` is a deterministic hash of the natural key, both
occurrences collapse into the same row on write rather than duplicating.

Does **not** call ``pybaseball.schedule_and_record`` directly: its last step
crashes on any in-progress season. See ``_fetch_team_table`` below for the
detail and for which of its steps this adapter does use.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import structlog
from pybaseball import team_results as _pb

from deportivas.domain.ids import fixture_id
from deportivas.ingest.base import DataSource
from deportivas.ingest.sources._scalars import to_optional_int

if TYPE_CHECKING:
    from deportivas.ingest.aliases import TeamAliasResolver
    from deportivas.ingest.ratelimit import RateLimiter
    from deportivas.storage.protocols import RawDocumentRepository

logger = structlog.get_logger(__name__)


class PybaseballSource(DataSource):
    name = "pybaseball"

    def __init__(
        self,
        *,
        raw_repo: RawDocumentRepository,
        rate_limiter: RateLimiter,
        aliases: TeamAliasResolver,
    ) -> None:
        super().__init__(raw_repo=raw_repo, rate_limiter=rate_limiter)
        self._aliases = aliases

    def fetch_schedule(
        self, *, competition_id: str, season: int, team_abbreviations: list[str]
    ) -> pd.DataFrame:
        """Returns rows shaped for the ``fixtures`` table, one call per team."""
        frames = []
        for team in team_abbreviations:
            self._wait()
            raw = _fetch_team_table(season, team)
            self._archive_bytes(
                endpoint="schedule_and_record",
                params={"season": season, "team": team},
                content=raw.to_parquet(),
                content_type="application/octet-stream",
                status_code=None,
            )
            frames.append(self._to_fixtures(raw, competition_id=competition_id, season=season))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _to_fixtures(self, raw: pd.DataFrame, *, competition_id: str, season: int) -> pd.DataFrame:
        now = datetime.now(UTC)
        rows: list[dict[str, object]] = []
        for data in raw.to_dict("records"):
            ts = _safe_parse_date(_with_year(data.get("Date"), season))
            if ts is None:
                continue
            kickoff = datetime(
                ts.year, ts.month, ts.day, 19, 0, tzinfo=UTC
            )  # hora nominal, ver docstring

            team = data.get("Tm")
            opponent = data.get("Opp")
            if team is None or opponent is None:
                continue
            # "@" marca visitante; ausente/NaN/vacio es local. No usar
            # bool(str(...)) aqui: str(float("nan")) == "nan", que es un
            # string no vacio y volteria erroneamente los partidos locales.
            home_away_raw = data.get("Home_Away")
            is_away = isinstance(home_away_raw, str) and home_away_raw.strip() == "@"
            team_id = self._aliases.resolve("pybaseball", str(team))
            opponent_id = self._aliases.resolve("pybaseball", str(opponent))
            home_team_id, away_team_id = (
                (opponent_id, team_id) if is_away else (team_id, opponent_id)
            )
            runs_for = to_optional_int(data.get("R"))
            runs_against = to_optional_int(data.get("RA"))
            home_score, away_score = (
                (runs_against, runs_for) if is_away else (runs_for, runs_against)
            )

            season_str = str(season)
            rows.append(
                {
                    "id": fixture_id(
                        competition_id, season_str, home_team_id, away_team_id, kickoff
                    ),
                    "competition_id": competition_id,
                    "season": season_str,
                    "kickoff_utc": kickoff,
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "status": "finished" if home_score is not None else "scheduled",
                    "stage": None,
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


def _fetch_team_table(season: int, team: str) -> pd.DataFrame:
    """Lo que ``pybaseball.schedule_and_record`` hace, menos su ultimo paso.

    ``schedule_and_record`` encadena cuatro pasos: ``get_soup`` (descarga),
    ``get_table`` (parsea el HTML), ``process_win_streak`` (racha) y
    ``make_numeric`` (convierte columnas a float). El ultimo esta roto para
    cualquier temporada en curso, y lo confirmo la primera corrida real de
    daily.yml con ``ValueError: could not convert string to float: 'Unknown'``.

    La causa es de la propia libreria: ``get_table`` rellena con el centinela
    ``"Unknown"`` tres columnas cuando la celda viene vacia (un partido que
    todavia no se juega no tiene marcador, entradas ni puesto en la tabla),
    pero solo convierte una de las tres -- ``Attendance`` -- de vuelta a NaN.
    ``make_numeric`` llega despues y hace ``astype(float)`` sobre
    ``["R", "RA", "Inn", "Rank", "Attendance"]``, y revienta con el
    ``"Unknown"`` que quedo en ``Rank``. En temporada baja no se nota; con la
    temporada en marcha falla siempre, que es justo cuando hace falta.

    Convertir texto a numero ahi es correcto y necesario -- baseball-reference
    publica una tabla HTML y todo llega como texto; lo que falta es tolerar el
    centinela, algo que ``pd.to_numeric(..., errors="coerce")`` daria gratis.

    Este adaptador no necesita ese paso en absoluto: lee ``R``/``RA`` con
    ``to_optional_int``, que ya devuelve ``None`` ante cualquier celda que no
    sea un numero (``"Unknown"`` incluido), y no mira ninguna de las otras
    columnas numericas. Tampoco necesita ``process_win_streak``: la racha no
    entra en ``fixtures``. Asi que se llaman los dos pasos que si sirven y se
    omiten los dos que no, usando las mismas funciones publicas de la libreria
    -- sin parchearla por dentro ni duplicar su scraping. ``get_soup`` sigue
    usando la sesion de pybaseball, que ya trae su propio rate limiting hacia
    baseball-reference.
    """
    # Lo unico que se pierde al no pasar por schedule_and_record: su guarda de
    # temporada futura. Se conserva aqui para no cambiar ese comportamiento.
    if season > datetime.now(UTC).year:
        raise ValueError("Season cannot be after current year")
    # pybaseball no publica py.typed, asi que mypy ve Any: se ancla aqui.
    table: pd.DataFrame = _pb.get_table(_pb.get_soup(season, team), team)
    return table


_DOUBLEHEADER_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")


def _with_year(date_str: object, season: int) -> str | None:
    """baseball-reference's "Date" column has no year (e.g. "Monday, Apr 7");
    the season the caller requested supplies it. Doubleheader games are
    suffixed "(1)"/"(2)", which breaks date parsing outright if left in —
    stripped here since it carries no date information anyway."""
    if not isinstance(date_str, str) or not date_str.strip():
        return None
    cleaned = _DOUBLEHEADER_SUFFIX.sub("", date_str.strip())
    return f"{cleaned}, {season}"


def _safe_parse_date(date_str: str | None) -> pd.Timestamp | None:
    """Given the acknowledged uncertainty about this source's exact format
    (see module docstring), a row baseball-reference formats unexpectedly
    must be skipped and logged, never crash the whole team's ingestion."""
    if date_str is None:
        return None
    try:
        return pd.Timestamp(date_str)
    except (ValueError, TypeError):
        logger.warning("fecha_no_parseable_pybaseball", raw_date=date_str)
        return None
