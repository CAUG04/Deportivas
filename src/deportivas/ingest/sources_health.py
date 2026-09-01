"""Fase 10: valida contra la fuente real -- no solo contra su propia
sintaxis -- los identificadores que ``config/competitions.yaml`` declaro
desde la Fase 1 como "no verificados todavia" (Eredivisie, Primeira Liga,
las tres competiciones UEFA, Liga BetPlay Dimayor) y las claves de deporte
de The Odds API. Pensado para ``.github/workflows/sources-health.yml``,
donde SI hay acceso de red real (esta sesion de desarrollo no lo tiene, ver
README) -- se ejecuta antes que ``daily.yml``/``odds.yml`` en el cron, para
atrapar un identificador roto (o una clave de The Odds API invalida) antes
de que la ingesta o la captura de cuotas de verdad lo intenten y desperdicien
creditos o tiempo de scraping en algo que iba a fallar de todas formas.

Reutiliza los mismos adaptadores que ``deportivas ingest`` (mismo rate
limiting, mismo archivado en la capa cruda) para no arriesgar un
comportamiento de red distinto entre "validar" y "ingerir de verdad" -- la
unica diferencia es que aqui el DataFrame que cada adaptador devuelve se
descarta, nunca se persiste en fixtures/team_match_stats/odds_snapshots. No
participa del data lake de la Fase 9: lo que archiva en la capa cruda
durante la corrida vive y muere con el runner efimero, nunca se publica ni
se restaura (``scripts/{restore,publish}_data_lake.sh``).

Un DataFrame vacio (liga fuera de temporada, sin partidos todavia) NO es un
fallo -- solo una excepcion real (error de red, HTTP, "liga invalida") lo
es. Tratar "vacio" como fallo produciria falsos positivos constantes en
temporada baja, que es exactamente el ruido que este chequeo existe para
evitar.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import pandas as pd

from deportivas.config.catalog import Competition, load_competitions
from deportivas.config.settings import get_settings
from deportivas.contracts.tables import TEAM_ALIASES, TEAMS
from deportivas.domain.enums import Sport
from deportivas.domain.seasons import season_labels
from deportivas.ingest.aliases import TeamAliasResolver
from deportivas.ingest.ratelimit import RateLimiter
from deportivas.ingest.soccerdata_config import ensure_custom_league_dict
from deportivas.storage.factory import get_raw_document_repository, get_table_repository

_ODDS_API_SPORTS_URL = "https://api.the-odds-api.com/v4/sports"


@dataclass(frozen=True)
class HealthIssue:
    competition_id: str
    field: str
    detail: str

    def __str__(self) -> str:
        return f"{self.competition_id}.{self.field}: {self.detail}"


def _try(competition_id: str, field: str, fetch: Callable[[], pd.DataFrame]) -> list[HealthIssue]:
    try:
        fetch()
    except Exception as exc:  # cualquier fallo de la fuente cuenta, no solo los tipos ya vistos
        return [HealthIssue(competition_id, field, f"{type(exc).__name__}: {exc}")]
    return []


def check_football_sources(competitions: list[Competition]) -> list[HealthIssue]:
    football = [c for c in competitions if c.sport is Sport.FOOTBALL]
    if not football:
        return []

    from deportivas.ingest.sources.espn import EspnSource
    from deportivas.ingest.sources.fbref import FBrefSource
    from deportivas.ingest.sources.footballdata import FootballDataSource
    from deportivas.ingest.sources.understat import UnderstatSource

    ensure_custom_league_dict()
    settings = get_settings()
    raw_repo = get_raw_document_repository()
    aliases = TeamAliasResolver(
        get_table_repository(TEAMS), get_table_repository(TEAM_ALIASES), sport="football"
    )
    quiet = RateLimiter(0.0)
    fbref = FBrefSource(
        raw_repo=raw_repo,
        rate_limiter=RateLimiter(settings.fbref_min_delay_seconds),
        data_dir=settings.cache_dir / "fbref",
        aliases=aliases,
    )
    understat = UnderstatSource(
        raw_repo=raw_repo,
        rate_limiter=quiet,
        data_dir=settings.cache_dir / "understat",
        aliases=aliases,
    )
    espn = EspnSource(
        raw_repo=raw_repo, rate_limiter=quiet, data_dir=settings.cache_dir / "espn", aliases=aliases
    )
    footballdata = FootballDataSource(
        raw_repo=raw_repo,
        rate_limiter=quiet,
        data_dir=settings.cache_dir / "footballdata",
        aliases=aliases,
    )

    issues: list[HealthIssue] = []
    for competition in football:
        season = season_labels(competition, count=1)[0]
        sources = competition.sources

        if sources.fbref is not None:
            issues += _try(
                competition.id,
                "sources.fbref",
                functools.partial(
                    fbref.fetch_schedule,
                    competition_id=competition.id,
                    fbref_league=sources.fbref,
                    seasons=[season],
                ),
            )
        if sources.understat is not None:
            issues += _try(
                competition.id,
                "sources.understat",
                functools.partial(
                    understat.fetch_team_match_stats,
                    competition_id=competition.id,
                    understat_league=sources.understat,
                    seasons=[season],
                ),
            )
        if sources.match_history is not None:
            issues += _try(
                competition.id,
                "sources.match_history",
                functools.partial(
                    footballdata.fetch_games,
                    competition_id=competition.id,
                    match_history_league=sources.match_history,
                    seasons=[season],
                ),
            )
        elif sources.espn is not None:
            # Respaldo de calendario solo cuando no hay football-data.co.uk
            # -- ver README, "unica fuente para Liga BetPlay Dimayor".
            issues += _try(
                competition.id,
                "sources.espn",
                functools.partial(
                    espn.fetch_schedule,
                    competition_id=competition.id,
                    espn_league=sources.espn,
                    seasons=[season],
                ),
            )
    return issues


def check_odds_api_sport_keys(
    competitions: list[Competition], *, client: httpx.Client | None = None
) -> list[HealthIssue]:
    """Valida cada ``odds.the_odds_api`` contra ``/v4/sports`` -- este
    endpoint no gasta creditos del plan gratuito (a diferencia de
    ``/v4/sports/{sport}/odds``, que si), asi que corre aqui sin miedo a
    consumir presupuesto. Sin clave configurada, no hay nada que preguntar:
    se salta en silencio en vez de fallar, para no bloquear la validacion
    de las fuentes de futbol cuando todavia no se ha configurado The Odds
    API. ``client`` es para los tests (``httpx.MockTransport``, mismo
    patron que ``TheOddsApiSource``); los llamadores reales lo omiten."""
    settings = get_settings()
    if not settings.has_odds_api_key:
        return []
    assert settings.the_odds_api_key is not None

    owns_client = client is None
    http_client = client if client is not None else httpx.Client(timeout=15.0)
    try:
        response = http_client.get(
            _ODDS_API_SPORTS_URL,
            params={"apiKey": settings.the_odds_api_key.get_secret_value()},
        )
    finally:
        if owns_client:
            http_client.close()
    if response.status_code != httpx.codes.OK:
        return [
            HealthIssue("*", "odds.the_odds_api", f"/v4/sports respondio {response.status_code}")
        ]

    known_keys = {sport["key"] for sport in response.json()}
    return [
        HealthIssue(
            competition.id,
            "odds.the_odds_api",
            f"'{competition.odds.the_odds_api}' no existe en /v4/sports",
        )
        for competition in competitions
        if competition.odds.the_odds_api not in known_keys
    ]


def run_health_check() -> list[HealthIssue]:
    competitions = list(load_competitions().enabled)
    return check_football_sources(competitions) + check_odds_api_sport_keys(competitions)
