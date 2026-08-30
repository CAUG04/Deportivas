"""``deportivas`` command-line entry point.

Every ``ingest`` subcommand is thin, deliberate wiring over one already-tested
adapter method: build the repositories for the active storage backend
(``storage/factory.py``), build a rate limiter and a team-alias resolver for
the right sport, call the adapter, and write the result inside a
:class:`BufferedUnitOfWork` so a failed write never lands half a batch.

Backfill (historical, explicit ``--seasons``) and incremental/daily ingestion
are the same commands run with a different season list — Fase 8's automation
is what schedules the narrow "current season" runs; this CLI does not guess
season windows on its own; run it from a workflow matrix that iterates
competitions and sources, matching ``config/competitions.yaml``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from deportivas.backtest.report import MetricSummary

import pandas as pd
import typer

from deportivas.config.catalog import load_competitions
from deportivas.config.settings import get_settings
from deportivas.contracts.tables import COMPETITIONS, FIXTURES, ODDS_SNAPSHOTS, TEAM_ALIASES, TEAMS
from deportivas.contracts.tables import NFL_TEAM_GAME_STATS as NFL_TEAM_GAME_STATS_TABLE
from deportivas.contracts.tables import TEAM_MATCH_STATS as TEAM_MATCH_STATS_TABLE
from deportivas.contracts.types import TableSpec
from deportivas.ingest.aliases import TeamAliasResolver
from deportivas.ingest.ratelimit import RateLimiter
from deportivas.storage.factory import get_raw_document_repository, get_table_repository
from deportivas.storage.unit_of_work import BufferedUnitOfWork

app = typer.Typer(help="Ingesta, features y backtest de la plataforma de pronosticos.")
ingest_app = typer.Typer(help="Un comando por adaptador de fuente.")
features_app = typer.Typer(help="Un comando por pipeline de features (uno por deporte).")
models_app = typer.Typer(help="Un comando por modelo entrenado (walk-forward por temporada).")
signals_app = typer.Typer(help="Generacion de senales a partir de predictions + odds_snapshots.")
backtest_app = typer.Typer(help="Liquidacion de senales y reporte de CLV/ROI.")
export_app = typer.Typer(help="Export estatico de JSON para el frontend (Fase 7).")
app.add_typer(ingest_app, name="ingest")
app.add_typer(features_app, name="features")
app.add_typer(models_app, name="models")
app.add_typer(signals_app, name="signals")
app.add_typer(backtest_app, name="backtest")
app.add_typer(export_app, name="export")


def _seasons_list(seasons: str) -> list[str]:
    return [s.strip() for s in seasons.split(",") if s.strip()]


def _int_seasons_list(seasons: str) -> list[int]:
    return [int(s) for s in _seasons_list(seasons)]


def _alias_resolver(sport: str) -> TeamAliasResolver:
    return TeamAliasResolver(
        get_table_repository(TEAMS), get_table_repository(TEAM_ALIASES), sport=sport
    )


def _fbref_rate_limiter() -> RateLimiter:
    return RateLimiter(get_settings().fbref_min_delay_seconds)


def _persist(table_spec: TableSpec, df: pd.DataFrame, *, label: str) -> None:
    if df.empty:
        typer.echo(f"{label}: 0 filas (nada que escribir)")
        return
    repo = get_table_repository(table_spec)
    with BufferedUnitOfWork() as uow:
        uow.stage(repo, df)
        written = uow.commit()
    typer.echo(f"{label}: {written} filas escritas")


CompetitionId = Annotated[
    str, typer.Option(help="Id de config/competitions.yaml, p.ej. eng-premier-league")
]
Seasons = Annotated[str, typer.Option(help="Temporadas separadas por coma, p.ej. 2223,2324,2425")]
LeagueKey = Annotated[
    str, typer.Option(help="Clave de liga en la fuente (ver sources: en competitions.yaml)")
]


@ingest_app.command("fbref-schedule")
def fbref_schedule(
    competition_id: CompetitionId, fbref_league: LeagueKey, seasons: Seasons
) -> None:
    """FBref: calendario y resultados -> fixtures."""
    from deportivas.ingest.sources.fbref import FBrefSource

    source = FBrefSource(
        raw_repo=get_raw_document_repository(),
        rate_limiter=_fbref_rate_limiter(),
        data_dir=get_settings().cache_dir / "fbref",
        aliases=_alias_resolver("football"),
    )
    df = source.fetch_schedule(
        competition_id=competition_id, fbref_league=fbref_league, seasons=_seasons_list(seasons)
    )
    _persist(FIXTURES, df, label="fbref fixtures")


@ingest_app.command("fbref-stats")
def fbref_stats(competition_id: CompetitionId, fbref_league: LeagueKey, seasons: Seasons) -> None:
    """FBref: estadisticas por equipo y partido -> team_match_stats."""
    from deportivas.ingest.sources.fbref import FBrefSource

    source = FBrefSource(
        raw_repo=get_raw_document_repository(),
        rate_limiter=_fbref_rate_limiter(),
        data_dir=get_settings().cache_dir / "fbref",
        aliases=_alias_resolver("football"),
    )
    df = source.fetch_team_match_stats(
        competition_id=competition_id, fbref_league=fbref_league, seasons=_seasons_list(seasons)
    )
    _persist(TEAM_MATCH_STATS_TABLE, df, label="fbref team_match_stats")


@ingest_app.command("understat-stats")
def understat_stats(
    competition_id: CompetitionId, understat_league: LeagueKey, seasons: Seasons
) -> None:
    """Understat: xG por partido -> team_match_stats."""
    from deportivas.ingest.sources.understat import UnderstatSource

    source = UnderstatSource(
        raw_repo=get_raw_document_repository(),
        rate_limiter=RateLimiter(0.0),
        data_dir=get_settings().cache_dir / "understat",
        aliases=_alias_resolver("football"),
    )
    df = source.fetch_team_match_stats(
        competition_id=competition_id,
        understat_league=understat_league,
        seasons=_seasons_list(seasons),
    )
    _persist(TEAM_MATCH_STATS_TABLE, df, label="understat team_match_stats")


@ingest_app.command("espn-schedule")
def espn_schedule(competition_id: CompetitionId, espn_league: LeagueKey, seasons: Seasons) -> None:
    """ESPN: calendario (sin resultado final) -> fixtures. Unica fuente para Liga BetPlay."""
    from deportivas.ingest.sources.espn import EspnSource

    source = EspnSource(
        raw_repo=get_raw_document_repository(),
        rate_limiter=RateLimiter(0.0),
        data_dir=get_settings().cache_dir / "espn",
        aliases=_alias_resolver("football"),
    )
    df = source.fetch_schedule(
        competition_id=competition_id, espn_league=espn_league, seasons=_seasons_list(seasons)
    )
    _persist(FIXTURES, df, label="espn fixtures")


@ingest_app.command("footballdata-games")
def footballdata_games(
    competition_id: CompetitionId, match_history_league: LeagueKey, seasons: Seasons
) -> None:
    """football-data.co.uk: resultados historicos -> fixtures."""
    from deportivas.ingest.sources.footballdata import FootballDataSource

    source = FootballDataSource(
        raw_repo=get_raw_document_repository(),
        rate_limiter=RateLimiter(0.0),
        data_dir=get_settings().cache_dir / "footballdata",
        aliases=_alias_resolver("football"),
    )
    df = source.fetch_games(
        competition_id=competition_id,
        match_history_league=match_history_league,
        seasons=_seasons_list(seasons),
    )
    _persist(FIXTURES, df, label="footballdata fixtures")


@ingest_app.command("footballdata-odds")
def footballdata_odds(
    competition_id: CompetitionId, match_history_league: LeagueKey, seasons: Seasons
) -> None:
    """football-data.co.uk: cuotas 1X2 historicas (incluye cierre Pinnacle) -> odds_snapshots."""
    from deportivas.ingest.sources.footballdata import FootballDataSource

    source = FootballDataSource(
        raw_repo=get_raw_document_repository(),
        rate_limiter=RateLimiter(0.0),
        data_dir=get_settings().cache_dir / "footballdata",
        aliases=_alias_resolver("football"),
    )
    df = source.fetch_1x2_odds(
        competition_id=competition_id,
        match_history_league=match_history_league,
        seasons=_seasons_list(seasons),
    )
    _persist(ODDS_SNAPSHOTS, df, label="footballdata odds")


@ingest_app.command("nfl-schedule")
def nfl_schedule(competition_id: CompetitionId, seasons: Seasons) -> None:
    """nfl_data_py: calendario NFL -> fixtures."""
    from deportivas.ingest.sources.nfl import NflSource

    source = NflSource(
        raw_repo=get_raw_document_repository(),
        rate_limiter=RateLimiter(0.0),
        aliases=_alias_resolver("american_football"),
    )
    df = source.fetch_schedules(competition_id=competition_id, seasons=_int_seasons_list(seasons))
    _persist(FIXTURES, df, label="nfl fixtures")


@ingest_app.command("nfl-team-game-stats")
def nfl_team_game_stats(competition_id: CompetitionId, seasons: Seasons) -> None:
    """nfl_data_py: play-by-play agregado -> nfl_team_game_stats.

    Requiere que ``nfl-schedule`` ya haya corrido para estas temporadas: cada
    partido de play-by-play se enlaza a su fixture ya ingerido.
    """
    from deportivas.ingest.sources.nfl import NflSource

    fixtures = get_table_repository(FIXTURES).read(filters={"competition_id": competition_id})
    source = NflSource(
        raw_repo=get_raw_document_repository(),
        rate_limiter=RateLimiter(0.0),
        aliases=_alias_resolver("american_football"),
    )
    df = source.fetch_team_game_stats(seasons=_int_seasons_list(seasons), fixtures=fixtures)
    _persist(NFL_TEAM_GAME_STATS_TABLE, df, label="nfl team_game_stats")


@ingest_app.command("nba-schedule")
def nba_schedule(competition_id: CompetitionId, seasons: Seasons) -> None:
    """sportsdataverse: calendario NBA -> fixtures."""
    from deportivas.ingest.sources.sportsdataverse_source import NbaSource

    source = NbaSource(
        raw_repo=get_raw_document_repository(),
        rate_limiter=RateLimiter(0.0),
        aliases=_alias_resolver("basketball"),
    )
    df = source.fetch_schedule(competition_id=competition_id, seasons=_int_seasons_list(seasons))
    _persist(FIXTURES, df, label="nba fixtures")


@ingest_app.command("nhl-schedule")
def nhl_schedule(competition_id: CompetitionId, seasons: Seasons) -> None:
    """sportsdataverse: calendario NHL -> fixtures."""
    from deportivas.ingest.sources.sportsdataverse_source import NhlSource

    source = NhlSource(
        raw_repo=get_raw_document_repository(),
        rate_limiter=RateLimiter(0.0),
        aliases=_alias_resolver("ice_hockey"),
    )
    df = source.fetch_schedule(competition_id=competition_id, seasons=_int_seasons_list(seasons))
    _persist(FIXTURES, df, label="nhl fixtures")


@ingest_app.command("mlb-schedule")
def mlb_schedule(
    competition_id: CompetitionId,
    season: Annotated[int, typer.Option(help="Temporada MLB, p.ej. 2025")],
    teams: Annotated[
        str, typer.Option(help="Abreviaturas de equipo separadas por coma, p.ej. PHI,ATL")
    ],
) -> None:
    """pybaseball: calendario MLB (schedule_and_record, por equipo) -> fixtures."""
    from deportivas.ingest.sources.pybaseball_source import PybaseballSource

    source = PybaseballSource(
        raw_repo=get_raw_document_repository(),
        rate_limiter=RateLimiter(0.0),
        aliases=_alias_resolver("baseball"),
    )
    df = source.fetch_schedule(
        competition_id=competition_id, season=season, team_abbreviations=_seasons_list(teams)
    )
    _persist(FIXTURES, df, label="mlb fixtures")


@ingest_app.command("odds-snapshot")
def odds_snapshot(
    competition_id: CompetitionId,
    sport_key: Annotated[
        str, typer.Option(help="Clave de deporte de The Odds API, p.ej. soccer_epl")
    ],
    season: Annotated[
        str, typer.Option(help="Temporada de config/competitions.yaml para este calendario")
    ],
    market_map: Annotated[
        str,
        typer.Option(
            help="raw:nuestro separados por coma, p.ej. h2h:1x2,spreads:asian_handicap,totals:over_under"
        ),
    ],
) -> None:
    """The Odds API: snapshot de cuotas en vivo -> odds_snapshots. Requiere DEPORTIVAS_THE_ODDS_API_KEY."""
    from deportivas.ingest.sources.theoddsapi import TheOddsApiSource

    settings = get_settings()
    if not settings.has_odds_api_key:
        raise typer.BadParameter("DEPORTIVAS_THE_ODDS_API_KEY no esta configurada")
    assert settings.the_odds_api_key is not None

    parsed_map = dict(pair.split(":", 1) for pair in market_map.split(",") if pair.strip())
    source = TheOddsApiSource(
        raw_repo=get_raw_document_repository(),
        rate_limiter=RateLimiter(0.0),
        aliases=_alias_resolver("football"),
        api_key=settings.the_odds_api_key.get_secret_value(),
    )
    df = source.fetch_odds(
        competition_id=competition_id, sport_key=sport_key, season=season, market_map=parsed_map
    )
    _persist(ODDS_SNAPSHOTS, df, label="theoddsapi odds")


@ingest_app.command("mark-closing")
def mark_closing_command(competition_id: CompetitionId) -> None:
    """Marca is_closing=True sobre el ultimo snapshot antes del kickoff de
    cada fixture ya arrancado -- optimizacion, no dependencia de
    correctitud: backtest/settlement.py ya calcula el precio de cierre en
    tiempo de lectura si esto nunca corre. Seguro de correr a diario contra
    toda la competicion: salta los fixtures ya marcados."""
    from deportivas.ingest.closing import mark_closing_lines

    changed = mark_closing_lines(competition_id)
    typer.echo(f"odds_snapshots: {changed} fila(s) marcadas is_closing")


@features_app.command("compute-football")
def compute_football_features_command(competition_id: CompetitionId) -> None:
    """Elo, ataque/defensa, xG rolling, descanso y congestion -> features (football_v1)."""
    from deportivas.features.football.pipeline import compute_and_write_football_features

    written = compute_and_write_football_features(competition_id)
    typer.echo(f"football_v1: {written} filas escritas")


@features_app.command("compute-nfl")
def compute_nfl_features_command(competition_id: CompetitionId) -> None:
    """EPA/jugada, success rate, descanso y DVOA aproximado -> features (nfl_v1)."""
    from deportivas.features.nfl.pipeline import compute_and_write_nfl_features

    written = compute_and_write_nfl_features(competition_id)
    typer.echo(f"nfl_v1: {written} filas escritas")


@features_app.command("compute-nba")
def compute_nba_features_command(competition_id: CompetitionId) -> None:
    """Descanso, back-to-back y margen de anotacion rolling -> features (nba_v1)."""
    from deportivas.features.nba.pipeline import compute_and_write_nba_features

    written = compute_and_write_nba_features(competition_id)
    typer.echo(f"nba_v1: {written} filas escritas")


@features_app.command("compute-nhl")
def compute_nhl_features_command(competition_id: CompetitionId) -> None:
    """Descanso, back-to-back y margen de anotacion rolling -> features (nhl_v1)."""
    from deportivas.features.nhl.pipeline import compute_and_write_nhl_features

    written = compute_and_write_nhl_features(competition_id)
    typer.echo(f"nhl_v1: {written} filas escritas")


@features_app.command("compute-mlb")
def compute_mlb_features_command(competition_id: CompetitionId) -> None:
    """Descanso, back-to-back (dobleheader) y margen de anotacion rolling -> features (mlb_v1)."""
    from deportivas.features.mlb.pipeline import compute_and_write_mlb_features

    written = compute_and_write_mlb_features(competition_id)
    typer.echo(f"mlb_v1: {written} filas escritas")


CalibrationMethod = Annotated[
    str | None,
    typer.Option(help="isotonic | platt. Sin especificar, usa config/thresholds.yaml"),
]


def _echo_training_result(label: str, written: list[int]) -> None:
    if not written:
        typer.echo(f"{label}: 0 ventanas entrenadas (datos insuficientes)")
        return
    typer.echo(f"{label}: {len(written)} ventana(s), {sum(written)} filas de predicciones")


@models_app.command("train-football")
def train_football_model_command(
    competition_id: CompetitionId, calibration_method: CalibrationMethod = None
) -> None:
    """Poisson bivariante walk-forward por temporada -> model_registry + predictions
    (1x2, over_under, btts). Requiere que fixtures ya este ingerido para esta
    competicion, con al menos dos temporadas terminadas."""
    from deportivas.models.football.train import compute_and_write_football_models

    written = compute_and_write_football_models(
        competition_id, calibration_method=calibration_method
    )
    _echo_training_result("football poisson", written)


@models_app.command("train-nfl")
def train_nfl_model_command(
    competition_id: CompetitionId, calibration_method: CalibrationMethod = None
) -> None:
    """Clasificador logistico walk-forward sobre nfl_v1 -> model_registry + predictions
    (moneyline). Requiere que features compute-nfl ya haya corrido para esta competicion."""
    from deportivas.models.nfl.train import compute_and_write_nfl_moneyline_model

    written = compute_and_write_nfl_moneyline_model(
        competition_id, calibration_method=calibration_method
    )
    _echo_training_result("nfl moneyline", written)


@models_app.command("train-nba")
def train_nba_model_command(
    competition_id: CompetitionId, calibration_method: CalibrationMethod = None
) -> None:
    """Clasificador logistico walk-forward sobre nba_v1 -> model_registry + predictions
    (moneyline). Requiere que features compute-nba ya haya corrido para esta competicion."""
    from deportivas.models.nba.train import compute_and_write_nba_moneyline_model

    written = compute_and_write_nba_moneyline_model(
        competition_id, calibration_method=calibration_method
    )
    _echo_training_result("nba moneyline", written)


@models_app.command("train-nhl")
def train_nhl_model_command(
    competition_id: CompetitionId, calibration_method: CalibrationMethod = None
) -> None:
    """Clasificador logistico walk-forward sobre nhl_v1 -> model_registry + predictions
    (moneyline). Requiere que features compute-nhl ya haya corrido para esta competicion."""
    from deportivas.models.nhl.train import compute_and_write_nhl_moneyline_model

    written = compute_and_write_nhl_moneyline_model(
        competition_id, calibration_method=calibration_method
    )
    _echo_training_result("nhl moneyline", written)


@models_app.command("train-mlb")
def train_mlb_model_command(
    competition_id: CompetitionId, calibration_method: CalibrationMethod = None
) -> None:
    """Clasificador logistico walk-forward sobre mlb_v1 -> model_registry + predictions
    (moneyline). Requiere que features compute-mlb ya haya corrido para esta competicion."""
    from deportivas.models.mlb.train import compute_and_write_mlb_moneyline_model

    written = compute_and_write_mlb_moneyline_model(
        competition_id, calibration_method=calibration_method
    )
    _echo_training_result("mlb moneyline", written)


@signals_app.command("generate")
def generate_signals_command(competition_id: CompetitionId) -> None:
    """Une predictions con odds_snapshots, quita el margen contra Pinnacle (o su
    fallback), calcula el edge, clasifica el tier de confianza y el stake de
    Kelly fraccionado -> signals. Incluye las senales 'descartar': saber a que
    no apostar tambien se persiste. Requiere que ya existan predictions para
    esta competicion (comandos ``models train-*``)."""
    from deportivas.signals.generate import compute_and_write_signals

    written = compute_and_write_signals(competition_id)
    typer.echo(f"signals: {written} fila(s) escritas")


def _format_pct(value: float | None) -> str:
    return "sin datos" if value is None else f"{value:+.2%}"


def _echo_metric_summary(label: str, summary: MetricSummary) -> None:
    if summary.n == 0:
        typer.echo(f"  {label}: sin datos liquidados")
        return
    ci_text = ""
    if summary.clv_ci is not None:
        lower, upper = summary.clv_ci
        ci_text = f" (IC [{lower:+.2%}, {upper:+.2%}])"
    typer.echo(
        f"  {label}: n={summary.n}  CLV medio={_format_pct(summary.mean_clv)}{ci_text}  "
        f"ROI={_format_pct(summary.roi)}"
    )


@backtest_app.command("settle")
def settle_backtest_command(competition_id: CompetitionId) -> None:
    """Liquida cada senal cuyo partido ya termino con marcador conocido:
    outcome, pnl y CLV (cuota de entrada contra la de cierre) -> results.
    Incluye las senales 'descartar'/'baja' (pnl siempre 0 porque no llevan
    stake): su CLV es la verificacion honesta de si el sistema de tiers esta
    descartando lo que debia. Requiere que ya existan signals (``deportivas
    signals generate``) para esta competicion."""
    from deportivas.backtest.settlement import compute_and_write_results

    written = compute_and_write_results(competition_id)
    typer.echo(f"results: {written} fila(s) escritas")


@backtest_app.command("report")
def backtest_report_command(competition_id: CompetitionId) -> None:
    """CLV medio -la metrica principal del proyecto, no el pnl- y ROI,
    globales y desglosados por tier y por mercado, junto con las baselines
    always_favourite/random calculadas en caliente sobre los mismos
    partidos. El intervalo de confianza de CLV se omite por debajo de
    ``backtest.min_matches_per_window`` liquidaciones. Requiere haber
    corrido ``deportivas backtest settle`` primero."""
    from deportivas.backtest.report import build_backtest_report

    report = build_backtest_report(competition_id)

    typer.echo("-- global --")
    _echo_metric_summary("total", report.overall)

    if report.by_tier:
        typer.echo("-- por tier --")
        for tier, summary in report.by_tier.items():
            _echo_metric_summary(tier, summary)

    if report.by_market:
        typer.echo("-- por mercado --")
        for market, summary in report.by_market.items():
            _echo_metric_summary(market, summary)

    typer.echo("-- baselines (stake plano de 1 unidad) --")
    for name, summary in report.baselines.items():
        _echo_metric_summary(name, summary)


@export_app.command("run")
def export_run_command(
    competition_id: Annotated[
        str | None,
        typer.Option(help="Si se omite, exporta todas las competiciones habilitadas."),
    ] = None,
) -> None:
    """Escribe JSON pre-calculado (competitions, y signals/backtest por
    competicion, con todos los tiers -incluido descartar-) bajo
    frontend/public/data/ -- lo que el frontend estatico de la Fase 7
    realmente consume, sin necesitar un servidor corriendo."""
    from deportivas.export.json_export import export_all, export_competition

    written = export_competition(competition_id) if competition_id is not None else export_all()
    for label, path in written.items():
        typer.echo(f"{label}: {path}")


@app.command("seed-competitions")
def seed_competitions() -> None:
    """Escribe config/competitions.yaml en la tabla competitions."""
    now = datetime.now(UTC)
    rows = [
        {
            "id": competition.id,
            "name": competition.name,
            "country": competition.country,
            "sport": competition.sport.value,
            "tier": competition.tier,
            "enabled": competition.enabled,
            "source": "config",
            "ingested_at": now,
        }
        for competition in load_competitions().competitions
    ]
    _persist(COMPETITIONS, pd.DataFrame(rows), label="competitions")


@app.command("current-seasons")
def current_seasons_command(
    competition_id: CompetitionId,
    count: Annotated[
        int, typer.Option(help="Cuantas temporadas recientes, de la mas nueva a la mas vieja")
    ] = 2,
) -> None:
    """Imprime, separadas por coma, las temporadas recientes de esta
    competicion -- lo que un workflow de ingesta incremental (Fase 8) pasa
    directo a --seasons, sin adivinar formatos de temporada a mano. Backfill
    e incremental son el mismo comando de ingesta; solo cambia esta lista."""
    from deportivas.domain.seasons import season_labels

    competition = load_competitions().get(competition_id)
    typer.echo(",".join(season_labels(competition, count=count)))


if __name__ == "__main__":  # pragma: no cover - invocado por el entry point, no por los tests
    app()
