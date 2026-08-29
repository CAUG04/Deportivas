"""The data model. This module is the single source of truth for the schema.

Nothing else in the project may declare a column. Postgres DDL, DuckDB DDL,
Parquet layout and Pandera validation are all generated from ``SCHEMA``.

Two conventions worth knowing before reading:

* **Denormalised partition keys.** ``competition_id`` and ``season`` are
  repeated on child tables that would otherwise need a join to be located on
  disk. That redundancy is what lets DuckDB prune Parquet partitions, and it
  costs nothing on Postgres beyond a few bytes per row.
* **Surrogate ids.** Tables whose natural key contains a nullable component
  (a handicap line, for instance) carry a deterministic ``id`` derived from
  that key, because SQL primary keys cannot span nullable columns.
"""

from __future__ import annotations

from deportivas.contracts.types import ColumnSpec as C
from deportivas.contracts.types import IndexSpec, LogicalType, SchemaRegistry, TableSpec

T = LogicalType


def _ingestion_columns() -> tuple[C, ...]:
    return (
        C("source", T.STR, max_length=64, description="Adaptador que produjo la fila"),
        C("ingested_at", T.TIMESTAMP, description="Cuando se escribio la fila"),
    )


COMPETITIONS = TableSpec(
    name="competitions",
    description="Catalogo de competiciones, espejo de config/competitions.yaml",
    columns=(
        C("id", T.STR, primary_key=True, max_length=64),
        C("name", T.STR, max_length=128),
        C("country", T.STR, max_length=64),
        C("sport", T.STR, max_length=32),
        C("tier", T.INT),
        C("enabled", T.BOOL),
        *_ingestion_columns(),
    ),
    natural_key=("id",),
)

TEAMS = TableSpec(
    name="teams",
    description="Equipos canonicos. El nombre canonico es el de FBref cuando existe",
    columns=(
        C("id", T.STR, primary_key=True, max_length=96),
        C("canonical_name", T.STR, max_length=128),
        C("sport", T.STR, max_length=32),
        C("country", T.STR, max_length=64, nullable=True),
        *_ingestion_columns(),
    ),
    natural_key=("id",),
    indexes=(IndexSpec("ix_teams_sport_name", ("sport", "canonical_name")),),
)

TEAM_ALIASES = TableSpec(
    name="team_aliases",
    description=(
        "Reconciliacion de nombres entre fuentes. FBref, ESPN y "
        "football-data.co.uk escriben el mismo equipo de tres maneras distintas"
    ),
    columns=(
        C("source", T.STR, primary_key=True, max_length=64),
        C("sport", T.STR, primary_key=True, max_length=32),
        C("alias", T.STR, primary_key=True, max_length=128),
        C("team_id", T.STR, foreign_key="teams.id", max_length=96),
        # 'source' ya es parte de la clave primaria aqui, no se repite.
        C("ingested_at", T.TIMESTAMP),
    ),
    natural_key=("source", "sport", "alias"),
    indexes=(IndexSpec("ix_team_aliases_team", ("team_id",)),),
)

FIXTURES = TableSpec(
    name="fixtures",
    description="Un partido. El id es un hash determinista de la clave natural",
    columns=(
        C("id", T.STR, primary_key=True, max_length=64),
        C("competition_id", T.STR, foreign_key="competitions.id", max_length=64),
        C("season", T.STR, max_length=16, description="Etiqueta de temporada, p.ej. '2425'"),
        C("kickoff_utc", T.TIMESTAMP, description="Siempre UTC, nunca hora local"),
        C("home_team_id", T.STR, foreign_key="teams.id", max_length=96),
        C("away_team_id", T.STR, foreign_key="teams.id", max_length=96),
        C("status", T.STR, max_length=16),
        C("stage", T.STR, max_length=64, nullable=True, description="Fase, para copas"),
        C("matchday", T.INT, nullable=True),
        C("home_score", T.INT, nullable=True),
        C("away_score", T.INT, nullable=True),
        C("home_score_ht", T.INT, nullable=True),
        C("away_score_ht", T.INT, nullable=True),
        C("kickoff_is_estimated", T.BOOL, description="La fuente no dio hora exacta"),
        *_ingestion_columns(),
    ),
    natural_key=("competition_id", "season", "home_team_id", "away_team_id", "kickoff_utc"),
    indexes=(
        IndexSpec("ix_fixtures_competition_kickoff", ("competition_id", "kickoff_utc")),
        IndexSpec("ix_fixtures_kickoff", ("kickoff_utc",)),
        IndexSpec("ix_fixtures_status", ("status",)),
    ),
    partition_by=("competition_id", "season"),
)

TEAM_MATCH_STATS = TableSpec(
    name="team_match_stats",
    description=(
        "Estadisticas por equipo, partido y fuente. 'source' es parte de la "
        "clave: FBref y Understat estiman xG por separado y ambas filas deben "
        "convivir en vez de que la ultima en llegar borre a la otra"
    ),
    columns=(
        C("fixture_id", T.STR, primary_key=True, foreign_key="fixtures.id", max_length=64),
        C("team_id", T.STR, primary_key=True, foreign_key="teams.id", max_length=96),
        C("source", T.STR, primary_key=True, max_length=64),
        C("competition_id", T.STR, max_length=64),
        C("season", T.STR, max_length=16),
        C("is_home", T.BOOL),
        C("goals", T.INT, nullable=True),
        C("shots", T.INT, nullable=True),
        C("shots_on_target", T.INT, nullable=True),
        C("xg", T.FLOAT, nullable=True),
        C("corners", T.INT, nullable=True),
        C("yellow_cards", T.INT, nullable=True),
        C("red_cards", T.INT, nullable=True),
        C("possession", T.FLOAT, nullable=True, description="Fraccion 0-1, no porcentaje"),
        C("ingested_at", T.TIMESTAMP),
    ),
    natural_key=("fixture_id", "team_id", "source"),
    partition_by=("competition_id", "season"),
)

NFL_TEAM_GAME_STATS = TableSpec(
    name="nfl_team_game_stats",
    description=(
        "EPA/jugada y tasa de exito por equipo y partido de NFL, ofensiva y "
        "defensiva, agregadas desde play-by-play (nflfastR via nfl_data_py). "
        "'source' es parte de la clave por el mismo motivo que en team_match_stats"
    ),
    columns=(
        C("fixture_id", T.STR, primary_key=True, foreign_key="fixtures.id", max_length=64),
        C("team_id", T.STR, primary_key=True, foreign_key="teams.id", max_length=96),
        C("source", T.STR, primary_key=True, max_length=64),
        C("competition_id", T.STR, max_length=64),
        C("season", T.STR, max_length=16),
        C("is_home", T.BOOL),
        C("offensive_plays", T.INT),
        C("offensive_epa_per_play", T.FLOAT, nullable=True),
        C("offensive_success_rate", T.FLOAT, nullable=True),
        C("defensive_plays", T.INT),
        C("defensive_epa_per_play_allowed", T.FLOAT, nullable=True),
        C("defensive_success_rate_allowed", T.FLOAT, nullable=True),
        C("ingested_at", T.TIMESTAMP),
    ),
    natural_key=("fixture_id", "team_id", "source"),
    partition_by=("competition_id", "season"),
)

ODDS_SNAPSHOTS = TableSpec(
    name="odds_snapshots",
    description=(
        "Cuota observada en un instante concreto. Sin captured_at una cuota no "
        "sirve: el backtest debe usar el precio disponible al decidir, no el cierre"
    ),
    columns=(
        C("id", T.STR, primary_key=True, max_length=64),
        C("fixture_id", T.STR, foreign_key="fixtures.id", max_length=64),
        C("competition_id", T.STR, max_length=64),
        C("season", T.STR, max_length=16),
        C("bookmaker", T.STR, max_length=64),
        C("market", T.STR, max_length=32),
        C("selection", T.STR, max_length=32),
        C("line", T.FLOAT, nullable=True, description="Handicap o total; None si no aplica"),
        C("price", T.FLOAT, description="Cuota decimal"),
        C("captured_at", T.TIMESTAMP, description="Momento real de captura, no el programado"),
        C("is_closing", T.BOOL, description="Ultima captura antes del kickoff"),
        *_ingestion_columns(),
    ),
    natural_key=("fixture_id", "bookmaker", "market", "selection", "captured_at"),
    indexes=(
        IndexSpec("ix_odds_fixture_captured", ("fixture_id", "captured_at")),
        IndexSpec("ix_odds_closing", ("fixture_id", "is_closing")),
    ),
    partition_by=("competition_id", "season"),
    append_only=True,
)

FEATURES = TableSpec(
    name="features",
    description=(
        "Vector de features de un partido. as_of_timestamp es el corte de "
        "informacion: nada en el vector puede conocerse despues de ese instante"
    ),
    columns=(
        C("fixture_id", T.STR, primary_key=True, foreign_key="fixtures.id", max_length=64),
        C("feature_set", T.STR, primary_key=True, max_length=64, description="Nombre y version"),
        C("competition_id", T.STR, max_length=64),
        C("season", T.STR, max_length=16),
        C("as_of_timestamp", T.TIMESTAMP),
        C("vector", T.JSON),
        C("computed_at", T.TIMESTAMP),
        *_ingestion_columns(),
    ),
    natural_key=("fixture_id", "feature_set"),
    indexes=(IndexSpec("ix_features_fixture_asof", ("fixture_id", "as_of_timestamp")),),
    partition_by=("competition_id", "season"),
)

MODEL_REGISTRY = TableSpec(
    name="model_registry",
    description="Una fila por version de modelo entrenada. Sin metricas no se despliega",
    columns=(
        C("id", T.STR, primary_key=True, max_length=64),
        C("model_name", T.STR, max_length=64),
        C("model_version", T.STR, max_length=32),
        C("sport", T.STR, max_length=32),
        C("feature_set", T.STR, max_length=64),
        C("trained_at", T.TIMESTAMP),
        C("train_window_start", T.TIMESTAMP),
        C("train_window_end", T.TIMESTAMP),
        C("hyperparameters", T.JSON),
        C("metrics", T.JSON, description="Brier, log loss y curva de fiabilidad por ventana"),
        C("calibration_method", T.STR, max_length=32, nullable=True),
        C("git_sha", T.STR, max_length=40, nullable=True),
        *_ingestion_columns(),
    ),
    natural_key=("model_name", "model_version"),
    append_only=True,
)

PREDICTIONS = TableSpec(
    name="predictions",
    description=(
        "Probabilidad cruda y calibrada por partido, mercado y seleccion. "
        "'id' es la clave natural real (no la tupla de columnas de negocio) "
        "porque 'line' distingue predicciones del mismo mercado/seleccion "
        "(over 1.5 vs. over 2.5) y una clave natural no puede tener un "
        "componente nullable — ver el docstring de modulo sobre ids surrogados"
    ),
    columns=(
        C("id", T.STR, primary_key=True, max_length=64),
        C("fixture_id", T.STR, foreign_key="fixtures.id", max_length=64),
        C("competition_id", T.STR, max_length=64),
        C("season", T.STR, max_length=16),
        C("model_name", T.STR, max_length=64),
        C("model_version", T.STR, max_length=32),
        C("market", T.STR, max_length=32),
        C("selection", T.STR, max_length=32),
        C("line", T.FLOAT, nullable=True),
        C("prob_raw", T.FLOAT),
        C(
            "prob_calibrated",
            T.FLOAT,
            nullable=True,
            description="None si el modelo no esta calibrado",
        ),
        C(
            "as_of_timestamp",
            T.TIMESTAMP,
            description="Corte de informacion de las features usadas",
        ),
        C("predicted_at", T.TIMESTAMP),
        *_ingestion_columns(),
    ),
    natural_key=("id",),
    indexes=(IndexSpec("ix_predictions_fixture", ("fixture_id", "market")),),
    partition_by=("competition_id", "season"),
)

SIGNALS = TableSpec(
    name="signals",
    description=(
        "Una apuesta senalada. tier='descartar' es una salida valida y se "
        "persiste: saber a que no apostar es parte del producto"
    ),
    columns=(
        C("id", T.STR, primary_key=True, max_length=64),
        C("fixture_id", T.STR, foreign_key="fixtures.id", max_length=64),
        C("competition_id", T.STR, max_length=64),
        C("season", T.STR, max_length=16),
        C("market", T.STR, max_length=32),
        C("selection", T.STR, max_length=32),
        C("line", T.FLOAT, nullable=True),
        C("model_name", T.STR, max_length=64),
        C("model_version", T.STR, max_length=32),
        C("prob_model", T.FLOAT, description="Probabilidad calibrada del modelo"),
        C("prob_fair", T.FLOAT, description="Probabilidad justa del mercado, sin margen"),
        C("fair_price", T.FLOAT),
        C("entry_price", T.FLOAT),
        C("entry_bookmaker", T.STR, max_length=64),
        C("entry_captured_at", T.TIMESTAMP),
        C("devig_method", T.STR, max_length=32),
        C("edge", T.FLOAT, description="prob_model - prob_fair"),
        C("tier", T.STR, max_length=16),
        C("tier_reasons", T.JSON, description="Que condicion de tier fallo o se cumplio"),
        C("stake_fraction", T.FLOAT, description="Fraccion del bankroll, Kelly fraccionado"),
        C("created_at", T.TIMESTAMP),
        *_ingestion_columns(),
    ),
    natural_key=("fixture_id", "market", "selection", "model_name", "model_version"),
    indexes=(IndexSpec("ix_signals_fixture_tier", ("fixture_id", "tier")),),
    partition_by=("competition_id", "season"),
)

RESULTS = TableSpec(
    name="results",
    description="Liquidacion de una senal. El CLV es la metrica principal del sistema",
    columns=(
        C("id", T.STR, primary_key=True, max_length=64),
        C("signal_id", T.STR, foreign_key="signals.id", max_length=64),
        C("fixture_id", T.STR, foreign_key="fixtures.id", max_length=64),
        C("competition_id", T.STR, max_length=64),
        C("season", T.STR, max_length=16),
        C("outcome", T.STR, max_length=16, description="win/loss/push/half_win/half_loss/void"),
        C("pnl", T.FLOAT, description="En unidades de bankroll, neto del stake"),
        C("closing_price", T.FLOAT, nullable=True),
        C("closing_captured_at", T.TIMESTAMP, nullable=True),
        C("clv", T.FLOAT, nullable=True, description="entry_price/closing_price - 1"),
        C("settled_at", T.TIMESTAMP),
        *_ingestion_columns(),
    ),
    natural_key=("signal_id",),
    indexes=(IndexSpec("ix_results_fixture", ("fixture_id",)),),
    partition_by=("competition_id", "season"),
)

RAW_DOCUMENTS = TableSpec(
    name="raw_documents",
    description=(
        "Capa cruda append-only. Se escribe la respuesta original ANTES de "
        "parsearla y no se sobrescribe nunca. Features y modelos deben poder "
        "reconstruirse desde aqui sin volver a raspar"
    ),
    columns=(
        C("id", T.STR, primary_key=True, max_length=64),
        C("source", T.STR, max_length=64),
        C("endpoint", T.STR, max_length=512),
        C("params", T.JSON),
        C("fetched_at", T.TIMESTAMP),
        C("content_type", T.STR, max_length=64),
        C("content_hash", T.STR, max_length=64),
        C("content_bytes", T.BIGINT),
        C("storage_path", T.STR, max_length=512, description="Ruta relativa dentro de raw/"),
        C("status_code", T.INT, nullable=True),
    ),
    natural_key=("source", "endpoint", "content_hash", "fetched_at"),
    indexes=(IndexSpec("ix_raw_source_fetched", ("source", "fetched_at")),),
    partition_by=("source",),
    append_only=True,
)

SCHEMA = SchemaRegistry(
    tables=(
        COMPETITIONS,
        TEAMS,
        TEAM_ALIASES,
        FIXTURES,
        TEAM_MATCH_STATS,
        NFL_TEAM_GAME_STATS,
        ODDS_SNAPSHOTS,
        FEATURES,
        MODEL_REGISTRY,
        PREDICTIONS,
        SIGNALS,
        RESULTS,
        RAW_DOCUMENTS,
    )
)
