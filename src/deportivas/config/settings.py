"""Runtime settings.

Everything configurable at *deployment* time lives here and comes from the
environment. Everything configurable at *modelling* time (thresholds, leagues,
markets) lives in ``config/*.yaml`` and is loaded by :mod:`deportivas.config.catalog`.

No secret is ever hard-coded. ``.env.example`` documents every variable.
"""

from __future__ import annotations

import functools
from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz del repositorio: .../src/deportivas/config/settings.py -> tres niveles arriba.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"


class StorageBackend(StrEnum):
    """Which repository implementation is active.

    ``duckdb`` is production (Parquet files, zero cost, no server). ``postgres``
    exists so the migration documented in the README is a swap of one
    implementation rather than a rewrite.
    """

    DUCKDB = "duckdb"
    POSTGRES = "postgres"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEPORTIVAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Almacenamiento ---------------------------------------------------
    storage_backend: StorageBackend = StorageBackend.DUCKDB
    data_dir: Path = PROJECT_ROOT / "data"
    duckdb_path: Path = PROJECT_ROOT / "data" / "deportivas.duckdb"
    database_url: str = "postgresql+psycopg://deportivas:deportivas@localhost:5432/deportivas"

    # --- Fuentes ----------------------------------------------------------
    the_odds_api_key: SecretStr | None = None
    http_user_agent: str = "deportivas/0.1 (uso personal)"
    fbref_min_delay_seconds: float = Field(default=4.0, ge=0.0)

    # --- Operacion --------------------------------------------------------
    log_level: str = "INFO"
    freshness_warning_hours: int = Field(default=36, gt=0)
    config_dir: Path = CONFIG_DIR

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level debe ser uno de {sorted(allowed)}")
        return upper

    # --- Rutas derivadas --------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        """Append-only raw layer: original payloads, never overwritten."""
        return self.data_dir / "raw"

    @property
    def parquet_dir(self) -> Path:
        """Partitioned Parquet lake, published as GitHub Release assets."""
        return self.data_dir / "parquet"

    @property
    def cache_dir(self) -> Path:
        """Scraper cache. Safe to delete; the raw layer is the real archive."""
        return self.data_dir / "cache"

    @property
    def export_dir(self) -> Path:
        """Pre-computed JSON consumed by the static frontend."""
        return PROJECT_ROOT / "frontend" / "public" / "data"

    @property
    def has_odds_api_key(self) -> bool:
        return self.the_odds_api_key is not None and bool(
            self.the_odds_api_key.get_secret_value().strip()
        )

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.raw_dir, self.parquet_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
