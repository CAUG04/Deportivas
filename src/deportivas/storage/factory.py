"""Picks the concrete repository implementation for the active storage backend.

Every caller in the project gets its repositories through here, never by
importing ``ParquetTableRepository`` or ``SqlTableRepository`` directly. That
is what makes the DuckDB -> Postgres migration a one-file change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deportivas.config.settings import StorageBackend, get_settings
from deportivas.storage.duckdb_repo.raw_store import ParquetRawDocumentRepository
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository
from deportivas.storage.sql_repo.engine import get_engine
from deportivas.storage.sql_repo.repository import SqlTableRepository

if TYPE_CHECKING:
    from deportivas.contracts.types import TableSpec
    from deportivas.storage.protocols import RawDocumentRepository, TableRepository


def get_table_repository(spec: TableSpec, *, temporal_column: str | None = None) -> TableRepository:
    settings = get_settings()
    if settings.storage_backend is StorageBackend.DUCKDB:
        return ParquetTableRepository(spec, settings.parquet_dir, temporal_column=temporal_column)
    if settings.storage_backend is StorageBackend.POSTGRES:
        return SqlTableRepository(spec, get_engine(), temporal_column=temporal_column)
    raise NotImplementedError(
        f"backend de almacenamiento sin implementar: {settings.storage_backend}"
    )


def get_raw_document_repository() -> RawDocumentRepository:
    settings = get_settings()
    if settings.storage_backend is StorageBackend.DUCKDB:
        settings.ensure_directories()
        return ParquetRawDocumentRepository(settings.raw_dir, settings.parquet_dir)
    raise NotImplementedError(
        "La capa cruda append-only solo esta implementada sobre DuckDB/Parquet; "
        "Postgres es el backend de desarrollo local para las tablas reconciliadas, "
        "no para el archivo crudo."
    )
