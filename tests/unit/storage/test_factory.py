"""storage.factory: picks the repository implementation by settings.storage_backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from deportivas.config.settings import Settings, StorageBackend
from deportivas.contracts.tables import COMPETITIONS
from deportivas.storage.duckdb_repo.raw_store import ParquetRawDocumentRepository
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository
from deportivas.storage.factory import get_raw_document_repository, get_table_repository
from deportivas.storage.sql_repo.repository import SqlTableRepository


def test_duckdb_backend_returns_parquet_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "deportivas.storage.factory.get_settings",
        lambda: Settings(_env_file=None, storage_backend=StorageBackend.DUCKDB, data_dir=tmp_path),
    )
    repo = get_table_repository(COMPETITIONS)
    assert isinstance(repo, ParquetTableRepository)


def test_duckdb_backend_returns_parquet_raw_document_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "deportivas.storage.factory.get_settings",
        lambda: Settings(_env_file=None, storage_backend=StorageBackend.DUCKDB, data_dir=tmp_path),
    )
    repo = get_raw_document_repository()
    assert isinstance(repo, ParquetRawDocumentRepository)


def test_postgres_backend_returns_sql_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deportivas.storage.factory.get_settings",
        lambda: Settings(_env_file=None, storage_backend=StorageBackend.POSTGRES),
    )
    repo = get_table_repository(COMPETITIONS)
    assert isinstance(repo, SqlTableRepository)


def test_postgres_backend_has_no_raw_document_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deportivas.storage.factory.get_settings",
        lambda: Settings(_env_file=None, storage_backend=StorageBackend.POSTGRES),
    )
    with pytest.raises(NotImplementedError, match="DuckDB/Parquet"):
        get_raw_document_repository()
