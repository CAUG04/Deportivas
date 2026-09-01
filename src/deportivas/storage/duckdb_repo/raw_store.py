"""The append-only raw layer: :class:`RawDocumentRepository` over local disk.

This is the single most important storage decision in the project (Fase 10):
every payload a source ever returns is written here, byte for byte, before
anything parses it. Nothing here is ever overwritten or deleted. Features and
models must be reconstructible from this layer alone, without re-scraping.

Layout::

    <raw_dir>/<source>/<content_hash[:2]>/<content_hash>.<ext>   payload bytes
    <parquet_dir>/raw_documents/source=<source>/data.parquet     metadata (via ParquetTableRepository)

The hash-prefixed subdirectory keeps any single directory from accumulating
too many files as a competition's history grows.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from deportivas.contracts.tables import RAW_DOCUMENTS
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository

_EXTENSION_BY_CONTENT_TYPE = {
    "text/html": "html",
    "application/json": "json",
    "text/csv": "csv",
    "application/octet-stream": "bin",
}


def _extension_for(content_type: str) -> str:
    return _EXTENSION_BY_CONTENT_TYPE.get(content_type, "bin")


class ParquetRawDocumentRepository:
    """Implements :class:`deportivas.storage.protocols.RawDocumentRepository`."""

    def __init__(self, raw_dir: Path, parquet_dir: Path) -> None:
        self._raw_dir = raw_dir
        self._metadata = ParquetTableRepository(
            RAW_DOCUMENTS, parquet_dir, temporal_column="fetched_at"
        )

    def store(
        self,
        *,
        source: str,
        endpoint: str,
        params: dict[str, object],
        content: bytes,
        content_type: str,
        status_code: int | None,
        fetched_at: datetime,
    ) -> str:
        content_hash = hashlib.sha256(content).hexdigest()
        extension = _extension_for(content_type)
        payload_path = self._raw_dir / source / content_hash[:2] / f"{content_hash}.{extension}"
        if not payload_path.exists():
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_bytes(content)
        # Si el hash ya existia (misma respuesta exacta que una vez previa),
        # no reescribimos el archivo: es contenido identico, byte a byte.

        document_id = str(uuid.uuid4())
        row = pd.DataFrame(
            [
                {
                    "id": document_id,
                    "source": source,
                    "endpoint": endpoint,
                    "params": json.dumps(params, sort_keys=True, default=str),
                    "fetched_at": fetched_at,
                    "content_type": content_type,
                    "content_hash": content_hash,
                    "content_bytes": len(content),
                    "storage_path": str(payload_path.relative_to(self._raw_dir)),
                    "status_code": status_code,
                }
            ]
        )
        self._metadata.write(row)
        return document_id

    def read(self, document_id: str) -> bytes:
        rows = self._metadata.read(filters={"id": document_id})
        if rows.empty:
            raise KeyError(f"raw_documents: no existe el documento {document_id!r}")
        storage_path = str(rows.iloc[0]["storage_path"])
        return (self._raw_dir / storage_path).read_bytes()

    def find(
        self, *, source: str, endpoint: str | None = None, since: datetime | None = None
    ) -> pd.DataFrame:
        filters: dict[str, object] = {"source": source}
        if endpoint is not None:
            filters["endpoint"] = endpoint
        rows = self._metadata.read(filters=filters, as_of=None)
        if since is not None:
            rows = rows[rows["fetched_at"] >= since]
        return rows
