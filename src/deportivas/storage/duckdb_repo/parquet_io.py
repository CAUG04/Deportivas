"""Low-level partitioned Parquet I/O. No business logic lives here.

Layout on disk, under ``settings.parquet_dir``::

    <table_name>/data.parquet                                  (no partition_by)
    <table_name>/<col1>=<val1>/<col2>=<val2>/data.parquet       (Hive-style, partition_by)

Hive-style partitioning is what lets DuckDB prune files by competition/season
instead of scanning the whole table on every query.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.parquet as pq

from deportivas.contracts.duckdb_adapter import arrow_schema
from deportivas.contracts.types import TableSpec

if TYPE_CHECKING:
    import pandas as pd


def table_dir(base_dir: Path, spec: TableSpec) -> Path:
    return base_dir / spec.name


def partition_dir(base_dir: Path, spec: TableSpec, row: dict[str, object]) -> Path:
    path = table_dir(base_dir, spec)
    for col in spec.partition_by:
        value = row[col]
        if value is None:
            raise ValueError(f"{spec.name}.{col}: columna de particion no puede ser None")
        # Sanea el valor para que nunca introduzca un separador de ruta: una
        # temporada o id de competicion con '/' partiria el layout Hive.
        safe_value = str(value).replace("/", "_")
        path = path / f"{col}={safe_value}"
    return path


def partition_file(base_dir: Path, spec: TableSpec, row: dict[str, object]) -> Path:
    return partition_dir(base_dir, spec, row) / "data.parquet"


def empty_frame(spec: TableSpec) -> pd.DataFrame:
    schema = arrow_schema(spec)
    # pyarrow no trae stubs de tipos: to_pandas() se ve como Any para mypy.
    return cast("pd.DataFrame", pa.Table.from_pylist([], schema=schema).to_pandas())


def read_parquet(path: Path, spec: TableSpec) -> pd.DataFrame:
    if not path.exists():
        return empty_frame(spec)
    table = pq.read_table(path, schema=arrow_schema(spec))
    return cast("pd.DataFrame", table.to_pandas())


def write_parquet_atomic(path: Path, table: pa.Table) -> None:
    """Writes ``table`` to ``path`` via a temp file + atomic rename.

    A crash or a concurrent job mid-write must never leave ``path`` holding a
    half-written file: readers would silently see a truncated partition.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(table, tmp_path)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def existing_partition_files(base_dir: Path, spec: TableSpec) -> list[Path]:
    directory = table_dir(base_dir, spec)
    if not directory.exists():
        return []
    return sorted(directory.rglob("data.parquet"))
