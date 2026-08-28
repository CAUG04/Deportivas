"""Generic DuckDB/Parquet implementation of :class:`TableRepository`.

One class, parametrised by a :class:`TableSpec`, backs every table. There is
no per-table subclass to maintain: adding a column to ``contracts/tables.py``
is enough for this repository to pick it up.

Upsert semantics: rows are merged into each affected partition file on the
table's natural key, keeping the most recently ingested row
(``ingested_at``) — every non-append-only table in ``contracts/tables.py``
carries that column, which is what makes re-running an ingestion job
idempotent (rule 1 of Fase 1). Append-only tables (``odds_snapshots``,
``raw_documents``) skip the merge and always append, because overwriting a
past odds snapshot would make CLV unrecoverable.

JSON columns (``LogicalType.JSON``) are stored as JSON-serialised strings —
callers must ``json.dumps`` before writing and ``json.loads`` after reading;
Parquet has no portable native JSON type.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, cast

import duckdb
import pandas as pd
import pyarrow as pa

from deportivas.contracts.duckdb_adapter import arrow_schema
from deportivas.contracts.types import TableSpec
from deportivas.storage.duckdb_repo import parquet_io
from deportivas.storage.validation import validate_rows

if TYPE_CHECKING:
    from pathlib import Path


def dataframe_to_arrow(df: pd.DataFrame, spec: TableSpec) -> pa.Table:
    ordered = df[list(spec.column_names)]
    return pa.Table.from_pandas(ordered, schema=arrow_schema(spec), preserve_index=False)


class ParquetTableRepository:
    """Implements :class:`deportivas.storage.protocols.TableRepository`."""

    def __init__(
        self,
        spec: TableSpec,
        base_dir: Path,
        *,
        temporal_column: str | None = None,
    ) -> None:
        self.spec = spec
        self.table_name = spec.name
        self._base_dir = base_dir
        self._temporal_column = temporal_column
        if temporal_column is not None and temporal_column not in spec.column_names:
            raise ValueError(f"{spec.name}: columna temporal {temporal_column!r} no existe")

    # -- write --------------------------------------------------------

    def write(self, rows: pd.DataFrame) -> int:
        if rows.empty:
            return 0
        valid = validate_rows(self.spec, rows)
        if valid.empty:
            return 0

        written = 0
        for partition_row, group in self._partition_groups(valid):
            path = self._partition_path(partition_row)
            existing = parquet_io.read_parquet(path, self.spec)
            combined = pd.concat([existing, group], ignore_index=True)
            if not self.spec.append_only:
                # Cada tabla mutable trae 'ingested_at' (ver
                # contracts/tables.py::_ingestion_columns); es lo que permite
                # "el ultimo gana" de forma determinista.
                combined = combined.sort_values("ingested_at").drop_duplicates(
                    subset=list(self.spec.natural_key), keep="last", ignore_index=True
                )
            table = dataframe_to_arrow(combined, self.spec)
            parquet_io.write_parquet_atomic(path, table)
            written += len(group)
        return written

    def _partition_groups(self, df: pd.DataFrame) -> list[tuple[dict[str, object], pd.DataFrame]]:
        if not self.spec.partition_by:
            return [({}, df)]
        groups = []
        for key, group in df.groupby(list(self.spec.partition_by), dropna=False, sort=False):
            keys = key if isinstance(key, tuple) else (key,)
            partition_row = dict(zip(self.spec.partition_by, keys, strict=True))
            groups.append((partition_row, group))
        return groups

    def _partition_path(self, partition_row: dict[str, object]) -> Path:
        if self.spec.partition_by:
            return parquet_io.partition_file(self._base_dir, self.spec, partition_row)
        return parquet_io.table_dir(self._base_dir, self.spec) / "data.parquet"

    # -- read ---------------------------------------------------------

    def read(
        self,
        *,
        filters: dict[str, object] | None = None,
        columns: list[str] | None = None,
        as_of: datetime | None = None,
    ) -> pd.DataFrame:
        filters = filters or {}
        for col in filters:
            if col not in self.spec.column_names:
                raise KeyError(f"{self.spec.name}: filtro sobre columna inexistente {col!r}")
        if as_of is not None and self._temporal_column is None:
            raise ValueError(f"{self.spec.name}: no tiene columna temporal, as_of no aplica")

        select_cols = columns or list(self.spec.column_names)
        files = [str(p) for p in parquet_io.existing_partition_files(self._base_dir, self.spec)]
        if not files:
            return parquet_io.empty_frame(self.spec)[select_cols]

        where_sql, params = self._build_where(filters, as_of)
        cols_sql = ", ".join(f'"{c}"' for c in select_cols)
        # hive_partitioning=false: los valores de particion viven en el propio
        # Parquet como columnas reales (ver write()); sin esto, DuckDB infiere
        # los valores del nombre de carpeta -que sanitizamos para el layout- y
        # los antepone a los del archivo, corrompiendo el dato leido.
        query = (
            f"SELECT {cols_sql} FROM read_parquet($files, hive_partitioning=false) t "
            f"{'WHERE ' + where_sql if where_sql else ''}"
        )

        con = duckdb.connect(":memory:")
        try:
            result = con.execute(query, {"files": files, **params}).to_arrow_table()
        finally:
            con.close()
        # duckdb no trae stubs: to_pandas() se ve como Any para mypy.
        return cast(pd.DataFrame, result.to_pandas())

    def _build_where(
        self, filters: dict[str, object], as_of: datetime | None
    ) -> tuple[str, dict[str, object]]:
        clauses: list[str] = []
        params: dict[str, object] = {}
        for i, (col, value) in enumerate(filters.items()):
            name = f"f{i}"
            clauses.append(f'"{col}" = ${name}')
            params[name] = value
        if as_of is not None:
            clauses.append(f'"{self._temporal_column}" <= $as_of')
            params["as_of"] = as_of
        return " AND ".join(clauses), params
