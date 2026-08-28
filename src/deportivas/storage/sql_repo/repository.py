"""Generic Postgres implementation of :class:`TableRepository`.

Mirrors :class:`deportivas.storage.duckdb_repo.repository.ParquetTableRepository`:
one class parametrised by a :class:`TableSpec`, upserting on the table's
natural key via ``INSERT ... ON CONFLICT ... DO UPDATE``. This is the
secondary backend (local dev / future migration target), not what production
runs on — see the README for the DuckDB-vs-Postgres split.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from deportivas.contracts.sqlalchemy_adapter import build_table
from deportivas.contracts.types import TableSpec
from deportivas.storage.validation import validate_rows


def _records(df: pd.DataFrame, spec: TableSpec) -> list[dict[str, object]]:
    ordered = df[list(spec.column_names)]
    # NaN/NaT no son NULL para psycopg: hay que convertirlos a None a mano.
    clean = ordered.astype(object).where(pd.notnull(ordered), None)
    return clean.to_dict("records")  # type: ignore[return-value]


class SqlTableRepository:
    """Implements :class:`deportivas.storage.protocols.TableRepository`.

    Builds its own single-table :class:`sqlalchemy.MetaData` from ``spec`` via
    :func:`build_table` rather than looking the table up in the project-wide
    metadata singleton: that keeps this class usable for any ``TableSpec``
    (production or a synthetic one in tests) as long as the table already
    exists in the database — creating it is Alembic's job, not this class's.
    """

    def __init__(
        self, spec: TableSpec, engine: sa.Engine, *, temporal_column: str | None = None
    ) -> None:
        self.spec = spec
        self.table_name = spec.name
        self._engine = engine
        self._table: sa.Table = build_table(spec, sa.MetaData())
        self._temporal_column = temporal_column
        if temporal_column is not None and temporal_column not in spec.column_names:
            raise ValueError(f"{spec.name}: columna temporal {temporal_column!r} no existe")

    def write(self, rows: pd.DataFrame) -> int:
        if rows.empty:
            return 0
        valid = validate_rows(self.spec, rows)
        if valid.empty:
            return 0

        if not self.spec.append_only:
            # Un INSERT ... ON CONFLICT no tolera dos filas del mismo lote
            # apuntando a la misma clave de conflicto ("cannot affect row a
            # second time"): hay que deduplicar el lote antes de enviarlo,
            # igual que hace el backend DuckDB.
            valid = valid.sort_values("ingested_at").drop_duplicates(
                subset=list(self.spec.natural_key), keep="last", ignore_index=True
            )

        records = _records(valid, self.spec)
        with self._engine.begin() as conn:
            if self.spec.append_only:
                conn.execute(self._table.insert(), records)
            else:
                stmt = pg_insert(self._table).values(records)
                update_cols = {
                    c.name: getattr(stmt.excluded, c.name)
                    for c in self._table.columns
                    if c.name not in self.spec.natural_key
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=list(self.spec.natural_key), set_=update_cols
                )
                conn.execute(stmt)
        return len(valid)

    def read(
        self,
        *,
        filters: dict[str, object] | None = None,
        columns: list[str] | None = None,
        as_of: datetime | None = None,
    ) -> pd.DataFrame:
        filters = filters or {}
        select_cols = columns or list(self.spec.column_names)
        for col in (*filters, *select_cols):
            if col not in self.spec.column_names:
                raise KeyError(f"{self.spec.name}: columna inexistente {col!r}")
        if as_of is not None and self._temporal_column is None:
            raise ValueError(f"{self.spec.name}: no tiene columna temporal, as_of no aplica")

        query = sa.select(*(self._table.c[c] for c in select_cols))
        for col, value in filters.items():
            query = query.where(self._table.c[col] == value)
        if as_of is not None:
            temporal_column = self._temporal_column
            assert temporal_column is not None  # garantizado por el chequeo de arriba
            query = query.where(self._table.c[temporal_column] <= as_of)

        with self._engine.connect() as conn:
            return pd.read_sql(query, conn)
