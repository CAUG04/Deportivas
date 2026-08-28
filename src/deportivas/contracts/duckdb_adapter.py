"""Derives DuckDB DDL and Parquet (Arrow) schemas from :data:`SCHEMA`.

The production backend stores Parquet files partitioned on disk and uses
DuckDB purely as a query engine over them. Two things are generated here:

* ``create_table_sql`` - typed tables, for the DuckDB catalog and for tests.
* ``arrow_schema`` - the exact Parquet types, so that a column written today
  and a column written next season are the same physical type.

Foreign keys are intentionally omitted from the DuckDB DDL: the Parquet lake
is loaded partition by partition and rejecting a child row because its parent
lands in the next batch would break incremental ingestion. Referential
integrity is enforced by the Postgres backend and by the contract tests.
"""

from __future__ import annotations

import pyarrow as pa

from deportivas.contracts.tables import SCHEMA
from deportivas.contracts.types import LogicalType, SchemaRegistry, TableSpec

_DUCKDB_TYPES: dict[LogicalType, str] = {
    LogicalType.INT: "INTEGER",
    LogicalType.BIGINT: "BIGINT",
    LogicalType.FLOAT: "DOUBLE",
    LogicalType.STR: "VARCHAR",
    LogicalType.BOOL: "BOOLEAN",
    LogicalType.TIMESTAMP: "TIMESTAMP WITH TIME ZONE",
    LogicalType.DATE: "DATE",
    LogicalType.JSON: "JSON",
}

_ARROW_TYPES: dict[LogicalType, pa.DataType] = {
    LogicalType.INT: pa.int32(),
    LogicalType.BIGINT: pa.int64(),
    LogicalType.FLOAT: pa.float64(),
    LogicalType.STR: pa.string(),
    LogicalType.BOOL: pa.bool_(),
    LogicalType.TIMESTAMP: pa.timestamp("us", tz="UTC"),
    LogicalType.DATE: pa.date32(),
    # El vector de features viaja como texto JSON: Parquet no tiene un tipo
    # JSON nativo estable entre versiones de pyarrow.
    LogicalType.JSON: pa.string(),
}


def duckdb_type(spec_type: LogicalType) -> str:
    try:
        return _DUCKDB_TYPES[spec_type]
    except KeyError as exc:  # pragma: no cover - defensivo, el enum es cerrado
        raise NotImplementedError(f"tipo logico sin mapear a DuckDB: {spec_type}") from exc


def arrow_type(spec_type: LogicalType) -> pa.DataType:
    try:
        return _ARROW_TYPES[spec_type]
    except KeyError as exc:  # pragma: no cover - defensivo, el enum es cerrado
        raise NotImplementedError(f"tipo logico sin mapear a Arrow: {spec_type}") from exc


def create_table_sql(spec: TableSpec, *, if_not_exists: bool = True) -> str:
    lines = []
    for col in spec.columns:
        null = "" if col.nullable else " NOT NULL"
        lines.append(f"    {col.name} {duckdb_type(col.type)}{null}")
    lines.append(f"    PRIMARY KEY ({', '.join(spec.primary_key_columns)})")
    if tuple(spec.natural_key) != spec.primary_key_columns:
        lines.append(f"    UNIQUE ({', '.join(spec.natural_key)})")
    exists = "IF NOT EXISTS " if if_not_exists else ""
    return f"CREATE TABLE {exists}{spec.name} (\n" + ",\n".join(lines) + "\n)"


def arrow_schema(spec: TableSpec) -> pa.Schema:
    return pa.schema(
        [pa.field(col.name, arrow_type(col.type), nullable=col.nullable) for col in spec.columns]
    )


def all_create_table_sql(schema: SchemaRegistry = SCHEMA) -> list[str]:
    return [create_table_sql(spec) for spec in schema]
