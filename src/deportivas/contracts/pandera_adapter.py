"""Derives Pandera schemas from :data:`SCHEMA`.

Every DataFrame is validated against one of these before it is written.
A bad row is rejected and logged; it never reaches the model as a silent null.
"""

from __future__ import annotations

import pandera.pandas as pa

from deportivas.contracts.tables import SCHEMA
from deportivas.contracts.types import LogicalType, SchemaRegistry, TableSpec

_PANDAS_DTYPES: dict[LogicalType, str] = {
    # Enteros nullable de pandas: un entero ausente no debe convertirse en NaN
    # float ni en 0.
    LogicalType.INT: "Int32",
    LogicalType.BIGINT: "Int64",
    LogicalType.FLOAT: "float64",
    LogicalType.STR: "string",
    LogicalType.BOOL: "boolean",
    LogicalType.TIMESTAMP: "datetime64[us, UTC]",
    LogicalType.DATE: "datetime64[us]",
    LogicalType.JSON: "string",
}


def pandas_dtype(spec_type: LogicalType) -> str:
    try:
        return _PANDAS_DTYPES[spec_type]
    except KeyError as exc:  # pragma: no cover - defensivo, el enum es cerrado
        raise NotImplementedError(f"tipo logico sin mapear a pandas: {spec_type}") from exc


def build_schema(spec: TableSpec, *, strict: bool = True) -> pa.DataFrameSchema:
    columns = {
        col.name: pa.Column(
            pandas_dtype(col.type),
            nullable=col.nullable,
            required=True,
            description=col.description or None,
        )
        for col in spec.columns
    }
    return pa.DataFrameSchema(
        columns=columns,
        unique=list(spec.natural_key),
        strict=strict,
        coerce=True,
        name=spec.name,
    )


def all_schemas(schema: SchemaRegistry = SCHEMA) -> dict[str, pa.DataFrameSchema]:
    return {spec.name: build_schema(spec) for spec in schema}
