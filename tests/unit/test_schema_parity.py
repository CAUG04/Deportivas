"""Parity tests between the three adapters derived from ``contracts.tables.SCHEMA``.

This is the test that makes the "declare the schema once" design safe: if the
SQLAlchemy, DuckDB or Pandera adapter ever silently drops a table or a column,
these tests fail here instead of two backends quietly disagreeing about what a
row looks like.
"""

from __future__ import annotations

import pytest

from deportivas.contracts.duckdb_adapter import all_create_table_sql, arrow_schema, create_table_sql
from deportivas.contracts.pandera_adapter import all_schemas as pandera_all_schemas
from deportivas.contracts.sqlalchemy_adapter import build_metadata
from deportivas.contracts.tables import SCHEMA
from deportivas.contracts.types import LogicalType

ALL_TABLE_NAMES = SCHEMA.names


def duckdb_all_schemas_dict() -> dict[str, str]:  # pragma: no cover - trivial helper
    return {name: create_table_sql(SCHEMA.get(name)) for name in ALL_TABLE_NAMES}


@pytest.mark.parametrize("table_name", ALL_TABLE_NAMES)
def test_sqlalchemy_has_every_table_and_column(table_name: str) -> None:
    metadata = build_metadata()
    assert table_name in metadata.tables
    spec = SCHEMA.get(table_name)
    sa_table = metadata.tables[table_name]
    assert set(sa_table.columns.keys()) == set(spec.column_names)


@pytest.mark.parametrize("table_name", ALL_TABLE_NAMES)
def test_sqlalchemy_primary_key_matches(table_name: str) -> None:
    metadata = build_metadata()
    spec = SCHEMA.get(table_name)
    sa_table = metadata.tables[table_name]
    pk_columns = {c.name for c in sa_table.primary_key.columns}
    assert pk_columns == set(spec.primary_key_columns)


@pytest.mark.parametrize("table_name", ALL_TABLE_NAMES)
def test_sqlalchemy_nullability_matches(table_name: str) -> None:
    metadata = build_metadata()
    spec = SCHEMA.get(table_name)
    sa_table = metadata.tables[table_name]
    for col in spec.columns:
        assert sa_table.columns[col.name].nullable == col.nullable, (
            f"{table_name}.{col.name}: nullable no coincide entre contrato y SQLAlchemy"
        )


@pytest.mark.parametrize("table_name", ALL_TABLE_NAMES)
def test_duckdb_ddl_mentions_every_column(table_name: str) -> None:
    spec = SCHEMA.get(table_name)
    ddl = create_table_sql(spec)
    assert ddl.startswith(f"CREATE TABLE IF NOT EXISTS {table_name} (")
    for col in spec.columns:
        assert f"    {col.name} " in ddl, f"{table_name}.{col.name} ausente en el DDL de DuckDB"
    assert f"PRIMARY KEY ({', '.join(spec.primary_key_columns)})" in ddl


@pytest.mark.parametrize("table_name", ALL_TABLE_NAMES)
def test_arrow_schema_matches_columns(table_name: str) -> None:
    spec = SCHEMA.get(table_name)
    schema = arrow_schema(spec)
    assert schema.names == list(spec.column_names)
    for col in spec.columns:
        field = schema.field(col.name)
        assert field.nullable == col.nullable


@pytest.mark.parametrize("table_name", ALL_TABLE_NAMES)
def test_pandera_schema_matches_columns(table_name: str) -> None:
    spec = SCHEMA.get(table_name)
    schemas = pandera_all_schemas()
    pa_schema = schemas[table_name]
    assert set(pa_schema.columns.keys()) == set(spec.column_names)
    for col in spec.columns:
        assert pa_schema.columns[col.name].nullable == col.nullable


def test_every_adapter_covers_every_logical_type() -> None:
    """A new LogicalType member must be mapped in all three adapters, or the
    dedicated NotImplementedError branches in each adapter would silently
    remain untested and a future column could target an unmapped type.
    """
    from deportivas.contracts.duckdb_adapter import _ARROW_TYPES, _DUCKDB_TYPES
    from deportivas.contracts.pandera_adapter import _PANDAS_DTYPES
    from deportivas.contracts.sqlalchemy_adapter import _TYPES

    types_needing_str_handling = {LogicalType.STR}
    for mapping, extra_ok in (
        (_DUCKDB_TYPES, set()),
        (_ARROW_TYPES, set()),
        (_PANDAS_DTYPES, set()),
        (_TYPES, types_needing_str_handling),  # STR se resuelve aparte en sqlalchemy_type()
    ):
        covered = set(mapping.keys()) | extra_ok
        assert covered == set(LogicalType), f"tipos sin mapear: {set(LogicalType) - covered}"


def test_all_tables_produce_valid_duckdb_ddl() -> None:
    ddls = duckdb_all_schemas_dict()
    assert len(ddls) == len(ALL_TABLE_NAMES)
    for name, ddl in ddls.items():
        assert ddl.count("(") == ddl.count(")")
        assert name in ddl


def test_all_create_table_sql_matches_per_table_output() -> None:
    from_bulk = all_create_table_sql()
    assert len(from_bulk) == len(ALL_TABLE_NAMES)
    assert from_bulk == [create_table_sql(SCHEMA.get(name)) for name in ALL_TABLE_NAMES]
