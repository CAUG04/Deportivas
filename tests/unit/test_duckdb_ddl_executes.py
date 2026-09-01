"""The DuckDB DDL generated from the contract must actually run on DuckDB.

``test_schema_parity`` checks the DDL string is well-formed; this test proves
it further by handing it to a real, in-memory DuckDB connection.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pytest

from deportivas.contracts.duckdb_adapter import create_table_sql
from deportivas.contracts.tables import SCHEMA


@pytest.fixture
def conn() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect(":memory:")
    try:
        yield connection
    finally:
        connection.close()


def test_all_tables_create_in_dependency_order(conn: duckdb.DuckDBPyConnection) -> None:
    # SCHEMA.tables ya viene en orden padre-antes-que-hijo (lo exige
    # SchemaRegistry.__post_init__), asi que ejecutar en ese orden no debe
    # fallar por referencias adelantadas.
    for spec in SCHEMA:
        conn.execute(create_table_sql(spec))

    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert tables == set(SCHEMA.names)


def test_primary_key_is_enforced(conn: duckdb.DuckDBPyConnection) -> None:
    spec = SCHEMA.get("competitions")
    conn.execute(create_table_sql(spec))
    conn.execute(
        "INSERT INTO competitions VALUES ('epl','Premier League','England','football',1,true,'seed',now())"
    )
    with pytest.raises(duckdb.ConstraintException):
        conn.execute(
            "INSERT INTO competitions VALUES ('epl','Premier League','England','football',1,true,'seed',now())"
        )
