"""A small synthetic TableSpec used to test the repository implementations in
isolation from the real (larger) production schema. Both DuckDB and SQL
repository test suites import this so their test cases stay comparable.
"""

from __future__ import annotations

from deportivas.contracts.types import ColumnSpec as C
from deportivas.contracts.types import LogicalType, TableSpec

T = LogicalType

WIDGETS = TableSpec(
    name="widgets",
    columns=(
        C("id", T.STR, primary_key=True, max_length=64),
        C("category", T.STR, max_length=32),
        C("season", T.STR, max_length=16),
        C("value", T.FLOAT, nullable=True),
        C("source", T.STR, max_length=64),
        C("ingested_at", T.TIMESTAMP),
    ),
    natural_key=("id",),
    partition_by=("category", "season"),
)

WIDGET_EVENTS = TableSpec(
    name="widget_events",
    columns=(
        # Igual que odds_snapshots/raw_documents: la clave primaria es un id
        # subrogado, distinto de la clave natural. Un append-only cuya PK
        # coincidiera con su natural_key no podria repetir esa clave natural
        # nunca (chocaria con la restriccion PK de Postgres) aunque en
        # DuckDB/Parquet no haya PK real que lo impida.
        C("id", T.STR, primary_key=True, max_length=64),
        C("widget_id", T.STR, max_length=64),
        C("captured_at", T.TIMESTAMP),
        C("payload", T.STR, max_length=256),
    ),
    natural_key=("widget_id", "captured_at"),
    append_only=True,
)
