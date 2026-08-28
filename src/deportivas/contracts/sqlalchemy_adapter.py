"""Derives SQLAlchemy metadata from :data:`SCHEMA`.

This is the Postgres flavour of the schema and the input to Alembic's
autogenerate. It is never hand-written: editing a table means editing
``contracts/tables.py``.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.schema import SchemaItem

from deportivas.contracts.tables import SCHEMA
from deportivas.contracts.types import LogicalType, SchemaRegistry, TableSpec

# JSON stays portable: JSONB on Postgres, plain JSON everywhere else. Without
# the variant the initial migration could not be applied outside Postgres and
# the round-trip test below would need a live server.
_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")

# TypeEngine[T] is generic over the Python type it maps to (TypeEngine[int],
# TypeEngine[bool]...), so a dict holding several concrete engines needs Any
# here rather than object: TypeEngine is invariant in T under mypy.
_TYPES: dict[LogicalType, sa.types.TypeEngine[Any]] = {
    LogicalType.INT: sa.Integer(),
    LogicalType.BIGINT: sa.BigInteger(),
    LogicalType.FLOAT: sa.Float(precision=53),
    LogicalType.BOOL: sa.Boolean(),
    LogicalType.TIMESTAMP: sa.DateTime(timezone=True),
    LogicalType.DATE: sa.Date(),
    LogicalType.JSON: _JSON,
}


def sqlalchemy_type(
    spec_type: LogicalType, max_length: int | None = None
) -> sa.types.TypeEngine[Any]:
    if spec_type is LogicalType.STR:
        return sa.String(max_length) if max_length else sa.Text()
    try:
        return _TYPES[spec_type]
    except KeyError as exc:  # pragma: no cover - defensivo, el enum es cerrado
        raise NotImplementedError(f"tipo logico sin mapear a SQLAlchemy: {spec_type}") from exc


def build_table(spec: TableSpec, metadata: sa.MetaData) -> sa.Table:
    columns: list[sa.Column[object]] = []
    for col in spec.columns:
        args: list[sa.ForeignKey] = []
        if col.foreign_key is not None:
            args.append(sa.ForeignKey(col.foreign_key))
        columns.append(
            sa.Column(
                col.name,
                sqlalchemy_type(col.type, col.max_length),
                *args,
                nullable=col.nullable,
                primary_key=col.primary_key,
                comment=col.description or None,
            )
        )

    constraints: list[SchemaItem] = []
    # La clave natural es lo que hace idempotente la ingesta va ON CONFLICT:
    # sin esta restriccion, reejecutar una fuente duplicaria filas. Las
    # tablas append-only (odds_snapshots, raw_documents...) nunca hacen
    # upsert -siempre insertan-, asi que su "clave natural" es solo
    # documental: exigirla como UNIQUE aqui rechazaria capturas reales que
    # coinciden en esa clave (dos snapshots en el mismo segundo, un reintento).
    if not spec.append_only and tuple(spec.natural_key) != spec.primary_key_columns:
        constraints.append(sa.UniqueConstraint(*spec.natural_key, name=f"uq_{spec.name}_natural"))
    for index in spec.indexes:
        constraints.append(sa.Index(index.name, *index.columns, unique=index.unique))

    return sa.Table(spec.name, metadata, *columns, *constraints, comment=spec.description or None)


def build_metadata(schema: SchemaRegistry = SCHEMA) -> sa.MetaData:
    metadata = sa.MetaData()
    for spec in schema:
        build_table(spec, metadata)
    return metadata


metadata = build_metadata()
