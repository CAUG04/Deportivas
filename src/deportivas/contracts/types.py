"""Schema primitives.

The whole project declares its tables exactly once, as :class:`TableSpec`
objects. Every persistence flavour (Pandera validation, SQLAlchemy/Postgres,
DuckDB/Parquet) is *derived* from those specs, so the two storage backends
cannot drift apart. Nothing here imports pandas, SQLAlchemy or DuckDB: the
adapters do that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class LogicalType(StrEnum):
    """Storage-agnostic column types.

    Deliberately small. Every adapter must map all of these, and a new member
    breaks the adapters loudly rather than silently degrading to text.
    """

    INT = "int"
    BIGINT = "bigint"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    TIMESTAMP = "timestamp"  # always tz-aware UTC
    DATE = "date"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    name: str
    type: LogicalType
    nullable: bool = False
    primary_key: bool = False
    foreign_key: str | None = None  # "table.column"
    max_length: int | None = None  # only meaningful for STR
    description: str = ""

    def __post_init__(self) -> None:
        if self.primary_key and self.nullable:
            raise ValueError(f"{self.name}: una clave primaria no puede ser nullable")
        if self.max_length is not None and self.type is not LogicalType.STR:
            raise ValueError(f"{self.name}: max_length solo aplica a columnas STR")
        if self.foreign_key is not None and self.foreign_key.count(".") != 1:
            raise ValueError(f"{self.name}: foreign_key debe ser 'tabla.columna'")


@dataclass(frozen=True, slots=True)
class IndexSpec:
    name: str
    columns: tuple[str, ...]
    unique: bool = False


@dataclass(frozen=True, slots=True)
class TableSpec:
    """One table, declared once for every backend.

    ``natural_key`` is what makes ingestion idempotent: re-running a source
    must upsert on this key instead of appending duplicates. ``partition_by``
    is how the DuckDB/Parquet backend lays files out on disk; it is ignored by
    the SQL backend, which relies on indexes instead.
    """

    name: str
    columns: tuple[ColumnSpec, ...]
    natural_key: tuple[str, ...]
    indexes: tuple[IndexSpec, ...] = ()
    partition_by: tuple[str, ...] = ()
    description: str = ""
    append_only: bool = False

    def __post_init__(self) -> None:
        names = [c.name for c in self.columns]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"{self.name}: columnas duplicadas {sorted(duplicates)}")
        if not self.primary_key_columns:
            raise ValueError(f"{self.name}: toda tabla necesita clave primaria")
        if not self.natural_key:
            raise ValueError(f"{self.name}: toda tabla necesita clave natural")
        known = set(names)
        for label, cols in (
            ("natural_key", self.natural_key),
            ("partition_by", self.partition_by),
        ):
            unknown = set(cols) - known
            if unknown:
                raise ValueError(
                    f"{self.name}: {label} referencia columnas inexistentes {sorted(unknown)}"
                )
        for index in self.indexes:
            unknown = set(index.columns) - known
            if unknown:
                raise ValueError(f"{self.name}: indice {index.name} referencia {sorted(unknown)}")
        for col in self.natural_key:
            if self.column(col).nullable:
                raise ValueError(
                    f"{self.name}: {col} forma parte de la clave natural y no puede ser nullable"
                )

    @property
    def primary_key_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns if c.primary_key)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def column(self, name: str) -> ColumnSpec:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"{self.name}: no existe la columna {name!r}")


@dataclass(frozen=True, slots=True)
class SchemaRegistry:
    """All tables, in dependency order (parents before children)."""

    tables: tuple[TableSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for table in self.tables:
            if table.name in seen:
                raise ValueError(f"tabla duplicada: {table.name}")
            for col in table.columns:
                if col.foreign_key is None:
                    continue
                parent = col.foreign_key.split(".")[0]
                if parent not in seen and parent != table.name:
                    raise ValueError(
                        f"{table.name}.{col.name}: la tabla padre {parent!r} debe declararse antes"
                    )
            seen.add(table.name)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.tables)

    def __len__(self) -> int:
        return len(self.tables)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tables)

    def get(self, name: str) -> TableSpec:
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(f"no existe la tabla {name!r}")
