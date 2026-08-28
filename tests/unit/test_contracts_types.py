"""Unit tests for the schema primitives themselves (contracts/types.py)."""

from __future__ import annotations

import pytest

from deportivas.contracts.types import ColumnSpec, IndexSpec, LogicalType, SchemaRegistry, TableSpec


def _col(name: str, **kwargs: object) -> ColumnSpec:
    return ColumnSpec(name, LogicalType.STR, **kwargs)  # type: ignore[arg-type]


def test_primary_key_cannot_be_nullable() -> None:
    with pytest.raises(ValueError, match="nullable"):
        ColumnSpec("id", LogicalType.STR, primary_key=True, nullable=True)


def test_max_length_only_for_str() -> None:
    with pytest.raises(ValueError, match="max_length"):
        ColumnSpec("n", LogicalType.INT, max_length=10)


def test_foreign_key_must_be_table_dot_column() -> None:
    with pytest.raises(ValueError, match="foreign_key"):
        ColumnSpec("team_id", LogicalType.STR, foreign_key="teams")


def test_table_rejects_duplicate_columns() -> None:
    with pytest.raises(ValueError, match="duplicadas"):
        TableSpec(
            name="t",
            columns=(_col("id", primary_key=True), _col("id")),
            natural_key=("id",),
        )


def test_table_requires_primary_key() -> None:
    with pytest.raises(ValueError, match="clave primaria"):
        TableSpec(name="t", columns=(_col("id"),), natural_key=("id",))


def test_table_requires_natural_key() -> None:
    with pytest.raises(ValueError, match="clave natural"):
        TableSpec(name="t", columns=(_col("id", primary_key=True),), natural_key=())


def test_natural_key_columns_must_exist() -> None:
    with pytest.raises(ValueError, match="natural_key"):
        TableSpec(
            name="t",
            columns=(_col("id", primary_key=True),),
            natural_key=("id", "ghost"),
        )


def test_natural_key_column_cannot_be_nullable() -> None:
    with pytest.raises(ValueError, match="clave natural"):
        TableSpec(
            name="t",
            columns=(_col("id", primary_key=True), _col("code", nullable=True)),
            natural_key=("id", "code"),
        )


def test_index_columns_must_exist() -> None:
    with pytest.raises(ValueError, match="indice"):
        TableSpec(
            name="t",
            columns=(_col("id", primary_key=True),),
            natural_key=("id",),
            indexes=(IndexSpec("ix", ("ghost",)),),
        )


def test_partition_by_columns_must_exist() -> None:
    with pytest.raises(ValueError, match="partition_by"):
        TableSpec(
            name="t",
            columns=(_col("id", primary_key=True),),
            natural_key=("id",),
            partition_by=("ghost",),
        )


def test_registry_rejects_duplicate_table_names() -> None:
    t = TableSpec(name="t", columns=(_col("id", primary_key=True),), natural_key=("id",))
    with pytest.raises(ValueError, match="duplicada"):
        SchemaRegistry(tables=(t, t))


def test_registry_requires_parent_before_child() -> None:
    parent = TableSpec(name="parent", columns=(_col("id", primary_key=True),), natural_key=("id",))
    child = TableSpec(
        name="child",
        columns=(
            _col("id", primary_key=True),
            ColumnSpec("parent_id", LogicalType.STR, foreign_key="parent.id"),
        ),
        natural_key=("id",),
    )
    with pytest.raises(ValueError, match="debe declararse antes"):
        SchemaRegistry(tables=(child, parent))
    # el orden correcto no falla
    SchemaRegistry(tables=(parent, child))


def test_registry_get_and_iteration() -> None:
    t = TableSpec(name="t", columns=(_col("id", primary_key=True),), natural_key=("id",))
    registry = SchemaRegistry(tables=(t,))
    assert registry.get("t") is t
    assert list(registry) == [t]
    assert len(registry) == 1
    with pytest.raises(KeyError):
        registry.get("ghost")


def test_table_spec_helpers() -> None:
    t = TableSpec(
        name="t",
        columns=(_col("id", primary_key=True), _col("name")),
        natural_key=("id",),
    )
    assert t.primary_key_columns == ("id",)
    assert t.column_names == ("id", "name")
    assert t.column("name").name == "name"
    with pytest.raises(KeyError):
        t.column("ghost")
