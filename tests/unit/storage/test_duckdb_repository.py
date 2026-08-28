"""Tests for ParquetTableRepository: idempotency, upsert, append-only,
partitioning, filtering, and rejection of bad rows are exactly the behaviours
Fase 1 (ingesta) depends on."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from tests.fixtures.synthetic_schema import WIDGET_EVENTS, WIDGETS

from deportivas.storage.duckdb_repo.repository import ParquetTableRepository
from deportivas.storage.validation import ValidationError


def _rows(**overrides: object) -> pd.DataFrame:
    base = {
        "id": "w1",
        "category": "cat-a",
        "season": "2025",
        "value": 1.0,
        "source": "test",
        "ingested_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    written = repo.write(_rows())
    assert written == 1

    result = repo.read()
    assert len(result) == 1
    assert result.iloc[0]["id"] == "w1"
    assert result.iloc[0]["value"] == pytest.approx(1.0)


def test_write_creates_hive_partition_layout(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    repo.write(_rows(category="cat-a", season="2025"))
    expected = tmp_path / "widgets" / "category=cat-a" / "season=2025" / "data.parquet"
    assert expected.exists()


def test_rewrite_same_natural_key_upserts_no_duplicate(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    repo.write(_rows(value=1.0, ingested_at=datetime(2026, 1, 1, tzinfo=UTC)))
    repo.write(_rows(value=2.0, ingested_at=datetime(2026, 1, 2, tzinfo=UTC)))

    result = repo.read()
    assert len(result) == 1
    assert result.iloc[0]["value"] == pytest.approx(2.0)


def test_upsert_keeps_latest_even_if_written_out_of_order(tmp_path: Path) -> None:
    """Re-running a backfill job out of order must not let an older row win."""
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    repo.write(_rows(value=2.0, ingested_at=datetime(2026, 1, 2, tzinfo=UTC)))
    repo.write(_rows(value=1.0, ingested_at=datetime(2026, 1, 1, tzinfo=UTC)))

    result = repo.read()
    assert len(result) == 1
    assert result.iloc[0]["value"] == pytest.approx(2.0)


def test_rerunning_ingestion_is_idempotent(tmp_path: Path) -> None:
    """Rule 1 of Fase 1: re-running a source must not duplicate rows."""
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    rows = _rows()
    repo.write(rows)
    repo.write(rows)
    repo.write(rows)

    assert len(repo.read()) == 1


def test_different_natural_keys_both_persist(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    repo.write(_rows(id="w1"))
    repo.write(_rows(id="w2"))

    result = repo.read()
    assert set(result["id"]) == {"w1", "w2"}


def test_read_with_equality_filter(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    repo.write(_rows(id="w1", category="cat-a"))
    repo.write(_rows(id="w2", category="cat-b", season="2025"))

    result = repo.read(filters={"category": "cat-b"})
    assert list(result["id"]) == ["w2"]


def test_read_unknown_filter_column_raises_keyerror(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    repo.write(_rows())
    with pytest.raises(KeyError):
        repo.read(filters={"does_not_exist": 1})


def test_read_selects_only_requested_columns(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    repo.write(_rows())
    result = repo.read(columns=["id", "value"])
    assert list(result.columns) == ["id", "value"]


def test_read_on_empty_table_returns_empty_with_correct_columns(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    result = repo.read()
    assert result.empty
    assert list(result.columns) == list(WIDGETS.column_names)


def test_write_empty_dataframe_is_a_noop(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    empty = pd.DataFrame(columns=list(WIDGETS.column_names))
    assert repo.write(empty) == 0
    assert repo.read().empty


def test_as_of_without_temporal_column_raises(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)  # sin temporal_column
    repo.write(_rows())
    with pytest.raises(ValueError, match="no tiene columna temporal"):
        repo.read(as_of=datetime.now(UTC))


def test_as_of_filters_on_temporal_column(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGET_EVENTS, tmp_path, temporal_column="captured_at")
    repo.write(
        pd.DataFrame(
            [
                {
                    "id": "e1",
                    "widget_id": "w1",
                    "captured_at": datetime(2026, 1, 1, tzinfo=UTC),
                    "payload": "early",
                },
                {
                    "id": "e2",
                    "widget_id": "w1",
                    "captured_at": datetime(2026, 1, 5, tzinfo=UTC),
                    "payload": "late",
                },
            ]
        )
    )
    result = repo.read(as_of=datetime(2026, 1, 2, tzinfo=UTC))
    assert list(result["id"]) == ["e1"]


def test_append_only_table_never_deduplicates(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGET_EVENTS, tmp_path, temporal_column="captured_at")
    same_natural_key = {"widget_id": "w1", "captured_at": datetime(2026, 1, 1, tzinfo=UTC)}
    repo.write(pd.DataFrame([{"id": "e1", "payload": "first", **same_natural_key}]))
    # misma clave natural (widget_id, captured_at), id subrogado distinto:
    # asi es como dos capturas reales de la misma cuota en el mismo instante
    # conviven sin pisarse.
    repo.write(pd.DataFrame([{"id": "e2", "payload": "second", **same_natural_key}]))

    result = repo.read()
    assert len(result) == 2


def test_invalid_rows_are_rejected_and_valid_rows_still_written(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    good = _rows(id="w-good")
    bad = _rows(id=None)  # id es primary key, no puede ser null
    batch = pd.concat([good, bad], ignore_index=True)

    written = repo.write(batch)
    assert written == 1

    result = repo.read()
    assert list(result["id"]) == ["w-good"]


def test_intra_batch_duplicate_natural_key_is_rejected(tmp_path: Path) -> None:
    """A single write() call reporting the same natural key twice is dirty
    source data, not a legitimate upsert (that happens across separate
    write() calls instead — see test_rewrite_same_natural_key_upserts_no_duplicate).
    Pandera's unique=natural_key check rejects both copies and logs why."""
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    batch = pd.concat(
        [_rows(id="w-clean"), _rows(id="w-dup", value=1.0), _rows(id="w-dup", value=2.0)],
        ignore_index=True,
    )
    written = repo.write(batch)
    assert written == 1
    assert list(repo.read()["id"]) == ["w-clean"]


def test_all_rows_invalid_returns_zero_written(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    bad = _rows(id=None)
    assert repo.write(bad) == 0
    assert repo.read().empty


def test_structurally_wrong_dataframe_raises_validation_error(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    wrong_shape = pd.DataFrame([{"totally": "different", "columns": 1}])
    with pytest.raises(ValidationError):
        repo.write(wrong_shape)


def test_constructor_rejects_unknown_temporal_column(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no existe"):
        ParquetTableRepository(WIDGETS, tmp_path, temporal_column="does_not_exist")


def test_none_partition_value_raises(tmp_path: Path) -> None:
    """No production TableSpec currently allows a nullable partition_by
    column (pandera would reject the row before this is ever reached via
    write()), but the guard exists for when one does — tested directly."""
    from deportivas.storage.duckdb_repo import parquet_io

    with pytest.raises(ValueError, match="no puede ser None"):
        parquet_io.partition_dir(tmp_path, WIDGETS, {"category": None, "season": "2025"})


def test_write_failure_cleans_up_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pyarrow.parquet as pq

    from deportivas.storage.duckdb_repo import parquet_io

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disco lleno, simulado")

    monkeypatch.setattr(pq, "write_table", boom)
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    with pytest.raises(OSError, match="disco lleno"):
        repo.write(_rows())

    leftovers = (
        list((tmp_path / "widgets").rglob("*.tmp")) if (tmp_path / "widgets").exists() else []
    )
    assert leftovers == []
    assert not parquet_io.existing_partition_files(tmp_path, WIDGETS)


def test_partition_value_with_slash_is_sanitised(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    repo.write(_rows(category="a/b"))
    result = repo.read()
    assert result.iloc[0]["category"] == "a/b"  # el valor logico no cambia
    assert not (tmp_path / "widgets" / "category=a" / "b").exists()  # pero no rompe el layout
