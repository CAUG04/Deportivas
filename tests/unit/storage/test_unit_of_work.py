"""BufferedUnitOfWork: stage writes, flush only on commit, discard on rollback
or on an exception inside the ``with`` block."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tests.fixtures.synthetic_schema import WIDGETS

from deportivas.storage.duckdb_repo.repository import ParquetTableRepository
from deportivas.storage.unit_of_work import BufferedUnitOfWork


def _row(widget_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": widget_id,
                "category": "cat-a",
                "season": "2025",
                "value": 1.0,
                "source": "test",
                "ingested_at": datetime(2026, 1, 1, tzinfo=UTC),
            }
        ]
    )


def test_staged_writes_are_flushed_on_commit(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    with BufferedUnitOfWork() as uow:
        uow.stage(repo, _row("w1"))
        uow.stage(repo, _row("w2"))
        assert repo.read().empty  # nada aterriza hasta el commit

    result = repo.read()
    assert set(result["id"]) == {"w1", "w2"}


def test_exception_inside_with_block_discards_everything(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    try:
        with BufferedUnitOfWork() as uow:
            uow.stage(repo, _row("w1"))
            raise RuntimeError("fallo a mitad del job")
    except RuntimeError:
        pass

    assert repo.read().empty


def test_explicit_rollback_discards_pending_writes(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    with BufferedUnitOfWork() as uow:
        uow.stage(repo, _row("w1"))
        uow.rollback()

    assert repo.read().empty


def test_commit_returns_total_rows_written(tmp_path: Path) -> None:
    repo = ParquetTableRepository(WIDGETS, tmp_path)
    with BufferedUnitOfWork() as uow:
        uow.stage(repo, _row("w1"))
        uow.stage(repo, _row("w2"))
        total = uow.commit()
    assert total == 2
