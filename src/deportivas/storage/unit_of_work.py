"""Backend-agnostic :class:`UnitOfWork`: stage writes, flush on commit.

Neither backend gets a true multi-table transaction in Fase 1: DuckDB writes
one Parquet partition file at a time (no cross-file transaction), and the SQL
repository opens its own connection per ``write()`` call. What this class
does provide is what ingestion jobs actually need — "if anything in this
batch fails, nothing lands" — by holding every ``(repository, rows)`` pair in
memory and only calling ``write`` once the caller reaches ``commit()``. A
failed job never leaves half a competition ingested.

True cross-table DB transactions are a reasonable future upgrade for the SQL
backend (share one connection across repositories) but are not needed by
anything built so far, so they are not built ahead of a real caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pandas as pd


class _WritableRepository(Protocol):
    def write(self, rows: pd.DataFrame) -> int: ...


class BufferedUnitOfWork:
    def __init__(self) -> None:
        self._pending: list[tuple[_WritableRepository, pd.DataFrame]] = []

    def stage(self, repository: _WritableRepository, rows: pd.DataFrame) -> None:
        self._pending.append((repository, rows))

    def __enter__(self) -> BufferedUnitOfWork:
        self._pending = []
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()

    def commit(self) -> int:
        total = 0
        for repository, rows in self._pending:
            total += repository.write(rows)
        self._pending = []
        return total

    def rollback(self) -> None:
        self._pending = []
