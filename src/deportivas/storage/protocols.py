"""Abstract repository layer.

Every other layer (ingest, features, models, backtest, api, export) depends
only on these protocols, never on DuckDB or SQLAlchemy directly. That is what
makes the DuckDB -> Postgres migration documented in the README a matter of
swapping which implementation is wired up in :mod:`deportivas.storage.factory`,
not a rewrite.

Two design choices worth calling out:

* **DataFrame in, DataFrame out.** The volume here (seasons of match data,
  odds snapshots, features) is naturally tabular, and every consumer downstream
  (pandas-based feature engineering, LightGBM, the backtest engine) wants a
  DataFrame anyway. Row-by-row ORM objects would just be unpacked back into
  one. ``pandas.DataFrame`` is the exchange type at this boundary.
* **Upsert, not insert.** ``write`` upserts on each table's natural key
  (see :mod:`deportivas.contracts.tables`). Re-running an ingestion job must
  never duplicate rows — rule 1 of Fase 1 (idempotencia).
* **append-only tables reject upsert.** ``odds_snapshots`` and
  ``raw_documents`` are append-only by design (a later snapshot must never
  overwrite an earlier one, or CLV becomes unrecoverable). ``write`` on those
  tables always inserts; a caller trying to "correct" a past snapshot is a bug.
* **One narrow, deliberate exception to "append-only never updates":**
  ``mark_closing``. Whether a captured price was the last one before kickoff
  is metadata that can only be known in hindsight, once the fixture has
  started and no further pre-kickoff snapshot can ever arrive — it was
  never knowable at capture time, unlike every other column on the row.
  Flagging it is not "correcting" a price (the price itself, and every
  other snapshot's price, stays exactly as captured); ``backtest/settlement.py``'s
  own runtime fallback (the latest snapshot at or before kickoff) already
  covers CLV correctly without this flag ever being set, so calling it is
  an optimization for the common case, never a correctness requirement.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class TableRepository(Protocol):
    """Generic read/write access to one table, keyed by its natural key.

    Table-specific repositories (``FixtureRepository``, ``OddsRepository``...)
    add typed convenience methods on top of this; they do not replace it.
    """

    table_name: str

    def write(self, rows: pd.DataFrame) -> int:
        """Upserts ``rows`` (or inserts, for an append-only table).

        Validates against the table's Pandera schema first and raises on a
        row that does not conform — see rule "validacion de datos en ingesta"
        in Fase 10. Returns the number of rows written.
        """
        ...

    def read(
        self,
        *,
        filters: dict[str, object] | None = None,
        columns: list[str] | None = None,
        as_of: datetime | None = None,
    ) -> pd.DataFrame:
        """Reads rows matching ``filters`` (equality only; keep this simple).

        ``as_of``, when the table carries a temporal column (``captured_at``
        for odds, ``as_of_timestamp`` for features), restricts to rows visible
        at that instant. This is the mechanism that makes point-in-time
        backtesting possible: the backtest never calls ``read`` without
        ``as_of`` on a temporal table.
        """
        ...

    def mark_closing(self, ids: Sequence[str]) -> int:
        """Sets ``is_closing = True`` on exactly the rows named by ``ids``
        (the table's own primary key) — see the module docstring's "one
        narrow, deliberate exception". Only ``odds_snapshots`` carries an
        ``is_closing`` column; calling this on any other table raises
        ``ValueError`` rather than silently doing nothing. Rows already
        flagged don't count towards the returned total, so re-running
        against the same ids is always cheap to observe.
        """
        ...


@runtime_checkable
class FixtureRepository(TableRepository, Protocol):
    def upcoming(
        self, *, competition_id: str | None = None, days_ahead: int = 7
    ) -> pd.DataFrame: ...

    def by_competition_season(self, competition_id: str, season: str) -> pd.DataFrame: ...


@runtime_checkable
class OddsRepository(TableRepository, Protocol):
    def latest_before(self, fixture_id: str, market: str, *, as_of: datetime) -> pd.DataFrame:
        """The most recent snapshot per (bookmaker, selection, line) at or before ``as_of``.

        This is the only correct way to get "the price available at decision
        time" (rule 2). Never call ``latest_before`` with ``as_of=None`` and
        never fall back to ``is_closing`` rows when building a backtest entry
        price — that would be training and testing on the same leak the whole
        engine exists to prevent.
        """
        ...

    def closing_line(self, fixture_id: str, market: str) -> pd.DataFrame:
        """Rows flagged ``is_closing``. Used only to compute CLV after settlement."""
        ...


@runtime_checkable
class FeatureRepository(TableRepository, Protocol):
    def vector_as_of(
        self, fixture_id: str, feature_set: str, *, as_of: datetime
    ) -> dict[str, object] | None: ...


@runtime_checkable
class RawDocumentRepository(Protocol):
    """The append-only raw layer. No update, no delete — see Fase 10."""

    def store(
        self,
        *,
        source: str,
        endpoint: str,
        params: dict[str, object],
        content: bytes,
        content_type: str,
        status_code: int | None,
        fetched_at: datetime,
    ) -> str:
        """Persists the untouched payload and returns the generated document id."""
        ...

    def read(self, document_id: str) -> bytes: ...

    def find(
        self, *, source: str, endpoint: str | None = None, since: datetime | None = None
    ) -> pd.DataFrame:
        """Metadata only (no payload bytes) — used to reconstruct without re-scraping."""
        ...


@runtime_checkable
class UnitOfWork(Protocol):
    """Transactional boundary. On Postgres this wraps a DB transaction; on the
    DuckDB/Parquet backend it wraps a batch of partition writes that either all
    land or none do, so a failed job never leaves half a competition ingested.
    """

    def __enter__(self) -> UnitOfWork: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...
