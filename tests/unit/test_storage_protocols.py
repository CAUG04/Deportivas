"""Structural tests for the abstract repository layer.

These protocols have no logic of their own (that is the point: DuckDB and
Postgres implementations plug into the same shape). What *is* worth testing is
that the shape is real: a class that implements every method structurally
satisfies the protocol via ``isinstance`` (thanks to ``@runtime_checkable``),
and one that is missing a method does not. That is what stops the DuckDB and
SQLAlchemy repositories built in Fase 1 from silently drifting apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from deportivas.storage.protocols import (
    FeatureRepository,
    FixtureRepository,
    OddsRepository,
    RawDocumentRepository,
    TableRepository,
    UnitOfWork,
)


class _FakeTableRepo:
    table_name = "fixtures"

    def write(self, rows: pd.DataFrame) -> int:
        return len(rows)

    def read(
        self,
        *,
        filters: dict[str, object] | None = None,
        columns: list[str] | None = None,
        as_of: datetime | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def mark_closing(self, ids: Sequence[str]) -> int:
        raise NotImplementedError


def test_fake_table_repo_satisfies_protocol() -> None:
    assert isinstance(_FakeTableRepo(), TableRepository)


def test_repo_missing_write_does_not_satisfy_protocol() -> None:
    class _Incomplete:
        table_name = "fixtures"

        def read(
            self,
            *,
            filters: dict[str, object] | None = None,
            columns: list[str] | None = None,
            as_of: datetime | None = None,
        ) -> pd.DataFrame:
            raise NotImplementedError

        def mark_closing(self, ids: Sequence[str]) -> int:
            raise NotImplementedError

    assert not isinstance(_Incomplete(), TableRepository)


class _FakeFixtureRepo(_FakeTableRepo):
    def upcoming(self, *, competition_id: str | None = None, days_ahead: int = 7) -> pd.DataFrame:
        raise NotImplementedError

    def by_competition_season(self, competition_id: str, season: str) -> pd.DataFrame:
        raise NotImplementedError


def test_fake_fixture_repo_satisfies_protocol() -> None:
    assert isinstance(_FakeFixtureRepo(), FixtureRepository)
    # tambien sigue satisfaciendo el protocolo generico del que hereda
    assert isinstance(_FakeFixtureRepo(), TableRepository)


class _FakeOddsRepo(_FakeTableRepo):
    def latest_before(self, fixture_id: str, market: str, *, as_of: datetime) -> pd.DataFrame:
        raise NotImplementedError

    def closing_line(self, fixture_id: str, market: str) -> pd.DataFrame:
        raise NotImplementedError


def test_fake_odds_repo_satisfies_protocol() -> None:
    assert isinstance(_FakeOddsRepo(), OddsRepository)


class _FakeFeatureRepo(_FakeTableRepo):
    def vector_as_of(
        self, fixture_id: str, feature_set: str, *, as_of: datetime
    ) -> dict[str, object] | None:
        raise NotImplementedError


def test_fake_feature_repo_satisfies_protocol() -> None:
    assert isinstance(_FakeFeatureRepo(), FeatureRepository)


class _FakeRawDocumentRepo:
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
        return "doc-id"

    def read(self, document_id: str) -> bytes:
        return b""

    def find(
        self, *, source: str, endpoint: str | None = None, since: datetime | None = None
    ) -> pd.DataFrame:
        raise NotImplementedError


def test_fake_raw_document_repo_satisfies_protocol() -> None:
    fake = _FakeRawDocumentRepo()
    assert isinstance(fake, RawDocumentRepository)
    assert (
        fake.store(
            source="fbref",
            endpoint="/matches",
            params={},
            content=b"<html></html>",
            content_type="text/html",
            status_code=200,
            fetched_at=datetime.now(UTC),
        )
        == "doc-id"
    )


class _FakeUnitOfWork:
    committed = False
    rolled_back = False

    def __enter__(self) -> _FakeUnitOfWork:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def test_fake_unit_of_work_satisfies_protocol_and_context_manager() -> None:
    uow = _FakeUnitOfWork()
    assert isinstance(uow, UnitOfWork)
    with uow as entered:
        assert entered is uow
        entered.commit()
    assert uow.committed is True
