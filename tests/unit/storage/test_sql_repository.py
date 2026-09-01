"""Tests for SqlTableRepository against a real Postgres instance.

Marked ``postgres``: skipped by default (see pyproject.toml addopts) because
most environments running the test suite don't have a live Postgres to talk
to. ``make db-up`` starts one locally; ``ci.yml`` runs a postgres service and
includes these via ``-m "not network"``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pandas as pd
import pytest
import sqlalchemy as sa
from tests.fixtures.synthetic_schema import WIDGET_EVENTS, WIDGETS

from deportivas.config.settings import get_settings
from deportivas.contracts.sqlalchemy_adapter import build_table
from deportivas.storage.sql_repo.repository import SqlTableRepository
from deportivas.storage.validation import ValidationError

pytestmark = pytest.mark.postgres


@pytest.fixture
def engine() -> sa.Engine:
    return sa.create_engine(get_settings().database_url, future=True)


@pytest.fixture(autouse=True)
def _fresh_tables(engine: sa.Engine) -> Iterator[None]:
    metadata = sa.MetaData()
    build_table(WIDGETS, metadata)
    build_table(WIDGET_EVENTS, metadata)
    metadata.drop_all(engine)
    metadata.create_all(engine)
    yield
    metadata.drop_all(engine)


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


def test_write_then_read_round_trips(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGETS, engine)
    assert repo.write(_rows()) == 1
    result = repo.read()
    assert len(result) == 1
    assert result.iloc[0]["id"] == "w1"


def test_rerunning_ingestion_is_idempotent(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGETS, engine)
    rows = _rows()
    repo.write(rows)
    repo.write(rows)
    repo.write(rows)
    assert len(repo.read()) == 1


def test_upsert_updates_non_key_columns(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGETS, engine)
    repo.write(_rows(value=1.0, ingested_at=datetime(2026, 1, 1, tzinfo=UTC)))
    repo.write(_rows(value=2.0, ingested_at=datetime(2026, 1, 2, tzinfo=UTC)))
    result = repo.read()
    assert len(result) == 1
    assert result.iloc[0]["value"] == pytest.approx(2.0)


def test_batch_with_intra_batch_duplicate_natural_key_is_rejected(engine: sa.Engine) -> None:
    """A single fetch reporting the same natural key twice is dirty source
    data, not a legitimate upsert (that happens across separate write() calls
    instead — see test_upsert_updates_non_key_columns). Pandera's own
    ``unique=natural_key`` check rejects both copies and logs why; it also
    means Postgres's ON CONFLICT never sees a batch with a repeated key
    (which it would reject outright with 'cannot affect row a second time')."""
    repo = SqlTableRepository(WIDGETS, engine)
    batch = pd.concat(
        [
            _rows(id="w-clean"),
            _rows(id="w-dup", value=1.0),
            _rows(id="w-dup", value=2.0),
        ],
        ignore_index=True,
    )
    written = repo.write(batch)
    assert written == 1
    assert list(repo.read()["id"]) == ["w-clean"]


def test_constructor_rejects_unknown_temporal_column(engine: sa.Engine) -> None:
    with pytest.raises(ValueError, match="no existe"):
        SqlTableRepository(WIDGETS, engine, temporal_column="does_not_exist")


def test_write_empty_dataframe_is_a_noop(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGETS, engine)
    empty = pd.DataFrame(columns=list(WIDGETS.column_names))
    assert repo.write(empty) == 0
    assert repo.read().empty


def test_all_rows_invalid_returns_zero_written(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGETS, engine)
    assert repo.write(_rows(id=None)) == 0
    assert repo.read().empty


def test_read_with_equality_filter(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGETS, engine)
    repo.write(_rows(id="w1", category="cat-a"))
    repo.write(_rows(id="w2", category="cat-b"))
    result = repo.read(filters={"category": "cat-b"})
    assert list(result["id"]) == ["w2"]


def test_read_unknown_column_raises_keyerror(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGETS, engine)
    repo.write(_rows())
    with pytest.raises(KeyError):
        repo.read(filters={"does_not_exist": 1})


def test_as_of_without_temporal_column_raises(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGETS, engine)
    with pytest.raises(ValueError, match="no tiene columna temporal"):
        repo.read(as_of=datetime.now(UTC))


def test_as_of_filters_on_temporal_column(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGET_EVENTS, engine, temporal_column="captured_at")
    repo.write(
        pd.DataFrame(
            [
                {
                    "id": "e1",
                    "widget_id": "w1",
                    "captured_at": datetime(2026, 1, 1, tzinfo=UTC),
                    "payload": "early",
                    "is_closing": False,
                },
                {
                    "id": "e2",
                    "widget_id": "w1",
                    "captured_at": datetime(2026, 1, 5, tzinfo=UTC),
                    "payload": "late",
                    "is_closing": False,
                },
            ]
        )
    )
    result = repo.read(as_of=datetime(2026, 1, 2, tzinfo=UTC))
    assert list(result["id"]) == ["e1"]


def test_append_only_table_never_deduplicates(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGET_EVENTS, engine, temporal_column="captured_at")
    same_natural_key = {
        "widget_id": "w1",
        "captured_at": datetime(2026, 1, 1, tzinfo=UTC),
        "is_closing": False,
    }
    repo.write(pd.DataFrame([{"id": "e1", "payload": "first", **same_natural_key}]))
    repo.write(pd.DataFrame([{"id": "e2", "payload": "second", **same_natural_key}]))
    assert len(repo.read()) == 2


def test_mark_closing_flags_only_the_given_ids(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGET_EVENTS, engine)
    repo.write(
        pd.DataFrame(
            [
                {
                    "id": "e1",
                    "widget_id": "w1",
                    "captured_at": datetime(2026, 1, 1, tzinfo=UTC),
                    "payload": "early",
                    "is_closing": False,
                },
                {
                    "id": "e2",
                    "widget_id": "w1",
                    "captured_at": datetime(2026, 1, 2, tzinfo=UTC),
                    "payload": "late",
                    "is_closing": False,
                },
            ]
        )
    )

    changed = repo.mark_closing(["e2"])

    assert changed == 1
    result = repo.read().set_index("id")
    assert not result.loc["e1", "is_closing"]
    assert result.loc["e2", "is_closing"]


def test_mark_closing_is_idempotent(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGET_EVENTS, engine)
    repo.write(
        pd.DataFrame(
            [
                {
                    "id": "e1",
                    "widget_id": "w1",
                    "captured_at": datetime(2026, 1, 1, tzinfo=UTC),
                    "payload": "x",
                    "is_closing": False,
                }
            ]
        )
    )

    assert repo.mark_closing(["e1"]) == 1
    assert repo.mark_closing(["e1"]) == 0


def test_mark_closing_empty_ids_is_a_noop(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGET_EVENTS, engine)
    assert repo.mark_closing([]) == 0


def test_mark_closing_raises_without_is_closing_column(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGETS, engine)
    with pytest.raises(ValueError, match="is_closing"):
        repo.mark_closing(["w1"])


def test_invalid_rows_are_rejected_and_valid_rows_still_written(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGETS, engine)
    good = _rows(id="w-good")
    bad = _rows(id=None)
    batch = pd.concat([good, bad], ignore_index=True)
    assert repo.write(batch) == 1
    assert list(repo.read()["id"]) == ["w-good"]


def test_structurally_wrong_dataframe_raises_validation_error(engine: sa.Engine) -> None:
    repo = SqlTableRepository(WIDGETS, engine)
    wrong_shape = pd.DataFrame([{"totally": "different", "columns": 1}])
    with pytest.raises(ValidationError):
        repo.write(wrong_shape)
