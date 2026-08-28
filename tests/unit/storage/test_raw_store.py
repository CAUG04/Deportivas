"""Tests for the append-only raw layer (Fase 10's most important decision,
built in Fase 1 because ingestion needs it from day one)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from deportivas.storage.duckdb_repo.raw_store import ParquetRawDocumentRepository


def _repo(tmp_path: Path) -> ParquetRawDocumentRepository:
    return ParquetRawDocumentRepository(tmp_path / "raw", tmp_path / "parquet")


def test_store_then_read_round_trips_bytes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    doc_id = repo.store(
        source="fbref",
        endpoint="/schedule",
        params={"league": "ENG-Premier League"},
        content=b"<html>hola</html>",
        content_type="text/html",
        status_code=200,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert repo.read(doc_id) == b"<html>hola</html>"


def test_read_unknown_document_raises_keyerror(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(KeyError):
        repo.read("does-not-exist")


def test_identical_content_is_not_duplicated_on_disk(tmp_path: Path) -> None:
    """Two fetches with byte-identical content share one file: the metadata
    row is what tracks that a fetch happened, not a fresh copy per fetch."""
    repo = _repo(tmp_path)
    content = b"same bytes"
    repo.store(
        source="fbref",
        endpoint="/schedule",
        params={},
        content=content,
        content_type="text/html",
        status_code=200,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    repo.store(
        source="fbref",
        endpoint="/schedule",
        params={},
        content=content,
        content_type="text/html",
        status_code=200,
        fetched_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    payload_files = list((tmp_path / "raw" / "fbref").rglob("*.html"))
    assert len(payload_files) == 1


def test_every_fetch_gets_its_own_metadata_row_even_if_content_repeats(tmp_path: Path) -> None:
    """The raw layer is an audit trail of fetch attempts, not a dedup cache:
    fetching the same page twice records two rows even though only one file
    is stored on disk."""
    repo = _repo(tmp_path)
    content = b"same bytes"
    for day in (1, 2):
        repo.store(
            source="fbref",
            endpoint="/schedule",
            params={},
            content=content,
            content_type="text/html",
            status_code=200,
            fetched_at=datetime(2026, 1, day, tzinfo=UTC),
        )
    rows = repo.find(source="fbref")
    assert len(rows) == 2


def test_find_filters_by_endpoint_and_since(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.store(
        source="fbref",
        endpoint="/schedule",
        params={},
        content=b"a",
        content_type="text/html",
        status_code=200,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    repo.store(
        source="fbref",
        endpoint="/team-stats",
        params={},
        content=b"b",
        content_type="text/html",
        status_code=200,
        fetched_at=datetime(2026, 1, 5, tzinfo=UTC),
    )
    only_schedule = repo.find(source="fbref", endpoint="/schedule")
    assert len(only_schedule) == 1
    assert only_schedule.iloc[0]["endpoint"] == "/schedule"

    recent = repo.find(source="fbref", since=datetime(2026, 1, 3, tzinfo=UTC))
    assert len(recent) == 1
    assert recent.iloc[0]["endpoint"] == "/team-stats"


def test_content_hash_is_deterministic_sha256(tmp_path: Path) -> None:
    import hashlib

    repo = _repo(tmp_path)
    content = b"deterministic content"
    doc_id = repo.store(
        source="fbref",
        endpoint="/x",
        params={},
        content=content,
        content_type="application/json",
        status_code=200,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    rows = repo.find(source="fbref")
    row = rows[rows["id"] == doc_id].iloc[0]
    assert row["content_hash"] == hashlib.sha256(content).hexdigest()


def test_status_code_can_be_none_for_a_local_or_library_source(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    doc_id = repo.store(
        source="pybaseball",
        endpoint="schedule_and_record",
        params={"season": 2024, "team": "PHI"},
        content=b"[]",
        content_type="application/json",
        status_code=None,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert repo.read(doc_id) == b"[]"
