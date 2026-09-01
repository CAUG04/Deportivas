"""archive_cache_dir and DataSource: the mechanism that recovers raw files
soccerdata-backed adapters never hand back directly, and archives them into
the append-only raw layer."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from deportivas.ingest.base import DataSource, archive_cache_dir, guess_content_type
from deportivas.ingest.ratelimit import RateLimiter
from deportivas.storage.duckdb_repo.raw_store import ParquetRawDocumentRepository


def _raw_repo(tmp_path: Path) -> ParquetRawDocumentRepository:
    return ParquetRawDocumentRepository(tmp_path / "raw", tmp_path / "parquet")


def test_guess_content_type_known_extensions() -> None:
    assert guess_content_type(Path("x.html")) == "text/html"
    assert guess_content_type(Path("x.json")) == "application/json"
    assert guess_content_type(Path("x.csv")) == "text/csv"


def test_guess_content_type_unknown_extension_falls_back() -> None:
    assert guess_content_type(Path("x.bin")) == "application/octet-stream"
    assert guess_content_type(Path("x")) == "application/octet-stream"


def test_archive_cache_dir_on_missing_dir_returns_empty(tmp_path: Path) -> None:
    repo = _raw_repo(tmp_path)
    result = archive_cache_dir(
        repo, source="fbref", cache_dir=tmp_path / "does-not-exist", since=datetime.now(UTC)
    )
    assert result == []


def test_archive_cache_dir_archives_files_modified_since(tmp_path: Path) -> None:
    repo = _raw_repo(tmp_path)
    cache_dir = tmp_path / "soccerdata_cache" / "FBref"
    cache_dir.mkdir(parents=True)
    since = datetime.now(UTC)
    time.sleep(0.05)  # margen real de reloj: mtime > since sin necesidad de mock
    (cache_dir / "schedule_ENG_2425.html").write_text("<html>ok</html>", encoding="utf-8")

    ids = archive_cache_dir(repo, source="fbref", cache_dir=cache_dir, since=since)

    assert len(ids) == 1
    stored = repo.find(source="fbref")
    assert len(stored) == 1
    assert stored.iloc[0]["endpoint"] == "schedule_ENG_2425.html"
    assert stored.iloc[0]["content_type"] == "text/html"


def test_archive_cache_dir_skips_files_older_than_since(tmp_path: Path) -> None:
    repo = _raw_repo(tmp_path)
    cache_dir = tmp_path / "soccerdata_cache" / "FBref"
    cache_dir.mkdir(parents=True)
    (cache_dir / "old_season.html").write_text("<html>old</html>", encoding="utf-8")
    time.sleep(0.05)
    since = datetime.now(UTC)

    ids = archive_cache_dir(repo, source="fbref", cache_dir=cache_dir, since=since)

    assert ids == []
    assert repo.find(source="fbref").empty


def test_archive_cache_dir_ignores_subdirectories(tmp_path: Path) -> None:
    repo = _raw_repo(tmp_path)
    cache_dir = tmp_path / "soccerdata_cache" / "FBref"
    (cache_dir / "nested").mkdir(parents=True)
    since = datetime.now(UTC)

    ids = archive_cache_dir(repo, source="fbref", cache_dir=cache_dir, since=since)

    assert ids == []


class _FakeSource(DataSource):
    name = "fake"

    def touch(self) -> float:
        return self._wait()

    def archive(self, cache_dir: Path, since: datetime) -> list[str]:
        return self._archive_cache_dir(cache_dir, since=since)

    def archive_one(self, content: bytes) -> str:
        return self._archive_bytes(
            endpoint="/x",
            params={},
            content=content,
            content_type="application/json",
            status_code=200,
        )


def test_data_source_wait_delegates_to_rate_limiter(tmp_path: Path) -> None:
    limiter = RateLimiter(0.0)
    source = _FakeSource(raw_repo=_raw_repo(tmp_path), rate_limiter=limiter)
    assert source.touch() == 0.0


def test_data_source_archive_cache_dir_uses_own_name(tmp_path: Path) -> None:
    repo = _raw_repo(tmp_path)
    limiter = RateLimiter(0.0)
    source = _FakeSource(raw_repo=repo, rate_limiter=limiter)
    cache_dir = tmp_path / "cache" / "fake"
    cache_dir.mkdir(parents=True)
    since = datetime.now(UTC)
    time.sleep(0.05)
    (cache_dir / "page.json").write_text("{}", encoding="utf-8")

    ids = source.archive(cache_dir, since)

    assert len(ids) == 1
    assert repo.find(source="fake").iloc[0]["endpoint"] == "page.json"


def test_data_source_archive_bytes(tmp_path: Path) -> None:
    repo = _raw_repo(tmp_path)
    source = _FakeSource(raw_repo=repo, rate_limiter=RateLimiter(0.0))
    doc_id = source.archive_one(b'{"ok": true}')
    assert repo.read(doc_id) == b'{"ok": true}'
    assert repo.find(source="fake").iloc[0]["endpoint"] == "/x"
