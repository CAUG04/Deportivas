"""Common contract every ingestion adapter follows.

There is no single ``fetch()`` method shared across FBref, The Odds API and
pybaseball: their real data (schedules, team stats, odds, box scores) has
different shapes per sport, and forcing them into one generic signature would
either lose type information or degrade to an untyped dict blob. What *is*
common — and what every concrete adapter in ``ingest/sources/`` gets from
this base class instead of reimplementing — is rate limiting and archiving
into the raw layer before anything gets parsed (rule of Fase 1 / Fase 10).

Two archiving paths exist because the underlying libraries differ in what
they expose:

* Adapters wrapping ``soccerdata`` (FBref, Understat, ClubElo, ESPN,
  football-data.co.uk) never see raw HTTP bytes directly — ``soccerdata``
  downloads to its own on-disk cache and hands back an already-parsed
  DataFrame. ``archive_cache_dir`` recovers the raw files it just wrote (by
  mtime, since the last call) and archives each one.
* Adapters that own the HTTP call themselves (The Odds API) archive the
  response bytes directly via ``archive_bytes`` — no cache-dir scanning
  needed.
* ``nfl_data_py``, ``pybaseball`` and ``sportsdataverse`` expose neither: they
  fetch remote CSV/Parquet straight into a DataFrame with no raw bytes and no
  on-disk cache. Their adapters archive the resulting DataFrame itself,
  serialised to Parquet, as the earliest available snapshot — documented in
  each adapter, not hidden.
"""

from __future__ import annotations

from abc import ABC
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deportivas.ingest.ratelimit import RateLimiter
    from deportivas.storage.protocols import RawDocumentRepository

_CONTENT_TYPE_BY_SUFFIX = {
    ".html": "text/html",
    ".json": "application/json",
    ".csv": "text/csv",
}


def guess_content_type(path: Path) -> str:
    return _CONTENT_TYPE_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")


def archive_cache_dir(
    raw_repo: RawDocumentRepository,
    *,
    source: str,
    cache_dir: Path,
    since: datetime,
) -> list[str]:
    """Archives every file under ``cache_dir`` modified at or after ``since``.

    Meant to be called right after a ``soccerdata`` reader method: capture
    ``since = datetime.now(UTC)`` immediately before the call, then archive
    afterwards. Files untouched by this call (older seasons already cached
    from a previous run) are correctly skipped — they were archived then.
    """
    if not cache_dir.exists():
        return []
    document_ids = []
    for path in sorted(cache_dir.rglob("*")):
        if not path.is_file():
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        if mtime < since:
            continue
        document_ids.append(
            raw_repo.store(
                source=source,
                endpoint=str(path.relative_to(cache_dir)),
                params={},
                content=path.read_bytes(),
                content_type=guess_content_type(path),
                status_code=None,
                fetched_at=mtime,
            )
        )
    return document_ids


class DataSource(ABC):
    """Base class for every source adapter in ``ingest/sources/``."""

    name: str

    def __init__(self, *, raw_repo: RawDocumentRepository, rate_limiter: RateLimiter) -> None:
        self._raw_repo = raw_repo
        self._rate_limiter = rate_limiter

    def _wait(self) -> float:
        """Blocks per this adapter's rate limit. Call before every real network call."""
        return self._rate_limiter.wait()

    def _archive_cache_dir(self, cache_dir: Path, *, since: datetime) -> list[str]:
        return archive_cache_dir(self._raw_repo, source=self.name, cache_dir=cache_dir, since=since)

    def _archive_bytes(
        self,
        *,
        endpoint: str,
        params: dict[str, object],
        content: bytes,
        content_type: str,
        status_code: int | None,
    ) -> str:
        return self._raw_repo.store(
            source=self.name,
            endpoint=endpoint,
            params=params,
            content=content,
            content_type=content_type,
            status_code=status_code,
            fetched_at=datetime.now(UTC),
        )
