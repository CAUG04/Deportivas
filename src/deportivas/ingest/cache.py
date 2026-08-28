"""A small disk cache for sources that do not manage their own (FBref,
Understat, ESPN and football-data.co.uk go through ``soccerdata``, which
already caches to disk — see ``ingest/base.py``). ``nfl_data_py``,
``pybaseball``, ``sportsdataverse`` and The Odds API do not, so re-running a
backfill would otherwise re-fetch everything every time.

This is a convenience cache, not the raw layer: safe to delete entirely
(``DEPORTIVAS_DATA_DIR/cache``), unlike ``data/raw`` which must never be
touched. A cache miss just means a real fetch; a cache hit still gets
archived into the raw layer by the caller exactly like a fresh fetch would,
because "this ran today" is what the raw layer's ``fetched_at`` records, not
"this changed today".
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path


def cache_key(source: str, endpoint: str, params: dict[str, object]) -> str:
    payload = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{source}_{digest}"


class DiskCache:
    """Age is tracked in a JSON sidecar (``stored_at``), not the file's mtime.

    mtime comes straight from the OS clock, which can't be faked in tests
    without actually touching the system clock; a self-written timestamp can
    be driven by the same injectable ``clock`` the rest of the cache uses.
    """

    def __init__(self, cache_dir: Path, *, clock: Callable[[], float] = time.time) -> None:
        self._cache_dir = cache_dir
        self._clock = clock

    def get(self, key: str, *, max_age_seconds: float) -> bytes | None:
        content_path, meta_path = self._paths(key)
        if not content_path.exists() or not meta_path.exists():
            return None
        stored_at = json.loads(meta_path.read_text(encoding="utf-8"))["stored_at"]
        age = self._clock() - stored_at
        if age > max_age_seconds:
            return None
        return content_path.read_bytes()

    def put(self, key: str, content: bytes) -> None:
        content_path, meta_path = self._paths(key)
        content_path.parent.mkdir(parents=True, exist_ok=True)
        content_path.write_bytes(content)
        meta_path.write_text(json.dumps({"stored_at": self._clock()}), encoding="utf-8")

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self._cache_dir / f"{key}.bin", self._cache_dir / f"{key}.meta.json"
