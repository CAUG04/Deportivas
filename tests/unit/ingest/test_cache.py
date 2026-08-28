from __future__ import annotations

from pathlib import Path

from deportivas.ingest.cache import DiskCache, cache_key


def test_cache_key_is_deterministic() -> None:
    a = cache_key("nfl", "import_schedules", {"years": [2024]})
    b = cache_key("nfl", "import_schedules", {"years": [2024]})
    assert a == b


def test_cache_key_differs_by_params() -> None:
    a = cache_key("nfl", "import_schedules", {"years": [2024]})
    b = cache_key("nfl", "import_schedules", {"years": [2023]})
    assert a != b


def test_cache_key_ignores_dict_key_order() -> None:
    a = cache_key("nfl", "x", {"a": 1, "b": 2})
    b = cache_key("nfl", "x", {"b": 2, "a": 1})
    assert a == b


def test_miss_on_empty_cache(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    assert cache.get("k", max_age_seconds=60) is None


def test_put_then_get_round_trips(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put("k", b"hello")
    assert cache.get("k", max_age_seconds=60) == b"hello"


def test_entry_older_than_max_age_is_a_miss(tmp_path: Path) -> None:
    clock = [1000.0]
    cache = DiskCache(tmp_path, clock=lambda: clock[0])
    cache.put("k", b"hello")
    clock[0] += 3600  # una hora despues
    assert cache.get("k", max_age_seconds=60) is None


def test_entry_within_max_age_is_a_hit(tmp_path: Path) -> None:
    clock = [1000.0]
    cache = DiskCache(tmp_path, clock=lambda: clock[0])
    cache.put("k", b"hello")
    clock[0] += 30
    assert cache.get("k", max_age_seconds=60) == b"hello"


def test_put_overwrites_previous_value(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put("k", b"first")
    cache.put("k", b"second")
    assert cache.get("k", max_age_seconds=60) == b"second"
