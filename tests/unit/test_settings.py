"""Settings must come from the environment, never be hard-coded, and the odds
API key must never leak into a repr/log line (it is a secret)."""

from __future__ import annotations

from pathlib import Path

import pytest

from deportivas.config.settings import PROJECT_ROOT, Settings, StorageBackend, get_settings


def test_defaults_do_not_require_any_env_var() -> None:
    settings = Settings(_env_file=None)
    assert settings.storage_backend is StorageBackend.DUCKDB
    assert settings.the_odds_api_key is None
    assert settings.has_odds_api_key is False


def test_odds_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPORTIVAS_THE_ODDS_API_KEY", "s3cr3t")
    settings = Settings(_env_file=None)
    assert settings.has_odds_api_key is True
    assert settings.the_odds_api_key is not None
    assert settings.the_odds_api_key.get_secret_value() == "s3cr3t"


def test_secret_never_appears_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPORTIVAS_THE_ODDS_API_KEY", "s3cr3t-value")
    settings = Settings(_env_file=None)
    assert "s3cr3t-value" not in repr(settings)
    assert "s3cr3t-value" not in str(settings)


def test_invalid_log_level_rejected() -> None:
    with pytest.raises(ValueError, match="log_level"):
        Settings(_env_file=None, log_level="not-a-level")


def test_storage_backend_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPORTIVAS_STORAGE_BACKEND", "postgres")
    settings = Settings(_env_file=None)
    assert settings.storage_backend is StorageBackend.POSTGRES


def test_derived_paths_are_under_data_dir(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    assert settings.raw_dir == tmp_path / "raw"
    assert settings.parquet_dir == tmp_path / "parquet"
    assert settings.cache_dir == tmp_path / "cache"


def test_ensure_directories_creates_tree(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    settings.ensure_directories()
    assert settings.raw_dir.exists()
    assert settings.parquet_dir.exists()
    assert settings.cache_dir.exists()


def test_export_dir_is_under_frontend_public_data() -> None:
    settings = Settings(_env_file=None)
    assert settings.export_dir == PROJECT_ROOT / "frontend" / "public" / "data"


def test_get_settings_is_cached_and_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("DEPORTIVAS_LOG_LEVEL", "DEBUG")
    try:
        first = get_settings()
        second = get_settings()
        assert first is second  # el cache devuelve la misma instancia
        assert first.log_level == "DEBUG"
    finally:
        get_settings.cache_clear()


def test_settings_are_frozen() -> None:
    settings = Settings(_env_file=None)
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError en frozen model
        settings.log_level = "DEBUG"  # type: ignore[misc]
