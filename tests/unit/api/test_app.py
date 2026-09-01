"""app.py: read-only FastAPI wrapper over api/views.py."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FIXTURES, SIGNALS
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository

_COMPETITION = "eng-premier-league"
_FIXTURE_ID = "fx1"
_KICKOFF = datetime(2024, 1, 10, tzinfo=UTC)
_ENTRY_AT = _KICKOFF - timedelta(days=1)


@pytest.fixture(autouse=True)
def _settings_in_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DEPORTIVAS_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client() -> TestClient:
    # el modulo lee get_settings().cors_allow_origins una sola vez al
    # importarse; recargarlo es lo unico que hace que un cambio de entorno
    # dentro de un test se refleje en el middleware CORS
    import deportivas.api.app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.app)


def _fixture_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": _FIXTURE_ID,
        "competition_id": _COMPETITION,
        "season": "s1",
        "kickoff_utc": _KICKOFF,
        "home_team_id": "football:arsenal",
        "away_team_id": "football:chelsea",
        "status": "scheduled",
        "stage": None,
        "matchday": None,
        "home_score": None,
        "away_score": None,
        "home_score_ht": None,
        "away_score_ht": None,
        "kickoff_is_estimated": False,
        "source": "test",
        "ingested_at": _ENTRY_AT,
    }
    base.update(overrides)
    return base


def _signal_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "sig1",
        "fixture_id": _FIXTURE_ID,
        "competition_id": _COMPETITION,
        "season": "s1",
        "market": "1x2",
        "selection": "home",
        "line": None,
        "model_name": "m1",
        "model_version": "v1",
        "prob_model": 0.6,
        "prob_fair": 0.5,
        "fair_price": 2.0,
        "entry_price": 1.8,
        "entry_bookmaker": "pinnacle",
        "entry_captured_at": _ENTRY_AT,
        "devig_method": "multiplicative",
        "edge": 0.1,
        "tier": "alta",
        "tier_reasons": "{}",
        "stake_fraction": 0.02,
        "created_at": _ENTRY_AT,
        "source": "test",
        "ingested_at": _ENTRY_AT,
    }
    base.update(overrides)
    return base


def test_health() -> None:
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_competitions_includes_known_competition() -> None:
    response = _client().get("/competitions")
    assert response.status_code == 200
    ids = {c["id"] for c in response.json()}
    assert _COMPETITION in ids


def test_list_signals_returns_empty_list_without_data() -> None:
    response = _client().get(f"/competitions/{_COMPETITION}/signals")
    assert response.status_code == 200
    assert response.json() == []


def test_list_signals_returns_enriched_data() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(pd.DataFrame([_signal_row()]))

    response = _client().get(f"/competitions/{_COMPETITION}/signals")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "sig1"
    assert body[0]["tier"] == "alta"


def test_list_signals_market_query_param() -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(
        pd.DataFrame(
            [_signal_row(id="sig-1x2", market="1x2"), _signal_row(id="sig-btts", market="btts")]
        )
    )

    response = _client().get(f"/competitions/{_COMPETITION}/signals", params={"market": "btts"})

    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ["sig-btts"]


def test_list_signals_404_for_unknown_competition() -> None:
    response = _client().get("/competitions/no-existe/signals")
    assert response.status_code == 404
    assert "no-existe" in response.json()["detail"]


def test_backtest_report_returns_empty_summary_without_data() -> None:
    response = _client().get(f"/competitions/{_COMPETITION}/backtest")
    assert response.status_code == 200
    body = response.json()
    assert body["overall"]["n"] == 0
    assert body["overall"]["mean_clv"] is None


def test_backtest_report_404_for_unknown_competition() -> None:
    response = _client().get("/competitions/no-existe/backtest")
    assert response.status_code == 404


def test_cors_disabled_by_default() -> None:
    response = _client().get("/health", headers={"Origin": "http://localhost:5173"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_enabled_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPORTIVAS_CORS_ALLOW_ORIGINS", '["http://localhost:5173"]')
    get_settings.cache_clear()
    client = _client()

    response = client.get("/health", headers={"Origin": "http://localhost:5173"})

    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
