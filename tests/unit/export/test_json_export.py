"""json_export.py: writes api/views.py's models to static JSON files for
the $0 static frontend."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from deportivas.config.catalog import load_competitions
from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FIXTURES, SIGNALS
from deportivas.export.json_export import export_all, export_competition
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


def test_export_competition_raises_for_unknown_competition(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        export_competition("no-existe-esta-competicion", out_dir=tmp_path)


def test_export_competition_writes_every_tier_including_descartar(tmp_path: Path) -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame([_fixture_row()]))
    ParquetTableRepository(SIGNALS, settings.parquet_dir).write(
        pd.DataFrame([_signal_row(tier="descartar")])
    )

    written = export_competition(_COMPETITION, out_dir=tmp_path)

    assert written["signals"] == tmp_path / _COMPETITION / "signals.json"
    signals = json.loads(written["signals"].read_text())
    assert len(signals) == 1
    assert signals[0]["tier"] == "descartar"  # a diferencia de la API en vivo (only_actionable)


def test_export_competition_writes_backtest_json(tmp_path: Path) -> None:
    written = export_competition(_COMPETITION, out_dir=tmp_path)

    backtest = json.loads(written["backtest"].read_text())
    assert backtest["overall"]["n"] == 0
    assert backtest["overall"]["mean_clv"] is None


def test_export_all_writes_competitions_json_and_per_competition_files(tmp_path: Path) -> None:
    written = export_all([_COMPETITION], out_dir=tmp_path)

    assert written["competitions"] == tmp_path / "competitions.json"
    competitions = json.loads(written["competitions"].read_text())
    assert any(c["id"] == _COMPETITION for c in competitions)
    assert written[f"{_COMPETITION}:signals"] == tmp_path / _COMPETITION / "signals.json"
    assert written[f"{_COMPETITION}:backtest"] == tmp_path / _COMPETITION / "backtest.json"


def test_export_all_defaults_to_every_enabled_competition(tmp_path: Path) -> None:
    written = export_all(out_dir=tmp_path)

    enabled_ids = {c.id for c in load_competitions().enabled}
    assert {f"{cid}:signals" for cid in enabled_ids} <= set(written)
    assert {f"{cid}:backtest" for cid in enabled_ids} <= set(written)
