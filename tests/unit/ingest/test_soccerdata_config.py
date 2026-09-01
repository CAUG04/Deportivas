"""ensure_custom_league_dict: every football competition's soccerdata_key
becomes an entry soccerdata itself can resolve, without ever importing
soccerdata (see the module docstring for why that matters)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deportivas.ingest.soccerdata_config import ensure_custom_league_dict


def test_writes_an_entry_per_football_competition_with_a_soccerdata_key(tmp_path: Path) -> None:
    path = ensure_custom_league_dict(config_dir=tmp_path)

    assert path == tmp_path / "league_dict.json"
    entries = json.loads(path.read_text(encoding="utf-8"))

    # Las 5 grandes ligas ya vienen de fabrica en soccerdata -- reescribirlas
    # con los mismos valores es inofensivo, no hace falta excluirlas.
    assert "ENG-Premier League" in entries
    assert entries["ENG-Premier League"]["FBref"] == "Premier League"
    assert entries["ENG-Premier League"]["Understat"] == "EPL"
    assert entries["ENG-Premier League"]["season_start"] == "Aug"
    assert entries["ENG-Premier League"]["season_end"] == "May"

    # Las que de verdad importan: las que soccerdata no trae de fabrica.
    assert entries["NED-Eredivisie"]["FBref"] == "Eredivisie"
    assert entries["NED-Eredivisie"]["ESPN"] == "ned.1"
    assert "Understat" not in entries["NED-Eredivisie"]  # null en competitions.yaml

    assert entries["INT-Champions League"]["season_start"] == "Sep"
    assert entries["INT-Champions League"]["season_end"] == "May"
    assert "MatchHistory" not in entries["INT-Champions League"]


def test_colombia_is_a_calendar_year_league(tmp_path: Path) -> None:
    path = ensure_custom_league_dict(config_dir=tmp_path)
    entries = json.loads(path.read_text(encoding="utf-8"))
    assert entries["COL-Primera A"]["season_start"] == "Jan"
    assert entries["COL-Primera A"]["season_end"] == "Dec"


def test_american_sports_have_no_soccerdata_key(tmp_path: Path) -> None:
    path = ensure_custom_league_dict(config_dir=tmp_path)
    entries = json.loads(path.read_text(encoding="utf-8"))
    assert not any(key.startswith("usa-") for key in entries)


def test_merges_with_and_does_not_clobber_unrelated_existing_entries(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    existing_path = tmp_path / "league_dict.json"
    existing_path.write_text(json.dumps({"BRA-Serie A": {"FBref": "Serie A"}}), encoding="utf-8")

    path = ensure_custom_league_dict(config_dir=tmp_path)
    entries = json.loads(path.read_text(encoding="utf-8"))

    assert entries["BRA-Serie A"] == {"FBref": "Serie A"}
    assert "NED-Eredivisie" in entries


def test_our_keys_win_over_a_stale_existing_value(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    existing_path = tmp_path / "league_dict.json"
    existing_path.write_text(
        json.dumps({"NED-Eredivisie": {"FBref": "algo-viejo-y-mal"}}), encoding="utf-8"
    )

    path = ensure_custom_league_dict(config_dir=tmp_path)
    entries = json.loads(path.read_text(encoding="utf-8"))

    assert entries["NED-Eredivisie"]["FBref"] == "Eredivisie"


def test_default_config_dir_resolves_under_soccerdata_dir_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOCCERDATA_DIR", str(tmp_path / "soccerdata"))

    path = ensure_custom_league_dict()

    assert path == tmp_path / "soccerdata" / "config" / "league_dict.json"
    assert path.is_file()


def test_default_config_dir_falls_back_to_home_when_env_var_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SOCCERDATA_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    path = ensure_custom_league_dict()

    assert path == tmp_path / "soccerdata" / "config" / "league_dict.json"
    assert path.is_file()
