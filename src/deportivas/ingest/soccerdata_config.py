"""Fixes a real gap, not just something Fase 10 verifies: ``soccerdata``
resolves every league name it's given against its own ``LEAGUE_DICT``,
extensible only through a JSON file it looks for at
``{SOCCERDATA_DIR:-~/soccerdata}/config/league_dict.json`` (see the
installed package's own ``soccerdata._config``, version pinned in
``pyproject.toml``). The stock dictionary only covers the 5 major European
leagues plus a few international tournaments — none of the
``soccerdata_key`` values this project declares for Eredivisie, Primeira
Liga, the three UEFA competitions or Liga BetPlay Dimayor exist in it.
Without this file, ``soccerdata.FBref(leagues=["NED-Eredivisie"])`` raises
``ValueError`` before attempting any network call at all — every one of
those six competitions' fbref/understat/espn/footballdata ingestion was
broken until this module started writing the file competitions.yaml's own
header comment already promised existed.

Deliberately does **not** import ``soccerdata`` (not even
``soccerdata._config``): that package computes its ``LEAGUE_DICT`` once, as
top-level module code, reading this same file at that exact moment and
never again. Importing it here — even just to read its config directory
constant — would trigger that computation before this function had a
chance to write anything, defeating the whole point. The path resolution
below is therefore a deliberate, documented duplicate of
``soccerdata._config``'s own convention (``SOCCERDATA_DIR`` env var,
default ``~/soccerdata``) that has to be kept in sync by hand if a future
soccerdata release changes it.

Callers (the ``ingest fbref-*``/``understat-*``/``espn-*``/
``footballdata-*`` CLI commands in ``cli.py``) must call
``ensure_custom_league_dict()`` before their own lazy import of the
adapter module — that adapter module is what actually triggers
``import soccerdata``.
"""

from __future__ import annotations

import calendar
import json
import os
from pathlib import Path

from deportivas.config.catalog import CompetitionSources, load_competitions
from deportivas.domain.enums import Sport


def _soccerdata_config_dir() -> Path:
    base_dir = Path(os.environ.get("SOCCERDATA_DIR", Path.home() / "soccerdata"))
    return base_dir / "config"


def _entry_for_sources(
    sources: CompetitionSources, season_start_month: int, season_end_month: int
) -> dict[str, str]:
    entry: dict[str, str] = {
        "season_start": calendar.month_abbr[season_start_month],
        "season_end": calendar.month_abbr[season_end_month],
    }
    field_to_key = {
        "fbref": "FBref",
        "understat": "Understat",
        "club_elo": "ClubElo",
        "match_history": "MatchHistory",
        "espn": "ESPN",
    }
    for field, league_dict_key in field_to_key.items():
        value = getattr(sources, field, None)
        if value is not None:
            entry[league_dict_key] = value
    return entry


def ensure_custom_league_dict(config_dir: Path | None = None) -> Path:
    """Writes/updates ``league_dict.json`` from every enabled football
    competition's ``sources.soccerdata_key`` in ``config/competitions.yaml``
    — adding a league stays "edit the YAML", never "hand-edit a
    third-party package's JSON". Merges with whatever the file already
    contains rather than overwriting it outright, so a key a user placed
    there for a league outside this project survives; this project's own
    keys always win, in case a ``soccerdata_key`` changes.

    ``config_dir`` is exposed purely for tests; real callers omit it and
    get soccerdata's own resolved config directory.
    """
    target_dir = config_dir if config_dir is not None else _soccerdata_config_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "league_dict.json"

    existing: dict[str, object] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            existing = json.load(fh)

    ours = {
        competition.sources.soccerdata_key: _entry_for_sources(
            competition.sources, competition.season_start_month, competition.season_end_month
        )
        for competition in load_competitions().enabled
        if competition.sport is Sport.FOOTBALL and competition.sources.soccerdata_key is not None
    }
    merged = {**existing, **ours}
    path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    return path
