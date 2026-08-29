"""NflSource mapping logic, tested against a small DataFrame shaped like
nfl_data_py.import_schedules's well-documented real output — never against
the live network."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from deportivas.contracts.tables import TEAM_ALIASES, TEAMS
from deportivas.ingest.aliases import TeamAliasResolver
from deportivas.ingest.ratelimit import RateLimiter
from deportivas.ingest.sources.nfl import NflSource
from deportivas.storage.duckdb_repo.raw_store import ParquetRawDocumentRepository
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository


def _source(tmp_path: Path) -> NflSource:
    raw_repo = ParquetRawDocumentRepository(tmp_path / "raw", tmp_path / "parquet")
    teams_repo = ParquetTableRepository(TEAMS, tmp_path / "parquet")
    aliases_repo = ParquetTableRepository(TEAM_ALIASES, tmp_path / "parquet")
    resolver = TeamAliasResolver(teams_repo, aliases_repo, sport="american_football")
    return NflSource(raw_repo=raw_repo, rate_limiter=RateLimiter(0.0), aliases=resolver)


def _schedule_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "game_id": "2025_10_KC_BUF",
        "season": 2025,
        "game_type": "REG",
        "week": 10,
        "gameday": "2025-11-16",
        "weekday": "Sunday",
        "gametime": "16:25",
        "away_team": "KC",
        "home_team": "BUF",
        "away_score": 24,
        "home_score": 27,
    }
    base.update(overrides)
    return base


def test_to_fixtures_maps_played_game(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row()])

    fixtures = source._to_fixtures(raw, competition_id="usa-nfl")

    row = fixtures.iloc[0]
    assert row["status"] == "finished"
    assert row["home_score"] == 27
    assert row["away_score"] == 24
    assert row["matchday"] == 10
    assert row["stage"] == "REG"
    # 16:25 ET en noviembre es EST (UTC-5) -> 21:25 UTC
    assert row["kickoff_utc"] == datetime(2025, 11, 16, 21, 25, tzinfo=UTC)
    assert bool(row["kickoff_is_estimated"]) is False


def test_to_fixtures_dst_summer_game_converts_edt(tmp_path: Path) -> None:
    """September games are still Eastern Daylight Time (UTC-4), not EST."""
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row(gameday="2025-09-07", week=1)])

    fixtures = source._to_fixtures(raw, competition_id="usa-nfl")

    assert fixtures.iloc[0]["kickoff_utc"] == datetime(2025, 9, 7, 20, 25, tzinfo=UTC)


def test_to_fixtures_unplayed_game_is_scheduled(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row(home_score=None, away_score=None)])

    fixtures = source._to_fixtures(raw, competition_id="usa-nfl")

    assert fixtures.iloc[0]["status"] == "scheduled"


def test_to_fixtures_missing_gametime_defaults_and_marks_estimated(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row(gametime=None)])

    fixtures = source._to_fixtures(raw, competition_id="usa-nfl")

    row = fixtures.iloc[0]
    assert bool(row["kickoff_is_estimated"]) is True
    assert row["kickoff_utc"].hour == 18  # 13:00 ET -> 18:00 UTC en EST


def test_to_fixtures_nan_game_type_is_none_stage(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row(game_type=float("nan"))])

    fixtures = source._to_fixtures(raw, competition_id="usa-nfl")

    assert fixtures.iloc[0]["stage"] is None


def test_to_fixtures_row_without_gameday_is_dropped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row(gameday=None)])

    fixtures = source._to_fixtures(raw, competition_id="usa-nfl")

    assert fixtures.empty


def test_to_fixtures_resolves_team_abbreviations(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = pd.DataFrame([_schedule_row()])

    fixtures = source._to_fixtures(raw, competition_id="usa-nfl")

    assert fixtures.iloc[0]["home_team_id"] == "american_football:buf"
    assert fixtures.iloc[0]["away_team_id"] == "american_football:kc"


def _play(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "game_id": "2025_10_KC_BUF",
        "season": 2025,
        "week": 10,
        "home_team": "BUF",
        "away_team": "KC",
        "posteam": "BUF",
        "defteam": "KC",
        "play_type": "run",
        "epa": 0.5,
        "success": 1,
    }
    base.update(overrides)
    return base


def _fixtures_lookup_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "fixture-buf-kc",
        "competition_id": "usa-nfl",
        "season": "2025",
        "home_team_id": "american_football:buf",
        "away_team_id": "american_football:kc",
    }
    base.update(overrides)
    return base


def test_to_team_game_stats_aggregates_offense_and_defense_epa(tmp_path: Path) -> None:
    source = _source(tmp_path)
    plays = pd.DataFrame(
        [
            _play(posteam="BUF", defteam="KC", epa=0.5, success=1),
            _play(posteam="BUF", defteam="KC", epa=0.3, success=0),
            _play(posteam="KC", defteam="BUF", epa=-0.2, success=0),
        ]
    )
    fixtures = pd.DataFrame([_fixtures_lookup_row()])

    result = source._to_team_game_stats(plays, fixtures=fixtures)

    buf = result[result["team_id"] == "american_football:buf"].iloc[0]
    assert buf["offensive_plays"] == 2
    assert buf["offensive_epa_per_play"] == pytest.approx(0.4)
    assert buf["offensive_success_rate"] == pytest.approx(0.5)
    assert buf["defensive_plays"] == 1
    assert buf["defensive_epa_per_play_allowed"] == pytest.approx(-0.2)
    assert bool(buf["is_home"]) is True

    kc = result[result["team_id"] == "american_football:kc"].iloc[0]
    assert kc["offensive_plays"] == 1
    assert kc["defensive_plays"] == 2
    assert bool(kc["is_home"]) is False


def test_to_team_game_stats_excludes_non_scrimmage_plays(tmp_path: Path) -> None:
    source = _source(tmp_path)
    plays = pd.DataFrame(
        [
            _play(posteam="BUF", defteam="KC", play_type="punt", epa=5.0),
            _play(posteam="BUF", defteam="KC", play_type="run", epa=0.5, success=1),
        ]
    )
    fixtures = pd.DataFrame([_fixtures_lookup_row()])

    result = source._to_team_game_stats(plays, fixtures=fixtures)

    buf = result[result["team_id"] == "american_football:buf"].iloc[0]
    assert buf["offensive_plays"] == 1
    assert buf["offensive_epa_per_play"] == pytest.approx(0.5)


def test_to_team_game_stats_excludes_plays_without_epa(tmp_path: Path) -> None:
    source = _source(tmp_path)
    plays = pd.DataFrame(
        [
            _play(posteam="BUF", defteam="KC", epa=None),
            _play(posteam="BUF", defteam="KC", epa=0.5, success=1),
        ]
    )
    fixtures = pd.DataFrame([_fixtures_lookup_row()])

    result = source._to_team_game_stats(plays, fixtures=fixtures)

    buf = result[result["team_id"] == "american_football:buf"].iloc[0]
    assert buf["offensive_plays"] == 1


def test_to_team_game_stats_skips_games_without_a_matching_fixture(tmp_path: Path) -> None:
    source = _source(tmp_path)
    plays = pd.DataFrame([_play()])
    fixtures = pd.DataFrame(
        columns=["id", "competition_id", "season", "home_team_id", "away_team_id"]
    )

    result = source._to_team_game_stats(plays, fixtures=fixtures)

    assert result.empty


def test_to_team_game_stats_team_with_no_plays_gets_none_not_zero_division(tmp_path: Path) -> None:
    """A team that never had the ball on offense (hypothetically) should read
    as 'no data' (None), never a spurious 0.0 average."""
    source = _source(tmp_path)
    plays = pd.DataFrame([_play(posteam="BUF", defteam="KC", epa=0.5, success=1)])
    fixtures = pd.DataFrame([_fixtures_lookup_row()])

    result = source._to_team_game_stats(plays, fixtures=fixtures)

    kc = result[result["team_id"] == "american_football:kc"].iloc[0]
    assert kc["offensive_plays"] == 0
    assert pd.isna(kc["offensive_epa_per_play"])


def test_to_team_game_stats_no_scrimmage_plays_returns_empty(tmp_path: Path) -> None:
    source = _source(tmp_path)
    plays = pd.DataFrame([_play(play_type="kickoff", epa=None)])
    fixtures = pd.DataFrame([_fixtures_lookup_row()])

    result = source._to_team_game_stats(plays, fixtures=fixtures)

    assert result.empty


def test_fetch_team_game_stats_waits_archives_and_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    waits: list[float] = []

    def fake_wait() -> float:
        waits.append(1.0)
        return 0.0

    monkeypatch.setattr(source, "_wait", fake_wait)
    monkeypatch.setattr(
        "deportivas.ingest.sources.nfl.nfl.import_pbp_data",
        lambda seasons, **kwargs: pd.DataFrame([_play()]),
    )
    fixtures = pd.DataFrame([_fixtures_lookup_row()])

    result = source.fetch_team_game_stats(seasons=[2025], fixtures=fixtures)

    assert len(waits) == 1
    assert len(result) == 2


def test_fetch_schedules_waits_archives_and_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    waits: list[float] = []

    def fake_wait() -> float:
        waits.append(1.0)
        return 0.0

    monkeypatch.setattr(source, "_wait", fake_wait)
    monkeypatch.setattr(
        "deportivas.ingest.sources.nfl.nfl.import_schedules",
        lambda years: pd.DataFrame([_schedule_row()]),
    )

    fixtures = source.fetch_schedules(competition_id="usa-nfl", seasons=[2025])

    assert len(waits) == 1
    assert len(fixtures) == 1
