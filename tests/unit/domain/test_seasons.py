"""seasons.py: recent season labels for a competition's incremental
ingestion (Fase 8's daily workflow), football's 2-year code vs every other
sport's single starting year."""

from __future__ import annotations

from datetime import date

from deportivas.config.catalog import Competition, CompetitionOdds, CompetitionSources
from deportivas.domain.enums import RefreshCadence, Sport
from deportivas.domain.seasons import season_labels


def _competition(**overrides: object) -> Competition:
    base: dict[str, object] = {
        "id": "eng-premier-league",
        "name": "Premier League",
        "country": "England",
        "sport": Sport.FOOTBALL,
        "tier": 1,
        "season_start_month": 8,
        "season_end_month": 5,
        "seasons_back": 5,
        "refresh": RefreshCadence.DAILY,
        "enabled": True,
        "sources": CompetitionSources(),
        "odds": CompetitionOdds(the_odds_api="soccer_epl"),
    }
    base.update(overrides)
    return Competition.model_validate(base)


def test_football_uses_hyphenated_two_year_code_after_season_start() -> None:
    competition = _competition(season_start_month=8)
    labels = season_labels(competition, count=1, today=date(2025, 9, 1))
    assert labels == ["2526"]


def test_football_still_in_last_seasons_code_before_season_start() -> None:
    competition = _competition(season_start_month=8)
    labels = season_labels(competition, count=1, today=date(2025, 3, 1))
    assert labels == ["2425"]


def test_football_on_the_exact_start_month_counts_as_started() -> None:
    competition = _competition(season_start_month=8)
    labels = season_labels(competition, count=1, today=date(2025, 8, 1))
    assert labels == ["2526"]


def test_american_sport_uses_single_starting_year() -> None:
    competition = _competition(
        sport=Sport.AMERICAN_FOOTBALL, season_start_month=9, season_end_month=2
    )
    labels = season_labels(competition, count=1, today=date(2025, 12, 1))
    assert labels == ["2025"]


def test_american_sport_before_season_start_is_still_last_year() -> None:
    competition = _competition(
        sport=Sport.AMERICAN_FOOTBALL, season_start_month=9, season_end_month=2
    )
    labels = season_labels(competition, count=1, today=date(2026, 1, 15))
    assert labels == ["2025"]


def test_count_returns_multiple_labels_newest_first() -> None:
    competition = _competition(season_start_month=8)
    labels = season_labels(competition, count=3, today=date(2025, 9, 1))
    assert labels == ["2526", "2425", "2324"]


def test_default_count_is_two() -> None:
    competition = _competition(season_start_month=8)
    assert len(season_labels(competition, today=date(2025, 9, 1))) == 2


def test_football_year_wraparound_at_century_boundary() -> None:
    competition = _competition(season_start_month=8)
    labels = season_labels(competition, count=1, today=date(2099, 9, 1))
    assert labels == ["9900"]
