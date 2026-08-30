"""closing.py: flags the true closing line on odds_snapshots once a fixture
has kicked off -- an optimization over settlement.py's runtime fallback,
never a correctness dependency."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FIXTURES, ODDS_SNAPSHOTS
from deportivas.ingest.closing import mark_closing_lines
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository

_COMPETITION = "test-comp"
_FIXTURE_ID = "fx1"
_KICKOFF = datetime(2024, 1, 10, tzinfo=UTC)

_odds_id_counter = itertools.count()


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
        "home_team_id": "home",
        "away_team_id": "away",
        "status": "finished",
        "stage": None,
        "matchday": None,
        "home_score": 2,
        "away_score": 1,
        "home_score_ht": None,
        "away_score_ht": None,
        "kickoff_is_estimated": False,
        "source": "test",
        "ingested_at": _KICKOFF,
    }
    base.update(overrides)
    return base


def _odds_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": f"odds{next(_odds_id_counter)}",
        "fixture_id": _FIXTURE_ID,
        "competition_id": _COMPETITION,
        "season": "s1",
        "bookmaker": "pinnacle",
        "market": "over_under",
        "selection": "over",
        "line": 2.5,
        "price": 1.9,
        "captured_at": _KICKOFF - timedelta(hours=1),
        "is_closing": False,
        "source": "test",
        "ingested_at": _KICKOFF,
    }
    base.update(overrides)
    return base


def _write_fixtures(*rows: dict[str, object]) -> None:
    settings = get_settings()
    ParquetTableRepository(FIXTURES, settings.parquet_dir).write(pd.DataFrame(list(rows)))


def _write_odds(*rows: dict[str, object]) -> None:
    settings = get_settings()
    ParquetTableRepository(ODDS_SNAPSHOTS, settings.parquet_dir).write(pd.DataFrame(list(rows)))


def _read_odds() -> pd.DataFrame:
    settings = get_settings()
    return ParquetTableRepository(ODDS_SNAPSHOTS, settings.parquet_dir).read()


def test_returns_zero_without_any_fixtures() -> None:
    assert mark_closing_lines(_COMPETITION) == 0


def test_returns_zero_when_no_fixture_has_kicked_off_yet() -> None:
    _write_fixtures(_fixture_row(kickoff_utc=_KICKOFF + timedelta(days=1)))
    assert mark_closing_lines(_COMPETITION, now=_KICKOFF) == 0


def test_returns_zero_without_any_odds() -> None:
    _write_fixtures(_fixture_row())
    assert mark_closing_lines(_COMPETITION, now=_KICKOFF + timedelta(hours=1)) == 0


def test_marks_the_latest_pre_kickoff_snapshot() -> None:
    _write_fixtures(_fixture_row())
    _write_odds(
        _odds_row(price=2.0, captured_at=_KICKOFF - timedelta(hours=2)),
        _odds_row(price=1.9, captured_at=_KICKOFF - timedelta(minutes=5)),
    )

    changed = mark_closing_lines(_COMPETITION, now=_KICKOFF + timedelta(hours=1))

    assert changed == 1
    odds = _read_odds().set_index("captured_at").sort_index()
    assert not odds.iloc[0]["is_closing"]
    assert odds.iloc[1]["is_closing"]


def test_excludes_snapshots_captured_after_kickoff() -> None:
    _write_fixtures(_fixture_row())
    _write_odds(
        _odds_row(price=1.9, captured_at=_KICKOFF - timedelta(minutes=5)),
        _odds_row(price=2.1, captured_at=_KICKOFF + timedelta(minutes=10)),
    )

    changed = mark_closing_lines(_COMPETITION, now=_KICKOFF + timedelta(hours=1))

    assert changed == 1
    odds = _read_odds().set_index("captured_at").sort_index()
    assert odds.iloc[0]["is_closing"]  # la de pre-kickoff
    assert not odds.iloc[1]["is_closing"]  # la de post-kickoff, nunca elegible


def test_marks_one_closing_row_per_bookmaker_market_selection_line() -> None:
    _write_fixtures(_fixture_row())
    _write_odds(
        _odds_row(
            bookmaker="pinnacle", selection="over", captured_at=_KICKOFF - timedelta(hours=1)
        ),
        _odds_row(
            bookmaker="pinnacle", selection="under", captured_at=_KICKOFF - timedelta(hours=1)
        ),
        _odds_row(bookmaker="bet365", selection="over", captured_at=_KICKOFF - timedelta(hours=1)),
    )

    changed = mark_closing_lines(_COMPETITION, now=_KICKOFF + timedelta(hours=1))

    assert changed == 3
    odds = _read_odds()
    assert odds["is_closing"].all()


def test_skips_a_fixture_already_marked() -> None:
    _write_fixtures(_fixture_row())
    _write_odds(
        _odds_row(price=1.9, captured_at=_KICKOFF - timedelta(hours=2), is_closing=False),
        _odds_row(price=1.85, captured_at=_KICKOFF - timedelta(minutes=5), is_closing=True),
    )

    changed = mark_closing_lines(_COMPETITION, now=_KICKOFF + timedelta(hours=1))

    assert changed == 0


def test_skips_a_fixture_whose_only_odds_are_post_kickoff() -> None:
    _write_fixtures(_fixture_row())
    _write_odds(_odds_row(captured_at=_KICKOFF + timedelta(minutes=10)))

    changed = mark_closing_lines(_COMPETITION, now=_KICKOFF + timedelta(hours=1))

    assert changed == 0


def test_skips_a_started_fixture_with_no_odds_at_all() -> None:
    _write_fixtures(
        _fixture_row(id="fx1"),
        _fixture_row(
            id="fx2",
            home_team_id="h2",
            away_team_id="a2",
            kickoff_utc=_KICKOFF + timedelta(hours=3),
        ),
    )
    _write_odds(_odds_row(fixture_id="fx1", captured_at=_KICKOFF - timedelta(minutes=5)))

    changed = mark_closing_lines(_COMPETITION, now=_KICKOFF + timedelta(days=1))

    assert changed == 1
