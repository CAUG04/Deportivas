"""TheOddsApiSource: mapping logic tested against small payloads shaped like
The Odds API's documented JSON response, and the real HTTP call tested with
httpx.MockTransport (no live network)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

from deportivas.contracts.tables import TEAM_ALIASES, TEAMS
from deportivas.ingest.aliases import TeamAliasResolver
from deportivas.ingest.ratelimit import RateLimiter
from deportivas.ingest.sources.theoddsapi import TheOddsApiSource, _parse_iso, _selection_name
from deportivas.storage.duckdb_repo.raw_store import ParquetRawDocumentRepository
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository

FOOTBALL_MARKET_MAP = {"h2h": "1x2", "spreads": "asian_handicap", "totals": "over_under"}


def _source(tmp_path: Path, *, transport: httpx.MockTransport | None = None) -> TheOddsApiSource:
    raw_repo = ParquetRawDocumentRepository(tmp_path / "raw", tmp_path / "parquet")
    teams_repo = ParquetTableRepository(TEAMS, tmp_path / "parquet")
    aliases_repo = ParquetTableRepository(TEAM_ALIASES, tmp_path / "parquet")
    resolver = TeamAliasResolver(teams_repo, aliases_repo, sport="football")
    client = httpx.Client(transport=transport) if transport else None
    return TheOddsApiSource(
        raw_repo=raw_repo,
        rate_limiter=RateLimiter(0.0),
        aliases=resolver,
        api_key="test-key",
        client=client,
    )


# -- _selection_name / _parse_iso ---------------------------------------------


def test_selection_name_matches_home_team() -> None:
    assert _selection_name("Arsenal", "Arsenal", "Chelsea") == "home"


def test_selection_name_matches_away_team() -> None:
    assert _selection_name("Chelsea", "Arsenal", "Chelsea") == "away"


def test_selection_name_draw() -> None:
    assert _selection_name("Draw", "Arsenal", "Chelsea") == "draw"


def test_selection_name_over_under() -> None:
    assert _selection_name("Over", "Arsenal", "Chelsea") == "over"
    assert _selection_name("Under", "Arsenal", "Chelsea") == "under"


def test_selection_name_unrecognised_is_none() -> None:
    assert _selection_name("Some Other Team", "Arsenal", "Chelsea") is None


def test_selection_name_non_string_is_none() -> None:
    assert _selection_name(None, "Arsenal", "Chelsea") is None


def test_parse_iso_with_z_suffix() -> None:
    assert _parse_iso("2026-01-10T17:30:00Z") == datetime(2026, 1, 10, 17, 30, tzinfo=UTC)


def test_parse_iso_malformed_is_none() -> None:
    assert _parse_iso("not a date") is None


def test_parse_iso_none_is_none() -> None:
    assert _parse_iso(None) is None


# -- _to_odds -------------------------------------------------------------------


def _event(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "abc123",
        "sport_key": "soccer_epl",
        "sport_title": "EPL",
        "commence_time": "2026-01-10T17:30:00Z",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "last_update": "2026-01-10T12:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": "2026-01-10T12:00:00Z",
                        "outcomes": [
                            {"name": "Arsenal", "price": 1.85},
                            {"name": "Draw", "price": 3.6},
                            {"name": "Chelsea", "price": 4.2},
                        ],
                    },
                    {
                        "key": "totals",
                        "last_update": "2026-01-10T12:00:00Z",
                        "outcomes": [
                            {"name": "Over", "price": 1.9, "point": 2.5},
                            {"name": "Under", "price": 1.9, "point": 2.5},
                        ],
                    },
                ],
            }
        ],
    }
    base.update(overrides)
    return base


def test_to_odds_maps_h2h_market(tmp_path: Path) -> None:
    source = _source(tmp_path)
    odds = source._to_odds(
        [_event()],
        competition_id="eng-premier-league",
        season="2526",
        market_map=FOOTBALL_MARKET_MAP,
    )

    h2h = odds[odds["market"] == "1x2"]
    assert set(h2h["selection"]) == {"home", "draw", "away"}
    home_row = h2h[h2h["selection"] == "home"].iloc[0]
    assert home_row["price"] == pytest.approx(1.85)
    assert pd.isna(home_row["line"])
    assert bool(home_row["is_closing"]) is False


def test_to_odds_maps_totals_market_with_line(tmp_path: Path) -> None:
    source = _source(tmp_path)
    odds = source._to_odds(
        [_event()],
        competition_id="eng-premier-league",
        season="2526",
        market_map=FOOTBALL_MARKET_MAP,
    )

    over = odds[(odds["market"] == "over_under") & (odds["selection"] == "over")].iloc[0]
    assert over["line"] == pytest.approx(2.5)
    assert over["price"] == pytest.approx(1.9)


def test_to_odds_captured_at_uses_bookmaker_last_update(tmp_path: Path) -> None:
    source = _source(tmp_path)
    odds = source._to_odds(
        [_event()],
        competition_id="eng-premier-league",
        season="2526",
        market_map=FOOTBALL_MARKET_MAP,
    )
    assert (odds["captured_at"] == datetime(2026, 1, 10, 12, 0, tzinfo=UTC)).all()


def test_to_odds_unmapped_market_key_is_skipped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    event = _event()
    event["bookmakers"][0]["markets"].append(  # type: ignore[index]
        {"key": "player_props", "outcomes": [{"name": "Someone", "price": 2.0}]}
    )
    odds = source._to_odds(
        [event],
        competition_id="eng-premier-league",
        season="2526",
        market_map=FOOTBALL_MARKET_MAP,
    )
    assert "player_props" not in set(odds["market"])


def test_to_odds_outcome_with_unrecognised_name_is_skipped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    event = _event()
    event["bookmakers"][0]["markets"][0]["outcomes"].append(  # type: ignore[index]
        {"name": "Some Third Team", "price": 9.0}
    )
    odds = source._to_odds(
        [event],
        competition_id="eng-premier-league",
        season="2526",
        market_map=FOOTBALL_MARKET_MAP,
    )
    assert 9.0 not in set(odds["price"])


def test_to_odds_event_without_commence_time_is_skipped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    odds = source._to_odds(
        [_event(commence_time=None)],
        competition_id="eng-premier-league",
        season="2526",
        market_map=FOOTBALL_MARKET_MAP,
    )
    assert odds.empty


def test_to_odds_all_ids_unique_and_within_length_limit(tmp_path: Path) -> None:
    source = _source(tmp_path)
    odds = source._to_odds(
        [_event()],
        competition_id="eng-premier-league",
        season="2526",
        market_map=FOOTBALL_MARKET_MAP,
    )
    assert odds["id"].is_unique
    assert (odds["id"].str.len() <= 64).all()


def test_to_odds_fixture_id_matches_fbref_style_hash(tmp_path: Path) -> None:
    from deportivas.domain.ids import fixture_id

    source = _source(tmp_path)
    odds = source._to_odds(
        [_event()],
        competition_id="eng-premier-league",
        season="2526",
        market_map=FOOTBALL_MARKET_MAP,
    )
    expected = fixture_id(
        "eng-premier-league",
        "2526",
        "football:arsenal",
        "football:chelsea",
        datetime(2026, 1, 10, 17, 30, tzinfo=UTC),
    )
    assert set(odds["fixture_id"]) == {expected}


# -- fetch_odds (real HTTP call over httpx.MockTransport) ---------------------


def test_fetch_odds_waits_calls_api_archives_strips_key_and_maps(tmp_path: Path) -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, json=[_event()])

    source = _source(tmp_path, transport=httpx.MockTransport(handler))
    waits: list[float] = []

    def fake_wait() -> float:
        waits.append(1.0)
        return 0.0

    source._wait = fake_wait  # type: ignore[method-assign]

    odds = source.fetch_odds(
        competition_id="eng-premier-league",
        sport_key="soccer_epl",
        season="2526",
        market_map=FOOTBALL_MARKET_MAP,
    )

    assert len(waits) == 1
    assert not odds.empty
    assert len(captured_requests) == 1
    assert "apiKey=test-key" in str(captured_requests[0].url)

    raw_repo = ParquetRawDocumentRepository(tmp_path / "raw", tmp_path / "parquet")
    stored = raw_repo.find(source="theoddsapi")
    assert len(stored) == 1
    assert "apiKey" not in stored.iloc[0]["params"]


def test_fetch_odds_raises_on_http_error_status(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "invalid key"})

    source = _source(tmp_path, transport=httpx.MockTransport(handler))
    source._wait = lambda: 0.0  # type: ignore[method-assign]

    with pytest.raises(httpx.HTTPStatusError):
        source.fetch_odds(
            competition_id="eng-premier-league",
            sport_key="soccer_epl",
            season="2526",
            market_map=FOOTBALL_MARKET_MAP,
        )


def test_fetch_odds_retries_transport_errors_then_raises(tmp_path: Path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("boom", request=request)

    source = _source(tmp_path, transport=httpx.MockTransport(handler))
    source._wait = lambda: 0.0  # type: ignore[method-assign]

    with pytest.raises(httpx.ConnectError):
        source.fetch_odds(
            competition_id="eng-premier-league",
            sport_key="soccer_epl",
            season="2526",
            market_map=FOOTBALL_MARKET_MAP,
        )
    assert attempts == 3
