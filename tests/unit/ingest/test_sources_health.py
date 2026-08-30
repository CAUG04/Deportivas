"""sources_health.py: valida en vivo (aqui, contra dobles) los
identificadores de fuente de competitions.yaml y las claves de The Odds
API, sin persistir nada -- ver el docstring del modulo."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pandas as pd
import pytest

from deportivas.config.catalog import Competition, CompetitionOdds, CompetitionSources
from deportivas.config.settings import get_settings
from deportivas.domain.enums import RefreshCadence, Sport
from deportivas.domain.seasons import season_labels
from deportivas.ingest.sources_health import (
    HealthIssue,
    check_football_sources,
    check_odds_api_sport_keys,
    run_health_check,
)


@pytest.fixture(autouse=True)
def _settings_in_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DEPORTIVAS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SOCCERDATA_DIR", str(tmp_path / "soccerdata"))
    monkeypatch.delenv("DEPORTIVAS_THE_ODDS_API_KEY", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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
        "sources": CompetitionSources(fbref="Premier League"),
        "odds": CompetitionOdds(the_odds_api="soccer_epl"),
    }
    base.update(overrides)
    return Competition.model_validate(base)


def test_no_football_competitions_returns_no_issues_and_touches_nothing() -> None:
    nfl = _competition(
        id="usa-nfl",
        sport=Sport.AMERICAN_FOOTBALL,
        sources=CompetitionSources(),
        odds=CompetitionOdds(the_odds_api="americanfootball_nfl"),
    )
    assert check_football_sources([nfl]) == []


def test_football_competition_without_any_configured_source_is_a_no_op() -> None:
    # No pasa hoy en config/competitions.yaml (toda liga declara fbref), pero
    # CompetitionSources lo permite: no debe intentar ni fallar nada.
    bare = _competition(sources=CompetitionSources())
    assert check_football_sources([bare]) == []


def test_fbref_only_success_reports_no_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeFBrefSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_schedule(self, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr("deportivas.ingest.sources.fbref.FBrefSource", _FakeFBrefSource)

    competition = _competition(sources=CompetitionSources(fbref="Premier League"))
    assert check_football_sources([competition]) == []


def test_fbref_failure_is_reported_by_competition_and_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingFBrefSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_schedule(self, **kwargs: object) -> pd.DataFrame:
            raise ValueError("Invalid league 'NED-Eredivisie'")

    monkeypatch.setattr("deportivas.ingest.sources.fbref.FBrefSource", _FailingFBrefSource)

    competition = _competition(
        id="ned-eredivisie",
        sources=CompetitionSources(fbref="Eredivisie", soccerdata_key="NED-Eredivisie"),
    )
    issues = check_football_sources([competition])

    assert issues == [
        HealthIssue(
            "ned-eredivisie", "sources.fbref", "ValueError: Invalid league 'NED-Eredivisie'"
        )
    ]
    assert (
        str(issues[0])
        == "ned-eredivisie.sources.fbref: ValueError: Invalid league 'NED-Eredivisie'"
    )


def test_empty_dataframe_is_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _OffSeasonFBrefSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_schedule(self, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame()  # liga real, simplemente sin partidos todavia

    monkeypatch.setattr("deportivas.ingest.sources.fbref.FBrefSource", _OffSeasonFBrefSource)

    assert check_football_sources([_competition()]) == []


def test_understat_checked_only_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _FakeFBrefSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_schedule(self, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame()

    class _FakeUnderstatSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_team_match_stats(self, **kwargs: object) -> pd.DataFrame:
            calls["understat"] = kwargs
            return pd.DataFrame()

    monkeypatch.setattr("deportivas.ingest.sources.fbref.FBrefSource", _FakeFBrefSource)
    monkeypatch.setattr("deportivas.ingest.sources.understat.UnderstatSource", _FakeUnderstatSource)

    with_understat = _competition(
        sources=CompetitionSources(fbref="Premier League", understat="EPL")
    )
    expected_season = season_labels(with_understat, count=1)[0]
    check_football_sources([with_understat])
    assert calls["understat"] == {
        "competition_id": "eng-premier-league",
        "understat_league": "EPL",
        "seasons": [expected_season],
    }

    calls.clear()
    without_understat = _competition(sources=CompetitionSources(fbref="Premier League"))
    check_football_sources([without_understat])
    assert "understat" not in calls


def test_match_history_checked_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _FakeFBrefSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_schedule(self, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame()

    class _FakeFootballDataSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_games(self, **kwargs: object) -> pd.DataFrame:
            calls["footballdata"] = kwargs
            return pd.DataFrame()

    class _FakeEspnSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_schedule(self, **kwargs: object) -> pd.DataFrame:
            calls["espn"] = kwargs
            return pd.DataFrame()

    monkeypatch.setattr("deportivas.ingest.sources.fbref.FBrefSource", _FakeFBrefSource)
    monkeypatch.setattr(
        "deportivas.ingest.sources.footballdata.FootballDataSource", _FakeFootballDataSource
    )
    monkeypatch.setattr("deportivas.ingest.sources.espn.EspnSource", _FakeEspnSource)

    with_match_history = _competition(
        sources=CompetitionSources(fbref="Premier League", match_history="E0", espn="eng.1")
    )
    check_football_sources([with_match_history])
    assert "footballdata" in calls
    assert "espn" not in calls


def test_espn_is_the_fallback_when_no_match_history(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _FakeFBrefSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_schedule(self, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame()

    class _FakeEspnSource:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fetch_schedule(self, **kwargs: object) -> pd.DataFrame:
            calls["espn"] = kwargs
            return pd.DataFrame()

    monkeypatch.setattr("deportivas.ingest.sources.fbref.FBrefSource", _FakeFBrefSource)
    monkeypatch.setattr("deportivas.ingest.sources.espn.EspnSource", _FakeEspnSource)

    colombia = _competition(
        id="col-primera-a", sources=CompetitionSources(fbref="Primera A", espn="col.1")
    )
    expected_season = season_labels(colombia, count=1)[0]
    check_football_sources([colombia])
    assert calls["espn"] == {
        "competition_id": "col-primera-a",
        "espn_league": "col.1",
        "seasons": [expected_season],
    }


# -- The Odds API sport keys ---------------------------------------------------


def test_no_api_key_skips_silently_and_makes_no_request() -> None:
    def _fail_if_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no deberia llamar a la red sin api key")

    with httpx.Client(transport=httpx.MockTransport(_fail_if_called)) as client:
        assert check_odds_api_sport_keys([_competition()], client=client) == []


def test_all_sport_keys_known_reports_no_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPORTIVAS_THE_ODDS_API_KEY", "test-key")
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"key": "soccer_epl"}, {"key": "basketball_nba"}])

    competition = _competition(odds=CompetitionOdds(the_odds_api="soccer_epl"))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert check_odds_api_sport_keys([competition], client=client) == []


def test_unknown_sport_key_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPORTIVAS_THE_ODDS_API_KEY", "test-key")
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"key": "soccer_epl"}])

    competition = _competition(
        id="col-primera-a", odds=CompetitionOdds(the_odds_api="soccer_colombia_primera_a")
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        issues = check_odds_api_sport_keys([competition], client=client)

    assert issues == [
        HealthIssue(
            "col-primera-a",
            "odds.the_odds_api",
            "'soccer_colombia_primera_a' no existe en /v4/sports",
        )
    ]


def test_non_200_response_is_reported_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPORTIVAS_THE_ODDS_API_KEY", "bad-key")
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "invalid key"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        issues = check_odds_api_sport_keys([_competition()], client=client)

    assert issues == [HealthIssue("*", "odds.the_odds_api", "/v4/sports respondio 401")]


def test_owns_and_closes_its_own_client_when_none_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPORTIVAS_THE_ODDS_API_KEY", "test-key")
    get_settings.cache_clear()

    class _FakeResponse:
        status_code = 200

        def json(self) -> list[dict[str, str]]:
            return [{"key": "soccer_epl"}]

    class _FakeClient:
        closed = False

        def get(self, *args: object, **kwargs: object) -> _FakeResponse:
            return _FakeResponse()

        def close(self) -> None:
            _FakeClient.closed = True

    monkeypatch.setattr("httpx.Client", lambda **kwargs: _FakeClient())

    check_odds_api_sport_keys([_competition(odds=CompetitionOdds(the_odds_api="soccer_epl"))])

    assert _FakeClient.closed is True


# -- run_health_check ----------------------------------------------------------


def test_run_health_check_combines_both_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    football_issue = HealthIssue("x", "sources.fbref", "boom")
    odds_issue = HealthIssue("y", "odds.the_odds_api", "boom")

    monkeypatch.setattr(
        "deportivas.ingest.sources_health.check_football_sources",
        lambda competitions: [football_issue],
    )
    monkeypatch.setattr(
        "deportivas.ingest.sources_health.check_odds_api_sport_keys",
        lambda competitions: [odds_issue],
    )

    assert run_health_check() == [football_issue, odds_issue]
