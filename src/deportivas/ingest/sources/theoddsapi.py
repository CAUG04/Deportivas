"""The Odds API (https://the-odds-api.com) via a direct HTTP call
(-> ``odds_snapshots``).

The only source in this project with real timestamped odds captures — no
approximation, unlike ``footballdata.py``'s pre-1990s-style CSV odds — and
the only source of genuine Liga BetPlay (Colombia) odds at all (rule of
Fase 1: no open historical Colombian odds source exists; this job starts
capturing from whenever it first runs, documented in the README).

Every event's ``h2h``/``spreads``/``totals`` markets from the API map onto
this project's own market ids via a caller-supplied ``market_map`` — e.g.
``{"h2h": "1x2", "spreads": "asian_handicap", "totals": "over_under"}`` for
football, ``{"h2h": "moneyline", "spreads": "spread", "totals": "total"}``
for the American sports — because the same raw market key means a different
thing (and a different set of selections) depending on the sport, and that
decision belongs to the caller (which knows the sport), not this adapter.

Every snapshot from this adapter is written with ``is_closing=False``: this
call cannot know, at fetch time, whether it is the last capture before
kickoff. Marking the true closing line is Fase 8's settlement job, run after
a fixture starts, comparing each fixture's snapshots by ``captured_at``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import httpx
import pandas as pd
import tenacity

from deportivas.domain.ids import deterministic_id, fixture_id
from deportivas.ingest.base import DataSource

if TYPE_CHECKING:
    from deportivas.ingest.aliases import TeamAliasResolver
    from deportivas.ingest.ratelimit import RateLimiter
    from deportivas.storage.protocols import RawDocumentRepository

_BASE_URL = "https://api.the-odds-api.com/v4"


class TheOddsApiSource(DataSource):
    name = "theoddsapi"

    def __init__(
        self,
        *,
        raw_repo: RawDocumentRepository,
        rate_limiter: RateLimiter,
        aliases: TeamAliasResolver,
        api_key: str,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(raw_repo=raw_repo, rate_limiter=rate_limiter)
        self._aliases = aliases
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=30.0)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        retry=tenacity.retry_if_exception_type(httpx.TransportError),
        reraise=True,
    )
    def fetch_odds(
        self,
        *,
        competition_id: str,
        sport_key: str,
        season: str,
        market_map: dict[str, str],
        regions: str = "uk,eu,us",
    ) -> pd.DataFrame:
        """Returns rows shaped for the ``odds_snapshots`` table."""
        self._wait()
        params = {
            "apiKey": self._api_key,
            "regions": regions,
            "markets": ",".join(market_map),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        response = self._client.get(f"{_BASE_URL}/sports/{sport_key}/odds", params=params)
        self._archive_bytes(
            endpoint=f"/sports/{sport_key}/odds",
            params={k: v for k, v in params.items() if k != "apiKey"},
            content=response.content,
            content_type="application/json",
            status_code=response.status_code,
        )
        response.raise_for_status()
        events = response.json()
        return self._to_odds(
            events,
            competition_id=competition_id,
            season=season,
            market_map=market_map,
        )

    def _to_odds(
        self,
        events: list[dict[str, object]],
        *,
        competition_id: str,
        season: str,
        market_map: dict[str, str],
    ) -> pd.DataFrame:
        now = datetime.now(UTC)
        rows: list[dict[str, object]] = []
        for event in events:
            kickoff = _parse_iso(event.get("commence_time"))
            home_name = event.get("home_team")
            away_name = event.get("away_team")
            if kickoff is None or not isinstance(home_name, str) or not isinstance(away_name, str):
                continue
            home_team_id = self._aliases.resolve(self.name, home_name)
            away_team_id = self._aliases.resolve(self.name, away_name)
            fid = fixture_id(competition_id, season, home_team_id, away_team_id, kickoff)

            bookmakers = cast("list[dict[str, object]]", event.get("bookmakers", []))
            for bookmaker in bookmakers:
                rows.extend(
                    self._bookmaker_rows(
                        bookmaker,
                        fid=fid,
                        competition_id=competition_id,
                        season=season,
                        home_name=home_name,
                        away_name=away_name,
                        market_map=market_map,
                        now=now,
                    )
                )
        return pd.DataFrame(rows)

    def _bookmaker_rows(
        self,
        bookmaker: dict[str, object],
        *,
        fid: str,
        competition_id: str,
        season: str,
        home_name: str,
        away_name: str,
        market_map: dict[str, str],
        now: datetime,
    ) -> list[dict[str, object]]:
        bookmaker_key = str(bookmaker.get("key"))
        captured_at = _parse_iso(bookmaker.get("last_update")) or now
        rows: list[dict[str, object]] = []

        markets = cast("list[dict[str, object]]", bookmaker.get("markets", []))
        for market in markets:
            raw_key = str(market.get("key"))
            our_market = market_map.get(raw_key)
            if our_market is None:
                continue
            outcomes = cast("list[dict[str, object]]", market.get("outcomes", []))
            for outcome in outcomes:
                selection = _selection_name(outcome.get("name"), home_name, away_name)
                price = outcome.get("price")
                if selection is None or not isinstance(price, int | float):
                    continue
                line = outcome.get("point")
                rows.append(
                    {
                        "id": deterministic_id(
                            fid, bookmaker_key, our_market, selection, captured_at.isoformat()
                        ),
                        "fixture_id": fid,
                        "competition_id": competition_id,
                        "season": season,
                        "bookmaker": bookmaker_key,
                        "market": our_market,
                        "selection": selection,
                        "line": float(line) if isinstance(line, int | float) else None,
                        "price": float(price),
                        "captured_at": captured_at,
                        "is_closing": False,  # ver docstring del modulo
                        "source": self.name,
                        "ingested_at": now,
                    }
                )
        return rows


def _selection_name(raw_name: object, home_name: str, away_name: str) -> str | None:
    if not isinstance(raw_name, str):
        return None
    if raw_name == home_name:
        return "home"
    if raw_name == away_name:
        return "away"
    lowered = raw_name.strip().lower()
    if lowered in ("draw", "tie"):
        return "draw"
    if lowered in ("over", "under"):
        return lowered
    return None


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
