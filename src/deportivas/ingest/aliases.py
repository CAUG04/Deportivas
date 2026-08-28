"""Team name reconciliation: FBref, ESPN and football-data.co.uk each spell
the same club differently ("Manchester Utd", "Man United", "Man Utd").

The canonical id is a deterministic slug of the *first name a team is ever
seen under*, scoped by sport (so "Arsenal" in football and a hypothetical
"Arsenal" in another sport never collide). Every other spelling from every
other source becomes an alias pointing at that id. Resolution never guesses:
an unknown alias becomes a brand new canonical team rather than being
silently merged into an existing one — a wrong merge corrupts a team's whole
history, which is far worse than one duplicate team sitting unresolved until
a human maps it.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from deportivas.storage.protocols import TableRepository


def slugify(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    if not slug:
        raise ValueError(f"nombre de equipo no produce un slug valido: {name!r}")
    return slug


class TeamAliasResolver:
    """Resolves raw source names to canonical team ids, creating new teams
    and aliases as needed. One instance per ingestion run keeps an in-memory
    cache so a batch never re-reads the alias table per row."""

    def __init__(
        self, teams_repo: TableRepository, aliases_repo: TableRepository, *, sport: str
    ) -> None:
        self._teams_repo = teams_repo
        self._aliases_repo = aliases_repo
        self._sport = sport
        self._alias_cache: dict[tuple[str, str, str], str] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        existing = self._aliases_repo.read(filters={"sport": self._sport})
        for row in existing.itertuples(index=False):
            key = (str(row.source), self._sport, str(row.alias))
            self._alias_cache[key] = str(row.team_id)
        self._loaded = True

    def resolve(self, source: str, raw_name: str, *, now: datetime | None = None) -> str:
        """Returns the canonical team id for ``raw_name`` as reported by ``source``.

        Creates a new canonical team (and its first alias) the first time a
        name is seen; every subsequent call, from any source, for a name
        already mapped returns the same id without writing anything new.
        """
        self._ensure_loaded()
        now = now or datetime.now(UTC)
        cache_key = (source, self._sport, raw_name)
        if cache_key in self._alias_cache:
            return self._alias_cache[cache_key]

        team_id = self._new_team_id(raw_name)
        self._write_team(team_id, raw_name, now)
        self._write_alias(source, raw_name, team_id, now)
        self._alias_cache[cache_key] = team_id
        return team_id

    def _new_team_id(self, raw_name: str) -> str:
        return f"{self._sport}:{slugify(raw_name)}"

    def _write_team(self, team_id: str, raw_name: str, now: datetime) -> None:
        row = pd.DataFrame(
            [
                {
                    "id": team_id,
                    "canonical_name": raw_name,
                    "sport": self._sport,
                    "country": None,
                    "source": "alias_resolver",
                    "ingested_at": now,
                }
            ]
        )
        self._teams_repo.write(row)

    def _write_alias(self, source: str, raw_name: str, team_id: str, now: datetime) -> None:
        row = pd.DataFrame(
            [
                {
                    "source": source,
                    "sport": self._sport,
                    "alias": raw_name,
                    "team_id": team_id,
                    "ingested_at": now,
                }
            ]
        )
        self._aliases_repo.write(row)
