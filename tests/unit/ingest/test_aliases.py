"""TeamAliasResolver reconciles differently-spelled team names into one
canonical id per real-world team, per sport."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from deportivas.contracts.tables import TEAM_ALIASES, TEAMS
from deportivas.ingest.aliases import TeamAliasResolver, slugify
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository


def _resolver(tmp_path: Path, *, sport: str = "football") -> TeamAliasResolver:
    teams_repo = ParquetTableRepository(TEAMS, tmp_path)
    aliases_repo = ParquetTableRepository(TEAM_ALIASES, tmp_path)
    return TeamAliasResolver(teams_repo, aliases_repo, sport=sport)


def test_slugify_basic() -> None:
    assert slugify("Manchester United") == "manchester-united"


def test_slugify_strips_accents_and_punctuation() -> None:
    assert slugify("Deportivo Cali!") == "deportivo-cali"


def test_slugify_empty_result_raises() -> None:
    with pytest.raises(ValueError, match="slug"):
        slugify("!!!")


def test_first_sighting_creates_a_canonical_team(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    team_id = resolver.resolve("fbref", "Manchester United")
    assert team_id == "football:manchester-united"


def test_same_name_same_source_resolves_to_same_id_without_rewriting(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    first = resolver.resolve("fbref", "Manchester United")
    second = resolver.resolve("fbref", "Manchester United")
    assert first == second


def test_different_spelling_different_source_maps_to_new_team(tmp_path: Path) -> None:
    """Without a human-curated alias, a different spelling from a different
    source is NOT guessed to be the same team — that would risk a silent,
    hard-to-detect merge of two different clubs' histories."""
    resolver = _resolver(tmp_path)
    fbref_id = resolver.resolve("fbref", "Manchester Utd")
    espn_id = resolver.resolve("espn", "Man United")
    assert fbref_id != espn_id


def test_curated_alias_maps_both_spellings_to_same_team(tmp_path: Path) -> None:
    """The intended workflow: seed team_aliases (by hand, or by a future
    curation step) so both spellings point at one canonical id."""
    aliases_repo = ParquetTableRepository(TEAM_ALIASES, tmp_path)
    teams_repo = ParquetTableRepository(TEAMS, tmp_path)

    now = datetime(2026, 1, 1, tzinfo=UTC)
    teams_repo.write(
        pd.DataFrame(
            [
                {
                    "id": "football:manchester-united",
                    "canonical_name": "Manchester United",
                    "sport": "football",
                    "country": "England",
                    "source": "curated",
                    "ingested_at": now,
                }
            ]
        )
    )
    aliases_repo.write(
        pd.DataFrame(
            [
                {
                    "source": "espn",
                    "sport": "football",
                    "alias": "Man United",
                    "team_id": "football:manchester-united",
                    "ingested_at": now,
                }
            ]
        )
    )
    resolver = TeamAliasResolver(teams_repo, aliases_repo, sport="football")
    assert resolver.resolve("espn", "Man United") == "football:manchester-united"


def test_same_name_different_sport_gets_different_id(tmp_path: Path) -> None:
    football_resolver = _resolver(tmp_path, sport="football")
    basketball_resolver = _resolver(tmp_path, sport="basketball")
    football_id = football_resolver.resolve("fbref", "Arsenal")
    basketball_id = basketball_resolver.resolve("espn", "Arsenal")
    assert football_id != basketball_id


def test_resolving_writes_team_and_alias_rows(tmp_path: Path) -> None:
    teams_repo = ParquetTableRepository(TEAMS, tmp_path)
    aliases_repo = ParquetTableRepository(TEAM_ALIASES, tmp_path)
    resolver = TeamAliasResolver(teams_repo, aliases_repo, sport="football")

    team_id = resolver.resolve("fbref", "Arsenal")

    teams = teams_repo.read()
    aliases = aliases_repo.read()
    assert list(teams["id"]) == [team_id]
    assert list(aliases["team_id"]) == [team_id]
    assert aliases.iloc[0]["alias"] == "Arsenal"
    assert aliases.iloc[0]["source"] == "fbref"


def test_second_resolver_instance_sees_previously_written_aliases(tmp_path: Path) -> None:
    """A fresh run (new process, new resolver instance) must not re-create a
    team that a previous run already resolved: idempotency across runs."""
    first_run = _resolver(tmp_path)
    team_id = first_run.resolve("fbref", "Arsenal")

    second_run = _resolver(tmp_path)
    assert second_run.resolve("fbref", "Arsenal") == team_id

    teams_repo = ParquetTableRepository(TEAMS, tmp_path)
    assert len(teams_repo.read()) == 1
