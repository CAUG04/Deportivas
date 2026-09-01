from __future__ import annotations

from datetime import UTC, datetime

import pytest

from deportivas.domain.ids import fixture_id


def test_same_inputs_produce_same_id() -> None:
    kickoff = datetime(2026, 1, 10, 20, 0, tzinfo=UTC)
    a = fixture_id("eng-premier-league", "2526", "t1", "t2", kickoff)
    b = fixture_id("eng-premier-league", "2526", "t1", "t2", kickoff)
    assert a == b


def test_different_teams_produce_different_id() -> None:
    kickoff = datetime(2026, 1, 10, 20, 0, tzinfo=UTC)
    a = fixture_id("eng-premier-league", "2526", "t1", "t2", kickoff)
    b = fixture_id("eng-premier-league", "2526", "t3", "t2", kickoff)
    assert a != b


def test_naive_kickoff_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        fixture_id("eng-premier-league", "2526", "t1", "t2", datetime(2026, 1, 10, 20, 0))  # noqa: DTZ001


def test_id_is_a_hex_digest() -> None:
    kickoff = datetime(2026, 1, 10, 20, 0, tzinfo=UTC)
    result = fixture_id("eng-premier-league", "2526", "t1", "t2", kickoff)
    assert len(result) == 40
    int(result, 16)  # no lanza si es hexadecimal valido
