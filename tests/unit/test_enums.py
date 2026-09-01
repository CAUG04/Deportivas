"""Behaviour of the domain enums' own properties (Fixture.is_settled, Tier.is_actionable)."""

from __future__ import annotations

from deportivas.domain.enums import FixtureStatus, Tier


def test_fixture_status_is_settled_only_when_finished() -> None:
    assert FixtureStatus.FINISHED.is_settled is True
    for status in (
        FixtureStatus.SCHEDULED,
        FixtureStatus.LIVE,
        FixtureStatus.POSTPONED,
        FixtureStatus.CANCELLED,
    ):
        assert status.is_settled is False


def test_tier_is_actionable_only_for_alta_and_media() -> None:
    assert Tier.ALTA.is_actionable is True
    assert Tier.MEDIA.is_actionable is True
    assert Tier.BAJA.is_actionable is False
    assert Tier.DESCARTAR.is_actionable is False
