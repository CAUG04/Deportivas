"""Closed vocabularies shared by every layer.

These are the values that end up in string columns. Keeping them as enums means
a typo in a market name fails at import time instead of producing an empty
backtest slice that looks like "no signals found".
"""

from __future__ import annotations

from enum import StrEnum


class Sport(StrEnum):
    FOOTBALL = "football"
    AMERICAN_FOOTBALL = "american_football"
    BASKETBALL = "basketball"
    BASEBALL = "baseball"
    ICE_HOCKEY = "ice_hockey"


class RefreshCadence(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"


class FixtureStatus(StrEnum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"

    @property
    def is_settled(self) -> bool:
        return self is FixtureStatus.FINISHED


class Market(StrEnum):
    ONE_X_TWO = "1x2"
    OVER_UNDER = "over_under"
    BTTS = "btts"
    ASIAN_HANDICAP = "asian_handicap"
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"


class Selection(StrEnum):
    HOME = "home"
    DRAW = "draw"
    AWAY = "away"
    OVER = "over"
    UNDER = "under"
    YES = "yes"
    NO = "no"


class Tier(StrEnum):
    """Confidence tier of a signal.

    ``DESCARTAR`` is a first-class outcome, not an error: telling the user what
    *not* to bet is half the product.
    """

    ALTA = "alta"
    MEDIA = "media"
    BAJA = "baja"
    DESCARTAR = "descartar"

    @property
    def is_actionable(self) -> bool:
        return self in {Tier.ALTA, Tier.MEDIA}


class DevigMethod(StrEnum):
    """How the bookmaker margin is removed to obtain a fair probability."""

    MULTIPLICATIVE = "multiplicative"
    POWER = "power"
    SHIN = "shin"


class CalibrationMethod(StrEnum):
    ISOTONIC = "isotonic"
    PLATT = "platt"


class BetOutcome(StrEnum):
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    HALF_WIN = "half_win"
    HALF_LOSS = "half_loss"
    VOID = "void"


class DerivedFrom(StrEnum):
    """Where a market's probability comes from."""

    SCORE_MATRIX = "score_matrix"
    CLASSIFIER = "classifier"
    MARGIN_REGRESSION = "margin_regression"
