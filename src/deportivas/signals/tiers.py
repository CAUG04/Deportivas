"""Confidence-tier classification — ``config/thresholds.yaml``'s ``tiers``
section, applied top-down (alta, then media, then baja). Nothing that
clears "baja" becomes ``descartar``, a first-class outcome the project
displays deliberately: telling the user what *not* to bet is half the
product, not a failure to classify.
"""

from __future__ import annotations

from dataclasses import dataclass

from deportivas.config.catalog import TierConfig, TiersConfig
from deportivas.domain.enums import Tier


@dataclass(frozen=True, slots=True)
class TierInputs:
    edge: float
    sample_matches: int
    is_calibrated: bool
    has_sharp_price: bool
    has_favourable_line_move: bool


@dataclass(frozen=True, slots=True)
class TierResult:
    tier: Tier
    reasons: dict[str, bool]


def _condition_results(config: TierConfig, inputs: TierInputs) -> dict[str, bool]:
    return {
        "min_edge": inputs.edge >= config.min_edge,
        "min_sample_matches": inputs.sample_matches >= config.min_sample_matches,
        "requires_calibration": (not config.requires_calibration) or inputs.is_calibrated,
        "requires_sharp_price": (not config.requires_sharp_price) or inputs.has_sharp_price,
        "requires_favourable_line_move": (not config.requires_favourable_line_move)
        or inputs.has_favourable_line_move,
    }


def classify_tier(inputs: TierInputs, tiers: TiersConfig) -> TierResult:
    """Returns the first tier (alta, then media, then baja) whose every
    condition holds, together with a per-condition breakdown for that tier
    — ``signals.tier_reasons``'s "que condicion de tier fallo o se
    cumplio". A signal meeting none of the three becomes ``descartar``,
    reported against ``baja``'s own conditions (the least strict, so it
    names exactly what was still missing to clear even that bar)."""
    for tier, config in (
        (Tier.ALTA, tiers.alta),
        (Tier.MEDIA, tiers.media),
        (Tier.BAJA, tiers.baja),
    ):
        reasons = _condition_results(config, inputs)
        if all(reasons.values()):
            return TierResult(tier=tier, reasons=reasons)
    return TierResult(tier=Tier.DESCARTAR, reasons=_condition_results(tiers.baja, inputs))
