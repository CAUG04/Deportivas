"""Fractional Kelly staking — ``config/thresholds.yaml``'s ``staking``
section. Project rule #6 (kelly_fraction): "Solo Kelly fraccionado. Nada de
progresiones." — full Kelly is never used directly, only a fixed fraction
of it, and the result is hard-clipped to ``max_stake_per_bet``.

Kelly's formula: ``f* = (p*(b+1) - 1) / b``, where ``b`` is the net
fractional payout (decimal price minus 1) and ``p`` is the model's true
(calibrated) probability of winning. A non-positive edge (``f* <= 0``)
stakes nothing rather than a negative fraction.
"""

from __future__ import annotations

from deportivas.config.catalog import StakingConfig


def kelly_stake_fraction(
    *, prob_model: float, decimal_price: float, config: StakingConfig
) -> float:
    """Returns 0.0 for a bet Kelly wouldn't take, and for one so small
    ``min_stake_per_bet`` says it isn't worth issuing at all — never a
    fraction rounded up into a bet that wasn't supposed to happen."""
    b = decimal_price - 1.0
    if b <= 0.0:
        return 0.0
    full_kelly = (prob_model * (b + 1.0) - 1.0) / b
    if full_kelly <= 0.0:
        return 0.0
    stake = min(full_kelly * config.kelly_fraction, config.max_stake_per_bet)
    if stake < config.min_stake_per_bet:
        return 0.0
    return stake
