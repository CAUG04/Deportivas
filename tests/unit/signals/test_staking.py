"""staking.py: fractional Kelly, clipped to config/thresholds.yaml's
staking bounds."""

from __future__ import annotations

import pytest

from deportivas.config.catalog import StakingConfig, load_thresholds
from deportivas.signals.staking import kelly_stake_fraction

_CONFIG = StakingConfig(
    kelly_fraction=0.25, max_stake_per_bet=0.02, min_stake_per_bet=0.001, bankroll_unit=1.0
)


def test_matches_the_fractional_kelly_formula_when_uncapped() -> None:
    # full Kelly = (0.52*2 - 1)/1 = 0.04; *0.25 = 0.01 (por debajo del tope 0.02)
    stake = kelly_stake_fraction(prob_model=0.52, decimal_price=2.0, config=_CONFIG)
    assert stake == pytest.approx(0.01)


def test_no_edge_stakes_nothing() -> None:
    # prob "justa" exacta para esta cuota (1/2.0 = 0.5): Kelly da 0
    stake = kelly_stake_fraction(prob_model=0.5, decimal_price=2.0, config=_CONFIG)
    assert stake == 0.0


def test_negative_edge_stakes_nothing() -> None:
    stake = kelly_stake_fraction(prob_model=0.3, decimal_price=2.0, config=_CONFIG)
    assert stake == 0.0


def test_large_edge_is_capped_at_max_stake_per_bet() -> None:
    # full Kelly = (0.9*2 - 1)/1 = 0.8; *0.25 = 0.2, muy por encima del tope 0.02
    stake = kelly_stake_fraction(prob_model=0.9, decimal_price=2.0, config=_CONFIG)
    assert stake == pytest.approx(0.02)


def test_tiny_edge_below_minimum_stakes_nothing_rather_than_rounding_up() -> None:
    config = StakingConfig(
        kelly_fraction=0.25, max_stake_per_bet=0.5, min_stake_per_bet=0.05, bankroll_unit=1.0
    )
    # full Kelly = (0.51*2 - 1)/1 = 0.02; *0.25 = 0.005, por debajo del minimo 0.05
    stake = kelly_stake_fraction(prob_model=0.51, decimal_price=2.0, config=config)
    assert stake == 0.0


def test_price_at_or_below_one_stakes_nothing() -> None:
    stake = kelly_stake_fraction(prob_model=0.9, decimal_price=1.0, config=_CONFIG)
    assert stake == 0.0


def test_higher_probability_means_a_larger_stake() -> None:
    low = kelly_stake_fraction(prob_model=0.51, decimal_price=2.0, config=_CONFIG)
    high = kelly_stake_fraction(prob_model=0.53, decimal_price=2.0, config=_CONFIG)
    assert high > low


def test_works_against_the_real_thresholds_yaml_config() -> None:
    config = load_thresholds().staking
    stake = kelly_stake_fraction(prob_model=0.9, decimal_price=2.5, config=config)
    assert 0.0 < stake <= config.max_stake_per_bet
