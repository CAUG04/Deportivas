"""metrics.py: Brier score, log loss and reliability curve, all computed by
flattening (probability, outcome) pairs regardless of how many selections a
market has."""

from __future__ import annotations

import pytest

from deportivas.models.metrics import brier_score, log_loss, reliability_curve


def test_brier_score_perfect_predictions_is_zero() -> None:
    assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == pytest.approx(0.0)


def test_brier_score_worst_predictions_is_one() -> None:
    assert brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)


def test_brier_score_coin_flip_binary_is_a_quarter() -> None:
    assert brier_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.25)


def test_log_loss_perfect_predictions_near_zero() -> None:
    assert log_loss([0.999999, 0.000001], [1, 0]) == pytest.approx(0.0, abs=1e-4)


def test_log_loss_confidently_wrong_is_heavily_penalised() -> None:
    confident_wrong = log_loss([0.01], [1])
    uncertain = log_loss([0.5], [1])
    assert confident_wrong > uncertain


def test_log_loss_clips_extreme_probabilities_instead_of_raising() -> None:
    # sin el clip, log(0) explota; con el clip el resultado es finito y grande.
    result = log_loss([0.0], [1])
    assert result > 0
    assert result < float("inf")


def test_reliability_curve_perfectly_calibrated_bin() -> None:
    probs = [0.9] * 10
    outcomes = [1] * 9 + [0]
    curve = reliability_curve(probs, outcomes, n_bins=10)
    assert len(curve) == 1
    assert curve[0].mean_predicted == pytest.approx(0.9)
    assert curve[0].observed_frequency == pytest.approx(0.9)
    assert curve[0].count == 10


def test_reliability_curve_omits_empty_bins() -> None:
    probs = [0.05, 0.95]
    outcomes = [0, 1]
    curve = reliability_curve(probs, outcomes, n_bins=10)
    assert len(curve) == 2


def test_reliability_curve_bin_mid_reflects_bin_position() -> None:
    curve = reliability_curve([0.05], [0], n_bins=10)
    assert curve[0].bin_mid == pytest.approx(0.05)
