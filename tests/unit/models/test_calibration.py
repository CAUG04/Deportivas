"""calibration.py: isotonic and Platt calibrators, both fit only on the
training data handed to fit_calibrator."""

from __future__ import annotations

import random

import pytest

from deportivas.models.calibration import fit_calibrator


def _synthetic_training_data(n: int = 300, *, seed: int = 0) -> tuple[list[float], list[int]]:
    """Raw probabilities that are systematically overconfident (true rate is
    closer to 0.5 than the raw probability claims) — a calibrator worth its
    salt should pull them back toward the middle."""
    rng = random.Random(seed)
    probs = []
    outcomes = []
    for _ in range(n):
        raw = rng.uniform(0.05, 0.95)
        true_rate = 0.5 + (raw - 0.5) * 0.4  # el rating real esta mucho mas cerca de 0.5
        probs.append(raw)
        outcomes.append(1 if rng.random() < true_rate else 0)
    return probs, outcomes


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibrated_predictions_are_valid_probabilities(method: str) -> None:
    probs, outcomes = _synthetic_training_data()
    calibrator = fit_calibrator(method, probs, outcomes)
    calibrated = calibrator.calibrate([0.1, 0.5, 0.9])
    assert all(0.0 <= p <= 1.0 for p in calibrated)


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibrator_pulls_overconfident_predictions_toward_the_middle(method: str) -> None:
    probs, outcomes = _synthetic_training_data()
    calibrator = fit_calibrator(method, probs, outcomes)
    calibrated = calibrator.calibrate([0.95])[0]
    assert calibrated < 0.95


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibrated_output_is_monotonic_in_raw_probability(method: str) -> None:
    probs, outcomes = _synthetic_training_data()
    calibrator = fit_calibrator(method, probs, outcomes)
    low, mid, high = calibrator.calibrate([0.1, 0.5, 0.9])
    assert low <= mid <= high


def test_unknown_calibration_method_raises() -> None:
    with pytest.raises(ValueError, match="metodo de calibracion desconocido"):
        fit_calibrator("bogus", [0.1, 0.9], [0, 1])


def test_isotonic_and_platt_give_different_results_on_the_same_data() -> None:
    probs, outcomes = _synthetic_training_data()
    isotonic = fit_calibrator("isotonic", probs, outcomes).calibrate([0.7])[0]
    platt = fit_calibrator("platt", probs, outcomes).calibrate([0.7])[0]
    assert isotonic != pytest.approx(platt, abs=1e-9)
