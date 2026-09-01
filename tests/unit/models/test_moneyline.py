"""moneyline.py: generic logistic-regression P(home win) classifier —
sport-agnostic, no hardcoded feature names."""

from __future__ import annotations

import random

from deportivas.models.moneyline import fit_moneyline_model


def _synthetic_training_data(
    n: int = 300, *, seed: int = 0
) -> tuple[list[dict[str, object]], list[int]]:
    """A single feature that genuinely drives the outcome, plus a team's
    first game (feature missing) mixed in to exercise imputation."""
    rng = random.Random(seed)
    vectors: list[dict[str, object]] = []
    outcomes: list[int] = []
    for _ in range(n):
        edge = rng.uniform(-3.0, 3.0)
        vector: dict[str, object] = {"rating_diff": edge if rng.random() > 0.05 else None}
        win_prob = 1.0 / (1.0 + 2.718281828 ** (-edge))
        vectors.append(vector)
        outcomes.append(1 if rng.random() < win_prob else 0)
    return vectors, outcomes


def test_predicts_valid_probabilities() -> None:
    vectors, outcomes = _synthetic_training_data()
    model = fit_moneyline_model(vectors, outcomes)
    prob = model.predict_proba_home({"rating_diff": 1.0})
    assert 0.0 <= prob <= 1.0


def test_higher_rating_diff_means_higher_home_win_probability() -> None:
    vectors, outcomes = _synthetic_training_data()
    model = fit_moneyline_model(vectors, outcomes)
    low = model.predict_proba_home({"rating_diff": -2.0})
    high = model.predict_proba_home({"rating_diff": 2.0})
    assert high > low


def test_missing_feature_at_predict_time_is_imputed() -> None:
    vectors, outcomes = _synthetic_training_data()
    model = fit_moneyline_model(vectors, outcomes)
    # no debe lanzar, y debe devolver una probabilidad valida
    prob = model.predict_proba_home({})
    assert 0.0 <= prob <= 1.0


def test_unseen_feature_key_at_predict_time_is_ignored() -> None:
    vectors, outcomes = _synthetic_training_data()
    model = fit_moneyline_model(vectors, outcomes)
    with_extra = model.predict_proba_home({"rating_diff": 1.0, "never_trained_on": 999.0})
    without_extra = model.predict_proba_home({"rating_diff": 1.0})
    assert with_extra == without_extra
