"""Generic logistic-regression moneyline classifier: P(home win) from an
arbitrary numeric feature vector.

Sport-agnostic on purpose: which fields exist in the vector is entirely up
to the ``feature_set`` trained against (``nfl_v1``, ``nba_v1``, ``nhl_v1``,
``mlb_v1``) — this module never hardcodes a sport's feature names, unlike
football's Poisson model which fits its own coefficients directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from deportivas.models.feature_matrix import FeatureMatrix, fit_feature_matrix, transform


@dataclass(frozen=True, slots=True)
class MoneylineModel:
    classifier: LogisticRegression
    matrix: FeatureMatrix

    def predict_proba_home(self, vector: dict[str, object]) -> float:
        x = transform(self.matrix, [vector])
        return float(self.classifier.predict_proba(x)[0, 1])


def fit_moneyline_model(vectors: list[dict[str, object]], home_win: list[int]) -> MoneylineModel:
    """``home_win`` must have at least two distinct values (0 and 1) —
    tied matches, uninformative for a home/away classifier, should already
    be excluded by the caller before this point."""
    matrix = fit_feature_matrix(vectors)
    x = transform(matrix, vectors)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(x, home_win)
    return MoneylineModel(classifier=classifier, matrix=matrix)
