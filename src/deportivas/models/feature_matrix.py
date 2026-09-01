"""Turns a list of feature-vector dicts into a numeric matrix a scikit-learn
estimator can consume — one column per key ever seen across the vectors,
missing or ``None`` values filled with that column's TRAINING-set mean.

Filling from the training set's own mean (never the validation set's)
mirrors rule #4's "fit only on training data" spirit even though this isn't
calibration: a validation-window fixture with a missing feature (a team's
first game after promotion, say) still gets a neutral, pre-computed value
rather than leaking anything from the window being validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    feature_names: tuple[str, ...]
    fill_values: dict[str, float]


def fit_feature_matrix(vectors: list[dict[str, object]]) -> FeatureMatrix:
    names = sorted({key for vector in vectors for key in vector})
    fill_values = {}
    for name in names:
        values = [float(vector[name]) for vector in vectors if vector.get(name) is not None]  # type: ignore[arg-type]
        fill_values[name] = sum(values) / len(values) if values else 0.0
    return FeatureMatrix(feature_names=tuple(names), fill_values=fill_values)


def transform(matrix: FeatureMatrix, vectors: list[dict[str, object]]) -> np.ndarray:
    rows = []
    for vector in vectors:
        row = []
        for name in matrix.feature_names:
            value = vector.get(name)
            row.append(float(value) if value is not None else matrix.fill_values[name])  # type: ignore[arg-type]
        rows.append(row)
    return np.array(rows, dtype=float)
