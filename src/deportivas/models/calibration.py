"""Isotonic and Platt probability calibration.

Project rule #4: "Todo modelo pasa por calibración... ajustada solo con
datos de entrenamiento." Both calibrators here are fit once, on a training
window's own raw probabilities and known outcomes, and then applied — frozen
— to a later, out-of-sample walk-forward validation window. Nothing here
ever sees a validation-window outcome.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ISOTONIC = "isotonic"
PLATT = "platt"

# Como en to_optional_float: evita log(0)/log(1) sin sesgar visiblemente la
# probabilidad calibrada resultante.
_LOGIT_EPS = 1e-6


class Calibrator(Protocol):
    def calibrate(self, raw_probs: Sequence[float]) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    model: IsotonicRegression

    def calibrate(self, raw_probs: Sequence[float]) -> list[float]:
        predicted = self.model.predict(np.asarray(raw_probs, dtype=float))
        return [float(v) for v in predicted]


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    model: LogisticRegression

    def calibrate(self, raw_probs: Sequence[float]) -> list[float]:
        predicted = self.model.predict_proba(_logit(raw_probs))[:, 1]
        return [float(v) for v in predicted]


def _logit(probs: Sequence[float]) -> np.ndarray:
    clipped = np.clip(np.asarray(probs, dtype=float), _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    odds = np.log(clipped / (1.0 - clipped))
    return np.asarray(odds, dtype=float).reshape(-1, 1)


def fit_calibrator(method: str, raw_probs: Sequence[float], outcomes: Sequence[int]) -> Calibrator:
    if method == ISOTONIC:
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(raw_probs, outcomes)
        return IsotonicCalibrator(model=model)
    if method == PLATT:
        logistic = LogisticRegression()
        logistic.fit(_logit(raw_probs), outcomes)
        return PlattCalibrator(model=logistic)
    raise ValueError(
        f"metodo de calibracion desconocido: {method!r} (usar {ISOTONIC!r} o {PLATT!r})"
    )
