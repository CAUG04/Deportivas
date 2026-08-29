"""Brier score, log loss and a reliability curve — the three numbers
``config/thresholds.yaml``'s ``calibration`` section cares about
(``max_brier_score``, ``reliability_bins``), computed generically over any
sequence of (predicted probability, binary outcome) pairs.

A market like ``1x2`` writes one ``predictions`` row per selection (home,
draw, away), each already a binary "did this selection happen" outcome. All
three functions here treat every such row independently and flatten them —
the standard one-hot/binary decomposition of a multi-class Brier score —
rather than grouping by fixture, so the same functions work unmodified for a
two-way market (btts, moneyline) and an N-way one alike.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(probs: Sequence[float], outcomes: Sequence[int], *, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    y = np.asarray(outcomes, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    bin_mid: float
    mean_predicted: float
    observed_frequency: float
    count: int


def reliability_curve(
    probs: Sequence[float], outcomes: Sequence[int], *, n_bins: int
) -> list[ReliabilityBin]:
    """Equal-width bins over ``[0, 1]``. Empty bins are omitted rather than
    reported as a misleading 0.0 — a bin with no predictions in it says
    nothing about calibration.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)

    curve = []
    for b in range(n_bins):
        mask = bin_index == b
        if not mask.any():
            continue
        curve.append(
            ReliabilityBin(
                bin_mid=float((edges[b] + edges[b + 1]) / 2.0),
                mean_predicted=float(p[mask].mean()),
                observed_frequency=float(y[mask].mean()),
                count=int(mask.sum()),
            )
        )
    return curve
