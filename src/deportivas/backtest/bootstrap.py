"""Bootstrap confidence intervals — ``config/thresholds.yaml``'s ``backtest``
section (``bootstrap_iterations``, ``confidence_level``). A sample of CLV or
pnl values is never symmetric (a handful of big-price wins skews it), so
resampling with replacement gives an honest interval instead of one that
assumes a bell curve that isn't there.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    iterations: int,
    confidence_level: float,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap on the mean: resamples ``values`` with
    replacement ``iterations`` times, and returns the ``confidence_level``
    central interval of the resampled means — e.g. 0.95 returns the
    (2.5th, 97.5th) percentile. Raises ``ValueError`` on fewer than two
    values: there is nothing to resample meaningfully out of one point."""
    if len(values) < 2:
        raise ValueError("se necesitan al menos 2 valores para un bootstrap")
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(iterations, array.size), replace=True)
    means = samples.mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lower), float(upper)
