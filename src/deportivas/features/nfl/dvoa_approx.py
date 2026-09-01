"""A deliberately simplified, "approximate DVOA": each side's rolling EPA
compared against the specific opponent's rolling numbers for this matchup,
rather than Football Outsiders' full methodology (situation-neutral play
values, iterative opponent-strength regression across the whole league,
week-by-week weighting). This is a pure transform of ``epa_rolling.py``'s
already point-in-time-safe output — both teams' pre-match snapshots already
live in the same fixture row, so no new walk-forward pass or ``as_of``
tracking is needed; the result is exactly as leakage-safe as its input.

Higher is better for both numbers, by construction:

* ``dvoa_off``: this side's typical offensive EPA/play minus what the
  opponent's defense typically allows — a positive edge for the attack.
* ``dvoa_def``: the opponent's typical offensive EPA/play minus what this
  side's defense typically allows — how far below their own norm this
  defense usually holds opponents.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_WINDOW = 8

_OPPONENT_SIDE = {"home": "away", "away": "home"}


def compute_dvoa_approx(
    epa_rolling_features: pd.DataFrame, *, window: int = DEFAULT_WINDOW
) -> pd.DataFrame:
    """``epa_rolling_features`` is ``epa_rolling.compute_epa_rolling``'s own
    output. ``window`` must be one of its ``EpaRollingConfig.windows``.
    """
    rows: list[dict[str, object]] = []
    for record in epa_rolling_features.to_dict("records"):
        vector = record["vector"]
        dvoa: dict[str, object] = {}
        for side, opponent in _OPPONENT_SIDE.items():
            off = vector[f"off_epa_rolling_{window}_{side}"]
            opponent_def = vector[f"def_epa_allowed_rolling_{window}_{opponent}"]
            opponent_off = vector[f"off_epa_rolling_{window}_{opponent}"]
            own_def = vector[f"def_epa_allowed_rolling_{window}_{side}"]
            dvoa[f"dvoa_off_{side}"] = _subtract(off, opponent_def)
            dvoa[f"dvoa_def_{side}"] = _subtract(opponent_off, own_def)
        rows.append(
            {
                "fixture_id": record["fixture_id"],
                "as_of_timestamp": record["as_of_timestamp"],
                "vector": dvoa,
            }
        )
    return pd.DataFrame(rows)


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right
