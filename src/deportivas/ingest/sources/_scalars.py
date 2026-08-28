"""Shared scalar coercion for source adapters.

Every adapter turns a DataFrame cell of unknown dynamic type (a real value, a
NaN float, a pandas NA, ``None``) into a clean Python ``int | None`` /
``float | None``. Small enough to look trivial, but every adapter needs the
exact same defensive handling, so it lives here once instead of five times.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def to_optional_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        result: int = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    return result


def to_optional_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        result: float = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result


def to_utc_from_timestamp(value: object) -> Any:
    """Returns a ``pd.Timestamp`` for a dynamically-typed date/datetime cell,
    or ``None`` if the cell is missing. Callers combine this with their own
    hour/minute handling (FBref has a separate time column; football-data.co.uk
    already has a combined datetime)."""
    dynamic_value: Any = value
    if dynamic_value is None or pd.isna(dynamic_value):
        return None
    return pd.Timestamp(dynamic_value)
