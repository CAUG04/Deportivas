"""Combines several feature modules' per-fixture vectors into one, shared by
every sport's ``pipeline.py``.
"""

from __future__ import annotations

import pandas as pd


def merge_vectors(base: pd.DataFrame, *extras: pd.DataFrame) -> pd.DataFrame:
    """Left-to-right dict-merges each frame's ``vector`` column into ``base``'s,
    joined on ``fixture_id``, keeping ``base``'s ``as_of_timestamp``. Every
    frame must cover exactly the same set of fixtures — true whenever every
    module in a pipeline walks the same ``fixtures`` DataFrame.
    """
    merged = base[["fixture_id", "as_of_timestamp", "vector"]].copy()
    for extra in extras:
        merged = merged.merge(
            extra[["fixture_id", "vector"]], on="fixture_id", suffixes=("", "_extra")
        )
        merged["vector"] = pd.Series(
            [
                {**left, **right}
                for left, right in zip(merged["vector"], merged["vector_extra"], strict=True)
            ],
            index=merged.index,
        )
        merged = merged.drop(columns="vector_extra")
    return merged
