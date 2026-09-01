"""Regla innegociable #1: ningun feature puede usar informacion posterior al
kickoff. Este test debe fallar (es decir, detectar la violacion) si alguien
introduce una feature con as_of_timestamp >= kickoff_utc.
"""

from __future__ import annotations

import pandas as pd
import pytest

from deportivas.domain.leakage import LeakageError, assert_no_leakage, find_leakage


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fixture_id": ["f1", "f2"],
            "kickoff_utc": pd.to_datetime(
                ["2026-01-10T20:00:00Z", "2026-01-11T18:00:00Z"], utc=True
            ),
        }
    )


def test_clean_features_pass() -> None:
    features = pd.DataFrame(
        {
            "fixture_id": ["f1", "f2"],
            "as_of_timestamp": pd.to_datetime(
                ["2026-01-10T19:00:00Z", "2026-01-11T10:00:00Z"], utc=True
            ),
        }
    )
    assert_no_leakage(features, _fixtures())  # no debe lanzar


def test_feature_at_exact_kickoff_is_leakage() -> None:
    """as_of == kickoff cuenta como leakage: el margen es estricto a proposito."""
    features = pd.DataFrame(
        {
            "fixture_id": ["f1"],
            "as_of_timestamp": pd.to_datetime(["2026-01-10T20:00:00Z"], utc=True),
        }
    )
    with pytest.raises(LeakageError, match="f1"):
        assert_no_leakage(features, _fixtures())


def test_feature_after_kickoff_is_leakage() -> None:
    features = pd.DataFrame(
        {
            "fixture_id": ["f1", "f2"],
            "as_of_timestamp": pd.to_datetime(
                ["2026-01-10T21:00:00Z", "2026-01-11T10:00:00Z"], utc=True
            ),
        }
    )
    violations = find_leakage(features, _fixtures())
    assert len(violations) == 1
    assert violations[0].fixture_id == "f1"


def test_naive_timestamps_are_rejected_not_silently_assumed() -> None:
    features = pd.DataFrame(
        {"fixture_id": ["f1"], "as_of_timestamp": pd.to_datetime(["2026-01-10T19:00:00"])}
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        assert_no_leakage(features, _fixtures())


def test_non_datetime_column_rejected() -> None:
    features = pd.DataFrame({"fixture_id": ["f1"], "as_of_timestamp": ["not-a-timestamp"]})
    with pytest.raises(TypeError, match="debe ser datetime"):
        assert_no_leakage(features, _fixtures())


def test_missing_column_raises_keyerror() -> None:
    features = pd.DataFrame({"fixture_id": ["f1"]})
    with pytest.raises(KeyError, match="as_of_timestamp"):
        assert_no_leakage(features, _fixtures())
