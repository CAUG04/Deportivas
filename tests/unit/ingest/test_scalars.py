from __future__ import annotations

import pandas as pd

from deportivas.ingest.sources._scalars import (
    to_optional_float,
    to_optional_int,
    to_utc_from_timestamp,
)


def test_to_optional_int_valid_value() -> None:
    assert to_optional_int(7) == 7


def test_to_optional_int_none_is_none() -> None:
    assert to_optional_int(None) is None


def test_to_optional_int_nan_is_none() -> None:
    assert to_optional_int(float("nan")) is None


def test_to_optional_int_malformed_string_is_none() -> None:
    assert to_optional_int("not-a-number") is None


def test_to_optional_float_valid_value() -> None:
    assert to_optional_float("1.5") == 1.5


def test_to_optional_float_none_is_none() -> None:
    assert to_optional_float(None) is None


def test_to_optional_float_nan_is_none() -> None:
    assert to_optional_float(float("nan")) is None


def test_to_optional_float_malformed_string_is_none() -> None:
    assert to_optional_float("not-a-number") is None


def test_to_utc_from_timestamp_valid_value() -> None:
    result = to_utc_from_timestamp(pd.Timestamp("2026-01-10"))
    assert result == pd.Timestamp("2026-01-10")


def test_to_utc_from_timestamp_none_is_none() -> None:
    assert to_utc_from_timestamp(None) is None


def test_to_utc_from_timestamp_nan_is_none() -> None:
    assert to_utc_from_timestamp(float("nan")) is None
