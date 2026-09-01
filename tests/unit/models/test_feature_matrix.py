"""feature_matrix.py: turns feature-vector dicts into a numeric matrix,
imputing missing values from the training set's own mean."""

from __future__ import annotations

import pytest

from deportivas.models.feature_matrix import fit_feature_matrix, transform


def test_feature_names_are_the_union_of_every_vector_sorted() -> None:
    matrix = fit_feature_matrix([{"a": 1.0, "b": 2.0}, {"b": 3.0, "c": 4.0}])
    assert matrix.feature_names == ("a", "b", "c")


def test_fill_value_is_the_mean_ignoring_missing_entries() -> None:
    matrix = fit_feature_matrix([{"a": 1.0}, {"a": 3.0}, {"a": None}, {}])
    assert matrix.fill_values["a"] == pytest.approx(2.0)


def test_fill_value_defaults_to_zero_when_never_observed() -> None:
    matrix = fit_feature_matrix([{"a": 1.0}, {"b": 2.0}])
    assert matrix.fill_values["b"] == pytest.approx(2.0)


def test_transform_uses_actual_values_when_present() -> None:
    matrix = fit_feature_matrix([{"a": 1.0}, {"a": 3.0}])
    result = transform(matrix, [{"a": 5.0}])
    assert result.tolist() == [[5.0]]


def test_transform_imputes_missing_values_with_training_mean() -> None:
    matrix = fit_feature_matrix([{"a": 1.0}, {"a": 3.0}])
    result = transform(matrix, [{"a": None}])
    assert result.tolist() == [[2.0]]


def test_transform_imputes_absent_key_the_same_as_none() -> None:
    matrix = fit_feature_matrix([{"a": 1.0}, {"a": 3.0}])
    result = transform(matrix, [{}])
    assert result.tolist() == [[2.0]]


def test_transform_column_order_matches_feature_names() -> None:
    matrix = fit_feature_matrix([{"b": 1.0, "a": 2.0}])
    result = transform(matrix, [{"a": 10.0, "b": 20.0}])
    assert matrix.feature_names == ("a", "b")
    assert result.tolist() == [[10.0, 20.0]]


def test_transform_handles_multiple_vectors() -> None:
    matrix = fit_feature_matrix([{"a": 1.0}, {"a": 3.0}])
    result = transform(matrix, [{"a": 5.0}, {"a": None}])
    assert result.tolist() == [[5.0], [2.0]]
