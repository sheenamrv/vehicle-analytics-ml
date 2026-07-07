import pandas as pd
import pytest

from src.preprocessing.preprocessing import (
    fill_missing_values_frontend,
    get_missing_values,
    normalize_columns,
    standardize_columns,
)


def test_get_missing_values_only_missing():
    df = pd.DataFrame({"a": [1, None], "b": [1, 2]})

    result = get_missing_values(df)

    assert list(result["column"]) == ["a"]
    assert list(result["missing_count"]) == [1]


def test_fill_missing_values_frontend_median_does_not_mutate_original():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})

    updated = fill_missing_values_frontend(df, column="a", method="median")

    assert updated["a"].isna().sum() == 0
    assert df["a"].isna().sum() == 1


def test_fill_missing_values_frontend_constant():
    df = pd.DataFrame({"a": ["x", None]})

    updated = fill_missing_values_frontend(df, column="a", method="constant", value="missing")

    assert list(updated["a"]) == ["x", "missing"]


def test_fill_missing_values_frontend_rejects_bad_column():
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError, match="Column does not exist"):
        fill_missing_values_frontend(df, column="b", method="median")


def test_standardize_columns_returns_copy():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    updated = standardize_columns(df, selected=["a"])

    assert round(updated["a"].mean(), 7) == 0
    assert list(df["a"]) == [1.0, 2.0, 3.0]


def test_normalize_columns_returns_values_between_zero_and_one():
    df = pd.DataFrame({"a": [10.0, 20.0, 30.0]})

    updated = normalize_columns(df, selected=["a"])

    assert updated["a"].min() == 0
    assert updated["a"].max() == 1
