import pandas as pd
import pytest

from src.model.model_utils import (
    align_features,
    get_categorical_feature_columns,
    get_num_feature_columns,
    prepare_training_data,
    prepare_training_features,
    scale_features,
    validate_dataset,
)


def test_validate_dataset_rejects_missing_label_values():
    df = pd.DataFrame({"x": [1, 2], "label": [0, None]})

    valid, message = validate_dataset(df, "label")

    assert valid is False
    assert "missing" in message.lower()


def test_get_feature_column_helpers_exclude_label():
    df = pd.DataFrame({
        "num": [1, 2],
        "cat": ["a", "b"],
        "label": [0, 1],
    })

    assert get_num_feature_columns(df, "label") == ["num"]
    assert get_categorical_feature_columns(df, "label") == ["cat"]


def test_prepare_training_features_fills_numeric_and_encodes_categorical():
    df = pd.DataFrame({
        "num": [1.0, None, 3.0],
        "cat": ["a", None, "b"],
        "label": [0, 1, 0],
    })

    X = prepare_training_features(df, features=["num", "cat"], fill_method="median")

    assert X["num"].isna().sum() == 0
    assert "cat_a" in X.columns
    assert "cat_b" in X.columns
    assert "cat___missing__" in X.columns


def test_prepare_training_features_rejects_unknown_fill_method():
    df = pd.DataFrame({"x": [1, None]})

    with pytest.raises(ValueError, match="Unsupported fill_method"):
        prepare_training_features(df, features=["x"], fill_method="bad")


def test_prepare_training_data_returns_X_and_y():
    df = pd.DataFrame({"x": [1, 2, 3], "label": [0, 1, 0]})

    X, y = prepare_training_data(df, label_col="label")

    assert list(X.columns) == ["x"]
    assert list(y) == [0, 1, 0]


def test_align_features_adds_missing_columns_and_orders_them():
    X = pd.DataFrame({"b": [2], "a": [1]})

    aligned = align_features(X, expected_columns=["a", "b", "c"], fill_value=0)

    assert list(aligned.columns) == ["a", "b", "c"]
    assert aligned.loc[0, "c"] == 0


def test_scale_features_returns_scaled_data_and_scaler():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})

    X_scaled, scaler = scale_features(X)

    assert X_scaled.shape == (3, 2)
    assert hasattr(scaler, "transform")
