import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.model.semisupervised_model import get_num_feature_columns, convert_ssl_labels, train_ssl, get_ssl_training_progress, evaluate_ssl_model, run_ssl_workflow

def generate_train_df():
    return pd.DataFrame({
        "feature1": [40, 42, 44, 70, 72, 74, 41, 73],
        "feature2": [2, 3, 2, 8, 9, 8, 2, 9],
        "feature3": [75, 77, 76, 95, 98, 97, 76, 96],
        "feature4": [1.1, 1.2, 1.0, 4.2, 4.5, 4.3, 1.1, 4.4],
        "label": [1, 1, 1, 0, 0, 0, None, None],
    })


def generate_test_df():
    return pd.DataFrame({
        "feature1": [43, 71],
        "feature2": [2, 8],
        "feature3": [76, 96],
        "feature4": [1.1, 4.4],
        "label": [1, 0],
    })


def train_base_model():
    df = generate_train_df().dropna()
    features = get_num_feature_columns(df, "label")

    X = df[features]
    y = df["label"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model


def test_get_num_feature_columns():
    df = generate_train_df()
    features = get_num_feature_columns(df, "label")

    assert features == ["feature1", "feature2", "feature3", "feature4"]


def test_convert_ssl_labels():
    df = generate_train_df()
    y = convert_ssl_labels(df, "label")

    assert list(y.tail(2)) == [-1, -1]


def test_train_ssl():
    df = generate_train_df()
    features = get_num_feature_columns(df, "label")
    pretrained_model = train_base_model()

    result = train_ssl(
        df=df,
        features=features,
        label="label",
        pretrained_model=pretrained_model,
        threshold=0.5,
        max_iter=5,
    )

    assert "model" in result
    assert "transduction" in result
    assert "labeled_iter" in result
    assert "progress_df" in result
    assert len(result["transduction"]) == len(df)


def test_get_ssl_training_progress():
    train_ssl_result = {
        "labeled_iter": np.array([0, 0, 1, -1, 0, 2, 1])
    }

    progress = get_ssl_training_progress(train_ssl_result)

    assert list(progress["iteration"]) == [-1, 0, 1, 2]
    assert list(progress["count"]) == [1, 3, 2, 1]
    assert list(progress["description"]) == ["Never labeled", "Originally labeled", "Pseudo-labeled on iteration 1", "Pseudo-labeled on iteration 2"]


def test_evaluate_ssl_model():
    train_df = generate_train_df()
    test_df = generate_test_df()
    features = get_num_feature_columns(train_df, "label")
    pretrained_model = train_base_model()

    ssl_result = train_ssl(
        train_df,
        features,
        "label",
        pretrained_model,
        threshold=0.5,
        max_iter=5,
    )

    evaluation = evaluate_ssl_model(
        model=ssl_result["model"],
        test_df=test_df,
        features=features,
        label="label",
    )

    assert "metrics" in evaluation
    assert "confusion_matrix" in evaluation
    assert "predictions" in evaluation


def test_run_ssl_workflow():
    train_df = generate_train_df()
    test_df = generate_test_df()
    pretrained_model = train_base_model()

    result = run_ssl_workflow(
        train_df=train_df,
        test_df=test_df,
        label="label",
        pretrained_model=pretrained_model,
        threshold=0.5,
        max_iter=5,
    )

    assert "ssl_model" in result
    assert "metrics" in result
    assert "confusion_matrix" in result
    assert "iteration_progress" in result