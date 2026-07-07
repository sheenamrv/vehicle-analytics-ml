import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

from src.model.model_training import (
    evaluate_classifier_predictions,
    evaluate_cluster_labels,
    fit_model,
    fit_predict_model,
    predict_model,
    split_train_test,
    train_test_classifier,
)


def _classification_data():
    X = pd.DataFrame({
        "x1": [0, 0, 1, 1, 0, 1],
        "x2": [0, 1, 0, 1, 0, 1],
    })
    y = pd.Series([0, 0, 1, 1, 0, 1])
    return X, y


def test_split_train_test_returns_expected_sizes():
    X, y = _classification_data()

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=0.5,
        random_state=42,
        stratify=False,
    )

    assert len(X_train) == 3
    assert len(X_test) == 3
    assert len(y_train) == 3
    assert len(y_test) == 3


def test_fit_and_predict_model_for_classifier():
    X, y = _classification_data()
    model = DecisionTreeClassifier(random_state=42)

    fit_model(model, X, y)
    predictions = predict_model(model, X)

    assert len(predictions) == len(X)


def test_train_test_classifier_uses_shared_helpers():
    X, y = _classification_data()
    model = DecisionTreeClassifier(random_state=42)

    result = train_test_classifier(
        model=model,
        X=X,
        y=y,
        test_size=0.5,
        random_state=42,
    )

    assert set(result.keys()) == {
        "model",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "predictions",
    }
    assert len(result["predictions"]) == len(result["y_test"])


def test_evaluate_classifier_predictions_returns_standard_keys():
    evaluation = evaluate_classifier_predictions(
        y_true=[0, 1, 1],
        y_pred=[0, 1, 0],
    )

    assert "metrics" in evaluation
    assert "confusion_matrix" in evaluation
    assert "predictions" in evaluation
    assert evaluation["metrics"]["accuracy"] >= 0


def test_fit_predict_model_for_clusterer():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [5.0, 5.0],
        [5.1, 5.1],
    ])
    model = KMeans(n_clusters=2, random_state=42, n_init=10)

    labels = fit_predict_model(model, X)

    assert len(labels) == len(X)
    assert set(labels) == {0, 1}


def test_evaluate_cluster_labels_returns_metrics_dict():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [5.0, 5.0],
        [5.1, 5.1],
    ])
    labels = np.array([0, 0, 1, 1])

    metrics = evaluate_cluster_labels(X, labels)

    assert "silhouette_score" in metrics
    assert "davies_bouldin_score" in metrics
    assert "calinski_harabasz_score" in metrics
