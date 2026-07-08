import numpy as np

from src.evaluation.metrics import (
    classification_metrics,
    classification_confusion_matrix,
    clustering_metrics,
)


def test_classification_metrics():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]

    metrics = classification_metrics(y_true, y_pred)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_classification_confusion_matrix():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]

    cm = classification_confusion_matrix(y_true, y_pred)

    assert cm.shape == (2, 2)


def test_clustering_metrics_valid():
    X = np.array([
        [1, 1],
        [1.1, 1],
        [5, 5],
        [5.1, 5],
    ])

    labels = np.array([0, 0, 1, 1])

    metrics = clustering_metrics(X, labels)

    assert "silhouette_score" in metrics
    assert "davies_bouldin_score" in metrics
    assert "calinski_harabasz_score" in metrics


def test_clustering_metrics_invalid_one_cluster():
    X = np.array([
        [1, 1],
        [1.1, 1],
        [1.2, 1],
    ])

    labels = np.array([0, 0, 0])

    metrics = clustering_metrics(X, labels)

    assert metrics["silhouette_score"] is None
    assert metrics["note"] is not None