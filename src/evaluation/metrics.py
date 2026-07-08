# Please add any functions for evaluating the model/ metrics

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def classification_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def clustering_metrics(X, cluster_labels):
    real_clusters = set(cluster_labels) - {-1}

    if len(real_clusters) < 2:
        return {
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
            "note": "Clustering metrics require at least 2 non-noise clusters."
        }

    return {
        "silhouette_score": silhouette_score(X, cluster_labels),
        "davies_bouldin_score": davies_bouldin_score(X, cluster_labels),
        "calinski_harabasz_score": calinski_harabasz_score(X, cluster_labels),
        "note": None
    }