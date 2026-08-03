# Please add any functions for evaluating the model/ metrics

import numpy as np

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
    labels = np.asarray(cluster_labels)
    non_noise_mask = labels != -1
    filtered_labels = labels[non_noise_mask]
    filtered_X = np.asarray(X)[non_noise_mask]
    unique_clusters = np.unique(filtered_labels)

    if len(unique_clusters) < 2 or len(filtered_labels) <= len(unique_clusters):
        return {
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
            "noise_count": int((labels == -1).sum()),
            "cluster_count": int(len(unique_clusters)),
            "note": "Clustering metrics require at least 2 non-noise clusters with enough samples.",
        }

    try:
        silhouette = silhouette_score(filtered_X, filtered_labels)
        davies = davies_bouldin_score(filtered_X, filtered_labels)
        calinski = calinski_harabasz_score(filtered_X, filtered_labels)
        note = None
    except ValueError as error:
        silhouette = davies = calinski = None
        note = str(error)

    return {
        "silhouette_score": silhouette,
        "davies_bouldin_score": davies,
        "calinski_harabasz_score": calinski,
        "noise_count": int((labels == -1).sum()),
        "cluster_count": int(len(unique_clusters)),
        "note": note,
    }