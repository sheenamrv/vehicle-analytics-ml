import pandas as pd
from sklearn.decomposition import PCA

from src.evaluation.metrics import (
    classification_confusion_matrix,
    classification_metrics,
    clustering_metrics,
)


def build_classification_results(model, X_test, y_test, predictions, feature_columns):
    return {
        "model": model,
        "features": list(feature_columns),
        "metrics": classification_metrics(y_test, predictions),
        "confusion_matrix": classification_confusion_matrix(y_test, predictions),
        "predictions": predictions,
    }


def create_cluster_summary(cluster_labels):
    counts = pd.Series(cluster_labels).value_counts().sort_index()
    return pd.DataFrame({"cluster": counts.index, "count": counts.values})


def create_clustered_dataframe(df, cluster_labels):
    clustered_df = df.copy()
    clustered_df["cluster"] = cluster_labels
    return clustered_df


def create_pca_cluster_data(X_scaled, cluster_labels):
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({
        "PC1": pca_data[:, 0],
        "PC2": pca_data[:, 1],
        "cluster": cluster_labels.astype(str),
    })
    return {
        "pca_df": pca_df,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance_sum": pca.explained_variance_ratio_.sum(),
    }


def build_clustering_results(method, features, model, scaler, labels, df, X_scaled):
    return {
        "method": method,
        "features": list(features),
        "model": model,
        "scaler": scaler,
        "cluster_labels": labels,
        "clustered_df": create_clustered_dataframe(df, labels),
        "summary_df": create_cluster_summary(labels),
        "metrics": clustering_metrics(X_scaled, labels),
        "pca_result": create_pca_cluster_data(X_scaled, labels),
    }
