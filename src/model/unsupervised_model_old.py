import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

from src.model.model_utils import (
    get_num_feature_columns,
    prepare_training_features,
)

from src.evaluation.metrics import clustering_metrics


def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def run_kmeans(X_scaled, n_clusters=3, random_state=42):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    return model, labels


def run_dbscan(X_scaled, eps=0.5, min_samples=5):
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(X_scaled)

    return model, labels


def run_hierarchical(X_scaled, n_clusters=3, linkage="ward"):
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )

    labels = model.fit_predict(X_scaled)

    return model, labels


def create_cluster_summary(cluster_labels):
    counts = pd.Series(cluster_labels).value_counts().sort_index()

    return pd.DataFrame({
        "cluster": counts.index,
        "count": counts.values
    })


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
        "cluster": cluster_labels.astype(str)
    })

    return {
        "pca_df": pca_df,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance_sum": pca.explained_variance_ratio_.sum()
    }


def run_unsupervised_workflow(
    df,
    method,
    label=None,
    features=None,
    n_clusters=3,
    eps=0.5,
    min_samples=5,
    linkage="ward",
    random_state=42
):
    if features is None:
        features = get_num_feature_columns(df, label)

    X = prepare_training_features(df, features)
    X_scaled, scaler = scale_features(X)

    if method == "kmeans":
        model, labels = run_kmeans(
            X_scaled,
            n_clusters=n_clusters,
            random_state=random_state
        )

    elif method == "dbscan":
        model, labels = run_dbscan(
            X_scaled,
            eps=eps,
            min_samples=min_samples
        )

    elif method == "hierarchical":
        model, labels = run_hierarchical(
            X_scaled,
            n_clusters=n_clusters,
            linkage=linkage
        )

    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    return {
        "method": method,
        "features": features,
        "model": model,
        "scaler": scaler,
        "cluster_labels": labels,
        "clustered_df": create_clustered_dataframe(df, labels),
        "summary_df": create_cluster_summary(labels),
        "metrics": clustering_metrics(X_scaled, labels),
        "pca_result": create_pca_cluster_data(X_scaled, labels)
    }