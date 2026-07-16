from src.model.model_factory import build_clusterer
from src.model.model_training import fit_predict_model
from src.model.model_utils import (
    get_num_feature_columns,
    prepare_training_features,
    scale_features,
)
from src.model.result_builders import (
    build_clustering_results,
    create_cluster_summary,
    create_clustered_dataframe,
    create_pca_cluster_data,
)


def run_kmeans(X_scaled, n_clusters=3, random_state=42):
    model = build_clusterer(
        "kmeans",
        {"n_clusters": n_clusters},
        random_state=random_state,
    )
    labels = fit_predict_model(model, X_scaled)
    return model, labels


def run_dbscan(X_scaled, eps=0.5, min_samples=5):
    model = build_clusterer(
        "dbscan",
        {"eps": eps, "min_samples": min_samples},
    )
    labels = fit_predict_model(model, X_scaled)
    return model, labels


def run_hierarchical(X_scaled, n_clusters=3, linkage="ward"):
    model = build_clusterer(
        "hierarchical",
        {"n_clusters": n_clusters, "linkage": linkage},
    )
    labels = fit_predict_model(model, X_scaled)
    return model, labels

# do not alter
def run_unsupervised_workflow(
    df,
    method,
    label=None,
    features=None,
    n_clusters=3,
    eps=0.5,
    min_samples=5,
    linkage="ward",
    random_state=42,
    fill_method="median",
    fill_value=None,
):
    if features is None:
        features = get_num_feature_columns(df, label)

    X = prepare_training_features(
        df=df,
        features=features,
        fill_method=fill_method,
        fill_value=fill_value,
    )
    X_scaled, scaler = scale_features(X)

    parameters = {
        "n_clusters": n_clusters,
        "eps": eps,
        "min_samples": min_samples,
        "linkage": linkage,
    }
    model = build_clusterer(method, parameters=parameters, random_state=random_state)
    labels = fit_predict_model(model, X_scaled)

    return build_clustering_results(
        method=method,
        features=features,
        model=model,
        scaler=scaler,
        labels=labels,
        df=df,
        X_scaled=X_scaled,
    )
