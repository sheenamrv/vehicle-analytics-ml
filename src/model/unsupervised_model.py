import numpy as np
from src.model.model_factory import build_clusterer
from src.model.model_training import fit_predict_model
from src.model.model_utils import (
    get_num_feature_columns,
    prepare_training_features,
    scale_features,
)
from src.model.result_builders import build_clustering_results

# Validate the selected clustering method and its parameters before training
def validate(method, rows, features, parameters):
    method = str(method).strip().lower()

    if method not in {"kmeans", "dbscan", "hierarchical"}:
        raise ValueError(f"Unsupported clustering method: {method}")

    if rows < 2:
        raise ValueError("Clustering requires at least two rows.")

    if not features:
        raise ValueError("At least one numeric feature is required for clustering.")

    # K-Means and hierarchical clustering require a fixed number of clusters
    if method in {"kmeans", "hierarchical"}:
        clusters = int(parameters.get("n_clusters", 3))

        if clusters < 2:
            raise ValueError("n_clusters must be at least 2.")

        if clusters > rows:
            raise ValueError(
                f"n_clusters ({clusters}) cannot exceed the number of rows ({rows})."
            )

    # DBSCAN requires a positive neighborhood size and at least one sample
    if method == "dbscan":
        if float(parameters.get("eps", 0.5)) <= 0:
            raise ValueError("eps must be greater than 0.")

        if int(parameters.get("min_samples", 5)) < 1:
            raise ValueError("min_samples must be at least 1.")

    # Ward linkage only works with Euclidean distance
    if method == "hierarchical":
        linkage = parameters.get("linkage", "ward")
        metric = parameters.get("metric", "euclidean")

        if linkage == "ward" and metric != "euclidean":
            raise ValueError("Ward linkage requires the euclidean metric.")

    return method


# Run K-Means directly on already scaled features
def run_kmeans(X_scaled, n_clusters=3, random_state=42, **parameters):
    parameters = {**parameters, "n_clusters": n_clusters}

    model = build_clusterer("kmeans", parameters, random_state=random_state)
    labels = fit_predict_model(model, X_scaled)

    return model, labels


# Run DBSCAN directly on already scaled features
def run_dbscan(X_scaled, eps=0.5, min_samples=5, **parameters):
    parameters = {**parameters, "eps": eps, "min_samples": min_samples}

    model = build_clusterer("dbscan", parameters)
    labels = fit_predict_model(model, X_scaled)

    return model, labels


# Run hierarchical clustering directly on already scaled features
def run_hierarchical(X_scaled, n_clusters=3, linkage="ward", **parameters):
    parameters = {**parameters, "n_clusters": n_clusters, "linkage": linkage}

    model = build_clusterer("hierarchical", parameters)
    labels = fit_predict_model(model, X_scaled)

    return model, labels


# do not alter
# Full unsupervised workflow function
# - Gets numeric features if no custom features are provided
# - Validates the selected clustering method and parameters
# - Prepares and scales the selected feature columns
# - Trains the selected clustering model
# - Builds clustering metrics, summaries, and PCA plotting data
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
    n_init=10,
    init="k-means++",
    max_iter=300,
    tol=0.0001,
    algorithm="auto",
    metric="euclidean",
    leaf_size=30,
    compute_distances=False,
    **extra_parameters,
):
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    # Get the numeric feature columns if features were not provided
    if features is None:
        features = get_num_feature_columns(df, label)

    method_name = str(method).strip().lower()

    # Modern scikit-learn versions no longer accept "auto" for K-Means
    if method_name == "kmeans" and algorithm == "auto":
        algorithm = "lloyd"

    # Store all supported clustering parameters in one dictionary
    # build_clusterer() selects only the settings required by each algorithm
    parameters = {
        "n_clusters": n_clusters,
        "eps": eps,
        "min_samples": min_samples,
        "linkage": linkage,
        "n_init": n_init,
        "init": init,
        "max_iter": max_iter,
        "tol": tol,
        "algorithm": algorithm,
        "metric": metric,
        "leaf_size": leaf_size,
        "compute_distances": compute_distances,
        **extra_parameters,
    }

    # Validate the method and parameter values before model training
    method = validate(
        method=method,
        rows=len(df),
        features=features,
        parameters=parameters,
    )

    # Prepare the selected feature matrix and handle missing values
    X = prepare_training_features(
        df=df,
        features=features,
        fill_method=fill_method,
        fill_value=fill_value,
    )

    # Ensure preprocessing did not leave missing or infinite values
    X_array = X.to_numpy(dtype=float)

    if X.isna().any().any() or not np.isfinite(X_array).all():
        raise ValueError(
            "Features contain missing or infinite values after preprocessing."
        )

    # Scale the data so large-valued columns do not dominate distance calculations
    X_scaled, scaler = scale_features(X)

    # Build and train the selected clustering model
    model = build_clusterer(
        method,
        parameters=parameters,
        random_state=random_state,
    )
    labels = fit_predict_model(model, X_scaled)

    # Preserve the original result format created by build_clustering_results()
    result = build_clustering_results(
        method=method,
        features=features,
        model=model,
        scaler=scaler,
        labels=labels,
        df=df,
        X_scaled=X_scaled,
    )

    return result
