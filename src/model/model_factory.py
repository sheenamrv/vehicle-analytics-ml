from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SUPPORTED_SUPERVISED_MODELS = {
    "svm",
    "knn",
    "decision_tree",
    "random_forest",
    "logistic_regression",
}

SUPPORTED_CLUSTERING_MODELS = {
    "kmeans",
    "dbscan",
    "hierarchical",
}

# Build supervised model
def build_model(model_type, parameters=None, random_state=42):
    parameters = parameters or {}

    if model_type == "svm":
        return SVC(
            C=parameters.get("C", 1.0),
            kernel=parameters.get("kernel", "rbf"),
            probability=parameters.get("probability", True),
            random_state=random_state,
        )

    if model_type == "knn":
        return KNeighborsClassifier(
            n_neighbors=parameters.get("n_neighbors", 5)
        )

    if model_type == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=parameters.get("max_depth"),
            random_state=random_state,
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=parameters.get("n_estimators", 100),
            max_depth=parameters.get("max_depth"),
            random_state=random_state,
        )

    if model_type == "logistic_regression":
        return LogisticRegression(
            C=parameters.get("C", 1.0),
            max_iter=parameters.get("max_iter", 1000),
            random_state=random_state,
        )

    raise ValueError(f"Unsupported supervised model_type: {model_type}")

# Build unsupervised model
def build_clusterer(method, parameters=None, random_state=42):
    parameters = parameters or {}

    if method == "kmeans":
        return KMeans(
            n_clusters=parameters.get("n_clusters", 3),
            random_state=random_state,
            n_init=parameters.get("n_init", 10),
        )

    if method == "dbscan":
        return DBSCAN(
            eps=parameters.get("eps", 0.5),
            min_samples=parameters.get("min_samples", 5),
        )

    if method == "hierarchical":
        return AgglomerativeClustering(
            n_clusters=parameters.get("n_clusters", 3),
            linkage=parameters.get("linkage", "ward"),
        )

    raise ValueError(f"Unsupported clustering method: {method}")
