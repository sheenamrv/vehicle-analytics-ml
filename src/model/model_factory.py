from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.model.model_controller import MODEL_CATALOG


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

SUPPORTED_MODELS = {

    "supervised":[

        "svm",

        "knn",

        ...

    ],

    "unsupervised":[

        "kmeans",

        ...

    ],

    "semi_supervised":[

        "self_training"

    ]

}

# do not alter
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
# do not alter
# Build unsupervised model
def build_clusterer(method, parameters=None, random_state=42):
    parameters = parameters or {}
    method = str(method).strip().lower()

    if method == "kmeans":
        return KMeans(
            n_clusters=int(parameters.get("n_clusters", 3)),
            init=parameters.get("init", "k-means++"),
            n_init=int(parameters.get("n_init", 10)),
            max_iter=int(parameters.get("max_iter", 300)),
            tol=float(parameters.get("tol", 0.0001)),
            algorithm=parameters.get("algorithm", "lloyd"),
            random_state=random_state,
        )

    if method == "dbscan":
        return DBSCAN(
            eps=float(parameters.get("eps", 0.5)),
            min_samples=int(parameters.get("min_samples", 5)),
            metric=parameters.get("metric", "euclidean"),
            algorithm=parameters.get("algorithm", "auto"),
            leaf_size=int(parameters.get("leaf_size", 30)),
        )

    if method == "hierarchical":
        return AgglomerativeClustering(
            n_clusters=int(parameters.get("n_clusters", 3)),
            linkage=parameters.get("linkage", "ward"),
            metric=parameters.get("metric", "euclidean"),
            compute_distances=bool(parameters.get("compute_distances", False)),
        )

    raise ValueError(f"Unsupported clustering method: {method}")

def get_supported_models():
    return {
        category: list(definitions.keys())
        for category, definitions in MODEL_CATALOG.items()
    }

def get_model_category():
    categories = {}
    for category, definitions in MODEL_CATALOG.items():
        for algorithm in definitions.keys():
            categories[algorithm] = category
    return categories