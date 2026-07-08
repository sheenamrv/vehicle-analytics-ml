import pytest
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.model.model_factory import build_clusterer, build_model


@pytest.mark.parametrize(
    "model_type, expected_class",
    [
        ("svm", SVC),
        ("knn", KNeighborsClassifier),
        ("decision_tree", DecisionTreeClassifier),
        ("random_forest", RandomForestClassifier),
        ("logistic_regression", LogisticRegression),
    ],
)
def test_build_model_supported_types(model_type, expected_class):
    model = build_model(model_type, parameters={}, random_state=42)

    assert isinstance(model, expected_class)


def test_build_model_applies_parameters():
    model = build_model(
        "random_forest",
        parameters={"n_estimators": 25, "max_depth": 3},
        random_state=7,
    )

    assert model.n_estimators == 25
    assert model.max_depth == 3
    assert model.random_state == 7


def test_build_model_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported supervised"):
        build_model("not_a_model")


@pytest.mark.parametrize(
    "method, expected_class",
    [
        ("kmeans", KMeans),
        ("dbscan", DBSCAN),
        ("hierarchical", AgglomerativeClustering),
    ],
)
def test_build_clusterer_supported_types(method, expected_class):
    model = build_clusterer(method, parameters={"n_clusters": 2}, random_state=42)

    assert isinstance(model, expected_class)


def test_build_clusterer_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported clustering"):
        build_clusterer("not_a_clusterer")
