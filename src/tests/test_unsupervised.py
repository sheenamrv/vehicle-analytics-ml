import pandas as pd

from src.model.unsupervised_model import (
    run_unsupervised_workflow,
    create_cluster_summary,
)


def generate_df():
    return pd.DataFrame({
        "feature1": [1, 1.1, 1.2, 5, 5.1, 5.2],
        "feature2": [1, 1.1, 1.0, 5, 5.1, 5.0],
        "label": [0, 0, 0, 1, 1, 1],
    })


def test_run_kmeans_workflow():
    df = generate_df()

    result = run_unsupervised_workflow(
        df=df,
        method="kmeans",
        label="label",
        n_clusters=2,
    )

    assert result["method"] == "kmeans"
    assert "cluster_labels" in result
    assert "summary_df" in result
    assert "metrics" in result
    assert "pca_result" in result
    assert len(result["cluster_labels"]) == len(df)


def test_run_hierarchical_workflow():
    df = generate_df()

    result = run_unsupervised_workflow(
        df=df,
        method="hierarchical",
        label="label",
        n_clusters=2,
    )

    assert result["method"] == "hierarchical"
    assert len(result["cluster_labels"]) == len(df)


def test_run_dbscan_workflow():
    df = generate_df()

    result = run_unsupervised_workflow(
        df=df,
        method="dbscan",
        label="label",
        eps=0.8,
        min_samples=2,
    )

    assert result["method"] == "dbscan"
    assert len(result["cluster_labels"]) == len(df)


def test_create_cluster_summary():
    labels = [0, 0, 1, 1, 1]

    summary = create_cluster_summary(labels)

    assert list(summary.columns) == ["cluster", "count"]
    assert summary["count"].sum() == 5