from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import auc, precision_recall_curve, roc_curve, silhouette_score
from sklearn.model_selection import StratifiedKFold, learning_curve, validation_curve
from sklearn.neighbors import NearestNeighbors

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
    frame = pd.DataFrame({"cluster": counts.index, "count": counts.values})
    frame["description"] = frame["cluster"].apply(lambda value: "Noise" if value == -1 else f"Cluster {value}")
    return frame


def create_clustered_dataframe(df, cluster_labels):
    clustered_df = df.copy()
    clustered_df["cluster"] = cluster_labels
    return clustered_df


def create_pca_cluster_data(X_scaled, cluster_labels, n_components=2):
    max_components = min(X_scaled.shape[0], X_scaled.shape[1])
    if max_components < 1:
        raise ValueError("PCA requires at least one row and one feature.")
    actual_components = min(int(n_components), max_components)
    pca = PCA(n_components=actual_components)
    pca_data = pca.fit_transform(X_scaled)
    pc2 = pca_data[:, 1] if actual_components >= 2 else np.zeros(len(pca_data))
    pca_df = pd.DataFrame({
        "PC1": pca_data[:, 0],
        "PC2": pc2,
        "cluster": np.asarray(cluster_labels).astype(str),
    })
    return {
        "pca_df": pca_df,
        "pca": pca,
        "requested_components": int(n_components),
        "actual_components": actual_components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
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


def _safe_name(value):
    text = str(value or "model").strip().replace(" ", "_")
    return "".join(char for char in text if char.isalnum() or char in {"_", "-"}) or "model"


def _numeric_metrics(metrics):
    output = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, (int, float)):
            output[str(key)] = float(value)
    return output


def _build_metrics_chart(metrics, path):
    if not metrics:
        return False
    labels = list(metrics.keys())
    values = [metrics[label] for label in labels]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values, color="#2f855a")
    plt.ylim(0, max(1.0, max(values) * 1.15))
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_confusion_chart(matrix, labels, path):
    if matrix is None:
        return False
    frame = pd.DataFrame(matrix)
    if frame.empty:
        return False
    if labels and len(labels) == frame.shape[0] == frame.shape[1]:
        frame.index = [str(label) for label in labels]
        frame.columns = [str(label) for label in labels]

    plt.figure(figsize=(6.2, 5.4))
    plt.imshow(frame.values, cmap="Greens")
    plt.title("Confusion Matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(frame.shape[1]), frame.columns, rotation=35, ha="right")
    plt.yticks(range(frame.shape[0]), frame.index)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for row in range(frame.shape[0]):
        for col in range(frame.shape[1]):
            plt.text(col, row, str(frame.iat[row, col]), ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_feature_importance_chart(model, feature_columns, path):
    if model is None or not hasattr(model, "feature_importances_"):
        return False
    values = list(np.array(getattr(model, "feature_importances_", []), dtype=float).ravel())
    if not values:
        return False
    labels = [str(col) for col in (feature_columns or [])]
    if len(labels) != len(values):
        labels = [f"feature_{index + 1}" for index in range(len(values))]

    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)[:15]
    top_labels = [labels[idx] for idx in order]
    top_values = [values[idx] for idx in order]

    plt.figure(figsize=(8, 4.8))
    plt.barh(top_labels[::-1], top_values[::-1], color="#276749")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_coefficients_chart(model, feature_columns, path):
    if model is None or not hasattr(model, "coef_"):
        return False
    coef = getattr(model, "coef_", None)
    if coef is None:
        return False

    coef_values = pd.Series(np.abs(np.array(coef)).mean(axis=0))
    if coef_values.empty:
        return False
    labels = [str(col) for col in (feature_columns or [])]
    if len(labels) != len(coef_values):
        labels = [f"feature_{index + 1}" for index in range(len(coef_values))]
    coef_values.index = labels
    coef_values = coef_values.sort_values(ascending=False).head(15)

    plt.figure(figsize=(8, 4.8))
    plt.barh(list(coef_values.index)[::-1], list(coef_values.values)[::-1], color="#2b6cb0")
    plt.title("Coefficient Magnitude")
    plt.xlabel("|Coefficient|")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_loss_chart(model, path):
    if model is None:
        return False

    if hasattr(model, "loss_curve_"):
        values = list(np.array(getattr(model, "loss_curve_", []), dtype=float).ravel())
        if not values:
            return False
        x_axis = list(range(1, len(values) + 1))
        title = "Training Loss Curve"
        xlabel = "Iteration"
    elif hasattr(model, "loss_") and isinstance(getattr(model, "loss_"), (int, float)):
        values = [float(getattr(model, "loss_"))]
        x_axis = [1]
        title = "Final Loss"
        xlabel = "Step"
    else:
        return False

    plt.figure(figsize=(8, 4.4))
    plt.plot(x_axis, values, marker="o", color="#9f1239")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_cluster_summary_chart(cluster_summary, path):
    if not cluster_summary:
        return False
    frame = pd.DataFrame(cluster_summary)
    if not {"cluster", "count"}.issubset(frame.columns):
        return False

    frame = frame.sort_values("cluster")
    plt.figure(figsize=(7.2, 4.4))
    plt.bar(frame["cluster"].astype(str), frame["count"], color="#1f7a8c")
    plt.title("Cluster Membership")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_cluster_plot(cluster_plot_data, component_count, path):
    """Plot PCA cluster separation or a one-dimensional cluster distribution."""
    if not cluster_plot_data:
        return False

    frame = pd.DataFrame(cluster_plot_data).copy()
    if not {"PC1", "cluster"}.issubset(frame.columns):
        return False

    frame["PC1"] = pd.to_numeric(frame["PC1"], errors="coerce")
    if "PC2" in frame.columns:
        frame["PC2"] = pd.to_numeric(frame["PC2"], errors="coerce")
    frame = frame.dropna(subset=["PC1", "cluster"])
    if frame.empty:
        return False

    clusters = list(pd.unique(frame["cluster"].astype(str)))
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(7.6, 5.0))

    if int(component_count or 2) >= 2 and "PC2" in frame.columns:
        frame = frame.dropna(subset=["PC2"])
        if frame.empty:
            plt.close()
            return False
        for index, cluster in enumerate(clusters):
            rows = frame[frame["cluster"].astype(str) == cluster]
            label = "Noise" if cluster == "-1" else f"Cluster {cluster}"
            plt.scatter(
                rows["PC1"],
                rows["PC2"],
                label=label,
                alpha=0.85,
                color=cmap(index % 10),
            )
        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        plt.title("Cluster Separation (PCA)")
    else:
        # When only one component is possible, show each cluster along PC1
        for index, cluster in enumerate(clusters):
            rows = frame[frame["cluster"].astype(str) == cluster]
            y = np.full(len(rows), index, dtype=float)
            label = "Noise" if cluster == "-1" else f"Cluster {cluster}"
            plt.scatter(
                rows["PC1"],
                y,
                label=label,
                alpha=0.85,
                color=cmap(index % 10),
            )
        plt.yticks(range(len(clusters)), [
            "Noise" if value == "-1" else f"Cluster {value}"
            for value in clusters
        ])
        plt.xlabel("Principal component 1")
        plt.ylabel("Cluster")
        plt.title("One-Dimensional Cluster Distribution")

    plt.grid(alpha=0.22)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_ssl_progress_chart(progress_rows, path):
    if not progress_rows:
        return False
    frame = pd.DataFrame(progress_rows)
    label_column = "description" if "description" in frame.columns else "status"
    if label_column not in frame.columns or "percentage" not in frame.columns:
        return False

    frame["percentage"] = pd.to_numeric(frame["percentage"], errors="coerce")
    frame = frame.dropna(subset=["percentage"])
    if frame.empty:
        return False
    plt.figure(figsize=(8, 4.8))
    plt.barh(frame[label_column].astype(str), frame["percentage"], color="#7c3aed")
    plt.title("Self-Training Progress")
    plt.xlabel("Percentage")
    plt.xlim(0, 100)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _normalize_ssl_iteration_frame(progress_rows):
    """Return a numeric, consistently named SSL iteration progress frame."""
    if not progress_rows:
        return pd.DataFrame()

    frame = pd.DataFrame(progress_rows).copy()
    if "iteration" not in frame.columns:
        return pd.DataFrame()

    aliases = {
        "pseudo_labeled_percentage": [
            "pseudo_labeled_percentage",
            "cumulative_pseudo_labeled_percentage",
            "percentage",
        ],
        "remaining_unlabeled_percentage": [
            "remaining_unlabeled_percentage",
            "remaining_percentage",
        ],
        "cumulative_pseudo_labeled": [
            "cumulative_pseudo_labeled",
            "pseudo_labeled_total",
        ],
    }
    for target, candidates in aliases.items():
        if target not in frame.columns:
            for candidate in candidates:
                if candidate in frame.columns:
                    frame[target] = frame[candidate]
                    break

    numeric_columns = [
        "iteration",
        "newly_pseudo_labeled",
        "cumulative_pseudo_labeled",
        "remaining_unlabeled",
        "pseudo_labeled_percentage",
        "remaining_unlabeled_percentage",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["iteration"]).sort_values("iteration")
    return frame.reset_index(drop=True)


def _build_ssl_iteration_progress_chart(progress_rows, path):
    """Plot cumulative pseudo-labeling and remaining percentages by iteration."""
    frame = _normalize_ssl_iteration_frame(progress_rows)
    if frame.empty or "pseudo_labeled_percentage" not in frame.columns:
        return False

    frame = frame.dropna(subset=["pseudo_labeled_percentage"])
    if frame.empty:
        return False

    labels = frame.get("description", frame["iteration"].astype(int).astype(str)).astype(str)
    plt.figure(figsize=(8.4, 4.8))
    plt.plot(
        labels,
        frame["pseudo_labeled_percentage"],
        marker="o",
        color="#2563eb",
        label="Pseudo-labeled",
    )
    if "remaining_unlabeled_percentage" in frame.columns:
        remaining = frame["remaining_unlabeled_percentage"]
        if remaining.notna().any():
            plt.plot(
                labels,
                remaining,
                marker="o",
                color="#dc2626",
                label="Remaining unlabeled",
            )

    plt.title("Self-Training Progress by Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Percentage of originally unlabeled rows")
    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_ssl_iteration_table(progress_rows, path):
    """Render the complete SSL iteration history as a report table image."""
    frame = _normalize_ssl_iteration_frame(progress_rows)
    if frame.empty:
        return False

    ordered = [
        "iteration",
        "description",
        "newly_pseudo_labeled",
        "cumulative_pseudo_labeled",
        "remaining_unlabeled",
        "pseudo_labeled_percentage",
        "remaining_unlabeled_percentage",
    ]
    columns = [column for column in ordered if column in frame.columns]
    display = frame[columns].copy()
    display = display.rename(columns={
        "iteration": "Iteration",
        "description": "Description",
        "newly_pseudo_labeled": "New",
        "cumulative_pseudo_labeled": "Cumulative",
        "remaining_unlabeled": "Remaining",
        "pseudo_labeled_percentage": "Pseudo-labeled %",
        "remaining_unlabeled_percentage": "Remaining %",
    })
    for column in ["Pseudo-labeled %", "Remaining %"]:
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.1f}%"
            )
    for column in ["Iteration", "New", "Cumulative", "Remaining"]:
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(int(value))
            )

    width = max(9.0, 1.35 * len(display.columns))
    height = max(2.8, 0.48 * (len(display) + 2))
    fig, axis = plt.subplots(figsize=(width, height))
    axis.axis("off")
    table = axis.table(
        cellText=display.astype(str).values,
        colLabels=list(display.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    axis.set_title("SSL Iteration Progress Details", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _clean_binary_chart_data(y_true, y_score):
    """Clean labels and scores used by binary classification report charts."""
    if y_true is None or y_score is None:
        return None, None, None

    true_series = pd.Series(list(y_true)).reset_index(drop=True)
    score_array = np.asarray(y_score)

    if score_array.ndim == 0 or len(true_series) != len(score_array):
        return None, None, None

    valid_mask = true_series.notna().to_numpy()
    if not valid_mask.any():
        return None, None, None

    cleaned_true = (
        true_series.loc[valid_mask]
        .astype(str)
        .reset_index(drop=True)
        .to_numpy()
    )
    cleaned_score = score_array[valid_mask]

    labels = np.unique(cleaned_true)
    if len(labels) != 2:
        return None, None, None

    if cleaned_score.ndim > 1:
        if cleaned_score.shape[1] < 2:
            return None, None, None
        cleaned_score = cleaned_score[:, -1]

    cleaned_score = pd.to_numeric(
        pd.Series(cleaned_score), errors="coerce"
    ).to_numpy(dtype=float)
    score_mask = np.isfinite(cleaned_score)
    cleaned_true = cleaned_true[score_mask]
    cleaned_score = cleaned_score[score_mask]

    if len(cleaned_true) == 0 or np.unique(cleaned_true).size != 2:
        return None, None, None

    positive_label = labels[-1]
    binary_true = (cleaned_true == positive_label).astype(int)
    return binary_true, cleaned_score, positive_label


def _build_roc_curve_chart(y_true, y_score, path):
    binary_true, score, positive_label = _clean_binary_chart_data(y_true, y_score)
    if binary_true is None:
        return False

    try:
        fpr, tpr, _ = roc_curve(binary_true, score)
        roc_auc = auc(fpr, tpr)
    except (TypeError, ValueError):
        return False

    plt.figure(figsize=(6.8, 5.2))
    plt.plot(fpr, tpr, color="#1f7a8c", label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="#9ca3af")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (positive class: {positive_label})")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_pr_curve_chart(y_true, y_score, path):
    binary_true, score, positive_label = _clean_binary_chart_data(y_true, y_score)
    if binary_true is None:
        return False

    try:
        precision, recall, _ = precision_recall_curve(binary_true, score)
        pr_auc = auc(recall, precision)
    except (TypeError, ValueError):
        return False

    plt.figure(figsize=(6.8, 5.2))
    plt.plot(recall, precision, color="#7c3aed", label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (positive class: {positive_label})")
    plt.legend(loc="lower left", frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_learning_curve_chart(model, X, y, path):
    if model is None or X is None or y is None:
        return False
    X = np.array(X)
    y = np.array(y)
    if X.ndim != 2 or X.shape[0] < 20:
        return False

    train_sizes, train_scores, valid_scores = learning_curve(
        clone(model),
        X,
        y,
        cv=3,
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=1,
    )
    train_mean = np.mean(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)

    plt.figure(figsize=(7.4, 5.0))
    plt.plot(train_sizes, train_mean, marker="o", color="#16a34a", label="Train")
    plt.plot(train_sizes, valid_mean, marker="o", color="#2563eb", label="Validation")
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_validation_curve_chart(model, X, y, path, param_name, param_range):
    if model is None or X is None or y is None or not param_name or not param_range:
        return False
    X = np.array(X)
    y = np.array(y)
    if X.ndim != 2 or len(param_range) < 2:
        return False
    if param_name not in clone(model).get_params():
        return False

    train_scores, valid_scores = validation_curve(
        clone(model),
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=3,
        scoring="accuracy",
        n_jobs=1,
    )

    train_mean = np.mean(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    x_ticks = [str(value) for value in param_range]

    plt.figure(figsize=(7.4, 5.0))
    plt.plot(x_ticks, train_mean, marker="o", color="#0f766e", label="Train")
    plt.plot(x_ticks, valid_mean, marker="o", color="#b91c1c", label="Validation")
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.title(f"Validation Curve ({param_name})")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_accuracy_vs_k_chart(X, y, path):
    if X is None or y is None:
        return False
    X = np.array(X)
    y = np.array(y)
    if X.ndim != 2 or X.shape[0] < 20:
        return False

    max_k = min(20, max(3, X.shape[0] // 5))
    ks = list(range(1, max_k + 1))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    from sklearn.neighbors import KNeighborsClassifier
    for k in ks:
        fold_scores = []
        for train_idx, test_idx in cv.split(X, y):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X[train_idx], y[train_idx])
            fold_scores.append(model.score(X[test_idx], y[test_idx]))
        scores.append(float(np.mean(fold_scores)))

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(ks, scores, marker="o", color="#1d4ed8")
    plt.xlabel("K")
    plt.ylabel("Cross-Validation Accuracy")
    plt.title("Accuracy vs K")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_calibration_curve_chart(y_true, y_score, path):
    binary_true, score, positive_label = _clean_binary_chart_data(y_true, y_score)
    if binary_true is None:
        return False

    try:
        frac_pos, mean_pred = calibration_curve(binary_true, score, n_bins=10)
    except (TypeError, ValueError):
        return False

    plt.figure(figsize=(6.8, 5.2))
    plt.plot(mean_pred, frac_pos, marker="o", color="#0f766e")
    plt.plot([0, 1], [0, 1], "--", color="#9ca3af")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve (positive class: {positive_label})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_cluster_scatter_chart(X, cluster_labels, path):
    if X is None or cluster_labels is None:
        return False
    X = np.array(X)
    labels = np.array(cluster_labels)
    if X.ndim != 2 or X.shape[0] != labels.shape[0]:
        return False

    if X.shape[1] > 2:
        points = PCA(n_components=2).fit_transform(X)
    elif X.shape[1] == 2:
        points = X
    else:
        return False

    plt.figure(figsize=(7.0, 5.4))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="viridis", s=38, alpha=0.85)
    plt.colorbar(scatter, fraction=0.046, pad=0.04)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Cluster Scatter Plot")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_elbow_curve_chart(X, path):
    if X is None:
        return False
    X = np.array(X)
    if X.ndim != 2 or X.shape[0] < 10:
        return False

    max_k = min(10, max(3, X.shape[0] // 10))
    ks = list(range(2, max_k + 1))
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertias.append(float(model.inertia_))

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(ks, inertias, marker="o", color="#0f766e")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_silhouette_vs_k_chart(X, path):
    if X is None:
        return False
    X = np.array(X)
    if X.ndim != 2 or X.shape[0] < 20:
        return False

    max_k = min(10, max(3, X.shape[0] // 10))
    ks = list(range(2, max_k + 1))
    scores = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        scores.append(float(silhouette_score(X, labels)))

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(ks, scores, marker="o", color="#334155")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs Number of Clusters")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_k_distance_plot(X, path):
    if X is None:
        return False
    X = np.array(X)
    if X.ndim != 2 or X.shape[0] < 10:
        return False

    neighbors = min(5, max(2, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=neighbors)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(k_distances, color="#dc2626")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{neighbors}-NN Distance")
    plt.title("k-Distance Plot")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_dendrogram_chart(X, path):
    if X is None:
        return False
    X = np.array(X)
    if X.ndim != 2 or X.shape[0] < 4:
        return False

    max_points = min(250, X.shape[0])
    sample = X[:max_points]
    linkage_matrix = linkage(sample, method="ward")

    plt.figure(figsize=(9.0, 5.2))
    dendrogram(linkage_matrix, no_labels=True, color_threshold=None)
    plt.title("Hierarchical Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_linkage_distance_chart(X, path):
    if X is None:
        return False
    X = np.array(X)
    if X.ndim != 2 or X.shape[0] < 4:
        return False

    max_points = min(250, X.shape[0])
    sample = X[:max_points]
    linkage_matrix = linkage(sample, method="ward")
    distances = linkage_matrix[:, 2]

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(range(1, len(distances) + 1), distances, color="#0f766e")
    plt.xlabel("Merge Step")
    plt.ylabel("Linkage Distance")
    plt.title("Linkage Distance Plot")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_decision_boundary_chart(model, X, y, path):
    if model is None or X is None or y is None:
        return False
    X = np.array(X)
    y = np.array(y)
    if X.ndim != 2 or X.shape[1] != 2:
        return False
    try:
        estimator = clone(model)
        estimator.fit(X, y)
    except Exception:
        return False

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = estimator.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7.0, 5.4))
    plt.contourf(xx, yy, zz, alpha=0.22, cmap="viridis")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap="viridis", edgecolors="k", linewidths=0.2)
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _build_support_vectors_chart(model, X, y, path):
    if model is None or X is None or y is None or not hasattr(model, "support_vectors_"):
        return False
    X = np.array(X)
    y = np.array(y)
    if X.ndim != 2 or X.shape[1] != 2:
        return False

    support_vectors = np.array(getattr(model, "support_vectors_", []))
    if support_vectors.size == 0:
        return False

    plt.figure(figsize=(7.0, 5.4))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=26, alpha=0.75)
    plt.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=90,
        facecolors="none",
        edgecolors="#dc2626",
        linewidths=1.2,
        label="Support Vectors",
    )
    plt.title("Support Vector Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _write_pdf_from_images(images, pdf_path, title):
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(pdf_path) as pdf:
        for image in images:
            figure = plt.figure(figsize=(11, 8.5))
            axis = figure.add_subplot(111)
            axis.imshow(plt.imread(str(image)))
            axis.set_title(title)
            axis.axis("off")
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)


def render_confusion_matrix_image(matrix, labels, path, title="Confusion Matrix"):
    """Public helper for rendering confusion matrices outside report export."""
    if matrix is None:
        return False
    frame = pd.DataFrame(matrix)
    if frame.empty:
        return False
    if labels and len(labels) == frame.shape[0] == frame.shape[1]:
        frame.index = [str(label) for label in labels]
        frame.columns = [str(label) for label in labels]

    plt.figure(figsize=(6.4, 5.4))
    plt.imshow(frame.values, cmap="Greens")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(frame.shape[1]), frame.columns, rotation=30, ha="right")
    plt.yticks(range(frame.shape[0]), frame.index)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for row in range(frame.shape[0]):
        for col in range(frame.shape[1]):
            plt.text(col, row, str(frame.iat[row, col]), ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def render_comparison_metrics_image(records, path):
    """Render a single comparison metrics image with a centered legend."""
    rows = []
    for record in records or []:
        metrics = _numeric_metrics(record.get("metrics", {}))
        rows.append({
            "name": record.get("name", ""),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
        })

    frame = pd.DataFrame(rows)
    required = {"name", "accuracy", "precision", "recall", "f1"}
    if frame.empty or not required.issubset(frame.columns):
        return False

    plot_df = frame[["name", "accuracy", "precision", "recall", "f1"]].melt(
        id_vars="name",
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    ).dropna()
    if plot_df.empty:
        return False

    plt.figure(figsize=(max(8.0, 1.2 * len(frame)), 5.2))
    unique_names = plot_df["name"].unique().tolist()
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(unique_names))
    width = 0.18
    palette = {
        "accuracy": "#1f7a8c",
        "precision": "#2b9348",
        "recall": "#f77f00",
        "f1": "#8338ec",
    }
    for idx, metric in enumerate(metrics):
        values = []
        for name in unique_names:
            entry = plot_df[(plot_df["name"] == name) & (plot_df["metric"] == metric)]
            values.append(float(entry.iloc[0]["score"]) if not entry.empty else np.nan)
        offset = (idx - 1.5) * width
        plt.bar(x + offset, values, width=width, label=metric.title(), color=palette[metric])

    plt.xticks(x, unique_names, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Comparison Metrics")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()
    return True


def render_combined_confusion_matrices_image(records, path):
    """Render all selected model confusion matrices into one combined image."""
    matrices = []
    for record in records or []:
        evaluation = record.get("evaluation", {}) or {}
        matrix = evaluation.get("confusion_matrix")
        if matrix is None:
            continue
        frame = pd.DataFrame(matrix)
        if frame.empty:
            continue
        labels = evaluation.get("confusion_labels")
        if labels and len(labels) == frame.shape[0] == frame.shape[1]:
            frame.index = [str(label) for label in labels]
            frame.columns = [str(label) for label in labels]
        matrices.append((record.get("name", "model"), frame))

    if not matrices:
        return False

    columns = 2 if len(matrices) > 1 else 1
    rows = int(np.ceil(len(matrices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.5 * columns, 5.5 * rows))
    axes_array = np.array(axes).reshape(-1)

    for axis, (name, frame) in zip(axes_array, matrices):
        image = axis.imshow(frame.values, cmap="Greens")
        axis.set_title(f"CM - {name}")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks(range(frame.shape[1]))
        axis.set_yticks(range(frame.shape[0]))
        axis.set_xticklabels(frame.columns, rotation=30, ha="right")
        axis.set_yticklabels(frame.index)
        for row in range(frame.shape[0]):
            for col in range(frame.shape[1]):
                axis.text(col, row, str(frame.iat[row, col]), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for axis in axes_array[len(matrices):]:
        axis.axis("off")

    fig.suptitle("Combined Confusion Matrices", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return True


def generate_model_report_assets(model_record, output_root, include_pdf=True):
    """Generate report images and optional PDF from model context.

    This function keeps type-based chart additions centralized so future model
    types only need to register new builders in one place.
    """
    output_root = Path(output_root)
    model_name = _safe_name(model_record.get("name") or model_record.get("display_name"))
    report_dir = output_root / model_name
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics = _numeric_metrics(model_record.get("metrics", {}))
    confusion_matrix = model_record.get("confusion_matrix")
    confusion_labels = model_record.get("confusion_labels")
    cluster_summary = model_record.get("cluster_summary")
    cluster_plot_data = model_record.get("cluster_plot_data")
    cluster_plot_components = model_record.get("cluster_plot_components")
    ssl_progress = model_record.get("ssl_progress")
    ssl_iteration_progress = model_record.get("ssl_iteration_progress")
    model_obj = model_record.get("model")
    feature_columns = model_record.get("feature_columns", [])
    category = str(model_record.get("category", ""))
    algorithm = str(model_record.get("algorithm", ""))

    X = model_record.get("X")
    y = model_record.get("y")
    y_true = model_record.get("y_true")
    y_pred = model_record.get("y_pred")
    y_score = model_record.get("y_score")
    correlated = bool(model_record.get("is_correlated", False))

    image_paths = []

    base_builders = [
        ("metrics.png", lambda p: _build_metrics_chart(metrics, p)),
        ("confusion_matrix.png", lambda p: _build_confusion_chart(confusion_matrix, confusion_labels, p)),
        # These model-type summaries do not require supervised correlation data.
        ("cluster_summary.png", lambda p: _build_cluster_summary_chart(cluster_summary, p)),
        (
            "cluster_plot.png",
            lambda p: _build_cluster_plot(
                cluster_plot_data, cluster_plot_components, p
            ),
        ),
        ("ssl_progress.png", lambda p: _build_ssl_progress_chart(ssl_progress, p)),
        (
            "ssl_iteration_progress.png",
            lambda p: _build_ssl_iteration_progress_chart(
                ssl_iteration_progress, p
            ),
        ),
        (
            "ssl_iteration_table.png",
            lambda p: _build_ssl_iteration_table(
                ssl_iteration_progress, p
            ),
        ),
    ]
    for filename, builder in base_builders:
        candidate = report_dir / filename
        try:
            created = builder(candidate)
        except Exception as error:
            print(f"Skipped report chart {filename}: {error}")
            created = False
        if created:
            image_paths.append(candidate)

    if not correlated:
        if not image_paths:
            return report_dir, None, []
        pdf_path = None
        if include_pdf:
            pdf_path = report_dir / f"{model_name}_report.pdf"
            _write_pdf_from_images(image_paths, pdf_path, model_name)
        return report_dir, pdf_path, image_paths

    generic_builders = [
        ("feature_importance.png", lambda p: _build_feature_importance_chart(model_obj, feature_columns, p)),
        ("coefficient_magnitude.png", lambda p: _build_coefficients_chart(model_obj, feature_columns, p)),
        ("loss.png", lambda p: _build_loss_chart(model_obj, p)),
        ("roc_curve.png", lambda p: _build_roc_curve_chart(y_true, y_score, p)),
        ("precision_recall_curve.png", lambda p: _build_pr_curve_chart(y_true, y_score, p)),
        ("learning_curve.png", lambda p: _build_learning_curve_chart(model_obj, X, y, p)),
        ("decision_boundary.png", lambda p: _build_decision_boundary_chart(model_obj, X, y, p)),
    ]

    for filename, builder in generic_builders:
        candidate = report_dir / filename
        try:
            created = builder(candidate)
        except Exception as error:
            print(f"Skipped report chart {filename}: {error}")
            created = False
        if created:
            image_paths.append(candidate)

    validation_param_map = {
        "svm": ("C", [0.1, 0.5, 1.0, 5.0, 10.0]),
        "knn": ("n_neighbors", [1, 3, 5, 7, 9, 11]),
        "decision_tree": ("max_depth", [2, 4, 6, 8, 10]),
        "random_forest": ("n_estimators", [20, 50, 100, 200]),
        "logistic_regression": ("C", [0.1, 0.5, 1.0, 5.0, 10.0]),
    }
    if algorithm in validation_param_map:
        param_name, param_values = validation_param_map[algorithm]
        candidate = report_dir / "validation_curve.png"
        if _build_validation_curve_chart(model_obj, X, y, candidate, param_name, param_values):
            image_paths.append(candidate)

    algorithm_specific = {
        "knn": [
            ("accuracy_vs_k.png", lambda p: _build_accuracy_vs_k_chart(X, y, p)),
        ],
        "logistic_regression": [
            ("calibration_curve.png", lambda p: _build_calibration_curve_chart(y_true, y_score, p)),
        ],
        "svm": [
            ("support_vectors.png", lambda p: _build_support_vectors_chart(model_obj, X, y, p)),
        ],
        "kmeans": [
            ("elbow_curve.png", lambda p: _build_elbow_curve_chart(X, p)),
            ("silhouette_vs_k.png", lambda p: _build_silhouette_vs_k_chart(X, p)),
        ],
        "dbscan": [
            ("k_distance_plot.png", lambda p: _build_k_distance_plot(X, p)),
        ],
        "hierarchical": [
            ("dendrogram.png", lambda p: _build_dendrogram_chart(X, p)),
            ("linkage_distance.png", lambda p: _build_linkage_distance_chart(X, p)),
        ],
    }
    for filename, builder in algorithm_specific.get(algorithm, []):
        candidate = report_dir / filename
        if builder(candidate):
            image_paths.append(candidate)

    if algorithm == "decision_tree" and hasattr(model_obj, "tree_"):
        tree_path = report_dir / "decision_tree_structure.png"
        try:
            from sklearn import tree as sklearn_tree

            plt.figure(figsize=(12, 7))
            sklearn_tree.plot_tree(model_obj, max_depth=3, fontsize=7, filled=True)
            plt.title("Decision Tree (Depth <= 3)")
            plt.tight_layout()
            plt.savefig(tree_path, dpi=160)
            plt.close()
            image_paths.append(tree_path)
        except Exception:
            pass

    if category == "unsupervised" or algorithm in {"kmeans", "dbscan", "hierarchical"}:
        scatter_path = report_dir / "cluster_scatter.png"
        if _build_cluster_scatter_chart(X, y_pred if y_pred is not None else model_record.get("cluster_labels"), scatter_path):
            image_paths.append(scatter_path)

    if not image_paths:
        return report_dir, None, []

    pdf_path = None
    if include_pdf:
        pdf_path = report_dir / f"{model_name}_report.pdf"
        _write_pdf_from_images(image_paths, pdf_path, model_name)
    return report_dir, pdf_path, image_paths


def export_model_report(model_record, output_root):
    """Export model charts as PNG plus a combined PDF in a model-specific folder."""
    return generate_model_report_assets(model_record, output_root, include_pdf=True)