import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

try:
    from mrmr import mrmr_classif
except ImportError:  # Allows the backend/tests to run when mrmr is not installed.
    mrmr_classif = None

from src.model.model_utils import (
    get_num_feature_columns,
    prepare_training_features,
)


# Part of TAB 1: PCA and Statistical Analysis
# - PCA Visualization
# - Dimensionality Reduction Tools
# - Feature Importance Analysis (mRMR, Correlation Analysis, Mutual information analysis, etc.)


# PCA Analysis
# Parameters include the dataset received from previous stage: ______,
# feature columns, label column and n_components
def pca_analysis(df, features, label, n_components=2, fill_method="median", fill_value=None):
    X = prepare_training_features(
        df=df,
        features=features,
        fill_method=fill_method,
        fill_value=fill_value,
    )
    y = df[label]  # PCA does not use labels -> keep for later

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])

    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(X_scaled)

    columns = []

    for i in range(n_components):
        columns.append(f"PC{i + 1}")

    pca_df = pd.DataFrame(pca_data, columns=columns)

    # Add labels
    pca_df[label] = y.values

    result = {
        "pca_df": pca_df,
        "features": features,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance_sum": pca.explained_variance_ratio_.sum(),
    }

    # We can modify based on needs
    return result

    # TO DO: Visualize PCA with streamlit / frontend


# Dimensionality Reduction Tools


# Feature Importance Analysis (mRMR, Correlation Analysis, Mutual information analysis, etc.)


# Correlation Analysis (feature to feature correlation)
def correlation_analysis(df, label):
    feature_columns = get_num_feature_columns(df, label)

    if len(feature_columns) == 0:
        raise ValueError("There are no numeric features/ columns.")

    corr_matrix = df[feature_columns].corr()

    return corr_matrix


# Mutual Information Analysis
def mutual_information_analysis(df, features, label, fill_method="median", fill_value=None):
    X = prepare_training_features(
        df=df,
        features=features,
        fill_method=fill_method,
        fill_value=fill_value,
    )
    y = df[label]

    # Calculate the Mutual Information scores
    mi_scores = mutual_info_classif(X, y, random_state=42)

    result = pd.DataFrame({
        "feature": X.columns,
        "mutual_information": mi_scores,
    }).sort_values("mutual_information", ascending=False)

    return result


# mRMR (minimum Redundancy - Maximum Relevance) Analysis
def mrmr_analysis(df, features, label, K=10, fill_method="median", fill_value=None):
    X = prepare_training_features(
        df=df,
        features=features,
        fill_method=fill_method,
        fill_value=fill_value,
    )
    y = df[label]

    # Take the smaller value for number of top features
    K = min(K, len(X.columns))

    if mrmr_classif is not None:
        top_features = mrmr_classif(X=X, y=y, K=K)
    else:
        # Fallback: rank by mutual information when optional mrmr package is unavailable.
        mi_scores = mutual_info_classif(X, y, random_state=42)

        top_features = (
            pd.DataFrame({
                "feature": X.columns,
                "score": mi_scores,
            })
            .sort_values("score", ascending=False)
            .head(K)["feature"]
            .tolist()
        )

    rank = range(1, len(top_features) + 1)

    result = pd.DataFrame({
        "rank": rank,
        "feature": top_features,
    })

    return result