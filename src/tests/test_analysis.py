import pandas as pd

from analysis.analysis import get_num_feature_columns, pca_analysis, correlation_analysis, mutual_information_analysis, mrmr_analysis

# These tests were created to test functions in analysis.py

def generate_df():
    df = pd.DataFrame({
        "feature1": [40, 42, 44, 70, 72, 74],
        "feature2": [2, 3, 2, 8, 9, 8],
        "feature3": [75, 77, 76, 95, 98, 97],
        "feature4": [1.1, 1.2, 1.0, 4.2, 4.5, 4.3],
        "label": [1, 1, 1, 0, 0, 0],
    })
    
    return df

def test_get_num_feature_columns():
    df = generate_df()

    features = get_num_feature_columns(df, "label")

    # Make sure the label is filtered out
    assert "label" not in features
    
    # Make sure the rest of the features are kept
    assert "feature1" in features
    assert "feature2" in features
    assert "feature3" in features
    assert "feature4" in features

def test_pca_analysis():
    df = generate_df()
    
    features = get_num_feature_columns(df, "label")

    result = pca_analysis(df=df, features=features, label="label", n_components=2)

    # Check contents of result
    assert "pca_df" in result
    assert "explained_variance_ratio" in result
    assert "explained_variance_sum" in result

    # Check on pca_df: PC1, PC2, label
    assert result["pca_df"].shape[1] == 3 
    
    assert "PC1" in result["pca_df"].columns
    assert "PC2" in result["pca_df"].columns
    assert "label" in result["pca_df"].columns

def test_correlation_analysis():
    df = generate_df()

    corr = correlation_analysis(df, "label")

    # Should result in 4 rows × 4 columns
    assert corr.shape[0] == 4
    assert corr.shape[1] == 4
    
    assert "feature1" in corr.columns
    assert "feature2" in corr.columns
    assert "feature3" in corr.columns
    assert "feature4" in corr.columns

def test_mutual_information_analysis():
    df = generate_df()
    
    features = get_num_feature_columns(df, "label")

    result = mutual_information_analysis(df=df, features=features, label="label")

    assert "feature" in result.columns
    assert "mutual_information" in result.columns
    assert len(result) == len(features)

def test_mrmr_analysis():
    df = generate_df()
    
    features = get_num_feature_columns(df, "label")

    result = mrmr_analysis(df=df, features=features, label="label", K=2)

    assert "rank" in result.columns
    assert "feature" in result.columns
    assert len(result) == 2