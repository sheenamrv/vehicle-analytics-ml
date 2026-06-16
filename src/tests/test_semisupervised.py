import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from semisupervised.semisupervised import get_num_feature_columns, convert_ssl_labels, train_ssl, get_ssl_training_progress

def generate_df():
    df = pd.DataFrame({
        "feature1": [40, 42, 44, 70, 72, 74],
        "feature2": [2, 3, 2, 8, 9, 8],
        "feature3": [75, 77, 76, 95, 98, 97],
        "feature4": [1.1, 1.2, 1.0, 4.2, 4.5, 4.3],
        "label": [1, 0, 1, None, None, None],
    })
    
    return df

def test_get_num_feature_columns():
    df = generate_df()

    features = get_num_feature_columns(df, "label")

    assert features == ["feature1", "feature2", "feature3", "feature4"]


def test_convert_ssl_labels():
    df = generate_df()

    y = convert_ssl_labels(df, "label")

    assert list(y) == [1, 0, 1, -1, -1, -1]


def test_get_ssl_training_progress():
    train_ssl_result = {"labeled_iter": np.array([0, 0, 1, -1, 0, 2, 1])}

    progress = get_ssl_training_progress(train_ssl_result)

    assert list(progress["iteration"]) == [-1, 0, 1, 2]
    assert list(progress["count"]) == [1, 3, 2, 1]
    assert list(progress["description"]) == ["Never labeled", "Originally labeled", "Pseudo-labeled on iteration 1", "Pseudo-labeled on iteration 2"]

