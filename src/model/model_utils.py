import pandas as pd
from sklearn.preprocessing import StandardScaler


def validate_dataset(df, label_col):
    if df is None or df.empty:
        return False, "Dataset is empty."
    if label_col not in df.columns:
        return False, "Label column does not exist."
    if df[label_col].isna().any():
        return False, "Label column contains missing values."
    return True, ""

# Returns list of names of all numeric columns except for the label/target column
def get_num_feature_columns(df, label_col=None):
    
    features = df.select_dtypes(include="number").columns.tolist()
    if label_col and label_col in features:
        features.remove(label_col)
    return features

# Returns list of names of all cateforical/string columns except for the label/target column
def get_categorical_feature_columns(df, label_col=None):
    features = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if label_col and label_col in features:
        features.remove(label_col)
    return features
# do not alter
# Prepare features for modeling (fill numeric missing values, categorical columns are one-hot encoded)
def prepare_training_features(df, features, fill_method="median", fill_value=None):
  
    if features is None:
        raise ValueError("features must be provided.")

    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}")

    X = df[list(features)].copy()

    numeric_cols = X.select_dtypes(include="number").columns
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns

    if len(numeric_cols) > 0:
        if fill_method == "median":
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        elif fill_method == "mean":
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        elif fill_method == "mode":
            for col in numeric_cols:
                mode = X[col].mode(dropna=True)
                if not mode.empty:
                    X[col] = X[col].fillna(mode.iloc[0])
        elif fill_method == "constant":
            X[numeric_cols] = X[numeric_cols].fillna(fill_value)
        elif fill_method in (None, "none"):
            pass
        else:
            raise ValueError(f"Unsupported fill_method: {fill_method}")

    if len(categorical_cols) > 0:
        X[categorical_cols] = X[categorical_cols].fillna("__missing__")
        X = pd.get_dummies(X, columns=list(categorical_cols), dummy_na=False)

    return X
# do not alter
#  Prepare X and y for training
def prepare_training_data(df, label_col, features=None, fill_method="median", fill_value=None):
    valid, message = validate_dataset(df, label_col)
    if not valid:
        raise ValueError(message)

    if features is None:
        features = [col for col in df.columns if col != label_col]

    X = prepare_training_features(
        df=df,
        features=features,
        fill_method=fill_method,
        fill_value=fill_value,
    )
    y = df[label_col].copy()
    return X, y

#  Align a feature matrixx to columns used during training
def align_features(X, expected_columns, fill_value=0):
    """Align a feature matrix to columns used during training."""
    return X.reindex(columns=list(expected_columns), fill_value=fill_value)

# Scale features with StandardScaler and return scaled data plus scaler
def scale_features(X, scaler=None):
    scaler = scaler or StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def get_common_training_config():
    return None

def merge_training_config():
    return None


# Old functions
def select_model():
    print("\nAvailable Models")
    print("1. SVM")
    print("2. KNN")
    print("3. Decision Tree")
    print("4. Random Forest")
    print("5. Logistic Regression")
    choice = input("\nSelect model: ").strip()
    return {
        "1": "svm",
        "2": "knn",
        "3": "decision_tree",
        "4": "random_forest",
        "5": "logistic_regression",
    }.get(choice)


def get_training_config(test_size=None, random_state=None):
    if test_size is None and random_state is None:
        print("\n Training Configuration")
        test_size = input("Test split (0.3): ").strip()
        random_state = input("Random state (42): ").strip()
        return {
            "test_size": float(test_size) if test_size else 0.3,
            "random_state": int(random_state) if random_state else 42,
        }

    return {
        "test_size": float(test_size) if test_size is not None else 0.3,
        "random_state": int(random_state) if random_state is not None else 42,
    }


def get_rf_config(n_estimators=None, max_depth=None):
    if n_estimators is None and max_depth is None:
        trees = input("Number of trees [100]: ").strip()
        depth = input("Max depth [None]: ").strip()
        return {
            "n_estimators": int(trees) if trees else 100,
            "max_depth": int(depth) if depth else None,
        }

    return {
        "n_estimators": int(n_estimators) if n_estimators is not None else 100,
        "max_depth": int(max_depth) if max_depth not in (None, "") else None,
    }
