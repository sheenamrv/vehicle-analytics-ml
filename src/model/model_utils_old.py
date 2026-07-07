import pandas as pd

def validate_dataset(df, label_col):
    
    if df.empty:
        return False, "Dataset is empty."
    
    if label_col not in df.columns:
        return False, "Label column does not exist"
    
    if df[label_col].isna().any():
        return(False, "Label column contains missing values.")
    
    return True, ""

def prepare_training_data(df,label_col):
    
    X = df.drop(columns=[label_col]).copy()
    y = df[label_col].copy()
    
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns
    
    if len(cat_cols) > 0:
        print("One Hot encoding: ")
        
        for col in cat_cols:
            print(f"_ {col}")
        
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
        
    return X, y

def select_model():
    
    print("\nAvailable Models")
    print("1. SVM")
    print("2. KNN")
    print("3. Decision Tree")
    print("4. Random Forest")
    print("5. Logistic Regression")

    choice = input(
        "\nSelect model: "
    ).strip()

    return {
        "1": "svm",
        "2": "knn",
        "3": "decision_tree",
        "4": "random_forest",
        "5": "logistic_regression"
    }.get(choice)
    
def get_training_config():
    
    print("\n Training Configuration")
    
    test_size = input("Test split (0.3): ").strip()
    
    random_state = input("Random state (42): ").strip()
    
    config = {
        "test_size" : float(test_size) if test_size else 0.3,
        "random_state" : int(random_state) if random_state else 42
    }
    
    return config

def get_rf_config():
    
    trees = input("Number of trees [100]: ").strip()
    
    max_depth = input("Max depth [None]: ").strip()
    
    return {
        "n_estimators": int(trees) if trees else 100,
        "max_depth" : int(max_depth) if max_depth else None
    }
    
# Returns list of names of all numeric columns except for the label/target column
def get_num_feature_columns(df, label_col=None):
    features = df.select_dtypes(include="number").columns.tolist()

    if label_col and label_col in features:
        features.remove(label_col)

    return features

# Should probably have options for this : median, mode, mean
def prepare_training_features(df, features):
    return df[features].fillna(df[features].median())