from sklearn.model_selection import train_test_split

from src.evaluation.metrics import (
    classification_confusion_matrix,
    classification_metrics,
    clustering_metrics,
)
from src.model.model_registry import add_model, select_saved_models
from src.model.model_utils import (
    align_features,
    get_training_config,
    prepare_training_data,
    select_model,
    validate_dataset,
)

# Contains the small shared functions used for supervised, semisupervised, and unsupervised worflows

def split_train_test(X, y, test_size=0.3, random_state=42, stratify=False):
    stratify_values = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )

# do not alter
def fit_model(model, X, y=None):
    if y is None:
        model.fit(X)
    else:
        model.fit(X, y)
    return model

# do not alter
def predict_model(model, X):
    return model.predict(X)

# do not alter
def fit_predict_model(model, X):
    if hasattr(model, "fit_predict"):
        return model.fit_predict(X)

    fit_model(model, X)
    if hasattr(model, "labels_"):
        return model.labels_

    return predict_model(model, X)

#do not alter
def train_test_classifier(
    model,
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=False,
    ):

    X_train, X_test, y_train, y_test = split_train_test(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    
    fit_model(model, X_train, y_train)
    
    predictions = predict_model(model, X_test)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
    }


def evaluate_classifier_predictions(y_true, y_pred):
    return {
        "metrics": classification_metrics(y_true, y_pred),
        "confusion_matrix": classification_confusion_matrix(y_true, y_pred),
        "predictions": y_pred,
    }


def evaluate_cluster_labels(X, cluster_labels):
    return clustering_metrics(X, cluster_labels)

# Evaluate selected saved models on a df and return results
#safely skips clustering artifacts
def evaluate_saved_models(models, df, label_col, fill_method="median", fill_value=None):
    
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")
    if df[label_col].isna().any():
        raise ValueError("Saved-model evaluation requires known labels for every evaluated row.")

    results = []
    for model_info in models:
        category = model_info.get("category", "supervised")
        if category == "unsupervised":
            results.append({
                "name": model_info.get("display_name", "Unnamed model"),
                "category": category,
                "metrics": model_info.get("metrics", {}),
                "confusion_matrix": None,
                "predictions": None,
                "note": "Clustering models are not evaluated as classifiers on labeled test data.",
            })
            continue

        features = model_info.get("feature_columns") or [column for column in df.columns if column != label_col]
        missing = [column for column in features if column not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found for {model_info.get('display_name')}: {missing}")

        model = model_info["model"]
        # SSL bundles own their fitted imputer/scaler; ordinary models use shared preparation.
        if category == "semi_supervised" or hasattr(model, "features"):
            X_test = df[features].copy()
        else:
            X, _ = prepare_training_data(
                df=df, label_col=label_col, features=features,
                fill_method=fill_method, fill_value=fill_value,
            )
            X_test = align_features(X, features, fill_value=0)

        predictions = predict_model(model, X_test)
        evaluation = evaluate_classifier_predictions(df[label_col], predictions)
        results.append({
            "name": model_info.get("display_name", "Unnamed model"),
            "category": category,
            "metrics": evaluation["metrics"],
            "confusion_matrix": evaluation["confusion_matrix"],
            "predictions": evaluation["predictions"],
            "note": None,
        })
    return results

def train_queue():
    return None

# use process pool executor
def train_parallel():
    return None


# Old functions
def train_new_model(project, working_df):
    valid, message = validate_dataset(working_df, project["label_column"])
    if not valid:
        print(message)
        return

    from src.model.supervised_model import get_model_parameters, run_supervised_workflow

    model_type = select_model()
    parameters = get_model_parameters(model_type)
    config = get_training_config()

    result = run_supervised_workflow(
        df=working_df,
        label_col=project["label_column"],
        model_type=model_type,
        parameters=parameters,
        test_size=config["test_size"],
        random_state=config["random_state"],
    )

    display_name = input("\nModel display name: ").strip()
    add_model(
        project,
        result["model"],
        display_name,
        model_type,
        {**config, **parameters},
        result["metrics"],
        result["features"],
    )
    print("\nModel added to project.")


def test_saved_models(project, working_df):
    models = select_saved_models(project)
    if not models:
        return

    print("\nTest Options")
    print("1 - Current working dataset")
    choice = input("\nChoice: ").strip()

    if choice == "1":
        results = evaluate_saved_models(models, working_df, project["label_column"])
        display_test_results(results)
    else:
        print("Invalid option.")


def test_models_current_data(models, df, label_col):
    results = evaluate_saved_models(models, df, label_col)
    display_test_results(results)
    return results


def display_test_results(results):
    print("\n========================================")
    print("MODEL TEST RESULTS")
    print("========================================")
    print(f"{'Model':20}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    print("-" * 62)

    for r in results:
        metrics = r.get("metrics", r)
        print(
            f"{r['name']:20}"
            f"{metrics['accuracy']:10.3f}"
            f"{metrics['precision']:12.3f}"
            f"{metrics['recall']:10.3f}"
            f"{metrics['f1']:10.3f}"
        )

    show = input("\nShow first 20 predictions (y/n): ").lower()
    if show == "y":
        for r in results:
            print(f"\n{r['name']}")
            print(r["predictions"][:20])