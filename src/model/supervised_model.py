from src.model.model_factory import build_model
from src.model.model_training import train_test_classifier
from src.model.model_utils import get_rf_config, prepare_training_data
from src.model.result_builders import build_classification_results

# Return model parameters. If parameters is None, this falls back to the old CLI prompts
# change param to config and advanced
def get_model_parameters(model_type, parameters=None):
    if parameters is not None:
        return parameters

    if model_type == "random_forest":
        return get_rf_config()

    if model_type == "knn":
        neighbours = input("Number of neighbors [5]: ").strip()
        return {"n_neighbors": int(neighbours) if neighbours else 5}

    if model_type == "svm":
        c = input("C value [1.0]: ").strip()
        return {"C": float(c) if c else 1.0}

    if model_type == "decision_tree":
        depth = input("Max depth [None]: ").strip()
        return {"max_depth": int(depth) if depth else None}

    if model_type == "logistic_regression":
        c = input("C value [1.0]: ").strip()
        max_iter = input("Max iterations [1000]: ").strip()
        return {
            "C": float(c) if c else 1.0,
            "max_iter": int(max_iter) if max_iter else 1000,
        }

    return {}

# do not alter
def run_supervised_workflow(
    df,
    label_col,
    model_type,
    parameters=None,
    features=None,
    test_size=0.3,
    random_state=42,
    fill_method="median",
    fill_value=None,
    stratify=False,
):
    parameters = parameters or {}
    X, y = prepare_training_data(
        df=df,
        label_col=label_col,
        features=features,
        fill_method=fill_method,
        fill_value=fill_value,
    )

    model = build_model(
        model_type=model_type,
        parameters=parameters,
        random_state=random_state,
    )

    trained = train_test_classifier(
        model=model,
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    result = build_classification_results(
        model=trained["model"],
        X_test=trained["X_test"],
        y_test=trained["y_test"],
        predictions=trained["predictions"],
        feature_columns=X.columns.tolist(),
    )
    result.update({
        "model_type": model_type,
        "parameters": parameters,
        "X_test": trained["X_test"],
        "y_test": trained["y_test"],
    })
    return result
