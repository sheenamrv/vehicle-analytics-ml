from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.model.model_utils import (
    validate_dataset,
    prepare_training_data,
    select_model,
    get_training_config
)

from src.model.supervised_model import (
    build_model,
    get_model_parameters
)

from src.model.model_registry import (
    add_model,
    select_saved_models
)

def train_model(model, X, y, config):

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=config["test_size"], random_state=config["random_state"]
        )
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),
        "precision":
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),
        "recall":
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),
        "f1":
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
    }

    return model, metrics

def test_saved_models(project, working_df):
    
    models = (select_saved_models(project))
    
    if not models:
        return
    
    print("\nTest Options")
    print("1 - Current working dataset")

    choice = input(
        "\nChoice: "
    ).strip()

    if choice == "1":

        test_models_current_data(
            models,
            working_df,
            project["label_column"]
        )

    else:

        print(
            "Invalid option."
        )
        
        
def train_new_model(project, working_df):
    
    valid, message = (
        validate_dataset(
            working_df,
            project["label_column"]
        )
    )

    if not valid:
        print(message)
        return

    X, y = (
        prepare_training_data(
            working_df,
            project["label_column"]
        )
    )

    feature_columns = (X.columns.tolist())
    model_type = (
        select_model()
    )

    parameters = (
        get_model_parameters(
            model_type
        )
    )

    config = (
        get_training_config()
    )

    model = (
        build_model(
            model_type,
            parameters
        )
    )

    trained_model, metrics = (
        train_model(
            model,
            X,
            y,
            config
        )
    )

    display_name = input(
        "\nModel display name: "
    ).strip()

    add_model(
        project,
        trained_model,
        display_name,
        model_type,
        {
            **config,
            **parameters
        },
        metrics,
        feature_columns
    )

    print(
        "\nModel added "
        "to project."
    )
    
def test_models_current_data(models, df, label_col):
    
    X, y = prepare_training_data(df, label_col)
    
    results = []
    
    for model_info in models:
        
        X_test = X.reindex(columns=model_info["feature_columns"], fill_value=0)
        
        preds = (model_info["model"].predict(X_test))
        
        results.append({
            "name":
                model_info[
                    "display_name"
                ],
            "accuracy":
                accuracy_score(
                    y,
                    preds
                ),
            "precision":
                precision_score(
                    y,
                    preds,
                    average="weighted",
                    zero_division=0
                ),
            "recall":
                recall_score(
                    y,
                    preds,
                    average="weighted",
                    zero_division=0
                ),
            "f1":
                f1_score(
                    y,
                    preds,
                    average="weighted",
                    zero_division=0
                ),
            "predictions":
                preds
        })

    display_test_results(
        results
    )
    
def display_test_results(results):

    print("\n========================================")
    print("MODEL TEST RESULTS")
    print("========================================")

    print(
        f"{'Model':20}"
        f"{'Accuracy':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )

    print("-" * 62)

    for r in results:

        print(
            f"{r['name']:20}"
            f"{r['accuracy']:10.3f}"
            f"{r['precision']:12.3f}"
            f"{r['recall']:10.3f}"
            f"{r['f1']:10.3f}"
        )

    show = input(
        "\nShow first 20 predictions (y/n): "
    ).lower()

    if show == "y":

        for r in results:

            print(
                f"\n{r['name']}"
            )

            print(
                r["predictions"][:20]
            )