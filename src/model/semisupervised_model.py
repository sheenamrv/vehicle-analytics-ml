import pandas as pd
from sklearn.base import clone
from sklearn.semi_supervised import SelfTrainingClassifier
from src.model.model_training import evaluate_classifier_predictions, fit_model, predict_model
from src.model.model_utils import (
    get_num_feature_columns,
    prepare_training_features,
)


# Convert unlabeled rows with a -1 (unlabeled) label for SelfTrainingClassifier
def convert_ssl_labels(df, label):
    return df[label].copy().fillna(-1)

# Main SSL training function
def train_ssl(
    df,
    features,
    label,
    pretrained_model,
    threshold=0.90,
    max_iter=10,
    fill_method="median",
    fill_value=None,
    verbose=False,
):
    # Prepare feature matrix using shared preprocessing utility
    X = prepare_training_features(
        df=df,
        features=features,
        fill_method=fill_method,
        fill_value=fill_value,
    )

    # Convert missing labels to -1, which SelfTrainingClassifier treats as unlabeled
    y = convert_ssl_labels(df, label)

    # Track original number of unlabeled labels
    original_unlabeled_mask = y == -1
    original_unlabeled_count = original_unlabeled_mask.sum()

    # Takes a new copy of the pretrained model and tracks progress through verbose parameter
    ssl_model = SelfTrainingClassifier(
        estimator=clone(pretrained_model),
        threshold=threshold,
        max_iter=max_iter,
        verbose=verbose,
    )

    fit_model(ssl_model, X, y)

    # Count only rows that were originally unlabeled and later received a label
    new_pseudo_labeled_mask = original_unlabeled_mask & (ssl_model.transduction_ != -1)

    pseudo_labeled_count = new_pseudo_labeled_mask.sum()
    unlabeled_remaining_count = (ssl_model.transduction_ == -1).sum()

    # Calculate pseudo-labeling progress percentages
    if original_unlabeled_count > 0:
        pseudo_labeled_percentage = (pseudo_labeled_count / original_unlabeled_count) * 100
        unlabeled_remaining_percentage = (
            unlabeled_remaining_count / original_unlabeled_count
        ) * 100
    else:
        pseudo_labeled_percentage = 0
        unlabeled_remaining_percentage = 0

    # Store overall pseudo-labeling progress for easier plotting
    progress = pd.DataFrame({
        "status": [
            "Originally unlabeled",
            "Pseudo-labeled",
            "Remaining unlabeled",
        ],
        "count": [
            original_unlabeled_count,
            pseudo_labeled_count,
            unlabeled_remaining_count,
        ],
        "percentage": [
            100.0,
            pseudo_labeled_percentage,
            unlabeled_remaining_percentage,
        ],
    })

    result = {
        "model": ssl_model,
        "transduction": ssl_model.transduction_,
        "labeled_iter": ssl_model.labeled_iter_,
        "pseudo_labeled_count": pseudo_labeled_count,
        "unlabeled_remaining": unlabeled_remaining_count,
        "progress_df": progress,
    }

    return result


# Get SSL training progress by iteration
def get_ssl_training_progress(train_ssl_result):
    labeled_iter = train_ssl_result["labeled_iter"]

    rows = []
    total_count = len(labeled_iter)

    for iteration in sorted(set(labeled_iter)):
        count = (labeled_iter == iteration).sum()

        if total_count > 0:
            percentage = (count / total_count) * 100
        else:
            percentage = 0

        if iteration == -1:
            description = "Never labeled"
        elif iteration == 0:
            description = "Originally labeled"
        else:
            description = f"Pseudo-labeled on iteration {iteration}"

        rows.append({
            "iteration": iteration,
            "description": description,
            "count": count,
            "percentage": percentage,
        })

    result = pd.DataFrame(rows)

    return result  # Returning a df for easier plotting

# Evaluate the SSL model on a test dataset
def evaluate_ssl_model(
    model,
    test_df,
    features,
    label,
    fill_method="median",
    fill_value=None,
):
    # Prepare test feature matrix using the same shared preprocessing utility
    X_test = prepare_training_features(
        df=test_df,
        features=features,
        fill_method=fill_method,
        fill_value=fill_value,
    )

    y_true = test_df[label]
    y_pred = predict_model(model, X_test)

    result = evaluate_classifier_predictions(y_true, y_pred)

    return result


# Full SSL workflow function
# - Gets numeric features if no custom features are provided
# - Trains the SSL model
# - Tracks pseudo-labeling progress
# - Evaluates the trained SSL model on the test dataset
def run_ssl_workflow(
    train_df,
    test_df,
    label,
    pretrained_model,
    threshold=0.90,
    max_iter=10,
    features=None,
    fill_method="median",
    fill_value=None,
    verbose=False,
):
    # Get the numeric feature columns if features were not provided
    if features is None:
        features = get_num_feature_columns(train_df, label)

    ssl_result = train_ssl(
        df=train_df,
        features=features,
        label=label,
        pretrained_model=pretrained_model,
        threshold=threshold,
        max_iter=max_iter,
        fill_method=fill_method,
        fill_value=fill_value,
        verbose=verbose,
    )

    evaluation = evaluate_ssl_model(
        model=ssl_result["model"],
        test_df=test_df,
        features=features,
        label=label,
        fill_method=fill_method,
        fill_value=fill_value,
    )

    result = {
        "features": features,
        "ssl_model": ssl_result["model"],
        "transduction": ssl_result["transduction"],
        "labeled_iter": ssl_result["labeled_iter"],
        "progress_df": ssl_result["progress_df"],
        "iteration_progress": get_ssl_training_progress(ssl_result),
        "metrics": evaluation["metrics"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "predictions": evaluation["predictions"],
    }

    return result