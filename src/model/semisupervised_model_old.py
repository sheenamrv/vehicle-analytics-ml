import pandas as pd
from sklearn.base import clone
from sklearn.semi_supervised import SelfTrainingClassifier
from src.model.model_utils import (
    get_num_feature_columns,
    prepare_training_features,
)
from src.evaluation.metrics import (
    classification_metrics,
    classification_confusion_matrix,
)

# Convert unlabeled rows with a -1 (unlabeled) label for SelfTrainingClassifier
def convert_ssl_labels(df, label):
    return df[label].copy().fillna(-1)


# Main SSL training function
def train_ssl(df, features, label, pretrained_model, threshold=0.90, max_iter=10):
    # Prepare feature matrix using shared preprocessing utility
    X = prepare_training_features(df, features)

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
        verbose=True
    )

    ssl_model.fit(X, y)

    # Count only rows that were originally unlabeled and later received a label
    new_pseudo_labeled_mask = (
        original_unlabeled_mask &
        (ssl_model.transduction_ != -1)
    )

    pseudo_labeled_count = new_pseudo_labeled_mask.sum()
    unlabeled_remaining_count = (ssl_model.transduction_ == -1).sum()

    # Calculate pseudo-labeling progress percentages
    if original_unlabeled_count > 0:
        pseudo_labeled_percentage = (pseudo_labeled_count / original_unlabeled_count) * 100
        unlabeled_remaining_percentage = (unlabeled_remaining_count / original_unlabeled_count) * 100
    else:
        pseudo_labeled_percentage = 0
        unlabeled_remaining_percentage = 0

    # Store overall pseudo-labeling progress for easier plotting
    progress = pd.DataFrame({
        "status": [
            "Originally unlabeled",
            "Pseudo-labeled",
            "Remaining unlabeled"
        ],
        "count": [
            original_unlabeled_count,
            pseudo_labeled_count,
            unlabeled_remaining_count
        ],
        "percentage": [
            100.0,
            pseudo_labeled_percentage,
            unlabeled_remaining_percentage
        ]
    })

    result = {
        "model": ssl_model,
        "transduction": ssl_model.transduction_,
        "labeled_iter": ssl_model.labeled_iter_,
        "pseudo_labeled_count": pseudo_labeled_count,
        "unlabeled_remaining": unlabeled_remaining_count,
        "progress_df": progress
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
            "percentage": percentage
        })

    return pd.DataFrame(rows)  # Returning a df for easier plotting


# Evaluate the SSL model on a test dataset
def evaluate_ssl_model(model, test_df, features, label):
    # Prepare test feature matrix using the same shared preprocessing utility
    X_test = prepare_training_features(test_df, features)
    y_true = test_df[label]

    y_pred = model.predict(X_test)

    result = {
        "metrics": classification_metrics(y_true, y_pred),
        "confusion_matrix": classification_confusion_matrix(y_true, y_pred),
        "predictions": y_pred
    }

    return result


# Full SSL workflow function
# - Gets numeric features
# - Trains the SSL model
# - Tracks pseudo-labeling progress
# - Evaluates the trained SSL model on the test dataset
def run_ssl_workflow(
    train_df,
    test_df,
    label,
    pretrained_model,
    threshold=0.90,
    max_iter=10
):
    # Get the numeric feature columns
    features = get_num_feature_columns(train_df, label)

    ssl_result = train_ssl(
        df=train_df,
        features=features,
        label=label,
        pretrained_model=pretrained_model,
        threshold=threshold,
        max_iter=max_iter
    )

    evaluation = evaluate_ssl_model(
        model=ssl_result["model"],
        test_df=test_df,
        features=features,
        label=label
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
        "predictions": evaluation["predictions"]
    }

    return result