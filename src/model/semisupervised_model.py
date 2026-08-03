import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from src.model.model_training import evaluate_classifier_predictions
from src.model.model_utils import get_num_feature_columns

# Store preprocessing, label encoding, and prediction behavior in one saved model
class SemiSupervisedModelBundle(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator, features, fill_method="median", fill_value=None, scale=True):
        self.estimator = estimator
        self.features = list(features)
        self.fill_method = fill_method
        self.fill_value = fill_value
        self.scale = scale

    # Build the missing-value handler from the selected fill method
    def make_imputer(self):
        strategy = self.fill_method
        
        if strategy in (None, "none"):
            strategy = "constant"
            value = 0 if self.fill_value is None else self.fill_value
            
            return SimpleImputer(strategy=strategy, fill_value=value)
        
        if strategy == "constant":
            value = 0 if self.fill_value is None else self.fill_value
            
            return SimpleImputer(strategy="constant", fill_value=value)
        
        if strategy not in {"median", "mean", "most_frequent"}:
            if strategy == "mode":
                strategy = "most_frequent"
            else:
                raise ValueError(f"Unsupported fill_method: {self.fill_method}")
            
        return SimpleImputer(strategy=strategy)

    # Select the original training features and validate incoming prediction data
    def select(self, X):
        if isinstance(X, pd.DataFrame):
            missing = [column for column in self.features if column not in X.columns]
            if missing:
                raise ValueError(f"Feature columns not found: {missing}")
            return X[self.features]
        array = np.asarray(X)
        if array.ndim != 2 or array.shape[1] != len(self.features):
            raise ValueError(
                f"Expected {len(self.features)} features, received shape {array.shape}."
            )
        return array

    # Fit preprocessing and the self-training classifier on labeled and unlabeled rows.
    def fit(self, X, y):
        selected = self.select(X)
        self.imputer_ = self.make_imputer()
        transformed = self.imputer_.fit_transform(selected)
        self.scaler_ = StandardScaler() if self.scale else None
        if self.scaler_ is not None:
            transformed = self.scaler_.fit_transform(transformed)

        labeled = pd.Series(y).notna().to_numpy()
        if labeled.sum() == 0:
            raise ValueError("At least one initially labeled row is required.")

        self.label_encoder_ = LabelEncoder()
        encoded = np.full(len(y), -1, dtype=int)
        encoded[labeled] = self.label_encoder_.fit_transform(pd.Series(y)[labeled])
        if len(self.label_encoder_.classes_) < 2:
            raise ValueError("At least two initially labeled classes are required.")

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(transformed, encoded)
        self.classes_ = self.label_encoder_.classes_
        return self

    # Reuse the training-fitted imputer and scaler for all future predictions.
    def transform(self, X):
        selected = self.select(X)
        transformed = self.imputer_.transform(selected)
        if self.scaler_ is not None:
            transformed = self.scaler_.transform(transformed)
        return transformed

    # Predict encoded classes, then convert them back to the original label values.
    def predict(self, X):
        encoded = np.asarray(self.estimator_.predict(self.transform(X)), dtype=int)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.transform(X))

    def decision_function(self, X):
        if not hasattr(self.estimator_, "decision_function"):
            raise AttributeError("The wrapped estimator has no decision_function().")
        return self.estimator_.decision_function(self.transform(X))

    @property
    def transduction_(self):
        return self.estimator_.transduction_

    @property
    def labeled_iter_(self):
        return self.estimator_.labeled_iter_


# Convert unlabeled rows with a -1 (unlabeled) label for SelfTrainingClassifier
# The saved model bundle later performs the final numeric encoding safely.
def convert_ssl_labels(df, label):
    if label not in df.columns:
        raise ValueError(f"Label column not found: {label}")
    return df[label].copy().fillna(-1)


# Validate the SSL inputs before model training so errors are clear and early.
def _validate_ssl_inputs(df, features, label, base_estimator, threshold, max_iter, criterion, k_best):
    if df is None or df.empty:
        raise ValueError("Training dataset is empty.")
    if label not in df.columns:
        raise ValueError(f"Label column not found: {label}")
    if not features:
        raise ValueError("At least one numeric feature is required.")
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}")
    labeled = df[label].dropna()
    if labeled.empty:
        raise ValueError("At least one initially labeled row is required.")
    if labeled.nunique() < 2:
        raise ValueError("At least two initially labeled classes are required.")
    if criterion == "threshold":
        if not 0 <= float(threshold) < 1:
            raise ValueError("threshold must be in the range [0, 1).")
        if not hasattr(base_estimator, "predict_proba"):
            raise TypeError("Threshold self-training requires a base model with predict_proba().")
    elif criterion == "k_best":
        if int(k_best) < 1:
            raise ValueError("k_best must be at least 1.")
    else:
        raise ValueError("criterion must be 'threshold' or 'k_best'.")
    if max_iter is not None and int(max_iter) < 1:
        raise ValueError("max_iter must be at least 1 or None.")


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
    criterion="threshold",
    k_best=10,
    scale=True,
):
    # Validate the dataset, labels, model capabilities, and SSL parameters
    _validate_ssl_inputs(
        df, features, label, pretrained_model, threshold, max_iter, criterion, k_best
    )

    # Takes a new copy of the pretrained model and tracks progress through verbose parameter
    # clone() preserves the model settings but intentionally starts from an unfitted estimator
    self_training = SelfTrainingClassifier(
        estimator=clone(pretrained_model),
        threshold=float(threshold),
        criterion=criterion,
        k_best=int(k_best),
        max_iter=max_iter,
        verbose=verbose,
    )
    # Keep feature preparation, scaling, label decoding, and prediction in one model artifact
    model = SemiSupervisedModelBundle(
        estimator=self_training,
        features=features,
        fill_method=fill_method,
        fill_value=fill_value,
        scale=scale,
    )
    # Convert missing labels internally to -1 when the model bundle is fitted
    y = df[label].copy()
    model.fit(df, y)

    # Track original number of unlabeled labels
    original_unlabeled_mask = y.isna().to_numpy()
    original_unlabeled_count = int(original_unlabeled_mask.sum())
    # Count only rows that were originally unlabeled and later received a label
    pseudo_mask = original_unlabeled_mask & (model.transduction_ != -1)
    pseudo_count = int(pseudo_mask.sum())
    remaining_count = int((model.transduction_ == -1).sum())

    # Calculate pseudo-labeling progress percentages
    denominator = original_unlabeled_count or 1
    # Store overall pseudo-labeling progress for easier plotting
    progress = pd.DataFrame({
        "status": ["Originally unlabeled", "Pseudo-labeled", "Remaining unlabeled"],
        "count": [original_unlabeled_count, pseudo_count, remaining_count],
        "percentage": [
            100.0 if original_unlabeled_count else 0.0,
            pseudo_count / denominator * 100 if original_unlabeled_count else 0.0,
            remaining_count / denominator * 100 if original_unlabeled_count else 0.0,
        ],
    })
    return {
        "model": model,
        "transduction": model.transduction_,
        "labeled_iter": model.labeled_iter_,
        "pseudo_labeled_count": pseudo_count,
        "unlabeled_remaining": remaining_count,
        "progress_df": progress,
    }


# Get SSL training progress by iteration
def get_ssl_training_progress(train_ssl_result):
    labeled_iter = np.asarray(train_ssl_result["labeled_iter"])
    rows = []
    total = len(labeled_iter)
    for iteration in sorted(set(labeled_iter.tolist())):
        count = int((labeled_iter == iteration).sum())
        description = (
            "Never labeled" if iteration == -1 else
            "Originally labeled" if iteration == 0 else
            f"Pseudo-labeled on iteration {iteration}"
        )
        rows.append({
            "iteration": int(iteration),
            "description": description,
            "count": count,
            "percentage": count / total * 100 if total else 0.0,
        })
    return pd.DataFrame(rows)  # Returning a df for easier plotting


# Evaluate the SSL model on a test dataset
def evaluate_ssl_model(model, test_df, features, label, **_):
    if test_df is None or test_df.empty:
        raise ValueError("Test dataset is empty.")
    if label not in test_df.columns:
        raise ValueError(f"Label column not found in test data: {label}")
    if test_df[label].isna().any():
        raise ValueError("Test labels must all be known for evaluation.")
    # The model bundle applies the same imputer and scaler learned from training
    predictions = model.predict(test_df[features])
    return evaluate_classifier_predictions(test_df[label], predictions)


# Check whether the same dataset was supplied for both training and testing.
def _same_dataset(left, right):
    if left is right:
        return True
    try:
        return left.shape == right.shape and left.equals(right)
    except Exception:
        return False


# Create a labeled evaluation holdout when the frontend supplies one dataframe.
# Unlabeled rows remain in the training set so they can be pseudo-labeled.
def _automatic_ssl_split(df, label, test_size, random_state):
    labeled = df[df[label].notna()].copy()
    unlabeled = df[df[label].isna()].copy()
    if labeled[label].nunique() < 2:
        raise ValueError("At least two labeled classes are required for an evaluation split.")
    counts = labeled[label].value_counts()
    stratify = labeled[label] if counts.min() >= 2 else None
    train_labeled, test_df = train_test_split(
        labeled,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=stratify,
    )
    train_df = pd.concat([train_labeled, unlabeled], axis=0).sort_index()
    return train_df, test_df


# do not alter
# Full SSL workflow function
# - Gets numeric features if no custom features are provided
# - Trains the SSL model
# - Tracks pseudo-labeling progress
# - Evaluates the trained SSL model on the test dataset
# - Creates a safe labeled holdout if train_df and test_df are the same dataset
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
    criterion="threshold",
    k_best=10,
    scale=True,
    test_size=0.3,
    random_state=42,
):
    # Get the numeric feature columns if features were not provided
    if features is None:
        features = get_num_feature_columns(train_df, label)
    # Prevent evaluation on the same labeled rows used to train the model
    if _same_dataset(train_df, test_df):
        train_df, test_df = _automatic_ssl_split(train_df, label, test_size, random_state)

    # Train the SSL model and collect pseudo-labeling progress
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
        criterion=criterion,
        k_best=k_best,
        scale=scale,
    )
    # Evaluate only on the separate labeled test dataset
    evaluation = evaluate_ssl_model(ssl_result["model"], test_df, features, label)

    # Preserve the original workflow result format used by the frontend.
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