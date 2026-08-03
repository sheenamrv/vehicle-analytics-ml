import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from copy import deepcopy
from src.model.model_training import evaluate_classifier_predictions
from src.model.model_utils import get_num_feature_columns


class TransferredSelfTrainingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        estimator,
        threshold=0.90,
        criterion="threshold",
        k_best=10,
        max_iter=10,
        verbose=False,
        use_pretrained_state=False,
        original_classes=None,
    ):
        self.estimator = estimator
        self.threshold = threshold
        self.criterion = criterion
        self.k_best = k_best
        self.max_iter = max_iter
        self.verbose = verbose
        self.use_pretrained_state = use_pretrained_state
        self.original_classes = original_classes

    def is_fitted(self, estimator):
        try:
            check_is_fitted(estimator)
            return True
        except Exception:
            return False

    def aligned_probabilities(self, estimator, X):
        probabilities = np.asarray(estimator.predict_proba(X), dtype=float)

       
        # Reorder the probabilitycolumns so they match the encoded class order
        if not self.original_classes or not hasattr(estimator, "classes_"):
            return probabilities

        original_classes = list(self.original_classes)
        estimator_classes = list(estimator.classes_)
        aligned = np.zeros((len(X), len(original_classes)), dtype=float)

        class_positions = {
            value: index for index, value in enumerate(estimator_classes)
        }
        for encoded_index, original_value in enumerate(original_classes):
            if original_value not in class_positions:
                raise ValueError(
                    "The selected pretrained model does not contain all classes "
                    "required by the SSL dataset."
                )
            aligned[:, encoded_index] = probabilities[
                :, class_positions[original_value]
            ]
        return aligned

    def _select_candidates(self, probabilities):
        if len(probabilities) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        confidence = probabilities.max(axis=1)
        predicted = probabilities.argmax(axis=1)

        if self.criterion == "threshold":
            selected = np.flatnonzero(confidence >= float(self.threshold))
        else:
            count = min(int(self.k_best), len(confidence))
            if count <= 0:
                selected = np.array([], dtype=int)
            else:
                selected = np.argsort(confidence)[-count:]

        return selected, predicted[selected]

    def _record_progress(self, iteration, newly_labeled, total_unlabeled, source):
        remaining = int(np.sum(self.transduction_ == -1))
        total_pseudo = int(total_unlabeled - remaining)
        denominator = total_unlabeled or 1
        self.iteration_history_.append({
            "iteration": int(iteration),
            "description": (
                "Pretrained model seed"
                if source == "pretrained"
                else f"Pseudo-labeled on iteration {iteration}"
            ),
            "newly_pseudo_labeled": int(newly_labeled),
            "pseudo_labeled_total": total_pseudo,
            "remaining_unlabeled": remaining,
            "percentage": total_pseudo / denominator * 100,
            "remaining_percentage": remaining / denominator * 100,
        })

    def fit(self, X, y):
        X_values = X
        y_values = np.asarray(y, dtype=int)
        self.transduction_ = y_values.copy()
        self.labeled_iter_ = np.where(y_values == -1, -1, 0)
        self.iteration_history_ = []
        self.pretrained_state_used_ = False

        original_unlabeled = self.transduction_ == -1
        total_unlabeled = int(original_unlabeled.sum())
        iteration = 0

        # Use the actual fitted estimator once, before any clone/refit occurs
        if (
            self.use_pretrained_state
            and total_unlabeled > 0
            and self.is_fitted(self.estimator)
        ):
            unlabeled_indexes = np.flatnonzero(self.transduction_ == -1)
            probabilities = self.aligned_probabilities(
                self.estimator, X_values.iloc[unlabeled_indexes]
                if isinstance(X_values, pd.DataFrame)
                else X_values[unlabeled_indexes]
            )
            selected_local, predicted = self._select_candidates(probabilities)
            selected_indexes = unlabeled_indexes[selected_local]

            if len(selected_indexes):
                iteration = 1
                self.transduction_[selected_indexes] = predicted
                self.labeled_iter_[selected_indexes] = iteration

            self.pretrained_state_used_ = True
            self._record_progress(
                iteration=iteration,
                newly_labeled=len(selected_indexes),
                total_unlabeled=total_unlabeled,
                source="pretrained",
            )

        start_iteration = max(iteration + 1, 1)
        # With max_iter=None, at most one new group of rows can be added per pass, so the number of originally unlabeled rows is a safe upper bound
        final_iteration = (
            total_unlabeled + 1
            if self.max_iter is None
            else int(self.max_iter)
        )

        for current_iteration in range(start_iteration, final_iteration + 1):
            labeled_mask = self.transduction_ != -1
            estimator = clone(self.estimator)
            estimator.fit(
                X_values.iloc[labeled_mask]
                if isinstance(X_values, pd.DataFrame)
                else X_values[labeled_mask],
                self.transduction_[labeled_mask],
            )

            unlabeled_indexes = np.flatnonzero(self.transduction_ == -1)
            if not len(unlabeled_indexes):
                self.estimator_ = estimator
                self.n_iter_ = current_iteration - 1
                self.termination_condition_ = "all_labeled"
                break

            probabilities = np.asarray(
                estimator.predict_proba(
                    X_values.iloc[unlabeled_indexes]
                    if isinstance(X_values, pd.DataFrame)
                    else X_values[unlabeled_indexes]
                ),
                dtype=float,
            )
            selected_local, predicted = self._select_candidates(probabilities)
            selected_indexes = unlabeled_indexes[selected_local]

            if len(selected_indexes):
                self.transduction_[selected_indexes] = predicted
                self.labeled_iter_[selected_indexes] = current_iteration

            self._record_progress(
                iteration=current_iteration,
                newly_labeled=len(selected_indexes),
                total_unlabeled=total_unlabeled,
                source="self_training",
            )

            if self.verbose:
                print(
                    f"SSL iteration {current_iteration}: "
                    f"pseudo-labeled {len(selected_indexes)} rows."
                )

            if not len(selected_indexes):
                self.estimator_ = estimator
                self.n_iter_ = current_iteration
                self.termination_condition_ = "no_change"
                break
        else:
            self.n_iter_ = final_iteration
            self.termination_condition_ = "max_iter"

        # Fit the final estimator on every original and pseudo-labeled row so the saved model contains the most complete training state
        
        labeled_mask = self.transduction_ != -1
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(
            X_values.iloc[labeled_mask]
            if isinstance(X_values, pd.DataFrame)
            else X_values[labeled_mask],
            self.transduction_[labeled_mask],
        )
        self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        if not hasattr(self.estimator_, "decision_function"):
            raise AttributeError("The wrapped estimator has no decision_function().")
        return self.estimator_.decision_function(X)


# Store preprocessing, label encoding, and prediction behavior in one saved model
class SemiSupervisedModelBundle(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        estimator,
        features,
        fill_method="median",
        fill_value=None,
        scale=True,
        use_pretrained_state=False,
    ):
        self.estimator = estimator
        self.features = list(features)
        self.fill_method = fill_method
        self.fill_value = fill_value
        self.scale = scale
        self.use_pretrained_state = use_pretrained_state

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

    # Fit preprocessing and the self-training classifier on labeled and unlabeled rows
    def fit(self, X, y):
        selected = self.select(X)
        self.imputer_ = self.make_imputer()
        transformed = self.imputer_.fit_transform(selected)

        # When reusing a fitted supervised estimator, avoid fitting a new scaler because the estimator expects the same feature representation it was originally trained with
        effective_scale = bool(self.scale and not self.use_pretrained_state)
        self.scaler_ = StandardScaler() if effective_scale else None
        if self.scaler_ is not None:
            transformed = self.scaler_.fit_transform(transformed)

        transformed = pd.DataFrame(
            transformed,
            columns=self.features,
            index=selected.index if isinstance(selected, pd.DataFrame) else None,
        )

        labeled = pd.Series(y).notna().to_numpy()
        if labeled.sum() == 0:
            raise ValueError("At least one initially labeled row is required.")

        self.label_encoder_ = LabelEncoder()
        encoded = np.full(len(y), -1, dtype=int)
        encoded[labeled] = self.label_encoder_.fit_transform(pd.Series(y)[labeled])
        if len(self.label_encoder_.classes_) < 2:
            raise ValueError("At least two initially labeled classes are required.")


        self.estimator_ = (
            deepcopy(self.estimator)
            if self.use_pretrained_state
            else clone(self.estimator)
        )
        if hasattr(self.estimator_, "set_params"):
            self.estimator_.set_params(
                original_classes=list(self.label_encoder_.classes_)
            )
        self.estimator_.fit(transformed, encoded)
        self.classes_ = self.label_encoder_.classes_
        self.pretrained_state_used_ = bool(
            getattr(self.estimator_, "pretrained_state_used_", False)
        )
        return self

    # Reuse the training-fitted imputer and scaler for all future predictions
    def transform(self, X):
        selected = self.select(X)
        transformed = self.imputer_.transform(selected)
        if self.scaler_ is not None:
            transformed = self.scaler_.transform(transformed)
        return pd.DataFrame(
            transformed,
            columns=self.features,
            index=selected.index if isinstance(selected, pd.DataFrame) else None,
        )

    # Predict encoded classes, then convert them back to the original label values
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

    @property
    def iteration_history_(self):
        return getattr(self.estimator_, "iteration_history_", [])


# Convert unlabeled rows with a -1 (unlabeled) label for SelfTrainingClassifier
# The saved model bundle later performs the final numeric encoding safely
def convert_ssl_labels(df, label):
    if label not in df.columns:
        raise ValueError(f"Label column not found: {label}")
    return df[label].copy().fillna(-1)


# Validate the SSL inputs before model training so errors are clear
def _validate_ssl_inputs(
    df,
    features,
    label,
    base_estimator,
    threshold,
    max_iter,
    criterion,
    k_best,
):
    """Validate the inputs required for semi-supervised training."""

    if df is None or df.empty:
        raise ValueError(
            "The selected dataset is empty."
        )

    if not features:
        raise ValueError(
            "At least one feature column must be selected."
        )

    if label not in df.columns:
        raise ValueError(
            f"Label column '{label}' was not found."
        )

    if base_estimator is None:
        raise ValueError(
            "A pretrained supervised model must be selected."
        )

    if max_iter is not None and int(max_iter) < 1:
        raise ValueError(
            "max_iter must be greater than 0."
        )

    if criterion not in {"threshold", "k_best"}:
        raise ValueError(
            "criterion must be 'threshold' or 'k_best'."
        )

    if not hasattr(base_estimator, "predict_proba"):
        raise TypeError(
            "The selected base model must implement predict_proba()."
        )

    if criterion == "threshold":
        threshold = float(threshold)

        if not 0.0 <= threshold < 1.0:
            raise ValueError(
                "threshold must be in the range [0, 1)."
            )

    if criterion == "k_best":
        if int(k_best) < 1:
            raise ValueError(
                "k_best must be greater than 0."
            )


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
    use_pretrained_state=False,
):
    # Validate the dataset, labels, model capabilities, and SSL parameters
    _validate_ssl_inputs(
        df, features, label, pretrained_model, threshold, max_iter, criterion, k_best
    )

    # A selected trained model can seed the first pseudo-labeling pass with itsb actual fitted state
    self_training = TransferredSelfTrainingClassifier(
        estimator=pretrained_model,
        threshold=float(threshold),
        criterion=criterion,
        k_best=int(k_best),
        max_iter=max_iter,
        verbose=verbose,
        use_pretrained_state=bool(use_pretrained_state),
    )

    # Keep feature preparation, scaling, label decoding, and prediction in one model artifact
    model = SemiSupervisedModelBundle(
        estimator=self_training,
        features=features,
        fill_method=fill_method,
        fill_value=fill_value,
        scale=scale,
        use_pretrained_state=bool(use_pretrained_state),
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
        "status": [
            "Originally unlabeled",
            "Pseudo-labeled",
            "Remaining unlabeled",
        ],
        "count": [
            original_unlabeled_count,
            pseudo_count,
            remaining_count,
        ],
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
    model = train_ssl_result.get("model")
    history = getattr(model, "iteration_history_", []) if model is not None else []
    if history:
        return pd.DataFrame(history)

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


# Create a labeled evaluation holdout when the frontend supplies one dataframe
# Unlabeled rows remain in the training set so they can be pseudo-labeled
def _automatic_ssl_split(df, label, test_size, random_state):
    labeled = df[df[label].notna()].copy()
    unlabeled = df[df[label].isna()].copy()
    counts = labeled[label].value_counts()

    if len(counts) < 2:
        raise ValueError("At least two labeled classes are required for an evaluation split.")
    if counts.min() < 2:
        raise ValueError(
            "Each labeled class needs at least two rows so one can remain in "
            "training and one can be used for evaluation."
        )

    # Keep at least one labeled row from every class in both partitions
    class_count = len(counts)
    requested = int(round(len(labeled) * float(test_size)))
    test_count = max(class_count, requested)
    test_count = min(test_count, len(labeled) - class_count)
    if test_count < class_count:
        raise ValueError("Not enough labeled rows to create a safe SSL evaluation split.")

    train_labeled, test_df = train_test_split(
        labeled,
        test_size=test_count,
        random_state=int(random_state),
        stratify=labeled[label],
    )
    train_df = pd.concat([train_labeled, unlabeled], axis=0).sort_index()
    return train_df, test_df


def _build_ssl_export_dataframe(
    original_df,
    training_df,
    label,
    model,
    transduction,
    labeled_iter,
    position_column,
):
    """Build the dataset produced by the fitted self-training workflow."""
    export_df = original_df.copy()
    source_column = f"{label}_source"
    iteration_column = f"{label}_ssl_iteration"

    export_df[label] = export_df[label].astype(object)
    export_df[source_column] = np.where(
        export_df[label].notna(),
        "original",
        "remaining_unlabeled",
    )
    export_df[iteration_column] = np.where(
        export_df[label].notna(),
        0,
        -1,
    )

    positions = training_df[position_column].to_numpy(dtype=int)
    training_labels = training_df[label]
    originally_unlabeled = training_labels.isna().to_numpy()
    transduction = np.asarray(transduction)
    labeled_iter = np.asarray(labeled_iter)

    if not (
        len(positions) == len(transduction) == len(labeled_iter)
    ):
        raise ValueError(
            "SSL export mapping does not match the fitted training rows."
        )

    accepted_mask = originally_unlabeled & (transduction != -1)
    remaining_mask = originally_unlabeled & (transduction == -1)

    if accepted_mask.any():
        accepted_positions = positions[accepted_mask]
        encoded_labels = transduction[accepted_mask].astype(int)
        decoded_labels = model.label_encoder_.inverse_transform(encoded_labels)

        label_column_index = export_df.columns.get_loc(label)
        source_column_index = export_df.columns.get_loc(source_column)
        iteration_column_index = export_df.columns.get_loc(iteration_column)

        export_df.iloc[accepted_positions, label_column_index] = decoded_labels
        export_df.iloc[accepted_positions, source_column_index] = "pseudo_labeled"
        export_df.iloc[accepted_positions, iteration_column_index] = (
            labeled_iter[accepted_mask].astype(int)
        )

    if remaining_mask.any():
        remaining_positions = positions[remaining_mask]
        source_column_index = export_df.columns.get_loc(source_column)
        iteration_column_index = export_df.columns.get_loc(iteration_column)
        export_df.iloc[remaining_positions, source_column_index] = (
            "remaining_unlabeled"
        )
        export_df.iloc[remaining_positions, iteration_column_index] = -1

    return export_df


# do not alter
# Full SSL workflow function
# - Gets numeric features if no custom features are provided
# - Trains the SSL model
# - Tracks pseudo-labeling progress
# - Evaluates the trained SSL model on the test dataset
# - Creates a safe labeled holdout if train_df and test_df are the same dataset
# - Builds an export-ready dataset without forcing labels onto rejected samples
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
    use_pretrained_state=False,
):
    # Get the numeric feature columns if features were not provided
    if features is None:
        features = get_num_feature_columns(train_df, label)

    # Keep an untouched copy for dataset export. A temporary positional column follows rows through the holdout split so transduction values can be mapped
    # back to the correct original records even when dataframe indexes differ
    original_train_df = train_df.copy()
    position_column = "__ssl_source_position__"
    while position_column in train_df.columns:
        position_column = f"_{position_column}"

    train_df = train_df.copy()
    train_df[position_column] = np.arange(len(train_df), dtype=int)

    same_dataset = _same_dataset(original_train_df, test_df)
    if same_dataset:
        test_df = train_df.copy()
    else:
        test_df = test_df.copy()

    # Prevent evaluation on the same labeled rows used to train the model
    if same_dataset:
        train_df, test_df = _automatic_ssl_split(
            train_df, label, test_size, random_state
        )

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
        use_pretrained_state=use_pretrained_state,
    )

    # Evaluate only on the separate labeled test dataset
    evaluation = evaluate_ssl_model(
        ssl_result["model"], test_df, features, label
    )

    # Build the exact dataset that can later be exported from the Results menu
    ssl_export_df = _build_ssl_export_dataframe(
        original_df=original_train_df,
        training_df=train_df,
        label=label,
        model=ssl_result["model"],
        transduction=ssl_result["transduction"],
        labeled_iter=ssl_result["labeled_iter"],
        position_column=position_column,
    )

    ssl_result["model"].ssl_export_df_ = ssl_export_df.copy()

    # Preserve the original workflow result format used by the fronten
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
        "ssl_export_df": ssl_export_df,
    }

    return result
