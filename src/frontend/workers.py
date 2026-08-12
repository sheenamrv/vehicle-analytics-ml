from pathlib import Path

import joblib
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from src.data.process import unpack_model_package
from src.model.model_controller import ModelController
from src.model.model_factory import build_model
from src.model.model_training import evaluate_saved_models
from src.model.semisupervised_model import run_ssl_workflow
from src.model.supervised_model import run_supervised_workflow
from src.model.unsupervised_model import run_unsupervised_workflow


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(Exception)


# Backend workflows can be CPU-heavy. Workers receive DataFrame copies so the
# GUI thread cannot mutate training input while a job is running.
class DataLoadWorker(QRunnable):
    def __init__(self, file_path, dataset, columns):
        super().__init__()
        self.file_path = Path(file_path)
        self.dataset = dataset
        self.columns = columns
        self.signals = WorkerSignals()

    def run(self):
        from src.data.test_load import select_col

        try:
            df = select_col(self.file_path, self.dataset, self.columns)
        except Exception as error:
            self.signals.error.emit(error)
            return

        self.signals.finished.emit(df)


class ModelTrainingWorker(QRunnable):
    def __init__(self, dataframe, label_column, options):
        super().__init__()
        self.dataframe = dataframe.copy()
        self.label_column = label_column
        self.options = options
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = run_supervised_workflow(
                df=self.dataframe,
                label_col=self.label_column,
                model_type=self.options["model_type"],
                parameters=self.options["parameters"],
                test_size=self.options["test_size"],
                random_state=self.options["random_state"],
                stratify=self.options["stratify"],
            )
        except Exception as error:
            self.signals.error.emit(error)
            return
        self.signals.finished.emit((self.options, result))


class ModelEvaluationWorker(QRunnable):
    def __init__(self, models, dataframe, label_column):
        super().__init__()
        self.models = models
        self.dataframe = dataframe.copy()
        self.label_column = label_column
        self.signals = WorkerSignals()

    def run(self):
        try:
            results = evaluate_saved_models(
                self.models,
                self.dataframe,
                self.label_column,
            )
        except Exception as error:
            self.signals.error.emit(error)
            return
        self.signals.finished.emit(results)


class SemiSupervisedTrainingWorker(QRunnable):
    def __init__(self, train_dataframe, test_dataframe, label_column, options):
        super().__init__()
        self.train_dataframe = train_dataframe.copy()
        self.test_dataframe = test_dataframe.copy()
        self.label_column = label_column
        self.options = options
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = run_ssl_workflow(
                train_df=self.train_dataframe,
                test_df=self.test_dataframe,
                label=self.label_column,
                pretrained_model=self.options["pretrained_model"]["model"],
                threshold=self.options["threshold"],
                max_iter=self.options["max_iter"],
            )
        except Exception as error:
            self.signals.error.emit(error)
            return
        self.signals.finished.emit((self.options, result))


class UnsupervisedTrainingWorker(QRunnable):
    def __init__(self, dataframe, label_column, options):
        super().__init__()
        self.dataframe = dataframe.copy()
        self.label_column = label_column
        self.options = options
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = run_unsupervised_workflow(
                df=self.dataframe,
                label=self.label_column,
                **self.options,
            )
        except Exception as error:
            self.signals.error.emit(error)
            return
        self.signals.finished.emit((self.options, result))


class UnifiedModelTrainingWorker(QRunnable):
    def __init__(self, dataframe, label_column, added_model_entry, saved_models):
        super().__init__()
        self.dataframe = dataframe.copy()
        self.label_column = label_column
        self.added_model_entry = dict(added_model_entry)
        self.saved_models = list(saved_models)
        self.signals = WorkerSignals()

    @staticmethod
    def _prepare_ssl_training_frame(dataframe, label_column):
        frame = dataframe.copy()
        if frame.empty or not label_column or label_column not in frame.columns:
            raise ValueError(
                "Semi-supervised training requires a dataset with a valid label column."
            )

        labels = frame[label_column]
        missing_mask = labels.isna()
        try:
            blank_mask = labels.astype("string").str.strip().eq("").fillna(False)
            if bool(blank_mask.any()):
                frame.loc[blank_mask, label_column] = np.nan
                missing_mask = missing_mask | blank_mask
        except Exception:
            pass

        labeled_count = int((~missing_mask).sum())
        missing_count = int(missing_mask.sum())

        if labeled_count == 0:
            raise ValueError(
                "Semi-supervised training requires at least one known label."
            )
        if missing_count == 0:
            raise ValueError(
                "Semi-supervised training requires at least one missing label."
            )

        return frame

    @staticmethod
    def _resolve_base_model_reference(token):
        text = str(token or "").strip()
        if not text:
            return "", ""
        if ":" not in text:
            if text in {
                "logistic_regression",
                "random_forest",
                "decision_tree",
                "knn",
                "svm",
            }:
                return "builtin", text
            return "saved", text
        source, value = text.split(":", 1)
        return source.strip().lower(), value.strip()

    def run(self):
        try:
            category = self.added_model_entry["category"]
            algorithm = self.added_model_entry["algorithm"]
            common = self.added_model_entry.get("common_parameters", {})
            parameters = ModelController.build_training_parameters(
                self.added_model_entry
            )

            if category == "supervised":
                result = run_supervised_workflow(
                    df=self.dataframe,
                    label_col=self.label_column,
                    model_type=algorithm,
                    parameters=parameters,
                    test_size=float(common.get("test_size", 0.3)),
                    random_state=int(common.get("random_state", 42)),
                    stratify=bool(common.get("stratify", False)),
                )
                payload = {
                    "name": self.added_model_entry["name"],
                    "category": category,
                    "algorithm": algorithm,
                    "result": result,
                    "trained_model": result["model"],
                    "metrics": result.get("metrics", {}),
                    # Save model-facing columns after shared training preparation
                    "feature_columns": result.get("features", []),
                    # Save raw dataset columns so Tab 4 can rebuild model input
                    "input_feature_columns": [
                        str(column)
                        for column in self.dataframe.columns
                        if str(column) != str(self.label_column)
                    ],
                    "preprocessing": {
                        "prepare_training_features": True,
                        "fill_method": "median",
                        "fill_value": None,
                    },
                    "parameters": parameters,
                }
            elif category == "semi_supervised":
                base_token = parameters.get("base_model_name", "")
                base_source, base_reference = self._resolve_base_model_reference(
                    base_token
                )
                use_pretrained_state = False
                base_features = None
                base_model = None

                if base_source == "saved":
                    base_model_info = next(
                        (
                            model
                            for model in self.saved_models
                            if model.get("display_name") == base_reference
                            and model.get("category") == "supervised"
                        ),
                        None,
                    )
                    if base_model_info is None:
                        raise ValueError(
                            "Select a valid saved supervised base model."
                        )
                    model_label = str(
                        base_model_info.get("label", "")
                    ).strip()
                    if (
                        model_label
                        and self.label_column
                        and model_label != self.label_column
                    ):
                        raise ValueError(
                            "The selected saved base model label does not match "
                            "the active dataset label."
                        )
                    base_model = base_model_info.get("model")
                    base_features = (
                        base_model_info.get("feature_columns") or None
                    )
                    if base_features and not self.dataframe.empty:
                        missing = [
                            column
                            for column in base_features
                            if column not in self.dataframe.columns
                        ]
                        if missing:
                            raise ValueError(
                                "The selected saved base model requires features "
                                "not found in the current dataset."
                            )
                    use_pretrained_state = True

                elif base_source == "exported":
                    model_path = Path(base_reference)
                    if not model_path.exists():
                        raise ValueError(
                            "The selected exported base model file was not found."
                        )
                    imported_payload = joblib.load(model_path)
                    (
                        imported_model,
                        package_metadata,
                        _,
                    ) = unpack_model_package(imported_payload)
                    imported_category = str(
                        package_metadata.get("category", "")
                    ).strip().lower()
                    if (
                        imported_model is None
                        or imported_category != "supervised"
                    ):
                        raise ValueError(
                            "Only supervised exported models can be used for "
                            "self-training."
                        )
                    exported_label = str(
                        package_metadata.get("label", "")
                    ).strip()
                    if (
                        exported_label
                        and self.label_column
                        and exported_label != self.label_column
                    ):
                        raise ValueError(
                            "The selected exported base model label does not "
                            "match the active dataset label."
                        )
                    base_features = [
                        str(column)
                        for column in package_metadata.get(
                            "feature_columns",
                            [],
                        )
                        or []
                    ]
                    if base_features and not self.dataframe.empty:
                        missing = [
                            column
                            for column in base_features
                            if column not in self.dataframe.columns
                        ]
                        if missing:
                            raise ValueError(
                                "The selected exported base model requires "
                                "features not found in the current dataset."
                            )
                    base_model = imported_model
                    use_pretrained_state = True

                elif base_source == "builtin":
                    base_model = build_model(
                        base_reference,
                        parameters={},
                        random_state=int(common.get("random_state", 42)),
                    )

                else:
                    raise ValueError(
                        "Select a built-in supervised base model, saved "
                        "supervised model, or exported supervised PKL model."
                    )

                ssl_train_df = self._prepare_ssl_training_frame(
                    dataframe=self.dataframe,
                    label_column=self.label_column,
                )

                result = run_ssl_workflow(
                    train_df=ssl_train_df,
                    test_df=ssl_train_df,
                    label=self.label_column,
                    pretrained_model=base_model,
                    features=base_features,
                    threshold=float(parameters.get("threshold", 0.9)),
                    max_iter=int(parameters.get("max_iter", 10)),
                    criterion=parameters.get("criterion", "threshold"),
                    k_best=int(parameters.get("k_best", 10)),
                    test_size=float(common.get("test_size", 0.3)),
                    random_state=int(common.get("random_state", 42)),
                    verbose=bool(common.get("verbose", False)),
                    use_pretrained_state=use_pretrained_state,
                )
                parameters["pretrained_state_used"] = bool(
                    getattr(
                        result["ssl_model"],
                        "pretrained_state_used_",
                        False,
                    )
                )
                payload = {
                    "name": self.added_model_entry["name"],
                    "category": category,
                    "algorithm": algorithm,
                    "result": result,
                    "trained_model": result["ssl_model"],
                    "metrics": result.get("metrics", {}),
                    "feature_columns": result.get("features", []),
                    "parameters": parameters,
                }
            else:
                result = run_unsupervised_workflow(
                    df=self.dataframe,
                    method=algorithm,
                    label=self.label_column,
                    random_state=int(common.get("random_state", 42)),
                    **parameters,
                )
                payload = {
                    "name": self.added_model_entry["name"],
                    "category": category,
                    "algorithm": algorithm,
                    "result": result,
                    "trained_model": result["model"],
                    "metrics": result.get("metrics", {}),
                    "feature_columns": result.get("features", []),
                    "parameters": parameters,
                }
        except Exception as error:
            self.signals.error.emit(error)
            return

        self.signals.finished.emit(payload)
