import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import pandas as pd

@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float | None = None
    error: str | None = None

# Operations for recorded/live playback  and annotation
class PlaybackAnnotationManager:

    ANNOTATION_COLUMNS = ["row_index", "timestamp", "predicted_label", "corrected_label", "annotation_note"]

    def __init__(self, max_stream_rows: int = 100_000) -> None:
        self.dataset = pd.DataFrame()
        self.annotations = pd.DataFrame(columns=self.ANNOTATION_COLUMNS)
        self.current_row = 0
        self.model: Any = None
        self.model_path: Optional[str] = None
        self.prediction_dataset = pd.DataFrame()
        self.max_stream_rows = max(100, int(max_stream_rows))

    @staticmethod
    def active_dataset(working_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        if working_df is not None and not working_df.empty:
            return working_df
        if original_df is not None and not original_df.empty:
            return original_df
        return pd.DataFrame()

    def set_dataset(self, dataset: pd.DataFrame, copy: bool = False) -> pd.DataFrame:
        self.dataset = dataset.copy() if copy and dataset is not None else (dataset if dataset is not None else pd.DataFrame())
        self.prediction_dataset = pd.DataFrame()
        if self.dataset.empty:
            self.current_row = 0
        else:
            self.current_row = int(max(0, min(self.current_row, len(self.dataset) - 1)))
        return self.dataset

    def clear_stream(self) -> None:
        self.dataset = pd.DataFrame()
        self.prediction_dataset = pd.DataFrame()
        self.current_row = 0

    def append_stream_sample(self, sample: dict[str, Any]) -> int:
        if not isinstance(sample, dict) or not sample:
            raise ValueError("A streamed sample must be a non-empty object.")
        self.dataset = pd.concat([self.dataset, pd.DataFrame([sample])], ignore_index=True, sort=False)
        if len(self.dataset) > self.max_stream_rows:
            dropped = len(self.dataset) - self.max_stream_rows
            self.dataset = self.dataset.iloc[dropped:].reset_index(drop=True)
            if not self.annotations.empty:
                self.annotations = self.annotations[self.annotations["row_index"] >= dropped].copy()
                self.annotations["row_index"] = self.annotations["row_index"] - dropped
        self.current_row = len(self.dataset) - 1
        return self.current_row

    def seek(self, row: int) -> int:
        if self.dataset.empty:
            self.current_row = 0
            return 0
        self.current_row = int(max(0, min(int(row), len(self.dataset) - 1)))
        return self.current_row

    def advance(self) -> Optional[int]:
        if self.dataset.empty or self.current_row >= len(self.dataset) - 1:
            return None
        return self.seek(self.current_row + 1)

    def current_sample(self) -> pd.DataFrame:
        return pd.DataFrame() if self.dataset.empty else self.dataset.iloc[[self.current_row]].copy()

    def signal_window(self, rows_back: int = 100) -> pd.DataFrame:
        if self.dataset.empty:
            return pd.DataFrame()
        start = max(0, self.current_row - int(rows_back))
        return self.dataset.iloc[start:self.current_row + 1]

    def numeric_signal_columns(self, label_col: str | None = None) -> list[str]:
        if self.dataset.empty:
            return []
        return [col for col in self.dataset.select_dtypes(include="number").columns if str(col) != str(label_col)]

    def load_model(self, path: str | Path) -> Any:
        path = Path(path)
        try:
            import joblib
            loaded = joblib.load(path)
        except Exception:
            with path.open("rb") as handle:
                loaded = pickle.load(handle)
        self.model = loaded.get("pipeline", loaded) if isinstance(loaded, dict) else loaded
        if not hasattr(self.model, "predict"):
            raise TypeError("The selected artifact does not provide a predict() method.")
        self.model_path = str(path)
        return self.model

    def set_model(self, model: Any, model_name: str | None = None) -> Any:
        if model is not None and not hasattr(model, "predict"):
            raise TypeError("Model must provide a predict() method.")
        self.model = model
        self.model_path = model_name
        return self.model

    def model_feature_names(self) -> list[str]:
        if self.model is None:
            return []

        names = getattr(self.model, "feature_names_in_", None)
        if names is not None:
            return [str(name) for name in names]

        # Some saved artifacts are sklearn Pipelines. The pipeline or its
        # first fitted step may retain the original input feature names.
        named_steps = getattr(self.model, "named_steps", None)
        if named_steps:
            for step in named_steps.values():
                names = getattr(step, "feature_names_in_", None)
                if names is not None:
                    return [str(name) for name in names]

        return []

    def predict_current(self, label_col: str | None = None, feature_columns: list[str] | None = None) -> PredictionResult:
        sample = self.current_sample()
        if sample.empty:
            return PredictionResult("no sample", error="No sample is available.")
        if self.model is None:
            if label_col and label_col in sample.columns:
                return PredictionResult(str(sample.iloc[0][label_col]))
            return PredictionResult("model not loaded", error="Load a pretrained model to classify samples.")
        try:
            if feature_columns:
                missing = [column for column in feature_columns if column not in sample.columns]
                if missing:
                    raise ValueError("Missing model features: " + ", ".join(missing))
                features = sample[feature_columns].copy()
            else:
                features = sample.drop(columns=[label_col], errors="ignore") if label_col else sample.copy()
            numeric_features = features.select_dtypes(include="number")
            if numeric_features.empty:
                raise ValueError("No numeric model features are available.")
            label = str(self.model.predict(numeric_features)[0])
            confidence = None
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(numeric_features)
                confidence = float(max(probabilities[0]))
            return PredictionResult(label=label, confidence=confidence)
        except Exception as error:
            return PredictionResult("prediction unavailable", error=str(error))


    # Classify every row and return a copy with prediction columns added. Confidence is the highest class probability returned by predict_proba().
    #  Models without predict_proba() receive an empty confidence column.
    def predict_dataset(
        self,
        feature_columns: list[str],
        prediction_column: str = "Predicted_Label",
        confidence_column: str = "Prediction_Confidence",
    ) -> pd.DataFrame:
        if self.dataset.empty:
            raise ValueError("No dataset is available for prediction.")
        if self.model is None:
            raise ValueError("Load a pretrained model before predicting the dataset.")
        if not feature_columns:
            raise ValueError("Select the model feature columns used during training.")

        missing = [column for column in feature_columns if column not in self.dataset.columns]
        if missing:
            raise ValueError("Missing model features: " + ", ".join(missing))

        features = self.dataset.loc[:, feature_columns].copy()
        non_numeric = [
            column for column in features.columns
            if not pd.api.types.is_numeric_dtype(features[column])
        ]
        if non_numeric:
            raise ValueError(
                "The current exported model expects numeric input. "
                "These selected columns are not numeric: " + ", ".join(non_numeric)
            )
        if features.isna().any().any():
            missing_counts = features.isna().sum()
            affected = [
                f"{column} ({int(count)})"
                for column, count in missing_counts.items()
                if count
            ]
            raise ValueError(
                "Prediction features contain missing values: " + ", ".join(affected)
            )

        predictions = self.model.predict(features)
        if len(predictions) != len(self.dataset):
            raise ValueError("The model returned an unexpected number of predictions.")

        result = self.dataset.copy()
        result[prediction_column] = predictions

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)
            if len(probabilities) != len(self.dataset):
                raise ValueError("The model returned an unexpected number of probabilities.")
            result[confidence_column] = probabilities.max(axis=1)
        else:
            result[confidence_column] = pd.NA

        self.prediction_dataset = result
        return result.copy()

    def export_prediction_dataset(self, path: str | Path) -> Path:
        if self.prediction_dataset.empty:
            raise ValueError("Run Predict Entire Dataset before exporting predictions.")
        path = Path(path)
        self._atomic_csv_write(self.prediction_dataset, path)
        return path

    @staticmethod
    def timestamp_column(dataset: pd.DataFrame) -> str | None:
        for column in dataset.columns:
            if "time" in str(column).lower():
                return column
        return None

    def playback_interval_ms(self, timestamp_col: str | None, fallback_ms: int) -> int:
        if not timestamp_col or timestamp_col not in self.dataset.columns or self.current_row >= len(self.dataset) - 1:
            return int(fallback_ms)
        current = self.dataset.iloc[self.current_row][timestamp_col]
        following = self.dataset.iloc[self.current_row + 1][timestamp_col]
        try:
            current_time = pd.to_datetime(current)
            next_time = pd.to_datetime(following)
            delta_ms = int((next_time - current_time).total_seconds() * 1000)
        except Exception:
            try:
                delta_ms = int((float(following) - float(current)) * 1000)
            except Exception:
                return int(fallback_ms)
        return max(10, min(delta_ms if delta_ms > 0 else int(fallback_ms), 10_000))

    def add_or_update_annotation(self, corrected_label: str, predicted_label: str, timestamp_col: str | None = None, annotation_note: str = "", row_index: int | None = None) -> pd.DataFrame:
        if self.dataset.empty:
            raise ValueError("No dataset is available for annotation.")
        corrected_label = str(corrected_label).strip()
        if not corrected_label:
            raise ValueError("A corrected label/state is required.")
        row_index = self.current_row if row_index is None else self.seek(row_index)
        resolved_timestamp_col = timestamp_col if timestamp_col in self.dataset.columns else self.timestamp_column(self.dataset)
        timestamp = self.dataset.iloc[row_index][resolved_timestamp_col] if resolved_timestamp_col else row_index
        new_row = pd.DataFrame([{"row_index": row_index, "timestamp": timestamp, "predicted_label": predicted_label, "corrected_label": corrected_label, "annotation_note": str(annotation_note).strip()}])
        existing = self.annotations if "row_index" in self.annotations.columns else pd.DataFrame(columns=self.ANNOTATION_COLUMNS)
        self.annotations = pd.concat([existing[existing["row_index"] != row_index], new_row], ignore_index=True).sort_values("row_index")
        return self.annotations.copy()

    def corrected_dataset(self) -> pd.DataFrame:
        if self.dataset.empty:
            raise ValueError("No dataset is available for export.")
        if self.annotations.empty:
            raise ValueError("No corrected labels have been created.")
        export_df = self.dataset.copy()
        export_df["corrected_label"] = export_df.get("corrected_label", pd.Series(pd.NA, index=export_df.index))
        column_index = export_df.columns.get_loc("corrected_label")
        for _, annotation in self.annotations.iterrows():
            row_position = int(annotation["row_index"])
            if 0 <= row_position < len(export_df):
                export_df.iloc[row_position, column_index] = annotation["corrected_label"]
        return export_df


    @staticmethod
    def _atomic_csv_write(frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, dir=path.parent, newline="", encoding="utf-8") as handle:
            temp_path = Path(handle.name)
        try:
            frame.to_csv(temp_path, index=False)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    def export_corrected_dataset(self, path: str | Path) -> Path:
        path = Path(path)
        self._atomic_csv_write(self.corrected_dataset(), path)
        return path

    def export_annotation_log(self, path: str | Path) -> Path:
        path = Path(path)
        if self.annotations.empty:
            raise ValueError("No annotations are available for export.")
        self._atomic_csv_write(self.annotations, path)
        return path