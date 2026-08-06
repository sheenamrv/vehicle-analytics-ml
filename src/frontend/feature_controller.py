"""Feature selection, windowing, extraction, preview, and export behavior."""

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.feature.feature import (
    build_window_index_table,
    extract_windowed_feature_dataset,
    to_feature_label,
)
from src.frontend.image_dialogs import WindowPreviewDialog
from src.frontend.widgets import (
    ColumnPicker,
    SIDEBAR_WIDTH,
    WheelLockedComboBox,
    WheelLockedDoubleSpinBox,
    WheelLockedSpinBox,
    data_panel,
    divider,
    primary_button,
    secondary_button,
    section_label,
    sidebar_base,
    taller_dropdown,
)


class FeaturePickerCompat:
    """Compatibility shim for legacy code/tests expecting a single feature picker."""

    def __init__(self, window):
        self.window = window

    def selected_items(self):
        return self.window._selected_feature_keys()

    def set_selected(self, values):
        desired_labels = []
        key_to_label = {
            key: label
            for label, key in self.window.feature_label_to_key.items()
        }
        for value in values:
            text = str(value)
            if text in self.window.feature_label_to_key:
                desired_labels.append(text)
            elif text in key_to_label:
                desired_labels.append(key_to_label[text])

        for picker in self.window.feature_category_pickers.values():
            available = set(picker.checkboxes.keys())
            picker.set_selected([label for label in desired_labels if label in available])

FEATURE_CATEGORIES = {
    "DISTRIBUTION FEATURES": [
        ("Skewness [Numeric Only]", "skew"),
        ("Kurtosis [Numeric Only]", "kurtosis"),
    ],
    "SIGNAL FEATURES": [
        ("RMS [Numeric Only]", "rms"),
        ("Energy [Numeric Only]", "energy"),
        ("Peak-to-Peak [Numeric Only]", "p2p"),
        ("Zero Crossing Rate [Numeric Only]", "zcr"),
    ],
    "DATA QUALITY": [
        ("Missing Count [Numeric / Non-Numeric]", "missing_count"),
        ("Missing Percentage [Numeric / Non-Numeric]", "missing_pct"),
    ],
    "DESCRIPTIVE STATISTICS": [
        ("Count [Numeric / Non-Numeric]", "count"),
        ("Unique [Numeric / Non-Numeric]", "unique"),
        ("Top [Non-Numeric Only]", "top"),
        ("Frequency [Non-Numeric Only]", "freq"),
        ("Mean [Numeric Only]", "mean"),
        ("Median [Numeric / Non-Numeric]", "median"),
        ("Standard Deviation [Numeric Only]", "std"),
        ("Variance [Numeric Only]", "variance"),
        ("Min [Numeric Only]", "min"),
        ("Max [Numeric Only]", "max"),
        ("Range [Numeric / Non-Numeric]", "range"),
    ],
}


class FeatureExtractionControllerMixin:
    """Feature workflow behavior mixed into the main application window."""

    def _feature_sidebar(self):
        """Build controls for selecting columns and feature metrics."""
        panel = sidebar_base()
        layout = panel.layout()
        sidebar_control_width = SIDEBAR_WIDTH - 52

        def _lock_sidebar_width(widget):
            widget.setMinimumWidth(sidebar_control_width)
            widget.setMaximumWidth(sidebar_control_width)

        layout.addWidget(section_label("SELECT COLUMNS"))
        layout.addWidget(QLabel("Columns to Analyze"))
        numeric_header_row = QHBoxLayout()
        numeric_header_row.setContentsMargins(0, 0, 0, 0)
        numeric_header_row.setSpacing(8)
        numeric_header_label = QLabel("Numeric Columns")
        self.feature_numeric_select_all_checkbox = QCheckBox("")
        self.feature_numeric_select_all_checkbox.toggled.connect(
            lambda checked: self._set_picker_checked(self.feature_numeric_picker, checked)
        )
        numeric_header_row.addWidget(self.feature_numeric_select_all_checkbox)
        numeric_header_row.addWidget(numeric_header_label)
        numeric_header_row.addStretch()
        numeric_header_container = QWidget()
        numeric_header_container.setLayout(numeric_header_row)
        _lock_sidebar_width(numeric_header_container)
        layout.addWidget(numeric_header_container)

        self.feature_numeric_picker = ColumnPicker("Search numeric columns")
        self.feature_numeric_picker.setMinimumHeight(132)
        _lock_sidebar_width(self.feature_numeric_picker)
        layout.addWidget(self.feature_numeric_picker)

        non_numeric_header_row = QHBoxLayout()
        non_numeric_header_row.setContentsMargins(0, 0, 0, 0)
        non_numeric_header_row.setSpacing(8)
        non_numeric_header_label = QLabel("Non-Numeric Columns")
        self.feature_non_numeric_select_all_checkbox = QCheckBox("")
        self.feature_non_numeric_select_all_checkbox.toggled.connect(
            lambda checked: self._set_picker_checked(self.feature_non_numeric_picker, checked)
        )
        non_numeric_header_row.addWidget(self.feature_non_numeric_select_all_checkbox)
        non_numeric_header_row.addWidget(non_numeric_header_label)
        non_numeric_header_row.addStretch()
        non_numeric_header_container = QWidget()
        non_numeric_header_container.setLayout(non_numeric_header_row)
        _lock_sidebar_width(non_numeric_header_container)
        layout.addWidget(non_numeric_header_container)

        self.feature_non_numeric_picker = ColumnPicker("Search non-numeric columns")
        self.feature_non_numeric_picker.setMinimumHeight(132)
        _lock_sidebar_width(self.feature_non_numeric_picker)
        layout.addWidget(self.feature_non_numeric_picker)

        self.feature_numeric_picker.selectionChange.connect(
            lambda: self._sync_select_all_checkbox("feature_numeric_select_all_checkbox", self.feature_numeric_picker)
        )
        self.feature_non_numeric_picker.selectionChange.connect(
            lambda: self._sync_select_all_checkbox("feature_non_numeric_select_all_checkbox", self.feature_non_numeric_picker)
        )

        self.feature_use_raw_checkbox = QCheckBox("Use raw dataset for feature extraction")
        self.feature_use_raw_checkbox.toggled.connect(self.on_feature_use_raw_toggled)
        _lock_sidebar_width(self.feature_use_raw_checkbox)
        layout.addWidget(self.feature_use_raw_checkbox)
        layout.addWidget(divider())

        layout.addWidget(section_label("WINDOWING CONFIGURATION"))
        self.windowing_config_widgets = []
        self.windowing_enabled_checkbox = QCheckBox("Enable Sliding Window")
        self.windowing_enabled_checkbox.toggled.connect(self.on_windowing_toggled)
        _lock_sidebar_width(self.windowing_enabled_checkbox)
        layout.addWidget(self.windowing_enabled_checkbox)

        self.window_size_label = QLabel("Window Size")
        layout.addWidget(self.window_size_label)
        self.windowing_config_widgets.append(self.window_size_label)
        window_size_row = QHBoxLayout()
        window_size_row.setContentsMargins(0, 0, 0, 0)
        window_size_row.setSpacing(8)
        self.window_size_input = WheelLockedDoubleSpinBox()
        self.window_size_input.setRange(0.001, 10_000_000)
        self.window_size_input.setDecimals(3)
        self.window_size_input.setValue(100)
        self.window_size_input.setSingleStep(1)
        self.window_size_input.setMinimumHeight(28)
        self.window_unit_combo = taller_dropdown(QComboBox())
        self.window_unit_combo.addItems(["Samples", "Seconds"])
        self.window_unit_combo.currentTextChanged.connect(self.on_window_unit_changed)
        unit_width = 96
        value_width = max(120, sidebar_control_width - unit_width - 8)
        self.window_size_input.setMinimumWidth(value_width)
        self.window_size_input.setMaximumWidth(value_width)
        self.window_unit_combo.setMinimumWidth(unit_width)
        self.window_unit_combo.setMaximumWidth(unit_width)
        window_size_row.addWidget(self.window_size_input, 2)
        window_size_row.addWidget(self.window_unit_combo, 1)
        layout.addLayout(window_size_row)
        self.windowing_config_widgets.extend([self.window_size_input, self.window_unit_combo])

        self.window_sample_rate_label = QLabel("Sample Rate (Hz, used for seconds)")
        layout.addWidget(self.window_sample_rate_label)
        self.windowing_config_widgets.append(self.window_sample_rate_label)
        self.window_sample_rate_input = WheelLockedDoubleSpinBox()
        self.window_sample_rate_input.setRange(0.001, 1_000_000)
        self.window_sample_rate_input.setDecimals(3)
        self.window_sample_rate_input.setValue(1.0)
        self.window_sample_rate_input.setSingleStep(0.5)
        _lock_sidebar_width(self.window_sample_rate_input)
        layout.addWidget(self.window_sample_rate_input)
        self.windowing_config_widgets.append(self.window_sample_rate_input)

        self.window_overlap_label = QLabel("Overlap Percentage (0-90%)")
        layout.addWidget(self.window_overlap_label)
        self.windowing_config_widgets.append(self.window_overlap_label)
        self.window_overlap_input = WheelLockedSpinBox()
        self.window_overlap_input.setRange(0, 90)
        self.window_overlap_input.setValue(50)
        _lock_sidebar_width(self.window_overlap_input)
        layout.addWidget(self.window_overlap_input)
        self.windowing_config_widgets.append(self.window_overlap_input)

        self.window_type_label = QLabel("Window Type")
        layout.addWidget(self.window_type_label)
        self.windowing_config_widgets.append(self.window_type_label)
        self.window_type_combo = taller_dropdown(WheelLockedComboBox())
        self.window_type_combo.addItems(["Fixed Windows", "Sliding Windows"])
        _lock_sidebar_width(self.window_type_combo)
        layout.addWidget(self.window_type_combo)
        self.windowing_config_widgets.append(self.window_type_combo)

        self.window_partial_label = QLabel("Partial Windows")
        layout.addWidget(self.window_partial_label)
        self.windowing_config_widgets.append(self.window_partial_label)
        self.window_partial_combo = taller_dropdown(WheelLockedComboBox())
        self.window_partial_combo.addItems(["Keep Partial Windows", "Ignore Partial Windows"])
        _lock_sidebar_width(self.window_partial_combo)
        layout.addWidget(self.window_partial_combo)
        self.windowing_config_widgets.append(self.window_partial_combo)

        self.preview_windows_button = secondary_button("Preview Windows")
        self.preview_windows_button.clicked.connect(self.preview_windows)
        self.preview_windows_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        _lock_sidebar_width(self.preview_windows_button)
        layout.addWidget(self.preview_windows_button)
        self.windowing_config_widgets.append(self.preview_windows_button)

        layout.addWidget(divider())

        layout.addWidget(section_label("SELECT FEATURES"))
        self.feature_category_pickers = {}
        self.feature_category_toggles = {}
        self.feature_label_to_key = {}
        for category, feature_pairs in FEATURE_CATEGORIES.items():
            category_toggle = QCheckBox(category.title())
            category_toggle.setChecked(True)
            _lock_sidebar_width(category_toggle)
            layout.addWidget(category_toggle)

            picker = ColumnPicker(f"Search {category.lower()}", rich_feature_labels=True)
            labels = [label for label, _key in feature_pairs]
            for label, key in feature_pairs:
                self.feature_label_to_key[label] = key
            picker.set_items(labels, checked=True)

            # Keep picker height proportional to option count so each category
            # box is sized to its checkbox list instead of a fixed tall panel.
            visible_rows = max(2, len(labels))
            picker_height = min(240, 42 + (visible_rows * 24))
            picker.setMinimumHeight(picker_height)
            picker.setMaximumHeight(picker_height)
            _lock_sidebar_width(picker)

            category_toggle.toggled.connect(
                lambda checked, p=picker: self._set_picker_checked(p, checked)
            )
            self.feature_category_pickers[category] = picker
            self.feature_category_toggles[category] = category_toggle
            layout.addWidget(picker)

        self.feature_picker = FeaturePickerCompat(self)

        extract_button = primary_button("Extract Features")
        extract_button.clicked.connect(self.extract_features)
        _lock_sidebar_width(extract_button)
        layout.addWidget(extract_button)

        import_features = secondary_button("Import Feature Dataset")
        import_features.clicked.connect(self.browse_feature_dataset)
        _lock_sidebar_width(import_features)
        layout.addWidget(import_features)

        export_features = secondary_button("Export Feature Dataset")
        export_features.clicked.connect(self.export_features)
        _lock_sidebar_width(export_features)
        layout.addWidget(export_features)

        self.on_windowing_toggled(False)
        layout.addStretch()
        return panel

    def _build_feature_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(
            data_panel("FEATURE DATASET PREVIEW", self.feature_preview_model),
            3,
        )
        layout.addWidget(
            data_panel("FEATURE DATASET SUMMARY", self.feature_summary_model),
            1,
        )
        if layout.count() > 1:
            summary_panel = layout.itemAt(1).widget()
            if summary_panel is not None:
                summary_panel.setMinimumHeight(240)
        self.feature_page = page
        self.feature_page_scroll = self._wrap_main_page(page)
        self.main_stack.addWidget(self.feature_page_scroll)

    def on_feature_use_raw_toggled(self, checked):
        if hasattr(self, "feature_use_raw_checkbox") and self.feature_use_raw_checkbox.isChecked() != checked:
            self.feature_use_raw_checkbox.blockSignals(True)
            self.feature_use_raw_checkbox.setChecked(checked)
            self.feature_use_raw_checkbox.blockSignals(False)
        self.feature_use_raw = checked
        self.refresh_column_pickers()
        if checked:
            # In raw mode, default to all raw columns so the feature output
            # reflects the full source dataset without extra clicks.
            self.feature_numeric_picker.set_selected(list(self.feature_numeric_picker.checkboxes.keys()))
            self.feature_non_numeric_picker.set_selected(list(self.feature_non_numeric_picker.checkboxes.keys()))
        self.refresh_feature_tables_for_active_dataset()
        self.populate_visualization_controls()

    def _feature_source_dataframe(self):
        """Return the dataset currently selected for feature extraction."""
        use_raw = self.feature_use_raw_checkbox.isChecked() if hasattr(self, "feature_use_raw_checkbox") else self.feature_use_raw
        return self.og_df if use_raw else (self.working_df if not self.working_df.empty else self.og_df)

    def on_windowing_toggled(self, checked):
        if not hasattr(self, "windowing_config_widgets"):
            return

        enabled = bool(checked)
        for widget in self.windowing_config_widgets:
            widget.setVisible(enabled)
        self.windowing_enabled_checkbox.setVisible(True)

    def on_window_type_changed(self, value):
        del value

    def on_window_unit_changed(self, value):
        del value

    def _selected_feature_keys(self):
        keys = []
        seen = set()
        for picker in getattr(self, "feature_category_pickers", {}).values():
            for label in picker.selected_items():
                key = self.feature_label_to_key.get(label)
                if key and key not in seen:
                    keys.append(key)
                    seen.add(key)
        return keys

    def _set_picker_checked(self, picker, checked):
        picker.blockSignals(True)
        for checkbox in picker.checkboxes.values():
            checkbox.setChecked(bool(checked))
        picker.blockSignals(False)

    def _compute_regular_feature_value(self, series, feature_key):
        key = str(feature_key)
        if key == "count":
            return int(series.count())
        if key == "unique":
            return int(series.dropna().nunique())
        if key == "top":
            non_null = series.dropna()
            if non_null.empty:
                return np.nan
            modes = non_null.mode(dropna=True)
            return modes.iloc[0] if not modes.empty else np.nan
        if key == "freq":
            non_null = series.dropna()
            if non_null.empty:
                return 0
            return int(non_null.value_counts(dropna=True).iloc[0])

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                return np.nan
            values = numeric.to_numpy(dtype=float)
            if key == "mean":
                return float(np.mean(values))
            if key == "median":
                return float(np.median(values))
            if key == "std":
                return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            if key == "variance":
                return float(np.var(values))
            if key == "min":
                return float(np.min(values))
            if key == "max":
                return float(np.max(values))
            if key == "range":
                return float(np.max(values) - np.min(values))
            if key == "skew":
                return float(pd.Series(values).skew()) if len(values) > 2 else 0.0
            if key == "kurtosis":
                return float(pd.Series(values).kurt()) if len(values) > 3 else 0.0
            if key == "rms":
                return float(np.sqrt(np.mean(values ** 2)))
            if key == "energy":
                return float(np.sum(values ** 2))
            if key == "p2p":
                return float(np.ptp(values))
            if key == "zcr":
                if len(values) < 2:
                    return 0.0
                sign_changes = np.count_nonzero(np.signbit(values[1:]) != np.signbit(values[:-1]))
                return float(sign_changes / (len(values) - 1))
            if key == "missing_count":
                return int(series.isna().sum())
            if key == "missing_pct":
                return float(series.isna().mean() * 100.0) if len(series) else 0.0
            return np.nan

        if key == "median":
            return series.dropna().mode().iloc[0] if not series.dropna().empty else np.nan
        if key == "range":
            return int(series.dropna().nunique())
        if key == "missing_count":
            return int(series.isna().sum())
        if key == "missing_pct":
            return float(series.isna().mean() * 100.0) if len(series) else 0.0
        if key in ("mean", "std", "variance", "min", "max", "skew", "kurtosis", "rms", "energy", "p2p", "zcr"):
            return np.nan
        return series.count() if key == "count" else np.nan

    def _feature_summary_rows(self):
        meta = self.feature_summary_meta or {}
        selected_columns = meta.get("selected_columns", [])
        selected_features = meta.get("selected_features", [])
        total_features_created = len([
            col
            for col in self.feature_df.columns
            if col not in (
                "signal",
                "Signal",
                "Window Number",
                "Start Sample",
                "End Sample",
                "Start Time (s)",
                "End Time (s)",
            )
        ])
        feature_shape = f"{self.feature_df.shape[0]} x {self.feature_df.shape[1]}"

        return [
            {"Metric": "Windowing", "Value": "Enabled" if meta.get("windowing_enabled") else "Disabled"},
            {"Metric": "Original Dataset Size", "Value": meta.get("original_dataset_size", 0)},
            {"Metric": "Window Size", "Value": meta.get("window_size_display", "N/A")},
            {"Metric": "Step Size", "Value": meta.get("step_size_display", "N/A")},
            {"Metric": "Overlap", "Value": meta.get("overlap_display", "N/A")},
            {"Metric": "Partial Windows", "Value": meta.get("partial_policy", "N/A")},
            {"Metric": "Number of Windows Generated", "Value": meta.get("num_windows", 0)},
            {"Metric": "Numeric Columns Processed", "Value": meta.get("numeric_columns_count", 0)},
            {"Metric": "Statistical Features Extracted", "Value": meta.get("statistical_features_count", 0)},
            {"Metric": "Selected Columns", "Value": ", ".join(selected_columns) if selected_columns else "None"},
            {"Metric": "Selected Features", "Value": ", ".join(selected_features) if selected_features else "None"},
            {"Metric": "Total Features Created", "Value": total_features_created},
            {"Metric": "Feature Dataset Shape", "Value": feature_shape},
        ]

    def _windowing_config(self):
        enabled = hasattr(self, "windowing_enabled_checkbox") and self.windowing_enabled_checkbox.isChecked()
        if not enabled:
            return {"enabled": False}

        unit = self.window_unit_combo.currentText().strip().lower()
        return {
            "enabled": True,
            "window_size": float(self.window_size_input.value()),
            "window_unit": unit,
            "window_type": self.window_type_combo.currentText().strip(),
            "overlap_pct": int(self.window_overlap_input.value()),
            "keep_partial": self.window_partial_combo.currentText().strip().lower().startswith("keep"),
            "sample_rate_hz": float(self.window_sample_rate_input.value()),
        }

    def _window_size_and_step_samples(self, cfg):
        unit = cfg["window_unit"]
        window_size = float(cfg["window_size"])
        sample_rate = float(cfg["sample_rate_hz"])
        window_size_samples = max(1, int(round(window_size * sample_rate))) if unit == "seconds" else max(1, int(round(window_size)))
        if cfg["window_type"].lower().startswith("sliding"):
            step_samples = max(1, int(window_size_samples * (1 - (cfg["overlap_pct"] / 100.0))))
        else:
            step_samples = window_size_samples
        return window_size_samples, step_samples

    def _validate_windowing_inputs(self, df, numeric_signals, requested_feature_keys, cfg):
        if not cfg.get("enabled"):
            return None

        if not numeric_signals:
            return "Select at least one numeric column."

        statistical_keys = [key for key in requested_feature_keys if key not in ("missing_count", "missing_pct")]
        if not statistical_keys:
            return "Select at least one statistical feature (for example Mean, Median, Standard Deviation, Variance, RMS, Min, or Max)."

        if float(cfg["window_size"]) <= 0:
            return "Window size must be greater than zero."

        overlap = float(cfg["overlap_pct"])
        if overlap < 0 or overlap > 90:
            return "Overlap must be between 0% and 90%."

        window_size_samples, step_samples = self._window_size_and_step_samples(cfg)
        if step_samples <= 0:
            return "Step size must be greater than zero."

        if len(df.index) == 0:
            return "Dataset is empty."

        if window_size_samples > len(df.index) and not cfg.get("keep_partial"):
            return (
                "Window size is larger than the dataset. "
                "Enable 'Keep Partial Windows' or choose a smaller window size."
            )

        return None

    def preview_windows(self):
        df = self._feature_source_dataframe()
        if df.empty:
            QMessageBox.warning(self, "Missing Dataset", "Open a dataset before previewing windows.")
            return

        numeric_signals = self.feature_numeric_picker.selected_items()
        requested_feature_keys = self._selected_feature_keys()

        cfg = self._windowing_config()
        if not cfg.get("enabled"):
            QMessageBox.information(self, "Windowing Disabled", "Enable windowing to preview data segmentation.")
            return

        validation_error = self._validate_windowing_inputs(df, numeric_signals, requested_feature_keys, cfg)
        if validation_error:
            QMessageBox.warning(self, "Invalid Windowing Configuration", validation_error)
            return

        try:
            unit = cfg["window_unit"]
            sample_rate = cfg["sample_rate_hz"]
            window_size_samples, step = self._window_size_and_step_samples(cfg)

            preview_df = build_window_index_table(
                length=len(df),
                window_size_samples=window_size_samples,
                step_samples=step,
                keep_partial=cfg["keep_partial"],
                use_time=(unit == "seconds"),
                sample_rate_hz=sample_rate,
            )
        except Exception as error:
            self.show_error("Window Preview Error", error)
            return

        if preview_df.empty:
            QMessageBox.warning(self, "No Windows", "No windows were generated with the current configuration.")
            return

        dialog = WindowPreviewDialog(self, preview_df)
        dialog.exec()

    def extract_features(self):
        """Extract selected features either over full columns or over windows."""
        self._run_feature_extraction(mark_dirty=True, show_messages=True, navigate=True)

    def _run_feature_extraction(self, mark_dirty, show_messages, navigate):
        df = self._feature_source_dataframe()
        if df.empty:
            if show_messages:
                QMessageBox.warning(self, "Missing Dataset", "Open a dataset before extracting features.")
            return False

        numeric_signals = self.feature_numeric_picker.selected_items()
        non_numeric_signals = self.feature_non_numeric_picker.selected_items()
        signals = numeric_signals + non_numeric_signals
        if not signals:
            if show_messages:
                QMessageBox.warning(self, "Missing Signals", "Select at least one signal.")
            return False

        requested_feature_keys = self._selected_feature_keys()
        if not requested_feature_keys:
            if show_messages:
                QMessageBox.warning(self, "Missing Features", "Select at least one feature to extract.")
            return False

        cfg = self._windowing_config()
        selected_feature_labels = [to_feature_label(key) for key in requested_feature_keys]
        validation_error = self._validate_windowing_inputs(df, numeric_signals, requested_feature_keys, cfg)
        if validation_error:
            if show_messages:
                QMessageBox.warning(self, "Invalid Windowing Configuration", validation_error)
            return False

        try:
            if cfg.get("enabled"):
                feature_df, window_meta = extract_windowed_feature_dataset(
                    df=df,
                    numeric_columns=numeric_signals,
                    feature_keys=requested_feature_keys,
                    window_size=cfg["window_size"],
                    window_unit=cfg["window_unit"],
                    overlap_pct=cfg["overlap_pct"],
                    window_type=cfg["window_type"],
                    keep_partial=cfg["keep_partial"],
                    sample_rate_hz=cfg["sample_rate_hz"],
                )

                self.feature_df = feature_df
                window_size_samples = int(window_meta.get("window_size_samples", 0))
                step_samples = int(window_meta.get("step_samples", 0))
                self.feature_summary_meta = {
                    "windowing_enabled": True,
                    "original_dataset_size": int(len(df.index)),
                    "window_size_display": f"{cfg['window_size']} {cfg['window_unit']}",
                    "step_size_display": f"{step_samples} samples",
                    "overlap_display": f"{cfg['overlap_pct']}%",
                    "partial_policy": "Keep Partial Windows" if cfg["keep_partial"] else "Ignore Partial Windows",
                    "num_windows": window_meta.get("num_windows", len(feature_df.index)),
                    "selected_columns": list(numeric_signals),
                    "selected_features": selected_feature_labels,
                    "numeric_columns_count": len(numeric_signals),
                    "statistical_features_count": len([k for k in requested_feature_keys if k not in ("missing_count", "missing_pct")]),
                    "window_size_samples": window_size_samples,
                    "step_samples": step_samples,
                    "window_config": cfg,
                }
            else:
                rows = []
                for signal in signals:
                    series = df[signal]
                    row = {"signal": signal}
                    for feature_key in requested_feature_keys:
                        value = self._compute_regular_feature_value(series, feature_key)
                        row[feature_key] = value
                    rows.append(row)

                self.feature_df = pd.DataFrame(rows)
                self.feature_summary_meta = {
                    "windowing_enabled": False,
                    "original_dataset_size": int(len(df.index)),
                    "window_size_display": "N/A",
                    "step_size_display": "N/A",
                    "overlap_display": "N/A",
                    "partial_policy": "N/A",
                    "num_windows": 1 if not self.feature_df.empty else 0,
                    "selected_columns": list(signals),
                    "selected_features": selected_feature_labels,
                    "numeric_columns_count": len(numeric_signals),
                    "statistical_features_count": len([k for k in requested_feature_keys if k not in ("missing_count", "missing_pct")]),
                    "window_config": {"enabled": False},
                }
        except Exception as error:
            if show_messages:
                self.show_error("Feature Extraction Error", error)
            return False

        if self.project is not None:
            self.project["feature_extraction"] = {
                "columns": list(self.feature_summary_meta.get("selected_columns", [])),
                "metrics": list(requested_feature_keys),
                "windowing": self.feature_summary_meta.get("window_config", {"enabled": False}),
            }

        if mark_dirty:
            self._set_dirty(True)

        self.refresh_feature_tables()
        if navigate:
            self.on_workflow_tab_changed(1)
        return True

    def refresh_feature_tables_for_active_dataset(self):
        """Recompute feature results when the source dataset toggle changes."""
        if self.feature_df.empty:
            return
        if not self._run_feature_extraction(mark_dirty=False, show_messages=False, navigate=False):
            self.feature_df = pd.DataFrame()
            self.feature_summary_meta = {}
            self.refresh_feature_tables()

    def refresh_feature_tables(self):
        """Refresh the feature dataset preview and summary models."""
        self.feature_preview_model.set_data(self.feature_df)
        summary_rows = self._feature_summary_rows() if self.feature_summary_meta else []
        if summary_rows:
            self.feature_summary_model.set_data(pd.DataFrame(summary_rows))
        else:
            self.feature_summary_model.set_data(pd.DataFrame(columns=["Metric", "Value"]))

    def export_features(self):
        """Export the current feature table to CSV."""
        if self.feature_df.empty:
            QMessageBox.warning(
                self,
                "Missing Features",
                "Extract or import features before exporting.",
            )
            return

        default_path = self.downloads_dir / "feature_dataset.csv"
        default_path.parent.mkdir(exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Feature Dataset",
            str(default_path.resolve()),
            "CSV Files (*.csv)",
        )
        if not path:
            return

        try:
            self.feature_df.to_csv(path, index=False)
        except Exception as error:
            self.show_error("Export Error", error)
            return

        QMessageBox.information(self, "Export Complete", f"Saved {path}")
