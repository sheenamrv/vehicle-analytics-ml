"""Dataset import, column selection, data quality, and preview behavior."""

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data.process import load_project
from src.data.test_load import (
    change_dtype,
    get_available_columns,
    get_datasets,
    select_col,
)
from src.frontend.charts import ChartCanvas
from src.frontend.data_quality_dialog import DataQualityDialog
from src.frontend.data_summary import file_summary, missing_summary
from src.frontend.widgets import (
    ColumnPicker,
    allowed_dtypes,
    data_panel,
    divider,
    primary_button,
    secondary_button,
    section_label,
    sidebar_base,
    table_view,
    taller_dropdown,
)
from src.frontend.workers import DataLoadWorker


class DataPrepControllerMixin:
    """Data-preparation behavior mixed into the main application window."""

    def _import_sidebar(self):
        """Build controls for opening datasets/projects and defining label/features."""
        panel = sidebar_base()
        layout = panel.layout()

        layout.addWidget(section_label("DATASET"))
        self.dataset_button = primary_button("Upload File")
        self.dataset_button.clicked.connect(self.browse_dataset)
        layout.addWidget(QLabel("Upload Dataset"))
        layout.addWidget(self.dataset_button)

        self.project_button = secondary_button("Open Project")
        self.project_button.clicked.connect(self.browse_project)
        layout.addWidget(self.project_button)
        layout.addWidget(divider())

        layout.addWidget(section_label("CONFIGURATION"))
        self.dataset_combo = taller_dropdown(QComboBox())
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)
        layout.addWidget(QLabel("Dataset"))
        layout.addWidget(self.dataset_combo)

        self.label_combo = taller_dropdown(QComboBox())

        self.label_combo.currentTextChanged.connect(self.update_selected_columns)

        layout.addWidget(QLabel("Label"))
        layout.addWidget(self.label_combo)

        layout.addWidget(QLabel("Features"))
        self.column_picker = ColumnPicker("Search columns")
        self.column_picker.setMinimumHeight(190)
        self.column_picker.selectionChange.connect(self.update_selected_columns)
        layout.addWidget(self.column_picker)

        self.select_all_columns_button = secondary_button("Select All Columns")
        self.select_all_columns_button.clicked.connect(self.select_all_columns)
        layout.addWidget(self.select_all_columns_button)

        self.project_name = QLineEdit()
        self.project_name.setMinimumWidth(210)
        self.project_name.setPlaceholderText("Project name")
        self.project_name.textChanged.connect(self.on_project_name_changed)
        layout.addWidget(QLabel("Project Name"))
        layout.addWidget(self.project_name)

        self.create_project_button = primary_button("Create Project")
        self.create_project_button.clicked.connect(self.create_project)
        layout.addWidget(self.create_project_button)
        layout.addStretch()
        return panel

    def _build_import_page(self):
        """Build file preview, summary, and data-quality panels."""
        page = QWidget()
        layout = QGridLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(20)
        layout.setVerticalSpacing(16)

        preview = data_panel("DATASET PREVIEW", self.preview_model)
        summary = data_panel("FILE SUMMARY", self.file_summary_model)
        # missing = data_panel("MISSING VALUES SUMMARY", self.missing_model)

        missing_widget = QWidget()
        missing_layout = QVBoxLayout(missing_widget)

        title = QLabel("MISSING VALUES SUMMARY")
        title.setProperty("panelTitle", True)

        self.missing_table = table_view(self.missing_model)
        self.missing_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        missing_layout.addWidget(title)
        missing_layout.addWidget(self.missing_table)
        self.missing_table.setMinimumHeight(220)

        # Missing-value chart lives in Import because it describes data quality.
        self.missing_chart = ChartCanvas(
            "Open a dataset to profile missing values.",
            min_height=180,
        )

        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)
        preview_title = QLabel("DATASET PREVIEW")
        preview_title.setProperty("panelTitle", True)
        preview_layout.addWidget(preview_title)
        self.preview_table = table_view(self.preview_model)
        self.preview_table.setContextMenuPolicy(Qt.NoContextMenu)
        self.preview_table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.preview_table.horizontalHeader().customContextMenuRequested.connect(self.on_preview_header_menu)
        preview_layout.addWidget(self.preview_table)

        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(8)
        self.preview_prev_button = secondary_button("Previous")
        self.preview_prev_button.setMaximumWidth(80)
        self.preview_prev_button.clicked.connect(lambda: self.change_preview_page(-1))
        self.preview_next_button = secondary_button("Next")
        self.preview_next_button.setMaximumWidth(80)
        self.preview_next_button.clicked.connect(lambda: self.change_preview_page(1))
        self.preview_page_label = QLabel("Rows 0-0 of 0")
        nav_layout.addWidget(self.preview_prev_button)
        nav_layout.addWidget(self.preview_next_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self.preview_page_label)
        preview_layout.addLayout(nav_layout)

        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(6)
        summary_title = QLabel("FILE SUMMARY")
        summary_title.setProperty("panelTitle", True)
        summary_layout.addWidget(summary_title)
        self.file_summary_table = table_view(self.file_summary_model)
        self.file_summary_table.setSelectionBehavior(QTableView.SelectRows)
        self.file_summary_table.selectionModel().selectionChanged.connect(self.on_summary_selection_changed)
        self.file_summary_table.setMinimumHeight(280)
        summary_layout.addWidget(self.file_summary_table)

        layout.addWidget(preview_widget, 0, 0, 1, 3)
        layout.addWidget(summary_widget, 0, 3, 1, 1)
        layout.addWidget(missing_widget, 1, 0, 1, 4)
        layout.addWidget(self.missing_chart, 2, 0, 1, 4)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 3)
        layout.setColumnStretch(2, 3)
        layout.setColumnStretch(3, 2)
        layout.setColumnMinimumWidth(3, 210)
        layout.setRowStretch(0, 3)
        layout.setRowStretch(1, 2)
        layout.setRowStretch(2, 1)
        self.import_page = page
        self.import_page_scroll = self._wrap_main_page(page)
        self.main_stack.addWidget(self.import_page_scroll)

    def change_column_type(self, column_name, new_dtype):
        df = self.working_df if not self.working_df.empty else self.og_df
        if df.empty or column_name not in df.columns:
            QMessageBox.warning(self, "No Data", f"Column '{column_name}' is not available for conversion.")
            return

        current_dtype = str(df[column_name].dtype)
        if current_dtype == new_dtype:
            return

        series = df[column_name]
        if new_dtype in ("int", "int64") and pd.api.types.is_float_dtype(series.dtype):
            fractional = series.dropna() % 1 != 0
            if fractional.any():
                result = QMessageBox.question(
                    self,
                    "Convert float to integer",
                    f"Column '{column_name}' contains non-integer float values.\n\n" \
                    "Press Yes to round up, No to round down, or Cancel to abort.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if result == QMessageBox.StandardButton.Cancel:
                    return
                if result == QMessageBox.StandardButton.Yes:
                    series = np.ceil(series)
                else:
                    series = np.floor(series)
                try:
                    converted = self._convert_series_dtype(series, new_dtype)
                except Exception as error:
                    QMessageBox.warning(
                        self,
                        "Conversion Failed",
                        f"Could not convert '{column_name}' to {new_dtype}:\n\n{error}"
                    )
                    return
                if self.working_df.empty:
                    self.working_df = df.copy()
                self.working_df[column_name] = converted
                self._set_dirty(True)
                self.refresh_column_pickers()
                self.refresh_import_tables()
                self.populate_visualization_controls()
                return

        if not self._can_convert_dtype(series, new_dtype):
            QMessageBox.warning(
                self,
                "Conversion Not Allowed",
                f"Column '{column_name}' cannot be converted to {new_dtype}.\n\nPlease change the type or correct the column values first."
            )
            return

        try:
            if self.working_df.empty:
                self.working_df = df.copy()
            self.working_df = change_dtype(
                self.working_df,
                column_name,
                new_dtype
            )
            self._set_dirty(True)
            self.refresh_column_pickers()
            self.refresh_import_tables()
            self.populate_visualization_controls()

        except Exception as e:
            QMessageBox.warning(
                self,
                "Conversion Failed",
                f"Could not convert '{column_name}' to {new_dtype}\n\n{e}"
            )
            self.refresh_import_tables()

    def _canonical_dtype(self, dtype_name):
        normalized = str(dtype_name).strip().lower()
        if normalized in ("int", "int64", "int32", "int16", "int8"):
            return "int64"
        if normalized in ("float", "float64", "float32", "float16"):
            return "float64"
        if normalized in ("bool", "boolean", "bool8"):
            return "boolean"
        if normalized in ("string", "str", "object", "string[python]", "string[pyarrow]", "str"):
            return "string"
        return dtype_name

    def _can_convert_dtype(self, series, dtype):
        canonical = self._canonical_dtype(dtype)
        try:
            if canonical == "int64":
                pd.to_numeric(series, errors="raise").astype("Int64")
            elif canonical == "float64":
                pd.to_numeric(series, errors="raise").astype("float64")
            elif canonical == "string":
                series.astype("string")
            elif canonical == "boolean":
                series.astype("boolean")
            else:
                series.astype(canonical)
            return True
        except Exception:
            return False

    def _convert_series_dtype(self, series, dtype, coerce=False):
        canonical = self._canonical_dtype(dtype)
        if canonical == "int64":
            if coerce:
                return pd.to_numeric(series, errors="coerce").astype("Int64")
            return pd.to_numeric(series, errors="raise").astype("Int64")
        if canonical == "float64":
            if coerce:
                return pd.to_numeric(series, errors="coerce").astype("float64")
            return pd.to_numeric(series, errors="raise").astype("float64")
        if canonical == "string":
            return series.astype("string")
        if canonical == "boolean":
            if coerce:
                converted = series.astype("string")
                return converted.replace({"True": True, "False": False}).astype("boolean")
            return series.astype("boolean")
        return series.astype(canonical)

    def browse_dataset(self):
        """Open a dataset and populate controls from its columns."""
        if not self.maybe_save_current_project("opening a new dataset"):
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            str(self.downloads_dir),
            "Supported Files (*.csv *.xlsx *.xls *.mat);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;MATLAB Files (*.mat)",
        )
        if not path:
            return

        self.file_path = Path(path)
        self.current_project_path = None
        self.project = None
        self._active_project_name = ""
        self.project_name.clear()
        self.reset_workflow_state()
        self._suppress_dirty = True
        try:
            datasets = get_datasets(self.file_path)
        except Exception as error:
            self._suppress_dirty = False
            self._set_dirty(False)
            self.show_error("Dataset Error", error)
            return

        if not datasets:
            QMessageBox.warning(self, "Dataset Error", "No datasets found in the selected file.")
            return

        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItems(datasets)
        self.dataset_combo.blockSignals(False)

        if datasets:
            self.dataset = datasets[0]
            self.load_dataset_metadata()
        self._suppress_dirty = False
        self._set_dirty(False)

    def browse_feature_dataset(self):
        """Import a precomputed feature table without replacing the raw dataset."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Feature Dataset",
            str(Path.home() / "Downloads"),
            "Supported Files (*.csv *.xlsx *.xls *.mat);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;MATLAB Files (*.mat)",
        )
        if not path:
            return

        try:
            datasets = get_datasets(path)
            if not datasets:
                QMessageBox.warning(
                    self,
                    "Feature Dataset Error",
                    "No datasets found in the selected file.",
                )
                return

            dataset = datasets[0]
            if len(datasets) > 1:
                dataset, accepted = QInputDialog.getItem(
                    self,
                    "Select Feature Dataset",
                    "Dataset",
                    datasets,
                    0,
                    False,
                )
                if not accepted:
                    return

            columns = get_available_columns(path, dataset)
            feature_df = select_col(path, dataset, columns)
        except Exception as error:
            self.show_error("Feature Dataset Error", error)
            return

        self.feature_df = feature_df
        self.feature_summary_meta = {}
        self._set_dirty(True)
        self.refresh_feature_tables()
        self.on_workflow_tab_changed(1)

    def browse_project(self):
        """Open an ICP project and restore the saved frontend state."""
        if not self.maybe_save_current_project("opening a new project"):
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            str(self.downloads_dir),
            "ICP Project Files (*.icp);;All Files (*.*)",
        )
        if not path:
            return

        self.reset_workflow_state()

        try:
            self.project, self.og_df, self.working_df, self.feature_df = load_project(path)
        except Exception as error:
            self.show_error("Project Error", error)
            return

        self.current_project_path = Path(path)
        self.file_path = Path(self.project.get("file_path", ""))
        self.dataset = self.project.get("dataset", "Data")
        self.columns = list(self.og_df.columns)
        self._active_project_name = str(self.project.get("project_name") or self.current_project_path.stem or "").strip()
        self.project["project_name"] = self._active_project_name
        self._suppress_dirty = True
        self.project_name.setText(self._active_project_name)

        # Populate original_dtypes in-place so delegates stay in sync.
        self.original_dtypes.clear()
        for col in self.og_df.columns:
            self.original_dtypes[col] = str(self.og_df[col].dtype)

        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItem(self.dataset)
        self.dataset_combo.blockSignals(False)

        self.populate_column_controls()
        # Ensure the working dataframe matches the project's selected columns.
        self.update_selected_columns()
        self.feature_summary_meta = {}
        self.refresh_feature_tables()
        self._suppress_dirty = False
        self._set_dirty(False)
        QMessageBox.information(self, "Project Loaded", "Project loaded successfully.")

    def on_dataset_changed(self, dataset):
        if not dataset or not self.file_path:
            return

        previous_dataset = self.dataset
        if not self.maybe_save_current_project("changing the dataset"):
            self.dataset_combo.blockSignals(True)
            self.dataset_combo.setCurrentText(previous_dataset)
            self.dataset_combo.blockSignals(False)
            return

        self.dataset = dataset
        self.current_project_path = None
        self.project = None
        self._active_project_name = ""
        self.reset_workflow_state()
        self.load_dataset_metadata()

    def maybe_save_current_project(self, action="continue"):
        if not self.is_dirty:
            return True

        result = QMessageBox.question(
            self,
            "Unsaved Changes",
            f"Save current changes before {action}?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )

        if result == QMessageBox.StandardButton.Save:
            return self.save_current_project()
        if result == QMessageBox.StandardButton.Discard:
            return True
        return False

    def load_dataset_metadata(self):
        """Read columns and preview data for the selected sheet/dataset."""
        try:
            self.columns = get_available_columns(self.file_path, self.dataset)
        except Exception as error:
            self.show_error("Load Error", error)
            return

        self.preview_page = 0
        self.og_df = pd.DataFrame()
        self.working_df = pd.DataFrame()
        self.original_dtypes.clear()

        if not self.project_name.text().strip() and self.file_path:
            self.project_name.setText(self.file_path.stem)

        self.populate_column_controls()
        self.refresh_import_tables()

        worker = DataLoadWorker(self.file_path, self.dataset, self.columns)
        worker.signals.finished.connect(self.on_dataset_loaded)
        worker.signals.error.connect(lambda err: self.show_error("Load Error", err))
        self.thread_pool.start(worker)

    def on_dataset_loaded(self, df):
        self.og_df = df
        self.working_df = self.og_df.copy()
        self.original_dtypes.clear()
        for col in self.og_df.columns:
            self.original_dtypes[col] = str(self.og_df[col].dtype)

        self.populate_column_controls()
        self.refresh_import_tables()
        self.refresh_realtime_dataset()

    def _refresh_label_combo(self):
        active_columns = list(self.working_df.columns) if not self.working_df.empty else list(self.og_df.columns if not self.og_df.empty else self.columns)
        current_label = self.get_selected_label()
        self.label_combo.blockSignals(True)
        self.label_combo.clear()
        self.label_combo.addItem("(None)")
        self.label_combo.addItems([str(col) for col in active_columns])
        if current_label and current_label in active_columns:
            self.label_combo.setCurrentText(current_label)
        elif active_columns:
            self.label_combo.setCurrentIndex(0)
        self.label_combo.blockSignals(False)

    def populate_column_controls(self):
        """Refresh every control whose choices come from dataset columns."""
        self._clear_picker_searches()
        self.column_picker.blockSignals(True)
        self.column_picker.set_items(self.columns, checked=True)
        self.column_picker.blockSignals(False)

        self._refresh_label_combo()

        selected = self.project.get("selected_columns", self.columns) if self.project else self.columns
        label = self.project.get("label_column") if self.project else None
        # Prevent programmatic selection from marking the project dirty.
        self.column_picker.blockSignals(True)
        self.feature_numeric_picker.blockSignals(True)
        self.feature_non_numeric_picker.blockSignals(True)
        self.analysis_numeric_picker.blockSignals(True)
        self.analysis_non_numeric_picker.blockSignals(True)
        try:
            self.column_picker.set_selected(selected)
            selected_columns = [str(col) for col in selected if str(col) in self.columns]
            feature_selected_only = not self.feature_use_raw
            analysis_selected_only = not self.analysis_use_raw
            feature_numeric_columns = self.numeric_columns(
                use_raw=self.feature_use_raw,
                selected_only=feature_selected_only,
                include_label=True,
            )
            feature_non_numeric_columns = self.non_numeric_columns(
                use_raw=self.feature_use_raw,
                selected_only=feature_selected_only,
                include_label=True,
            )
            analysis_numeric_columns = self.numeric_columns(
                use_raw=self.analysis_use_raw,
                selected_only=analysis_selected_only,
                include_label=True,
            )
            analysis_non_numeric_columns = self.non_numeric_columns(
                use_raw=self.analysis_use_raw,
                selected_only=analysis_selected_only,
                include_label=True,
            )
            self._populate_picker_choices(self.feature_numeric_picker, feature_numeric_columns, checked=False, preserve_selection=False)
            self._populate_picker_choices(self.feature_non_numeric_picker, feature_non_numeric_columns, checked=False, preserve_selection=False)
            self._populate_picker_choices(self.analysis_numeric_picker, analysis_numeric_columns, checked=False, preserve_selection=False)
            self._populate_picker_choices(self.analysis_non_numeric_picker, analysis_non_numeric_columns, checked=False, preserve_selection=False)
            self._resize_picker_to_items(self.feature_numeric_picker, len(feature_numeric_columns))
            self._resize_picker_to_items(self.feature_non_numeric_picker, len(feature_non_numeric_columns))
            self.feature_numeric_picker.set_selected([c for c in selected_columns if c in feature_numeric_columns])
            self.feature_non_numeric_picker.set_selected([c for c in selected_columns if c in feature_non_numeric_columns])
            self.analysis_numeric_picker.set_selected([c for c in selected_columns if c in analysis_numeric_columns])
            self.analysis_non_numeric_picker.set_selected([c for c in selected_columns if c in analysis_non_numeric_columns])
        finally:
            self.column_picker.blockSignals(False)
            self.feature_numeric_picker.blockSignals(False)
            self.feature_non_numeric_picker.blockSignals(False)
            self.analysis_numeric_picker.blockSignals(False)
            self.analysis_non_numeric_picker.blockSignals(False)
        self._sync_select_all_checkbox("feature_numeric_select_all_checkbox", self.feature_numeric_picker)
        self._sync_select_all_checkbox("feature_non_numeric_select_all_checkbox", self.feature_non_numeric_picker)
        if label:
            index = self.label_combo.findText(label)
            if index >= 0:
                self.label_combo.setCurrentIndex(index)

        self.populate_visualization_controls()

    def _populate_picker_choices(self, picker, items, checked=False, preserve_selection=False):
        picker.blockSignals(True)
        selected = picker.selected_items() if preserve_selection else []
        picker.set_items(items, checked=checked)
        if preserve_selection:
            picker.set_selected([item for item in selected if item in items])
        picker.blockSignals(False)

    def _resize_picker_to_items(self, picker, item_count, minimum_rows=2, maximum_rows=8):
        del maximum_rows
        visible_rows = max(minimum_rows, int(item_count) if item_count else minimum_rows)
        picker_height = 42 + (visible_rows * 24)
        picker.setMinimumHeight(picker_height)
        picker.setMaximumHeight(picker_height)

    def _sync_select_all_checkbox(self, checkbox, picker):
        if not hasattr(self, checkbox):
            return
        widget = getattr(self, checkbox)
        if not picker.checkboxes:
            widget.blockSignals(True)
            widget.setChecked(False)
            widget.setEnabled(False)
            widget.blockSignals(False)
            return
        widget.setEnabled(True)
        selected_count = len(picker.selected_items())
        all_selected = selected_count == len(picker.checkboxes)
        widget.blockSignals(True)
        widget.setChecked(all_selected)
        widget.blockSignals(False)

    def _get_active_dataframe(self, use_raw=False):
        if use_raw:
            return self.og_df
        return self.working_df if not self.working_df.empty else self.og_df

    def _selected_import_columns(self, use_raw=False, include_label=False):
        df = self._get_active_dataframe(use_raw=use_raw)
        available_columns = list(df.columns) if not df.empty else list(self.columns)
        selected = [str(col) for col in self.column_picker.selected_items() if str(col) in available_columns]
        label = self.get_selected_label()
        if not include_label and label and label in selected:
            selected.remove(label)
        return selected

    def _is_effectively_numeric(self, series):
        if series is None:
            return False
        return pd.api.types.is_numeric_dtype(series)

    def refresh_column_pickers(self):
        """Refresh the feature and analysis pickers from the active dataset state."""
        feature_selected_only = not self.feature_use_raw
        analysis_selected_only = not self.analysis_use_raw
        self._populate_picker_choices(
            self.feature_numeric_picker,
            self.numeric_columns(
                use_raw=self.feature_use_raw,
                selected_only=feature_selected_only,
                include_label=True,
            ),
            checked=False,
            preserve_selection=True,
        )
        self._populate_picker_choices(
            self.feature_non_numeric_picker,
            self.non_numeric_columns(
                use_raw=self.feature_use_raw,
                selected_only=feature_selected_only,
                include_label=True,
            ),
            checked=False,
            preserve_selection=True,
        )
        self._resize_picker_to_items(self.feature_numeric_picker, len(self.feature_numeric_picker.checkboxes))
        self._resize_picker_to_items(self.feature_non_numeric_picker, len(self.feature_non_numeric_picker.checkboxes))
        self._populate_picker_choices(
            self.analysis_numeric_picker,
            self.numeric_columns(
                use_raw=self.analysis_use_raw,
                selected_only=analysis_selected_only,
                include_label=True,
            ),
            checked=False,
            preserve_selection=True,
        )
        self._sync_select_all_checkbox("feature_numeric_select_all_checkbox", self.feature_numeric_picker)
        self._sync_select_all_checkbox("feature_non_numeric_select_all_checkbox", self.feature_non_numeric_picker)
        self._populate_picker_choices(
            self.analysis_non_numeric_picker,
            self.non_numeric_columns(
                use_raw=self.analysis_use_raw,
                selected_only=analysis_selected_only,
                include_label=True,
            ),
            checked=False,
            preserve_selection=True,
        )

    def _clear_picker_searches(self):
        picker_names = [
            "column_picker",
            "feature_numeric_picker",
            "feature_non_numeric_picker",
            "analysis_numeric_picker",
            "analysis_non_numeric_picker",
        ]
        for name in picker_names:
            picker = getattr(self, name, None)
            if picker is not None and hasattr(picker, "search"):
                picker.search.blockSignals(True)
                picker.search.clear()
                picker.search.blockSignals(False)
                picker.filter_items("")

        for picker in getattr(self, "feature_category_pickers", {}).values():
            if hasattr(picker, "search"):
                picker.search.blockSignals(True)
                picker.search.clear()
                picker.search.blockSignals(False)
                picker.filter_items("")

    def get_selected_label(self):
        """Return the selected label column or None when the user chose no label."""
        label = self.label_combo.currentText().strip()
        if not label or label == "(None)":
            return None
        return label

    def selected_columns(self):
        """Return selected features, always retaining the chosen label when one is selected."""
        columns = self.column_picker.selected_items()
        label = self.get_selected_label()
        if label and label not in columns:
            columns.append(label)
        return columns

    def select_all_columns(self):
        """Select all available columns without deselecting ones already selected."""
        if not self.columns:
            return
        current = set(self.column_picker.selected_items())
        all_columns = set(self.columns)
        combined = current.union(all_columns)
        self.column_picker.set_selected(list(combined))
        self.update_selected_columns()

    def update_selected_columns(self):
        """Return the selected columns, and retain the chosen label"""
        if self.og_df.empty:
            return

        columns = self.selected_columns()

        if not columns:
            self.working_df = pd.DataFrame()
        else:
            if self.working_df.empty:
                self.working_df = self.og_df[columns].copy()
            else:
                # Retain preprocessing already applied to selected columns.
                # Working-frame indices are source-row identities, so restored
                # columns must be selected with those same indices after rows
                # have been removed during data-quality cleanup.
                existing = [c for c in columns if c in self.working_df.columns]
                current = self.working_df[existing].copy()
                missing = [c for c in columns if c not in existing]
                for col in missing:
                    if col in self.og_df.columns:
                        current[col] = self.og_df.loc[current.index, col]
                self.working_df = current[columns].copy()

        self._set_dirty(True)
        # self._refresh_label_combo()
        # self.refresh_column_pickers()

        # Keep project label metadata and added-model labels synchronized with
        # the Data & Features selection.
        if self.project is not None:
            new_label = self.label_combo.currentText()
            if self.project.get("label_column") != new_label:
                self.project["label_column"] = new_label
                for added in self.project.get("added_models", []):
                    added["label"] = new_label
                self.model_sidebar.set_project_label(new_label)

        self.refresh_import_tables()
        self.populate_visualization_controls()
        self.refresh_realtime_dataset()

    def apply_selected_preprocessing(self):
        """Apply one existing preprocessing operation to the checked project columns."""
        columns = self.column_picker.selected_items()
        if not columns:
            QMessageBox.warning(self, "Missing Columns", "Select at least one column before preprocessing.")
            return

        method = self.preprocessing_method.currentText()
        applied_columns = []
        recipe = None
        if method == "Impute with Mean":
            applied_columns = self.impute_columns(columns, method="mean")
            recipe = {"operation": "impute", "strategy": "mean"}
        elif method == "Impute with Median":
            applied_columns = self.impute_columns(columns, method="median")
            recipe = {"operation": "impute", "strategy": "median"}
        elif method == "Impute with Mode":
            applied_columns = self.impute_columns(columns, method="mode")
            recipe = {"operation": "impute", "strategy": "mode"}
        elif method == "Standardize Numeric":
            applied_columns = self.standardize_columns(columns)
            recipe = {"operation": "standardize"}
        elif method == "Normalize Numeric":
            applied_columns = self.normalize_columns(columns)
            recipe = {"operation": "normalize"}

        if self.project is not None and applied_columns and recipe is not None:
            # Store only effective transformations. This keeps the project
            # recipe reproducible instead of recording rejected or no-op work.
            self.project.setdefault("preprocessing", []).append({**recipe, "columns": applied_columns})

    def reset_workflow_state(self):
        """Clear analysis and visualization state when a new dataset/project loads."""
        # Project references are cleared before asynchronous dataset loading so
        # stale feature/analysis output cannot leak into the next project.
        self.project = None
        self.feature_df = pd.DataFrame()
        self.feature_summary_meta = {}
        self.latest_correlation_matrix = pd.DataFrame()
        self.analysis_model.set_data(pd.DataFrame())
        self.visualization_model.set_data(pd.DataFrame())
        self.analysis_chart.show_empty("Run an analysis from the sidebar.")
        self.chart_canvas.show_empty("Open a dataset to visualize it.")
        if hasattr(self, "analysis_title"):
            self.analysis_title.setText("ANALYSIS PREVIEW")
        if hasattr(self, "visualization_title"):
            self.visualization_title.setText("PROJECT SUMMARY")
        if hasattr(self, "model_page"):
            self.unified_model_page.set_added_models([])
            self.unified_model_page.set_queue([])
        if hasattr(self, "playback_page"):
            self.playback_page.reset()

    def on_preview_header_menu(self, pos):
        header = self.preview_table.horizontalHeader()
        col_index = header.logicalIndexAt(pos)
        if col_index < 0 or self.preview_model.columnCount() == 0:
            return

        column_name = self.preview_model._data.columns[col_index]
        menu = QMenu(header)
        menu.addAction("Review Missing/Duplicates", self.open_quality_dialog)
        menu.addAction("Highlight Missing Rows", self.highlight_missing_rows)
        menu.addAction("Highlight Duplicate Rows", self.highlight_duplicate_rows)
        menu.addAction("Remove Duplicate Rows", self.remove_duplicate_rows)
        menu.addAction("Remove Rows with Missing Values", self.remove_missing_rows)
        menu.addSeparator()

        impute_menu = menu.addMenu("Impute Missing")
        impute_menu.addAction("Mean", lambda: self.impute_columns([column_name], method="mean"))
        impute_menu.addAction("Median", lambda: self.impute_columns([column_name], method="median"))
        impute_menu.addAction("Mode", lambda: self.impute_columns([column_name], method="mode"))
        impute_menu.addAction("Custom", lambda: self.impute_columns([column_name], method="custom"))
        menu.addAction("Revert Imputation", lambda: self.revert_imputation([column_name]))
        menu.addSeparator()

        df = self.working_df if not self.working_df.empty else self.og_df
        original_dtype = self.original_dtypes.get(column_name)
        current_dtype = str(df[column_name].dtype)

        action_group = QActionGroup(self)
        action_group.setExclusive(True)
        dtype_choices = ["int64", "float64", "string", "boolean"]
        current_canonical = self._canonical_dtype(current_dtype)
        choices = [dtype for dtype in dtype_choices if dtype != current_canonical]

        for dtype in choices:
            action = QAction(f"Change Type to {dtype}", self)
            action.setCheckable(True)
            action.setChecked(dtype == current_canonical)
            action.triggered.connect(lambda checked, dtype=dtype: self.change_column_type(column_name, dtype))
            action_group.addAction(action)
            menu.addAction(action)
        menu.addAction("Revert Data Type", lambda: self.revert_dtype_conversion([column_name]))
        menu.addSeparator()

        transform_menu = menu.addMenu("Scale / Transform")
        transform_menu.addAction("Standardize", lambda: self.standardize_columns([column_name]))
        transform_menu.addAction("Normalize", lambda: self.normalize_columns([column_name]))
        transform_menu.addAction("Revert Scale/Transform", lambda: self.revert_transform([column_name]))
        menu.exec(header.mapToGlobal(pos))

    def open_quality_dialog(self):
        dialog = DataQualityDialog(self, self.working_df if not self.working_df.empty else self.og_df)
        if dialog.exec() == QDialog.Accepted and dialog.removed_indices:
            df = self.working_df if not self.working_df.empty else self.og_df
            self.working_df = df.drop(index=dialog.removed_indices)
            self._set_dirty(True)
            self.refresh_import_tables()
            self.refresh_realtime_dataset()

    def on_summary_selection_changed(self, selected, deselected):
        if not selected.indexes():
            return
        row = selected.indexes()[0].row()
        metric_name = self.file_summary_model._data.iloc[row]["metric"]
        model = self.preview_table.selectionModel()
        model.clearSelection()

        if metric_name == "Missing Cells":
            self.highlight_missing_rows()
        elif metric_name == "Duplicate Rows":
            self.highlight_duplicate_rows()

    def impute_columns(self, columns, method="mean"):
        # Keep og_df as the immutable baseline used by the revert actions.
        df = self.working_df if not self.working_df.empty else self.og_df.copy()
        if df.empty:
            QMessageBox.warning(self, "No Data", "Load a dataset before imputing values.")
            return []

        applied_columns = []
        for col in columns:
            before = df[col].copy()
            if method == "custom":
                value, ok = QInputDialog.getText(self, "Custom Imputation", f"Value for {col}:")
                if not ok:
                    continue
                if value == "":
                    QMessageBox.warning(self, "Invalid Value", "Custom imputation value cannot be empty.")
                    continue
                impute_value = self._parse_custom_impute_value(value, df[col].dtype)
                df[col] = df[col].fillna(impute_value)
                if isinstance(impute_value, str):
                    df[col] = df[col].astype("string")
            else:
                if df[col].isna().all():
                    QMessageBox.warning(self, "Cannot Impute", f"Column '{col}' contains only missing values.")
                    continue
                if method == "mean":
                    if not pd.api.types.is_numeric_dtype(df[col].dtype):
                        QMessageBox.warning(
                            self,
                            "Invalid Imputation",
                            f"Mean imputation is not valid for column '{col}' of type {df[col].dtype}."
                        )
                        continue
                    fill = df[col].mean()
                elif method == "median":
                    if not pd.api.types.is_numeric_dtype(df[col].dtype):
                        QMessageBox.warning(
                            self,
                            "Invalid Imputation",
                            f"Median imputation is not valid for column '{col}' of type {df[col].dtype}."
                        )
                        continue
                    fill = df[col].median()
                elif method == "mode":
                    mode_series = df[col].mode()
                    fill = mode_series.iloc[0] if not mode_series.empty else pd.NA
                else:
                    fill = pd.NA

                try:
                    if pd.api.types.is_integer_dtype(df[col].dtype) and not pd.isna(fill):
                        fill = int(round(fill))
                    if pd.api.types.is_bool_dtype(df[col].dtype) and not pd.isna(fill):
                        fill = bool(fill)
                    if pd.api.types.is_string_dtype(df[col].dtype) and not pd.isna(fill):
                        fill = str(fill)
                except Exception as error:
                    QMessageBox.warning(
                        self,
                        "Imputation Not Allowed",
                        f"Cannot apply imputation to column '{col}' with dtype {df[col].dtype}: {error}"
                    )
                    continue

                try:
                    df[col] = df[col].fillna(fill)
                except Exception as error:
                    QMessageBox.warning(
                        self,
                        "Imputation Failed",
                        f"Could not impute column '{col}': {error}"
                    )
                    continue

            if not df[col].equals(before):
                applied_columns.append(col)

        self.working_df = df
        self._set_dirty(True)
        self.refresh_import_tables()
        return applied_columns

    def _parse_custom_impute_value(self, value, dtype):
        try:
            if pd.api.types.is_integer_dtype(dtype):
                return int(value)
            if pd.api.types.is_float_dtype(dtype):
                return float(value)
            if pd.api.types.is_bool_dtype(dtype):
                lower = value.strip().lower()
                if lower in ("true", "1", "yes", "y"):
                    return True
                if lower in ("false", "0", "no", "n"):
                    return False
                raise ValueError("Invalid boolean value")
        except Exception:
            return value
        return value

    def revert_imputation(self, columns):
        if self.og_df.empty:
            QMessageBox.warning(self, "No Original Data", "Original dataset is not available to revert changes.")
            return
        if self.working_df.empty:
            self.working_df = self.og_df.copy()
        for col in columns:
            if col in self.og_df.columns and col in self.working_df.columns:
                original = self.og_df[col]
                mask = original.isna() & self.working_df[col].notna()
                self.working_df.loc[mask, col] = pd.NA
        self._set_dirty(True)
        self.refresh_import_tables()

    def revert_dtype_conversion(self, columns):
        if self.og_df.empty:
            QMessageBox.warning(self, "No Original Data", "Original dataset is not available to revert data type changes.")
            return
        for col in columns:
            if col not in self.og_df.columns or col not in self.working_df.columns:
                continue
            original_dtype = self.original_dtypes.get(col)
            if not original_dtype:
                continue
            if str(self.working_df[col].dtype) == original_dtype:
                continue

            original_series = self.og_df[col]
            current_series = self.working_df[col]
            if self._canonical_dtype(original_dtype) == "float64":
                restored = pd.to_numeric(current_series.astype("string"), errors="coerce").astype("float64")
            else:
                restored = current_series.astype("object").copy()

            # Preserve imputed values, but restore original non-missing data from the source.
            original_mask = original_series.notna()
            if self._canonical_dtype(original_dtype) == "float64":
                restored.loc[original_mask] = pd.to_numeric(original_series.loc[original_mask], errors="coerce").astype("float64")
            else:
                restored.loc[original_mask] = original_series.loc[original_mask].astype("object")

            if self._can_convert_dtype(restored, original_dtype):
                try:
                    self.working_df[col] = self._convert_series_dtype(restored, original_dtype)
                except Exception as error:
                    QMessageBox.warning(
                        self,
                        "Revert Failed",
                        f"Could not revert '{col}' to original dtype {original_dtype}:\n\n{error}"
                    )
            else:
                result = QMessageBox.question(
                    self,
                    "Revert Data Type",
                    f"Column '{col}' contains values that cannot be safely converted to original dtype '{original_dtype}'.\n\n" \
                    "Press Yes to attempt conversion and coerce invalid values to missing, or No to keep the current column dtype.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if result == QMessageBox.StandardButton.Yes:
                    try:
                        self.working_df[col] = self._convert_series_dtype(restored, original_dtype, coerce=True)
                    except Exception as error:
                        QMessageBox.warning(
                            self,
                            "Revert Failed",
                            f"Could not coerce '{col}' to {original_dtype}:\n\n{error}"
                        )
        self._set_dirty(True)
        self.refresh_import_tables()

    def revert_transform(self, columns):
        if self.og_df.empty:
            QMessageBox.warning(self, "No Original Data", "Original dataset is not available to revert transformations.")
            return
        if self.working_df.empty:
            self.working_df = self.og_df.copy()
        for col in columns:
            if col in self.og_df.columns and col in self.working_df.columns:
                original = self.og_df[col]
                current = self.working_df[col]
                mask = original.isna() & current.notna()
                restored = original.copy()
                restored.loc[mask] = current.loc[mask]
                try:
                    restored = restored.astype(current.dtype)
                except Exception:
                    pass
                self.working_df[col] = restored
        self._set_dirty(True)
        self.refresh_import_tables()

    def standardize_columns(self, columns):
        df = self.working_df if not self.working_df.empty else self.og_df.copy()
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_cols:
            QMessageBox.warning(self, "Standardize", "Select numeric columns to standardize.")
            return []
        before = df[numeric_cols].copy()
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.working_df = df
        self._set_dirty(True)
        self.refresh_import_tables()
        return [col for col in numeric_cols if not df[col].equals(before[col])]

    def normalize_columns(self, columns):
        df = self.working_df if not self.working_df.empty else self.og_df.copy()
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_cols:
            QMessageBox.warning(self, "Normalize", "Select numeric columns to normalize.")
            return []
        before = df[numeric_cols].copy()
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.working_df = df
        self._set_dirty(True)
        self.refresh_import_tables()
        return [col for col in numeric_cols if not df[col].equals(before[col])]

    def highlight_missing_rows(self):
        df = self.working_df if not self.working_df.empty else self.og_df
        rows = [i for i, row in df.iterrows() if row.isna().any()]
        self._select_preview_rows(rows)

    def highlight_duplicate_rows(self):
        df = self.working_df if not self.working_df.empty else self.og_df
        rows = list(df[df.duplicated(keep=False)].index)
        self._select_preview_rows(rows)

    def _select_preview_rows(self, row_indices):
        model = self.preview_table.model()
        selection = self.preview_table.selectionModel()
        selection.clearSelection()
        df = self.working_df if not self.working_df.empty else self.og_df
        page_start = self.preview_page * self.preview_page_size
        page_end = page_start + self.preview_page_size
        selected_labels = set(row_indices)
        # Preview pages are positional, while cleaned working data retains
        # source row labels. Match labels within the visible positional slice.
        for preview_row, row_label in enumerate(df.index[page_start:page_end]):
            if row_label in selected_labels:
                index = model.index(preview_row, 0)
                selection.select(index, QItemSelectionModel.Select | QItemSelectionModel.Rows)

    def remove_duplicate_rows(self):
        df = self.working_df if not self.working_df.empty else self.og_df
        if df.empty:
            QMessageBox.warning(self, "No Data", "Load a dataset before removing duplicate rows.")
            return
        duplicates = df[df.duplicated(keep=False)]
        if duplicates.empty:
            QMessageBox.information(self, "No Duplicates", "No duplicate rows were found.")
            return
        if QMessageBox.question(
            self,
            "Remove Duplicate Rows",
            f"Remove {len(duplicates)} duplicate row(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        self.working_df = df.drop(index=duplicates.index)
        self._set_dirty(True)
        self.refresh_import_tables()

    def remove_missing_rows(self):
        df = self.working_df if not self.working_df.empty else self.og_df
        if df.empty:
            QMessageBox.warning(self, "No Data", "Load a dataset before removing rows with missing values.")
            return
        missing_rows = df[df.isna().any(axis=1)]
        if missing_rows.empty:
            QMessageBox.information(self, "No Missing Values", "No rows with missing values were found.")
            return
        if QMessageBox.question(
            self,
            "Remove Missing Rows",
            f"Remove {len(missing_rows)} row(s) containing missing values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        self.working_df = df.drop(index=missing_rows.index)
        self._set_dirty(True)
        self.refresh_import_tables()

    def change_selected_column_dtype(self, columns):
        if not columns:
            return
        if len(columns) > 1:
            QMessageBox.warning(self, "Change Data Type", "Select only one column to change its data type.")
            return

        column = columns[0]
        df = self.working_df if not self.working_df.empty else self.og_df
        if column not in df.columns:
            return

        original_dtype = self.original_dtypes.get(column)
        current_dtype = str(df[column].dtype)
        choices = allowed_dtypes(original_dtype if original_dtype else current_dtype)
        if original_dtype and original_dtype not in choices:
            choices.insert(0, original_dtype)

        initial = choices.index(current_dtype) if current_dtype in choices else 0
        dtype, ok = QInputDialog.getItem(
            self,
            "Change Data Type",
            f"Select new dtype for '{column}':",
            choices,
            current=initial,
            editable=False,
        )
        if not ok or not dtype:
            return

        self.change_column_type(column, dtype)

    def refresh_import_tables(self):
        """Refresh import-tab preview, summary, missing table, and missing chart."""
        df = self.working_df if not self.working_df.empty else self.og_df
        self._refresh_label_combo()
        self.refresh_column_pickers()
        self.populate_visualization_controls()
        self.refresh_preview_page()
        self.file_summary_model.set_data(file_summary(df))
        self.missing_model.set_data(missing_summary(df))
        self.missing_chart.plot_missing_values(df)

    def refresh_preview_page(self):
        df = self.working_df if not self.working_df.empty else self.og_df
        if df.empty:
            self.preview_page = 0
            self.preview_model.set_data(pd.DataFrame())
            self.preview_page_label.setText("Rows 0-0 of 0")
            self.preview_prev_button.setEnabled(False)
            self.preview_next_button.setEnabled(False)
            return

        start = self.preview_page * self.preview_page_size
        end = min(start + self.preview_page_size, len(df))
        self.preview_model.set_data(df.iloc[start:end])
        self.preview_page_label.setText(f"Rows {start + 1}-{end} of {len(df)}")
        self.preview_prev_button.setEnabled(self.preview_page > 0)
        self.preview_next_button.setEnabled(end < len(df))

    def change_preview_page(self, delta):
        df = self.working_df if not self.working_df.empty else self.og_df
        if df.empty:
            return
        new_page = max(0, self.preview_page + delta)
        max_page = max(0, (len(df) - 1) // self.preview_page_size)
        self.preview_page = min(new_page, max_page)
        self.refresh_preview_page()
