from pathlib import Path

import pandas as pd
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.analysis.analysis import (
    correlation_analysis,
    get_num_feature_columns,
    mutual_information_analysis,
    pca_analysis,
)
from src.data.process import load_project, save_project
from src.data.test_load import get_available_columns, get_datasets, select_col
from src.feature.feature import feature_extract
from src.frontend.charts import ChartCanvas
from src.frontend.data_summary import file_summary, missing_summary
from src.frontend.styles import apply_app_styles
from src.frontend.table_model import PandasTableModel
from src.frontend.widgets import (
    ColumnPicker,
    data_panel,
    divider,
    primary_button,
    secondary_button,
    section_label,
    sidebar_base,
    SIDEBAR_WIDTH,
    tab_row,
    table_view,
    taller_dropdown,
)


FEATURES = [
    # Extension point: add/remove selectable summary metrics here.
    # Names should match keys returned by src.feature.feature_extract().
    "count",
    "mean",
    "std",
    "min",
    "25%",
    "50%",
    "75%",
    "max",
    "skew",
    "kurtosis",
    "rms",
    "p2p",
    "variance",
    "missing_count",
    "missing_pct",
]

CHART_TYPES = [
    # Extension point: add Visualization-tab chart names here, then route them
    # in ChartCanvas.plot() and update_chart_controls().
    "Histogram",
    "Scatter",
    "Line",
    "Box Plot",
    "Bar Chart",
    "Grouped Box Plot",
]


class AnalyticsWindow(QMainWindow):
    """Main desktop window and shared frontend state."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classify & Learn Lab")
        self.resize(1180, 760)

        self.file_path = None
        self.dataset = None
        self.columns = []
        self.project = None
        # Raw and working data mirror the existing project file format.
        self.og_df = pd.DataFrame()
        self.working_df = pd.DataFrame()
        self.feature_df = pd.DataFrame()

        # Each table view owns a lightweight model so data refreshes are cheap.
        self.preview_model = PandasTableModel()
        self.file_summary_model = PandasTableModel()
        self.missing_model = PandasTableModel()
        self.feature_preview_model = PandasTableModel()
        self.feature_summary_model = PandasTableModel()
        self.analysis_model = PandasTableModel()
        self.visualization_model = PandasTableModel()

        self._build_ui()
        self._build_menu()
        apply_app_styles(self)

    def _build_menu(self):
        open_dataset = QAction("Open Dataset", self)
        open_dataset.triggered.connect(self.browse_dataset)
        open_project = QAction("Open Project", self)
        open_project.triggered.connect(self.browse_project)
        export_features = QAction("Export Feature Dataset", self)
        export_features.triggered.connect(self.export_features)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_dataset)
        file_menu.addAction(open_project)
        file_menu.addSeparator()
        file_menu.addAction(export_features)

    def _build_ui(self):
        """Build the shell: sidebars on the left, pages on the right."""
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar_stack = QStackedWidget()
        self.sidebar_stack.setFixedWidth(SIDEBAR_WIDTH)
        root_layout.addWidget(self.sidebar_stack)

        # Main content keeps the two-level tab layout from the design mockup.
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(34, 22, 34, 24)
        content_layout.setSpacing(16)
        root_layout.addWidget(content)

        self.top_tabs = tab_row(
            self,
            ["Data & Features", "Models", "Results", "Playback & Annotation"],
            self.on_top_tab_changed,
        )
        content_layout.addLayout(self.top_tabs["layout"])

        self.workflow_tabs = tab_row(
            self,
            ["Import", "Feature Extraction", "Analysis", "Visualization"],
            self.on_workflow_tab_changed,
            compact=True,
        )
        content_layout.addLayout(self.workflow_tabs["layout"])

        self.main_stack = QStackedWidget()
        content_layout.addWidget(self.main_stack, 1)

        # Page order must match sidebar_stack order and workflow tab indexes.
        self._build_import_page()
        self._build_feature_page()
        self._build_analysis_page()
        self._build_visualization_page()

        self.sidebar_stack.addWidget(self._import_sidebar())
        self.sidebar_stack.addWidget(self._feature_sidebar())
        self.sidebar_stack.addWidget(self._analysis_sidebar())
        self.sidebar_stack.addWidget(self._visualization_sidebar())

        self.on_top_tab_changed(0)
        self.on_workflow_tab_changed(0)

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
        layout.addWidget(QLabel("Label"))
        layout.addWidget(self.label_combo)

        layout.addWidget(QLabel("Features"))
        self.column_picker = ColumnPicker("Search columns")
        self.column_picker.setMinimumHeight(190)
        layout.addWidget(self.column_picker)

        self.project_name = QLineEdit()
        self.project_name.setMinimumWidth(210)
        self.project_name.setPlaceholderText("Project name")
        layout.addWidget(QLabel("Project Name"))
        layout.addWidget(self.project_name)

        self.create_project_button = primary_button("Create Project")
        self.create_project_button.clicked.connect(self.create_project)
        layout.addWidget(self.create_project_button)
        layout.addStretch()
        return panel

    def _feature_sidebar(self):
        """Build controls for selecting columns and feature metrics."""
        panel = sidebar_base()
        layout = panel.layout()

        layout.addWidget(section_label("SELECT COLUMNS"))
        layout.addWidget(QLabel("Columns to Analyze"))
        self.signal_picker = ColumnPicker("Search columns")
        self.signal_picker.setMinimumHeight(125)
        layout.addWidget(self.signal_picker)
        layout.addWidget(divider())

        layout.addWidget(section_label("SELECT FEATURES"))
        layout.addWidget(QLabel("Features to Extract"))
        self.feature_picker = ColumnPicker("Search features")
        self.feature_picker.setMinimumHeight(135)
        self.feature_picker.set_items(FEATURES, checked=True)
        layout.addWidget(self.feature_picker)

        extract_button = primary_button("Extract Features")
        extract_button.clicked.connect(self.extract_features)
        layout.addWidget(extract_button)

        import_features = secondary_button("Import Feature Dataset")
        import_features.clicked.connect(self.browse_dataset)
        layout.addWidget(import_features)

        export_features = secondary_button("Export Feature Dataset")
        export_features.clicked.connect(self.export_features)
        layout.addWidget(export_features)
        layout.addStretch()
        return panel

    def _analysis_sidebar(self):
        """Build analysis actions; each fills both chart and table output."""
        panel = sidebar_base()
        layout = panel.layout()
        layout.addWidget(section_label("ANALYSIS"))

        corr_button = primary_button("Correlation")
        corr_button.clicked.connect(self.show_correlation)
        layout.addWidget(corr_button)

        pca_button = secondary_button("PCA")
        pca_button.clicked.connect(self.show_pca)
        layout.addWidget(pca_button)

        mi_button = secondary_button("Mutual Information")
        mi_button.clicked.connect(self.show_mutual_information)
        layout.addWidget(mi_button)
        # Extension point: add new analysis actions here and implement the
        # matching show_* method using existing backend helpers where possible.
        layout.addStretch()
        return panel

    def _visualization_sidebar(self):
        """Build exploratory chart controls for the active dataset."""
        panel = sidebar_base()
        layout = panel.layout()
        layout.addWidget(section_label("VISUALIZATION"))

        self.chart_type_combo = taller_dropdown(QComboBox())
        self.chart_type_combo.addItems(CHART_TYPES)
        self.chart_type_combo.currentTextChanged.connect(self.update_chart_controls)
        layout.addWidget(QLabel("Chart Type"))
        layout.addWidget(self.chart_type_combo)

        self.chart_x_combo = taller_dropdown(QComboBox())
        self.chart_y_combo = taller_dropdown(QComboBox())
        layout.addWidget(QLabel("X / Primary Column"))
        layout.addWidget(self.chart_x_combo)
        layout.addWidget(QLabel("Y Column"))
        layout.addWidget(self.chart_y_combo)

        render = primary_button("Render Chart")
        render.clicked.connect(self.render_visualization)
        layout.addWidget(render)
        layout.addWidget(divider())

        refresh = primary_button("Refresh Summary")
        refresh.clicked.connect(self.refresh_visualization_summary)
        layout.addWidget(refresh)
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
        missing = data_panel("MISSING VALUES SUMMARY", self.missing_model)
        # Missing-value chart lives in Import because it describes data quality.
        self.missing_chart = ChartCanvas(
            "Open a dataset to profile missing values.",
            min_height=180,
        )

        layout.addWidget(preview, 0, 0, 1, 3)
        layout.addWidget(summary, 0, 3, 1, 1)
        layout.addWidget(missing, 1, 0, 1, 4)
        layout.addWidget(self.missing_chart, 2, 0, 1, 4)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 3)
        layout.setColumnStretch(2, 3)
        layout.setColumnStretch(3, 2)
        layout.setColumnMinimumWidth(3, 210)
        layout.setRowStretch(0, 3)
        layout.setRowStretch(1, 2)
        layout.setRowStretch(2, 1)
        self.main_stack.addWidget(page)

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
        self.main_stack.addWidget(page)

    def _build_analysis_page(self):
        """Build the shared analysis chart plus result table."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.analysis_chart = ChartCanvas(
            "Run an analysis from the sidebar.",
            min_height=260,
        )
        layout.addWidget(self.analysis_chart, 2)
        self.analysis_title = section_label("ANALYSIS PREVIEW")
        self.analysis_table = table_view(self.analysis_model)
        layout.addWidget(self.analysis_title)
        layout.addWidget(self.analysis_table, 1)
        self.main_stack.addWidget(page)

    def _build_visualization_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Generic plots share one canvas and are re-rendered from sidebar inputs.
        self.chart_canvas = ChartCanvas()
        layout.addWidget(self.chart_canvas, 3)

        self.visualization_title = section_label("PROJECT SUMMARY")
        self.visualization_table = table_view(self.visualization_model)
        layout.addWidget(self.visualization_title)
        layout.addWidget(self.visualization_table, 1)
        self.main_stack.addWidget(page)

    def on_top_tab_changed(self, index):
        if index != 0:
            QMessageBox.information(
                self,
                "Coming Soon",
                "This section is ready in the shell and will be connected as model and result workflows are added.",
            )
            self.top_tabs["buttons"][0].setChecked(True)
            return
        self.top_tabs["buttons"][index].setChecked(True)

    def on_workflow_tab_changed(self, index):
        """Switch the visible sidebar/page pair for the selected workflow tab."""
        self.workflow_tabs["buttons"][index].setChecked(True)
        self.main_stack.setCurrentIndex(index)
        self.sidebar_stack.setCurrentIndex(index)
        if index == 3:
            self.refresh_visualization_summary()
            self.render_visualization()

    def browse_dataset(self):
        """Open a dataset and populate controls from its columns."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            str(Path.home() / "Downloads"),
            "Supported Files (*.csv *.xlsx *.xls *.mat);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;MATLAB Files (*.mat)",
        )
        if not path:
            return

        self.file_path = Path(path)
        try:
            datasets = get_datasets(self.file_path)
        except Exception as error:
            self.show_error("Dataset Error", error)
            return

        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItems(datasets)
        self.dataset_combo.blockSignals(False)

        if datasets:
            self.dataset = datasets[0]
            self.load_dataset_metadata()

    def browse_project(self):
        """Open an ICP project and restore the saved frontend state."""
        project_dir = Path("Projects")
        project_dir.mkdir(exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            str(project_dir.resolve()),
            "ICP Project Files (*.icp);;All Files (*.*)",
        )
        if not path:
            return

        try:
            self.project, self.og_df, self.working_df = load_project(path)
        except Exception as error:
            self.show_error("Project Error", error)
            return

        self.file_path = Path(self.project.get("file_path", ""))
        self.dataset = self.project.get("dataset", "Data")
        self.columns = list(self.og_df.columns)
        self.project_name.setText(self.project.get("project_name", ""))

        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItem(self.dataset)
        self.dataset_combo.blockSignals(False)

        self.populate_column_controls()
        self.refresh_import_tables()
        QMessageBox.information(self, "Project Loaded", "Project loaded successfully.")

    def on_dataset_changed(self, dataset):
        if dataset and self.file_path:
            self.dataset = dataset
            self.load_dataset_metadata()

    def load_dataset_metadata(self):
        """Read columns and preview data for the selected sheet/dataset."""
        try:
            self.columns = get_available_columns(self.file_path, self.dataset)
            self.og_df = select_col(self.file_path, self.dataset, self.columns)
            self.working_df = self.og_df.copy()
        except Exception as error:
            self.show_error("Load Error", error)
            return

        if not self.project_name.text().strip() and self.file_path:
            self.project_name.setText(self.file_path.stem)

        self.populate_column_controls()
        self.refresh_import_tables()

    def populate_column_controls(self):
        """Refresh every control whose choices come from dataset columns."""
        self.column_picker.set_items(self.columns, checked=True)
        self.signal_picker.set_items(self.columns, checked=True)

        self.label_combo.clear()
        self.label_combo.addItems([str(col) for col in self.columns])
        if self.columns:
            self.label_combo.setCurrentIndex(len(self.columns) - 1)

        if self.project:
            selected = self.project.get("selected_columns", self.columns)
            label = self.project.get("label_column")
            self.column_picker.set_selected(selected)
            self.signal_picker.set_selected(selected)
            if label:
                index = self.label_combo.findText(label)
                if index >= 0:
                    self.label_combo.setCurrentIndex(index)

        self.populate_visualization_controls()

    def selected_columns(self):
        """Return selected features, always retaining the chosen label."""
        columns = self.column_picker.selected_items()
        label = self.label_combo.currentText()
        if label and label not in columns:
            columns.append(label)
        return columns

    def create_project(self):
        """Persist the selected dataset configuration as an ICP project."""
        if self.og_df.empty:
            QMessageBox.warning(self, "Missing Dataset", "Open a dataset before creating a project.")
            return

        project_name = self.project_name.text().strip()
        if not project_name:
            QMessageBox.warning(self, "Missing Project Name", "Enter a project name.")
            return

        columns = self.selected_columns()
        if not columns:
            QMessageBox.warning(self, "Missing Columns", "Select at least one predictor or response column.")
            return

        label_col = self.label_combo.currentText()
        try:
            self.working_df = self.og_df[columns].copy()
            # Keep project metadata compatible with src.data.process.save_project().
            self.project = {
                "project_name": project_name,
                "file_path": str(self.file_path),
                "dataset": self.dataset,
                "selected_columns": list(self.working_df.columns),
                "label_column": label_col,
                "column_types": {
                    col: str(self.working_df[col].dtype)
                    for col in self.working_df.columns
                },
                "preprocessing": [],
                "visualizations": [],
                "models": [],
            }
            save_project(self.project, self.og_df, self.working_df)
        except Exception as error:
            self.show_error("Save Error", error)
            return

        self.refresh_import_tables()
        QMessageBox.information(self, "Project Created", f"Saved Projects/{project_name}.icp")

    def refresh_import_tables(self):
        """Refresh import-tab preview, summary, missing table, and missing chart."""
        df = self.working_df if not self.working_df.empty else self.og_df
        self.preview_model.set_data(df.head(200))
        self.file_summary_model.set_data(file_summary(df))
        self.missing_model.set_data(missing_summary(df))
        self.missing_chart.plot_missing_values(df)

    def extract_features(self):
        """Run the existing feature_extract helper for selected columns."""
        df = self.working_df if not self.working_df.empty else self.og_df
        if df.empty:
            QMessageBox.warning(self, "Missing Dataset", "Open a dataset before extracting features.")
            return

        signals = self.signal_picker.selected_items()
        if not signals:
            QMessageBox.warning(self, "Missing Signals", "Select at least one signal.")
            return

        requested_features = self.feature_picker.selected_items()
        try:
            raw_features = feature_extract(df, signals)
        except Exception as error:
            self.show_error("Feature Extraction Error", error)
            return

        rows = []
        for signal, values in raw_features.items():
            row = {"signal": signal}
            for feature in requested_features:
                row[feature] = values.get(feature, "")
            rows.append(row)

        self.feature_df = pd.DataFrame(rows)
        self.feature_preview_model.set_data(self.feature_df)
        self.feature_summary_model.set_data(file_summary(self.feature_df))
        self.on_workflow_tab_changed(1)

    def export_features(self):
        """Export the current feature table to CSV."""
        if self.feature_df.empty:
            QMessageBox.warning(self, "Missing Features", "Extract features before exporting.")
            return

        default_path = Path("ExportedModels") / "feature_dataset.csv"
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

    def show_correlation(self):
        """Run correlation analysis and show matrix plus heatmap."""
        df = self.working_df if not self.working_df.empty else self.og_df
        label = self.label_combo.currentText()
        if df.empty:
            QMessageBox.warning(self, "Missing Dataset", "Open a dataset first.")
            return

        try:
            # Backend call: do not duplicate correlation logic in the frontend.
            result = correlation_analysis(df, label)
        except Exception as error:
            self.show_error("Correlation Error", error)
            return

        self.analysis_title.setText("CORRELATION MATRIX")
        self.analysis_model.set_data(result.round(4))
        self.analysis_chart.plot_correlation_heatmap(df)
        self.on_workflow_tab_changed(2)

    def show_pca(self):
        """Run the existing PCA analysis and show table plus PC scatter chart."""
        df = self.working_df if not self.working_df.empty else self.og_df
        label = self.label_combo.currentText()
        features = get_num_feature_columns(df, label) if not df.empty else []
        if len(features) < 2:
            QMessageBox.warning(self, "PCA Unavailable", "PCA needs at least two numeric feature columns.")
            return

        try:
            # Backend call: pca_analysis owns scaling and component generation.
            result = pca_analysis(df, features, label, n_components=2)
            pca_df = result["pca_df"].head(200).copy()
            pca_df.attrs["explained_variance_sum"] = result["explained_variance_sum"]
        except Exception as error:
            self.show_error("PCA Error", error)
            return

        self.analysis_title.setText(
            f"PCA PREVIEW - EXPLAINED VARIANCE {result['explained_variance_sum']:.2%}"
        )
        self.analysis_model.set_data(pca_df.round(4))
        self.analysis_chart.plot_pca_scatter(pca_df, label)
        self.on_workflow_tab_changed(2)

    def show_mutual_information(self):
        """Run mutual information analysis and show table plus score chart."""
        df = self.working_df if not self.working_df.empty else self.og_df
        label = self.label_combo.currentText()
        features = get_num_feature_columns(df, label) if not df.empty else []
        if not features or not label:
            QMessageBox.warning(self, "Analysis Unavailable", "Select numeric features and a response column.")
            return

        try:
            # Backend call: frontend only displays returned feature scores.
            result = mutual_information_analysis(df, features, label)
        except Exception as error:
            self.show_error("Mutual Information Error", error)
            return

        self.analysis_title.setText("MUTUAL INFORMATION")
        self.analysis_model.set_data(result.round(4))
        self.analysis_chart.plot_mutual_information(result)
        self.on_workflow_tab_changed(2)

    def refresh_visualization_summary(self):
        """Refresh the small project/dataset summary under the Visualization chart."""
        df = self.working_df if not self.working_df.empty else self.og_df
        summary = file_summary(df)
        if self.project:
            extra = pd.DataFrame(
                [
                    ("Project", self.project.get("project_name", "")),
                    ("Dataset", self.project.get("dataset", "")),
                    ("Label", self.project.get("label_column", self.label_combo.currentText())),
                    ("Feature Rows", len(self.feature_df)),
                ],
                columns=["metric", "value"],
            )
            summary = pd.concat([extra, summary], ignore_index=True)
        self.visualization_model.set_data(summary)

    def populate_visualization_controls(self):
        """Populate chart column dropdowns from the active dataset columns."""
        for combo in (self.chart_x_combo, self.chart_y_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([str(col) for col in self.columns])
            combo.blockSignals(False)

        numeric_columns = self.numeric_columns()
        if numeric_columns:
            self.chart_x_combo.setCurrentText(numeric_columns[0])
            self.chart_y_combo.setCurrentText(
                numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0]
            )
        self.update_chart_controls()

    def update_chart_controls(self):
        """Enable only the chart inputs required by the selected chart type."""
        chart_type = self.chart_type_combo.currentText()
        # Extension point: keep this mapping aligned with CHART_TYPES.
        needs_y = chart_type in ("Scatter", "Line", "Bar Chart", "Grouped Box Plot")
        needs_x = chart_type in (
            "Histogram",
            "Scatter",
            "Line",
            "Box Plot",
            "Bar Chart",
            "Grouped Box Plot",
        )
        self.chart_x_combo.setEnabled(needs_x)
        self.chart_y_combo.setEnabled(needs_y)

    def render_visualization(self):
        """Render the selected generic visualization against the active dataset."""
        df = self.working_df if not self.working_df.empty else self.og_df
        if df.empty:
            self.chart_canvas.show_empty("Open a dataset to visualize it.")
            return

        self.chart_canvas.plot(
            df,
            self.chart_type_combo.currentText(),
            self.chart_x_combo.currentText(),
            self.chart_y_combo.currentText(),
            self.label_combo.currentText(),
        )

    def numeric_columns(self):
        """Return numeric columns from the active DataFrame for chart defaults."""
        df = self.working_df if not self.working_df.empty else self.og_df
        if df.empty:
            return []
        return [str(col) for col in df.select_dtypes(include="number").columns]

    def show_error(self, title, error):
        QMessageBox.critical(self, title, str(error))
