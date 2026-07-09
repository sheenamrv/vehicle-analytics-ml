from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
    QMenu,
    QDialog,
    QPushButton,
    QInputDialog,
    QTableView,
    QCheckBox,
    QButtonGroup,
)
from PySide6.QtCore import (
    Qt,
    QThreadPool,
    QRunnable,
    QObject,
    Signal,
    QAbstractTableModel,
    QModelIndex,
    QItemSelectionModel,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.analysis.analysis import (
    correlation_analysis,
    get_num_feature_columns,
    mutual_information_analysis,
    pca_analysis,
)
from src.data.process import load_project, save_project
from src.model.model_registry import add_model, delete_model
from src.model.model_training import test_models_current_data
from src.model.model_utils import validate_dataset, prepare_training_data
from src.model.supervised_model import build_model
from src.frontend.model_panel import ModelParameterPanel, MODEL_TYPES
from src.data.test_load import get_available_columns, get_datasets, select_col, change_dtype
from src.feature.feature import feature_extract
from src.frontend.charts import ChartCanvas
from src.frontend.data_summary import file_summary, missing_summary
from src.frontend.styles import apply_app_styles
from src.frontend.table_model import PandasTableModel
from src.visualize.visualization import export_plot_image
from src.frontend.widgets import (
    ColumnPicker,
    allowed_dtypes,
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


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(Exception)


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


class QualityIssueTableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data.copy()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._data.index)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._data.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            value = self._data.iat[index.row(), index.column()]
            if pd.isna(value):
                return ""
            return str(value)

        if role == Qt.BackgroundRole:
            column_name = self._data.columns[index.column()]
            value = self._data.iat[index.row(), index.column()]
            issue_type = self._data.iloc[index.row()]["issue"]

            if pd.isna(value) and column_name not in ("issue", "__original_index__"):
                return QColor("#fff3c4")
            if "Duplicate" in str(issue_type):
                return QColor("#e8f4ff")

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return str(self._data.columns[section])
        return str(self._data.index[section])


class DataQualityDialog(QDialog):
    def __init__(self, parent, working_df):
        super().__init__(parent)
        self.setWindowTitle("Review Missing and Duplicate Rows")
        self.setMinimumSize(1000, 520)

        self.issue_df = self._build_issue_df(working_df)
        self.removed_indices = []

        layout = QVBoxLayout(self)

        if self.issue_df.empty:
            layout.addWidget(QLabel("No missing values or duplicate rows were found."))
            close_button = QPushButton("Close")
            close_button.clicked.connect(self.reject)
            layout.addWidget(close_button)
            return

        self.issue_model = QualityIssueTableModel(self.issue_df)
        self.table = table_view(self.issue_model)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        layout.addWidget(self.table)

        button_layout = QHBoxLayout()
        remove_button = QPushButton("Remove Selected Rows")
        remove_button.clicked.connect(self.on_remove_selected)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(remove_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def _build_issue_df(self, df):
        if df.empty:
            return pd.DataFrame()

        missing = set(df[df.isna().any(axis=1)].index.tolist())
        duplicates = set(df[df.duplicated(keep=False)].index.tolist())
        issue_rows = []

        for row_index in sorted(missing.union(duplicates)):
            row = df.loc[row_index].copy()
            issue_labels = []
            if row_index in missing:
                issue_labels.append("Missing")
            if row_index in duplicates:
                issue_labels.append("Duplicate")
            row["issue"] = " & ".join(issue_labels)
            row["__original_index__"] = row_index
            issue_rows.append(row)

        if not issue_rows:
            return pd.DataFrame()

        issue_df = pd.DataFrame(issue_rows)
        columns = ["__original_index__", "issue"] + [c for c in issue_df.columns if c not in ("__original_index__", "issue")]
        return issue_df[columns]

    def on_remove_selected(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, self.windowTitle(), "Select at least one row to remove.")
            return

        indices = sorted({
            self.issue_df.iloc[index.row()]["__original_index__"]
            for index in selected_rows
        })

        result = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Remove {len(indices)} selected row(s) from the working dataset?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if result == QMessageBox.StandardButton.Yes:
            self.removed_indices = indices
            self.accept()


'''
    Main font end window for the application
'''

# ============================================================================
# Constants
# ============================================================================
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
    "Class Separation",
    "3D Scatter",
    "Time Series (All Signals)",
    "Feature Distribution Comparison",
]


# ============================================================================
# Main Application Window
# ============================================================================

class AnalyticsWindow(QMainWindow):
    """Main desktop window and shared frontend state."""

# ============================================================================
# Window Initialization
# ============================================================================
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classify & Learn Lab")
        self.resize(1180, 760)

        self.file_path = None
        self.dataset = None
        self.columns = []
        self.project = None
        self.current_project_path = None
        self.is_dirty = False
        self._suppress_dirty = False
        self.downloads_dir = Path.home() / "Downloads"
        self.thread_pool = QThreadPool.globalInstance()
        self.preview_page = 0
        self.preview_page_size = 1000
        # Track original dtypes for the DTypeDelegate; start empty until a
        # dataset is loaded.
        self.original_dtypes = {}
        # Raw and working data mirror the existing project file format.
        self.og_df = pd.DataFrame()
        self.working_df = pd.DataFrame()
        self.feature_df = pd.DataFrame()
        self.latest_correlation_matrix = pd.DataFrame()
        self.feature_use_raw = False
        self.analysis_use_raw = False
        self.analysis_include_label = True
        self.active_analysis = "Correlation"
        self.analysis_cmap = "viridis"
        self.analysis_matrix_type = "Numeric"
        self.feature_use_raw = False
        self.analysis_use_raw = False
        self.visualization_use_raw = False
        self.active_analysis = "Correlation"
        self.analysis_include_label = True
        self.analysis_cmap = "viridis"

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

# ============================================================================
# Menu Bar
# ============================================================================

    def _build_menu(self):
        open_dataset = QAction("Open Dataset", self)
        open_dataset.triggered.connect(self.browse_dataset)
        open_project = QAction("Open Project", self)
        open_project.triggered.connect(self.browse_project)
        save_project_action = QAction("Save Project", self)
        save_project_action.triggered.connect(self.save_current_project)
        save_project_action.setShortcut(QKeySequence.StandardKey.Save)
        save_as_action = QAction("Save Project As...", self)
        save_as_action.triggered.connect(self.save_project_as)
        export_features = QAction("Export Feature Dataset", self)
        export_features.triggered.connect(self.export_features)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_dataset)
        file_menu.addAction(open_project)
        file_menu.addAction(save_project_action)
        file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(export_features)

        # Single QAction handles the Save shortcut; avoid QShortcut duplicate.

# ============================================================================
# Main UI Construction
# ============================================================================

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
        self.workflow_tabs_container = QWidget()
        self.workflow_tabs_container.setLayout(self.workflow_tabs["layout"])
        content_layout.addWidget(self.workflow_tabs_container)

        self.main_stack = QStackedWidget()
        content_layout.addWidget(self.main_stack, 1)

        # Page order must match sidebar_stack order and workflow tab indexes.
        self._build_import_page()
        self._build_feature_page()
        self._build_analysis_page()
        self._build_visualization_page()
        self._build_models_page()

        self.sidebar_stack.addWidget(self._import_sidebar())
        self.sidebar_stack.addWidget(self._feature_sidebar())
        self.sidebar_stack.addWidget(self._analysis_sidebar())
        self.sidebar_stack.addWidget(self._visualization_sidebar())
        self.sidebar_stack.addWidget(self._models_sidebar())

        self.on_top_tab_changed(0)
        self.on_workflow_tab_changed(0)

# ============================================================================
# Sidebar Construction
# ============================================================================

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

    def _feature_sidebar(self):
        """Build controls for selecting columns and feature metrics."""
        panel = sidebar_base()
        layout = panel.layout()

        layout.addWidget(section_label("SELECT COLUMNS"))
        layout.addWidget(QLabel("Columns to Analyze"))
        self.feature_numeric_picker = ColumnPicker("Search numeric columns")
        self.feature_numeric_picker.setMinimumHeight(110)
        layout.addWidget(QLabel("Numeric Columns"))
        layout.addWidget(self.feature_numeric_picker)

        self.feature_non_numeric_picker = ColumnPicker("Search non-numeric columns")
        self.feature_non_numeric_picker.setMinimumHeight(110)
        layout.addWidget(QLabel("Non-Numeric Columns"))
        layout.addWidget(self.feature_non_numeric_picker)

        self.feature_use_raw_checkbox = QCheckBox("Use raw dataset for feature extraction")
        self.feature_use_raw_checkbox.toggled.connect(self.on_feature_use_raw_toggled)
        layout.addWidget(self.feature_use_raw_checkbox)
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
        import_features.clicked.connect(self.browse_feature_dataset)
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

        self.analysis_type_combo = taller_dropdown(QComboBox())
        self.analysis_type_combo.addItems(["Correlation", "PCA", "Mutual Information"])
        self.analysis_type_combo.currentTextChanged.connect(self.on_analysis_type_changed)
        layout.addWidget(QLabel("Analysis Type"))
        layout.addWidget(self.analysis_type_combo)

        self.run_analysis_button = primary_button("Run Analysis")
        self.run_analysis_button.clicked.connect(self.run_current_analysis)
        layout.addWidget(self.run_analysis_button)

        self.analysis_include_label_checkbox = QCheckBox("Include label")
        self.analysis_include_label_checkbox.setChecked(self.analysis_include_label)
        self.analysis_include_label_checkbox.toggled.connect(self.on_analysis_include_label_toggled)
        layout.addWidget(self.analysis_include_label_checkbox)

        self.analysis_use_raw_checkbox = QCheckBox("Use raw dataset")
        self.analysis_use_raw_checkbox.toggled.connect(self.on_analysis_use_raw_toggled)
        layout.addWidget(self.analysis_use_raw_checkbox)

        self.analysis_cmap_label = QLabel("Heatmap Color")
        self.analysis_cmap_combo = taller_dropdown(QComboBox())
        self.analysis_cmap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis"])
        self.analysis_cmap_combo.currentTextChanged.connect(self.on_analysis_cmap_changed)
        layout.addWidget(self.analysis_cmap_label)
        layout.addWidget(self.analysis_cmap_combo)

        self.analysis_non_numeric_checkbox = QCheckBox("Use non-numeric columns")
        self.analysis_non_numeric_checkbox.toggled.connect(self.on_analysis_matrix_type_changed)
        layout.addWidget(self.analysis_non_numeric_checkbox)

        self.analysis_numeric_label = QLabel("Numeric Columns")
        self.analysis_non_numeric_label = QLabel("Non-Numeric Columns")
        self.analysis_numeric_picker = ColumnPicker("Search numeric columns")
        self.analysis_numeric_picker.setMinimumHeight(100)
        layout.addWidget(self.analysis_numeric_label)
        layout.addWidget(self.analysis_numeric_picker)

        self.analysis_non_numeric_picker = ColumnPicker("Search non-numeric columns")
        self.analysis_non_numeric_picker.setMinimumHeight(100)
        layout.addWidget(self.analysis_non_numeric_label)
        layout.addWidget(self.analysis_non_numeric_picker)

        self.export_analysis_button = secondary_button("Export Analysis Image")
        self.export_analysis_button.clicked.connect(self.export_analysis_image)
        layout.addWidget(self.export_analysis_button)

        self._refresh_analysis_column_picker_visibility()
        self.on_analysis_type_changed(self.analysis_type_combo.currentText())
        layout.addStretch()
        return panel

    def _visualization_sidebar(self):
        """Build exploratory chart controls for the active dataset."""
        panel = sidebar_base()
        layout = panel.layout()
        layout.addWidget(section_label("VISUALIZATION"))

        self.visualization_use_raw_checkbox = QCheckBox("Use raw dataset")
        self.visualization_use_raw_checkbox.toggled.connect(self.on_visualization_use_raw_toggled)
        layout.addWidget(self.visualization_use_raw_checkbox)

        self.chart_type_combo = taller_dropdown(QComboBox())
        self.chart_type_combo.addItems(CHART_TYPES)
        self.chart_type_combo.currentTextChanged.connect(self.on_chart_type_changed)
        layout.addWidget(QLabel("Chart Type"))
        layout.addWidget(self.chart_type_combo)

        self.chart_input_hint = QLabel("")
        self.chart_input_hint.setWordWrap(True)
        layout.addWidget(self.chart_input_hint)

        self.chart_x_combo = taller_dropdown(QComboBox())
        self.chart_y_combo = taller_dropdown(QComboBox())
        self.chart_z_combo = taller_dropdown(QComboBox())
        self.chart_x_label = QLabel("X / Primary Column")
        self.chart_y_label = QLabel("Y Column")
        self.chart_z_label = QLabel("Z Column (3D Scatter only)")
        layout.addWidget(self.chart_x_label)
        layout.addWidget(self.chart_x_combo)
        layout.addWidget(self.chart_y_label)
        layout.addWidget(self.chart_y_combo)
        layout.addWidget(self.chart_z_label)
        layout.addWidget(self.chart_z_combo)

        self.chart_bins_spin = QSpinBox()
        self.chart_bins_spin.setRange(2, 200)
        self.chart_bins_spin.setValue(24)
        self.chart_bins_label = QLabel("Bins (Histogram only)")
        layout.addWidget(self.chart_bins_label)
        layout.addWidget(self.chart_bins_spin)

        self.chart_mean_line_checkbox = QCheckBox("Show mean line")
        self.chart_median_line_checkbox = QCheckBox("Show median line")
        layout.addWidget(self.chart_mean_line_checkbox)
        layout.addWidget(self.chart_median_line_checkbox)

        self.chart_multi_picker = ColumnPicker("Search columns")
        self.chart_multi_picker.setMinimumHeight(100)
        self.chart_multi_label = QLabel("Signals to Compare (Time Series / Distribution Comparison)")
        layout.addWidget(self.chart_multi_label)
        layout.addWidget(self.chart_multi_picker)

        render = primary_button("Render Chart")
        render.clicked.connect(self.render_visualization)
        layout.addWidget(render)
        layout.addWidget(divider())

        refresh = primary_button("Refresh Summary")
        refresh.clicked.connect(self.refresh_visualization_summary)
        layout.addWidget(refresh)

        export_chart = secondary_button("Export Chart")
        export_chart.clicked.connect(self.export_chart_gui)
        layout.addWidget(export_chart)

        layout.addStretch()
        return panel

    def _models_sidebar(self):
        """Build training, selection, and testing controls for the Models tab."""
        panel = sidebar_base()
        layout = panel.layout()

        layout.addWidget(section_label("TRAIN MODEL"))

        self.model_type_combo = taller_dropdown(QComboBox())
        self.model_type_combo.addItems([label for label, _ in MODEL_TYPES])
        self.model_type_combo.currentIndexChanged.connect(self.update_model_parameter_fields)
        layout.addWidget(QLabel("Model Type"))
        layout.addWidget(self.model_type_combo)

        self.model_param_panel = ModelParameterPanel()
        layout.addWidget(self.model_param_panel)

        self.model_test_size_spin = QDoubleSpinBox()
        self.model_test_size_spin.setRange(0.05, 0.95)
        self.model_test_size_spin.setSingleStep(0.05)
        self.model_test_size_spin.setValue(0.3)
        layout.addWidget(QLabel("Test Split"))
        layout.addWidget(self.model_test_size_spin)

        self.model_random_state_spin = QSpinBox()
        self.model_random_state_spin.setRange(0, 9999)
        self.model_random_state_spin.setValue(42)
        layout.addWidget(QLabel("Random State"))
        layout.addWidget(self.model_random_state_spin)

        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("Model display name")
        layout.addWidget(QLabel("Model Name"))
        layout.addWidget(self.model_name_edit)

        train_button = primary_button("Train Model")
        train_button.clicked.connect(self.train_new_model_gui)
        layout.addWidget(train_button)
        layout.addWidget(divider())

        layout.addWidget(section_label("MANAGE MODELS"))

        self.model_select_combo = taller_dropdown(QComboBox())
        self.model_select_combo.currentTextChanged.connect(self.show_model_details_gui)
        layout.addWidget(QLabel("Select Model"))
        layout.addWidget(self.model_select_combo)

        delete_button = secondary_button("Delete Selected Model")
        delete_button.clicked.connect(self.delete_selected_model)
        layout.addWidget(delete_button)

        self.model_test_picker = ColumnPicker("Search models")
        self.model_test_picker.setMinimumHeight(100)
        layout.addWidget(QLabel("Models to Test"))
        layout.addWidget(self.model_test_picker)

        test_button = primary_button("Test Selected Models")
        test_button.clicked.connect(self.test_selected_models_gui)
        layout.addWidget(test_button)

        layout.addStretch()
        return panel


# ============================================================================
# Workflow Page Construction
# ============================================================================
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
        self.main_stack.addWidget(page)

    # ============================================================================
    # Data Type Conversion
    # ============================================================================
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

    def _build_models_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.models_title = section_label("SAVED MODELS")
        self.models_model = PandasTableModel(pd.DataFrame())
        self.models_table = table_view(self.models_model)
        layout.addWidget(self.models_title)
        layout.addWidget(self.models_table, 1)
        self.main_stack.addWidget(page)

# ============================================================================
# Navigation & Tab Management
# ============================================================================

    def on_top_tab_changed(self, index):
        if index == 0:
            self.top_tabs["buttons"][0].setChecked(True)
            self.workflow_tabs_container.setVisible(True)
            current_workflow = 0
            for i, button in enumerate(self.workflow_tabs["buttons"]):
                if button.isChecked():
                    current_workflow = i
                    break
            self.on_workflow_tab_changed(current_workflow)
        elif index == 1:
            self.top_tabs["buttons"][1].setChecked(True)
            self.workflow_tabs_container.setVisible(False)
            self.main_stack.setCurrentIndex(4)
            self.sidebar_stack.setCurrentIndex(4)
            self.refresh_models_list()
        else:
            QMessageBox.information(
                self,
                "Coming Soon",
                "This section is ready in the shell and will be connected as result workflows are added.",
            )
            self.top_tabs["buttons"][0].setChecked(True)
            self.workflow_tabs_container.setVisible(True)

    # ============================================================================
    # Models
    # ============================================================================
    def update_model_parameter_fields(self):
        """Swap the visible parameter fields for the selected model type."""
        _, model_type = MODEL_TYPES[self.model_type_combo.currentIndex()]
        self.model_param_panel.set_model_type(model_type)

    def refresh_models_list(self):
        """Refresh the model dropdown/picker and the saved-models table."""
        names = [m["display_name"] for m in self.project.get("models", [])] if self.project else []

        self.model_select_combo.blockSignals(True)
        self.model_select_combo.clear()
        self.model_select_combo.addItems(names)
        self.model_select_combo.blockSignals(False)

        self.model_test_picker.blockSignals(True)
        self.model_test_picker.set_items(names, checked=False)
        self.model_test_picker.blockSignals(False)

        self.show_model_details_gui()

    def show_model_details_gui(self):
        """Show details for the selected model, or a summary list if none selected."""
        if not self.project or not self.project.get("models"):
            self.models_title.setText("SAVED MODELS")
            self.models_model.set_data(pd.DataFrame())
            return

        name = self.model_select_combo.currentText()
        match = next((m for m in self.project["models"] if m["display_name"] == name), None)

        if not match:
            self.models_title.setText("SAVED MODELS")
            rows = [(m["display_name"], m["algorithm"]) for m in self.project["models"]]
            self.models_model.set_data(pd.DataFrame(rows, columns=["display_name", "algorithm"]))
            return

        self.models_title.setText(f"MODEL DETAILS - {match['display_name']}")
        rows = [("algorithm", match["algorithm"])]
        for key, value in match.get("parameters", {}).items():
            rows.append((f"param: {key}", value))
        for key, value in match.get("metrics", {}).items():
            rows.append((f"metric: {key}", round(value, 4) if isinstance(value, float) else value))
        self.models_model.set_data(pd.DataFrame(rows, columns=["field", "value"]))

    def train_new_model_gui(self):
        """Train a model using the sidebar's selections and add it to the project."""
        if not self.project:
            QMessageBox.warning(self, "No Project", "Create or load a project before training a model.")
            return

        df = self.working_df if not self.working_df.empty else self.og_df
        label = self.project.get("label_column") or self.get_selected_label()

        valid, message = validate_dataset(df, label)
        if not valid:
            QMessageBox.warning(self, "Invalid Dataset", message)
            return

        try:
            X, y = prepare_training_data(df, label)
            feature_columns = X.columns.tolist()
            _, model_type = MODEL_TYPES[self.model_type_combo.currentIndex()]
            parameters = self.model_param_panel.get_parameters()
            config = {
                "test_size": self.model_test_size_spin.value(),
                "random_state": self.model_random_state_spin.value(),
            }
            model = build_model(model_type, parameters)
            trained_model, metrics = build_model(model_type, parameters), {"accuracy": None}
        except Exception as error:
            self.show_error("Training Error", error)
            return

        display_name = self.model_name_edit.text().strip() or (
            f"{model_type}_{len(self.project.get('models', [])) + 1}"
        )

        add_model(self.project, trained_model, display_name, model_type, {**config, **parameters}, metrics, feature_columns)
        self._set_dirty(True)
        self.model_name_edit.clear()
        self.refresh_models_list()
        QMessageBox.information(self, "Model Trained", f"'{display_name}' trained and added to the project.")

    def delete_selected_model(self):
        """Delete the model currently selected in the sidebar dropdown."""
        if not self.project or not self.project.get("models"):
            return

        name = self.model_select_combo.currentText()
        if not name:
            return

        result = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete model '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            return

        delete_model(self.project, name)
        self._set_dirty(True)
        self.refresh_models_list()

    def test_selected_models_gui(self):
        """Test the checked models from the picker against the working dataset."""
        if not self.project or not self.project.get("models"):
            QMessageBox.warning(self, "No Models", "Train or load at least one model first.")
            return

        names = self.model_test_picker.selected_items()
        if not names:
            QMessageBox.warning(self, "No Selection", "Select at least one model to test.")
            return

        df = self.working_df if not self.working_df.empty else self.og_df
        label = self.project.get("label_column") or self.get_selected_label()
        models = [m for m in self.project["models"] if m["display_name"] in names]

        try:
            results = test_models_current_data(models, df, label)
        except Exception as error:
            self.show_error("Test Error", error)
            return

        rows = [
            {
                "model": r["name"],
                "accuracy": round(r["accuracy"], 4),
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1": round(r["f1"], 4),
            }
            for r in results
        ]
        self.models_title.setText("MODEL TEST RESULTS")
        self.models_model.set_data(pd.DataFrame(rows))

    def on_workflow_tab_changed(self, index):
        """Switch the visible sidebar/page pair for the selected workflow tab."""
        self.workflow_tabs["buttons"][index].setChecked(True)
        self.main_stack.setCurrentIndex(index)
        self.sidebar_stack.setCurrentIndex(index)
        if index == 3:
            self.refresh_visualization_summary()
            self.render_visualization()

    def _set_dirty(self, dirty=True):
        if self._suppress_dirty and dirty:
            return
        self.is_dirty = dirty
        icon = "• " if dirty else ""
        title = self.project.get("project_name") if self.project else "Classify & Learn Lab"
        self.setWindowTitle(f"{icon}{title}")

    def on_project_name_changed(self, text):
        if self._suppress_dirty:
            return
        if self.project or self.file_path or self.current_project_path:
            self._set_dirty(True)

    def closeEvent(self, event):
        if self.is_dirty:
            result = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to exit without saving?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if result == QMessageBox.StandardButton.No:
                event.ignore()
                return
        event.accept()

# ============================================================================
# Dataset & Project Loading
# ============================================================================
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
        self._suppress_dirty = True
        self.project_name.setText(self.project.get("project_name", ""))

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
        self.feature_preview_model.set_data(self.feature_df)
        self.feature_summary_model.set_data(file_summary(self.feature_df))
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

    def on_analysis_use_raw_toggled(self, checked):
        self.analysis_use_raw = checked
        self.refresh_column_pickers()
        self.populate_visualization_controls()

    def on_visualization_use_raw_toggled(self, checked):
        self.visualization_use_raw = checked
        self.refresh_column_pickers()
        self.populate_visualization_controls()

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
                existing = [c for c in columns if c in self.working_df.columns]
                current = self.working_df[existing].copy()
                missing = [c for c in columns if c not in existing]
                for col in missing:
                    if col in self.og_df.columns:
                        current[col] = self.og_df[col]
                self.working_df = current[columns].copy()

        self._set_dirty(True)
        self._refresh_label_combo()
        self.refresh_column_pickers()
        self.refresh_import_tables()
        self.populate_visualization_controls()

    def reset_workflow_state(self):
        """Clear analysis and visualization state when a new dataset/project loads."""
        self.project = None
        self.feature_df = pd.DataFrame()
        self.latest_correlation_matrix = pd.DataFrame()
        self.analysis_model.set_data(pd.DataFrame())
        self.visualization_model.set_data(pd.DataFrame())
        self.analysis_chart.show_empty("Run an analysis from the sidebar.")
        self.chart_canvas.show_empty("Open a dataset to visualize it.")
        if hasattr(self, "analysis_title"):
            self.analysis_title.setText("ANALYSIS PREVIEW")
        if hasattr(self, "visualization_title"):
            self.visualization_title.setText("PROJECT SUMMARY")

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
            self.working_df = df.drop(index=dialog.removed_indices).reset_index(drop=True)
            self._set_dirty(True)
            self.refresh_import_tables()

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
        df = self.working_df if not self.working_df.empty else self.og_df
        if df.empty:
            QMessageBox.warning(self, "No Data", "Load a dataset before imputing values.")
            return

        for col in columns:
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

        self.working_df = df
        self._set_dirty(True)
        self.refresh_import_tables()

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
        df = self.working_df if not self.working_df.empty else self.og_df
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_cols:
            QMessageBox.warning(self, "Standardize", "Select numeric columns to standardize.")
            return
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.working_df = df
        self._set_dirty(True)
        self.refresh_import_tables()

    def normalize_columns(self, columns):
        df = self.working_df if not self.working_df.empty else self.og_df
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_cols:
            QMessageBox.warning(self, "Normalize", "Select numeric columns to normalize.")
            return
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.working_df = df
        self._set_dirty(True)
        self.refresh_import_tables()

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
        for row_index in row_indices:
            page_start = self.preview_page * self.preview_page_size
            page_end = page_start + self.preview_page_size
            if page_start <= row_index < page_end:
                preview_row = row_index - page_start
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
        self.working_df = df.drop(index=duplicates.index).reset_index(drop=True)
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
        self.working_df = df.drop(index=missing_rows.index).reset_index(drop=True)
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

# ============================================================================
# Project Management
# ============================================================================

    def create_project(self):
        """Persist the selected dataset configuration as an ICP project."""
        if self.og_df.empty:
            QMessageBox.warning(self, "Missing Dataset", "Open a dataset before creating a project.")
            return False

        project_name = self.project_name.text().strip()
        if not project_name:
            QMessageBox.warning(self, "Missing Project Name", "Enter a project name.")
            return False

        columns = self.selected_columns()
        if not columns:
            QMessageBox.warning(self, "Missing Columns", "Select at least one predictor or response column.")
            return False

        label_col = self.get_selected_label()
        try:
            if self.working_df.empty:
                self.working_df = self.og_df[columns].copy()
            else:
                self.working_df = self.working_df.reindex(columns=columns).copy()

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
            if self.current_project_path is None:
                self.current_project_path = self._choose_project_save_path(default_name=project_name)
                if self.current_project_path is None:
                    return False
            save_project(
                self.project,
                self.og_df,
                self.working_df,
                str(self.current_project_path),
                feature_df=self.feature_df.copy(),
            )
            self._set_dirty(False)
        except Exception as error:
            self.show_error("Save Error", error)
            return False

        self.refresh_import_tables()
        QMessageBox.information(self, "Project Created", f"Saved {self.current_project_path}")
        return True

    def save_current_project(self):
        if self.project is None:
            return self.create_project()

        if self.working_df.empty and not self.og_df.empty:
            self.working_df = self.og_df[self.selected_columns()].copy()

        if self.current_project_path is None:
            return self.save_project_as()

        self.project["selected_columns"] = list(self.working_df.columns)
        self.project["column_types"] = {col: str(self.working_df[col].dtype) for col in self.working_df.columns}

        try:
            self.project["file_path"] = str(self.file_path) if self.file_path is not None else self.project.get("file_path")
            save_project(
                self.project,
                self.og_df.copy(),
                self.working_df.copy(),
                str(self.current_project_path),
                feature_df=self.feature_df.copy(),
            )
            self._set_dirty(False)
            QMessageBox.information(self, "Project Saved", f"Saved {self.current_project_path}")
            return True
        except Exception as error:
            self.show_error("Save Error", error)
            return False

    def save_project_as(self):
        if self.project is None:
            QMessageBox.warning(self, "No Project", "Create or open a project before saving.")
            return False

        project_name = self.project.get("project_name") or self.project_name.text().strip() or "project"
        chosen_path = self._choose_project_save_path(default_name=project_name)
        if chosen_path is None:
            return False

        if self.working_df.empty and not self.og_df.empty:
            self.working_df = self.og_df[self.selected_columns()].copy()

        try:
            # Update metadata on the active project before saving a copy.
            self.project["selected_columns"] = list(self.working_df.columns)
            self.project["column_types"] = {col: str(self.working_df[col].dtype) for col in self.working_df.columns}
            self.project["file_path"] = str(self.file_path) if self.file_path is not None else self.project.get("file_path")

            # Save a copy under the chosen path, but keep the active project bound
            # to the original project for future saves.
            project_copy = dict(self.project)
            project_copy["project_name"] = chosen_path.stem
            save_project(
                project_copy,
                self.og_df.copy(),
                self.working_df.copy(),
                str(chosen_path),
                feature_df=self.feature_df.copy(),
            )
            # Keep the active project path unchanged when saving a copy.
            self._set_dirty(False)
            QMessageBox.information(self, "Project Saved", f"Saved {chosen_path}")
            return True
        except Exception as error:
            self.show_error("Save Error", error)
            return False

    def _choose_project_save_path(self, default_name=None):
        default_name = default_name or "project"
        default_file = self.downloads_dir / f"{default_name}.icp"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            str(default_file),
            "ICP Project Files (*.icp)",
        )
        if not path:
            return None

        chosen_path = Path(path)
        if chosen_path.suffix.lower() != ".icp":
            chosen_path = chosen_path.with_suffix(".icp")

        # if chosen_path.exists():
        #     result = QMessageBox.question(
        #         self,
        #         "Overwrite Project",
        #         f"A project named '{chosen_path.name}' already exists. Overwrite it?",
        #         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        #         QMessageBox.StandardButton.No,
        #     )
        #     if result == QMessageBox.StandardButton.No:
        #         return None

        chosen_path.parent.mkdir(parents=True, exist_ok=True)
        return chosen_path

# ============================================================================
# Import Page Refresh
# ============================================================================

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

    # ============================================================================
    # Feature Extraction
    # ============================================================================
    def extract_features(self):
        """Run the existing feature_extract helper for selected columns."""
        df = self._feature_source_dataframe()
        if df.empty:
            QMessageBox.warning(self, "Missing Dataset", "Open a dataset before extracting features.")
            return

        signals = self.feature_numeric_picker.selected_items() + self.feature_non_numeric_picker.selected_items()
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
        self.refresh_feature_tables()
        self.on_workflow_tab_changed(1)

    def refresh_feature_tables_for_active_dataset(self):
        """Recompute feature results when the source dataset toggle changes."""
        if self.feature_df.empty:
            return

        df = self._feature_source_dataframe()
        if df.empty:
            self.feature_df = pd.DataFrame()
            self.refresh_feature_tables()
            return

        signals = self.feature_numeric_picker.selected_items() + self.feature_non_numeric_picker.selected_items()
        if not signals:
            self.feature_df = pd.DataFrame()
            self.refresh_feature_tables()
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
        self.refresh_feature_tables()

    def refresh_feature_tables(self):
        """Refresh the feature dataset preview and summary models."""
        self.feature_preview_model.set_data(self.feature_df)
        self.feature_summary_model.set_data(file_summary(self.feature_df))

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

    # ============================================================================
    # Analysis
    # ============================================================================
    def on_analysis_type_changed(self, value):
        self.active_analysis = value
        uses_heatmap = value == "Correlation"
        non_numeric_allowed = value == "Correlation"
        self.analysis_include_label_checkbox.setVisible(False)
        self.analysis_cmap_label.setVisible(uses_heatmap)
        self.analysis_cmap_combo.setVisible(uses_heatmap)
        self.analysis_non_numeric_checkbox.setEnabled(non_numeric_allowed)
        self.analysis_non_numeric_checkbox.setVisible(non_numeric_allowed)
        if not non_numeric_allowed:
            self.analysis_non_numeric_checkbox.blockSignals(True)
            self.analysis_non_numeric_checkbox.setChecked(False)
            self.analysis_non_numeric_checkbox.blockSignals(False)
            self.analysis_matrix_type = "Numeric"
        self._refresh_analysis_column_picker_visibility()
        if value == "Mutual Information":
            self.analysis_include_label_checkbox.blockSignals(True)
            self.analysis_include_label_checkbox.setChecked(True)
            self.analysis_include_label = True
            self.analysis_include_label_checkbox.blockSignals(False)
        else:
            self.analysis_include_label = False

    def on_analysis_include_label_toggled(self, checked):
        if self.analysis_type_combo.currentText() == "Mutual Information":
            checked = True
            self.analysis_include_label_checkbox.blockSignals(True)
            self.analysis_include_label_checkbox.setChecked(True)
            self.analysis_include_label_checkbox.blockSignals(False)
        self.analysis_include_label = checked

    def on_analysis_matrix_type_changed(self, checked):
        self.analysis_matrix_type = "Non-Numeric" if checked else "Numeric"
        self._refresh_analysis_column_picker_visibility()

    def _refresh_analysis_column_picker_visibility(self):
        use_non_numeric = self.analysis_matrix_type == "Non-Numeric"
        self.analysis_numeric_picker.setVisible(not use_non_numeric)
        self.analysis_non_numeric_picker.setVisible(use_non_numeric)
        self.analysis_numeric_label.setVisible(not use_non_numeric)
        self.analysis_non_numeric_label.setVisible(use_non_numeric and self.analysis_non_numeric_checkbox.isVisible())

    def on_analysis_cmap_changed(self, value):
        self.analysis_cmap = value
        if not self.latest_correlation_matrix.empty:
            self.analysis_chart.plot_correlation_heatmap(self.latest_correlation_matrix, cmap=self.analysis_cmap)

    def on_chart_type_changed(self, _value):
        self.populate_visualization_controls()

    def run_current_analysis(self):
        analysis_type = self.analysis_type_combo.currentText()
        if analysis_type == "Correlation":
            self.show_correlation()
        elif analysis_type == "PCA":
            self.show_pca()
        else:
            self.show_mutual_information()

    def show_correlation(self):
        """Run correlation analysis and show matrix plus heatmap."""
        df = self._get_active_dataframe(use_raw=self.analysis_use_raw)
        if df.empty:
            QMessageBox.warning(self, "Missing Dataset", "Open a dataset first.")
            return

        if self.analysis_matrix_type == "Numeric":
            candidate_columns = self.analysis_numeric_picker.selected_items() or self.numeric_columns(
                use_raw=self.analysis_use_raw,
                selected_only=True,
                include_label=True,
            )
            if len(candidate_columns) < 2:
                QMessageBox.warning(self, "Correlation Unavailable", "Select at least two numeric columns for a numeric correlation matrix.")
                return
            matrix_df = df[candidate_columns].select_dtypes(include="number")
        else:
            candidate_columns = self.analysis_non_numeric_picker.selected_items() or self.non_numeric_columns(
                use_raw=self.analysis_use_raw,
                selected_only=True,
                include_label=True,
            )
            if not candidate_columns:
                QMessageBox.warning(self, "Correlation Unavailable", "Select at least one non-numeric column for a non-numeric correlation matrix.")
                return
            matrix_df = pd.get_dummies(df[candidate_columns].copy(), dummy_na=False)
            if matrix_df.shape[1] < 2:
                QMessageBox.warning(self, "Correlation Unavailable", "The selected non-numeric columns do not produce a useful correlation matrix.")
                return

        try:
            corr = matrix_df.corr().round(4)
        except Exception as error:
            self.show_error("Correlation Error", error)
            return

        self.analysis_title.setText(f"CORRELATION MATRIX - {self.analysis_matrix_type.upper()}")
        self.latest_correlation_matrix = corr.copy()
        self.analysis_model.set_data(corr)
        self.analysis_chart.plot_correlation_heatmap(corr, cmap=self.analysis_cmap)
        self.on_workflow_tab_changed(2)

    def show_pca(self):
        """Run the existing PCA analysis and show table plus PC scatter chart."""
        df = self.og_df if self.analysis_use_raw else (self.working_df if not self.working_df.empty else self.og_df)
        selected_label = self.get_selected_label()
        features = self.analysis_numeric_picker.selected_items() or self.numeric_columns(
            use_raw=self.analysis_use_raw,
            selected_only=True,
            include_label=False,
        ) if not df.empty else []
        features = [feature for feature in features if feature in df.columns and feature != selected_label]
        if len(features) < 2:
            QMessageBox.warning(self, "PCA Unavailable", "PCA needs at least two numeric feature columns.")
            return

        try:
            result = pca_analysis(df, features, None, n_components=2)
            pca_df = result["pca_df"].head(200).copy()
            pca_df.attrs["explained_variance_sum"] = result["explained_variance_sum"]
        except Exception as error:
            self.show_error("PCA Error", error)
            return

        self.analysis_title.setText(
            f"PCA PREVIEW - EXPLAINED VARIANCE {result['explained_variance_sum']:.2%}"
        )
        self.analysis_model.set_data(pca_df.round(4))
        self.analysis_chart.plot_pca_scatter(pca_df, None, cmap=self.analysis_cmap)
        self.on_workflow_tab_changed(2)

    def show_mutual_information(self):
        """Run mutual information analysis and show table plus score chart."""
        df = self.og_df if self.analysis_use_raw else (self.working_df if not self.working_df.empty else self.og_df)
        label = self.get_selected_label()
        features = self.analysis_numeric_picker.selected_items() or self.numeric_columns(
            use_raw=self.analysis_use_raw,
            selected_only=True,
            include_label=True,
        ) if not df.empty else []
        features = [feature for feature in features if feature in df.columns and feature != label]
        if not features or not label:
            QMessageBox.warning(self, "Analysis Unavailable", "Select numeric features and a response column.")
            return

        try:
            result = mutual_information_analysis(df, features, label)
        except Exception as error:
            self.show_error("Mutual Information Error", error)
            return

        self.analysis_title.setText("MUTUAL INFORMATION")
        self.analysis_model.set_data(result.round(4))
        self.analysis_chart.plot_mutual_information(result)
        self.on_workflow_tab_changed(2)

    # ============================================================================
    # Visualization
    # ============================================================================

    def refresh_visualization_summary(self):
        """Refresh the small project/dataset summary under the Visualization chart."""
        df = self._get_active_dataframe(use_raw=self.visualization_use_raw)
        summary = file_summary(df)
        if self.project:
            extra = pd.DataFrame(
                [
                    ("Project", self.project.get("project_name", "")),
                    ("Dataset", self.project.get("dataset", "")),
                    ("Label", self.project.get("label_column", self.get_selected_label())),
                    ("Feature Rows", len(self.feature_df)),
                ],
                columns=["metric", "value"],
            )
            summary = pd.concat([extra, summary], ignore_index=True)
        self.visualization_model.set_data(summary)

    def populate_visualization_controls(self):
        """Populate chart column dropdowns from the active dataset columns."""
        df = self._get_active_dataframe(use_raw=self.visualization_use_raw)
        numeric_columns = self.numeric_columns(use_raw=self.visualization_use_raw)
        date_columns = self.date_columns(use_raw=self.visualization_use_raw)
        categorical_columns = self.categorical_columns(use_raw=self.visualization_use_raw)
        chart_type = self.chart_type_combo.currentText()
        if chart_type in ("Histogram", "Box Plot", "Scatter", "Class Separation", "3D Scatter"):
            x_options = list(dict.fromkeys(numeric_columns))
            y_options = list(dict.fromkeys(numeric_columns))
            z_options = list(dict.fromkeys(numeric_columns))
        elif chart_type == "Line":
            x_options = list(dict.fromkeys(date_columns + categorical_columns + list(df.columns)))
            y_options = list(dict.fromkeys(numeric_columns))
            z_options = []
        elif chart_type == "Bar Chart":
            x_options = categorical_columns or list(df.columns)
            y_options = numeric_columns or list(df.columns)
            z_options = []
        elif chart_type == "Grouped Box Plot":
            x_options = categorical_columns or list(df.columns)
            y_options = numeric_columns or list(df.columns)
            z_options = []
        else:
            x_options = list(df.columns)
            y_options = list(df.columns)
            z_options = list(df.columns)

        for combo, options in ((self.chart_x_combo, x_options), (self.chart_y_combo, y_options), (self.chart_z_combo, z_options)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([str(col) for col in options])
            combo.blockSignals(False)

        if x_options:
            self.chart_x_combo.setCurrentText(x_options[0])
        if y_options:
            self.chart_y_combo.setCurrentText(y_options[1] if len(y_options) > 1 else y_options[0])
        if z_options:
            self.chart_z_combo.setCurrentText(z_options[2] if len(z_options) > 2 else z_options[0])

        self.chart_multi_picker.blockSignals(True)
        self.chart_multi_picker.set_items(numeric_columns, checked=True)
        self.chart_multi_picker.blockSignals(False)

        self.chart_input_hint.setText("")
        self.chart_input_hint.setVisible(False)
        self.update_chart_controls()

    def update_chart_controls(self):
        """Enable only the chart inputs required by the selected chart type."""
        chart_type = self.chart_type_combo.currentText()
        needs_y = chart_type in (
            "Scatter", "Line", "Bar Chart", "Grouped Box Plot",
            "Class Separation", "3D Scatter",
        )
        needs_x = chart_type in (
            "Histogram", "Scatter", "Line", "Box Plot", "Bar Chart",
            "Grouped Box Plot", "Class Separation", "3D Scatter",
        )
        needs_z = chart_type == "3D Scatter"
        needs_bins = chart_type == "Histogram"
        needs_multi = chart_type in ("Time Series (All Signals)", "Feature Distribution Comparison")
        needs_lines = chart_type in ("Histogram", "Box Plot")

        field_label = {
            "Histogram": "Numeric X column",
            "Scatter": "Numeric X/Y columns",
            "Line": "Date, category, or text X column with numeric Y values",
            "Box Plot": "Numeric column",
            "Bar Chart": "Category + numeric value",
            "Grouped Box Plot": "Category + numeric value",
            "Class Separation": "Numeric X/Y columns",
            "3D Scatter": "Numeric X/Y/Z columns",
            "Time Series (All Signals)": "Numeric signal columns",
            "Feature Distribution Comparison": "Numeric signal columns",
            "Correlation Heatmap": "Numeric columns",
        }.get(chart_type, "Columns")

        self.chart_x_combo.setVisible(needs_x)
        self.chart_y_combo.setVisible(needs_y)
        self.chart_z_combo.setVisible(needs_z)
        self.chart_x_label.setVisible(needs_x)
        self.chart_y_label.setVisible(needs_y)
        self.chart_z_label.setVisible(needs_z)
        self.chart_bins_spin.setVisible(needs_bins)
        self.chart_bins_label.setVisible(needs_bins)
        self.chart_multi_picker.setVisible(needs_multi)
        self.chart_multi_label.setVisible(needs_multi)
        self.chart_mean_line_checkbox.setVisible(needs_lines)
        self.chart_median_line_checkbox.setVisible(needs_lines)
        self.chart_input_hint.setText("")
        self.chart_input_hint.setVisible(False)

    def render_visualization(self):
        """Render the selected generic visualization against the active dataset."""
        df = self._get_active_dataframe(use_raw=self.visualization_use_raw)
        if df.empty:
            self.chart_canvas.show_empty("Open a dataset to visualize it.")
            return

        self.chart_canvas.plot(
            df,
            self.chart_type_combo.currentText(),
            self.chart_x_combo.currentText(),
            self.chart_y_combo.currentText(),
            self.get_selected_label(),
            z_col=self.chart_z_combo.currentText(),
            bins=self.chart_bins_spin.value(),
            extra_cols=self.chart_multi_picker.selected_items(),
            show_median_line=self.chart_median_line_checkbox.isChecked(),
            show_mean_line=self.chart_mean_line_checkbox.isChecked(),
        )

    def export_chart_gui(self):
        """Save the currently rendered Visualization-tab chart to an image file."""
        default_path = self.downloads_dir / "chart.png"
        default_path.parent.mkdir(exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Chart",
            str(default_path.resolve()),
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Image (*.svg)",
        )
        if not path:
            return

        try:
            export_plot_image(self.chart_canvas.figure, path)
        except Exception as error:
            self.show_error("Export Error", error)
            return

        QMessageBox.information(self, "Chart Exported", f"Saved {path}")

    def date_columns(self, use_raw=False):
        """Return columns that can be plotted as date-like values."""
        df = self._get_active_dataframe(use_raw=use_raw)
        if df.empty:
            return []

        date_columns = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_columns.append(str(col))
                continue
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > 0 and parsed.notna().sum() >= max(1, len(parsed) // 2):
                    date_columns.append(str(col))
            except Exception:
                continue
        return date_columns

    def numeric_columns(self, use_raw=False, selected_only=False, include_label=False):
        """Return numeric columns from the active DataFrame for chart defaults."""
        df = self._get_active_dataframe(use_raw=use_raw)
        if df.empty:
            return []

        candidate_columns = list(df.columns)
        if selected_only:
            candidate_columns = self._selected_import_columns(use_raw=use_raw, include_label=include_label)

        return [str(col) for col in candidate_columns if col in df.columns and self._is_effectively_numeric(df[col])]

    def non_numeric_columns(self, use_raw=False, selected_only=False, include_label=False):
        """Return non-numeric columns from the active DataFrame."""
        df = self._get_active_dataframe(use_raw=use_raw)
        if df.empty:
            return []

        candidate_columns = list(df.columns)
        if selected_only:
            candidate_columns = self._selected_import_columns(use_raw=use_raw, include_label=include_label)

        return [str(col) for col in candidate_columns if col in df.columns and not self._is_effectively_numeric(df[col])]

    def categorical_columns(self, use_raw=False):
        """Return categorical-like columns for visualization defaults."""
        df = self._get_active_dataframe(use_raw=use_raw)
        if df.empty:
            return []
        return [str(col) for col in df.select_dtypes(exclude="number").columns]

    def export_analysis_image(self):
        """Export the currently rendered analysis figure to an image file."""
        default_path = self.downloads_dir / "analysis.png"
        default_path.parent.mkdir(exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis Image",
            str(default_path.resolve()),
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Image (*.svg)",
        )
        if not path:
            return
        try:
            export_plot_image(self.analysis_chart.figure, path)
        except Exception as error:
            self.show_error("Export Error", error)
            return
        QMessageBox.information(self, "Export Complete", f"Saved {path}")

    # ============================================================================
    # Utility Functions
    # ============================================================================
    def show_error(self, title, error):
        QMessageBox.critical(self, title, str(error))