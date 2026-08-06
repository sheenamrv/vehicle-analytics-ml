from pathlib import Path
from datetime import datetime
import tempfile
import json
import joblib

import numpy as np
import pandas as pd
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
    QMenu,
    QDialog,
    QPushButton,
    QTableView,
    QCheckBox,
    QButtonGroup,
    QAbstractSpinBox,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QSplitter,
)
from PySide6.QtCore import (
    Qt,
    QThreadPool,
    QItemSelectionModel,
)
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.analysis.analysis import (
    correlation_analysis,
    get_num_feature_columns,
    mutual_information_analysis,
    pca_analysis,
)
from src.data.process import (
    create_model_package,
    load_project,
    make_json_safe,
    save_project,
    unpack_model_package,
)
from src.model.model_registry import add_model, delete_model
from src.model.model_training import test_models_current_data
from src.model.model_utils import validate_dataset, prepare_training_data
from src.model.supervised_model import build_model
from src.frontend.model_panel import ModelParameterPanel, MODEL_TYPES
from src.data.test_load import get_available_columns, get_datasets, select_col, change_dtype
from src.feature.feature import (
    build_window_index_table,
    extract_windowed_feature_dataset,
    feature_extract,
    to_feature_label,
)
from src.frontend.chart_specs import (
    CHART_TYPES,
    chart_column_options,
    get_chart_spec,
    visualization_validation_error,
)
from src.frontend.charts import ChartCanvas
from src.frontend.data_summary import file_summary, missing_summary
from src.frontend.data_quality_dialog import (
    DataQualityDialog,
    QualityIssueTableModel,
)
from src.frontend.image_dialogs import (
    ImageInspectDialog,
    ReportImagesDialog,
    WindowPreviewDialog,
)
from src.frontend.unified_model_panel import (
    UnifiedModelPage,
    UnifiedModelSidebar,
)
from src.playback_annotation.playback_annotation import PlaybackAnnotationManager
from src.frontend.playback_page import PlaybackAnnotationPage, PlaybackSidebar
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
    WheelLockedComboBox,
    WheelLockedDoubleSpinBox,
    WheelLockedSpinBox,
    tab_row,
    table_view,
    taller_dropdown,
)
from src.frontend.workers import (
    DataLoadWorker,
    ModelEvaluationWorker,
    ModelTrainingWorker,
    SemiSupervisedTrainingWorker,
    UnifiedModelTrainingWorker,
    UnsupervisedTrainingWorker,
    WorkerSignals,
)
from src.model.model_controller import ModelController
from src.model.model_factory import build_model
from src.model.result_builders import (
    export_model_report,
    generate_model_report_assets,
    render_comparison_metrics_image,
    render_combined_confusion_matrices_image,
    render_model_specific_comparison_image,
)
from src.model.model_utils import prepare_training_data, align_features

from src.frontend.results_controller import ResultsControllerMixin

from src.frontend.model_training_controller import ModelTrainingControllerMixin

from src.frontend.data_prep_controller import DataPrepControllerMixin



from src.frontend.feature_controller import (
    FeatureExtractionControllerMixin,
    FEATURE_CATEGORIES,
    FeaturePickerCompat,
)

from src.frontend.analysis_visualization_controller import AnalysisVisualizationControllerMixin


'''
    Main font end window for the application
'''

# ============================================================================
# Constants
# ============================================================================

# ============================================================================
# Main Application Window
# ============================================================================

class AnalyticsWindow(AnalysisVisualizationControllerMixin, FeatureExtractionControllerMixin, DataPrepControllerMixin, ModelTrainingControllerMixin, ResultsControllerMixin, QMainWindow):
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
        self._active_project_name = ""
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
        self.feature_summary_meta = {}
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
        self._result_records = []
        self._comparison_records = []
        self._comparison_selected_names = set()
        self._comparison_metric_image_path = None
        self._comparison_cm_image_path = None
        self._comparison_model_specific_ssl_image_path = None
        self._comparison_model_specific_unsup_image_path = None
        self._comparison_zoom_levels = {
            "metrics": 1.0,
            "cm": 1.0,
            "model_ssl": 1.0,
            "model_unsup": 1.0,
        }
        self._comparison_base_pixmaps = {
            "metrics": None,
            "cm": None,
            "model_ssl": None,
            "model_unsup": None,
        }
        self._preview_report_root = Path(tempfile.gettempdir()) / "vehicle_analytics_result_previews"
        self._preview_report_root.mkdir(parents=True, exist_ok=True)

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
        # self.workflow_tabs_container = QWidget()
        # self.workflow_tabs_container.setLayout(self.workflow_tabs["layout"])
        # content_layout.addWidget(self.workflow_tabs_container)
        self.workflow_tab_container = QWidget()
        self.workflow_tab_container.setLayout(self.workflow_tabs["layout"])
        content_layout.addWidget(self.workflow_tab_container)

        self.main_stack = QStackedWidget()
        content_layout.addWidget(self.main_stack, 1)

        # Page order must match sidebar_stack order and workflow tab indexes.
        self._build_import_page()
        self._build_feature_page()
        self._build_analysis_page()
        self._build_visualization_page()
        self._build_models_page()
        self._build_results_page()

        # Playback/annotation
        # always reads the current working dataset
        self.playback_sidebar = PlaybackSidebar()
        self.playback_page = PlaybackAnnotationPage(
            self.playback_sidebar,
            self._active_realtime_dataset,
            self,
        )
        self.playback_page_scroll = self._wrap_main_page(self.playback_page)
        self.main_stack.addWidget(self.playback_page_scroll)

        self.import_sidebar = self._scrollable_sidebar(self._import_sidebar())
        self.feature_sidebar = self._scrollable_sidebar(self._feature_sidebar())
        self.analysis_sidebar = self._scrollable_sidebar(self._analysis_sidebar())
        self.visualization_sidebar = self._scrollable_sidebar(self._visualization_sidebar())

        self.sidebar_stack.addWidget(self.import_sidebar)
        self.sidebar_stack.addWidget(self.feature_sidebar)
        self.sidebar_stack.addWidget(self.analysis_sidebar)
        self.sidebar_stack.addWidget(self.visualization_sidebar)
        # self.sidebar_stack.addWidget(self._models_sidebar())
        self.model_sidebar = UnifiedModelSidebar()
        self.model_sidebar.add_model_requested.connect(self.add_or_update_model_definition)
        self.model_sidebar.import_external_requested.connect(self.import_external_model)
        self.model_sidebar_scroll = self._scrollable_sidebar(self.model_sidebar)
        self.sidebar_stack.addWidget(self.model_sidebar_scroll)
        self.results_sidebar = self._scrollable_sidebar(self._results_sidebar())
        self.sidebar_stack.addWidget(self.results_sidebar)
        self.playback_sidebar_scroll = self._scrollable_sidebar(self.playback_sidebar)
        self.sidebar_stack.addWidget(self.playback_sidebar_scroll)

        self.on_top_tab_changed(0)
        self.on_workflow_tab_changed(0)

    def _scrollable_sidebar(self, content):
        scroll = QScrollArea()
        scroll.setProperty("sidebarScroll", True)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    def _wrap_main_page(self, page):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(page)
        return scroll

# ============================================================================
# Sidebar Construction
# ============================================================================





    # def _models_sidebar(self):
    #     """Build training, selection, and testing controls for the Models tab."""
    #     panel = sidebar_base()
    #     layout = panel.layout()

    #     layout.addWidget(section_label("TRAIN MODEL"))

    #     self.model_type_combo = taller_dropdown(QComboBox())
    #     self.model_type_combo.addItems([label for label, _ in MODEL_TYPES])
    #     self.model_type_combo.currentIndexChanged.connect(self.update_model_parameter_fields)
    #     layout.addWidget(QLabel("Model Type"))
    #     layout.addWidget(self.model_type_combo)

    #     self.model_param_panel = ModelParameterPanel()
    #     layout.addWidget(self.model_param_panel)

    #     self.model_test_size_spin = QDoubleSpinBox()
    #     self.model_test_size_spin.setRange(0.05, 0.95)
    #     self.model_test_size_spin.setSingleStep(0.05)
    #     self.model_test_size_spin.setValue(0.3)
    #     layout.addWidget(QLabel("Test Split"))
    #     layout.addWidget(self.model_test_size_spin)

    #     self.model_random_state_spin = QSpinBox()
    #     self.model_random_state_spin.setRange(0, 9999)
    #     self.model_random_state_spin.setValue(42)
    #     layout.addWidget(QLabel("Random State"))
    #     layout.addWidget(self.model_random_state_spin)

    #     self.model_name_edit = QLineEdit()
    #     self.model_name_edit.setPlaceholderText("Model display name")
    #     layout.addWidget(QLabel("Model Name"))
    #     layout.addWidget(self.model_name_edit)

    #     train_button = primary_button("Train Model")
    #     train_button.clicked.connect(self.train_new_model_gui)
    #     layout.addWidget(train_button)
    #     layout.addWidget(divider())

    #     layout.addWidget(section_label("MANAGE MODELS"))

    #     self.model_select_combo = taller_dropdown(QComboBox())
    #     self.model_select_combo.currentTextChanged.connect(self.show_model_details_gui)
    #     layout.addWidget(QLabel("Select Model"))
    #     layout.addWidget(self.model_select_combo)

    #     delete_button = secondary_button("Delete Selected Model")
    #     delete_button.clicked.connect(self.delete_selected_model)
    #     layout.addWidget(delete_button)

    #     self.model_test_picker = ColumnPicker("Search models")
    #     self.model_test_picker.setMinimumHeight(100)
    #     layout.addWidget(QLabel("Models to Test"))
    #     layout.addWidget(self.model_test_picker)

    #     test_button = primary_button("Test Selected Models")
    #     test_button.clicked.connect(self.test_selected_models_gui)
    #     layout.addWidget(test_button)

    #     layout.addStretch()
    #     return panel



# ============================================================================
# Workflow Page Construction
# ============================================================================

    # ============================================================================
    # Data Type Conversion
    # ============================================================================









# ============================================================================
# Navigation & Tab Management
# ============================================================================

    def on_top_tab_changed(self, index):
        if index < 0 or index >= len(self.top_tabs["buttons"]):
            return

        # Check only the selected top navigation button
        for button_index, button in enumerate(self.top_tabs["buttons"]):
            button.setChecked(button_index == index)

        # Data & Features
        if index == 0:
            self.workflow_tab_container.setVisible(True)

            workflow_index = self.workflow_tabs["group"].checkedId()
            if workflow_index < 0:
                workflow_index = 0

            self.on_workflow_tab_changed(workflow_index)
            return

        # Models, Results, and Playback do not use the workflow subtabs
        self.workflow_tab_container.setVisible(False)

        # Models
        if index == 1:
            self.main_stack.setCurrentWidget(self.model_page_scroll)
            self.sidebar_stack.setCurrentWidget(self.model_sidebar_scroll)
            self.refresh_model_page()
            return

        # Results
        if index == 2:
            self.main_stack.setCurrentWidget(self.results_page)
            self.sidebar_stack.setCurrentWidget(self.results_sidebar)
            self.refresh_results_page()
            return

        # Playback & Annotation
        if index == 3:
            self.main_stack.setCurrentWidget(self.playback_page_scroll)
            self.sidebar_stack.setCurrentWidget(self.playback_sidebar_scroll)
            self.refresh_realtime_dataset()
            return

    # ============================================================================
    # Models
    # ============================================================================
    # def update_model_parameter_fields(self):
    #     """Swap the visible parameter fields for the selected model type."""
    #     _, model_type = MODEL_TYPES[self.model_type_combo.currentIndex()]
    #     self.model_param_panel.set_model_type(model_type)

    # def refresh_models_list(self):
    #     """Refresh the model dropdown/picker and the saved-models table."""
    #     names = [m["display_name"] for m in self.project.get("models", [])] if self.project else []

    #     self.model_select_combo.blockSignals(True)
    #     self.model_select_combo.clear()
    #     self.model_select_combo.addItems(names)
    #     self.model_select_combo.blockSignals(False)

    #     self.model_test_picker.blockSignals(True)
    #     self.model_test_picker.set_items(names, checked=False)
    #     self.model_test_picker.blockSignals(False)

    #     self.show_model_details_gui()

    # def show_model_details_gui(self):
    #     """Show details for the selected model, or a summary list if none selected."""
    #     if not self.project or not self.project.get("models"):
    #         self.models_title.setText("SAVED MODELS")
    #         self.models_model.set_data(pd.DataFrame())
    #         return

    #     name = self.model_select_combo.currentText()
    #     match = next((m for m in self.project["models"] if m["display_name"] == name), None)

    #     if not match:
    #         self.models_title.setText("SAVED MODELS")
    #         rows = [(m["display_name"], m["algorithm"]) for m in self.project["models"]]
    #         self.models_model.set_data(pd.DataFrame(rows, columns=["display_name", "algorithm"]))
    #         return

    #     self.models_title.setText(f"MODEL DETAILS - {match['display_name']}")
    #     rows = [("algorithm", match["algorithm"])]
    #     for key, value in match.get("parameters", {}).items():
    #         rows.append((f"param: {key}", value))
    #     for key, value in match.get("metrics", {}).items():
    #         rows.append((f"metric: {key}", round(value, 4) if isinstance(value, float) else value))
    #     self.models_model.set_data(pd.DataFrame(rows, columns=["field", "value"]))

    # def train_new_model_gui(self):
    #     """Train a model using the sidebar's selections and add it to the project."""
    #     if not self.project:
    #         QMessageBox.warning(self, "No Project", "Create or load a project before training a model.")
    #         return

    #     df = self.working_df if not self.working_df.empty else self.og_df
    #     label = self.project.get("label_column") or self.get_selected_label()

    #     valid, message = validate_dataset(df, label)
    #     if not valid:
    #         QMessageBox.warning(self, "Invalid Dataset", message)
    #         return

    #     try:
    #         X, y = prepare_training_data(df, label)
    #         feature_columns = X.columns.tolist()
    #         _, model_type = MODEL_TYPES[self.model_type_combo.currentIndex()]
    #         parameters = self.model_param_panel.get_parameters()
    #         config = {
    #             "test_size": self.model_test_size_spin.value(),
    #             "random_state": self.model_random_state_spin.value(),
    #         }
    #         model = build_model(model_type, parameters)
    #         trained_model, metrics = build_model(model_type, parameters), {"accuracy": None}
    #     except Exception as error:
    #         self.show_error("Training Error", error)
    #         return

    #     display_name = self.model_name_edit.text().strip() or (
    #         f"{model_type}_{len(self.project.get('models', [])) + 1}"
    #     )

    #     add_model(self.project, trained_model, display_name, model_type, {**config, **parameters}, metrics, feature_columns)
    #     self._set_dirty(True)
    #     self.model_name_edit.clear()
    #     self.refresh_models_list()
    #     QMessageBox.information(self, "Model Trained", f"'{display_name}' trained and added to the project.")

    # def delete_selected_model(self):
    #     """Delete the model currently selected in the sidebar dropdown."""
    #     if not self.project or not self.project.get("models"):
    #         return

    #     name = self.model_select_combo.currentText()
    #     if not name:
    #         return

    #     result = QMessageBox.question(
    #         self,
    #         "Confirm Delete",
    #         f"Delete model '{name}'?",
    #         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    #         QMessageBox.StandardButton.No,
    #     )
    #     if result != QMessageBox.StandardButton.Yes:
    #         return

    #     delete_model(self.project, name)
    #     self._set_dirty(True)
    #     self.refresh_models_list()

    # def test_selected_models_gui(self):
    #     """Test the checked models from the picker against the working dataset."""
    #     if not self.project or not self.project.get("models"):
    #         QMessageBox.warning(self, "No Models", "Train or load at least one model first.")
    #         return

    #     names = self.model_test_picker.selected_items()
    #     if not names:
    #         QMessageBox.warning(self, "No Selection", "Select at least one model to test.")
    #         return

    #     df = self.working_df if not self.working_df.empty else self.og_df
    #     label = self.project.get("label_column") or self.get_selected_label()
    #     models = [m for m in self.project["models"] if m["display_name"] in names]

    #     try:
    #         results = test_models_current_data(models, df, label)
    #     except Exception as error:
    #         self.show_error("Test Error", error)
    #         return

    #     rows = [
    #         {
    #             "model": r["name"],
    #             "accuracy": round(r["accuracy"], 4),
    #             "precision": round(r["precision"], 4),
    #             "recall": round(r["recall"], 4),
    #             "f1": round(r["f1"], 4),
    #         }
    #         for r in results
    #     ]
    #     self.models_title.setText("MODEL TEST RESULTS")
    #     self.models_model.set_data(pd.DataFrame(rows))
        # self.top_tabs["buttons"][index].setChecked(True)
        # self.workflow_tab_container.setVisible(True)
        # self.on_workflow_tab_changed(self.workflow_tabs["group"].checkedId())

    def on_workflow_tab_changed(self, index):
        """Switch the Data & Features workflow page and sidebar."""
        if index < 0 or index >= len(self.workflow_tabs["buttons"]):
            return

        for button_index, button in enumerate(self.workflow_tabs["buttons"]):
            button.setChecked(button_index == index)

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
        title = self.project.get("project_name") if self.project else (self.project_name.text().strip() or "Classify & Learn Lab")
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



























    # ============================================================================
    # Supervised Models
    # ============================================================================































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

            if self.project is None:
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
                    "added_models": [],
                    "model_queue": [],
                }
                base_dir = self.current_project_path.parent if self.current_project_path else self.downloads_dir
                target_path = base_dir / f"{project_name}.icp"
                target_path.parent.mkdir(parents=True, exist_ok=True)
                self.current_project_path = target_path

                save_project(
                    self.project,
                    self.og_df,
                    self.working_df,
                    str(self.current_project_path),
                    feature_df=self.feature_df.copy(),
                )
                self._active_project_name = project_name
                self.project["project_name"] = self._active_project_name
                self._set_dirty(False)
            else:
                active_name = str(
                    self._active_project_name
                    or self.project.get("project_name")
                    or (self.current_project_path.stem if self.current_project_path else "project")
                ).strip()
                self._active_project_name = active_name
                self.project["project_name"] = active_name
                self.project["selected_columns"] = list(self.working_df.columns)
                self.project["label_column"] = label_col
                self.project["column_types"] = {
                    col: str(self.working_df[col].dtype)
                    for col in self.working_df.columns
                }
                self.project["file_path"] = str(self.file_path) if self.file_path is not None else self.project.get("file_path")

                if project_name == active_name:
                    # Do not create a duplicate copy with the currently active name.
                    return self.save_current_project()

                base_dir = self.current_project_path.parent if self.current_project_path else self.downloads_dir
                copy_path = base_dir / f"{project_name}.icp"
                copy_path.parent.mkdir(parents=True, exist_ok=True)
                project_copy = dict(self.project)
                project_copy["project_name"] = project_name
                save_project(
                    project_copy,
                    self.og_df.copy(),
                    self.working_df.copy(),
                    str(copy_path),
                    feature_df=self.feature_df.copy(),
                )

                # Continue on the active project after creating the copy.
                self._suppress_dirty = True
                try:
                    self.project_name.setText(active_name)
                finally:
                    self._suppress_dirty = False
                self._set_dirty(False)
                QMessageBox.information(
                    self,
                    "Project Copy Created",
                    f"Saved copy as {copy_path}. Continuing with {active_name}.",
                )
                return True
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

        active_name = str(self._active_project_name or self.project.get("project_name") or self.current_project_path.stem or "project").strip()
        self._active_project_name = active_name
        self.project["project_name"] = active_name

        self.project["selected_columns"] = list(self.working_df.columns)
        self.project["column_types"] = {col: str(self.working_df[col].dtype) for col in self.working_df.columns}

        try:
            self.project["file_path"] = str(self.file_path) if self.file_path is not None else self.project.get("file_path")
            if not self._save_named_copy_if_requested(active_name):
                return False
            # feature_df is persisted separately from raw and working data so
            # imported/extracted features survive project reloads.
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

    def _save_named_copy_if_requested(self, active_name):
        requested_name = self.project_name.text().strip()
        if not requested_name or requested_name == active_name:
            return True

        copy_path = self.current_project_path.with_name(f"{requested_name}.icp")
        if copy_path.exists():
            result = QMessageBox.question(
                self,
                "Overwrite Project Copy",
                f"A project named '{copy_path.name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if result != QMessageBox.StandardButton.Yes:
                return False

        project_copy = dict(self.project)
        project_copy["project_name"] = requested_name
        save_project(
            project_copy,
            self.og_df.copy(),
            self.working_df.copy(),
            str(copy_path),
            feature_df=self.feature_df.copy(),
        )

        # Keep working on the current project after creating the renamed copy.
        self._suppress_dirty = True
        try:
            self.project_name.setText(active_name)
        finally:
            self._suppress_dirty = False

        QMessageBox.information(
            self,
            "Project Copy Saved",
            f"Saved copy as {copy_path.name}. Continuing with {active_name}.",
        )
        return True

    def save_project_as(self):
        if self.project is None:
            QMessageBox.warning(self, "No Project", "Create or open a project before saving.")
            return False

        project_name = self.project_name.text().strip() or self.project.get("project_name") or "project"
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




    # ============================================================================
    # Feature Extraction
    # ============================================================================
















    # ============================================================================
    # Analysis
    # ============================================================================










    # ============================================================================
    # Visualization
    # ============================================================================



        # Keep these choices synchronized with visualization_validation_error;
        # filtering improves UX while validation remains the safety boundary.
        # df = self.working_df if not self.working_df.empty else self.og_df
        # numeric = [str(column) for column in df.select_dtypes(include="number").columns] if not df.empty else []
        # temporal = [str(column) for column in df.select_dtypes(include=["datetime", "datetimetz"]).columns] if not df.empty else []
        # categorical = [str(column) for column in df.columns if str(column) not in numeric and str(column) not in temporal] if not df.empty else []
        # x_choices, y_choices = numeric, numeric
        # if chart_type == "Line":
        #     x_choices = numeric + temporal
        # elif chart_type == "Bar Chart":
        #     x_choices = categorical
        # elif chart_type == "Grouped Box Plot":
        #     x_choices = categorical

        # for combo, choices in ((self.chart_x_combo, x_choices), (self.chart_y_combo, y_choices)):
        #     current = combo.currentText()
        #     combo.blockSignals(True)
        #     combo.clear()
        #     combo.addItems(choices)
        #     if current in choices:
        #         combo.setCurrentText(current)
        #     combo.blockSignals(False)
        # # Extension point: keep this mapping aligned with CHART_TYPES.
        # needs_y = chart_type in ("Scatter", "Line", "Bar Chart", "Grouped Box Plot")
        # needs_x = chart_type in (
        #     "Histogram",
        #     "Scatter",
        #     "Line",
        #     "Box Plot",
        #     "Bar Chart",
        #     "Grouped Box Plot",
        # )
        # self.chart_x_combo.setEnabled(needs_x)
        # self.chart_y_combo.setEnabled(needs_y)
        # self.chart_label_combo.setEnabled(chart_type == "Scatter")

        # if self.project is not None:
        #         "chart_type": self.chart_type_combo.currentText(),
        #         "x_column": self.chart_x_combo.currentText(),
        #         "y_column": self.chart_y_combo.currentText(),
        #         "group_column": label or None,
        #     }
        #     visualizations = self.project.setdefault("visualizations", [])
        #     if not visualizations or visualizations[-1] != configuration:
        #         visualizations.append(configuration)
        #         self._set_dirty(True)








    # For Playback, ML Prediction, and Annotation
    
    # Return the active imported dataset 
    def _active_realtime_dataset(self):
        return PlaybackAnnotationManager.active_dataset(self.working_df, self.og_df)

    # Synchronize the playback page with the current imported dataset
    def refresh_realtime_dataset(self):
        if hasattr(self, "playback_page"):
            self.playback_page.sync_dataset()

    # ============================================================================
    # Utility Functions
    # ============================================================================
    def show_error(self, title, error):
        QMessageBox.critical(self, title, str(error))
