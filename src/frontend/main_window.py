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
from src.frontend.project_controller import ProjectControllerMixin
from src.frontend.app_identity import APP_DISPLAY_NAME, application_icon


class AnalyticsWindow(
    ProjectControllerMixin,
    AnalysisVisualizationControllerMixin,
    FeatureExtractionControllerMixin,
    DataPrepControllerMixin,
    ModelTrainingControllerMixin,
    ResultsControllerMixin,
    QMainWindow,
):
    """Main desktop window and shared frontend state."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.setWindowIcon(application_icon())
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
        self.visualization_use_raw = False
        self.analysis_include_label = True
        self.active_analysis = "Correlation"
        self.analysis_cmap = "viridis"
        self.analysis_matrix_type = "Numeric"

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
        title = self.project.get("project_name") if self.project else (self.project_name.text().strip() or APP_DISPLAY_NAME)
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

    # Playback and annotation always use the active imported dataset.
    def _active_realtime_dataset(self):
        return PlaybackAnnotationManager.active_dataset(self.working_df, self.og_df)

    # Synchronize the playback page with the current imported dataset
    def refresh_realtime_dataset(self):
        if hasattr(self, "playback_page"):
            self.playback_page.sync_dataset()

    def show_error(self, title, error):
        QMessageBox.critical(self, title, str(error))
