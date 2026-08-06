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


'''
    Main font end window for the application
'''

# ============================================================================
# Constants
# ============================================================================
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

# ============================================================================
# Main Application Window
# ============================================================================

class AnalyticsWindow(DataPrepControllerMixin, ModelTrainingControllerMixin, ResultsControllerMixin, QMainWindow):
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
        # self.chart_label_combo = taller_dropdown(QComboBox())
        # layout.addWidget(QLabel("X / Primary Column"))
        layout.addWidget(self.chart_x_combo)
        layout.addWidget(self.chart_y_label)
        layout.addWidget(self.chart_y_combo)
        layout.addWidget(self.chart_z_label)
        layout.addWidget(self.chart_z_combo)

        self.chart_bins_spin = QSpinBox()
        self.chart_bins_spin.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
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
        # layout.addWidget(QLabel("Color / Group By"))
        # layout.addWidget(self.chart_label_combo)

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

        self.update_chart_controls()
        layout.addStretch()
        return panel

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
        self.analysis_page = page
        self.analysis_page_scroll = self._wrap_main_page(page)
        self.main_stack.addWidget(self.analysis_page_scroll)

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
        self.visualization_page = page
        self.visualization_page_scroll = self._wrap_main_page(page)
        self.main_stack.addWidget(self.visualization_page_scroll)



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
        x_options, y_options, z_options = chart_column_options(
            chart_type=chart_type,
            all_columns=df.columns,
            numeric_columns=numeric_columns,
            date_columns=date_columns,
            categorical_columns=categorical_columns,
        )

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
        # self.chart_label_combo.blockSignals(True)
        # self.chart_label_combo.clear()
        # self.chart_label_combo.addItem("None", "")
        # df = self.working_df if not self.working_df.empty else self.og_df
        # categorical = [
        #     str(column) for column in df.columns
        #     if not pd.api.types.is_numeric_dtype(df[column])
        #     or str(column) == self.label_combo.currentText()
        # ] if not df.empty else []
        # for column in categorical:
        #     self.chart_label_combo.addItem(column, column)
        # label_index = self.chart_label_combo.findData(self.label_combo.currentText())
        # if label_index >= 0:
        #     self.chart_label_combo.setCurrentIndex(label_index)
        # self.chart_label_combo.blockSignals(False)

        # numeric_columns = self.numeric_columns()
        # if numeric_columns:
        #     self.chart_x_combo.setCurrentText(numeric_columns[0])
        #     self.chart_y_combo.setCurrentText(
        #         numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0]
        #     )
        self.update_chart_controls()

    def update_chart_controls(self):
        """Enable only the chart inputs required by the selected chart type."""
        chart_type = self.chart_type_combo.currentText()
        spec = get_chart_spec(chart_type)
        needs_x = spec.needs_x if spec is not None else False
        needs_y = spec.needs_y if spec is not None else False
        needs_z = spec.needs_z if spec is not None else False
        needs_bins = spec.needs_bins if spec is not None else False
        needs_multi = spec.needs_multi if spec is not None else False
        needs_lines = spec.needs_lines if spec is not None else False

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

    def render_visualization(self):
        """Render the selected generic visualization against the active dataset."""
        df = self._get_active_dataframe(use_raw=self.visualization_use_raw)
        if df.empty:
            self.chart_canvas.show_empty("Open a dataset to visualize it.")
            return

        # error = self.visualization_validation_error(
        #     df,
        #     self.chart_type_combo.currentText(),
        #     self.chart_x_combo.currentText(),
        #     self.chart_y_combo.currentText(),
        # )
        # if error:
        #     QMessageBox.warning(self, "Invalid Visualization", error)
        #     return

        # label = self.chart_label_combo.currentData()
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
            # label,
        )
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

    @staticmethod
    def visualization_validation_error(df, chart_type, x_column, y_column):
        return visualization_validation_error(
            df,
            chart_type,
            x_column,
            y_column,
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
