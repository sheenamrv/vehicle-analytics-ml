"""Results page construction, selection, reporting, and comparison behavior."""

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.data.process import make_json_safe
from src.frontend.image_dialogs import (
    ImageInspectDialog,
    ReportImagesDialog,
)
from src.frontend.table_model import PandasTableModel
from src.frontend.widgets import (
    data_panel,
    divider,
    primary_button,
    secondary_button,
    section_label,
    sidebar_base,
    tab_row,
    table_view,
    taller_dropdown,
)
from src.model.model_utils import align_features, prepare_training_data
from src.model.result_builders import (
    export_model_report,
    generate_model_report_assets,
    render_combined_confusion_matrices_image,
    render_comparison_metrics_image,
    render_model_specific_comparison_image,
)


class ResultsControllerMixin:
    """Behavior extracted from AnalyticsWindow for results workflows."""

    def _results_sidebar(self):
        panel = sidebar_base()
        layout = panel.layout()
        layout.addWidget(section_label("RESULTS"))
        refresh = primary_button("Refresh Results")
        refresh.clicked.connect(self.refresh_results_page)
        layout.addWidget(refresh)
        export = secondary_button("Export Comparison")
        export.clicked.connect(self.export_results_comparison)
        layout.addWidget(export)

        self.results_comparison_controls = QWidget()
        controls_layout = QVBoxLayout(self.results_comparison_controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        controls_layout.addWidget(divider())
        controls_layout.addWidget(section_label("COMPARE MODELS"))
        self.comparison_model_list = QListWidget()
        self.comparison_model_list.setMinimumHeight(210)
        self.comparison_model_list.setStyleSheet(
            "QListWidget { border: 1px solid #9fb8aa; background: #ffffff; }"
            "QListWidget::item { padding: 3px 8px; }"
            "QCheckBox { font-weight: bold; font-size: 12px; color: #1f2d24; spacing: 8px; }"
            "QListWidget::indicator { width: 16px; height: 16px; }"
            "QListWidget::indicator:unchecked { width: 16px; height: 16px; }"
            "QListWidget::indicator:checked { width: 16px; height: 16px; }"
        )
        controls_layout.addWidget(self.comparison_model_list)

        selection_buttons = QHBoxLayout()
        select_all_button = secondary_button("Select All")
        select_all_button.clicked.connect(self.select_all_comparison_models)
        clear_all_button = secondary_button("Clear")
        clear_all_button.clicked.connect(self.clear_all_comparison_models)
        selection_buttons.addWidget(select_all_button)
        selection_buttons.addWidget(clear_all_button)
        controls_layout.addLayout(selection_buttons)

        controls_layout.addWidget(section_label("COMPARE EXPORT"))
        self.comparison_export_mode = taller_dropdown(QComboBox())
        self.comparison_export_mode.addItems([
            "Metric Image",
            "Confusion Matrix Image",
            "Semi-Supervised Image",
            "Unsupervised Image",
            "Export All 4 Images",
        ])
        controls_layout.addWidget(self.comparison_export_mode)
        export_png = primary_button("Export PNG")
        export_png.clicked.connect(self.export_comparison_images)
        controls_layout.addWidget(export_png)

        self.results_comparison_controls.setVisible(False)
        layout.addWidget(self.results_comparison_controls)
        layout.addStretch()
        return panel

    def _build_results_page(self):
        self.results_page = QWidget()
        layout = QVBoxLayout(self.results_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.results_tabs = tab_row(self, ["Results", "Comparisons"], self.on_results_tab_changed, compact=True)
        layout.addLayout(self.results_tabs["layout"])
        self.results_pages = QStackedWidget()
        self.results_model = PandasTableModel()
        self.result_details_model = PandasTableModel()
        self.result_confusion_model = PandasTableModel()
        self.result_ssl_iteration_model = PandasTableModel()

        results_view = QWidget()
        results_layout = QVBoxLayout(results_view)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(6)
        results_layout.addWidget(section_label("TRAINED MODELS"))
        self.results_table = table_view(self.results_model)
        self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.results_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.results_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.results_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        trained_models_visible_rows = 12
        trained_models_height = (
            self.results_table.horizontalHeader().sizeHint().height()
            + (self.results_table.verticalHeader().defaultSectionSize() * trained_models_visible_rows)
            + (self.results_table.frameWidth() * 2)
        )
        self.results_table.setMinimumHeight(trained_models_height)
        self.results_table.setMaximumHeight(trained_models_height)
        self.results_table.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.results_table.selectionModel().selectionChanged.connect(self.on_result_selection_changed)
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self.on_results_context_menu)
        results_layout.addWidget(self.results_table, 0)

        details_layout = QHBoxLayout()
        details_layout.setSpacing(16)

        self.result_training_panel = data_panel(
            "TRAINING INFO", self.result_details_model
        )
        self.result_training_panel.setFixedHeight(200)
        self.result_training_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        details_layout.addWidget(self.result_training_panel, 1)

        # The right-hand panel changes with the learning category
        self.result_secondary_panel = QWidget()
        self.result_secondary_panel.setFixedHeight(200)
        self.result_secondary_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        secondary_layout = QVBoxLayout(self.result_secondary_panel)
        secondary_layout.setContentsMargins(0, 0, 0, 0)
        secondary_layout.setSpacing(6)

        self.result_secondary_title = QLabel("CONFUSION MATRIX")
        self.result_secondary_title.setProperty("panelTitle", True)
        secondary_layout.addWidget(self.result_secondary_title)

        self.result_secondary_table = table_view(self.result_confusion_model)
        self.result_secondary_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.result_secondary_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.result_secondary_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.result_secondary_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)

        secondary_layout.addWidget(self.result_secondary_table)
        details_layout.addWidget(self.result_secondary_panel, 1)
        results_layout.addLayout(details_layout, 0)

        # Self-training iteration history is SSL-specific
        self.result_ssl_iteration_panel = data_panel(
            "SSL ITERATION PROGRESS", self.result_ssl_iteration_model
        )
        self.result_ssl_iteration_panel.setMaximumHeight(200)
        self.result_ssl_iteration_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.result_ssl_iteration_panel.setVisible(False)
        results_layout.addWidget(self.result_ssl_iteration_panel, 0)
        results_layout.addStretch(1)

        comparison_view = QWidget()
        comparison_layout = QVBoxLayout(comparison_view)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        comparison_layout.setSpacing(12)

        comparison_layout.addWidget(section_label("METRICS IMAGE"))
        self.comparison_metric_scroll = QScrollArea()
        self.comparison_metric_scroll.setWidgetResizable(True)
        self.comparison_metric_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.comparison_metric_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.comparison_metric_scroll.setMinimumHeight(280)
        self.comparison_metric_label = QLabel("Select model(s) to generate metrics comparison image.")
        self.comparison_metric_label.setAlignment(Qt.AlignCenter)
        self.comparison_metric_label.mousePressEvent = lambda event: self.inspect_comparison_image("metrics", event)
        self.comparison_metric_scroll.setWidget(self.comparison_metric_label)
        comparison_layout.addWidget(self.comparison_metric_scroll, 1)
        metric_controls = QHBoxLayout()
        metric_zoom_out = secondary_button("-")
        metric_zoom_out.clicked.connect(lambda: self._change_comparison_zoom("metrics", 0.85))
        metric_zoom_in = secondary_button("+")
        metric_zoom_in.clicked.connect(lambda: self._change_comparison_zoom("metrics", 1.15))
        metric_reset = secondary_button("Reset")
        metric_reset.clicked.connect(lambda: self._reset_comparison_zoom("metrics"))
        metric_controls.addWidget(metric_zoom_out)
        metric_controls.addWidget(metric_zoom_in)
        metric_controls.addWidget(metric_reset)
        metric_controls.addStretch()
        comparison_layout.addLayout(metric_controls)

        comparison_layout.addWidget(section_label("COMBINED CONFUSION MATRIX IMAGE"))
        self.comparison_cm_scroll = QScrollArea()
        self.comparison_cm_scroll.setWidgetResizable(True)
        self.comparison_cm_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.comparison_cm_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.comparison_cm_scroll.setMinimumHeight(280)
        self.comparison_cm_label = QLabel("Select model(s) with confusion matrices to generate combined image.")
        self.comparison_cm_label.setAlignment(Qt.AlignCenter)
        self.comparison_cm_label.mousePressEvent = lambda event: self.inspect_comparison_image("cm", event)
        self.comparison_cm_scroll.setWidget(self.comparison_cm_label)
        comparison_layout.addWidget(self.comparison_cm_scroll, 1)
        cm_controls = QHBoxLayout()
        cm_zoom_out = secondary_button("-")
        cm_zoom_out.clicked.connect(lambda: self._change_comparison_zoom("cm", 0.85))
        cm_zoom_in = secondary_button("+")
        cm_zoom_in.clicked.connect(lambda: self._change_comparison_zoom("cm", 1.15))
        cm_reset = secondary_button("Reset")
        cm_reset.clicked.connect(lambda: self._reset_comparison_zoom("cm"))
        cm_controls.addWidget(cm_zoom_out)
        cm_controls.addWidget(cm_zoom_in)
        cm_controls.addWidget(cm_reset)
        cm_controls.addStretch()
        comparison_layout.addLayout(cm_controls)

        comparison_layout.addWidget(section_label("SEMI-SUPERVISED MODEL-SPECIFIC IMAGE"))
        self.comparison_model_specific_ssl_scroll = QScrollArea()
        self.comparison_model_specific_ssl_scroll.setWidgetResizable(True)
        self.comparison_model_specific_ssl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.comparison_model_specific_ssl_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.comparison_model_specific_ssl_scroll.setMinimumHeight(260)
        self.comparison_model_specific_ssl_label = QLabel(
            "Select semi-supervised models to generate a semi-supervised comparison image."
        )
        self.comparison_model_specific_ssl_label.setAlignment(Qt.AlignCenter)
        self.comparison_model_specific_ssl_label.mousePressEvent = (
            lambda event: self.inspect_comparison_image("model_specific_ssl", event)
        )
        self.comparison_model_specific_ssl_scroll.setWidget(self.comparison_model_specific_ssl_label)
        comparison_layout.addWidget(self.comparison_model_specific_ssl_scroll, 1)
        ssl_controls = QHBoxLayout()
        ssl_zoom_out = secondary_button("-")
        ssl_zoom_out.clicked.connect(lambda: self._change_comparison_zoom("model_ssl", 0.85))
        ssl_zoom_in = secondary_button("+")
        ssl_zoom_in.clicked.connect(lambda: self._change_comparison_zoom("model_ssl", 1.15))
        ssl_reset = secondary_button("Reset")
        ssl_reset.clicked.connect(lambda: self._reset_comparison_zoom("model_ssl"))
        ssl_controls.addWidget(ssl_zoom_out)
        ssl_controls.addWidget(ssl_zoom_in)
        ssl_controls.addWidget(ssl_reset)
        ssl_controls.addStretch()
        comparison_layout.addLayout(ssl_controls)

        comparison_layout.addWidget(section_label("UNSUPERVISED MODEL-SPECIFIC IMAGE"))
        self.comparison_model_specific_unsup_scroll = QScrollArea()
        self.comparison_model_specific_unsup_scroll.setWidgetResizable(True)
        self.comparison_model_specific_unsup_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.comparison_model_specific_unsup_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.comparison_model_specific_unsup_scroll.setMinimumHeight(260)
        self.comparison_model_specific_unsup_label = QLabel(
            "Select unsupervised models to generate an unsupervised comparison image."
        )
        self.comparison_model_specific_unsup_label.setAlignment(Qt.AlignCenter)
        self.comparison_model_specific_unsup_label.mousePressEvent = (
            lambda event: self.inspect_comparison_image("model_specific_unsup", event)
        )
        self.comparison_model_specific_unsup_scroll.setWidget(self.comparison_model_specific_unsup_label)
        comparison_layout.addWidget(self.comparison_model_specific_unsup_scroll, 1)
        unsup_controls = QHBoxLayout()
        unsup_zoom_out = secondary_button("-")
        unsup_zoom_out.clicked.connect(lambda: self._change_comparison_zoom("model_unsup", 0.85))
        unsup_zoom_in = secondary_button("+")
        unsup_zoom_in.clicked.connect(lambda: self._change_comparison_zoom("model_unsup", 1.15))
        unsup_reset = secondary_button("Reset")
        unsup_reset.clicked.connect(lambda: self._reset_comparison_zoom("model_unsup"))
        unsup_controls.addWidget(unsup_zoom_out)
        unsup_controls.addWidget(unsup_zoom_in)
        unsup_controls.addWidget(unsup_reset)
        unsup_controls.addStretch()
        comparison_layout.addLayout(unsup_controls)
        self.results_pages.addWidget(results_view)
        self.results_pages.addWidget(comparison_view)
        self.results_pages.currentChanged.connect(self._sync_results_stack_to_current_page)
        self.results_page_scroll = QScrollArea()
        self.results_page_scroll.setWidgetResizable(True)
        self.results_page_scroll.setFrameShape(QFrame.NoFrame)
        self.results_page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.results_page_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.results_page_scroll.setWidget(self.results_pages)
        self._sync_results_stack_to_current_page()
        layout.addWidget(self.results_page_scroll, 1)
        self.main_stack.addWidget(self.results_page)

    def on_results_tab_changed(self, index):
        self.results_tabs["buttons"][index].setChecked(True)
        self.results_pages.setCurrentIndex(index)
        if hasattr(self, "results_comparison_controls"):
            self.results_comparison_controls.setVisible(index == 1)
        self._sync_results_stack_to_current_page()

    def _sync_results_stack_to_current_page(self, index=None):
        del index
        if not hasattr(self, "results_pages"):
            return
        current_page = self.results_pages.currentWidget()
        if current_page is None:
            return

        # Keep the scroll target sized to the active tab only so vertical
        # scrolling appears only when the visible Results content exceeds
        # the available viewport.
        current_page.adjustSize()
        page_hint = current_page.sizeHint()
        self.results_pages.setMinimumHeight(max(0, page_hint.height()))
        self.results_pages.updateGeometry()

    @staticmethod
    def _round_metric(value):
        if isinstance(value, (int, float)):
            return round(float(value), 4)
        return value

    def _normalize_evaluation_snapshot(self, result):
        snapshot = {}
        if not isinstance(result, dict):
            return snapshot

        # Preserve metrics and prediction context so saved projects and exported modelcan restore previously evaluated results
        metrics = result.get("metrics")
        if isinstance(metrics, dict):
            snapshot["metrics"] = make_json_safe(metrics)

        for source_key, snapshot_key in (
            ("predictions", "predictions"),
            ("y_test", "y_true"),
            ("y_true", "y_true"),
            ("y_score", "y_score"),
        ):
            if source_key in result and result.get(source_key) is not None:
                try:
                    snapshot[snapshot_key] = make_json_safe(result.get(source_key))
                except Exception:
                    pass

        matrix = result.get("confusion_matrix")
        if matrix is not None:
            try:
                matrix_df = pd.DataFrame(matrix)
                if not matrix_df.empty:
                    snapshot["confusion_matrix"] = matrix_df.fillna(0).astype(int).values.tolist()
                    if "y_test" in result:
                        labels = [str(label) for label in sorted(pd.unique(result["y_test"]))]
                    elif getattr(result.get("ssl_model"), "classes_", None) is not None:
                        labels = [str(label) for label in result["ssl_model"].classes_]
                    else:
                        labels = [str(index) for index in range(matrix_df.shape[0])]
                    snapshot["confusion_labels"] = labels
            except Exception:
                pass

        # Preserve the overall SSL counts and the each iteration history separately
        if "progress_df" in result:
            try:
                snapshot["ssl_progress"] = pd.DataFrame(result["progress_df"]).to_dict(orient="records")
            except Exception:
                pass

        if "iteration_progress" in result:
            try:
                snapshot["ssl_iteration_progress"] = pd.DataFrame(
                    result["iteration_progress"]
                ).to_dict(orient="records")
            except Exception:
                pass

        # Preserve the export-ready SSL dataset so it remains available after
        # project save/reload and inside the self-contained model PKL package.
        if "ssl_export_df" in result:
            try:
                snapshot["ssl_export_data"] = pd.DataFrame(
                    result["ssl_export_df"]
                ).to_dict(orient="records")
            except Exception:
                pass

        if "summary_df" in result:
            try:
                snapshot["cluster_summary"] = pd.DataFrame(
                    result["summary_df"]
                ).to_dict(orient="records")
            except Exception:
                pass

        # Preserve the original dataset with its fitted cluster assignment 
        if "clustered_df" in result:
            try:
                clustered_export = pd.DataFrame(result["clustered_df"]).copy()
                if "cluster" in clustered_export.columns:
                    clustered_export["cluster_description"] = clustered_export[
                        "cluster"
                    ].map(lambda value: "Noise" if value == -1 else "Cluster")
                snapshot["clustered_export_data"] = clustered_export.to_dict(
                    orient="records"
                )
            except Exception:
                pass

        # PCA coordinates are report data only
        if "pca_result" in result:
            try:
                pca_result = result.get("pca_result") or {}
                snapshot["cluster_plot_data"] = pd.DataFrame(
                    pca_result.get("pca_df", pd.DataFrame())
                ).to_dict(orient="records")
                snapshot["cluster_plot_components"] = int(
                    pca_result.get("actual_components", 2)
                )
            except Exception:
                pass

        return snapshot

    def _load_exported_results(self):
        exported = []
        exported_dir = Path("ExportedModels")
        if not exported_dir.exists():
            return exported

        for json_path in sorted(exported_dir.glob("*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue

            display_name = str(payload.get("display_name") or json_path.stem)
            exported.append({
                "name": display_name,
                "display_name": display_name,
                "algorithm": str(payload.get("algorithm", "")),
                "category": str(payload.get("category", "")),
                "metrics": payload.get("metrics", {}),
                "parameters": payload.get("parameters", {}),
                "feature_columns": payload.get("feature_columns", []),
                "evaluation": {
                    "confusion_matrix": payload.get("confusion_matrix"),
                    "confusion_labels": payload.get("confusion_labels"),
                    "cluster_summary": payload.get("cluster_summary"),
                    "ssl_progress": payload.get("ssl_progress"),
                    "ssl_iteration_progress": payload.get("ssl_iteration_progress"),
                    "ssl_export_data": payload.get("ssl_export_data"),
                },
                "source": f"exported:{json_path.name}",
                "trained": True,
                "model": None,
            })
        return exported

    def _collect_trained_result_records(self):
        records = []
        seen = set()

        if self.project:
            added_lookup = {
                item.get("name"): item
                for item in self.project.get("added_models", [])
            }
            for model in self.project.get("models", []):
                name = model.get("display_name", "")
                if not name:
                    continue
                added = added_lookup.get(name, {})
                key = ("project", name)
                if key in seen:
                    continue
                seen.add(key)
                records.append({
                    "name": name,
                    "display_name": name,
                    "label": added.get("label") or model.get("label") or self.project.get("label_column", ""),
                    "algorithm": model.get("algorithm", ""),
                    "category": model.get("category") or added.get("category", ""),
                    "metrics": (
                        model.get("metrics")
                        or (model.get("evaluation", {}) or {}).get("metrics")
                        or added.get("metrics")
                        or {}
                    ),
                    "parameters": model.get("parameters", {}),
                    "common_parameters": added.get("common_parameters", {}),
                    "required_parameters": added.get("required_parameters", {}),
                    "advanced_parameters": added.get("advanced_parameters", {}),
                    "feature_columns": model.get("feature_columns", []),
                    "evaluation": model.get("evaluation", {}),
                    "source": "project",
                    "trained": True,
                    "model": model.get("model"),
                })

        return records

    def _records_to_table_rows(self, records):
        """Create one shared results table while leaving invalid metrics blank."""
        rows = []
        metric_keys = []
        static_keys = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "silhouette_score",
            "davies_bouldin_score",
            "calinski_harabasz_score",
            "cluster_count",
            "noise_count",
            "note",
        }
        for record in records:
            metrics = record.get("metrics", {}) or {}
            for key, value in metrics.items():
                if key in static_keys:
                    continue
                if value is None:
                    continue
                if key not in metric_keys:
                    metric_keys.append(key)

        for record in records:
            metrics = record.get("metrics", {}) or {}
            evaluation = record.get("evaluation", {}) or {}
            ssl_counts = {
                str(item.get("status", "")): item.get("count")
                for item in evaluation.get("ssl_progress", []) or []
            }
            row = {
                "name": record.get("name", ""),
                "label": record.get("label") or "N/A",
                "source": record.get("source", "project"),
                "category": record.get("category", ""),
                "algorithm": record.get("algorithm", ""),
                "accuracy": self._round_metric(metrics.get("accuracy")),
                "precision": self._round_metric(metrics.get("precision")),
                "recall": self._round_metric(metrics.get("recall")),
                "f1": self._round_metric(metrics.get("f1")),
                "silhouette": self._round_metric(metrics.get("silhouette_score")),
                "davies_bouldin": self._round_metric(metrics.get("davies_bouldin_score")),
                "calinski_harabasz": self._round_metric(metrics.get("calinski_harabasz_score")),
                "clusters": metrics.get("cluster_count"),
                "noise": metrics.get("noise_count"),
                "originally_unlabeled": ssl_counts.get("Originally unlabeled"),
                "pseudo_labeled": ssl_counts.get("Pseudo-labeled"),
                "remaining_unlabeled": ssl_counts.get("Remaining unlabeled"),
            }
            for key in metric_keys:
                row[key] = self._round_metric(metrics.get(key))
            rows.append(row)
        return rows

    def refresh_results_page(self):
        previous_selection = set(self._comparison_selected_names)
        records = self._collect_trained_result_records() if self.project else []
        self._result_records = records

        rows = self._records_to_table_rows(records)
        table = pd.DataFrame(
            rows,
            columns=[
                "name", "label", "source", "category", "algorithm",
                "accuracy", "precision", "recall", "f1",
                "silhouette", "davies_bouldin", "calinski_harabasz",
                "clusters", "noise", "originally_unlabeled",
                "pseudo_labeled", "remaining_unlabeled",
            ] + [
                key
                for key in pd.DataFrame(rows).columns
                if key not in {
                    "name", "label", "source", "category", "algorithm",
                    "accuracy", "precision", "recall", "f1",
                    "silhouette", "davies_bouldin", "calinski_harabasz",
                    "clusters", "noise", "originally_unlabeled",
                    "pseudo_labeled", "remaining_unlabeled",
                }
            ],
        )
        self.results_model.set_data(table)
        self._comparison_records = list(records)
        self._comparison_selected_names = previous_selection
        self._refresh_comparison_model_list(records)
        if self._comparison_selected_names:
            selected_records = [
                record for record in self._comparison_records
                if record.get("name") in self._comparison_selected_names
            ]
            self._render_comparison_images(selected_records)
        else:
            self._clear_comparison_images()
        self.result_details_model.set_data(pd.DataFrame())
        self.result_confusion_model.set_data(pd.DataFrame())
        self.result_ssl_iteration_model.set_data(pd.DataFrame())
        self.result_ssl_iteration_panel.setVisible(False)
        self.result_secondary_title.setText("CONFUSION MATRIX")
        self._sync_results_stack_to_current_page()

    def on_result_selection_changed(self, selected, deselected):
        del selected, deselected
        indexes = self.results_table.selectionModel().selectedRows()
        if not indexes or not self._result_records:
            self.result_details_model.set_data(pd.DataFrame())
            self.result_confusion_model.set_data(pd.DataFrame())
            self.result_ssl_iteration_model.set_data(pd.DataFrame())
            self.result_ssl_iteration_panel.setVisible(False)
            self.result_secondary_title.setText("CONFUSION MATRIX")
            self._sync_results_stack_to_current_page()
            return
        name = self.results_model._data.iloc[indexes[0].row()]["name"]
        model = next((item for item in self._result_records if item.get("name") == name), {})
        category = model.get("category", "")
        details = [
            ("source", model.get("source", "")),
            ("label", model.get("label") or "N/A"),
            ("category", category),
            ("algorithm", model.get("algorithm", "")),
            ("features", ", ".join(model.get("feature_columns", []))),
        ]

        is_ssl = category == "semi_supervised"
        self.result_ssl_iteration_panel.setVisible(is_ssl)
        if not is_ssl:
            self.result_ssl_iteration_model.set_data(pd.DataFrame())
        common_parameters = model.get("common_parameters", {}) or {}
        relevant_common = {
            "supervised": {"test_size", "verbose", "random_state", "stratify"},
            "semi_supervised": {"test_size", "verbose", "random_state", "stratify"},
            "unsupervised": {"verbose", "random_state"},
        }.get(category, set(common_parameters))
        details.extend(
            (f"common: {key}", value)
            for key, value in common_parameters.items()
            if key in relevant_common
        )

        required_parameters = model.get("required_parameters", {}) or {}
        criterion = required_parameters.get("criterion") or (
            model.get("advanced_parameters", {}) or {}
        ).get("criterion")
        for key, value in required_parameters.items():
            if category == "semi_supervised":
                if key == "threshold" and criterion == "k_best":
                    continue
                if key == "k_best" and criterion != "k_best":
                    continue
            details.append((f"required: {key}", value))

        for key, value in (model.get("advanced_parameters", {}) or {}).items():
            if category == "semi_supervised":
                if key == "threshold" and criterion == "k_best":
                    continue
                if key == "k_best" and criterion != "k_best":
                    continue
            details.append((f"advanced: {key}", value))

        for key, value in (model.get("metrics", {}) or {}).items():
            if value is not None and key != "note":
                details.append((f"metric: {key}", self._round_metric(value)))

        evaluation = model.get("evaluation", {}) or {}
        if category == "semi_supervised":
            model_obj = model.get("model")
            if model_obj is not None:
                details.append((
                    "SSL: Pretrained state used",
                    bool(getattr(model_obj, "pretrained_state_used_", False)),
                ))

            for item in evaluation.get("ssl_progress", []) or []:
                count = item.get("count", "")
                percentage = item.get("percentage")
                value = (
                    f"{count} ({float(percentage):.1f}%)"
                    if percentage is not None
                    else count
                )
                details.append((f"SSL: {item.get('status', '')}", value))

            iteration_rows = []
            for item in evaluation.get("ssl_iteration_progress", []) or []:
                iteration_rows.append({
                    "iteration": item.get("iteration", ""),
                    "description": item.get(
                        "description", f"Iteration {item.get('iteration', '')}"
                    ),
                    "newly_pseudo_labeled": item.get(
                        "newly_pseudo_labeled", item.get("count", "")
                    ),
                    "cumulative_pseudo_labeled": item.get(
                        "cumulative_pseudo_labeled",
                        item.get("pseudo_labeled_total", ""),
                    ),
                    "remaining_unlabeled": item.get(
                        "remaining_unlabeled", ""
                    ),
                    "pseudo_labeled_percentage": item.get("percentage", ""),
                    "remaining_unlabeled_percentage": item.get(
                        "remaining_percentage", ""
                    ),
                })
            self.result_ssl_iteration_model.set_data(pd.DataFrame(
                iteration_rows,
                columns=[
                    "iteration",
                    "description",
                    "newly_pseudo_labeled",
                    "cumulative_pseudo_labeled",
                    "remaining_unlabeled",
                    "pseudo_labeled_percentage",
                    "remaining_unlabeled_percentage",
                ],
            ))

        self.result_details_model.set_data(
            pd.DataFrame(details, columns=["field", "value"])
        )

        if category == "unsupervised":
            # Reuse this panel for the cluster summary instead
            self.result_secondary_title.setText("CLUSTER SUMMARY")
            summary = pd.DataFrame(evaluation.get("cluster_summary", []) or [])
            if not summary.empty:
                preferred = [
                    column
                    for column in ["cluster", "description", "count"]
                    if column in summary.columns
                ]
                if preferred:
                    summary = summary[preferred]
            self.result_confusion_model.set_data(summary)
            self._sync_results_stack_to_current_page()
            return

        self.result_secondary_title.setText("CONFUSION MATRIX")
        matrix = evaluation.get("confusion_matrix")
        labels = evaluation.get("confusion_labels")
        if matrix:
            try:
                matrix_df = pd.DataFrame(matrix)
                if labels and len(labels) == matrix_df.shape[0] == matrix_df.shape[1]:
                    matrix_df.index = [str(label) for label in labels]
                    matrix_df.columns = [str(label) for label in labels]
                matrix_df.index.name = "actual"
                self.result_confusion_model.set_data(matrix_df)
            except Exception:
                self.result_confusion_model.set_data(pd.DataFrame())
        else:
            self.result_confusion_model.set_data(pd.DataFrame())

        self._sync_results_stack_to_current_page()

    def on_results_context_menu(self, position):
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        menu = QMenu(self)
        menu.addAction("Inspect Model", self.inspect_selected_result_model)
        menu.addAction("View Report Images", self.view_selected_result_report_images)
        menu.addSeparator()
        menu.addAction("Export Model Report", self.export_selected_model_reports)

        # Dataset export is learning category specific 
        record = self._selected_result_record()
        if record and record.get("category") == "semi_supervised":
            menu.addAction("Export SSL Dataset", self.export_selected_ssl_dataset)
        elif record and record.get("category") == "unsupervised":
            menu.addAction(
                "Export Clustered Dataset",
                self.export_selected_clustered_dataset,
            )

        menu.exec(self.results_table.viewport().mapToGlobal(position))

    def export_selected_ssl_dataset(self):
        """Export the selected SSL model's self-training dataset.
        Original labels remain unchanged, accepted pseudo-labels are inserted,
        and samples rejected by ``SelfTrainingClassifier`` stay unlabeled. The
        added source and iteration columns provide an audit trail.
        """
        record = self._selected_result_record()
        if not record or record.get("category") != "semi_supervised":
            QMessageBox.warning(
                self,
                "SSL Model Required",
                "Select a trained semi-supervised model before exporting.",
            )
            return

        evaluation = record.get("evaluation", {}) or {}
        export_data = evaluation.get("ssl_export_data")

        if not export_data:
            model = record.get("model")
            export_frame = getattr(model, "ssl_export_df_", None)
            if isinstance(export_frame, pd.DataFrame):
                export_data = export_frame.to_dict(orient="records")

        if not export_data:
            QMessageBox.warning(
                self,
                "SSL Dataset Unavailable",
                "This SSL result does not contain row-mapping data. Retrain the "
                "model with the current version, then export again.",
            )
            return

        export_df = pd.DataFrame(export_data)
        if export_df.empty:
            QMessageBox.warning(
                self,
                "SSL Dataset Unavailable",
                "The saved SSL export dataset is empty.",
            )
            return

        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(record.get("name", "ssl_model"))
        ).strip("_") or "ssl_model"

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export SSL Dataset",
            str(self.downloads_dir / f"{safe_name}_ssl_dataset.csv"),
            "CSV Files (*.csv);;Excel Files (*.xlsx)",
        )
        if not path:
            return

        output_path = Path(path)
        if "Excel" in selected_filter and output_path.suffix.lower() != ".xlsx":
            output_path = output_path.with_suffix(".xlsx")
        elif "CSV" in selected_filter and output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(".csv")

        try:
            if output_path.suffix.lower() == ".xlsx":
                export_df.to_excel(output_path, index=False)
            else:
                export_df.to_csv(output_path, index=False)
        except Exception as error:
            self.show_error("SSL Dataset Export Error", error)
            return

        QMessageBox.information(
            self,
            "SSL Dataset Exported",
            f"The SSL dataset was exported to:\n{output_path}",
        )

    def export_selected_clustered_dataset(self):
        """Export the selected clustering result with cluster assignments.
        The original dataset columns are preserved. A cluster column stores
        the fitted assignment and cluster_description identifies DBSCAN
        noise without changing scikit-learn's standard -1 label.
        """
        record = self._selected_result_record()
        if not record or record.get("category") != "unsupervised":
            QMessageBox.warning(
                self,
                "Unsupervised Model Required",
                "Select a trained unsupervised model before exporting.",
            )
            return

        evaluation = record.get("evaluation", {}) or {}
        export_data = evaluation.get("clustered_export_data")

        if not export_data:
            QMessageBox.warning(
                self,
                "Clustered Dataset Unavailable",
                "This clustering result does not contain the original rows and "
                "cluster assignments. Retrain the model with the current version, "
                "then export again.",
            )
            return

        export_df = pd.DataFrame(export_data)
        if export_df.empty or "cluster" not in export_df.columns:
            QMessageBox.warning(
                self,
                "Clustered Dataset Unavailable",
                "The saved clustered dataset is empty or has no cluster column.",
            )
            return

        if "cluster_description" not in export_df.columns:
            export_df["cluster_description"] = export_df["cluster"].map(
                lambda value: "Noise" if value == -1 else "Cluster"
            )

        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(record.get("name", "cluster_model"))
        ).strip("_") or "cluster_model"

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Clustered Dataset",
            str(self.downloads_dir / f"{safe_name}_clustered_dataset.csv"),
            "CSV Files (*.csv);;Excel Files (*.xlsx)",
        )
        if not path:
            return

        output_path = Path(path)
        if "Excel" in selected_filter and output_path.suffix.lower() != ".xlsx":
            output_path = output_path.with_suffix(".xlsx")
        elif "CSV" in selected_filter and output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(".csv")

        try:
            if output_path.suffix.lower() == ".xlsx":
                export_df.to_excel(output_path, index=False)
            else:
                export_df.to_csv(output_path, index=False)
        except Exception as error:
            self.show_error("Clustered Dataset Export Error", error)
            return

        QMessageBox.information(
            self,
            "Clustered Dataset Exported",
            f"The clustered dataset was exported to:\n{output_path}",
        )

    def _selected_result_record(self):
        rows = self.results_table.selectionModel().selectedRows()
        if not rows:
            return None
        name = self.results_model._data.iloc[rows[0].row()].get("name", "")
        return next((record for record in self._result_records if record.get("name") == name), None)

    def inspect_selected_result_model(self):
        record = self._selected_result_record()
        if not record:
            return

        evaluation = record.get("evaluation", {}) or {}
        details = [
            f"Name: {record.get('name', '')}",
            f"Source: {record.get('source', '')}",
            f"Category: {record.get('category', '')}",
            f"Algorithm: {record.get('algorithm', '')}",
            f"Features: {', '.join(record.get('feature_columns', [])) or 'None'}",
            "",
            "Metrics:",
        ]
        excluded_for_ssl = {
            "silhouette_score",
            "davies_bouldin_score",
            "calinski_harabasz_score",
            "cluster_count",
            "noise_count",
        }
        for key, value in record.get("metrics", {}).items():
            if value is None:
                continue
            if record.get("category") == "semi_supervised" and key in excluded_for_ssl:
                continue
            details.append(f"  {key}: {self._round_metric(value)}")

        if record.get("category") == "semi_supervised":
            status_lookup = {
                str(item.get("status", "")): item
                for item in evaluation.get("ssl_progress", []) or []
            }
            details.append("")
            details.append("Semi-Supervised Table Metrics:")
            for status in [
                "Originally unlabeled",
                "Pseudo-labeled",
                "Remaining unlabeled",
            ]:
                item = status_lookup.get(status)
                if not item:
                    continue
                count = item.get("count")
                percentage = item.get("percentage")
                details.append(
                    f"  {status}: count={count}, percentage={self._round_metric(percentage)}"
                )
        details.append("")
        details.append(f"Confusion Matrix Available: {'Yes' if evaluation.get('confusion_matrix') else 'No'}")
        details.append(f"Report PDF: {evaluation.get('report_pdf', 'Not generated')}")
        QMessageBox.information(self, "Model Inspect", "\n".join(details))

    def view_selected_result_report_images(self):
        record = self._selected_result_record()
        if not record:
            return

        evaluation = record.get("evaluation", {}) or {}
        images = [path for path in evaluation.get("report_images", []) if Path(path).exists()]
        if not images:
            _, _, generated = self._generate_preview_report_assets(record)
            images = [str(path) for path in generated]
        if not images:
            QMessageBox.information(self, "No Report Images", "No report images were found for the selected model.")
            return

        dialog = ReportImagesDialog(self, f"Report Images - {record.get('name', '')}", images)
        dialog.exec()

    def _generate_preview_report_assets(self, record):
        report_context = self._build_report_context(record)
        evaluation = dict(record.get("evaluation", {}) or {})

        if report_context.get("is_correlated") and not evaluation.get("confusion_matrix"):
            y_true = report_context.get("y_true")
            y_pred = report_context.get("y_pred")
            try:
                labels = sorted(pd.unique(y_true))
                matrix = pd.crosstab(
                    pd.Series(y_true, name="actual"),
                    pd.Series(y_pred, name="predicted"),
                    dropna=False,
                )
                matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)
                evaluation["confusion_matrix"] = matrix.values.tolist()
                evaluation["confusion_labels"] = [str(label) for label in labels]
            except Exception:
                pass

        model_record = {
            "name": record.get("name"),
            "display_name": record.get("display_name"),
            "algorithm": record.get("algorithm"),
            "category": record.get("category"),
            "metrics": record.get("metrics", {}),
            "feature_columns": record.get("feature_columns", []),
            "confusion_matrix": evaluation.get("confusion_matrix"),
            "confusion_labels": evaluation.get("confusion_labels"),
            "cluster_summary": evaluation.get("cluster_summary"),
            "cluster_plot_data": evaluation.get("cluster_plot_data"),
            "cluster_plot_components": evaluation.get(
                "cluster_plot_components"
            ),
            "ssl_progress": evaluation.get("ssl_progress"),
            "ssl_iteration_progress": evaluation.get("ssl_iteration_progress"),
            "ssl_export_data": evaluation.get("ssl_export_data"),
            "model": record.get("model"),
            "X": report_context.get("X"),
            "y": report_context.get("y"),
            "y_true": report_context.get("y_true"),
            "y_pred": report_context.get("y_pred"),
            "y_score": report_context.get("y_score"),
            "is_correlated": report_context.get("is_correlated", False),
        }
        preview_root = self._preview_report_root / datetime.now().strftime("%Y%m%d")
        return generate_model_report_assets(model_record, preview_root, include_pdf=False)

    def _build_report_context(self, record):
        context = {
            "is_correlated": False,
            "X": None,
            "y": None,
            "y_true": None,
            "y_pred": None,
            "y_score": None,
        }

        if not self.project or self.working_df.empty:
            return context

        model_obj = record.get("model")
        label_col = self.project.get("label_column")
        feature_columns = record.get("feature_columns", [])
        category = record.get("category", "supervised")
        if category == "unsupervised":
            evaluation = record.get("evaluation", {}) or {}
            clustered_rows = evaluation.get("clustered_export_data") or []
            if not clustered_rows:
                return context
            clustered_df = pd.DataFrame(clustered_rows)
            if clustered_df.empty or "cluster" not in clustered_df.columns:
                return context

            candidate_features = [
                str(column)
                for column in (feature_columns or [])
                if str(column) in clustered_df.columns
            ]
            if not candidate_features:
                candidate_features = [
                    str(column)
                    for column in clustered_df.columns
                    if str(column) not in {"cluster", "cluster_description"}
                    and pd.api.types.is_numeric_dtype(clustered_df[column])
                ]
            if not candidate_features:
                return context

            try:
                X = clustered_df[candidate_features].apply(pd.to_numeric, errors="coerce")
                y_pred = pd.to_numeric(clustered_df["cluster"], errors="coerce")
                valid = (~X.isna().any(axis=1)) & y_pred.notna()
                X = X.loc[valid]
                y_pred = y_pred.loc[valid]
                if X.empty:
                    return context
            except Exception:
                return context

            context.update({
                "is_correlated": False,
                "X": X.to_numpy(dtype=float),
                "y_pred": y_pred.to_numpy(dtype=float),
            })
            return context
        if model_obj is None or not label_col:
            return context
        if label_col not in self.working_df.columns:
            return context

        try:
            # Reports evaluate classifiers only on rows with known ground-truth labels
            # SSL unlabeled rows belong in progress reporting
            evaluation_df = self.working_df[self.working_df[label_col].notna()].copy()
            if evaluation_df.empty:
                return context

            if category == "semi_supervised" or hasattr(model_obj, "features"):
                expected = list(feature_columns) if feature_columns else [c for c in evaluation_df.columns if c != label_col]
                X = evaluation_df[expected].copy()
                y_values = evaluation_df[label_col].copy()
            else:
                X_raw, y_values = prepare_training_data(
                    df=evaluation_df,
                    label_col=label_col,
                    features=feature_columns if feature_columns else None,
                )
                expected = list(feature_columns) if feature_columns else list(X_raw.columns)
                X = align_features(X_raw, expected, fill_value=0)
        except Exception:
            return context

        try:
            y_pred = model_obj.predict(X)
        except Exception:
            return context

        y_score = None
        try:
            if hasattr(model_obj, "predict_proba"):
                y_score = model_obj.predict_proba(X)
            elif hasattr(model_obj, "decision_function"):
                y_score = model_obj.decision_function(X)
        except Exception:
            y_score = None

        context.update({
            "is_correlated": True,
            "X": X.values,
            "y": np.array(y_values),
            "y_true": np.array(y_values),
            "y_pred": np.array(y_pred),
            "y_score": y_score,
        })
        return context

    def _refresh_comparison_model_list(self, records):
        available_names = {str(record.get("name", "")) for record in records}
        self._comparison_selected_names = {
            name for name in self._comparison_selected_names if name in available_names
        }
        self.comparison_model_list.blockSignals(True)
        self.comparison_model_list.clear()
        for record in records:
            name = record.get("name", "")
            checkbox = QCheckBox(name)
            checkbox.setChecked(name in self._comparison_selected_names)
            checkbox.toggled.connect(self.on_comparison_list_changed)
            item = QListWidgetItem()
            item.setSizeHint(checkbox.sizeHint())
            self.comparison_model_list.addItem(item)
            self.comparison_model_list.setItemWidget(item, checkbox)
        self.comparison_model_list.blockSignals(False)

    def select_all_comparison_models(self):
        for index in range(self.comparison_model_list.count()):
            checkbox = self.comparison_model_list.itemWidget(self.comparison_model_list.item(index))
            if checkbox is not None:
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)
        self.on_comparison_list_changed(None)

    def clear_all_comparison_models(self):
        for index in range(self.comparison_model_list.count()):
            checkbox = self.comparison_model_list.itemWidget(self.comparison_model_list.item(index))
            if checkbox is not None:
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(False)
        self.on_comparison_list_changed(None)

    def on_comparison_list_changed(self, item):
        del item
        selected_names = []
        for index in range(self.comparison_model_list.count()):
            list_item = self.comparison_model_list.item(index)
            checkbox = self.comparison_model_list.itemWidget(list_item)
            if checkbox is not None and checkbox.isChecked():
                selected_names.append(checkbox.text())

        self._comparison_selected_names = set(selected_names)
        selected_records = [record for record in self._comparison_records if record.get("name") in selected_names]
        self._render_comparison_images(selected_records)

    def _clear_comparison_images(self):
        self._comparison_metric_image_path = None
        self._comparison_cm_image_path = None
        self._comparison_model_specific_ssl_image_path = None
        self._comparison_model_specific_unsup_image_path = None
        self.comparison_metric_label.clear()
        self.comparison_metric_label.setText("Select model(s) to generate metrics comparison image.")
        self.comparison_cm_label.clear()
        self.comparison_cm_label.setText("Select model(s) with confusion matrices to generate combined image.")
        self.comparison_model_specific_ssl_label.clear()
        self.comparison_model_specific_ssl_label.setText(
            "Select semi-supervised models to generate a semi-supervised comparison image."
        )
        self.comparison_model_specific_unsup_label.clear()
        self.comparison_model_specific_unsup_label.setText(
            "Select unsupervised models to generate an unsupervised comparison image."
        )
        for key in self._comparison_zoom_levels:
            self._comparison_zoom_levels[key] = 1.0
            self._comparison_base_pixmaps[key] = None

    def _comparison_label_for_key(self, image_key):
        return {
            "metrics": self.comparison_metric_label,
            "cm": self.comparison_cm_label,
            "model_ssl": self.comparison_model_specific_ssl_label,
            "model_unsup": self.comparison_model_specific_unsup_label,
        }.get(image_key)

    def _change_comparison_zoom(self, image_key, factor):
        label = self._comparison_label_for_key(image_key)
        base_pixmap = self._comparison_base_pixmaps.get(image_key)
        if label is None or base_pixmap is None or base_pixmap.isNull():
            return
        self._comparison_zoom_levels[image_key] = max(
            0.1,
            min(8.0, self._comparison_zoom_levels.get(image_key, 1.0) * factor),
        )
        self._apply_comparison_zoom(image_key)

    def _reset_comparison_zoom(self, image_key):
        self._comparison_zoom_levels[image_key] = 1.0
        self._apply_comparison_zoom(image_key)

    def _apply_comparison_zoom(self, image_key):
        label = self._comparison_label_for_key(image_key)
        base_pixmap = self._comparison_base_pixmaps.get(image_key)
        if label is None or base_pixmap is None or base_pixmap.isNull():
            return
        zoom = float(self._comparison_zoom_levels.get(image_key, 1.0))
        width = max(1, int(base_pixmap.width() * zoom))
        height = max(1, int(base_pixmap.height() * zoom))
        scaled = base_pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        label.adjustSize()

    def _set_comparison_image(self, image_key, image_path, empty_text):
        label = self._comparison_label_for_key(image_key)
        if label is None:
            return
        if image_path is None:
            self._comparison_base_pixmaps[image_key] = None
            label.setText(empty_text)
            label.adjustSize()
            return

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self._comparison_base_pixmaps[image_key] = None
            label.setText("Failed to render image.")
            label.adjustSize()
            return

        self._comparison_base_pixmaps[image_key] = pixmap
        self._comparison_zoom_levels[image_key] = 1.0
        self._apply_comparison_zoom(image_key)

    def inspect_comparison_image(self, image_type, event):
        del event
        if image_type == "metrics":
            path = self._comparison_metric_image_path
            title = "Inspect - Comparison Metrics"
        elif image_type == "model_specific_ssl":
            path = self._comparison_model_specific_ssl_image_path
            title = "Inspect - Semi-Supervised Comparison"
        elif image_type == "model_specific_unsup":
            path = self._comparison_model_specific_unsup_image_path
            title = "Inspect - Unsupervised Comparison"
        else:
            path = self._comparison_cm_image_path
            title = "Inspect - Combined Confusion Matrix"

        if not path or not Path(path).exists():
            return
        dialog = ImageInspectDialog(self, title, path)
        dialog.exec()

    def _render_comparison_images(self, selected_records):
        self._comparison_metric_image_path = None
        self._comparison_cm_image_path = None
        self._comparison_model_specific_ssl_image_path = None
        self._comparison_model_specific_unsup_image_path = None
        if not selected_records:
            self._clear_comparison_images()
            return

        output_dir = self._preview_report_root / "comparison" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir.mkdir(parents=True, exist_ok=True)

        metric_path = output_dir / "comparison_metrics.png"
        if render_comparison_metrics_image(selected_records, metric_path):
            self._comparison_metric_image_path = metric_path
            self._set_comparison_image("metrics", metric_path, "No metrics available for selected models.")
        else:
            self._set_comparison_image("metrics", None, "No metrics available for selected models.")

        cm_path = output_dir / "comparison_combined_cm.png"
        if render_combined_confusion_matrices_image(selected_records, cm_path):
            self._comparison_cm_image_path = cm_path
            self._set_comparison_image("cm", cm_path, "No confusion matrices available for selected models.")
        else:
            self._set_comparison_image("cm", None, "No confusion matrices available for selected models.")

        ssl_records = [record for record in selected_records if record.get("category") == "semi_supervised"]
        ssl_model_specific_path = output_dir / "comparison_model_specific_ssl.png"
        if ssl_records and render_model_specific_comparison_image(ssl_records, ssl_model_specific_path):
            self._comparison_model_specific_ssl_image_path = ssl_model_specific_path
            self._set_comparison_image(
                "model_ssl",
                ssl_model_specific_path,
                "No comparable semi-supervised values available for selected models.",
            )
        else:
            self._set_comparison_image(
                "model_ssl",
                None,
                "No comparable semi-supervised values available for selected models.",
            )

        unsup_records = [record for record in selected_records if record.get("category") == "unsupervised"]
        unsup_model_specific_path = output_dir / "comparison_model_specific_unsup.png"
        if unsup_records and render_model_specific_comparison_image(unsup_records, unsup_model_specific_path):
            self._comparison_model_specific_unsup_image_path = unsup_model_specific_path
            self._set_comparison_image(
                "model_unsup",
                unsup_model_specific_path,
                "No comparable unsupervised values available for selected models.",
            )
        else:
            self._set_comparison_image(
                "model_unsup",
                None,
                "No comparable unsupervised values available for selected models.",
            )

    def _prompt_png_name(self, title, default_name):
        name, accepted = QInputDialog.getText(self, title, "PNG filename (without extension)", text=default_name)
        if not accepted or not name.strip():
            return None
        return f"{name.strip()}.png"

    def _export_generated_image(self, source_path, filename):
        if not source_path or not Path(source_path).exists():
            return False
        output_dir = QFileDialog.getExistingDirectory(self, "Choose Export Folder", str(self.downloads_dir))
        if not output_dir:
            return False
        target_path = Path(output_dir) / filename
        return QPixmap(str(source_path)).save(str(target_path), "PNG")

    def export_comparison_images(self):
        mode = self.comparison_export_mode.currentText()
        if mode == "Metric Image":
            if not self._comparison_metric_image_path:
                QMessageBox.information(self, "No Metrics Image", "Select model(s) to generate a metrics image first.")
                return
            filename = self._prompt_png_name("Export Metrics PNG", "comparison_metrics")
            if not filename:
                return
            if not self._export_generated_image(self._comparison_metric_image_path, filename):
                QMessageBox.warning(self, "Export Failed", "Unable to export metrics image.")
            return

        if mode == "Confusion Matrix Image":
            if not self._comparison_cm_image_path:
                QMessageBox.information(self, "No Confusion Image", "Select model(s) with confusion matrix data first.")
                return
            filename = self._prompt_png_name("Export Confusion Matrix PNG", "comparison_combined_cm")
            if not filename:
                return
            if not self._export_generated_image(self._comparison_cm_image_path, filename):
                QMessageBox.warning(self, "Export Failed", "Unable to export confusion matrix image.")
            return

        if mode == "Semi-Supervised Image":
            if not self._comparison_model_specific_ssl_image_path:
                QMessageBox.information(
                    self,
                    "No Semi-Supervised Image",
                    "Select semi-supervised models to generate this image first.",
                )
                return
            filename = self._prompt_png_name("Export Semi-Supervised PNG", "comparison_model_specific_ssl")
            if not filename:
                return
            if not self._export_generated_image(self._comparison_model_specific_ssl_image_path, filename):
                QMessageBox.warning(self, "Export Failed", "Unable to export semi-supervised image.")
            return

        if mode == "Unsupervised Image":
            if not self._comparison_model_specific_unsup_image_path:
                QMessageBox.information(
                    self,
                    "No Unsupervised Image",
                    "Select unsupervised models to generate this image first.",
                )
                return
            filename = self._prompt_png_name("Export Unsupervised PNG", "comparison_model_specific_unsup")
            if not filename:
                return
            if not self._export_generated_image(self._comparison_model_specific_unsup_image_path, filename):
                QMessageBox.warning(self, "Export Failed", "Unable to export unsupervised image.")
            return

        # Export All 4 Images
        if (
            not self._comparison_metric_image_path
            or not self._comparison_cm_image_path
            or not self._comparison_model_specific_ssl_image_path
            or not self._comparison_model_specific_unsup_image_path
        ):
            QMessageBox.information(
                self,
                "Missing Images",
                "Generate metric, confusion matrix, semi-supervised, and unsupervised images before exporting all 4.",
            )
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Choose Export Folder", str(self.downloads_dir))
        if not output_dir:
            return
        targets = [
            (self._comparison_metric_image_path, "comparison_metrics.png"),
            (self._comparison_cm_image_path, "comparison_combined_cm.png"),
            (self._comparison_model_specific_ssl_image_path, "comparison_model_specific_ssl.png"),
            (self._comparison_model_specific_unsup_image_path, "comparison_model_specific_unsup.png"),
        ]
        failures = []
        for source_path, filename in targets:
            if not source_path:
                continue
            saved = QPixmap(str(source_path)).save(str(Path(output_dir) / filename), "PNG")
            if not saved:
                failures.append(filename)

        if failures:
            QMessageBox.warning(
                self,
                "Export Failed",
                "One or more images could not be exported:\n" + "\n".join(failures),
            )
            return
        QMessageBox.information(self, "Export Complete", f"Saved comparison images to {output_dir}")

    def export_selected_model_reports(self):
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Select at least one trained model from Results.")
            return

        output_root = QFileDialog.getExistingDirectory(
            self,
            "Choose Export Folder",
            str(self.downloads_dir),
        )
        if not output_root:
            return

        batch_folder = Path(output_root) / f"model_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_folder.mkdir(parents=True, exist_ok=True)

        exported = 0
        skipped = []
        for index in selected_rows:
            row = self.results_model._data.iloc[index.row()]
            name = row.get("name", "")
            record = next((item for item in self._result_records if item.get("name") == name), None)
            if record is None:
                skipped.append(name)
                continue

            evaluation = record.get("evaluation", {}) or {}
            report_context = self._build_report_context(record)

            if report_context.get("is_correlated") and not evaluation.get("confusion_matrix"):
                y_true = report_context.get("y_true")
                y_pred = report_context.get("y_pred")
                try:
                    true_series = pd.Series(y_true).dropna().astype(str).reset_index(drop=True)
                    pred_series = pd.Series(y_pred).iloc[:len(true_series)].astype(str).reset_index(drop=True)
                    labels = sorted(set(true_series.tolist()) | set(pred_series.tolist()))
                    matrix = pd.crosstab(
                        true_series.rename("actual"),
                        pred_series.rename("predicted"),
                        dropna=False,
                    )
                    matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)
                    evaluation["confusion_matrix"] = matrix.values.tolist()
                    evaluation["confusion_labels"] = labels
                except Exception:
                    pass

            model_record = {
                "name": record.get("name"),
                "display_name": record.get("display_name"),
                "algorithm": record.get("algorithm"),
                "category": record.get("category"),
                "metrics": record.get("metrics", {}),
                "feature_columns": record.get("feature_columns", []),
                "confusion_matrix": evaluation.get("confusion_matrix"),
                "confusion_labels": evaluation.get("confusion_labels"),
                "cluster_summary": evaluation.get("cluster_summary"),
                "cluster_plot_data": evaluation.get("cluster_plot_data"),
                "cluster_plot_components": evaluation.get(
                    "cluster_plot_components"
                ),
                "ssl_progress": evaluation.get("ssl_progress"),
                "ssl_iteration_progress": evaluation.get("ssl_iteration_progress"),
                "ssl_export_data": evaluation.get("ssl_export_data"),
                "model": record.get("model"),
                "X": report_context.get("X"),
                "y": report_context.get("y"),
                "y_true": report_context.get("y_true"),
                "y_pred": report_context.get("y_pred"),
                "y_score": report_context.get("y_score"),
                "is_correlated": report_context.get("is_correlated", False),
            }
            try:
                _, pdf_path, images = export_model_report(model_record, batch_folder)
            except Exception as error:
                skipped.append(f"{name} ({error})")
                continue

            if pdf_path or images:
                exported += 1
                if self.project:
                    for model in self.project.get("models", []):
                        if model.get("display_name") == record.get("name"):
                            model.setdefault("evaluation", {})["report_pdf"] = str(pdf_path) if pdf_path else ""
                            model.setdefault("evaluation", {})["report_images"] = [str(path) for path in images]
                            break
            else:
                skipped.append(f"{name} (no plottable results)")

        if exported == 0:
            QMessageBox.warning(
                self,
                "Export Incomplete",
                "No report files were generated.\n" + ("\n".join(skipped) if skipped else ""),
            )
            return

        message = f"Generated {exported} model report folder(s) in:\n{batch_folder}"
        if skipped:
            message += "\n\nSkipped:\n" + "\n".join(skipped)
        if self.project and exported:
            self._set_dirty(True)
        QMessageBox.information(self, "Report Export Complete", message)

    def export_results_comparison(self):
        selected_names = []
        for index in range(self.comparison_model_list.count()):
            item = self.comparison_model_list.item(index)
            if item.checkState() == Qt.Checked:
                selected_names.append(item.text())

        selected_records = [record for record in self._comparison_records if record.get("name") in selected_names]
        if not selected_records:
            QMessageBox.warning(self, "No Results", "There are no saved model results to export.")
            return

        frame = pd.DataFrame(
            self._records_to_table_rows(selected_records),
            columns=["name", "label", "source", "category", "algorithm", "accuracy", "precision", "recall", "f1"],
        )
        path, _ = QFileDialog.getSaveFileName(self, "Export Model Comparison", str(self.downloads_dir / "model_results.csv"), "CSV Files (*.csv)")
        if not path:
            return
        try:
            frame.to_csv(path, index=False)
        except Exception as error:
            self.show_error("Export Error", error)
            return
        QMessageBox.information(self, "Export Complete", f"Saved {path}")
