import numpy as np
import pandas as pd

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.frontend.styles import DARK_GREEN, GREEN, MUTED
from src.frontend.table_model import PandasTableModel
from src.frontend.widgets import ColumnPicker, divider, primary_button, section_label, sidebar_base, table_view, taller_dropdown

# Matplotlib plot canvas for one signal
class PlotCanvas(FigureCanvas):
    def __init__(self, title="Signal"):
        self.figure = Figure(figsize=(4.1, 2.25), facecolor="white")
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.update_plot([], [], title=title)

    def update_plot(self, x_values, y_values, current_x=None, current_y=None, title="Signal"):
        self.ax.clear()
        self.ax.set_facecolor("white")

        if len(x_values) > 0 and len(y_values) > 0:
            self.ax.plot(x_values, y_values, color=DARK_GREEN, linewidth=1.4)

            if current_x is not None and current_y is not None:
                self.ax.scatter([current_x], [current_y], color=GREEN, s=42, zorder=3)

        self.ax.set_title(title, fontsize=9, color=DARK_GREEN, fontweight="bold")
        self.ax.grid(True, alpha=0.25)
        self.ax.tick_params(labelsize=8)
        self.figure.tight_layout()
        self.draw()

# Main view page
class PlaybackAnnotationPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = pd.DataFrame()
        self.current_index = 0
        self.preview_model = PandasTableModel()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_next_row)

        # Assigned by SimplePlaybackSidebar
        self.sidebar = None
        self.timestamp_combo = None
        self.signal_picker = None
        self.window_spin = None
        self.delay_spin = None

        self._build_ui()
        self._set_page_controls_enabled(False)

    # ============================================================
    # UI
    # ============================================================
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        # title = QLabel("Recorded Playback")
        # title.setStyleSheet(f"font-size: 22px; font-weight: 700; color: {DARK_GREEN};")

        # layout.addWidget(title)

        self.dataset_status = QLabel("Dataset: not loaded")
        self.dataset_status.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.dataset_status)

        layout.addWidget(divider())

        # Playback controls
        layout.addWidget(section_label("PLAYBACK / NAVIGATION"))

        nav = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.previous_button = QPushButton("Previous")
        self.play_button = primary_button("▶ Play")
        self.pause_button = QPushButton("Pause")
        self.next_button = QPushButton("Next")

        self.start_button.clicked.connect(self.go_start)
        self.previous_button.clicked.connect(self.go_previous)
        self.play_button.clicked.connect(self.start_playback)
        self.pause_button.clicked.connect(self.pause_playback)
        self.next_button.clicked.connect(self.go_next)

        for button in [
            self.start_button,
            self.previous_button,
            self.play_button,
            self.pause_button,
            self.next_button,
        ]:
            nav.addWidget(button)

        layout.addLayout(nav)

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.valueChanged.connect(self.timeline_changed)
        layout.addWidget(self.timeline_slider)

        # Metrics
        metrics = QGridLayout()
        metrics.setHorizontalSpacing(14)

        self.index_metric = self._metric("Current Index", "N/A")
        self.timestamp_metric = self._metric("Timestamp", "N/A")
        self.playback_metric = self._metric("Playback", "Paused")

        metrics.addWidget(self.index_metric, 0, 0)
        metrics.addWidget(self.timestamp_metric, 0, 1)
        metrics.addWidget(self.playback_metric, 0, 2)

        layout.addLayout(metrics)

        # Plots
        layout.addWidget(section_label("VISUALIZATIONS"))

        plots = QGridLayout()
        plots.setHorizontalSpacing(12)
        plots.setVerticalSpacing(12)

        self.plots = []
        for i in range(6):
            canvas = PlotCanvas(f"Signal {i + 1}")
            self.plots.append(canvas)
            plots.addWidget(canvas, i // 3, i % 3)

        layout.addLayout(plots, 2)

        # Preview
        layout.addWidget(section_label("DATASET PREVIEW"))
        self.preview_table = table_view(self.preview_model)
        layout.addWidget(self.preview_table, 1)

    def _metric(self, title, value):
        label = QLabel(f"{title}\n{value}")
        label.setProperty("metric", True)
        label.setMinimumHeight(58)
        label.setWordWrap(True)
        return label

    def _set_metric(self, label, title, value):
        label.setText(f"{title}\n{value}")

    def _set_page_controls_enabled(self, enabled):
        for widget in [
            self.start_button,
            self.previous_button,
            self.play_button,
            self.pause_button,
            self.next_button,
            self.timeline_slider,
        ]:
            widget.setEnabled(enabled)

    def set_sidebar(self, sidebar):
        self.sidebar = sidebar
        self.timestamp_combo = sidebar.timestamp_combo
        self.signal_picker = sidebar.signal_picker
        self.window_spin = sidebar.window_spin
        self.delay_spin = sidebar.delay_spin

    # ============================================================
    # Loading
    # ============================================================
    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Dataset",
            "",
            "Supported Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)",
        )

        if not path:
            return

        try:
            if path.lower().endswith(".csv"):
                self.data = pd.read_csv(path)
            else:
                self.data = pd.read_excel(path)
        except Exception as error:
            QMessageBox.critical(self, "Dataset Error", f"Could not load dataset:\n{error}")
            return

        if self.data.empty:
            QMessageBox.warning(self, "Dataset Error", "The selected dataset is empty.")
            return

        self.current_index = 0

        self.dataset_status.setText(
            f"Dataset: {path} | Rows: {len(self.data)} | Columns: {len(self.data.columns)}"
        )

        if self.sidebar:
            self.sidebar.populate_controls()
            self.sidebar.set_controls_enabled(True)

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setMaximum(len(self.data) - 1)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)

        self._set_page_controls_enabled(True)
        self.update_display()

    # ============================================================
    # Helpers
    # ============================================================
    def numeric_columns(self):
        if self.data.empty:
            return []
        return self.data.select_dtypes(include=[np.number]).columns.astype(str).tolist()

    def selected_signal_columns(self):
        if self.signal_picker is None:
            return []

        selected = self.signal_picker.selected_items()
        numeric = self.numeric_columns()

        for col in numeric:
            if len(selected) >= 6:
                break
            if col not in selected:
                selected.append(col)

        return selected[:6]

    # ============================================================
    # Playback
    # ============================================================
    def go_start(self):
        if self.data.empty:
            return

        self.pause_playback()
        self.current_index = 0
        self._sync_timeline()
        self.update_display()

    def go_previous(self):
        if self.data.empty:
            return

        self.pause_playback()
        self.current_index = max(0, self.current_index - 1)
        self._sync_timeline()
        self.update_display()

    def go_next(self):
        if self.data.empty:
            return

        self.pause_playback()
        self.current_index = min(len(self.data) - 1, self.current_index + 1)
        self._sync_timeline()
        self.update_display()

    def start_playback(self):
        if self.data.empty or self.delay_spin is None:
            return

        self._set_metric(self.playback_metric, "Playback", "Playing")
        self.timer.start(self.delay_spin.value())

    def pause_playback(self):
        self.timer.stop()
        self._set_metric(self.playback_metric, "Playback", "Paused")

    def play_next_row(self):
        if self.data.empty:
            return

        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self._sync_timeline()
            self.update_display()
        else:
            self.pause_playback()

    def timeline_changed(self, value):
        if self.data.empty:
            return

        self.current_index = value
        self.update_display()

    def _sync_timeline(self):
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(self.current_index)
        self.timeline_slider.blockSignals(False)

    # ============================================================
    # Display
    # ============================================================
    def update_display(self):
        if self.data.empty or self.timestamp_combo is None:
            return

        self.current_index = max(0, min(self.current_index, len(self.data) - 1))
        row = self.data.iloc[self.current_index]

        timestamp_col = self.timestamp_combo.currentText()
        timestamp_value = row[timestamp_col] if timestamp_col != "None" else self.current_index

        self._set_metric(self.index_metric, "Current Index", self.current_index)
        self._set_metric(self.timestamp_metric, "Timestamp", timestamp_value)

        self.update_plots()
        self.update_preview()

    def update_plots(self):
        if self.data.empty or self.window_spin is None:
            return

        signals = self.selected_signal_columns()

        window_size = self.window_spin.value()
        start = max(0, self.current_index - window_size // 2)
        end = min(len(self.data), self.current_index + window_size // 2)
        window_df = self.data.iloc[start:end]

        timestamp_col = self.timestamp_combo.currentText()

        if timestamp_col != "None":
            x_values = window_df[timestamp_col].values
            current_x = self.data.iloc[self.current_index][timestamp_col]
        else:
            x_values = window_df.index.values
            current_x = self.current_index

        row = self.data.iloc[self.current_index]

        for i, canvas in enumerate(self.plots):
            if i < len(signals):
                signal = signals[i]
                canvas.update_plot(
                    x_values=x_values,
                    y_values=window_df[signal].values,
                    current_x=current_x,
                    current_y=row[signal],
                    title=f"{i + 1}. {signal}",
                )
            else:
                canvas.update_plot([], [], title=f"Signal {i + 1}")

    def update_preview(self):
        if self.data.empty:
            return

        start = max(0, self.current_index - 5)
        end = min(len(self.data), self.current_index + 6)
        self.preview_model.set_data(self.data.iloc[start:end])
        self.preview_table.resizeColumnsToContents()


class PlaybackSidebar(QWidget):
    """Left sidebar controls for the simplified playback tab."""

    def __init__(self, page: PlaybackAnnotationPage, parent=None):
        super().__init__(parent)

        self.page = page

        self._build_ui()
        self.page.set_sidebar(self)
        self.set_controls_enabled(False)

    def _build_ui(self):
        panel = sidebar_base()
        panel_layout = panel.layout()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(panel)

        panel_layout.addWidget(section_label("UPLOAD"))

        self.dataset_button = primary_button("Open Dataset")
        self.dataset_button.clicked.connect(self.page.load_dataset)
        panel_layout.addWidget(QLabel("Dataset"))
        panel_layout.addWidget(self.dataset_button)

        panel_layout.addWidget(divider())

        panel_layout.addWidget(section_label("DATASET SETTINGS"))

        self.timestamp_combo = taller_dropdown(QComboBox())
        self.timestamp_combo.currentTextChanged.connect(self.page.update_display)
        panel_layout.addWidget(QLabel("Timestamp Column"))
        panel_layout.addWidget(self.timestamp_combo)

        panel_layout.addWidget(divider())

        panel_layout.addWidget(section_label("SIGNALS TO MONITOR"))

        self.signal_picker = ColumnPicker("Search signals")
        self.signal_picker.setMinimumHeight(190)
        panel_layout.addWidget(self.signal_picker)

        panel_layout.addWidget(divider())

        panel_layout.addWidget(section_label("PLAYBACK SETTINGS"))

        self.window_spin = QSpinBox()
        self.window_spin.setRange(10, 1000)
        self.window_spin.setValue(100)
        self.window_spin.valueChanged.connect(self.page.update_display)
        panel_layout.addWidget(QLabel("Plot Window Size"))
        panel_layout.addWidget(self.window_spin)

        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(50, 2000)
        self.delay_spin.setValue(300)
        self.delay_spin.setSuffix(" ms")
        panel_layout.addWidget(QLabel("Playback Delay"))
        panel_layout.addWidget(self.delay_spin)

        panel_layout.addStretch()

    def set_controls_enabled(self, enabled):
        for widget in [
            self.timestamp_combo,
            self.signal_picker,
            self.window_spin,
            self.delay_spin,
        ]:
            widget.setEnabled(enabled)

        self.dataset_button.setEnabled(True)

    def populate_controls(self):
        data = self.page.data

        if data.empty:
            return

        all_columns = [str(col) for col in data.columns]
        numeric_columns = self.page.numeric_columns()

        self.timestamp_combo.blockSignals(True)
        self.timestamp_combo.clear()
        self.timestamp_combo.addItem("None")
        self.timestamp_combo.addItems(all_columns)
        self.timestamp_combo.blockSignals(False)

        self.signal_picker.set_items(numeric_columns, checked=False)

        default_numeric = numeric_columns[: min(6, len(numeric_columns))]
        self.signal_picker.set_selected(default_numeric)

        self._select_combo_if_exists(
            self.timestamp_combo,
            ["timestamp", "time", "index"],
        )

        for checkbox in self.signal_picker.checkboxes.values():
            checkbox.stateChanged.connect(self.page.update_display)

    def _select_combo_if_exists(self, combo, candidates):
        names = [combo.itemText(i).lower() for i in range(combo.count())]

        for candidate in candidates:
            if candidate.lower() in names:
                combo.setCurrentIndex(names.index(candidate.lower()))
                return
