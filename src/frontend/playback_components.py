from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtMultimediaWidgets import QVideoWidget
from src.frontend.charts import ChartCanvas
from src.frontend.widgets import data_panel, primary_button, secondary_button, section_label
from src.playback_annotation.video_controller import VIDEO_AVAILABLE


def status_card(title: str, value: str) -> dict[str, QWidget | QLabel]:
    widget = QWidget()
    widget.setProperty("card", True)
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(12, 8, 12, 8)

    label = QLabel(title)
    label.setProperty("panelTitle", True)
    value_label = QLabel(value)
    value_label.setWordWrap(True)
    layout.addWidget(label)
    layout.addWidget(value_label)
    return {"widget": widget, "value": value_label}

class PlaybackStatusPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)

        self.values: dict[str, QLabel] = {}
        for index, key in enumerate(
            ["Current Index", "Timestamp", "Prediction", "Confidence", "Playback"]
        ):
            card = status_card(key, "—")
            self.values[key] = card["value"]  # type: ignore[assignment]
            layout.addWidget(card["widget"], 0, index)  # type: ignore[arg-type]

class DatasetPlaybackControls(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)

        self.start_button = secondary_button("◀ Start")
        self.prev_button = secondary_button("‹ Prev")
        self.play_button = primary_button("▶ Play")
        self.pause_button = secondary_button("▮▮ Pause")
        self.next_button = secondary_button("Next ›")

        for button in (
            self.start_button,
            self.prev_button,
            self.play_button,
            self.pause_button,
            self.next_button,
        ):
            self._make_compact(button)
            buttons.addWidget(button)
        buttons.addStretch()

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        layout.addLayout(buttons)
        layout.addWidget(self.timeline_slider)

    @staticmethod
    def _make_compact(button: QPushButton) -> None:
        button.setFixedHeight(28)
        button.setMaximumWidth(92)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

class SignalChartsPanel(QWidget):
    def __init__(self, chart_height: int = 180, parent: QWidget | None = None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        title = QLabel("Signal Visualizations (Max 6)")
        title.setProperty("sectionTitle", True)
        outer.addWidget(title)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.charts = [
            ChartCanvas("Open a dataset to monitor incoming data.", min_height=chart_height)
            for _ in range(6)
        ]
        for index, chart in enumerate(self.charts):
            chart.setFixedHeight(chart_height)
            chart.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            grid.addWidget(chart, index // 3, index % 3)

        for column in range(3):
            grid.setColumnStretch(column, 1)

        container = QWidget()
        container.setLayout(grid)
        container.setFixedHeight(chart_height * 2 + grid.verticalSpacing())
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        outer.addWidget(container, alignment=Qt.AlignmentFlag.AlignTop)


class VideoPlaybackPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(8)

        title = QLabel("Video Playback")
        title.setProperty("sectionTitle", True)
        layout.addWidget(title)

        self.video_widget = None
        self.controls_widget = None
        self.play_button = None
        self.pause_button = None
        self.restart_button = None
        self.stop_button = None

        if VIDEO_AVAILABLE and QVideoWidget is not None:
            self.video_widget = QVideoWidget()
            self.video_widget.setMinimumHeight(420)
            self.video_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            layout.addWidget(self.video_widget, 1)

            self.controls_widget = QWidget()
            controls = QHBoxLayout(self.controls_widget)
            controls.setContentsMargins(0, 0, 0, 0)
            controls.setSpacing(6)

            self.play_button = primary_button("▶ Video Play")
            self.pause_button = secondary_button("▮▮ Video Pause")
            self.restart_button = secondary_button("↻ Restart Video")
            self.stop_button = secondary_button("■ Video Stop")

            for button in (
                self.play_button,
                self.pause_button,
                self.restart_button,
                self.stop_button,
            ):
                button.setMinimumHeight(32)
                controls.addWidget(button)
            controls.addStretch()
            layout.addWidget(self.controls_widget)
            self.hide()
        else:
            unavailable = QLabel("Qt Multimedia is unavailable in this PySide6 installation.")
            unavailable.setWordWrap(True)
            layout.addWidget(unavailable)


class PlaybackPreviewTabs(QTabWidget):
    def __init__(self, preview_model, annotation_model, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("playbackBottomTabs")
        self.setDocumentMode(True)
        self.setMovable(False)
        self.setTabsClosable(False)

        self.addTab(data_panel("CURRENT SAMPLE / DATASET PREVIEW", preview_model), "Dataset Preview")
        self.addTab(
            data_panel("MANUAL ANNOTATIONS AND LABEL CORRECTIONS", annotation_model),
            "Annotations & Corrections",
        )
        self.setMinimumHeight(135)
        self.setMaximumHeight(190)


class AnnotationPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(section_label("MANUAL ANNOTATION / LABEL CORRECTION"))

        self.existing_label = status_card("Existing Label", "No label selected")
        self.predicted_label = status_card("Prediction", "No label selected")
        self.corrected_label = status_card("Corrected Label", "Not corrected")
        for card in (self.existing_label, self.predicted_label, self.corrected_label):
            layout.addWidget(card["widget"])  # type: ignore[arg-type]

        self.corrected_label_input = QLineEdit()
        self.corrected_label_input.setPlaceholderText("Corrected label")
        layout.addWidget(QLabel("Corrected label"))
        layout.addWidget(self.corrected_label_input)

        self.annotation_note = QTextEdit()
        self.annotation_note.setPlaceholderText("Annotation note")
        self.annotation_note.setMinimumHeight(48)
        self.annotation_note.setMaximumHeight(70)
        layout.addWidget(QLabel("Annotation note"))
        layout.addWidget(self.annotation_note)

        self.save_correction_button = primary_button("Save Correction")
        layout.addWidget(self.save_correction_button)
        layout.addWidget(section_label("DOWNLOADS"))

        self.export_dataset_button = primary_button("Export Corrected Dataset")
        self.export_log_button = secondary_button("Export Annotation Log")
        self.export_predictions_button = secondary_button("Export Predictions + Confidence")
        self.export_predictions_button.setVisible(False)
        self.export_predictions_button.setEnabled(False)
        self.export_udp_data_button = secondary_button("Export UDP Data")
        self.export_udp_data_button.setVisible(False)
        self.export_udp_data_button.setEnabled(False)

        layout.addWidget(self.export_dataset_button)
        layout.addWidget(self.export_log_button)
        layout.addWidget(self.export_predictions_button)
        layout.addWidget(self.export_udp_data_button)
        layout.addStretch()