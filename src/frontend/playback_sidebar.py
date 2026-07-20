import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.frontend.widgets import (
    ColumnPicker,
    SIDEBAR_WIDTH,
    divider,
    primary_button,
    secondary_button,
    section_label,
    taller_dropdown,
)


class PlaybackSidebar(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.setProperty("sidebar", True)
        self.setFixedWidth(SIDEBAR_WIDTH)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.scroll_content = QWidget()
        self.scroll_content.setProperty("sidebar", True)

        layout = QVBoxLayout(self.scroll_content)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(6)

        self.scroll_area.setWidget(self.scroll_content)
        outer_layout.addWidget(self.scroll_area)

        title = QLabel("Classify & Learn Lab")
        title.setObjectName("brand")
        layout.addWidget(title)
        layout.addSpacing(12)

        layout.addWidget(section_label("SOURCE"))

        self.source_combo = taller_dropdown(QComboBox())
        self.source_combo.addItems(
            [
                "Imported Dataset",
                "Upload Dataset",
                "UDP / JSON Stream",
            ]
        )
        layout.addWidget(QLabel("Data source"))
        layout.addWidget(self.source_combo)

        self.dataset_status = QLabel("No imported dataset is available yet.")
        self.dataset_status.setWordWrap(True)
        layout.addWidget(self.dataset_status)

        self.upload_dataset_button = secondary_button("Upload Dataset...")
        self.upload_dataset_button.setEnabled(False)
        layout.addWidget(self.upload_dataset_button)

        self.udp_controls_container = QWidget()
        udp_layout = QVBoxLayout(self.udp_controls_container)
        udp_layout.setContentsMargins(0, 8, 0, 0)
        udp_layout.setSpacing(6)

        self.udp_host = QLineEdit("0.0.0.0")
        self.udp_port = QSpinBox()
        self.udp_port.setRange(1, 65535)
        self.udp_port.setValue(5005)
        self.stream_button = primary_button("Start UDP Stream")
        self.stop_stream_button = secondary_button("Stop UDP Stream")
        self.stream_status = QLabel("UDP stream stopped")
        self.stream_status.setWordWrap(True)

        udp_layout.addWidget(QLabel("UDP interface"))
        udp_layout.addWidget(self.udp_host)
        udp_layout.addWidget(QLabel("UDP port"))
        udp_layout.addWidget(self.udp_port)
        udp_layout.addWidget(self.stream_button)
        udp_layout.addWidget(self.stop_stream_button)
        udp_layout.addWidget(self.stream_status)
        layout.addWidget(self.udp_controls_container)

        self.load_video_button = secondary_button("Open Video Stream/File")
        self.video_status = QLabel("No video selected")
        self.video_status.setWordWrap(True)
        layout.addWidget(self.load_video_button)
        layout.addWidget(self.video_status)
        layout.addWidget(divider())

        layout.addWidget(section_label("DATASET SETTINGS"))

        self.timestamp_combo = taller_dropdown(QComboBox())
        self.label_combo = taller_dropdown(QComboBox())
        self.confidence_combo = taller_dropdown(QComboBox())

        layout.addWidget(QLabel("Timestamp column"))
        layout.addWidget(self.timestamp_combo)
        layout.addWidget(QLabel("Label / prediction column"))
        layout.addWidget(self.label_combo)
        layout.addWidget(QLabel("Confidence column"))
        layout.addWidget(self.confidence_combo)

        layout.addWidget(QLabel("Model features"))
        self.feature_picker = ColumnPicker("Search features")
        self.feature_picker.setMinimumHeight(115)
        self.feature_picker.setMaximumHeight(145)
        layout.addWidget(self.feature_picker)

        layout.addWidget(QLabel("Signals to monitor"))
        self.signal_picker = ColumnPicker("Search signals")
        self.signal_picker.setMinimumHeight(125)
        self.signal_picker.setMaximumHeight(160)
        layout.addWidget(self.signal_picker)

        self.window_size = QSpinBox()
        self.window_size.setRange(10, 10000)
        self.window_size.setValue(100)
        layout.addWidget(QLabel("Plot window size"))
        layout.addWidget(self.window_size)

        self.playback_delay = QSpinBox()
        self.playback_delay.setRange(10, 10000)
        self.playback_delay.setValue(300)
        self.playback_delay.setSuffix(" ms")
        layout.addWidget(QLabel("Playback delay"))
        layout.addWidget(self.playback_delay)
        layout.addWidget(divider())

        layout.addWidget(section_label("MODEL"))
        self.load_model_button = secondary_button("Load Pretrained Model")
        layout.addWidget(self.load_model_button)

        self.model_status = QLabel("No model loaded")
        self.model_status.setWordWrap(True)
        layout.addWidget(self.model_status)

        self.predict_dataset_button = primary_button("Predict Entire Dataset")
        self.predict_dataset_button.setEnabled(False)
        layout.addWidget(self.predict_dataset_button)
        layout.addStretch()

        self.set_udp_controls_visible(False)

    def set_udp_controls_visible(self, visible: bool) -> None:
        self.udp_controls_container.setVisible(visible)

    def set_columns(self, df: pd.DataFrame) -> None:
        columns = [str(column) for column in df.columns] if df is not None and not df.empty else []
        optional_columns = ["None", *columns]

        for combo in (self.timestamp_combo, self.label_combo, self.confidence_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(optional_columns)
            combo.blockSignals(False)

        if columns:
            lower = {str(column).lower(): str(column) for column in df.columns}
            for key, combo in (
                ("timestamp", self.timestamp_combo),
                ("time", self.timestamp_combo),
                ("predicted_label", self.label_combo),
                ("label", self.label_combo),
                ("true_label", self.label_combo),
                ("confidence", self.confidence_combo),
            ):
                match = lower.get(key)
                if match and combo.currentText() == "None":
                    combo.setCurrentText(match)

        numeric = (
            [str(column) for column in df.select_dtypes(include="number").columns]
            if df is not None and not df.empty
            else []
        )

        self.feature_picker.blockSignals(True)
        self.signal_picker.blockSignals(True)
        self.feature_picker.set_items(columns, checked=False)

        excluded = {
            self.timestamp_combo.currentText(),
            self.label_combo.currentText(),
            self.confidence_combo.currentText(),
            "None",
        }
        self.feature_picker.set_selected([column for column in numeric if column not in excluded])

        self.signal_picker.set_items(numeric or columns, checked=True)
        if len(numeric) > 6:
            self.signal_picker.set_selected(numeric[:6])

        self.feature_picker.blockSignals(False)
        self.signal_picker.blockSignals(False)

        if df is None or df.empty:
            self.dataset_status.setText("No imported dataset is available yet.")
        else:
            self.dataset_status.setText(
                f"Imported dataset: {len(df):,} rows × {len(df.columns):,} columns"
            )