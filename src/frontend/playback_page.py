from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMessageBox,
    QProgressDialog,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from src.frontend.playback_components import (
    AnnotationPanel,
    DatasetPlaybackControls,
    PlaybackPreviewTabs,
    PlaybackStatusPanel,
    SignalChartsPanel,
    VideoPlaybackPanel,
)
from src.frontend.playback_sidebar import PlaybackSidebar
from src.frontend.table_model import PandasTableModel
from src.playback_annotation.playback_annotation import PlaybackAnnotationManager
from src.playback_annotation.udp_stream import UdpStreamReceiver
from src.playback_annotation.video_controller import (
    VIDEO_AVAILABLE,
    VideoPlaybackController,
)
from src.playback_annotation.video_sync import VideoSyncMapper


class PlaybackAnnotationPage(QWidget):
    def __init__(
        self,
        sidebar: PlaybackSidebar,
        data_provider: Callable[[], pd.DataFrame],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.sidebar = sidebar
        self.data_provider = data_provider

        self.backend = PlaybackAnnotationManager()
        self._dataset_signature = None
        self._uploaded_dataset = pd.DataFrame()
        self._uploaded_dataset_path: Optional[Path] = None

        self.preview_model = PandasTableModel()
        self.annotation_model = PandasTableModel()
        self.current_sample_model = PandasTableModel()

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._advance_frame)

        self.stream_receiver = UdpStreamReceiver(self)

        # Store every UDP packet immediately, but redraw the UI at a certain rate
        self._udp_ui_update_pending = False

        self.udp_refresh_timer = QTimer(self)
        self.udp_refresh_timer.setInterval(100)  # 10 UI refreshes per second
        self.udp_refresh_timer.timeout.connect(
            self._refresh_udp_display
        )
        self.udp_refresh_timer.start()

        self._annotation_row = None

        self.video_controller = VideoPlaybackController(self)
        self.video_sync = VideoSyncMapper()

        self._build_ui()
        self._connect_controls()

        self._on_source_changed(
            self.sidebar.source_combo.currentText()
        )

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(18)

        self.visual_scroll_area = QScrollArea()
        self.visual_scroll_area.setWidgetResizable(True)
        self.visual_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.visual_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.visual_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        self.center_content = QWidget()
        center_layout = QVBoxLayout(self.center_content)
        center_layout.setContentsMargins(0, 0, 12, 12)
        center_layout.setSpacing(10)
        self.visual_scroll_area.setWidget(self.center_content)
        root.addWidget(self.visual_scroll_area, 1)

        self.status_panel = PlaybackStatusPanel()
        self.status_cards = self.status_panel.values
        center_layout.addWidget(self.status_panel)

        self.playback_controls = DatasetPlaybackControls()
        self.start_button = self.playback_controls.start_button
        self.prev_button = self.playback_controls.prev_button
        self.play_button = self.playback_controls.play_button
        self.pause_button = self.playback_controls.pause_button
        self.next_button = self.playback_controls.next_button
        self.timeline_slider = self.playback_controls.timeline_slider
        center_layout.addWidget(self.playback_controls)

        self.signal_panel = SignalChartsPanel(chart_height=180)
        self.charts = self.signal_panel.charts
        center_layout.addWidget(self.signal_panel)

        self.video_panel = VideoPlaybackPanel()
        self.video_section = self.video_panel
        self.video_widget = self.video_panel.video_widget
        self.video_controls_widget = self.video_panel.controls_widget
        self.video_play_button = self.video_panel.play_button
        self.video_pause_button = self.video_panel.pause_button
        self.video_restart_button = self.video_panel.restart_button
        self.video_stop_button = self.video_panel.stop_button
        center_layout.addWidget(self.video_panel)

        self.bottom_tabs = PlaybackPreviewTabs(
            self.preview_model,
            self.annotation_model,
        )
        center_layout.addWidget(self.bottom_tabs)

        self.annotation_panel = AnnotationPanel()
        self.existing_label = self.annotation_panel.existing_label
        self.predicted_label = self.annotation_panel.predicted_label
        self.corrected_label = self.annotation_panel.corrected_label
        self.corrected_label_input = self.annotation_panel.corrected_label_input
        self.annotation_note = self.annotation_panel.annotation_note
        self.save_correction_button = self.annotation_panel.save_correction_button
        self.export_dataset_button = self.annotation_panel.export_dataset_button
        self.export_log_button = self.annotation_panel.export_log_button
        self.export_predictions_button = self.annotation_panel.export_predictions_button
        self.export_udp_data_button = self.annotation_panel.export_udp_data_button
        root.addWidget(self.annotation_panel)

    def _connect_controls(self):
        # Source selection
        self.sidebar.source_combo.currentTextChanged.connect(
            self._on_source_changed
        )

        # Direct dataset upload for Tab 4
        self.sidebar.upload_dataset_button.clicked.connect(
            self.upload_dataset
        )

        # UDP controls
        self.sidebar.stream_button.clicked.connect(
            self.start_udp_stream
        )
        self.sidebar.stop_stream_button.clicked.connect(
            self.stop_udp_stream
        )

        self.stream_receiver.sample_received.connect(
            self._receive_stream_sample
        )
        self.stream_receiver.status_changed.connect(
            self.sidebar.stream_status.setText
        )
        self.stream_receiver.error_occurred.connect(
            self._stream_error
        )

        # Video loading
        self.sidebar.load_video_button.clicked.connect(
            self.load_video
        )
        self.video_controller.status_changed.connect(
            self.sidebar.video_status.setText
        )
        self.video_controller.error_occurred.connect(
            self._video_error
        )
        if VIDEO_AVAILABLE and self.video_widget is not None:
            self.video_controller.set_video_output(self.video_widget)

        if VIDEO_AVAILABLE:
            self.video_play_button.clicked.connect(
                self.play_video
            )
            self.video_pause_button.clicked.connect(
                self.pause_video
            )
            self.video_restart_button.clicked.connect(
                self.restart_video
            )
            self.video_stop_button.clicked.connect(
                self.stop_video
            )

        # Dataset column controls
        self.sidebar.timestamp_combo.currentTextChanged.connect(
            self._on_timestamp_column_changed
        )
        self.sidebar.label_combo.currentTextChanged.connect(
            self.render_current_frame
        )
        self.sidebar.confidence_combo.currentTextChanged.connect(
            self.render_current_frame
        )

        self.sidebar.feature_picker.selectionChange.connect(
            self.render_current_frame
        )
        self.sidebar.signal_picker.selectionChange.connect(
            self.render_current_frame
        )

        self.sidebar.window_size.valueChanged.connect(
            self.render_current_frame
        )
        self.sidebar.playback_delay.valueChanged.connect(
            self._update_timer_interval
        )

        # Model
        self.sidebar.load_model_button.clicked.connect(
            self.load_model
        )

        # Dataset playback
        self.start_button.clicked.connect(
            lambda: self.seek(0)
        )

        self.prev_button.clicked.connect(
            lambda: self.seek(
                self.backend.current_row - 1
            )
        )

        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)

        self.next_button.clicked.connect(
            lambda: self.seek(
                self.backend.current_row + 1
            )
        )

        self.timeline_slider.valueChanged.connect(
            self.seek
        )

        # Annotation
        self.corrected_label_input.editingFinished.connect(
            self.pause_for_annotation
        )

        self.save_correction_button.clicked.connect(
            self.save_correction
        )

        self.sidebar.predict_dataset_button.clicked.connect(
            self.predict_entire_dataset
        )

        self.export_predictions_button.clicked.connect(
            self.export_predictions
        )

        self.export_dataset_button.clicked.connect(
            self.export_corrected_dataset
        )

        self.export_log_button.clicked.connect(
            self.export_annotation_log
        )

        self.export_udp_data_button.clicked.connect(
            self.export_udp_data
        )

    def _on_timestamp_column_changed(self):
        self.video_sync.reset()
        self.render_current_frame()
        self.sync_video_to_current_row(force=True)

    def _on_source_changed(self, source_text: str):
        using_udp = source_text == "UDP / JSON Stream"
        using_upload = source_text == "Upload Dataset"

        self.sidebar.set_udp_controls_visible(using_udp)
        self.sidebar.upload_dataset_button.setVisible(using_upload)
        self.sidebar.upload_dataset_button.setEnabled(using_upload)

        # Changing the active data source invalidates any previous batch export.
        self.backend.prediction_dataset = pd.DataFrame()
        self.export_predictions_button.setVisible(False)
        self.export_predictions_button.setEnabled(False)

        udp_data_available = (
            using_udp
            and not self.backend.dataset.empty
        )
        self.export_udp_data_button.setVisible(udp_data_available)
        self.export_udp_data_button.setEnabled(udp_data_available)

        self.sync_dataset()

    @staticmethod
    def _make_dataset_signature(
        df: pd.DataFrame,
    ):
        if df is None or df.empty:
            return None

        return (
            len(df),
            tuple(
                str(column)
                for column in df.columns
            ),
            tuple(
                str(dtype)
                for dtype in df.dtypes
            ),
        )

    def sync_dataset(self):
        source = self.sidebar.source_combo.currentText()

        if source == "UDP / JSON Stream":
            self.pause()

            self.sidebar.dataset_status.setText(
                "Live UDP source selected. "
                "Start the stream to receive samples."
            )

            self.render_current_frame()
            return

        self.stop_udp_stream()

        if source == "Upload Dataset":
            df = self._uploaded_dataset
        else:
            df = self.data_provider()

        if df is None:
            df = pd.DataFrame()

        self.pause()

        new_signature = (
            source,
            self._make_dataset_signature(df),
        )

        if new_signature != self._dataset_signature:
            self.backend.set_dataset(df)

            self.backend.annotations = pd.DataFrame(
                columns=(
                    PlaybackAnnotationManager
                    .ANNOTATION_COLUMNS
                )
            )

            self._dataset_signature = new_signature
        else:
            self.backend.set_dataset(df)

        self.video_sync.reset()
        self.sidebar.set_columns(df)

        if source == "Upload Dataset":
            if df.empty:
                self.sidebar.dataset_status.setText(
                    "No Tab 4 dataset uploaded. Click Upload Dataset..."
                )
            else:
                file_name = (
                    self._uploaded_dataset_path.name
                    if self._uploaded_dataset_path is not None
                    else "Uploaded dataset"
                )
                self.sidebar.dataset_status.setText(
                    f"Uploaded: {file_name} — "
                    f"{len(df):,} rows × {len(df.columns):,} columns"
                )

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setRange(
            0,
            max(0, len(df) - 1),
        )
        self.timeline_slider.setValue(
            self.backend.current_row
        )
        self.timeline_slider.blockSignals(False)

        self.annotation_model.set_data(
            self.backend.annotations.copy()
        )

        self.corrected_label_input.clear()
        self.annotation_note.clear()

        self.render_current_frame()

    def upload_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Upload Dataset for Playback and Prediction",
            str(Path.home() / "Downloads"),
            (
                "Dataset Files (*.csv *.xlsx *.xls);;"
                "CSV Files (*.csv);;"
                "Excel Files (*.xlsx *.xls);;"
                "All Files (*.*)"
            ),
        )

        if not path:
            return

        dataset_path = Path(path)

        try:
            suffix = dataset_path.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(dataset_path)
            elif suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(dataset_path)
            else:
                raise ValueError(
                    "Unsupported dataset format. Use CSV, XLSX, or XLS."
                )
        except Exception as error:
            QMessageBox.critical(
                self,
                "Dataset Upload Error",
                f"Could not load the selected dataset.\n\n{error}",
            )
            return

        if df.empty:
            QMessageBox.warning(
                self,
                "Empty Dataset",
                "The selected dataset contains no rows.",
            )
            return

        self._uploaded_dataset = df.copy()
        self._uploaded_dataset_path = dataset_path

        # Predictions made for a previous dataset are no longer valid
        self.backend.prediction_dataset = pd.DataFrame()
        self.export_predictions_button.setVisible(False)
        self.export_predictions_button.setEnabled(False)

        self._dataset_signature = None

        if self.sidebar.source_combo.currentText() != "Upload Dataset":
            self.sidebar.source_combo.setCurrentText("Upload Dataset")
        else:
            self.sync_dataset()

        QMessageBox.information(
            self,
            "Dataset Loaded",
            (
                f"Loaded {dataset_path.name}\n"
                f"{len(df):,} rows × {len(df.columns):,} columns"
            ),
        )

    def reset(self):
        self.timer.stop()
        self.stop_udp_stream()

        self.video_controller.clear()
        self.video_sync.reset()

        self.backend = PlaybackAnnotationManager()
        self._dataset_signature = None
        self._uploaded_dataset = pd.DataFrame()
        self._uploaded_dataset_path = None

        self.preview_model.set_data(
            pd.DataFrame()
        )

        self.annotation_model.set_data(
            pd.DataFrame(
                columns=(
                    PlaybackAnnotationManager
                    .ANNOTATION_COLUMNS
                )
            )
        )

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)

        self.corrected_label_input.clear()
        self.annotation_note.clear()

        self.sidebar.model_status.setText(
            "No model loaded"
        )
        self.sidebar.predict_dataset_button.setEnabled(False)
        self.export_predictions_button.setVisible(False)
        self.export_predictions_button.setEnabled(False)
        self.export_udp_data_button.setVisible(False)
        self.export_udp_data_button.setEnabled(False)
        self._udp_ui_update_pending = False

        self.sidebar.video_status.setText(
            "No video selected"
        )

        self.sidebar.set_columns(
            pd.DataFrame()
        )

        if VIDEO_AVAILABLE:
            self.video_section.hide()

        self.render_current_frame()


    # Dataset playback
    def play(self):
        if self.backend.dataset.empty:
            QMessageBox.warning(
                self,
                "Missing Dataset",
                "Open a recorded dataset or start a UDP "
                "stream before playback.",
            )
            return

        self._update_timer_interval()
        self.sync_video_to_current_row(force=True)

        if self.video_controller.has_media:
            self.video_controller.play()

        self.timer.start(self.timer.interval())
        self.status_cards["Playback"].setText("Playing")

    def pause(self):
        self.timer.stop()

        if self.video_controller.has_media:
            self.video_controller.pause()

        self.status_cards["Playback"].setText("Paused")

    def _update_timer_interval(self):
        timestamp_column = self._selected_column(
            self.sidebar.timestamp_combo
        )

        interval = self.backend.playback_interval_ms(
            timestamp_column,
            self.sidebar.playback_delay.value(),
        )

        self.timer.setInterval(interval)

    def _advance_frame(self):
        row = self.backend.advance()

        if row is None:
            self.pause()
            return

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(row)
        self.timeline_slider.blockSignals(False)

        self.render_current_frame()
        self.sync_video_to_current_row(force=False)
        self._update_timer_interval()
        self.timer.start(self.timer.interval())

    def seek(self, row):
        if self.backend.dataset.empty:
            return

        resolved = self.backend.seek(row)

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(resolved)
        self.timeline_slider.blockSignals(False)

        self.render_current_frame()
        self.sync_video_to_current_row(force=True)

    def _selected_column(
        self,
        combo: QComboBox,
    ) -> Optional[str]:
        value = combo.currentText()

        if value and value != "None":
            return value

        return None

    def render_current_frame(self):
        df = self.backend.dataset

        if df.empty:
            self.preview_model.set_data(
                pd.DataFrame()
            )

            for chart in self.charts:
                chart.show_empty(
                    "Open a dataset to monitor incoming data."
                )

            self.status_cards["Current Index"].setText(
                "—"
            )
            self.status_cards["Timestamp"].setText(
                "—"
            )
            self.status_cards["Prediction"].setText(
                "No label selected"
            )
            self.status_cards["Confidence"].setText(
                "N/A"
            )
            self.status_cards["Playback"].setText(
                "Paused"
            )

            return

        current_row = self.backend.current_row
        sample = self.backend.current_sample()

        preview_start = max(
            0,
            current_row - 4,
        )

        preview_end = min(
            len(df),
            current_row + 5,
        )

        self.preview_model.set_data(
            df.iloc[
                preview_start:preview_end
            ].copy()
        )

        timestamp_column = self._selected_column(
            self.sidebar.timestamp_combo
        )

        label_column = self._selected_column(
            self.sidebar.label_combo
        )

        confidence_column = self._selected_column(
            self.sidebar.confidence_combo
        )

        if (
            timestamp_column
            and timestamp_column in sample.columns
        ):
            timestamp = sample.iloc[0][
                timestamp_column
            ]
        else:
            timestamp = current_row

        if (
            label_column
            and label_column in sample.columns
        ):
            existing = sample.iloc[0][
                label_column
            ]
        else:
            existing = "Unlabelled dataset"

        if (
            confidence_column
            and confidence_column in sample.columns
        ):
            confidence = sample.iloc[0][
                confidence_column
            ]
        else:
            confidence = "N/A"

        selected_features = (
            self.sidebar.feature_picker.selected_items()
        )

        prediction_result = (
            self.backend.predict_current(
                label_col=label_column,
                feature_columns=selected_features,
            )
        )

        prediction = prediction_result.label

        if prediction_result.confidence is not None:
            confidence = (
                f"{prediction_result.confidence:.3f}"
            )
        elif prediction_result.error:
            confidence = "Unavailable"

        corrected = self._corrected_label_for_row(
            current_row
        )

        self.status_cards["Current Index"].setText(
            f"{current_row + 1:,} / {len(df):,}"
        )

        self.status_cards["Timestamp"].setText(
            str(timestamp)
        )

        self.status_cards["Prediction"].setText(
            str(prediction)
        )

        self.status_cards["Confidence"].setText(
            str(confidence)
        )

        if not self.timer.isActive():
            self.status_cards["Playback"].setText(
                "Paused"
            )

        self.existing_label["value"].setText(
            str(existing)
        )

        self.predicted_label["value"].setText(
            str(prediction)
        )

        if corrected is None:
            corrected_text = "Not corrected"
        else:
            corrected_text = str(corrected)

        self.corrected_label["value"].setText(
            corrected_text
        )

        rows_back = self.sidebar.window_size.value()

        signal_columns = (
            self.sidebar.signal_picker.selected_items()
        )

        if not signal_columns:
            signal_columns = (
                self.backend.numeric_signal_columns(
                    label_col=label_column
                )[:6]
            )

        window_df = self.backend.signal_window(
            rows_back=rows_back
        )

        for index, chart in enumerate(self.charts):
            if (
                index < len(signal_columns)
                and signal_columns[index]
                in window_df.columns
            ):
                chart.plot_realtime_signal(
                    window_df,
                    signal_columns[index],
                    current_row=current_row,
                )
            else:
                chart.show_empty(
                    "Video stream or additional "
                    "signal unavailable."
                )

    def _corrected_label_for_row(
        self,
        row_index: int,
    ):
        annotations = self.backend.annotations

        if (
            annotations.empty
            or "row_index" not in annotations.columns
        ):
            return None

        matches = annotations[
            annotations["row_index"] == row_index
        ]

        if matches.empty:
            return None

        return matches.iloc[-1].get(
            "corrected_label"
        )


    # Model
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Pretrained Model",
            str(Path.home() / "Downloads"),
            (
                "Pickle/Joblib Models "
                "(*.pkl *.pickle *.joblib);;"
                "All Files (*.*)"
            ),
        )

        if not path:
            return

        try:
            self.backend.load_model(path)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Model Load Error",
                str(error),
            )
            return

        expected_features = self.backend.model_feature_names()
        available_features = set(
            self.sidebar.feature_picker.checkboxes
        )
        matched_features = [
            feature for feature in expected_features
            if feature in available_features
        ]

        if expected_features and len(matched_features) == len(expected_features):
            self.sidebar.feature_picker.set_selected(matched_features)
            feature_note = f" — {len(matched_features)} features matched"
        elif expected_features:
            missing = [
                feature for feature in expected_features
                if feature not in available_features
            ]
            feature_note = (
                " — missing dataset features: "
                + ", ".join(missing)
            )
        else:
            feature_note = " — select the training features manually"

        self.sidebar.model_status.setText(
            Path(path).name + feature_note
        )

        self.sidebar.predict_dataset_button.setEnabled(True)

        # Loading a model does not create exportable predictions.
        # Keep the export button hidden until prediction succeeds.
        self.export_predictions_button.setVisible(False)
        self.export_predictions_button.setEnabled(False)

        self.render_current_frame()


    # Annotation
    def pause_for_annotation(self):
        if not self.backend.dataset.empty:
            self.pause()
            self._annotation_row = (
                self.backend.current_row
            )

    def save_correction(self):
        if self.backend.dataset.empty:
            QMessageBox.warning(
                self,
                "Missing Dataset",
                "Open a dataset in the Import tab "
                "before annotating.",
            )
            return

        corrected = (
            self.corrected_label_input
            .text()
            .strip()
        )

        if not corrected:
            QMessageBox.warning(
                self,
                "Missing Label",
                "Enter a corrected label before saving.",
            )
            return

        predicted = self.predicted_label[
            "value"
        ].text()

        note = (
            self.annotation_note
            .toPlainText()
            .strip()
        )

        timestamp_column = self._selected_column(
            self.sidebar.timestamp_combo
        )

        try:
            annotations = (
                self.backend.add_or_update_annotation(
                    corrected_label=corrected,
                    predicted_label=predicted,
                    timestamp_col=timestamp_column,
                    annotation_note=note,
                    row_index=self._annotation_row,
                )
            )
        except ValueError as error:
            QMessageBox.warning(
                self,
                "Annotation Error",
                str(error),
            )
            return

        self.annotation_model.set_data(
            annotations
        )

        self.corrected_label_input.clear()
        self.annotation_note.clear()
        self._annotation_row = None

        self.render_current_frame()

    def sync_video_to_current_row(self, force: bool = False):
        """Align the loaded video with the current dataset row."""

        if not self.video_controller.has_media or self.backend.dataset.empty:
            return

        timestamp_column = self._selected_column(
            self.sidebar.timestamp_combo
        )

        expected_ms = self.video_sync.position_for_row(
            dataframe=self.backend.dataset,
            row_index=self.backend.current_row,
            timestamp_column=timestamp_column,
            fallback_interval_ms=self.sidebar.playback_delay.value(),
        )

        if expected_ms is None:
            return

        drift_ms = abs(expected_ms - self.video_controller.position_ms)

        if force or drift_ms > 300:
            self.video_controller.seek_ms(expected_ms)


    # Video
    def load_video(self):
        if not VIDEO_AVAILABLE:
            QMessageBox.warning(
                self,
                "Video Unavailable",
                "This PySide6 installation does not include Qt Multimedia.",
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*.*)",
        )
        if not path:
            return

        try:
            self.video_controller.load(path, autoplay=False)
        except (OSError, RuntimeError) as error:
            QMessageBox.warning(self, "Video Playback Error", str(error))
            return

        self.video_section.show()
        self.video_sync.reset()
        self.sync_video_to_current_row(force=True)
        QTimer.singleShot(0, self._scroll_to_video)

    def play_video(self):
        if not self.video_controller.play():
            QMessageBox.information(self, "No Video", "Open a video file first.")

    def pause_video(self):
        self.video_controller.pause()

    def restart_video(self):
        if not self.video_controller.restart():
            QMessageBox.information(self, "No Video", "Open a video file first.")

    def stop_video(self):
        self.video_controller.stop()

    def _scroll_to_video(self):
        scrollbar = self.visual_scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _video_error(self, message: str):
        QMessageBox.warning(self, "Video Playback Error", message)


    # UDP
    def start_udp_stream(self):
        self.sidebar.source_combo.setCurrentText(
            "UDP / JSON Stream"
        )

        self.pause()

        self.backend.clear_stream()
        self._udp_ui_update_pending = False
        self.export_udp_data_button.setVisible(False)
        self.export_udp_data_button.setEnabled(False)

        self.backend.annotations = pd.DataFrame(
            columns=(
                PlaybackAnnotationManager
                .ANNOTATION_COLUMNS
            )
        )

        self.annotation_model.set_data(
            self.backend.annotations.copy()
        )

        started = self.stream_receiver.start(
            self.sidebar.udp_host.text(),
            self.sidebar.udp_port.value(),
        )

        if started:
            self.sidebar.dataset_status.setText(
                "Waiting for live UDP samples…"
            )

            self.render_current_frame()

    def stop_udp_stream(self):
        self.stream_receiver.stop()

    def _stream_error(self, message: str):
        self.sidebar.stream_status.setText(
            message
        )

    def _receive_stream_sample(
        self,
        sample: dict,
    ):

        previous_columns = tuple(
            self.backend.dataset.columns
        )

        try:
            self.backend.append_stream_sample(
                sample
            )
        except ValueError as error:
            self._stream_error(
                str(error)
            )
            return

        if (
            tuple(self.backend.dataset.columns)
            != previous_columns
        ):
            self.sidebar.set_columns(
                self.backend.dataset
            )

        self.export_udp_data_button.setVisible(True)
        self.export_udp_data_button.setEnabled(True)
        self._udp_ui_update_pending = True

    def _refresh_udp_display(self):
        if not self._udp_ui_update_pending:
            return

        if (
            self.sidebar.source_combo.currentText()
            != "UDP / JSON Stream"
        ):
            self._udp_ui_update_pending = False
            return

        df = self.backend.dataset

        if df.empty:
            self._udp_ui_update_pending = False
            return

        current_row = len(df) - 1
        self.backend.seek(current_row)

        self.sidebar.dataset_status.setText(
            f"Live buffer: "
            f"{len(df):,} rows × "
            f"{len(df.columns):,} columns"
        )

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setRange(
            0,
            max(0, len(df) - 1),
        )
        self.timeline_slider.setValue(current_row)
        self.timeline_slider.blockSignals(False)

        self.render_current_frame()
        self._udp_ui_update_pending = False


    # Export
    def predict_entire_dataset(self):
        if self.backend.dataset.empty:
            QMessageBox.warning(
                self,
                "No Dataset",
                "Upload or select an unlabelled dataset first.",
            )
            return

        if self.backend.model is None:
            QMessageBox.warning(
                self,
                "No Model",
                "Load the saved PKL model before running batch prediction.",
            )
            return

        selected_features = self.sidebar.feature_picker.selected_items()
        if not selected_features:
            QMessageBox.warning(
                self,
                "No Model Features",
                "Select the exact feature columns used to train the PKL model.",
            )
            return

        progress = QProgressDialog(
            "Predicting every row in the dataset…",
            None,
            0,
            0,
            self,
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        try:
            result = self.backend.predict_dataset(
                selected_features,
                prediction_column="Predicted_Label",
                confidence_column="Prediction_Confidence",
            )
        except Exception as error:
            progress.close()
            self.export_predictions_button.setEnabled(False)
            QMessageBox.critical(
                self,
                "Prediction Error",
                str(error),
            )
            return

        progress.close()

        # The export action appears only after predictions have been generated
        self.export_predictions_button.setVisible(True)
        self.export_predictions_button.setEnabled(True)

        confidence_available = result["Prediction_Confidence"].notna().any()
        confidence_text = (
            "Confidence values were generated using predict_proba()."
            if confidence_available
            else "The model does not provide predict_proba(); confidence is blank."
        )
        QMessageBox.information(
            self,
            "Dataset Prediction Complete",
            f"Predicted {len(result):,} rows.\n\n"
            f"Added columns:\n"
            f"• Predicted_Label\n"
            f"• Prediction_Confidence\n\n"
            f"{confidence_text}\n\n"
            "Click Export Predictions + Confidence to save the CSV.",
        )

    def export_udp_data(self):
        if self.backend.dataset.empty:
            QMessageBox.warning(
                self,
                "No UDP Data",
                "Start the UDP stream and receive at least one sample first.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export UDP Data",
            str(
                (
                    Path.home()
                    / "Downloads"
                    / "udp_stream_data.csv"
                ).resolve()
            ),
            "CSV Files (*.csv)",
        )

        if not path:
            return

        if not path.lower().endswith(".csv"):
            path += ".csv"

        try:
            self.backend.dataset.to_csv(
                path,
                index=False,
            )
        except Exception as error:
            QMessageBox.critical(
                self,
                "UDP Export Error",
                str(error),
            )
            return

        QMessageBox.information(
            self,
            "UDP Export Complete",
            (
                f"Saved {len(self.backend.dataset):,} UDP samples "
                f"to {path}"
            ),
        )

    def export_predictions(self):
        if self.backend.prediction_dataset.empty:
            QMessageBox.warning(
                self,
                "No Predictions",
                "Run Predict Entire Dataset before exporting.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Predictions and Confidence",
            str(
                (
                    Path.home()
                    / "Downloads"
                    / "dataset_predictions.csv"
                ).resolve()
            ),
            "CSV Files (*.csv)",
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        try:
            self.backend.export_prediction_dataset(path)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Export Error",
                str(error),
            )
            return

        QMessageBox.information(
            self,
            "Export Complete",
            f"Saved predictions and confidence to {path}",
        )

    def export_corrected_dataset(self):
        if self.backend.dataset.empty:
            QMessageBox.warning(
                self,
                "No Dataset",
                "Open a dataset before exporting.",
            )
            return

        if self.backend.annotations.empty:
            QMessageBox.warning(
                self,
                "No Annotations",
                "Save at least one correction before "
                "exporting.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Corrected Dataset",
            str(
                (
                    Path.home()
                    / "Downloads"
                    / "corrected_dataset.csv"
                ).resolve()
            ),
            "CSV Files (*.csv)",
        )

        if not path:
            return

        progress = QProgressDialog(
            "Exporting corrected dataset…",
            None,
            0,
            0,
            self,
        )

        progress.setWindowModality(
            Qt.WindowModality.WindowModal
        )
        progress.show()

        try:
            self.backend.export_corrected_dataset(
                path
            )
        except Exception as error:
            progress.close()

            QMessageBox.critical(
                self,
                "Export Error",
                str(error),
            )
            return

        progress.close()

        QMessageBox.information(
            self,
            "Export Complete",
            f"Saved corrected dataset to {path}",
        )

    def export_annotation_log(self):
        if self.backend.annotations.empty:
            QMessageBox.warning(
                self,
                "No Annotations",
                "Save at least one correction before "
                "exporting.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Annotation Log",
            str(
                (
                    Path.home()
                    / "Downloads"
                    / "annotation_log.csv"
                ).resolve()
            ),
            "CSV Files (*.csv)",
        )

        if not path:
            return

        progress = QProgressDialog(
            "Exporting annotation log…",
            None,
            0,
            0,
            self,
        )

        progress.setWindowModality(
            Qt.WindowModality.WindowModal
        )
        progress.show()

        try:
            self.backend.export_annotation_log(
                path
            )
        except Exception as error:
            progress.close()

            QMessageBox.critical(
                self,
                "Export Error",
                str(error),
            )
            return

        progress.close()

        QMessageBox.information(
            self,
            "Export Complete",
            f"Saved annotation log to {path}",
        )