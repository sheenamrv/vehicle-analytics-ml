from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
)

from src.frontend.table_model import PandasTableModel
from src.frontend.widgets import (
    primary_button,
    secondary_button,
    table_view,
)


class ReportImagesDialog(QDialog):
    def __init__(self, parent, title, image_paths):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(980, 680)
        self._images = [
            Path(path)
            for path in image_paths
            if Path(path).exists()
        ]
        self._index = 0
        self._current_image_path = None
        self._zoom = 1.0

        layout = QVBoxLayout(self)

        self.image_name = QLabel("")
        layout.addWidget(self.image_name)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.image_label)
        layout.addWidget(self.scroll, 1)

        controls = QHBoxLayout()
        self.prev_button = secondary_button("Previous")
        self.prev_button.clicked.connect(lambda: self._step(-1))
        self.next_button = secondary_button("Next")
        self.next_button.clicked.connect(lambda: self._step(1))
        zoom_out = secondary_button("-")
        zoom_out.clicked.connect(lambda: self._change_zoom(0.85))
        zoom_in = secondary_button("+")
        zoom_in.clicked.connect(lambda: self._change_zoom(1.15))
        reset_zoom = secondary_button("Reset")
        reset_zoom.clicked.connect(self._reset_zoom)
        close_button = primary_button("Close")
        close_button.clicked.connect(self.accept)

        controls.addWidget(self.prev_button)
        controls.addWidget(self.next_button)
        controls.addWidget(zoom_out)
        controls.addWidget(zoom_in)
        controls.addWidget(reset_zoom)
        controls.addStretch()
        controls.addWidget(close_button)
        layout.addLayout(controls)

        self._refresh_view()

    def _step(self, delta):
        if not self._images:
            return
        self._index = (self._index + delta) % len(self._images)
        self._refresh_view()

    def _refresh_view(self):
        if not self._images:
            self.image_name.setText(
                "No report images were found for this model."
            )
            self.image_label.clear()
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return

        image_path = self._images[self._index]
        self._current_image_path = image_path
        self._zoom = 1.0
        self._apply_zoomed_pixmap()

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.image_name.setText(
                f"Could not load image: {image_path.name}"
            )
            self.image_label.clear()
        else:
            self.image_name.setText(
                f"{self._index + 1}/{len(self._images)} - {image_path.name}"
            )

        enable_nav = len(self._images) > 1
        self.prev_button.setEnabled(enable_nav)
        self.next_button.setEnabled(enable_nav)

    def _change_zoom(self, factor):
        if self._current_image_path is None:
            return
        self._zoom = max(0.1, min(6.0, self._zoom * factor))
        self._apply_zoomed_pixmap()

    def _reset_zoom(self):
        self._zoom = 1.0
        self._apply_zoomed_pixmap()

    def _apply_zoomed_pixmap(self):
        if self._current_image_path is None:
            return
        pixmap = QPixmap(str(self._current_image_path))
        if pixmap.isNull():
            self.image_label.clear()
            return
        width = max(1, int(pixmap.width() * self._zoom))
        height = max(1, int(pixmap.height() * self._zoom))
        scaled = pixmap.scaled(
            width,
            height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.adjustSize()


class WindowPreviewDialog(QDialog):
    def __init__(self, parent, preview_df):
        super().__init__(parent)
        self.setWindowTitle("Window Segmentation Preview")
        self.setMinimumSize(860, 520)

        layout = QVBoxLayout(self)
        title = QLabel(
            "Preview of generated windows before feature extraction"
        )
        title.setProperty("panelTitle", True)
        layout.addWidget(title)

        self.model = PandasTableModel(preview_df)
        self.table = table_view(self.model)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table, 1)

        controls = QHBoxLayout()
        controls.addStretch()
        close_button = primary_button("Close")
        close_button.clicked.connect(self.accept)
        controls.addWidget(close_button)
        layout.addLayout(controls)


class ImageInspectDialog(QDialog):
    def __init__(self, parent, title, image_path):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(980, 720)
        self._base_pixmap = QPixmap(str(image_path))
        self._zoom = 1.0

        layout = QVBoxLayout(self)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.image_label)
        layout.addWidget(self.scroll, 1)

        controls = QHBoxLayout()
        zoom_out = secondary_button("-")
        zoom_out.clicked.connect(lambda: self._change_zoom(0.85))
        zoom_in = secondary_button("+")
        zoom_in.clicked.connect(lambda: self._change_zoom(1.15))
        reset = secondary_button("Reset")
        reset.clicked.connect(self._reset_zoom)
        close_button = primary_button("Close")
        close_button.clicked.connect(self.accept)

        controls.addWidget(zoom_out)
        controls.addWidget(zoom_in)
        controls.addWidget(reset)
        controls.addStretch()
        controls.addWidget(close_button)
        layout.addLayout(controls)

        self._apply_zoom()

    def _change_zoom(self, factor):
        if self._base_pixmap.isNull():
            return
        self._zoom = max(0.1, min(6.0, self._zoom * factor))
        self._apply_zoom()

    def _reset_zoom(self):
        self._zoom = 1.0
        self._apply_zoom()

    def _apply_zoom(self):
        if self._base_pixmap.isNull():
            self.image_label.setText("Unable to load image.")
            return
        width = max(1, int(self._base_pixmap.width() * self._zoom))
        height = max(1, int(self._base_pixmap.height() * self._zoom))
        scaled = self._base_pixmap.scaled(
            width,
            height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.adjustSize()
