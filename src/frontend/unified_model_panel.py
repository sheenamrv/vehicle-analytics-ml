from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, Signal, QEvent, QRect
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTableView,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from src.frontend.table_model import PandasTableModel
from src.frontend.widgets import (
    divider,
    primary_button,
    section_label,
    secondary_button,
    taller_dropdown,
)
from src.model.model_controller import COMMON_TRAINING_PARAMETERS, ModelController


class BorderedCheckBox(QCheckBox):
    """Checkbox with a thick border and explicit tick for readability."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("")
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(22, 22)

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(max(hint.width(), 22))
        hint.setHeight(max(hint.height(), 22))
        return hint

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        side = min(self.width(), self.height()) - 4
        x = (self.width() - side) // 2
        y = (self.height() - side) // 2
        rect = QRect(x, y, side, side)

        border_color = QColor("#1f7f43") if self.isChecked() else QColor("#6c7881")
        painter.setPen(QPen(border_color, 2))
        painter.setBrush(Qt.white)
        painter.drawRoundedRect(rect, 4, 4)

        if self.isChecked():
            tick_pen = QPen(QColor("#1f7f43"), 2.5)
            tick_pen.setCapStyle(Qt.RoundCap)
            tick_pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(tick_pen)
            left = rect.left()
            top = rect.top()
            width = rect.width()
            height = rect.height()
            painter.drawLine(left + int(width * 0.22), top + int(height * 0.56), left + int(width * 0.45), top + int(height * 0.78))
            painter.drawLine(left + int(width * 0.45), top + int(height * 0.78), left + int(width * 0.80), top + int(height * 0.30))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self.toggle()
            event.accept()
            return
        super().keyPressEvent(event)


class AddedModelCheckDelegate(QStyledItemDelegate):
    """Draw and toggle bold checkbox cells in the Added Models first column."""

    def paint(self, painter, option, index):
        checked = index.data(Qt.CheckStateRole) == Qt.Checked
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        side = min(option.rect.width(), option.rect.height()) - 8
        side = max(side, 14)
        x = option.rect.x() + (option.rect.width() - side) // 2
        y = option.rect.y() + (option.rect.height() - side) // 2
        box = QRect(x, y, side, side)

        border_color = QColor("#1f7f43") if checked else QColor("#6c7881")
        painter.setPen(QPen(border_color, 2))
        painter.setBrush(Qt.white)
        painter.drawRoundedRect(box, 3, 3)

        if checked:
            tick_pen = QPen(QColor("#1f7f43"), 2.3)
            tick_pen.setCapStyle(Qt.RoundCap)
            tick_pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(tick_pen)
            left = box.left()
            top = box.top()
            width = box.width()
            height = box.height()
            painter.drawLine(left + int(width * 0.20), top + int(height * 0.56), left + int(width * 0.44), top + int(height * 0.78))
            painter.drawLine(left + int(width * 0.44), top + int(height * 0.78), left + int(width * 0.82), top + int(height * 0.28))

        painter.restore()

    def editorEvent(self, event, model, option, index):
        del option
        if event.type() == QEvent.MouseButtonRelease:
            current = index.data(Qt.CheckStateRole)
            next_state = Qt.Unchecked if current == Qt.Checked else Qt.Checked
            return model.setData(index, next_state, Qt.CheckStateRole)
        if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Space, Qt.Key_Select):
            current = index.data(Qt.CheckStateRole)
            next_state = Qt.Unchecked if current == Qt.Checked else Qt.Checked
            return model.setData(index, next_state, Qt.CheckStateRole)
        return False


class AddedModelsTableModel(PandasTableModel):
    """Table model with a leading checkbox and optional row highlighting."""

    def __init__(self, data=None):
        super().__init__(data)
        self.highlight_mode = None

    def set_highlight_mode(self, mode):
        self.highlight_mode = mode
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))

    def flags(self, index):
        flags = super().flags(index)
        if not index.isValid():
            return flags
        if index.column() == 0 and self._is_row_selectable(index.row()):
            return flags | Qt.ItemIsUserCheckable | Qt.ItemIsEditable
        return flags

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if index.column() == 0:
            if role == Qt.CheckStateRole:
                value = bool(self._data.iat[index.row(), index.column()])
                return Qt.Checked if value else Qt.Unchecked
            if role == Qt.TextAlignmentRole:
                return Qt.AlignCenter
            if role in (Qt.DisplayRole, Qt.ToolTipRole):
                return ""
            return None

        if role == Qt.BackgroundRole and self.highlight_mode in {"trained", "external"}:
            row = self._data.iloc[index.row()]
            if self.highlight_mode == "trained" and str(row.get("trained", "No")) == "Yes":
                return QColor("#d9f2d9")
            if self.highlight_mode == "external" and str(row.get("external", "No")) == "Yes":
                return QColor("#d9f2d9")

        return super().data(index, role)

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid():
            return False

        if index.column() == 0 and role in (Qt.EditRole, Qt.CheckStateRole):
            if not self._is_row_selectable(index.row()):
                return False
            self._data.iat[index.row(), index.column()] = value == Qt.Checked if role == Qt.CheckStateRole else bool(value)
            self.dataChanged.emit(index, index)
            return True
        return super().setData(index, value, role)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal and section == 0:
            return ""
        return super().headerData(section, orientation, role)

    def _is_row_selectable(self, row):
        if row < 0 or row >= len(self._data.index):
            return False
        if "selectable" not in self._data.columns:
            return True
        return bool(self._data.iloc[row].get("selectable", True))


class QueueTableWidget(QTableWidget):
    order_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["name", "category", "algorithm", "trained"])
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDragDropOverwriteMode(False)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)

    def set_rows(self, rows):
        self.setRowCount(0)
        for row_data in rows:
            row = self.rowCount()
            self.insertRow(row)
            for col, key in enumerate(["name", "category", "algorithm", "trained"]):
                item = QTableWidgetItem(str(row_data.get(key, "")))
                if col == 0:
                    item.setData(Qt.UserRole, row_data.get("name", ""))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.setItem(row, col, item)

    def selected_names(self):
        names = []
        for index in self.selectionModel().selectedRows(0):
            item = self.item(index.row(), 0)
            if item is not None:
                names.append(str(item.data(Qt.UserRole) or item.text()))
        return names

    def ordered_names(self):
        names = []
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            if item is not None:
                names.append(str(item.data(Qt.UserRole) or item.text()))
        return names

    def dropEvent(self, event):
        if event.source() is not self:
            event.ignore()
            return

        selected_rows = sorted({index.row() for index in self.selectionModel().selectedRows()})
        if not selected_rows:
            event.ignore()
            return

        drop_row = self._drop_row_from_event(event)
        rows_payload = [self._row_payload(row) for row in selected_rows]

        for row in reversed(selected_rows):
            self.removeRow(row)

        removed_before_drop = sum(1 for row in selected_rows if row < drop_row)
        drop_row -= removed_before_drop

        for offset, payload in enumerate(rows_payload):
            target_row = drop_row + offset
            self.insertRow(target_row)
            for col, item in enumerate(payload):
                self.setItem(target_row, col, item)

        self.clearSelection()
        for offset in range(len(rows_payload)):
            self.selectRow(drop_row + offset)

        event.accept()
        self.order_changed.emit(self.ordered_names())

    def _drop_row_from_event(self, event):
        indicator = self.dropIndicatorPosition()
        if indicator == QAbstractItemView.OnViewport:
            return self.rowCount()

        point = event.position().toPoint()
        row = self.rowAt(point.y())
        if row < 0:
            return self.rowCount()
        if indicator == QAbstractItemView.BelowItem:
            return row + 1
        return row

    def _row_payload(self, row):
        payload = []
        for col in range(self.columnCount()):
            source = self.item(row, col)
            if source is None:
                payload.append(QTableWidgetItem(""))
                continue
            copied = QTableWidgetItem(source.text())
            copied.setData(Qt.UserRole, source.data(Qt.UserRole))
            copied.setFlags(source.flags())
            payload.append(copied)
        return payload


class AdvancedParametersDialog(QDialog):
    def __init__(self, specs, values=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Parameters")
        self.controls = {}
        self.values = dict(values or {})

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        layout.addLayout(form)

        for key, spec in specs.items():
            control = self._create_control(spec)
            value = self.values.get(key, spec.get("default"))
            self._set_control_value(control, value)
            self.controls[key] = control
            form.addRow(key.replace("_", " ").title(), control)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = secondary_button("Cancel")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = primary_button("Apply")
        ok_btn.clicked.connect(self.accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons)

    def _create_control(self, spec):
        kind = spec.get("type")
        if kind == "int":
            widget = QSpinBox()
            widget.setRange(spec.get("min", -2147483648), spec.get("max", 2147483647))
            widget.setSingleStep(spec.get("step", 1))
            return widget
        if kind == "float":
            widget = QDoubleSpinBox()
            widget.setRange(spec.get("min", -1e12), spec.get("max", 1e12))
            widget.setSingleStep(spec.get("step", 0.1))
            widget.setDecimals(spec.get("decimals", 4))
            return widget
        if kind == "choice":
            widget = taller_dropdown(QComboBox())
            widget.addItems([str(item) for item in spec.get("choices", [])])
            return widget
        if kind == "bool":
            return BorderedCheckBox()
        return QLineEdit()

    def _set_control_value(self, control, value):
        if isinstance(control, (QSpinBox, QDoubleSpinBox)) and value is not None:
            control.setValue(value)
        elif isinstance(control, QComboBox) and value is not None:
            control.setCurrentText(str(value))
        elif isinstance(control, QCheckBox):
            control.setChecked(bool(value))
        elif isinstance(control, QLineEdit):
            control.setText("" if value is None else str(value))

    def collected_values(self):
        output = {}
        for key, control in self.controls.items():
            if isinstance(control, (QSpinBox, QDoubleSpinBox)):
                output[key] = control.value()
            elif isinstance(control, QComboBox):
                output[key] = control.currentText()
            elif isinstance(control, QCheckBox):
                output[key] = control.isChecked()
            elif isinstance(control, QLineEdit):
                output[key] = control.text().strip()
        return output


class UnifiedModelSidebar(QWidget):
    add_model_requested = Signal(dict)
    import_external_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._editing_original_name = None
        self.required_controls = {}
        self.advanced_specs = {}
        self.advanced_values = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 28, 26, 22)
        layout.setSpacing(9)

        brand = QLabel("Classify & Learn Lab")
        brand.setObjectName("brand")
        layout.addWidget(brand)
        layout.addSpacing(26)
        layout.addWidget(section_label("MODEL BUILDER"))

        self.category_combo = taller_dropdown(QComboBox())
        self.category_combo.addItems(ModelController.category_options())
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        layout.addWidget(QLabel("Category"))
        layout.addWidget(self.category_combo)

        self.algorithm_combo = taller_dropdown(QComboBox())
        self.algorithm_combo.currentTextChanged.connect(self._on_algorithm_changed)
        layout.addWidget(QLabel("Model"))
        layout.addWidget(self.algorithm_combo)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Model name")
        layout.addWidget(QLabel("Saved Name"))
        layout.addWidget(self.name_input)

        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Label column")
        self.label_input.setReadOnly(True)
        layout.addWidget(QLabel("Label"))
        layout.addWidget(self.label_input)

        layout.addWidget(divider())
        layout.addWidget(section_label("COMMON PARAMETERS"))
        self.common_form = QFormLayout()
        self.common_form.setContentsMargins(0, 0, 0, 0)
        self.common_controls = self._build_controls(COMMON_TRAINING_PARAMETERS)
        for key, control in self.common_controls.items():
            self.common_form.addRow(key.replace("_", " ").title(), control)
        layout.addLayout(self.common_form)

        layout.addWidget(divider())
        layout.addWidget(section_label("REQUIRED PARAMETERS"))
        self.required_form = QFormLayout()
        self.required_form.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.required_form)

        self.show_advanced = QCheckBox("Use Advanced Parameters")
        self.advanced_button = secondary_button("Configure Advanced Parameters")
        self.advanced_button.clicked.connect(self.open_advanced_dialog)
        layout.addWidget(self.advanced_button)

        self.submit_button = primary_button("Add Model")
        self.submit_button.clicked.connect(self._submit)
        layout.addWidget(self.submit_button)

        clear_button = secondary_button("Clear Form")
        clear_button.clicked.connect(self.reset_form)
        layout.addWidget(clear_button)
        import_external = secondary_button("Import External PKL")
        import_external.clicked.connect(self.import_external_requested.emit)
        layout.addWidget(import_external)
        layout.addStretch()

        self._on_category_changed(self.category_combo.currentText())

    def _build_controls(self, schema):
        controls = {}
        for key, spec in schema.items():
            controls[key] = self._create_control(spec)
        return controls

    def _create_control(self, spec):
        control_type = spec.get("type")
        if control_type == "int":
            control = QSpinBox()
            control.setRange(spec.get("min", -2147483648), spec.get("max", 2147483647))
            control.setSingleStep(spec.get("step", 1))
            control.setValue(int(spec.get("default", 0)))
            return control
        if control_type == "float":
            control = QDoubleSpinBox()
            control.setRange(spec.get("min", -1e12), spec.get("max", 1e12))
            control.setSingleStep(spec.get("step", 0.1))
            control.setDecimals(spec.get("decimals", 4))
            control.setValue(float(spec.get("default", 0.0)))
            return control
        if control_type == "choice":
            control = taller_dropdown(QComboBox())
            choices = [str(item) for item in spec.get("choices", [])]
            control.addItems(choices)
            default = str(spec.get("default", choices[0] if choices else ""))
            if default:
                control.setCurrentText(default)
            return control
        if control_type == "bool":
            control = BorderedCheckBox()
            control.setChecked(bool(spec.get("default", False)))
            return control
        control = QLineEdit()
        control.setText(str(spec.get("default", "")))
        return control

    def _clear_form_layout(self, form):
        while form.rowCount() > 0:
            form.removeRow(0)

    def _on_category_changed(self, category):
        algorithms = ModelController.algorithms_for_category(category)
        self.algorithm_combo.blockSignals(True)
        self.algorithm_combo.clear()
        self.algorithm_combo.addItems(algorithms)
        self.algorithm_combo.blockSignals(False)
        if algorithms:
            self.algorithm_combo.setCurrentIndex(0)
            self._on_algorithm_changed(algorithms[0])

    def _on_algorithm_changed(self, algorithm):
        category = self.category_combo.currentText()
        definition = ModelController.get_definition(category, algorithm)
        self._clear_form_layout(self.required_form)
        self.required_controls = self._build_controls(definition.get("required", {}))
        self.advanced_specs = definition.get("advanced", {})
        self.advanced_values = {
            key: spec.get("default")
            for key, spec in self.advanced_specs.items()
        }
        self.advanced_button.setEnabled(bool(self.advanced_specs))

        for key, control in self.required_controls.items():
            self.required_form.addRow(key.replace("_", " ").title(), control)

        base_name = definition.get("label", algorithm.replace("_", " ").title())
        if not self.name_input.text().strip() or self._editing_original_name is None:
            self.name_input.setText(base_name)

    def open_advanced_dialog(self):
        if not self.advanced_specs:
            QMessageBox.information(self, "Advanced Parameters", "No advanced parameters are available for this model.")
            return
        dialog = AdvancedParametersDialog(self.advanced_specs, self.advanced_values, self)
        if dialog.exec() == QDialog.Accepted:
            self.advanced_values = dialog.collected_values()

    def _collect_values(self, controls):
        values = {}
        for key, control in controls.items():
            if isinstance(control, (QSpinBox, QDoubleSpinBox)):
                values[key] = control.value()
            elif isinstance(control, QComboBox):
                values[key] = control.currentText()
            elif isinstance(control, QCheckBox):
                values[key] = control.isChecked()
            elif isinstance(control, QLineEdit):
                values[key] = control.text().strip()
        return values

    def _submit(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Model Name", "Enter a saved model name.")
            return
        payload = {
            "original_name": self._editing_original_name,
            "name": name,
            "label": self.label_input.text().strip(),
            "category": self.category_combo.currentText(),
            "algorithm": self.algorithm_combo.currentText(),
            "common_parameters": self._collect_values(self.common_controls),
            "required_parameters": self._collect_values(self.required_controls),
            "advanced_parameters": dict(self.advanced_values),
        }
        self.add_model_requested.emit(payload)

    def reset_form(self):
        self._editing_original_name = None
        self.submit_button.setText("Add Model")
        self.advanced_values = {
            key: spec.get("default")
            for key, spec in self.advanced_specs.items()
        }
        self.advanced_button.setEnabled(bool(self.advanced_specs))
        for key, spec in COMMON_TRAINING_PARAMETERS.items():
            control = self.common_controls[key]
            if isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.setValue(spec.get("default", 0))
            elif isinstance(control, QCheckBox):
                control.setChecked(bool(spec.get("default", False)))
        self._on_category_changed(self.category_combo.currentText())

    def set_project_label(self, label):
        self.label_input.setText(label or "")

    def load_for_edit(self, entry):
        self._editing_original_name = entry.get("name")
        self.submit_button.setText("Update Model")
        self.name_input.setText(entry.get("name", ""))
        self.label_input.setText(entry.get("label", ""))
        self.category_combo.setCurrentText(entry.get("category", "supervised"))
        self.algorithm_combo.setCurrentText(entry.get("algorithm", ""))

        for key, value in entry.get("common_parameters", {}).items():
            control = self.common_controls.get(key)
            if control is None:
                continue
            if isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.setValue(value)
            elif isinstance(control, QCheckBox):
                control.setChecked(bool(value))

        for key, value in entry.get("required_parameters", {}).items():
            control = self.required_controls.get(key)
            if control is None:
                continue
            if isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.setValue(value)
            elif isinstance(control, QComboBox):
                control.setCurrentText(str(value))
            elif isinstance(control, QCheckBox):
                control.setChecked(bool(value))
            elif isinstance(control, QLineEdit):
                control.setText(str(value))

        self.advanced_values = {
            key: spec.get("default")
            for key, spec in self.advanced_specs.items()
        }
        self.advanced_values.update(entry.get("advanced_parameters", {}))
        self.advanced_button.setEnabled(bool(self.advanced_specs))


class UnifiedModelPage(QWidget):
    model_action_requested = Signal(str, str)
    queue_add_requested = Signal(list)
    queue_remove_requested = Signal(list)
    queue_reordered = Signal(list)
    train_queue_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.added_model = AddedModelsTableModel()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(section_label("ADDED MODELS"))
        self.added_table = QTableView()
        self.added_table.setModel(self.added_model)
        self.added_table.setSelectionBehavior(QTableView.SelectRows)
        self.added_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.added_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.added_table.customContextMenuRequested.connect(self._show_added_context_menu)
        self.added_table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.added_table.horizontalHeader().customContextMenuRequested.connect(self._show_header_context_menu)
        self.added_table.horizontalHeader().setStretchLastSection(True)
        self.added_table.verticalHeader().setVisible(True)
        self.added_table.setAlternatingRowColors(True)
        self.added_table.setItemDelegateForColumn(0, AddedModelCheckDelegate(self.added_table))
        layout.addWidget(self.added_table, 2)

        added_buttons = QHBoxLayout()
        self.add_to_queue = primary_button("Add Selected To Queue")
        self.add_to_queue.clicked.connect(self._on_add_to_queue)
        added_buttons.addWidget(self.add_to_queue)
        added_buttons.addStretch()
        layout.addLayout(added_buttons)

        layout.addWidget(divider())
        layout.addWidget(section_label("MODEL QUEUE"))
        self.queue_table = QueueTableWidget()
        self.queue_table.order_changed.connect(self.queue_reordered.emit)
        layout.addWidget(self.queue_table, 1)

        queue_buttons = QHBoxLayout()
        self.remove_from_queue = secondary_button("Remove")
        self.remove_from_queue.clicked.connect(self._on_remove_from_queue)
        self.train_queue_button = primary_button("Train Queue")
        self.train_queue_button.clicked.connect(self.train_queue_requested.emit)

        queue_buttons.addWidget(self.remove_from_queue)
        queue_buttons.addStretch()
        queue_buttons.addWidget(self.train_queue_button)
        layout.addLayout(queue_buttons)

    def set_added_models(self, models):
        rows = []
        for model in models:
            is_trained = bool(model.get("trained"))
            is_external = bool(model.get("externally_added"))
            editable_external = bool(model.get("editable_external", True))
            selectable = (not is_trained) and (not is_external or editable_external)
            rows.append({
                "": selectable,
                "name": model.get("name", ""),
                "category": model.get("category", ""),
                "algorithm": model.get("algorithm", ""),
                "label": model.get("label", ""),
                "trained": "Yes" if is_trained else "No",
                "external": "Yes" if is_external else "No",
                "selectable": selectable,
            })
        self.added_model.set_data(pd.DataFrame(rows, columns=["", "name", "category", "algorithm", "label", "trained", "external", "selectable"]))
        selectable_index = self.added_model._data.columns.get_loc("selectable")
        self.added_table.setColumnHidden(selectable_index, True)
        if self.added_model._data.shape[1] > 0:
            self.added_table.setColumnWidth(0, 34)

    def set_queue(self, rows):
        self.queue_table.set_rows(rows)

    def set_training(self, training):
        self.train_queue_button.setEnabled(not training)
        self.add_to_queue.setEnabled(not training)
        self.remove_from_queue.setEnabled(not training)
        self.train_queue_button.setText("Training..." if training else "Train Queue")

    def _selected_names(self, table, model=None):
        names = []
        if model is self.added_model:
            if "" in model._data.columns:
                checked = model._data[model._data[""] == True]  # noqa: E712
                names = [str(name) for name in checked["name"].tolist()]
            if not names:
                selection = table.selectionModel().selectedRows()
                for index in selection:
                    names.append(model._data.iloc[index.row()]["name"])
            return names

        return self.queue_table.selected_names()

    def _on_add_to_queue(self):
        names = self._selected_names(self.added_table, self.added_model)
        self.queue_add_requested.emit(names)

    def _on_remove_from_queue(self):
        names = self._selected_names(self.queue_table)
        self.queue_remove_requested.emit(names)

    def _show_added_context_menu(self, position):
        indexes = self.added_table.selectionModel().selectedRows()
        if not indexes:
            return
        name = self.added_model._data.iloc[indexes[0].row()]["name"]
        menu = QMenu(self)
        menu.addAction("Inspect", lambda: self.model_action_requested.emit("inspect", name))
        menu.addAction("Edit", lambda: self.model_action_requested.emit("edit", name))
        menu.addAction("Duplicate", lambda: self.model_action_requested.emit("duplicate", name))
        menu.addAction("Delete", lambda: self.model_action_requested.emit("delete", name))
        menu.addSeparator()
        menu.addAction("Export Config JSON", lambda: self.model_action_requested.emit("export_json", name))
        menu.addAction("Export PKL", lambda: self.model_action_requested.emit("export_pkl", name))
        menu.exec(self.added_table.viewport().mapToGlobal(position))

    def _show_header_context_menu(self, pos):
        if self.added_model._data.empty:
            return
        index = self.added_table.horizontalHeader().logicalIndexAt(pos)
        if index < 0:
            return
        column = self.added_model._data.columns[index]
        menu = QMenu(self)
        if column == "trained":
            menu.addAction("Highlight Trained Algorithms", self.highlight_trained_algorithms)
            menu.addAction("Clear Highlight", self.clear_highlight)
            menu.exec(self.added_table.horizontalHeader().mapToGlobal(pos))
            return
        if column == "external":
            menu.addAction("Highlight External = Yes", self.highlight_external_models)
            menu.addAction("Clear Highlight", self.clear_highlight)
            menu.exec(self.added_table.horizontalHeader().mapToGlobal(pos))

    def highlight_trained_algorithms(self):
        self.added_model.set_highlight_mode("trained")

    def highlight_external_models(self):
        self.added_model.set_highlight_mode("external")

    def clear_highlight(self):
        self.added_model.set_highlight_mode(None)
