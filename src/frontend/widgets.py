import html

from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QStyledItemDelegate
)
from PySide6.QtCore import Qt, Signal

SIDEBAR_WIDTH = 320
DROPDOWN_MIN_HEIGHT = 32


'''
    Contains reusable frontend widgets and helper functions used throughout the application
'''
class ColumnPicker(QWidget):
    """Searchable checkbox list for column/feature selection."""

    selectionChange = Signal()

    def __init__(self, empty_text, rich_feature_labels=False):
        super().__init__()
        self.checkboxes = {}
        self.item_widgets = {}
        self.rich_feature_labels = rich_feature_labels

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.search = QLineEdit()
        self.search.setMinimumWidth(210)
        self.search.setPlaceholderText(empty_text)
        self.search.textChanged.connect(self.filter_items)
        layout.addWidget(self.search)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.scroll.setFixedWidth(SIDEBAR_WIDTH - 52)
        self.list_widget = QWidget()
        self.list_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(4)
        self.list_layout.addStretch()
        self.scroll.setWidget(self.list_widget)
        layout.addWidget(self.scroll)

    def set_items(self, items, checked=True):
        """Replace the checkbox list while preserving the trailing layout stretch."""
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.checkboxes = {}
        self.item_widgets = {}
        widest_text = 0
        for item in items:
            item_name = str(item)
            checkbox = QCheckBox(item_name)
            display_widget = checkbox

            if self.rich_feature_labels and " [" in item_name and item_name.endswith("]"):
                feature_name, applicability = item_name.split(" [", 1)
                applicability = f"[{applicability}"
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                checkbox = QCheckBox("")
                label = QLabel(
                    f"<b>{html.escape(feature_name)}</b> {html.escape(applicability)}"
                )
                label.setTextFormat(Qt.RichText)
                label.setToolTip(item_name)
                row_layout.addWidget(checkbox)
                row_layout.addWidget(label)
                row_layout.addStretch()
                display_widget = row

                def _toggle_label(_event, box=checkbox):
                    box.setChecked(not box.isChecked())

                label.mousePressEvent = _toggle_label

            checkbox.setChecked(checked)
            checkbox.setToolTip(item_name)
            checkbox.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            checkbox.stateChanged.connect(self._selection_changed)

            self.checkboxes[item_name] = checkbox
            self.item_widgets[item_name] = display_widget
            self.list_layout.insertWidget(self.list_layout.count() - 1, display_widget)
            widest_text = max(widest_text, checkbox.fontMetrics().horizontalAdvance(item_name))

        content_width = max(self.search.minimumWidth(), widest_text + 56)
        self.list_widget.setMinimumWidth(content_width)
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self.filter_items("")

    def selected_items(self):
        return [
            name
            for name, checkbox in self.checkboxes.items()
            if checkbox.isChecked()
        ]

    def set_selected(self, selected):
        selected = set(selected)
        for name, checkbox in self.checkboxes.items():
            checkbox.setChecked(name in selected)

    def filter_items(self, text):
        needle = text.strip().lower()
        for name, checkbox in self.checkboxes.items():
            widget = self.item_widgets.get(name, checkbox)
            widget.setVisible(needle in name.lower())

    def _selection_changed(self):
        self.selectionChange.emit()


class WheelLockedComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class WheelLockedSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.UpDownArrows)

    def wheelEvent(self, event):
        event.ignore()


class WheelLockedDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.UpDownArrows)

    def wheelEvent(self, event):
        event.ignore()


def tab_row(parent, labels, callback, compact=False):
    """Create design-matched tab rows from QPushButtons."""
    layout = QVBoxLayout()
    buttons = []
    row = QHBoxLayout()
    row.setSpacing(34 if not compact else 22)

    group = QButtonGroup(parent)
    group.setExclusive(True)

    for index, label in enumerate(labels):
        # Escape ampersands so Qt does not treat them as keyboard mnemonics.
        button = QPushButton(label.replace("&", "&&"))
        button.setCheckable(True)
        button.setProperty("tabButton", True)
        button.setProperty("compactTab", compact)
        button.clicked.connect(lambda checked=False, i=index: callback(i))
        # Selected tabs become bold; reserve extra width to avoid clipping.
        text_width = button.fontMetrics().horizontalAdvance(label) + (40 if not compact else 28)
        button.setMinimumWidth(text_width)
        button.setMinimumHeight(42 if not compact else 34)
        button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        group.addButton(button, index)
        row.addWidget(button)
        buttons.append(button)

    row.addStretch()
    line = divider("tabLine")
    layout.addLayout(row)
    layout.addWidget(line)
    buttons[0].setChecked(True)
    return {"layout": layout, "buttons": buttons, "group": group, "line": line}


def sidebar_base():
    """Create the shared left rail shell."""
    panel = QWidget()
    panel.setProperty("sidebar", True)
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(26, 28, 26, 22)
    layout.setSpacing(9)

    title = QLabel("Classify & Learn Lab")
    title.setObjectName("brand")
    layout.addWidget(title)
    layout.addSpacing(26)
    return panel


def section_label(text):
    label = QLabel(text)
    label.setProperty("sectionTitle", True)
    return label


def divider(property_name="divider"):
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setProperty(property_name, True)
    return line


def primary_button(text):
    button = QPushButton(text)
    button.setProperty("primary", True)
    return button


def secondary_button(text):
    button = QPushButton(text)
    button.setProperty("secondary", True)
    return button


def taller_dropdown(combo):
    """Size native dropdowns without restyling arrows/focus states."""
    combo.setMinimumHeight(DROPDOWN_MIN_HEIGHT)
    combo.setMinimumWidth(210)
    return combo


def data_panel(title, model):
    """Create a titled table panel."""
    group = QWidget()
    layout = QVBoxLayout(group)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    # QLabel titles avoid QGroupBox clipping over table borders on macOS.
    title_label = QLabel(title)
    title_label.setProperty("panelTitle", True)
    layout.addWidget(title_label)
    layout.addWidget(table_view(model))
    return group


def table_view(model):
    """Create a standard read-only-feeling table view for DataFrame models."""
    table = QTableView()
    table.setModel(model)
    table.setAlternatingRowColors(True)
    table.setSortingEnabled(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    table.horizontalHeader().setStretchLastSection(True)
    table.verticalHeader().setDefaultSectionSize(24)
    table.setSelectionBehavior(QTableView.SelectRows)
    table.setWordWrap(False)
    return table

# Dtype conversion

def allowed_dtypes(current_dtype):

    current_dtype = str(current_dtype)

    mapping = {
        "int64": ["int64", "float64", "string"],
        "Int64": ["Int64", "float64", "string"],
        "float64": ["float64", "Int64", "string"],
        "string": ["string"],
        "object": ["string"],
        "boolean": ["boolean", "string"],
        "bool": ["bool", "boolean", "string"],
    }

    return mapping.get(
        current_dtype,
        [current_dtype]
    )

class DTypeDelegate(QStyledItemDelegate):
    
    DEFAULT_TYPES = [
        "int64",
        "float64",
        "string",
        "boolean"
    ]

    def __init__(self, parent=None, original_dtypes=None):
        super().__init__(parent)
        self.original_dtypes = original_dtypes or {}
        
    def createEditor(
        self,
        parent,
        option,
        index
    ):

        combo = QComboBox(parent)

        column_name = index.model()._data.iloc[index.row()]["column"]

        current_dtype = index.model()._data.iloc[index.row()]["dtype"]
        
        original_dtype = self.original_dtypes.get(column_name)

        choices = allowed_dtypes(current_dtype)

        if original_dtype and original_dtype not in choices:
            choices.insert(0, original_dtype)

        combo.addItems(choices)

        return combo

    def setEditorData(
        self,
        editor,
        index
    ):

        value = index.data()

        pos = editor.findText(value)

        if pos >= 0:
            editor.setCurrentIndex(pos)

    def setModelData(
        self,
        editor,
        model,
        index
    ):

        model.setData(
            index,
            editor.currentText(),
            Qt.EditRole
        )
