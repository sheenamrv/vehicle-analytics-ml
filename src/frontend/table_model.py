import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal

'''
    Provides a Qt table model that allows pandas DataFrames to be displayed inside the QTableView widgets
'''
class PandasTableModel(QAbstractTableModel):
    """Expose a pandas DataFrame through Qt's table model interface."""

    dtypeChanged = Signal(str, str)
    def __init__(self, data=None):
        super().__init__()
        self._data = pd.DataFrame() if data is None else data

    def set_data(self, data):
        """Replace the displayed DataFrame and notify any attached table views."""
        self.beginResetModel()
        self._data = pd.DataFrame() if data is None else data
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._data.index)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._data.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None

        value = self._data.iat[index.row(), index.column()]
        # Qt expects display values as strings; keep missing values visually blank.
        if pd.isna(value):
            return ""
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return str(self._data.columns[section])
        return str(self._data.index[section])

    def flags(self, index):
        flags = super().flags(index)

        if (len(self._data.columns) > 1 and self._data.columns[index.column()] == "dtype"):
            return flags | Qt.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.EditRole):

        if role != Qt.EditRole:
            return False

        column_name = self._data.columns[index.column()]

        if column_name != "dtype":
            return False

        row = index.row()

        actual_column = self._data.iloc[row]["column"]

        self._data.iat[row, index.column()] = value

        self.dtypeChanged.emit(
            actual_column,
            value
        )

        self.dataChanged.emit(index, index)

        return True