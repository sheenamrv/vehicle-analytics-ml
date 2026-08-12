import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
)

from src.frontend.widgets import table_view


class QualityIssueTableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data.copy()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._data.index)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._data.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            value = self._data.iat[index.row(), index.column()]
            if pd.isna(value):
                return ""
            return str(value)

        if role == Qt.BackgroundRole:
            column_name = self._data.columns[index.column()]
            value = self._data.iat[index.row(), index.column()]
            issue_type = self._data.iloc[index.row()]["issue"]

            if (
                pd.isna(value)
                and column_name not in ("issue", "__original_index__")
            ):
                return QColor("#fff3c4")
            if "Duplicate" in str(issue_type):
                return QColor("#e8f4ff")

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return str(self._data.columns[section])
        return str(self._data.index[section])


class DataQualityDialog(QDialog):
    def __init__(self, parent, working_df):
        super().__init__(parent)
        self.setWindowTitle("Review Missing and Duplicate Rows")
        self.setMinimumSize(1000, 520)

        self.issue_df = self._build_issue_df(working_df)
        self.removed_indices = []

        layout = QVBoxLayout(self)

        if self.issue_df.empty:
            layout.addWidget(
                QLabel("No missing values or duplicate rows were found.")
            )
            close_button = QPushButton("Close")
            close_button.clicked.connect(self.reject)
            layout.addWidget(close_button)
            return

        self.issue_model = QualityIssueTableModel(self.issue_df)
        self.table = table_view(self.issue_model)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        layout.addWidget(self.table)

        button_layout = QHBoxLayout()
        remove_button = QPushButton("Remove Selected Rows")
        remove_button.clicked.connect(self.on_remove_selected)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(remove_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def _build_issue_df(self, df):
        if df.empty:
            return pd.DataFrame()

        missing = set(df[df.isna().any(axis=1)].index.tolist())
        duplicates = set(df[df.duplicated(keep=False)].index.tolist())
        issue_rows = []

        for row_index in sorted(missing.union(duplicates)):
            row = df.loc[row_index].copy()
            issue_labels = []
            if row_index in missing:
                issue_labels.append("Missing")
            if row_index in duplicates:
                issue_labels.append("Duplicate")
            row["issue"] = " & ".join(issue_labels)
            row["__original_index__"] = row_index
            issue_rows.append(row)

        if not issue_rows:
            return pd.DataFrame()

        issue_df = pd.DataFrame(issue_rows)
        columns = [
            "__original_index__",
            "issue",
            *[
                column
                for column in issue_df.columns
                if column not in ("__original_index__", "issue")
            ],
        ]
        return issue_df[columns]

    def on_remove_selected(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(
                self,
                self.windowTitle(),
                "Select at least one row to remove.",
            )
            return

        indices = sorted(
            {
                self.issue_df.iloc[index.row()]["__original_index__"]
                for index in selected_rows
            }
        )

        result = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Remove {len(indices)} selected row(s) from the working dataset?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if result == QMessageBox.StandardButton.Yes:
            self.removed_indices = indices
            self.accept()
