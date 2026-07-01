from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication


GREEN = "#36b66b"
DARK_GREEN = "#1f7f43"
TEXT = "#2f3237"
MUTED = "#6b7280"
BORDER = "#d7dbe0"
BG = "#fbfbfa"

'''
    Defines the application's global QT stylesheet and shared colour palette.
'''
def apply_app_styles(window):
    """Apply the app-wide Qt stylesheet.

    Dropdowns intentionally keep native styling; their dimensions are controlled
    in widgets.py so platform arrows and focus states remain readable.
    """
    QApplication.instance().setFont(QFont("Arial", 10))
    window.setStyleSheet(
        f"""
        QMainWindow, QWidget {{
            background: {BG};
            color: {TEXT};
        }}
        QMenuBar, QMenu {{
            background: white;
            color: {TEXT};
        }}
        QWidget[sidebar="true"] {{
            background: white;
            border-right: 1px solid #eceef1;
        }}
        QLabel#brand {{
            color: {DARK_GREEN};
            font-size: 17px;
            font-weight: 700;
        }}
        QLabel[sectionTitle="true"] {{
            color: {DARK_GREEN};
            font-size: 12px;
            font-weight: 700;
            margin-top: 8px;
        }}
        QLabel {{
            color: {TEXT};
            font-size: 12px;
        }}
        QLineEdit, QSpinBox {{
            min-height: 26px;
            border: 1px solid {BORDER};
            border-radius: 4px;
            background: #ffffff;
            padding: 2px 7px;
        }}
        QPushButton {{
            border: none;
            border-radius: 4px;
            min-height: 28px;
            padding: 5px 12px;
            font-weight: 600;
        }}
        QPushButton[primary="true"] {{
            background: {GREEN};
            color: white;
        }}
        QPushButton[secondary="true"] {{
            background: {DARK_GREEN};
            color: white;
        }}
        QPushButton[tabButton="true"] {{
            background: transparent;
            color: {TEXT};
            border-radius: 0;
            font-size: 15px;
            font-weight: 500;
            min-height: 42px;
            padding: 6px 0 12px 0;
        }}
        QPushButton[compactTab="true"] {{
            font-size: 12px;
            min-height: 34px;
            padding: 5px 0 10px 0;
        }}
        QPushButton[tabButton="true"]:checked {{
            color: {DARK_GREEN};
            border-bottom: 2px solid {GREEN};
            font-weight: 700;
        }}
        QFrame[tabLine="true"], QFrame[divider="true"] {{
            color: #7d8389;
            background: #7d8389;
            max-height: 1px;
            margin: 10px 0;
        }}
        QGroupBox[dataPanel="true"] {{
            border: none;
            color: {DARK_GREEN};
            font-size: 12px;
            font-weight: 700;
            margin-top: 8px;
        }}
        QLabel[panelTitle="true"] {{
            color: {DARK_GREEN};
            font-size: 12px;
            font-weight: 700;
            min-height: 18px;
            padding: 0;
            margin: 0;
        }}
        QTableView {{
            background: white;
            alternate-background-color: #f8faf9;
            gridline-color: #e4e7ea;
            border: 1px solid {BORDER};
            selection-background-color: #ccebd9;
            selection-color: {TEXT};
        }}
        QHeaderView::section {{
            background: #f1f4f2;
            border: 0;
            border-right: 1px solid #e1e5e8;
            border-bottom: 1px solid #dce1e5;
            padding: 5px;
            font-weight: 700;
        }}
        QCheckBox {{
            color: {MUTED};
            font-size: 11px;
            spacing: 7px;
        }}
        QSlider::groove:horizontal {{
            height: 5px;
            background: #c7c9cc;
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
            background: {GREEN};
        }}
        """
    )
