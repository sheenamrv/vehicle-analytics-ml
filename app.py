import sys

from PySide6.QtWidgets import QApplication

from src.frontend.app_identity import (
    configure_qt_application,
    configure_windows_app_identity,
)
from src.frontend.main_window import AnalyticsWindow


def main():
    configure_windows_app_identity()
    app = QApplication(sys.argv)
    configure_qt_application(app)
    window = AnalyticsWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
