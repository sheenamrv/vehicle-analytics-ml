import sys

from PySide6.QtWidgets import QApplication

from src.frontend.main_window import AnalyticsWindow


def main():
    app = QApplication(sys.argv)
    window = AnalyticsWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
