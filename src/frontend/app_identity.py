"""Shared application naming, icon, and Windows process identity."""

from pathlib import Path
import ctypes
import sys

from PySide6.QtGui import QIcon


APP_DISPLAY_NAME = "Classify & Learn Lab"
APP_ORGANIZATION_NAME = "Classify & Learn Lab"
WINDOWS_APP_USER_MODEL_ID = "ClassifyAndLearnLab.Desktop"


def resource_path(relative_path):
    """Resolve an asset in source checkouts and PyInstaller bundles."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    root = Path(bundle_root) if bundle_root else Path(__file__).resolve().parents[2]
    return root / relative_path


def application_icon():
    """Return the shared application icon."""
    return QIcon(str(resource_path("assets/icon.png")))


def configure_windows_app_identity():
    """Set the process identity Windows uses for taskbar grouping and shortcuts."""
    if sys.platform != "win32":
        return

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            WINDOWS_APP_USER_MODEL_ID
        )
    except (AttributeError, OSError):
        # Older or restricted Windows environments can omit this shell API.
        pass


def configure_qt_application(application):
    """Apply the shared identity to the QApplication instance."""
    application.setApplicationName(APP_DISPLAY_NAME)
    application.setApplicationDisplayName(APP_DISPLAY_NAME)
    application.setOrganizationName(APP_ORGANIZATION_NAME)
    application.setWindowIcon(application_icon())
