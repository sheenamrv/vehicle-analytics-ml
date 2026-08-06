from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from src.frontend import app_identity


def _qt_app():
    return QApplication.instance() or QApplication([])


def test_application_icon_resolves_in_source_checkout():
    _qt_app()
    icon_path = app_identity.resource_path("assets/icon.png")

    assert icon_path.is_file()
    assert not app_identity.application_icon().isNull()


def test_configure_qt_application_sets_shared_name_and_icon():
    application = _qt_app()

    app_identity.configure_qt_application(application)

    assert application.applicationName() == app_identity.APP_DISPLAY_NAME
    assert application.applicationDisplayName() == app_identity.APP_DISPLAY_NAME
    assert not application.windowIcon().isNull()


def test_configure_windows_app_identity_sets_explicit_process_id(monkeypatch):
    received_ids = []
    shell32 = SimpleNamespace(
        SetCurrentProcessExplicitAppUserModelID=received_ids.append
    )
    monkeypatch.setattr(app_identity.sys, "platform", "win32")
    monkeypatch.setattr(
        app_identity.ctypes,
        "windll",
        SimpleNamespace(shell32=shell32),
        raising=False,
    )

    app_identity.configure_windows_app_identity()

    assert received_ids == [app_identity.WINDOWS_APP_USER_MODEL_ID]
