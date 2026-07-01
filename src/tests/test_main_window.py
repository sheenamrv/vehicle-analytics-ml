import pandas as pd
from PySide6.QtWidgets import QApplication

import src.frontend.main_window as main_window
from src.frontend.main_window import AnalyticsWindow


def _qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_update_selected_columns_does_not_mark_dirty_during_programmatic_population(monkeypatch):
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"a": [1], "b": [2]})
    window.working_df = window.og_df.copy()

    monkeypatch.setattr(window, "refresh_import_tables", lambda: None)
    monkeypatch.setattr(window, "populate_visualization_controls", lambda: None)

    window._suppress_dirty = True
    window.update_selected_columns()

    assert window.is_dirty is False


def test_save_project_as_saves_copy_without_changing_active_project(monkeypatch, tmp_path):
    _qt_app()
    window = AnalyticsWindow()
    window.project = {"project_name": "Original", "file_path": "data.csv"}
    window.working_df = pd.DataFrame({"a": [1]})
    window.og_df = pd.DataFrame({"a": [1]})
    window.current_project_path = tmp_path / "original.icp"

    saved = {}

    def fake_choose_project_save_path(default_name=None):
        return tmp_path / "copy.icp"

    def fake_save_project(project, original_df, mod_df, target_path=None, feature_df=None):

    monkeypatch.setattr(window, "_choose_project_save_path", fake_choose_project_save_path)
    monkeypatch.setattr(main_window, "save_project", fake_save_project)

    window.save_project_as()

    assert saved["target_path"] == str(tmp_path / "copy.icp")
    assert saved["project"]["project_name"] == "copy"
    assert window.current_project_path == tmp_path / "original.icp"
    assert window.project["project_name"] == "Original"
