import pandas as pd
from PySide6.QtWidgets import QApplication

import src.frontend.main_window as main_window
from src.frontend.charts import ChartCanvas
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
        saved["project"] = project
        saved["target_path"] = target_path
        saved["feature_df"] = feature_df
    monkeypatch.setattr(window, "_choose_project_save_path", fake_choose_project_save_path)
    monkeypatch.setattr(main_window, "save_project", fake_save_project)

    window.save_project_as()

    assert saved["target_path"] == str(tmp_path / "copy.icp")
    assert saved["project"]["project_name"] == "copy"
    assert window.current_project_path == tmp_path / "original.icp"
    assert window.project["project_name"] == "Original"


def test_change_column_type_refreshes_feature_and_analysis_pickers():
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"num": [1, 2], "label": ["a", "b"]})
    window.working_df = window.og_df.copy()
    window.columns = list(window.og_df.columns)
    window.populate_column_controls()

    window.change_column_type("num", "string")

    assert list(window.feature_numeric_picker.checkboxes) == []
    assert "num" in window.feature_non_numeric_picker.checkboxes


def test_refresh_column_pickers_uses_selected_columns_only():
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "label": ["x", "y"]})
    window.working_df = window.og_df.copy()
    window.columns = list(window.og_df.columns)
    window.populate_column_controls()

    window.column_picker.set_selected(["a", "label"])
    window.label_combo.setCurrentText("label")
    window.refresh_column_pickers()

    assert list(window.feature_numeric_picker.checkboxes) == ["a"]
    assert list(window.analysis_numeric_picker.checkboxes) == ["a"]
    assert list(window.feature_non_numeric_picker.checkboxes) == ["label"]
    assert list(window.analysis_non_numeric_picker.checkboxes) == ["label"]


def test_revert_dtype_conversion_restores_integer_dtype_after_string_conversion():
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"a": [1, 2, 3]})
    window.working_df = window.og_df.copy()
    window.original_dtypes = {"a": "int64"}
    window.columns = ["a"]

    window.change_column_type("a", "string")
    window.revert_dtype_conversion(["a"])

    assert pd.api.types.is_integer_dtype(window.working_df["a"].dtype)
    assert window.working_df["a"].tolist() == [1, 2, 3]


def test_revert_dtype_conversion_restores_float_dtype_and_imputed_values():
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"a": [1.5, None, 3.25]})
    window.working_df = window.og_df.copy()
    window.original_dtypes = {"a": "float64"}
    window.columns = ["a"]

    window.change_column_type("a", "int64")
    window.change_column_type("a", "string")
    window.working_df.loc[1, "a"] = "2"
    window.revert_dtype_conversion(["a"])

    assert pd.api.types.is_float_dtype(window.working_df["a"].dtype)
    assert window.working_df["a"].tolist() == [1.5, 2.0, 3.25]


def test_refresh_import_tables_updates_label_combo_from_working_dataframe():
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"a": [1], "b": [2]})
    window.working_df = pd.DataFrame({"a": [1]})
    window.columns = ["a", "b"]
    window.populate_column_controls()

    window.working_df = pd.DataFrame({"b": [2]})
    window.refresh_import_tables()

    assert "b" in [window.label_combo.itemText(i) for i in range(window.label_combo.count())]
    assert "a" not in [window.label_combo.itemText(i) for i in range(window.label_combo.count())]


def test_feature_pickers_include_label_and_raw_only_columns_when_using_raw_data():
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"num": [1, 2], "label": ["a", "b"], "raw_only": [10, 20]})
    window.working_df = pd.DataFrame({"num": [1, 2], "label": ["a", "b"]})
    window.columns = list(window.og_df.columns)
    window.populate_column_controls()
    window.label_combo.setCurrentText("label")

    window.on_feature_use_raw_toggled(True)

    assert "label" in window.feature_non_numeric_picker.checkboxes
    assert "raw_only" in window.feature_numeric_picker.checkboxes


def test_feature_raw_toggle_shows_raw_columns_not_selected_in_import_picker():
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"num": [1, 2], "raw_only": [10, 20], "label": ["a", "b"]})
    window.working_df = pd.DataFrame({"num": [1, 2], "label": ["a", "b"]})
    window.columns = list(window.og_df.columns)
    window.populate_column_controls()

    window.column_picker.set_selected(["num", "label"])
    window.refresh_column_pickers()
    assert "raw_only" not in window.feature_numeric_picker.checkboxes

    window.on_feature_use_raw_toggled(True)

    assert "raw_only" in window.feature_numeric_picker.checkboxes
    assert "raw_only" in window.feature_numeric_picker.selected_items()


def test_analysis_raw_toggle_shows_raw_columns_not_selected_in_import_picker():
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"num": [1, 2], "raw_only": [10, 20], "label": ["a", "b"]})
    window.working_df = pd.DataFrame({"num": [1, 2], "label": ["a", "b"]})
    window.columns = list(window.og_df.columns)
    window.populate_column_controls()

    window.column_picker.set_selected(["num", "label"])
    window.refresh_column_pickers()
    assert "raw_only" not in window.analysis_numeric_picker.checkboxes

    window.on_analysis_use_raw_toggled(True)

    assert "raw_only" in window.analysis_numeric_picker.checkboxes


def test_feature_results_refresh_when_toggling_raw_dataset():
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"x": [100, 200], "label": ["a", "b"]})
    window.working_df = pd.DataFrame({"x": [1, 2], "label": ["a", "b"]})
    window.columns = list(window.og_df.columns)
    window.populate_column_controls()
    window.feature_numeric_picker.set_selected(["x"])
    window.feature_picker.set_selected(["mean"])

    window.extract_features()
    working_mean = float(window.feature_df.loc[window.feature_df["signal"] == "x", "mean"].iloc[0])

    window.on_feature_use_raw_toggled(True)
    raw_mean = float(window.feature_df.loc[window.feature_df["signal"] == "x", "mean"].iloc[0])

    assert working_mean == 1.5
    assert raw_mean == 150.0


def test_extract_features_uses_checkbox_state_when_flag_is_stale():
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"x": [100, 200], "label": ["a", "b"]})
    window.working_df = pd.DataFrame({"x": [1, 2], "label": ["a", "b"]})
    window.columns = list(window.og_df.columns)
    window.populate_column_controls()
    window.feature_numeric_picker.set_selected(["x"])
    window.feature_picker.set_selected(["mean"])

    # Simulate stale internal flag; extraction should still follow checkbox state.
    window.feature_use_raw = False
    window.feature_use_raw_checkbox.setChecked(True)
    window.extract_features()

    raw_mean = float(window.feature_df.loc[window.feature_df["signal"] == "x", "mean"].iloc[0])
    assert raw_mean == 150.0


def test_analysis_non_numeric_option_is_disabled_for_mutual_information_and_pca():
    _qt_app()
    window = AnalyticsWindow()

    window.analysis_non_numeric_checkbox.setChecked(True)
    window.on_analysis_type_changed("Mutual Information")

    assert window.analysis_non_numeric_checkbox.isEnabled() is False
    assert window.analysis_non_numeric_checkbox.isHidden() is True
    assert window.analysis_non_numeric_checkbox.isChecked() is False
    assert window.analysis_matrix_type == "Numeric"
    assert window.analysis_cmap_combo.isHidden() is True

    window.on_analysis_type_changed("PCA")

    assert window.analysis_non_numeric_checkbox.isEnabled() is False
    assert window.analysis_matrix_type == "Numeric"
    assert window.analysis_cmap_combo.isHidden() is True

    window.on_analysis_type_changed("Correlation")

    assert window.analysis_non_numeric_checkbox.isEnabled() is True
    assert window.analysis_non_numeric_checkbox.isHidden() is False
    assert window.analysis_cmap_combo.isHidden() is False


def test_analysis_cmap_change_reruns_correlation_heatmap(monkeypatch):
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    window.working_df = window.og_df.copy()
    window.columns = list(window.og_df.columns)
    window.populate_column_controls()

    calls = []
    monkeypatch.setattr(window.analysis_chart, "plot_correlation_heatmap", lambda df, cmap=None: calls.append(cmap))

    window.show_correlation()
    window.on_analysis_cmap_changed("plasma")

    assert calls[-1] == "plasma"
    assert window.latest_correlation_matrix.equals(window.analysis_model._data)


def test_analysis_cmap_change_updates_rendered_heatmap_cmap():
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    window.working_df = window.og_df.copy()
    window.columns = list(window.og_df.columns)
    window.populate_column_controls()

    window.show_correlation()
    window.on_analysis_cmap_changed("plasma")

    axis = window.analysis_chart.figure.axes[0]
    assert axis.images[0].get_cmap().name == "plasma"


def test_3d_scatter_requires_three_distinct_numeric_columns():
    canvas = ChartCanvas(empty_message="", min_height=10)
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "label": ["a", "b"]})

    canvas.plot(df, "3D Scatter", "x", "x", "y")

    axis = canvas.figure.axes[0]
    assert axis.axison is False


def test_3d_scatter_handles_label_equal_to_axis_column_without_series_truth_error():
    canvas = ChartCanvas(empty_message="", min_height=10)
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4], "z": [3, 4, 5]})

    canvas.plot(df, "3D Scatter", "x", "y", "x", z_col="z")

    axis = canvas.figure.axes[0]
    assert len(axis.collections) >= 1


def test_pca_ignores_selected_label_column(monkeypatch):
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"label": [0, 1, 0], "x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    window.working_df = window.og_df.copy()
    window.columns = list(window.og_df.columns)
    window.populate_column_controls()
    window.label_combo.setCurrentText("label")
    window.analysis_type_combo.setCurrentText("PCA")

    calls = {}
    monkeypatch.setattr(main_window, "pca_analysis", lambda df, features, label, n_components=2: calls.update({"features": features, "label": label}) or {"pca_df": pd.DataFrame({"PC1": [0.1], "PC2": [0.2]}), "explained_variance_sum": 0.9})

    window.show_pca()

    assert calls["label"] is None
    assert calls["features"] == ["x", "y"]


def test_line_chart_x_options_include_string_and_date_columns():
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame(
        {
            "event_date": ["2024-01-01", "2024-01-02"],
            "category": ["start", "stop"],
            "value": [1, 2],
        }
    )
    window.working_df = window.og_df.copy()
    window.columns = list(window.og_df.columns)
    window.populate_column_controls()
    window.chart_type_combo.setCurrentText("Line")

    window.populate_visualization_controls()

    x_items = [window.chart_x_combo.itemText(i) for i in range(window.chart_x_combo.count())]
    y_items = [window.chart_y_combo.itemText(i) for i in range(window.chart_y_combo.count())]
    assert "event_date" in x_items
    assert "category" in x_items
    assert y_items == ["value"]


def test_scatter_chart_options_are_numeric_only():
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"category": ["a", "b"], "event_date": ["2024-01-01", "2024-01-02"], "value": [1, 2]})
    window.working_df = window.og_df.copy()
    window.columns = list(window.og_df.columns)
    window.populate_column_controls()
    window.chart_type_combo.setCurrentText("Scatter")

    x_items = [window.chart_x_combo.itemText(i) for i in range(window.chart_x_combo.count())]
    y_items = [window.chart_y_combo.itemText(i) for i in range(window.chart_y_combo.count())]

    assert x_items == ["value"]
    assert y_items == ["value"]


def test_chart_type_change_repopulates_visualization_columns():
    _qt_app()
    window = AnalyticsWindow()
    window.og_df = pd.DataFrame({"category": ["a", "b"], "value": [1, 2]})
    window.working_df = window.og_df.copy()
    window.columns = list(window.og_df.columns)
    window.populate_column_controls()

    window.chart_type_combo.setCurrentText("Histogram")
    histogram_x_items = [window.chart_x_combo.itemText(i) for i in range(window.chart_x_combo.count())]
    window.chart_type_combo.setCurrentText("Bar Chart")
    bar_x_items = [window.chart_x_combo.itemText(i) for i in range(window.chart_x_combo.count())]

    assert histogram_x_items == ["value"]
    assert "category" in bar_x_items


def test_chart_canvas_line_accepts_string_dates_for_x_axis():
    canvas = ChartCanvas(empty_message="", min_height=10)
    df = pd.DataFrame({"event_date": ["2024-01-01", "2024-01-02"], "value": [1, 2]})

    canvas.plot(df, "Line", "event_date", "value")

    axis = canvas.figure.axes[0]
    assert len(axis.lines) == 1
    assert axis.get_xlabel() == "event_date"
