import pandas as pd

from src.frontend.chart_specs import (
    CHART_TYPES,
    chart_column_options,
    get_chart_spec,
    visualization_validation_error,
)


def test_chart_types_preserve_visualization_order():
    assert CHART_TYPES == (
        "Histogram",
        "Scatter",
        "Line",
        "Box Plot",
        "Bar Chart",
        "Grouped Box Plot",
        "Class Separation",
        "3D Scatter",
        "Time Series (All Signals)",
        "Feature Distribution Comparison",
    )


def test_chart_column_options_preserve_type_filtering():
    columns = ["event_date", "category", "value"]
    numeric = ["value"]
    dates = ["event_date"]
    categorical = ["category"]

    scatter = chart_column_options(
        "Scatter",
        columns,
        numeric,
        dates,
        categorical,
    )
    line = chart_column_options(
        "Line",
        columns,
        numeric,
        dates,
        categorical,
    )
    bar = chart_column_options(
        "Bar Chart",
        columns,
        numeric,
        dates,
        categorical,
    )

    assert scatter == (["value"], ["value"], ["value"])
    assert line == (
        ["event_date", "category", "value"],
        ["value"],
        [],
    )
    assert bar == (["category"], ["value"], [])


def test_chart_spec_describes_required_controls():
    scatter_3d = get_chart_spec("3D Scatter")
    multi_signal = get_chart_spec("Time Series (All Signals)")

    assert scatter_3d is not None
    assert scatter_3d.needs_x is True
    assert scatter_3d.needs_y is True
    assert scatter_3d.needs_z is True
    assert multi_signal is not None
    assert multi_signal.needs_multi is True


def test_visualization_validation_remains_ui_independent():
    df = pd.DataFrame(
        {
            "category": ["a", "b"],
            "x": [1, 2],
            "y": [3, 4],
        }
    )

    assert visualization_validation_error(
        df,
        "Scatter",
        "x",
        "y",
    ) is None
    assert visualization_validation_error(
        df,
        "Grouped Box Plot",
        "x",
        "y",
    ) == (
        "Grouped box plots require a categorical X column and numeric Y "
        "column."
    )
