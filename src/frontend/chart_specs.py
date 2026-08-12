from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ChartSpec:
    name: str
    renderer: str
    x_options: str = "all"
    y_options: str = "all"
    z_options: str = "all"
    needs_x: bool = False
    needs_y: bool = False
    needs_z: bool = False
    needs_bins: bool = False
    needs_multi: bool = False
    needs_lines: bool = False


_SPECS = (
    ChartSpec(
        name="Histogram",
        renderer="histogram",
        x_options="numeric",
        y_options="numeric",
        z_options="numeric",
        needs_x=True,
        needs_bins=True,
        needs_lines=True,
    ),
    ChartSpec(
        name="Scatter",
        renderer="scatter",
        x_options="numeric",
        y_options="numeric",
        z_options="numeric",
        needs_x=True,
        needs_y=True,
    ),
    ChartSpec(
        name="Line",
        renderer="line",
        x_options="line",
        y_options="numeric",
        z_options="none",
        needs_x=True,
        needs_y=True,
    ),
    ChartSpec(
        name="Box Plot",
        renderer="box_plot",
        x_options="numeric",
        y_options="numeric",
        z_options="numeric",
        needs_x=True,
        needs_lines=True,
    ),
    ChartSpec(
        name="Bar Chart",
        renderer="bar_chart",
        x_options="categorical_or_all",
        y_options="numeric_or_all",
        z_options="none",
        needs_x=True,
        needs_y=True,
    ),
    ChartSpec(
        name="Grouped Box Plot",
        renderer="grouped_box_plot",
        x_options="categorical_or_all",
        y_options="numeric_or_all",
        z_options="none",
        needs_x=True,
        needs_y=True,
    ),
    ChartSpec(
        name="Class Separation",
        renderer="class_separation",
        x_options="numeric",
        y_options="numeric",
        z_options="numeric",
        needs_x=True,
        needs_y=True,
    ),
    ChartSpec(
        name="3D Scatter",
        renderer="scatter_3d",
        x_options="numeric",
        y_options="numeric",
        z_options="numeric",
        needs_x=True,
        needs_y=True,
        needs_z=True,
    ),
    ChartSpec(
        name="Time Series (All Signals)",
        renderer="time_series",
        needs_multi=True,
    ),
    ChartSpec(
        name="Feature Distribution Comparison",
        renderer="distribution_comparison",
        needs_multi=True,
    ),
)

CHART_SPECS = {spec.name: spec for spec in _SPECS}
CHART_TYPES = tuple(CHART_SPECS)


def get_chart_spec(chart_type: str) -> ChartSpec | None:
    return CHART_SPECS.get(chart_type)


def chart_column_options(
    chart_type: str,
    all_columns: Iterable,
    numeric_columns: Iterable,
    date_columns: Iterable,
    categorical_columns: Iterable,
) -> tuple[list, list, list]:
    all_options = list(all_columns)
    numeric_options = _unique(numeric_columns)
    date_options = list(date_columns)
    categorical_options = list(categorical_columns)
    spec = get_chart_spec(chart_type)

    if spec is None:
        return all_options, all_options.copy(), all_options.copy()

    context = {
        "all": all_options,
        "numeric": numeric_options,
        "line": _unique(
            [*date_options, *categorical_options, *all_options]
        ),
        "categorical_or_all": categorical_options or all_options,
        "numeric_or_all": numeric_options or all_options,
        "none": [],
    }
    return (
        list(context[spec.x_options]),
        list(context[spec.y_options]),
        list(context[spec.z_options]),
    )


def visualization_validation_error(
    df,
    chart_type,
    x_column,
    y_column,
):
    numeric = lambda column: (
        column in df.columns
        and pd.api.types.is_numeric_dtype(df[column])
    )
    if chart_type in ("Histogram", "Box Plot") and not numeric(x_column):
        return f"{chart_type} requires one numeric primary column."
    if chart_type == "Scatter" and not (
        numeric(x_column) and numeric(y_column)
    ):
        return "Scatter plots require numeric X and Y columns."
    if chart_type == "Line":
        if x_column not in df.columns:
            return "Line charts require an X column."
        if not (
            numeric(x_column)
            or pd.api.types.is_datetime64_any_dtype(df[x_column])
        ):
            return "Line charts require a numeric or datetime X column."
        if not numeric(y_column):
            return "Line charts require a numeric Y column."
    if chart_type == "Bar Chart" and (
        x_column not in df.columns or not numeric(y_column)
    ):
        return (
            "Bar charts require a categorical X column and numeric Y column."
        )
    if chart_type == "Grouped Box Plot" and (
        x_column not in df.columns
        or numeric(x_column)
        or not numeric(y_column)
    ):
        return (
            "Grouped box plots require a categorical X column and numeric "
            "Y column."
        )
    return None


def _unique(values: Iterable) -> list:
    return list(dict.fromkeys(values))
