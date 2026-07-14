import os
import tempfile
from pathlib import Path

# Matplotlib wants writable config/cache directories even when embedded in Qt.
# Use temp locations so packaged or sandboxed desktop runs do not fail on import.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "vehicle_analytics_matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "vehicle_analytics_cache"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

'''
    Central location for every chart rendered by the application
'''
class ChartCanvas(FigureCanvasQTAgg):
    """Reusable matplotlib canvas for all frontend chart areas."""

    def __init__(self, empty_message="Open a dataset to visualize it.", min_height=360):
        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        super().__init__(self.figure)
        self.setMinimumHeight(min_height)
        self.empty_message = empty_message
        self.show_empty(empty_message)

    def show_empty(self, message):
        axis = self._single_axis()
        axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=12,
            color="#6b7280",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        self.draw()

    def plot(self, df, chart_type, x_col=None, y_col=None, label_col=None):
        """Route Visualization-tab chart selections to plotting methods."""
        if df.empty:
            self.show_empty("Open a dataset to visualize it.")
            return

        # Extension point: add new generic chart types to this dispatch.
        if chart_type == "Histogram":
            self._histogram(df, x_col)
        elif chart_type == "Scatter":
            self._scatter(df, x_col, y_col, label_col)
        elif chart_type == "Line":
            self._line(df, x_col, y_col)
        elif chart_type == "Box Plot":
            self._box_plot(df, x_col)
        elif chart_type == "Bar Chart":
            self._bar_chart(df, x_col, y_col)
        elif chart_type == "Grouped Box Plot":
            self._grouped_box_plot(df, x_col, y_col)
        else:
            self.show_empty("Choose a chart type.")

        self.draw()

    def plot_correlation_heatmap(self, df):
        """Draw correlation output produced for the Analysis tab."""
        if df.empty:
            self.show_empty("Open a dataset to analyze it.")
            return
        self._correlation_heatmap(df)
        self.draw()

    def plot_missing_values(self, df):
        """Render the Import-tab data quality chart."""
        if df.empty:
            self.show_empty("Open a dataset to profile missing values.")
            return
        self._missing_values(df)
        self.draw()

    def plot_pca_scatter(self, pca_df, label_col):
        """Draw PC1 vs PC2 from existing pca_analysis output."""
        if pca_df.empty or not self._has_columns(pca_df, ["PC1", "PC2"]):
            self.show_empty("Run PCA with at least two components.")
            return

        axis = self._single_axis()
        plot_df = pca_df[["PC1", "PC2"] + ([label_col] if label_col in pca_df.columns else [])].dropna()
        # Color by label only when the legend stays readable.
        if label_col in plot_df.columns and plot_df[label_col].nunique() <= 12:
            labels = plot_df[label_col].astype("category")
            scatter = axis.scatter(
                plot_df["PC1"],
                plot_df["PC2"],
                c=labels.cat.codes,
                cmap="viridis",
                alpha=0.82,
                s=38,
            )
            handles, _ = scatter.legend_elements()
            axis.legend(
                handles,
                labels.cat.categories.astype(str),
                title=label_col,
                loc="best",
                frameon=False,
            )
        else:
            axis.scatter(plot_df["PC1"], plot_df["PC2"], color="#36b66b", alpha=0.82, s=38)

        axis.set_title("PCA Scatter Plot")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        self._style_axis(axis)
        self.draw()

    def plot_mutual_information(self, scores_df):
        """Draw feature scores returned by mutual_information_analysis()."""
        if scores_df.empty or not self._has_columns(scores_df, ["feature", "mutual_information"]):
            self.show_empty("Run Mutual Information to see feature scores.")
            return

        # Limit bars so long feature lists still fit in the desktop panel.
        plot_df = scores_df.sort_values("mutual_information", ascending=True).tail(15)
        axis = self._single_axis()
        axis.barh(plot_df["feature"], plot_df["mutual_information"], color="#36b66b")
        axis.set_title("Mutual Information Scores")
        axis.set_xlabel("Score")
        axis.set_ylabel("Feature")
        self._style_axis(axis)
        self.draw()

    def _single_axis(self):
        """Reset the figure and return a fresh axis for single-chart canvases."""
        self.figure.clear()
        return self.figure.add_subplot(111)

    def _histogram(self, df, column):
        axis = self._single_axis()
        series = self._numeric_series(df, column)
        if series is None:
            self.show_empty("Choose a numeric column for the histogram.")
            return

        sns.histplot(series.dropna(), bins=24, color="#36b66b", edgecolor="white", kde=True, ax=axis)
        axis.axvline(series.mean(), color="#1f7f43", linestyle="--", linewidth=2)
        axis.set_title(f"Histogram: {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Count")
        self._style_axis(axis)

    def _scatter(self, df, x_col, y_col, label_col=None):
        axis = self._single_axis()
        if not self._has_columns(df, [x_col, y_col]):
            self.show_empty("Choose X and Y columns for the scatter plot.")
            return

        plot_df = df[[x_col, y_col] + ([label_col] if label_col in df.columns else [])].dropna()
        if plot_df.empty:
            self.show_empty("The selected columns do not contain plottable values.")
            return

        if label_col in plot_df.columns and plot_df[label_col].nunique() <= 12:
            labels = plot_df[label_col].astype("category")
            sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=label_col, palette="viridis", alpha=0.78, s=48, ax=axis)
        else:
            sns.scatterplot(data=plot_df, x=x_col, y=y_col, color="#36b66b", alpha=0.78, s=48, ax=axis)

        axis.set_title(f"{y_col} vs {x_col}")
        axis.set_xlabel(x_col)
        axis.set_ylabel(y_col)
        self._style_axis(axis)

    def _line(self, df, x_col, y_col):
        axis = self._single_axis()
        if not self._has_columns(df, [x_col, y_col]):
            self.show_empty("Choose X and Y columns for the line chart.")
            return

        # Cap row count to keep line rendering responsive on large datasets.
        plot_df = df[[x_col, y_col]].dropna().head(1000)
        if plot_df.empty:
            self.show_empty("The selected columns do not contain plottable values.")
            return

        sns.lineplot(data=plot_df, x=x_col, y=y_col, color="#1f7f43", linewidth=2, ax=axis)
        axis.set_title(f"{y_col} over {x_col}")
        axis.set_xlabel(x_col)
        axis.set_ylabel(y_col)
        self._style_axis(axis)

    def _box_plot(self, df, column):
        axis = self._single_axis()
        series = self._numeric_series(df, column)
        if series is None:
            self.show_empty("Choose a numeric column for the box plot.")
            return

        sns.boxplot(x=series.dropna(), color="#9ad6af", ax=axis)
        axis.set_title(f"Distribution: {column}")
        axis.set_xlabel(column)
        axis.set_yticks([])
        self._style_axis(axis)

    def _bar_chart(self, df, category_col, value_col=None):
        axis = self._single_axis()
        if category_col not in df.columns:
            self.show_empty("Choose a category column for the bar chart.")
            return

        # If a numeric Y column is selected, show grouped means; otherwise show
        # frequency counts for the selected category column.
        if value_col in df.columns and pd.api.types.is_numeric_dtype(df[value_col]):
            values = df.groupby(category_col, dropna=False)[value_col].mean().sort_values(ascending=False)
            ylabel = f"Mean {value_col}"
            title = f"Mean {value_col} by {category_col}"
        else:
            values = df[category_col].astype("string").fillna("Missing").value_counts().head(20)
            ylabel = "Count"
            title = f"Counts by {category_col}"

        if values.empty:
            self.show_empty("The selected column does not contain plottable values.")
            return

        # Keep categorical labels readable; show top groups only.
        values = values.head(20)
        sns.barplot(x=values.index.astype(str), y=values.values, color="#36b66b", ax=axis)
        axis.set_title(title)
        axis.set_xlabel(category_col)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
        self._style_axis(axis)

    def _grouped_box_plot(self, df, group_col, value_col):
        axis = self._single_axis()
        if not self._has_columns(df, [group_col, value_col]):
            self.show_empty("Choose a group column and a numeric value column.")
            return
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            self.show_empty("Grouped box plot needs a numeric Y column.")
            return

        plot_df = df[[group_col, value_col]].dropna()
        if plot_df.empty:
            self.show_empty("The selected columns do not contain plottable values.")
            return

        top_groups = plot_df[group_col].value_counts().head(12).index
        sns.boxplot(data=plot_df[plot_df[group_col].isin(top_groups)], x=group_col, y=value_col, color="#9ad6af", ax=axis)
        axis.set_title(f"{value_col} by {group_col}")
        axis.set_xlabel(group_col)
        axis.set_ylabel(value_col)
        axis.tick_params(axis="x", rotation=35)
        self._style_axis(axis)

    def _correlation_heatmap(self, df):
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            self.show_empty("Correlation needs at least two numeric columns.")
            return

        corr = numeric_df.corr()
        axis = self._single_axis()
        # aspect="auto" prevents square heatmaps from clinging to one side of
        # the wide desktop canvas when many labels and a colorbar are present.
        image = axis.imshow(corr, cmap="viridis", vmin=-1, vmax=1, aspect="auto")
        axis.set_title("Correlation Heatmap")
        axis.set_xticks(range(len(corr.columns)))
        axis.set_xticklabels(corr.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(corr.columns)))
        axis.set_yticklabels(corr.columns)
        self.figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
        self.figure.subplots_adjust(left=0.16, right=0.92, bottom=0.28, top=0.88)

    def _missing_values(self, df):
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=True)
        if missing.empty:
            self.show_empty("No missing values found in the current dataset.")
            return

        axis = self._single_axis()
        axis.barh(missing.index.astype(str), missing.values, color="#36b66b")
        axis.set_title("Missing Values by Column")
        axis.set_xlabel("Missing cells")
        axis.set_ylabel("Column")
        self._style_axis(axis)

    def _numeric_series(self, df, column):
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return None
        return df[column].replace([np.inf, -np.inf], np.nan)

    def _has_columns(self, df, columns):
        return all(column in df.columns for column in columns)

    def _style_axis(self, axis):
        axis.grid(True, color="#e5e7eb", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
