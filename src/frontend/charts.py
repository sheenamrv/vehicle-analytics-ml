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
import matplotlib
import seaborn as sns
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from src.frontend.chart_specs import get_chart_spec


DEFAULT_CMAP = "viridis"

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

    def plot(self, df, chart_type, x_col=None, y_col=None, label_col=None,
             z_col=None, bins=24, extra_cols=None, show_median_line=False,
             show_mean_line=False, cmap=DEFAULT_CMAP):
        """Route Visualization-tab chart selections to plotting methods."""
        if df.empty:
            self.show_empty("Open a dataset to visualize it.")
            return

        spec = get_chart_spec(chart_type)
        renderer = spec.renderer if spec is not None else None

        if renderer == "histogram":
            self._histogram(df, x_col, bins=bins, show_mean_line=show_mean_line,
                            show_median_line=show_median_line, cmap=cmap)
        elif renderer == "scatter":
            self._scatter(df, x_col, y_col, label_col, cmap=cmap)
        elif renderer == "line":
            self._line(df, x_col, y_col, show_median_line=show_median_line)
        elif renderer == "box_plot":
            self._box_plot(df, x_col, label_col=label_col, show_median_line=show_median_line)
        elif renderer == "bar_chart":
            self._bar_chart(df, x_col, y_col)
        elif renderer == "grouped_box_plot":
            self._grouped_box_plot(df, x_col, y_col)
        elif renderer == "class_separation":
            self._class_separation(df, x_col, y_col, label_col, cmap=cmap)
        elif renderer == "scatter_3d":
            self._scatter_3d(df, x_col, y_col, z_col, label_col, cmap=cmap)
        elif renderer == "time_series":
            self._time_series(df, extra_cols)
        elif renderer == "distribution_comparison":
            self._distribution_comparison(df, extra_cols)
        else:
            self.show_empty("Choose a chart type.")

        self.draw()

    def plot_correlation_heatmap(self, df, cmap=DEFAULT_CMAP):
        """Draw correlation output produced for the Analysis tab."""
        if df.empty:
            self.show_empty("Open a dataset to analyze it.")
            return
        self._correlation_heatmap(df, cmap=cmap)
        self.draw()

    def plot_missing_values(self, df):
        """Render the Import-tab data quality chart."""
        if df.empty:
            self.show_empty("Open a dataset to profile missing values.")
            return
        self._missing_values(df)
        self.draw()

    def plot_pca_scatter(self, pca_df, label_col, cmap=DEFAULT_CMAP):
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
                cmap=cmap,
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

    def _single_axis(self, projection=None):
    # def plot_model_comparison(self, metrics_df):
    #     """Render comparable saved-model classification metrics."""
    #     required = {"name", "accuracy", "precision", "recall", "f1"}
    #     if metrics_df.empty or not required.issubset(metrics_df.columns):
    #         self.show_empty("Train at least one model to compare evaluation metrics.")
    #         return

    #     # Long-form data lets Seaborn create one consistent metric group per
    #     # model without manually calculating bar offsets.
    #     plot_df = metrics_df[["name", "accuracy", "precision", "recall", "f1"]].melt(
    #         id_vars="name",
    #         value_vars=["accuracy", "precision", "recall", "f1"],
    #         var_name="metric",
    #         value_name="score",
    #     ).dropna()
    #     if plot_df.empty:
    #         self.show_empty("No comparable classification metrics are available.")
    #         return

    #     axis = self._single_axis()
    #     sns.barplot(data=plot_df, x="name", y="score", hue="metric", palette="Greens", ax=axis)
    #     axis.set_title("Model Metrics Comparison")
    #     axis.set_xlabel("Model")
    #     axis.set_ylabel("Score")
    #     axis.set_ylim(0, 1.05)
    #     axis.tick_params(axis="x", rotation=25)
    #     axis.legend(title="Metric", frameon=False, loc="best")
    #     self._style_axis(axis)
    #     self.draw()

    # def _single_axis(self):
        """Reset the figure and return a fresh axis for single-chart canvases."""
        self.figure.clear()
        if projection:
            return self.figure.add_subplot(111, projection=projection)
        return self.figure.add_subplot(111)

    def _histogram(self, df, column, bins=24, show_mean_line=False, show_median_line=False, cmap=DEFAULT_CMAP):
        axis = self._single_axis()
        series = self._numeric_series(df, column)
        if series is None:
            self.show_empty("Choose a numeric column for the histogram.")
            return

        bins = int(bins) if bins else 24
        bins = max(2, min(bins, 200))

        cmap_name = self._resolve_cmap(cmap)
        color = self._sample_cmap_color(cmap_name)
        axis.hist(series.dropna(), bins=bins, color=color, edgecolor="white", alpha=0.85)
        if show_mean_line:
            axis.axvline(series.mean(), color="#1f7f43", linestyle="--", linewidth=2)
        if show_median_line:
            axis.axvline(series.median(), color="#0f3e2e", linestyle=":", linewidth=2)
        axis.set_title(f"Histogram: {column} ({bins} bins)")
        axis.set_xlabel(column)
        axis.set_ylabel("Count")
        self._style_axis(axis)

    def _scatter(self, df, x_col, y_col, label_col=None, cmap=DEFAULT_CMAP):
        axis = self._single_axis()
        if not self._has_columns(df, [x_col, y_col]):
            self.show_empty("Choose X and Y columns for the scatter plot.")
            return
        if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
            self.show_empty("Choose numeric X and Y columns for the scatter plot.")
            return

        plot_df = df[[x_col, y_col] + ([label_col] if label_col in df.columns else [])].dropna()
        if plot_df.empty:
            self.show_empty("The selected columns do not contain plottable values.")
            return

        if label_col in plot_df.columns and plot_df[label_col].nunique() <= 12:
            labels = plot_df[label_col].astype("category")
            scatter = axis.scatter(
                plot_df[x_col],
                plot_df[y_col],
                c=labels.cat.codes,
                cmap=self._resolve_cmap(cmap),
                alpha=0.78,
                s=34,
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
            sns.scatterplot(data=plot_df, x=x_col, y=y_col, color="#36b66b", alpha=0.78, s=48, ax=axis)

        axis.set_title(f"{y_col} vs {x_col}")
        axis.set_xlabel(x_col)
        axis.set_ylabel(y_col)
        self._style_axis(axis)

    def _class_separation(self, df, x_col, y_col, label_col, cmap=DEFAULT_CMAP):
        """Scatter plot that requires a label column, colored per class."""
        axis = self._single_axis()
        if not self._has_columns(df, [x_col, y_col]):
            self.show_empty("Choose X and Y columns for class separation.")
            return
        if not label_col or label_col not in df.columns:
            self.show_empty("Choose a label column to separate classes.")
            return

        plot_df = df[[x_col, y_col, label_col]].dropna()
        if plot_df.empty:
            self.show_empty("The selected columns do not contain plottable values.")
            return

        labels = plot_df[label_col].astype("category")
        if labels.cat.categories.size > 20:
            self.show_empty("Too many distinct classes to color legibly (limit 20).")
            return

        scatter = axis.scatter(
            plot_df[x_col],
            plot_df[y_col],
            c=labels.cat.codes,
            cmap="tab20" if labels.cat.categories.size > 10 else cmap,
            alpha=0.8,
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
        axis.set_title(f"Class Separation: {y_col} vs {x_col} by {label_col}")
        axis.set_xlabel(x_col)
        axis.set_ylabel(y_col)
        self._style_axis(axis)

    def _scatter_3d(self, df, x_col, y_col, z_col, label_col=None, cmap=DEFAULT_CMAP):
        if not self._has_columns(df, [x_col, y_col, z_col]):
            self.show_empty("Choose X, Y, and Z columns for the 3D scatter plot.")
            return
        if len({x_col, y_col, z_col}) < 3:
            self.show_empty("Choose three different numeric columns for the 3D scatter plot.")
            return
        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in (x_col, y_col, z_col)):
            self.show_empty("3D scatter requires numeric X, Y, and Z columns.")
            return

        columns = list(dict.fromkeys([x_col, y_col, z_col] + ([label_col] if label_col in df.columns else [])))
        plot_df = df[columns].dropna()
        if plot_df.empty:
            self.show_empty("The selected columns do not contain plottable values.")
            return

        axis = self._single_axis(projection="3d")

        if label_col in plot_df.columns:
            label_values = plot_df[label_col]
            if isinstance(label_values, pd.DataFrame):
                label_values = label_values.iloc[:, 0]
            if label_values.nunique() <= 12:
                labels = label_values.astype("category")
                scatter = axis.scatter(
                    plot_df[x_col],
                    plot_df[y_col],
                    plot_df[z_col],
                    c=labels.cat.codes,
                    cmap=self._resolve_cmap(cmap),
                    alpha=0.8,
                    s=30,
                )
                handles, _ = scatter.legend_elements()
                axis.legend(handles, labels.cat.categories.astype(str), title=label_col, loc="best")
            else:
                axis.scatter(plot_df[x_col], plot_df[y_col], plot_df[z_col], color="#36b66b", alpha=0.8, s=30)
        else:
            scatter = axis.scatter(
                plot_df[x_col], plot_df[y_col], plot_df[z_col], color="#36b66b", alpha=0.8, s=30
            )

        axis.set_title(f"3D Scatter: {x_col}, {y_col}, {z_col}")
        axis.set_xlabel(x_col)
        axis.set_ylabel(y_col)
        axis.set_zlabel(z_col)

    def _time_series(self, df, columns=None):
        """Plot all (or selected) numeric signals against the row index."""
        numeric_df = df.select_dtypes(include="number")
        if columns:
            keep = [c for c in columns if c in numeric_df.columns]
            if keep:
                numeric_df = numeric_df[keep]

        if numeric_df.empty:
            self.show_empty("No numeric signals available for a time series plot.")
            return

        # Cap both series count and row count so the plot stays legible/responsive.
        numeric_df = numeric_df.iloc[:2000, :12]

        axis = self._single_axis()
        for column in numeric_df.columns:
            axis.plot(numeric_df.index, numeric_df[column], linewidth=1.4, label=str(column))

        axis.set_title("Time Series - All Signals")
        axis.set_xlabel("Row Index")
        axis.set_ylabel("Value")
        if numeric_df.shape[1] <= 12:
            axis.legend(loc="best", fontsize=8, frameon=False, ncol=2)
        self._style_axis(axis)

    def _distribution_comparison(self, df, columns=None):
        """Overlay histograms for multiple numeric columns to compare shape/spread."""
        numeric_df = df.select_dtypes(include="number")
        if columns:
            keep = [c for c in columns if c in numeric_df.columns]
            if keep:
                numeric_df = numeric_df[keep]

        if numeric_df.empty:
            self.show_empty("No numeric columns available to compare.")
            return

        numeric_df = numeric_df.iloc[:, :8]  # keep legend readable
        axis = self._single_axis()
        colors = plt_colormap(numeric_df.shape[1], cmap="tab10")

        for color, column in zip(colors, numeric_df.columns):
            series = numeric_df[column].replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                continue
            axis.hist(series, bins=30, alpha=0.45, label=str(column), color=color, density=True)

        axis.set_title("Feature Distribution Comparison")
        axis.set_xlabel("Value")
        axis.set_ylabel("Density")
        axis.legend(loc="best", fontsize=8, frameon=False)
        self._style_axis(axis)

    def _line(self, df, x_col, y_col, show_median_line=False):
        axis = self._single_axis()
        if not self._has_columns(df, [x_col, y_col]):
            self.show_empty("Choose X and Y columns for the line chart.")
            return

        if not pd.api.types.is_numeric_dtype(df[y_col]):
            self.show_empty("Choose a numeric Y column for the line chart.")
            return

        # Cap row count to keep line rendering responsive on large datasets.
        plot_df = df[[x_col, y_col]].dropna().head(1000)
        if plot_df.empty:
            self.show_empty("The selected columns do not contain plottable values.")
            return

        x_values = self._prepare_plot_values(plot_df[x_col])
        y_values = plot_df[y_col]
        axis.plot(x_values, y_values, color="#1f7f43", linewidth=2)
        if show_median_line and pd.api.types.is_numeric_dtype(plot_df[y_col]):
            axis.axhline(plot_df[y_col].median(), color="#0f3e2e", linestyle=":", linewidth=1.5)
        axis.set_title(f"{y_col} over {x_col}")
        axis.set_xlabel(x_col)
        axis.set_ylabel(y_col)
        self._style_axis(axis)

    def _box_plot(self, df, column, label_col=None, show_median_line=False):
        axis = self._single_axis()
        series = self._numeric_series(df, column)
        if series is None:
            if label_col and label_col in df.columns and pd.api.types.is_numeric_dtype(df[label_col]):
                plot_df = df[[column, label_col]].dropna()
                if plot_df.empty:
                    self.show_empty("The selected columns do not contain plottable values.")
                    return
                groups = []
                labels = []
                for group_name, group_values in plot_df.groupby(column, dropna=False):
                    groups.append(group_values[label_col].to_numpy())
                    labels.append(str(group_name))
                if not groups:
                    self.show_empty("The selected columns do not contain plottable values.")
                    return
                axis.boxplot(
                    groups,
                    labels=labels,
                    vert=False,
                    patch_artist=True,
                    boxprops={"facecolor": "#ccebd9", "edgecolor": "#1f7f43"},
                    medianprops={"color": "#1f7f43", "linewidth": 2},
                )
                axis.set_title(f"{label_col} by {column}")
                axis.set_xlabel(column)
                axis.set_ylabel(label_col)
                axis.tick_params(axis="x", rotation=35)
                self._style_axis(axis)
                return
            self.show_empty("Choose a numeric column for the box plot.")
            return

        median_color = "#1f7f43" if show_median_line else "#1f7f43"
        axis.boxplot(
            series.dropna(),
            vert=False,
            patch_artist=True,
            boxprops={"facecolor": "#ccebd9", "edgecolor": "#1f7f43"},
            medianprops={"color": median_color, "linewidth": 2},
        )
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

    def _correlation_heatmap(self, df, cmap=DEFAULT_CMAP):
        if df.empty:
            self.show_empty("Correlation needs at least two numeric columns.")
            return

        if isinstance(df, pd.DataFrame) and list(df.index.astype(str)) == list(df.columns.astype(str)):
            corr = df.copy()
        else:
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.shape[1] < 2:
                self.show_empty("Correlation needs at least two numeric columns.")
                return
            corr = numeric_df.corr()

        if corr.shape[1] < 2:
            self.show_empty("Correlation needs at least two numeric columns.")
            return

        # Constant or near-constant columns can produce NaN correlations; fill
        # them so the heatmap still renders and colormap changes stay visible.
        corr = corr.fillna(0.0)

        axis = self._single_axis()
        # aspect="auto" prevents square heatmaps from clinging to one side of
        # the wide desktop canvas when many labels and a colorbar are present.
        resolved_cmap = self._resolve_cmap(cmap)
        image = axis.imshow(corr, cmap=resolved_cmap, vmin=-1, vmax=1, aspect="auto")
        axis.set_title(f"Correlation Heatmap ({resolved_cmap})")
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

    def _resolve_cmap(self, cmap):
        if not cmap:
            return DEFAULT_CMAP
        try:
            matplotlib.colormaps[cmap]
            return cmap
        except Exception:
            return DEFAULT_CMAP

    def _sample_cmap_color(self, cmap, value=0.5):
        try:
            return matplotlib.colormaps[self._resolve_cmap(cmap)](value)
        except Exception:
            return "#36b66b"

    def _prepare_plot_values(self, series):
        if pd.api.types.is_numeric_dtype(series):
            return series
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        try:
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().sum() > 0 and parsed.notna().sum() >= max(1, len(parsed) // 2):
                return parsed
        except Exception:
            pass
        return series

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
        
    
    # For Playback functionality
    def plot_realtime_signal(self, df, column, current_row=None):
        if (
            df.empty
            or column not in df.columns
            or not pd.api.types.is_numeric_dtype(df[column])
        ):
            self.show_empty("No numeric signal available.")
            return

        axis = self._single_axis()
        plot_df = (
            df[[column]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if plot_df.empty:
            self.show_empty("No plottable values in this signal window.")
            return

        axis.plot(plot_df.index, plot_df[column], linewidth=1.8)

        if current_row is not None and current_row in plot_df.index:
            axis.axvline(
                current_row,
                linestyle="--",
                linewidth=1.2,
            )

        axis.set_title(str(column))
        axis.set_xlabel("row / time")
        axis.set_ylabel("value")
        self._style_axis(axis)
        self.draw()

    def plt_colormap(n, cmap="tab10"):
        """Return n visually distinct colors for overlay-style charts."""
        try:
            cmap_obj = matplotlib.colormaps[cmap if cmap else "tab10"]
        except Exception:
            cmap_obj = matplotlib.colormaps["tab10"]
        if n <= 1:
            return [cmap_obj(0.5)]
        return [cmap_obj(i / max(n - 1, 1)) for i in range(n)]
