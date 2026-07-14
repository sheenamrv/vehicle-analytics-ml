import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from src.frontend.table_model import PandasTableModel
from src.frontend.charts import ChartCanvas
from src.frontend.widgets import data_panel, divider, primary_button, secondary_button, section_label, taller_dropdown


MODEL_OPTIONS = {
    "Support Vector Machine": "svm",
    "K-Nearest Neighbors": "knn",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "Logistic Regression": "logistic_regression",
}


# These widgets collect and display state only. Training orchestration belongs
# to AnalyticsWindow so panels stay reusable and never call model backends.
class SupervisedModelSidebar(QWidget):
    """Collect configuration for the classifiers supported by the backend."""

    train_requested = Signal(dict)
    save_configuration_requested = Signal(dict)
    load_configuration_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 28, 26, 22)
        layout.setSpacing(9)

        brand = QLabel("Classify & Learn Lab")
        brand.setObjectName("brand")
        layout.addWidget(brand)
        layout.addSpacing(26)
        layout.addWidget(section_label("SUPERVISED MODEL"))

        self.model_type = taller_dropdown(QComboBox())
        self.model_type.addItems(MODEL_OPTIONS)
        self.model_type.currentTextChanged.connect(self._update_parameter_controls)
        layout.addWidget(QLabel("Algorithm"))
        layout.addWidget(self.model_type)

        self.model_name = QLineEdit()
        self.model_name.setPlaceholderText("Model name")
        layout.addWidget(QLabel("Model Name"))
        layout.addWidget(self.model_name)
        layout.addWidget(divider())

        layout.addWidget(section_label("PARAMETERS"))
        self.parameter_form = QFormLayout()
        self.parameter_form.setContentsMargins(0, 0, 0, 0)
        self.parameter_form.setSpacing(7)

        self.c_value = QDoubleSpinBox()
        self.c_value.setRange(0.001, 100000.0)
        self.c_value.setValue(1.0)
        self.c_value.setDecimals(3)
        self.c_value.setSingleStep(0.1)

        self.kernel = taller_dropdown(QComboBox())
        self.kernel.addItems(["rbf", "linear", "poly", "sigmoid"])

        self.neighbors = QSpinBox()
        self.neighbors.setRange(1, 1000)
        self.neighbors.setValue(5)

        self.max_depth = QSpinBox()
        self.max_depth.setRange(0, 1000)
        self.max_depth.setSpecialValueText("None")
        self.max_depth.setValue(0)

        self.n_estimators = QSpinBox()
        self.n_estimators.setRange(1, 5000)
        self.n_estimators.setValue(100)

        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(100, 100000)
        self.max_iterations.setValue(1000)
        self.max_iterations.setSingleStep(100)

        self.parameter_rows = []
        for label, field, types in (
            ("C", self.c_value, {"svm", "logistic_regression"}),
            ("Kernel", self.kernel, {"svm"}),
            ("Neighbors", self.neighbors, {"knn"}),
            ("Max depth", self.max_depth, {"decision_tree", "random_forest"}),
            ("Trees", self.n_estimators, {"random_forest"}),
            ("Max iterations", self.max_iterations, {"logistic_regression"}),
        ):
            self.parameter_form.addRow(label, field)
            self.parameter_rows.append((self.parameter_form.labelForField(field), field, types))
        layout.addLayout(self.parameter_form)
        layout.addWidget(divider())

        layout.addWidget(section_label("TRAINING"))
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.05, 0.95)
        self.test_size.setValue(0.3)
        self.test_size.setSingleStep(0.05)
        self.random_state = QSpinBox()
        self.random_state.setRange(0, 999999)
        self.random_state.setValue(42)
        self.stratify = QComboBox()
        self.stratify.addItems(["Off", "On"])
        training_form = QFormLayout()
        training_form.setContentsMargins(0, 0, 0, 0)
        training_form.addRow("Test split", self.test_size)
        training_form.addRow("Random state", self.random_state)
        training_form.addRow("Stratify", self.stratify)
        layout.addLayout(training_form)

        self.train_button = primary_button("Train Model")
        self.train_button.clicked.connect(self._request_training)
        layout.addWidget(self.train_button)
        save_configuration = secondary_button("Save Configuration")
        save_configuration.clicked.connect(lambda: self.save_configuration_requested.emit(self.configuration()))
        layout.addWidget(save_configuration)
        load_configuration = secondary_button("Load Configuration")
        load_configuration.clicked.connect(self.load_configuration_requested.emit)
        layout.addWidget(load_configuration)
        layout.addStretch()
        self._update_parameter_controls()

    def _update_parameter_controls(self):
        model_type = self.selected_model_type()
        self.c_value.setVisible(model_type in {"svm", "logistic_regression"})
        self.kernel.setVisible(model_type == "svm")
        self.neighbors.setVisible(model_type == "knn")
        self.max_depth.setVisible(model_type in {"decision_tree", "random_forest"})
        self.n_estimators.setVisible(model_type == "random_forest")
        self.max_iterations.setVisible(model_type == "logistic_regression")
        for label, field, types in self.parameter_rows:
            label.setVisible(model_type in types)
            field.setVisible(model_type in types)

    def selected_model_type(self):
        return MODEL_OPTIONS[self.model_type.currentText()]

    def parameters(self):
        model_type = self.selected_model_type()
        if model_type == "svm":
            return {"C": self.c_value.value(), "kernel": self.kernel.currentText()}
        if model_type == "knn":
            return {"n_neighbors": self.neighbors.value()}
        if model_type == "decision_tree":
            return {"max_depth": self.max_depth.value() or None}
        if model_type == "random_forest":
            return {
                "n_estimators": self.n_estimators.value(),
                "max_depth": self.max_depth.value() or None,
            }
        return {"C": self.c_value.value(), "max_iter": self.max_iterations.value()}

    def _request_training(self):
        self.train_requested.emit(self.configuration())

    def configuration(self):
        return {
            "model_type": self.selected_model_type(),
            "model_name": self.model_name.text().strip(),
            "parameters": self.parameters(),
            "test_size": self.test_size.value(),
            "random_state": self.random_state.value(),
            "stratify": self.stratify.currentText() == "On",
        }

    def set_configuration(self, configuration):
        # Configuration files store backend identifiers; translate them back to
        # the human-readable labels used by the algorithm dropdown.
        model_type = configuration.get("model_type")
        for label, value in MODEL_OPTIONS.items():
            if value == model_type:
                self.model_type.setCurrentText(label)
                break
        self.model_name.setText(configuration.get("model_name", ""))
        parameters = configuration.get("parameters", {})
        self.c_value.setValue(float(parameters.get("C", 1.0)))
        if "kernel" in parameters:
            self.kernel.setCurrentText(parameters["kernel"])
        self.neighbors.setValue(int(parameters.get("n_neighbors", 5)))
        self.max_depth.setValue(int(parameters.get("max_depth") or 0))
        self.n_estimators.setValue(int(parameters.get("n_estimators", 100)))
        self.max_iterations.setValue(int(parameters.get("max_iter", 1000)))
        self.test_size.setValue(float(configuration.get("test_size", 0.3)))
        self.random_state.setValue(int(configuration.get("random_state", 42)))
        self.stratify.setCurrentText("On" if configuration.get("stratify") else "Off")

    def set_training(self, training):
        self.train_button.setEnabled(not training)
        self.train_button.setText("Training..." if training else "Train Model")


class SupervisedModelPage(QWidget):
    """Display trained models and the most recent classifier evaluation."""

    delete_requested = Signal(str)
    test_requested = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_table_model = PandasTableModel()
        self.metrics_model = PandasTableModel()
        self.confusion_model = PandasTableModel()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = section_label("TRAINED MODELS")
        header.addWidget(title)
        header.addStretch()
        self.delete_button = secondary_button("Delete Selected")
        self.delete_button.clicked.connect(self._delete_selected)
        header.addWidget(self.delete_button)
        self.test_button = primary_button("Test Selected")
        self.test_button.clicked.connect(self._test_selected)
        header.addWidget(self.test_button)
        layout.addLayout(header)

        self.model_table = QTableView()
        self.model_table.setModel(self.model_table_model)
        self.model_table.setSelectionBehavior(QTableView.SelectRows)
        self.model_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.model_table, 2)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        layout.addWidget(section_label("LATEST EVALUATION"))
        evaluation_layout = QHBoxLayout()
        evaluation_layout.setSpacing(16)
        evaluation_layout.addWidget(data_panel("METRICS", self.metrics_model), 1)
        evaluation_layout.addWidget(data_panel("CONFUSION MATRIX", self.confusion_model), 2)
        layout.addLayout(evaluation_layout, 2)

    def set_models(self, models):
        rows = [
            {
                "name": model.get("display_name", ""),
                "algorithm": model.get("algorithm", ""),
                "accuracy": model.get("metrics", {}).get("accuracy"),
                "f1": model.get("metrics", {}).get("f1"),
            }
            for model in models
        ]
        self.model_table_model.set_data(pd.DataFrame(rows, columns=["name", "algorithm", "accuracy", "f1"]))

    def set_result(self, result):
        metrics = pd.DataFrame(
            [{"metric": key.replace("_", " ").title(), "value": value}
             for key, value in result["metrics"].items()]
        )
        self.metrics_model.set_data(metrics.round(4))
        labels = [str(label) for label in sorted(pd.unique(result["y_test"]))]
        matrix = pd.DataFrame(result["confusion_matrix"], index=labels, columns=labels)
        matrix.index.name = "actual"
        self.confusion_model.set_data(matrix)

    def set_training(self, training):
        self.progress.setVisible(training)
        if training:
            self.progress.setRange(0, 0)
            self.progress.setFormat("Training model...")
        else:
            self.progress.setRange(0, 1)

    def _delete_selected(self):
        indexes = self.model_table.selectionModel().selectedRows()
        if indexes:
            self.delete_requested.emit(self.model_table_model._data.iloc[indexes[0].row()]["name"])

    def _test_selected(self):
        names = [
            self.model_table_model._data.iloc[index.row()]["name"]
            for index in self.model_table.selectionModel().selectedRows()
        ]
        self.test_requested.emit(names)


class SemiSupervisedSidebar(QWidget):
    """Configure self-training from a saved supervised classifier."""

    train_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._models = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 28, 26, 22)
        layout.setSpacing(9)

        brand = QLabel("Classify & Learn Lab")
        brand.setObjectName("brand")
        layout.addWidget(brand)
        layout.addSpacing(26)
        layout.addWidget(section_label("SEMI-SUPERVISED MODEL"))

        self.pretrained_model = taller_dropdown(QComboBox())
        layout.addWidget(QLabel("Starting model"))
        layout.addWidget(self.pretrained_model)
        layout.addWidget(divider())

        layout.addWidget(section_label("SELF-TRAINING"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.05, 0.99)
        self.threshold.setValue(0.90)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(1, 100)
        self.max_iterations.setValue(10)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Confidence", self.threshold)
        form.addRow("Max iterations", self.max_iterations)
        layout.addLayout(form)

        self.train_button = primary_button("Train Self-Training Model")
        self.train_button.clicked.connect(self._request_training)
        layout.addWidget(self.train_button)
        layout.addStretch()

    def set_pretrained_models(self, models):
        # A semi-supervised estimator must start from a classifier. Exclude SSL
        # and clustering entries if project model types expand in the future.
        self._models = [
            model for model in models
            if model.get("algorithm") in set(MODEL_OPTIONS.values())
        ]
        current_name = self.pretrained_model.currentData()
        self.pretrained_model.blockSignals(True)
        self.pretrained_model.clear()
        for model in self._models:
            self.pretrained_model.addItem(model.get("display_name", "Unnamed model"), model)
        self.pretrained_model.blockSignals(False)
        if current_name:
            index = self.pretrained_model.findData(current_name)
            if index >= 0:
                self.pretrained_model.setCurrentIndex(index)

    def _request_training(self):
        model = self.pretrained_model.currentData()
        if model is None:
            return
        self.train_requested.emit({
            "pretrained_model": model,
            "threshold": self.threshold.value(),
            "max_iter": self.max_iterations.value(),
        })

    def set_training(self, training):
        self.train_button.setEnabled(not training)
        self.pretrained_model.setEnabled(not training)
        self.train_button.setText("Training..." if training else "Train Self-Training Model")


class SemiSupervisedPage(QWidget):
    """Show the latest self-training evaluation and labelling progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics_model = PandasTableModel()
        self.confusion_model = PandasTableModel()
        self.progress_model = PandasTableModel()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(section_label("SELF-TRAINING RESULTS"))

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        top = QHBoxLayout()
        top.setSpacing(16)
        top.addWidget(data_panel("EVALUATION METRICS", self.metrics_model), 1)
        top.addWidget(data_panel("CONFUSION MATRIX", self.confusion_model), 2)
        layout.addLayout(top, 1)
        layout.addWidget(data_panel("PSEUDO-LABELLING PROGRESS", self.progress_model), 1)

    def set_result(self, result):
        metrics = pd.DataFrame(
            [{"metric": key.replace("_", " ").title(), "value": value}
             for key, value in result["metrics"].items()]
        )
        self.metrics_model.set_data(metrics.round(4))
        self.confusion_model.set_data(pd.DataFrame(result["confusion_matrix"]))
        self.progress_model.set_data(result["iteration_progress"].round(2))

    def clear(self):
        self.metrics_model.set_data(pd.DataFrame())
        self.confusion_model.set_data(pd.DataFrame())
        self.progress_model.set_data(pd.DataFrame())

    def set_training(self, training):
        self.progress.setVisible(training)
        if training:
            self.progress.setRange(0, 0)
            self.progress.setFormat("Training self-training model...")
        else:
            self.progress.setRange(0, 1)


UNSUPERVISED_OPTIONS = {
    "K-Means": "kmeans",
    "DBSCAN": "dbscan",
    "Hierarchical": "hierarchical",
}


class UnsupervisedSidebar(QWidget):
    """Configure a backend-supported clustering run."""

    run_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 28, 26, 22)
        layout.setSpacing(9)

        brand = QLabel("Classify & Learn Lab")
        brand.setObjectName("brand")
        layout.addWidget(brand)
        layout.addSpacing(26)
        layout.addWidget(section_label("UNSUPERVISED CLUSTERING"))

        self.method = taller_dropdown(QComboBox())
        self.method.addItems(UNSUPERVISED_OPTIONS)
        self.method.currentTextChanged.connect(self._update_parameter_controls)
        layout.addWidget(QLabel("Method"))
        layout.addWidget(self.method)
        layout.addWidget(divider())

        layout.addWidget(section_label("PARAMETERS"))
        self.n_clusters = QSpinBox()
        self.n_clusters.setRange(2, 100)
        self.n_clusters.setValue(3)
        self.eps = QDoubleSpinBox()
        self.eps.setRange(0.01, 1000.0)
        self.eps.setValue(0.5)
        self.eps.setDecimals(2)
        self.eps.setSingleStep(0.1)
        self.min_samples = QSpinBox()
        self.min_samples.setRange(1, 1000)
        self.min_samples.setValue(5)
        self.linkage = taller_dropdown(QComboBox())
        self.linkage.addItems(["ward", "complete", "average", "single"])
        self.random_state = QSpinBox()
        self.random_state.setRange(0, 999999)
        self.random_state.setValue(42)

        self.parameter_form = QFormLayout()
        self.parameter_form.setContentsMargins(0, 0, 0, 0)
        self.parameter_form.setSpacing(7)
        self.parameter_rows = []
        for label, field, methods in (
            ("Clusters", self.n_clusters, {"kmeans", "hierarchical"}),
            ("Epsilon", self.eps, {"dbscan"}),
            ("Min samples", self.min_samples, {"dbscan"}),
            ("Linkage", self.linkage, {"hierarchical"}),
            ("Random state", self.random_state, {"kmeans"}),
        ):
            self.parameter_form.addRow(label, field)
            self.parameter_rows.append((self.parameter_form.labelForField(field), field, methods))
        layout.addLayout(self.parameter_form)

        self.run_button = primary_button("Run Clustering")
        self.run_button.clicked.connect(self._request_run)
        layout.addWidget(self.run_button)
        layout.addStretch()
        self._update_parameter_controls()

    def selected_method(self):
        return UNSUPERVISED_OPTIONS[self.method.currentText()]

    def _update_parameter_controls(self):
        method = self.selected_method()
        # Parameter rows are created once to keep sidebar geometry stable; only
        # controls supported by the selected backend clusterer are exposed.
        for label, field, methods in self.parameter_rows:
            label.setVisible(method in methods)
            field.setVisible(method in methods)

    def _request_run(self):
        self.run_requested.emit({
            "method": self.selected_method(),
            "n_clusters": self.n_clusters.value(),
            "eps": self.eps.value(),
            "min_samples": self.min_samples.value(),
            "linkage": self.linkage.currentText(),
            "random_state": self.random_state.value(),
        })

    def set_running(self, running):
        self.run_button.setEnabled(not running)
        self.method.setEnabled(not running)
        self.run_button.setText("Clustering..." if running else "Run Clustering")


class UnsupervisedPage(QWidget):
    """Display cluster assignments, quality scores, and a PCA projection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics_model = PandasTableModel()
        self.summary_model = PandasTableModel()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(section_label("CLUSTERING RESULTS"))

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        upper = QHBoxLayout()
        upper.setSpacing(16)
        self.chart = ChartCanvas("Run clustering to view the PCA projection.", min_height=300)
        upper.addWidget(self.chart, 2)
        upper.addWidget(data_panel("CLUSTER COUNTS", self.summary_model), 1)
        layout.addLayout(upper, 2)
        layout.addWidget(data_panel("CLUSTER QUALITY", self.metrics_model), 1)

    def set_result(self, result):
        self.summary_model.set_data(result["summary_df"])
        metrics = pd.DataFrame([
            {"metric": key.replace("_", " ").title(), "value": value}
            for key, value in result["metrics"].items()
            if value is not None
        ])
        self.metrics_model.set_data(metrics.round(4))
        self.chart.plot_pca_scatter(result["pca_result"]["pca_df"], "cluster")

    def clear(self):
        self.summary_model.set_data(pd.DataFrame())
        self.metrics_model.set_data(pd.DataFrame())
        self.chart.show_empty("Run clustering to view the PCA projection.")

    def set_running(self, running):
        self.progress.setVisible(running)
        if running:
            self.progress.setRange(0, 0)
            self.progress.setFormat("Running clustering...")
        else:
            self.progress.setRange(0, 1)
