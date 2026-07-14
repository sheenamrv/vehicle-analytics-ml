from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QLabel,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

'''
    Central location for the Models tab's dynamic parameter fields.
    Mirrors the algorithms and parameters supported by
    src/model/supervised_model.py so the GUI never falls out of sync
    with what build_model() actually accepts.
'''

MODEL_TYPES = [
    ("SVM", "svm"),
    ("KNN", "knn"),
    ("Decision Tree", "decision_tree"),
    ("Random Forest", "random_forest"),
    ("Logistic Regression", "logistic_regression"),
]


class ModelParameterPanel(QWidget):
    """Shows only the parameter fields relevant to the selected model type."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self.pages = {
            "svm": self._build_svm_page(),
            "knn": self._build_knn_page(),
            "decision_tree": self._build_tree_page(),
            "random_forest": self._build_forest_page(),
            "logistic_regression": self._build_empty_page(),
        }

        for page in self.pages.values():
            self.stack.addWidget(page)

        # Use a fixed pixel height (not a computed sizeHint) so this stays
        # correct regardless of when the app-wide font/stylesheet gets
        # applied relative to widget construction. Random Forest (the
        # tallest page: 2 labels + 2 spin boxes) needs roughly this much
        # room; shorter pages just leave blank space below instead of
        # causing overlap.
        self.stack.setFixedHeight(150)

        self.set_model_type("svm")

    def _build_svm_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("C value"))
        self.svm_c = QDoubleSpinBox()
        self.svm_c.setRange(0.01, 100.0)
        self.svm_c.setSingleStep(0.1)
        self.svm_c.setValue(1.0)
        layout.addWidget(self.svm_c)
        layout.addStretch()
        return page

    def _build_knn_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Number of neighbors"))
        self.knn_neighbors = QSpinBox()
        self.knn_neighbors.setRange(1, 100)
        self.knn_neighbors.setValue(5)
        layout.addWidget(self.knn_neighbors)
        layout.addStretch()
        return page

    def _build_tree_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Max depth (0 = unlimited)"))
        self.tree_max_depth = QSpinBox()
        self.tree_max_depth.setRange(0, 100)
        self.tree_max_depth.setValue(0)
        layout.addWidget(self.tree_max_depth)
        layout.addStretch()
        return page

    def _build_forest_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Number of trees"))
        self.forest_trees = QSpinBox()
        self.forest_trees.setRange(1, 2000)
        self.forest_trees.setValue(100)
        layout.addWidget(self.forest_trees)
        layout.addWidget(QLabel("Max depth (0 = unlimited)"))
        self.forest_max_depth = QSpinBox()
        self.forest_max_depth.setRange(0, 100)
        self.forest_max_depth.setValue(0)
        layout.addWidget(self.forest_max_depth)
        layout.addStretch()
        return page

    def _build_empty_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("No extra parameters for Logistic Regression."))
        layout.addStretch()
        return page

    def set_model_type(self, model_type):
        page = self.pages.get(model_type, self.pages["svm"])
        self.stack.setCurrentWidget(page)

    def get_parameters(self):
        """Return the parameter dict matching build_model()'s expected keys."""
        current = self.stack.currentWidget()

        if current is self.pages["svm"]:
            return {"C": self.svm_c.value()}

        if current is self.pages["knn"]:
            return {"n_neighbors": self.knn_neighbors.value()}

        if current is self.pages["decision_tree"]:
            depth = self.tree_max_depth.value()
            return {"max_depth": depth if depth > 0 else None}

        if current is self.pages["random_forest"]:
            depth = self.forest_max_depth.value()
            return {
                "n_estimators": self.forest_trees.value(),
                "max_depth": depth if depth > 0 else None,
            }

        return {}