from __future__ import annotations

import copy
from typing import Any, Dict, List


MODEL_CATALOG: Dict[str, Dict[str, Dict[str, Any]]] = {
    "supervised": {
        "svm": {
            "label": "Support Vector Machine",
            "required": {
                "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100000.0, "step": 0.1},
                "kernel": {"type": "choice", "default": "rbf", "choices": ["rbf", "linear", "poly", "sigmoid"]},
            },
            "advanced": {
                "probability": {"type": "bool", "default": True},
            },
        },
        "knn": {
            "label": "K-Nearest Neighbors",
            "required": {
                "n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 1000, "step": 1},
            },
            "advanced": {},
        },
        "decision_tree": {
            "label": "Decision Tree",
            "required": {
                "max_depth": {"type": "int", "default": 0, "min": 0, "max": 1000, "step": 1, "zero_as_none": True},
            },
            "advanced": {},
        },
        "random_forest": {
            "label": "Random Forest",
            "required": {
                "n_estimators": {"type": "int", "default": 100, "min": 1, "max": 5000, "step": 1},
                "max_depth": {"type": "int", "default": 0, "min": 0, "max": 1000, "step": 1, "zero_as_none": True},
            },
            "advanced": {},
        },
        "logistic_regression": {
            "label": "Logistic Regression",
            "required": {
                "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100000.0, "step": 0.1},
                "max_iter": {"type": "int", "default": 1000, "min": 100, "max": 100000, "step": 100},
            },
            "advanced": {},
        },
    },
    "semi_supervised": {
        "self_training": {
            "label": "Self-Training",
            "required": {
                "threshold": {"type": "float", "default": 0.9, "min": 0.05, "max": 0.99, "step": 0.05},
                "max_iter": {"type": "int", "default": 10, "min": 1, "max": 100, "step": 1},
                "base_model_name": {"type": "text", "default": ""},
            },
            "advanced": {},
        },
    },
    "unsupervised": {
        "kmeans": {
            "label": "K-Means",
            "required": {
                "n_clusters": {"type": "int", "default": 3, "min": 2, "max": 100, "step": 1},
            },
            "advanced": {
                "n_init": {"type": "int", "default": 10, "min": 1, "max": 500, "step": 1},
            },
        },
        "dbscan": {
            "label": "DBSCAN",
            "required": {
                "eps": {"type": "float", "default": 0.5, "min": 0.01, "max": 1000.0, "step": 0.1},
                "min_samples": {"type": "int", "default": 5, "min": 1, "max": 1000, "step": 1},
            },
            "advanced": {},
        },
        "hierarchical": {
            "label": "Hierarchical",
            "required": {
                "n_clusters": {"type": "int", "default": 3, "min": 2, "max": 100, "step": 1},
                "linkage": {"type": "choice", "default": "ward", "choices": ["ward", "complete", "average", "single"]},
            },
            "advanced": {},
        },
    },
}

COMMON_TRAINING_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "test_size": {"type": "float", "default": 0.3, "min": 0.05, "max": 0.95, "step": 0.05},
    "epochs": {"type": "int", "default": 10, "min": 1, "max": 100000, "step": 1},
    "batch_size": {"type": "int", "default": 32, "min": 1, "max": 100000, "step": 1},
    "verbose": {"type": "bool", "default": False},
    "random_state": {"type": "int", "default": 42, "min": 0, "max": 999999, "step": 1},
    "stratify": {"type": "bool", "default": False},
}


class ModelController:
    @staticmethod
    def ensure_project_state(project: Dict[str, Any]) -> None:
        project.setdefault("added_models", [])
        project.setdefault("model_queue", [])

    @staticmethod
    def category_options() -> List[str]:
        return ["supervised", "semi_supervised", "unsupervised"]

    @staticmethod
    def algorithms_for_category(category: str) -> List[str]:
        return list(MODEL_CATALOG.get(category, {}).keys())

    @staticmethod
    def get_definition(category: str, algorithm: str) -> Dict[str, Any]:
        return copy.deepcopy(MODEL_CATALOG.get(category, {}).get(algorithm, {}))

    @staticmethod
    def default_common_parameters() -> Dict[str, Any]:
        return {key: spec["default"] for key, spec in COMMON_TRAINING_PARAMETERS.items()}

    @staticmethod
    def default_required_parameters(category: str, algorithm: str) -> Dict[str, Any]:
        definition = ModelController.get_definition(category, algorithm)
        return {key: spec.get("default") for key, spec in definition.get("required", {}).items()}

    @staticmethod
    def default_advanced_parameters(category: str, algorithm: str) -> Dict[str, Any]:
        definition = ModelController.get_definition(category, algorithm)
        return {key: spec.get("default") for key, spec in definition.get("advanced", {}).items()}

    @staticmethod
    def unique_name(base_name: str, existing_names: List[str]) -> str:
        if base_name not in existing_names:
            return base_name
        index = 1
        while f"{base_name}_{index}" in existing_names:
            index += 1
        return f"{base_name}_{index}"

    @staticmethod
    def create_added_model_entry(
        name: str,
        category: str,
        algorithm: str,
        label: str,
        common_parameters: Dict[str, Any],
        required_parameters: Dict[str, Any],
        advanced_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "category": category,
            "algorithm": algorithm,
            "label": label,
            "trained": False,
            "common_parameters": dict(common_parameters),
            "required_parameters": dict(required_parameters),
            "advanced_parameters": dict(advanced_parameters),
        }

    @staticmethod
    def duplicate_entry(entry: Dict[str, Any], existing_names: List[str]) -> Dict[str, Any]:
        duplicate = copy.deepcopy(entry)
        duplicate["name"] = ModelController.unique_name(entry["name"], existing_names)
        duplicate["trained"] = False
        return duplicate

    @staticmethod
    def find_added_model(project: Dict[str, Any], name: str) -> Dict[str, Any] | None:
        for model in project.get("added_models", []):
            if model.get("name") == name:
                return model
        return None

    @staticmethod
    def build_training_parameters(entry: Dict[str, Any]) -> Dict[str, Any]:
        params = {}
        params.update(entry.get("required_parameters", {}))
        params.update(entry.get("advanced_parameters", {}))

        definition = ModelController.get_definition(entry.get("category", ""), entry.get("algorithm", ""))
        for key, spec in definition.get("required", {}).items():
            if spec.get("zero_as_none") and params.get(key) == 0:
                params[key] = None
        for key, spec in definition.get("advanced", {}).items():
            if spec.get("zero_as_none") and params.get(key) == 0:
                params[key] = None
        return params

    @staticmethod
    def queue_rows(project: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = []
        for name in project.get("model_queue", []):
            model = ModelController.find_added_model(project, name)
            if not model:
                continue
            rows.append({
                "name": model.get("name", ""),
                "category": model.get("category", ""),
                "algorithm": model.get("algorithm", ""),
                "trained": "Yes" if model.get("trained") else "No",
            })
        return rows
