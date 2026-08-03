import json
import joblib
import zipfile
import tempfile
import re
from pathlib import Path

import pandas as pd
import numpy as np


MODEL_PACKAGE_MARKER = "vehicle_analytics_ml_model_package"
MODEL_PACKAGE_VERSION = 1


def create_model_package(model_info, added_entry=None, label=None):
    """Create one self-contained PKL payload for a trained model."""
    model_info = dict(model_info or {})
    added_entry = dict(added_entry or {})
    evaluation = dict(model_info.get("evaluation", {}) or {})

    # Preserve result values that may exist at the top level in older records
    for key in (
        "confusion_matrix",
        "confusion_labels",
        "cluster_summary",
        "ssl_progress",
        "ssl_iteration_progress",
        "ssl_export_data",
        "clustered_export_data",
        "predictions",
        "y_true",
        "y_score",
        "labels",
        "transduction",
        "labeled_iter",
    ):
        if key in model_info and key not in evaluation:
            evaluation[key] = model_info.get(key)

    metrics = (
        model_info.get("metrics")
        or evaluation.get("metrics")
        or {}
    )
    evaluation.setdefault("metrics", metrics)

    resolved_label = (
        label
        if label is not None
        else model_info.get(
            "label",
            added_entry.get("label", ""),
        )
    )

    return {
        "package_type": MODEL_PACKAGE_MARKER,
        "package_version": MODEL_PACKAGE_VERSION,
        "model": model_info.get("model"),
        "display_name": model_info.get(
            "display_name",
            added_entry.get("name", ""),
        ),
        "category": model_info.get(
            "category",
            added_entry.get("category", ""),
        ),
        "algorithm": model_info.get(
            "algorithm",
            added_entry.get("algorithm", ""),
        ),
        "label": resolved_label or "",
        "feature_columns": list(
            model_info.get(
                "feature_columns",
                added_entry.get("feature_columns", []),
            )
            or []
        ),
        "parameters": model_info.get("parameters", {}) or {},
        "common_parameters": (
            model_info.get("common_parameters")
            or added_entry.get("common_parameters")
            or {}
        ),
        "required_parameters": (
            model_info.get("required_parameters")
            or added_entry.get("required_parameters")
            or {}
        ),
        "advanced_parameters": (
            model_info.get("advanced_parameters")
            or added_entry.get("advanced_parameters")
            or {}
        ),
        "metrics": metrics,
        "evaluation": evaluation,
    }


def unpack_model_package(payload):
    """Return ``(model, metadata, is_package)`` for new and legacy PKLs."""

    # Legacy estimator-only PKL
    if not isinstance(payload, dict):
        return payload, {}, False

    model = payload.get("model")

    if model is None:
        model = payload.get("estimator")

    if model is None:
        model = payload.get("trained_model")

    if model is None:
        return None, dict(payload), False

    evaluation = dict(payload.get("evaluation", {}) or {})

    for key in (
        "confusion_matrix",
        "confusion_labels",
        "cluster_summary",
        "ssl_progress",
        "ssl_iteration_progress",
        "ssl_export_data",
        "clustered_export_data",
        "predictions",
        "y_true",
        "y_score",
        "labels",
        "transduction",
        "labeled_iter",
    ):
        if key in payload and key not in evaluation:
            evaluation[key] = payload.get(key)

    metrics = (
        payload.get("metrics")
        or evaluation.get("metrics")
        or {}
    )
    evaluation.setdefault("metrics", metrics)

    metadata = {
        "display_name": payload.get(
            "display_name",
            payload.get("name", ""),
        ),
        "category": payload.get("category", ""),
        "algorithm": payload.get("algorithm", ""),
        "label": payload.get(
            "label",
            payload.get("label_column", ""),
        ),
        "feature_columns": list(
            payload.get(
                "feature_columns",
                payload.get("features", []),
            )
            or []
        ),
        "parameters": payload.get("parameters", {}) or {},
        "common_parameters": payload.get("common_parameters", {}) or {},
        "required_parameters": payload.get("required_parameters", {}) or {},
        "advanced_parameters": payload.get("advanced_parameters", {}) or {},
        "metrics": metrics,
        "evaluation": evaluation,
        "package_version": payload.get("package_version"),
    }

    is_package = payload.get("package_type") == MODEL_PACKAGE_MARKER

    return model, metadata, is_package


def make_json_safe(value):
    """Convert project metadata into values accepted by json.dump."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, pd.DataFrame):
        return make_json_safe(value.to_dict(orient="records"))

    if isinstance(value, pd.Series):
        return make_json_safe(value.tolist())

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    if value is pd.NA:
        return None

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def _safe_filename(value, fallback="model"):
    value = str(value or fallback).strip()
    value = re.sub(r'[<>:"/\\|?*]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._")

    return value or fallback


def _sanitize_dataframe_for_pickle(df):
    # sanitized = df.copy()
    # for column in sanitized.columns:
    #     dtype = sanitized[column].dtype
    #     if is_extension_array_dtype(dtype) and not (
    #         is_datetime64_any_dtype(dtype) or is_timedelta64_dtype(dtype)
    #     ):
    #         sanitized[column] = sanitized[column].astype(object)
    # return sanitized

    df = df.copy()

    for col in df.columns:
        dtype = str(df[col].dtype)

        if "[pyarrow]" in dtype:
            if dtype.startswith("string"):
                df[col] = df[col].astype("object")
            elif dtype.startswith("int"):
                df[col] = df[col].astype("Int64")
            elif dtype.startswith("float"):
                df[col] = df[col].astype("float64")
            elif dtype.startswith("bool"):
                df[col] = df[col].astype("boolean")
            else:
                df[col] = df[col].astype("object")

    return df

# def save_project(project, original_df, mod_df, target_path=None, feature_df=None):
#
#     project_name = project["project_name"]
#     if target_path is None:
#         icp_path = Path("Projects") / f"{project_name}.icp"
#     else:
#         icp_path = Path(target_path)
#         if icp_path.suffix.lower() != ".icp":
#             icp_path = icp_path.with_suffix(".icp")
#
#     icp_path.parent.mkdir(parents=True, exist_ok=True)
#
#     with tempfile.TemporaryDirectory() as temp_dir:
#
#         temp_dir = Path(temp_dir)
#
#         json_path = temp_dir / "project.json"
#         og_path = temp_dir / "original_data.pkl"
#         work_path = temp_dir / "working_data.pkl"
#         feature_path = temp_dir / "feature_data.pkl"
#
#         models_dir = temp_dir / "models"
#         models_dir.mkdir(exist_ok=True)
#
#         project_copy = dict(project)
#         project_copy["models"] = []
#
#         for model_info in project.get("models", []):
#
#             save_name = (
#                 model_info["display_name"]
#                 .replace(" ", "_")
#                 .replace(".", "_")
#             )
#
#             model_path = models_dir / f"{save_name}.pkl"
#             metadata_path = models_dir / f"{save_name}.json"
#
#             joblib.dump(model_info["model"], model_path)
#
#             metadata = {
#                 "display_name": model_info["display_name"],
#                 "algorithm": model_info["algorithm"],
#                 "parameters": model_info["parameters"],
#                 "metrics": model_info["metrics"],
#                 "category": model_info.get("category", ""),
#                 "feature_columns": model_info.get("feature_columns", []),
#                 "evaluation": model_info.get("evaluation", {}),
#             }
#
#             with open(metadata_path, "w") as f:
#                 json.dump(metadata, f, indent=4)
#
#             model_copy = model_info.copy()
#             model_copy.pop("model", None)
#
#             model_copy["model_file"] = f"models/{save_name}.pkl"
#             model_copy["metadata_file"] = f"models/{save_name}.json"
#
#             project_copy["models"].append(model_copy)
#
#         if feature_df is not None and not feature_df.empty:
#             joblib.dump(
#                 _sanitize_dataframe_for_pickle(feature_df),
#                 feature_path,
#             )
#             project_copy["feature_file"] = "feature_data.pkl"
#             project_copy["has_feature_data"] = True
#         else:
#             project_copy["has_feature_data"] = False
#             project_copy["feature_file"] = None
#
#         with open(json_path, "w") as f:
#             json.dump(project_copy, f, indent=4)
#
#         joblib.dump(
#             _sanitize_dataframe_for_pickle(original_df),
#             og_path,
#         )
#         joblib.dump(
#             _sanitize_dataframe_for_pickle(mod_df),
#             work_path,
#         )
#
#         with zipfile.ZipFile(
#             icp_path,
#             "w",
#             zipfile.ZIP_DEFLATED,
#         ) as z:
#             for file in temp_dir.rglob("*"):
#                 z.write(file, file.relative_to(temp_dir))
#
#     print(f"Saved : {icp_path}")


def save_project(project, original_df, mod_df, target_path=None, feature_df=None):
    """Save the project and all trained models into one ``.icp`` archive."""
    project_name = project["project_name"]

    if target_path is None:
        icp_path = Path("Projects") / f"{project_name}.icp"
    else:
        icp_path = Path(target_path)

        if icp_path.suffix.lower() != ".icp":
            icp_path = icp_path.with_suffix(".icp")

    icp_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        json_path = temp_dir / "project.json"
        og_path = temp_dir / "original_data.pkl"
        work_path = temp_dir / "working_data.pkl"
        feature_path = temp_dir / "feature_data.pkl"

        models_dir = temp_dir / "models"
        models_dir.mkdir(exist_ok=True)

        project_copy = dict(project)
        project_copy["models"] = []

        added_by_name = {
            item.get("name"): item
            for item in project.get("added_models", [])
            if item.get("name")
        }

        used_save_names = set()

        for index, model_info in enumerate(project.get("models", []), start=1):
            display_name = model_info.get("display_name", f"model_{index}")

            save_name = _safe_filename(display_name, fallback=f"model_{index}")

            # Prevent duplicate display names from overwriting one another
            original_save_name = save_name
            duplicate_number = 2

            while save_name.lower() in used_save_names:
                save_name = f"{original_save_name}_{duplicate_number}"
                duplicate_number += 1

            used_save_names.add(save_name.lower())

            model_path = models_dir / f"{save_name}.pkl"

            # Each project model is stored as one self-contained package
            added_entry = added_by_name.get(display_name, {})

            package = create_model_package(
                model_info,
                added_entry=added_entry,
                label=added_entry.get(
                    "label",
                    model_info.get("label", project.get("label_column", "")),
                ),
            )

            if package.get("model") is None:
                raise ValueError(
                    f"Model '{display_name}' does not contain a fitted "
                    "estimator and cannot be saved."
                )

            joblib.dump(package, model_path)

            model_copy = model_info.copy()
            model_copy.pop("model", None)
            model_copy["model_file"] = f"models/{save_name}.pkl"

            model_copy.pop("metadata_file", None)

            project_copy["models"].append(model_copy)

        if feature_df is not None and not feature_df.empty:
            joblib.dump(_sanitize_dataframe_for_pickle(feature_df), feature_path)
            project_copy["feature_file"] = "feature_data.pkl"
            project_copy["has_feature_data"] = True
        else:
            project_copy["has_feature_data"] = False
            project_copy["feature_file"] = None

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(make_json_safe(project_copy), file, indent=4)

        joblib.dump(_sanitize_dataframe_for_pickle(original_df), og_path)
        joblib.dump(_sanitize_dataframe_for_pickle(mod_df), work_path)

        with zipfile.ZipFile(icp_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in temp_dir.rglob("*"):
                if file.is_file():
                    archive.write(file, file.relative_to(temp_dir))

    print(f"Saved : {icp_path}")


def save_checkpoint(project_dir, checkpoint_name, obj, project):
    file_path = Path(project_dir) / f"{checkpoint_name}.pkl"

    joblib.dump(obj, file_path)

    project["files"][checkpoint_name] = str(file_path)

    with open(
        Path(project_dir) / f"{project['project_name']}.icp",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(make_json_safe(project), file, indent=4)

# def load_project(icp_path):
#
#     with tempfile.TemporaryDirectory() as temp_dir:
#
#         temp_dir = Path(temp_dir)
#
#         with zipfile.ZipFile(icp_path, "r") as z:
#             z.extractall(temp_dir)
#
#         with open(
#             temp_dir / "project.json",
#             "r"
#         ) as f:
#
#             project = json.load(f)
#
#         project.setdefault("added_models", [])
#         project.setdefault("model_queue", [])
#
#         for added in project.get("added_models", []):
#             added.setdefault("trained", False)
#             added.setdefault("externally_added", False)
#             added.setdefault("editable_external", True)
#
#         for model_info in project.get("models", []):
#
#             model_path = temp_dir / model_info["model_file"]
#             model_info["model"] = joblib.load(model_path)
#
#             with open(
#                 temp_dir / model_info["metadata_file"]
#             ) as f:
#                 metadata = json.load(f)
#
#             model_info["algorithm"] = metadata["algorithm"]
#             model_info["parameters"] = metadata["parameters"]
#             model_info["metrics"] = metadata["metrics"]
#             model_info["category"] = metadata.get("category", "")
#             model_info["feature_columns"] = metadata.get(
#                 "feature_columns",
#                 model_info.get("feature_columns", []),
#             )
#             model_info["evaluation"] = metadata.get("evaluation", {})
#
#         try:
#             og_df = joblib.load(temp_dir / "original_data.pkl")
#             working_df = joblib.load(temp_dir / "working_data.pkl")
#             feature_df = pd.DataFrame()
#
#             if project.get("has_feature_data") and project.get("feature_file"):
#                 feature_df = joblib.load(temp_dir / project["feature_file"])
#         except ModuleNotFoundError as error:
#             print(f"Missing mdoule: {error}.")
#             print(error)
#             raise
#
#     return project, og_df, working_df, feature_df


def load_project(icp_path):
    """Load projects created by both the new and original save formats."""
    icp_path = Path(icp_path)

    if not icp_path.exists():
        raise FileNotFoundError(f"Project file was not found: {icp_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        with zipfile.ZipFile(icp_path, "r") as archive:
            archive.extractall(temp_dir)

        project_json_path = temp_dir / "project.json"

        if not project_json_path.exists():
            raise FileNotFoundError(
                "The project archive does not contain project.json."
            )

        with open(project_json_path, "r", encoding="utf-8") as file:
            project = json.load(file)

        project.setdefault("added_models", [])
        project.setdefault("model_queue", [])
        project.setdefault("models", [])

        for added in project.get("added_models", []):
            added.setdefault("trained", False)
            added.setdefault("externally_added", False)
            added.setdefault("editable_external", True)

        for model_info in project.get("models", []):
            model_file = model_info.get("model_file")

            if not model_file:
                raise ValueError(
                    "A project model record is missing its 'model_file' value."
                )

            model_path = temp_dir / model_file

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Project model file was not found: {model_file}"
                )

            loaded_payload = joblib.load(model_path)
            loaded_model, package_metadata, _ = unpack_model_package(loaded_payload)

            if loaded_model is None:
                raise ValueError(
                    "The model package does not contain a fitted estimator: "
                    f"{model_file}"
                )

            model_info["model"] = loaded_model

            # New project archives keep all model metadata in the PKL package
            metadata = package_metadata
            metadata_file = model_info.get("metadata_file")

            if metadata_file and (temp_dir / metadata_file).exists():
                with open(
                    temp_dir / metadata_file,
                    "r",
                    encoding="utf-8",
                ) as file:
                    legacy_metadata = json.load(file)

                # New package metadata
                metadata = {**legacy_metadata, **metadata}

            defaults = {
                "display_name": "",
                "algorithm": "",
                "category": "",
                "label": project.get("label_column", ""),
                "feature_columns": [],
                "parameters": {},
                "common_parameters": {},
                "required_parameters": {},
                "advanced_parameters": {},
                "metrics": {},
                "evaluation": {},
                "package_version": None,
            }

            for key, default in defaults.items():
                model_info[key] = metadata.get(
                    key,
                    model_info.get(key, default),
                )

        try:
            original_path = temp_dir / "original_data.pkl"
            working_path = temp_dir / "working_data.pkl"

            if not original_path.exists():
                raise FileNotFoundError(
                    "The project archive does not contain original_data.pkl."
                )

            if not working_path.exists():
                raise FileNotFoundError(
                    "The project archive does not contain working_data.pkl."
                )

            og_df = joblib.load(original_path)
            working_df = joblib.load(working_path)

            feature_df = pd.DataFrame()

            if project.get("has_feature_data") and project.get("feature_file"):
                feature_path = temp_dir / project["feature_file"]

                if feature_path.exists():
                    feature_df = joblib.load(feature_path)

        except ModuleNotFoundError as error:
            # raise ModuleNotFoundError(
            #     f"Failed to load project data: {error}. "
            #     "This project may contain pandas extension arrays requiring "
            #     "pyarrow. Install pyarrow or recreate the project with "
            #     "compatible dtypes."
            # ) from error
            print(f"Missing module: {error}.")
            print(error)
            raise

    return project, og_df, working_df, feature_df


# def load_project(icp_path):
#     icp_path = Path(icp_path)
#
#     parent = icp_path.parent
#
#     with open(icp_path, "r") as file:
#         project = json.load(file)
#
#     og_path = parent / "selected_data.pkl"
#     working_path = parent / "working_data.pkl"
#
#     og_df = joblib.load(og_path)
#     working_df = joblib.load(working_path)
#
#     return project, og_df, working_df
