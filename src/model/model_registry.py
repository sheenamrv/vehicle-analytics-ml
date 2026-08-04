import json
import joblib
import re
from pathlib import Path

from src.data.process import create_model_package, make_json_safe
from src.model.model_utils import prepare_training_data


def add_model(
    project,
    model,
    model_name,
    algorithm,
    parameters,
    metrics,
    feature_column,
):

    # in icp folder it sets the json to have an empty list
    project.setdefault("models", [])

    project["models"].append({
        "display_name": model_name,
        "algorithm": algorithm,
        "model": model,
        "parameters": parameters,
        "metrics": metrics,
        "feature_columns": list(feature_column),
    })


def add_model_queue():
    return None


def list_models(project):

    if not project.get("models"):
        print("No trained models.")
        return

    for i, model in enumerate(project["models"], 1):
        print(
            f"{i}. {model['display_name']} {model['algorithm']}"
        )


def delete_model(project, display_name):

    project["models"] = [
        model
        for model in project.get("models", [])
        if model["display_name"] != display_name
    ]


def delete_model_queue():
    return None


def edit_model(project):

    list_models(project)

    if not project.get("models"):
        return

    index = int(input("Model #: ").strip()) - 1

    model = project["models"][index]

    new_name = input("New display name: ").strip()

    if new_name:
        model["display_name"] = new_name

    for key in model["parameters"]:

        value = input(
            f"{key} {model['parameters'][key]} "
        ).strip()

        if value:
            current = model["parameters"][key]

            if isinstance(current, int):
                model["parameters"][key] = int(value)
            elif isinstance(current, float):
                model["parameters"][key] = float(value)
            else:
                model["parameters"][key] = value


def edit_model_queue():
    return None


def clear_training_queue():
    return None


def get_training_queue():
    return None


def enable_queue_model():
    return None


def disable_queue_model():
    return None


def reorder_queue():
    return None


def show_model_details(project):

    list_models(project)

    if not project.get("models"):
        return

    index = int(input("Model #: ").strip()) - 1

    model = project["models"][index]

    print("\nMODEL DETAILS")
    print(
        f"Name       : "
        f"{model['display_name']}"
    )
    print(
        f"Algorithm  : "
        f"{model['algorithm']}"
    )
    print(
        f"Parameters : "
        f"{model['parameters']}"
    )
    print(
        f"Metrics    : "
        f"{model['metrics']}"
    )


def run_saved_model(project, working_df):

    list_models(project)

    if not project.get("models"):
        return

    index = int(input("Model #: ").strip()) - 1

    model = project["models"][index]

    X, y = prepare_training_data(
        working_df,
        project["label_column"],
    )

    predictions = model["model"].predict(X)

    print("First 20 predictions: ")
    print(predictions[:20])


def select_saved_models(project):

    models = project.get("models", [])

    if not models:
        print("No saved modesl")
        return []

    list_models(project)

    selection = input("Select models: ").strip()

    try:
        indices = [
            int(x.strip()) - 1
            for x in selection.split(",")
        ]

        selected = [
            models[i]
            for i in indices
            if 0 <= i < len(models)
        ]

        return selected

    except Exception:
        print("Invalid selection")
        return []


def _safe_export_name(value, fallback="model"):
    """Create a filename-safe export name."""
    value = str(value or fallback).strip()
    value = re.sub(r'[<>:"/\\|?*]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._")

    return value or fallback


# def export_model(project):
#
#     if not project.get("models"):
#         return
#
#     list_models(project)
#
#     try:
#         index = int(input("\nModel #: ").strip()) - 1
#
#         if index < 0 or index >= len(project["models"]):
#             print("\nInvalid model number.")
#             return
#
#         model_info = project["models"][index]
#         export_name = input("Export filename: ").strip()
#
#         if not export_name:
#             export_name = model_info["display_name"]
#
#         # Create export directory if needed
#         export_dir = Path("ExportedModels")
#         export_dir.mkdir(parents=True, exist_ok=True)
#
#         model_path = export_dir / f"{export_name}.pkl"
#
#         # Export one self-contained package rather than creating a separate
#         # metadata file. The resulting PKL restores the fitted estimator and
#         # all saved evaluation information in one import operation.
#         added_entry = next(
#             (
#                 item
#                 for item in project.get("added_models", [])
#                 if item.get("name") == model_info.get("display_name")
#             ),
#             {},
#         )
#
#         package = create_model_package(
#             model_info,
#             added_entry=added_entry,
#             label=added_entry.get(
#                 "label",
#                 project.get("label_column", ""),
#             ),
#         )
#
#         joblib.dump(package, model_path)
#
#         print("\nModel exported successfully.")
#         print(f"Model Package: {model_path}")
#
#     except ValueError:
#         print("\nPlease enter a valid model number.")
#
#     except Exception as e:
#         print(f"\nExport failed:\n{e}")


def export_model(project):
    """Export one self-contained model PKL and an optional metadata backup."""

    if not project.get("models"):
        print("No trained models.")
        return

    list_models(project)

    try:
        index = int(input("\nModel #: ").strip()) - 1

        if index < 0 or index >= len(project["models"]):
            print("\nInvalid model number.")
            return

        model_info = project["models"][index]
        export_name = input("Export filename: ").strip()

        if not export_name:
            export_name = model_info.get(
                "display_name",
                "model",
            )

        export_name = _safe_export_name(
            export_name,
            fallback="model",
        )

        # Create export directory if needed
        export_dir = Path("ExportedModels")
        export_dir.mkdir(parents=True, exist_ok=True)

        model_path = export_dir / f"{export_name}.pkl"
        metadata_path = export_dir / f"{export_name}.json"

        added_entry = next(
            (
                item
                for item in project.get("added_models", [])
                if item.get("name") == model_info.get("display_name")
            ),
            {},
        )

        # Export one self-contained package
        package = create_model_package(
            model_info,
            added_entry=added_entry,
            label=added_entry.get(
                "label",
                model_info.get(
                    "label",
                    project.get("label_column", ""),
                ),
            ),
        )

        if package.get("model") is None:
            raise ValueError(
                "The selected record does not contain a fitted model."
            )

        joblib.dump(package, model_path)

        # Keep a JSON copy only as an optional backup
        metadata = {
            "package_type": package.get("package_type"),
            "package_version": package.get("package_version"),
            "display_name": package.get("display_name", ""),
            "category": package.get("category", ""),
            "algorithm": package.get("algorithm", ""),
            "label": package.get("label", ""),
            "feature_columns": package.get(
                "feature_columns",
                [],
            ),
            "parameters": package.get("parameters", {}),
            "common_parameters": package.get(
                "common_parameters",
                {},
            ),
            "required_parameters": package.get(
                "required_parameters",
                {},
            ),
            "advanced_parameters": package.get(
                "advanced_parameters",
                {},
            ),
            "metrics": package.get("metrics", {}),
            "evaluation": package.get("evaluation", {}),
            "model_file": model_path.name,
        }

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                make_json_safe(metadata),
                file,
                indent=4,
            )

        print("\nModel exported successfully.")
        print(f"Model Package : {model_path}")
        print(f"Metadata Copy : {metadata_path}")
        print(
            "The PKL is self-contained; the JSON file is an "
            "optional readable backup."
        )

    except ValueError as error:
        print(f"\nExport failed:\n{error}")

    except Exception as error:
        print(f"\nExport failed:\n{error}")


def export_predictions(df, results):

    output = df.copy()

    for result in results:

        output[
            result["name"] + "_prediction"
        ] = result["predictions"]

        file_name = input("Output CSV name: ").strip()

        if not file_name:
            file_name = "predictions"

        output.to_csv(
            f"{file_name}.csv",
            index=False,
        )

        print(f"Saved {file_name}.csv")
