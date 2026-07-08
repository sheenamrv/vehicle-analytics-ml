import json
import joblib
from pathlib import Path
import pandas as pd

from src.model.model_utils import prepare_training_data

def add_model(project, model, model_name, algorithm, parameters, metrics, feature_column):
    
    # in icp folder it sets the json to have an empty list
    project.setdefault("models", [])
    
    project["models"].append({
        "display_name": model_name,
        "algorithm": algorithm,
        "model": model,
        "parameters": parameters,
        "metrics" : metrics,
        "feature_columns" : list(feature_column)
    })

def list_models(project):
    
    if not project.get("models"):
        print("No trained models.")
        return
    
    for i, model in enumerate(project["models"], 1):
        print(
            f"{i}. {model['display_name']} {model['algorithm']}"
        )

def delete_model(project, display_name):
    
    project["models"] = [model for model in project.get("models", []) if model["display_name"] != display_name]

def edit_model(project):
    
    list_models(project)
    
    if not project.get("models"):
        return
    
    index = (int(input("Model #: ").strip()) - 1)
    
    model = project["models"][index]
    
    new_name = input("New display name: ").strip()
    
    if new_name:
        model["display_name"] = (new_name)
        
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
            

def show_model_details(project):
    
    list_models(project)
    
    if not project.get("models"):
        return
    
    index = (int(input("Model #: ").strip()) - 1)
    
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
    
    index = (int(input("Model #: ").strip()) - 1)
    
    model = (project["models"][index])
    
    X, y = prepare_training_data(working_df, project["label_column"])
    
    predictions = (model["model"].predict(X))
    
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
        indices = [int(x.strip()) - 1 for x in selection.split(",")]
        
        selected = [models[i] for i in indices if 0 <= i < len(models)]
        
        return selected
    
    except Exception:
        print("Invalid selection")
        return []
    
def export_model(project):
    
    if not project.get("models"):
        return
    
    list_models(project)
    
    try:

        index = (
            int(
                input(
                    "\nModel #: "
                ).strip()
            ) - 1
        )

        if (
            index < 0 or
            index >= len(project["models"])
        ):

            print(
                "\nInvalid model number."
            )

            return

        model_info = (
            project["models"][index]
        )

        export_name = input(
            "Export filename: "
        ).strip()

        if not export_name:
            export_name = (
                model_info[
                    "display_name"
                ]
            )

        # Create export directory if needed
        export_dir = Path(
            "ExportedModels"
        )

        export_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # File paths
        model_path = (
            export_dir /
            f"{export_name}.pkl"
        )

        metadata_path = (
            export_dir /
            f"{export_name}.json"
        )

        # Save trained model
        joblib.dump(
            model_info["model"],
            model_path
        )

        # Save metadata
        metadata = {
            "display_name":
                model_info[
                    "display_name"
                ],
            "algorithm":
                model_info[
                    "algorithm"
                ],
            "parameters":
                model_info.get(
                    "parameters",
                    {}
                ),
            "metrics":
                model_info.get(
                    "metrics",
                    {}
                ),
            "feature_columns":
                model_info.get(
                    "feature_columns",
                    []
                )
        }

        with open(
            metadata_path,
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

        print(
            "\nModel exported "
            "successfully."
        )

        print(
            f"Model File   : "
            f"{model_path}"
        )

        print(
            f"Metadata File: "
            f"{metadata_path}"
        )

    except ValueError:

        print(
            "\nPlease enter a "
            "valid model number."
        )

    except Exception as e:

        print(
            f"\nExport failed:\n{e}"
        )
    
def export_predictions(df, results):
    
    output = df.copy()
    
    for result in results:
        
        output[result["name"] + "_prediction"] = result["predictions"]
        
        file_name = input("Output CSV name: ").strip()
        
        if not file_name:
            file_name = ("predictions")
            
        output.to_csv(f"{file_name}.csv", index=False)
        
        print(f"Saved {file_name}.csv")
        
