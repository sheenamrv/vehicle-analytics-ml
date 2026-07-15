import json
import joblib
import zipfile
import tempfile
from pathlib import Path

import pandas as pd


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


def save_project(project, original_df, mod_df, target_path=None, feature_df=None):

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
        
        for model_info in project.get("models", []):
            
            save_name = (model_info["display_name"].replace(" ", "_").replace(".", "_"))
            
            model_path = (models_dir / f"{save_name}.pkl")
            
            metadata_path = (models_dir / f"{save_name}.json")
            
            joblib.dump(model_info["model"], model_path)
            
            metadata = {
                "display_name" : model_info["display_name"],
                "algorithm" : model_info["algorithm"],
                "parameters" : model_info["parameters"],
                "metrics" : model_info["metrics"]
            }
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
                
            model_copy = model_info.copy()
            model_copy.pop("model", None)
            
            model_copy["model_file"] = (f"models/{save_name}.pkl")
            model_copy["metadata_file"] = (f"models/{save_name}.json")
            
            project_copy["models"].append(model_copy)
            
        if feature_df is not None and not feature_df.empty:
            joblib.dump(_sanitize_dataframe_for_pickle(feature_df), feature_path)
            project_copy["feature_file"] = "feature_data.pkl"
            project_copy["has_feature_data"] = True
        else:
            project_copy["has_feature_data"] = False
            project_copy["feature_file"] = None

        with open(json_path, "w") as f:
            json.dump(project_copy, f, indent=4)

        joblib.dump(_sanitize_dataframe_for_pickle(original_df), og_path)
        joblib.dump(_sanitize_dataframe_for_pickle(mod_df), work_path)
            
        # with zipfile.ZipFile(icp_path, "w", zipfile.ZIP_DEFLATED) as z:
        #     z.write(json_path, "project.json")
        #     z.write(og_path, "original_data.pkl")
        #     z.write(work_path, "working_data.pkl")
        
        with zipfile.ZipFile(icp_path, "w", zipfile.ZIP_DEFLATED) as z:
            
            for file in temp_dir.rglob("*"):
                z.write(file, file.relative_to(temp_dir))
    
    print(f"Saved : {icp_path}")
    # og_data_file = project_dir / "selected_data.pkl"
    # mod_data_file = project_dir / "working_data.pkl"

    # icp_file = project_dir / f"{project_name}.icp"

    # project["files"] = {
    #     "selected_data" : str(og_data_file),
    #     "working_data" : str(mod_data_file)
    # }

    # with open(icp_file, "w") as f:
    #     json.dump(project, f, indent=4)

    # og_df = joblib.dump(original_df, og_data_file)
    # working_df = joblib.dump(mod_df, mod_data_file)
    # print(f"Project saved to {icp_file}")


def save_checkpoint(project_dir, checkpoint_name, obj, project):

    file_path = Path(project_dir) / f"{checkpoint_name}.pkl"

    joblib.dump(obj, file_path)

    project["files"][checkpoint_name] = str(file_path)
    
    with open(Path(project_dir) / f"{project['project_name']}.icp", "w") as f:

        json.dump(project,f, indent=4)
    
def load_project(icp_path):

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)
                
        with zipfile.ZipFile(icp_path, "r") as z:
            z.extractall(temp_dir)

        with open(
            temp_dir / "project.json",
            "r"
        ) as f:

            project = json.load(f)

        project.setdefault("added_models", [])
        project.setdefault("model_queue", [])
        for added in project.get("added_models", []):
            added.setdefault("trained", False)
            added.setdefault("externally_added", False)
            added.setdefault("editable_external", True)

        for model_info in project.get("models", []):
            
            model_path = (temp_dir/model_info["model_file"])
            model_info["model"] = (joblib.load(model_path))
            
            with open(
                temp_dir / model_info["metadata_file"]
            ) as f:
                
                metadata = json.load(f)
            
            model_info["algorithm"] = (metadata["algorithm"])
            model_info["parameters"] = (metadata["parameters"])
            model_info["metrics"] = (metadata["metrics"])
            
        try:
            og_df = joblib.load(
                temp_dir / "original_data.pkl"
            )
            working_df = joblib.load(
                temp_dir / "working_data.pkl"
            )
            feature_df = pd.DataFrame()
            if project.get("has_feature_data") and project.get("feature_file"):
                feature_df = joblib.load(temp_dir / project["feature_file"])
        except ModuleNotFoundError as error:
            # raise ModuleNotFoundError(
            #     f"Failed to load project data: {error}. "
            #     "This project may contain pandas extension arrays requiring pyarrow. "
            #     "Install pyarrow or recreate the project with compatible dtypes."
            # ) from error
            print(f"Missing mdoule: {error}.")
            print(error)
            raise
    
    return (project, og_df, working_df, feature_df)
    # icp_path = Path(icp_path)

    # parent = icp_path.parent

    # with open(icp_path, "r") as file:
    #     project = json.load(file)

    # og_path = (parent / "selected_data.pkl")
    # working_path = (parent / "working_data.pkl")

    # og_df = joblib.load(og_path)
    # working_df = joblib.load(working_path)

    # return project, og_df, working_df