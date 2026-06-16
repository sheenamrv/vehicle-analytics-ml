import json
import joblib
import zipfile
import tempfile
from pathlib import Path

def save_project(project, original_df, mod_df):

    project_name = project["project_name"]
    # project_dir = Path("Projects") / project_name

    # project_dir.mkdir(parents=True, exist_ok=True)

    icp_path = Path(f"Projects/{project_name}.icp")

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        json_path = temp_dir / "project.json"
        og_path = temp_dir / "original_data.pkl"
        work_path = temp_dir / "working_data.pkl"
        
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
            
        with open(json_path, "w") as f:
            json.dump(project_copy, f, indent=4)

        joblib.dump(original_df, og_path)
        joblib.dump(mod_df, work_path)
            
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
            
        og_df = joblib.load(
            temp_dir / "original_data.pkl"
        )

        working_df = joblib.load(
            temp_dir / "working_data.pkl"
        )
    
    return (project, og_df, working_df)
    # icp_path = Path(icp_path)

    # parent = icp_path.parent

    # with open(icp_path, "r") as file:
    #     project = json.load(file)

    # og_path = (parent / "selected_data.pkl")
    # working_path = (parent / "working_data.pkl")

    # og_df = joblib.load(og_path)
    # working_df = joblib.load(working_path)

    # return project, og_df, working_df