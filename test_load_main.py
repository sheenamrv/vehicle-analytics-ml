from src.data.test_load import *
from src.feature.feature import *
from src.data.process import *
from src.preprocessing.preprocessing import *
from src.visualize.visualization import create_visualization

from pathlib import Path

def run_project(project, og_df, working_df):

    label_col = project.get("label_column")
    project_modified = False

    while True:

        print("\n====================")
        print("PROJECT MENU")
        print("====================")

        print("1 - Show Current Columns")
        print("2 - Add Columns")
        print("3 - Remove Columns")
        print("4 - Change Data Type")
        print("5 - Change Label Column")
        print("6 - Run Feature Extraction")
        print("7 - Save Project")
        print("8 - Fill Missing Values")
        print("9 - Standardize")
        print("10 - Normalize")
        print("11 - Visualize")
        print("0 - Exit")

        choice = input("\nChoice: ").strip()

        if choice == "1":

            print("\nCurrent Columns:")

            for col in working_df.columns:
                print(col)

        elif choice == "2":

            working_df = add_col(
                working_df,
                og_df
            )

            project["selected_columns"] = list(working_df.columns)
            project_modified = True

        elif choice == "3":

            working_df = remove_col(
                working_df,
                og_df
            )

            project["selected_columns"] = list(working_df.columns)
            project_modified = True

        elif choice == "4":

            working_df = change_dtype(
                working_df
            )

            project["column_types"] = {col: str(working_df[col].dtype) for col in working_df.columns}
            project_modified = True

        elif choice == "5":

            label_col = set_label(
                working_df
            )

            project["selected_columns"] = label_col
            project_modified = True

        elif choice == "6":

            results = feature_extract(
                working_df,
                list(working_df.columns)
            )

            display_configuration(
                project["dataset"],
                list(working_df.columns),
                label_col,
                results
            )

        elif choice == "7":

            project["selected_columns"] = (
                list(working_df.columns)
            )

            project["label_column"] = (
                label_col
            )

            project["column_types"] = {
                col: str(
                    working_df[col].dtype
                )
                for col in working_df.columns
            }

            save_project(
                project,
                og_df,
                working_df
            )

            project_modified = False
            print("\nProject saved.")

        elif choice == "8":

            working_df, column, method = (
                fill_missing_values(
                    working_df
                )
            )

            project["preprocessing"].append({
                "operation": "fill_missing",
                "column": column,
                "method": method
            })
          
            project_modified = True

        elif choice == "9":

            working_df, column, mean, std = (
                standardize_col(
                    working_df
                )
            )

            project["preprocessing"].append({
                "operation": "standardize",
                "column": column,
                "mean": mean,
                "std": std
            })

            project_modified = True

        elif choice == "10":

            working_df, column = (
                normalize_col(
                    working_df
                )
            )

            project["preprocessing"].append({
                "operation": "normalize",
                "column": column
            })

            project_modified = True

        elif choice == "11":

            create_visualization(
                working_df
            )

        elif choice == "0":

            if project_modified:
                save_choice = input("Save changes y/n: ").lower()

                if save_choice == "y":

                    project["selected_columns"] = (
                        list(working_df.columns)
                    )

                    project["label_column"] = (
                        label_col
                    )

                    project["column_types"] = {
                        col: str(
                            working_df[col].dtype
                        )
                        for col in working_df.columns
                    }

                    save_project(
                        project,
                        og_df,
                        working_df
                    )

                    print("\nProject saved.")

            break

        else:

            print("\nInvalid option.")


if __name__ == "__main__":

    try:

        print("\n====================")
        print("PROJECT STARTUP")
        print("====================")

        print("1 - New Project")
        print("2 - Load Project")

        startup_choice = (
            input("\nChoice: ")
            .strip()
        )

        # ====================================
        # NEW PROJECT
        # ====================================

        if startup_choice == "1":

            data_dir = Path("data")

            file_name = input(
                "\nFile path: "
            ).strip()

            file = data_dir / file_name

            datasets = get_datasets(
                file
            )

            print("\nAvailable Datasets:")

            for i, ds in enumerate(datasets):
                print(f"{i} : {ds}")

            dataset = input(
                "\nDataset: "
            ).strip()

            all_columns = (
                get_available_columns(
                    file,
                    dataset
                )
            )

            print("\nAvailable Columns:")

            for col in all_columns:
                print(col)

            selected = input(
                "\nColumns to keep: "
            )

            selected_columns = [
                c.strip()
                for c in selected.split(",")
            ]

            og_df = select_col(
                file,
                dataset,
                all_columns
            )

            working_df = (
                og_df[selected_columns]
                .copy()
            )

            label_col = input(
                "\nLabel column: "
            ).strip()

            project_name = input(
                "\nProject name: "
            ).strip()

            project = {
                "project_name": project_name,
                "file_path": str(file),
                "dataset": dataset,
                "selected_columns":
                    list(
                        working_df.columns
                    ),
                "label_column":
                    label_col,
                "column_types": {
                    col: str(
                        working_df[col].dtype
                    )
                    for col in working_df.columns
                },
                "preprocessing": [],
                "visualizations": []
            }

            save_project(
                project,
                og_df,
                working_df
            )

        # ====================================
        # LOAD PROJECT
        # ====================================

        elif startup_choice == "2":

            project_name = input(
                "\nProject name: "
            ).strip()

            icp_path = (
                f"Projects/"
                f"{project_name}.icp"
            )

            project, og_df, working_df, feature_df = (
                load_project(
                    icp_path
                )
            )

            print("\nProject loaded.")

            print("\nProject Info:")
            print(project)

            print("\nCurrent Data:")
            print(
                working_df.head()
            )

        else:

            print("Invalid option.")
            exit()

        # ====================================
        # SHARED MENU
        # ====================================

        run_project(
            project,
            og_df,
            working_df
        )

    except FileNotFoundError as e:

        print(
            f"File not found: {e}"
        )

    except Exception as e:

        print(
            f"Unexpected error: {e}"
        )