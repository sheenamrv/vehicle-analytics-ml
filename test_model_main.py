from src.data.process import load_project, save_project
from pathlib import Path

from src.model.model_registry import *
from src.model.model_training import *
from src.model.model_utils import *
from src.model.supervised_model import *

def display_project_info(project):

    print("\n------------------------")
    print("PROJECT INFORMATION")
    print("------------------------")

    print(
        f"Project Name : "
        f"{project['project_name']}"
    )

    print(
        f"Dataset      : "
        f"{project.get('dataset')}"
    )

    print(
        f"Label Column : "
        f"{project.get('label_column')}"
    )

    print(
        f"Columns      : "
        f"{project.get('selected_columns')}"
    )

    print(
        f"Saved Models : "
        f"{len(project.get('models', []))}"
    )


def display_dataframe(df, title):

    print(f"\n------------------------")
    print(title)
    print("------------------------")

    print(df.head())

    print(
        f"\nRows: "
        f"{df.shape[0]}"
    )

    print(
        f"Columns: "
        f"{df.shape[1]}"
    )


def project_menu(
    project,
    og_df,
    working_df
):

    project_modified = False

    while True:

        print("\n================================")
        print("PROJECT MENU")
        print("================================")

        print("1 - Project Information")
        print("2 - View Original Dataset")
        print("3 - View Working Dataset")
        print("4 - List Saved Models")
        print("5 - View Model Details")
        print("6 - Train New Model")
        print("7 - Test Saved Models(s)")
        print("8 - Edit Saved Model")
        print("9 - Delete Saved Model")
        print("10 - Export Saved Model")
        print("11 - Save Project")
        print("0 - Exit")

        choice = input(
            "\nChoice: "
        ).strip()

        if choice == "1":

            display_project_info(
                project
            )

        elif choice == "2":

            display_dataframe(
                og_df,
                "ORIGINAL DATA"
            )

        elif choice == "3":

            display_dataframe(
                working_df,
                "WORKING DATA"
            )

        elif choice == "4":

            list_models(
                project
            )

        elif choice == "5":

            show_model_details(
                project
            )

        elif choice == "6":

            train_new_model(
                project,
                working_df
            )
            
            project_modified = True


        elif choice == "7":

            test_saved_models(
                project, working_df
            )

            project_modified = True

        elif choice == "8":

            edit_model(
                project
            )

            project_modified = True

        elif choice == "9":

            list_models(
                project
            )

            model_name = input(
                "\nModel display name to delete: "
            ).strip()

            confirm = input(
                f"Delete '{model_name}' (y/n): "
            ).lower()

            if confirm == "y":

                delete_model(
                    project,
                    model_name
                )

                print(
                    "\nModel deleted."
                )
                project_modified = True


        elif choice == "10":

            export_model(
                project
            )

        elif choice == "11":

            save_project(
                project,
                og_df,
                working_df
            )

            print(
                "\nProject saved."
            )

            project_modified = False


        elif choice == "0":

            if project_modified:
                save_choice = input(
                    "\nSave project before exit (y/n): "
                ).lower()

                if save_choice == "y":

                    save_project(
                        project,
                        og_df,
                        working_df
                    )

                    print(
                        "\nProject saved."
                    )

                print(
                    "\nClosing project."
                )

            break

        else:

            print(
                "\nInvalid option."
            )


def test_load_project():

    print("\n========================")
    print("PROJECT LOADER")
    print("========================")

    project_name = input(
        "Project name: "
    ).strip()

    icp_path = Path(
        f"Projects/{project_name}.icp"
    )

    if not icp_path.exists():

        print(
            f"\nError: "
            f"{icp_path} does not exist."
        )

        return

    try:

        project, og_df, working_df, feature_df = (
            load_project(
                icp_path
            )
        )

        print(
            f"\nProject "
            f"'{project['project_name']}' "
            f"loaded successfully."
        )

        project_menu(
            project,
            og_df,
            working_df
        )

    except Exception as e:

        print(
            f"\nError loading project:"
        )

        print(e)


if __name__ == "__main__":

    test_load_project()