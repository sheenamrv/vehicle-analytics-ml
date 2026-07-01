import tkinter as tk
from tkinter import Tk, filedialog
import streamlit as st
import os
from pathlib import Path

from src.data.test_load import *
from src.data.process import *

def open_file_dialog():

    root = Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)

    downloads_dir = Path.home() / "Downloads"

    file_path = filedialog.askopenfilename(
        title="Select Dataset",
        initialdir=str(downloads_dir),
        filetypes = [
            ("Supported Files", "*.csv *.xlsx *.xls *.mat"),
            ("CSV Files", "*.csv"),
            ("Excel Files", "*.xlsx *.xls"),
            ("MATLAB Files", "*.mat"),
        ]
    )

    root.quit()
    root.destroy()

    # st.write(f"DEBUG: {file_path}")

    if file_path:
        st.session_state["selected_file"] = file_path
        st.rerun()

def open_project_dialog():

    root = Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)

    project_dir = Path("Projects")
    project_dir.mkdir(exist_ok=True)

    file_path = filedialog.askopenfilename(
        title="Open Project",
        initialdir=str(project_dir.resolve()),
        filetypes = [
            ("ICP Project Files", "*.icp"),
            ("All Files", "*.*")
        ]
    )

    root.quit()
    root.destroy()

    # st.write(f"DEBUG: {file_path}")

    if file_path:
        st.session_state["selected_project"] = file_path
        st.rerun()

def load_page():

    startup_choice = st.radio("Select an option",
                              ["New Project", "Load Project"])
    
    if startup_choice == "New Project":

        if st.button("Browse Dataset"):
            open_file_dialog()
        
        if "selected_file" in st.session_state:

            file = st.session_state["selected_file"]
            st.success(f"Loaded: {Path(file).name}")

            dataset = get_datasets(file)
            dataset = st.selectbox("Select Dataset", dataset)

            all_columns = get_available_columns(file, dataset)
            selected_col = st.multiselect(
                "Columns to keep",
                all_columns,
                default=all_columns
            )

            label_col = st.selectbox(
                "Label Column",
                selected_col
            )

            project_name = st.text_input("Project Name")

            if st.button("Create Project"):

                og_df = select_col(file, dataset, all_columns)
                working_df = og_df[selected_col].copy()

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

                st.session_state.project = project
                st.session_state.og_df = og_df
                st.session_state.working_df = working_df

                save_project(project, og_df, working_df)

                st.rerun()

    if startup_choice == "Load Project":

        if st.button("Browse Dataset"):
            open_project_dialog()
            st.write("Session state:", st.session_state)
       
        if "selected_project" in st.session_state:

            st.write("Before load_project")

            project, og_df, working_df, feature_df = load_project(st.session_state["selected_project"])

            st.write("After load_project")

            st.session_state.project = project
            st.session_state.og_df = og_df
            st.session_state.working_df = working_df
            st.session_state.feature_df = feature_df

            st.success("Project loaded")
            # st.rerun()