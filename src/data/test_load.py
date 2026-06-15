"""
Test loading files and selecting specific columns
"""
from pathlib import Path
import pandas as pd
from scipy.io import loadmat
import numpy as np
# from pymatreader import read_mat


def get_datasets(file):

    file_ext = Path(file).suffix.lower()

    if file_ext == ".csv":
        return ["Data"]
    elif file_ext in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(file)
        return xls.sheet_names
    elif file_ext == ".mat":
        mat = loadmat(file)

        return [key for key in mat.keys() if not key.startswith("__")]
    
def get_available_columns(file, dataset=None):

    file_ext = Path(file).suffix.lower()

    if file_ext == ".csv":
        return pd.read_csv(file, nrows=0).columns.to_list()
    elif file_ext in [".xlsx", ".xls"]:
        return pd.read_excel(file, sheet_name=dataset, nrows=0).columns.to_list()
    elif file_ext == ".mat":
        mat = loadmat(file)

        arr = mat[dataset]

        if arr.ndim == 1:
            return [dataset]
        elif arr.ndim == 2:
            return [f"{dataset}_{i}" for i in range(arr.shape[1])]
        
    else:
        raise ValueError(f"Unsupported format {file}")
    

def select_col(file, dataset, cols):

    file_ext = Path(file).suffix.lower()

    if file_ext == ".csv":
        return pd.read_csv(file, usecols=cols)
    elif file_ext in [".xlsx", ".xls"]:
        return pd.read_excel(file, sheet_name=dataset, usecols=cols)
    elif file_ext == ".mat":
        mat = loadmat(file)

        arr = mat[dataset]

        if arr.ndim == 1:
            df = pd.DataFrame({dataset:arr})
        else: 
            all_cols = [f"{dataset}_{i}" for i in range(arr.shape[1])]
            df = pd.DataFrame(arr, columns=all_cols)
        return df[cols]
    else:
        raise ValueError(f"Could not load {cols}")

def add_col(working_df, og_df):

    print("Available Cols")

    for col in og_df.columns:

        if col not in working_df.columns:
            print(col)

    cols = input("Columns to add: ")

    cols_to_add = [c.strip() for c in cols.split(",")]

    for col in cols_to_add:
        if col in og_df.columns:
            working_df[col] = og_df[col]

    return working_df

def remove_col(working_df, og_df):

    print("Current Cols")

    for col in working_df.columns:
        print(col)

    cols = input("Columns to remove: ")

    cols_to_remove = [c.strip() for c in cols.split(",")]

    return working_df.drop(columns=cols_to_remove, errors="ignore")

def change_dtype(working_df):

    print("Current Cols: ")

    for col in working_df.columns:
        print(f"{col} : {working_df[col].dtype}")

    col = input("Columns: ").strip()
    dtype = input("New dtype: ").strip()

    try:
        
        if dtype in ["int", "int64"]:
            working_df[col] = (pd.to_numeric(working_df[col], errors="raise").astype(int))
        elif dtype in ["float", "float64"]:
            working_df[col] = pd.to_numeric(working_df[col], errors="raise")
        else:
            working_df[col] = (working_df[col].astype(dtype))

        print(f"{col} is now {dtype}")

    except Exception as e:
        print(e)
    
    return working_df

def set_label(working_df):

    print("Columns")

    for col in working_df.columns:
        print(col)

    label = input("Label Col: ").strip()

    return label