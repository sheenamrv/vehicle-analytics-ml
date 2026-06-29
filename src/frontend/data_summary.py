import pandas as pd

'''
    Contains helper functions that generate summary tables displayed for the front end.
    Only prepare pandas Dataframe, do not modify the original dataset.
'''

## Returns overall dataset summary for the Import and Visualization tab
## Add new summary metric here

def file_summary(df):
    """Build the compact dataset summary shown in the Import and Visualization tabs."""
    if df.empty:
        return pd.DataFrame(columns=["metric", "value"])

    return pd.DataFrame(
        [
            ("Rows", len(df)),
            ("Columns", len(df.columns)),
            ("Missing Cells", int(df.isna().sum().sum())),
            ("Duplicate Rows", int(df.duplicated().sum())),
            ("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"),
        ],
        columns=["metric", "value"],
    )

## Returns a table describing each column
## Add new per-column diagnostics here
def missing_summary(df):
    """Build per-column missing-value diagnostics for the Import tab."""
    if df.empty:
        return pd.DataFrame(columns=["column", "dtype", "missing", "missing_pct"])

    missing = df.isna().sum()
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing": [int(missing[col]) for col in df.columns],
            "missing_pct": [
                f"{(missing[col] / len(df) * 100):.2f}%" if len(df) else "0.00%"
                for col in df.columns
            ],
        }
    )
