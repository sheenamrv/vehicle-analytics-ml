import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from pprint import pprint

def format_value(value):
    if value == 0:
        return 0

    if abs(value) < 0.001:
        return f"{value:.3e}"

    return round(value, 3)


def feature_extract(df, cols):
    results = {}

    for col in cols:
        signal = df[col].dropna()
        total_row = int(len(df[col]))
        total_non_null = int(signal.count())
        missing_count = int(total_row - total_non_null)
        missing_pct = (missing_count / total_row) * 100 if total_row else 0

        if pd.api.types.is_numeric_dtype(signal):
            results[col] = signal.describe().to_dict()
            results[col]["skew"] = float(skew(signal)) if len(signal) else None
            results[col]["kurtosis"] = float(kurtosis(signal)) if len(signal) else None
            results[col]["rms"] = float(np.sqrt(np.mean(signal**2))) if len(signal) else None
            results[col]["p2p"] = float(np.ptp(signal)) if len(signal) else None
            results[col]["variance"] = float(np.var(signal)) if len(signal) else None

            for key, value in results[col].items():
                if isinstance(value, (int, float)) and value is not None:
                    results[col][key] = format_value(value)

            results[col]["total_non_null"] = total_non_null
            results[col]["missing_count"] = missing_count
            results[col]["missing_pct"] = missing_pct
        else:
            results[col] = signal.describe().to_dict()
            results[col]["total_non_null"] = total_non_null
            results[col]["missing_count"] = missing_count
            results[col]["missing_pct"] = f"{missing_pct:.2f}%"

            for key, value in results[col].items():
                if isinstance(value, (int, float)):
                    results[col][key] = format_value(value)

    return results


def display_configuration(ds, cols, label, results):
    print("\n===== CONFIGURATION =====")
    print(f"\nDataset: {ds}")

    print("\nSelected Columns:")
    for col in cols:
        print(f"{col}")

    print(f"\nLabel Column: {label}")
    print("\nFeature Extraction:")
    pprint(results, indent=4)
