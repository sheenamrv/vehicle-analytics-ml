import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from pprint import pprint

def format_value(value):

    if value == 0:
        return 0
    
    if abs(value) < 0.001:
        return f"{value:.3e}"
    
    return round(value,3)
def feature_extract(df, cols):

    results = {}

    for col in cols:

        get_signal = df[col]
        signal = get_signal.dropna()

        total_row = int(len(signal))
        total_non_null = int(signal.count())
        missing_count = int(total_row - total_non_null)
        missing_pct = (missing_count / total_row) * 100
        
        if pd.api.types.is_numeric_dtype(signal):

            results[col] = signal.describe().to_dict()
            results[col]["skew"] = float(skew(signal))
            results[col]["kurtosis"] = float(kurtosis(signal))
            results[col]["rms"] = float(np.sqrt(np.mean(signal**2)))
            results[col]["p2p"] = float(np.ptp(signal))
            results[col]["variance"] = float(np.var(signal))

            for key, value in results[col].items():
                if isinstance(value, (int, float)):
                    results[col][key] = format_value(value)

            # results[col]["total_row"] = total_row
            results[col]["total_non_null"] = total_non_null
            results[col]["missing_count"] = missing_count
            results[col]["missing_pct"] = missing_pct

            # results[col] = {
            #     "median": float(np.median(signal)),
            #     "min": float(np.min(signal)),
            #     "25%" : float(signal.quantile(0.25)),
            #     "mean": float(np.mean(signal)),
            #     "75%" : float(signal.quantile(0.75)),
            #     "max": float(np.max(signal)),
            #     "p2p": float(np.ptp(signal)),
            #     "variance": float(np.var(signal)),
            #     "std": float(np.std(signal)),
            #     "rms": float(np.sqrt(np.mean(signal**2))),
            #     "skewness": float(skew(signal)),
            #     "kurtosis": float(kurtosis(signal))
            # }
        else:

            results[col] = signal.describe().to_dict()

            # results[col]["total_row"] = total_row
            results[col]["total_non_null"] = total_non_null
            results[col]["missing_count"] = missing_count
            results[col]["missing_pct"] = f"{missing_pct:.2f}%"

            for key, value in results[col].items():
                if isinstance(value, (int, float)):
                    results[col][key] = format_value(value)

            # # mode stats
            # val_counts = signal.value_counts()

            # if not val_counts.empty:
            #     top_val = val_counts.index[0]
            # else:
            #     top_val = None

            # results[col] = {
            #     "total_count" : total_row,
            #     "non_null_count" : total_non_null,
            #     "missing_count" : missing_count,
            #     "missing_pct" : missing_pct,
            #     "unique_values" : signal.nunique(),

                
            # }
    
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