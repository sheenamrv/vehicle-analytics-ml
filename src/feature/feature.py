import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from pprint import pprint


FEATURE_LABELS = {
    "mean": "Mean",
    "median": "Median",
    "std": "Standard Deviation",
    "variance": "Variance",
    "min": "Min",
    "max": "Max",
    "range": "Range",
    "skew": "Skewness",
    "kurtosis": "Kurtosis",
    "rms": "RMS",
    "energy": "Energy",
    "p2p": "Peak-to-Peak",
    "zcr": "Zero Crossing Rate",
    "unique": "Unique",
    "top": "Top",
    "freq": "Frequency",
    "missing_count": "Missing Count",
    "missing_pct": "Missing Percentage",
}

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


def to_feature_label(feature_key):
    return FEATURE_LABELS.get(str(feature_key), str(feature_key))


def _compute_feature_value(series, feature_key):
    key = str(feature_key)
    total_count = int(len(series))
    missing_count = int(series.isna().sum())

    if key == "unique":
        return int(series.dropna().nunique())
    if key == "top":
        non_null = series.dropna()
        if non_null.empty:
            return np.nan
        modes = non_null.mode(dropna=True)
        return modes.iloc[0] if not modes.empty else np.nan
    if key == "freq":
        non_null = series.dropna()
        if non_null.empty:
            return 0
        return int(non_null.value_counts(dropna=True).iloc[0])

    if key == "missing_count":
        return missing_count
    if key == "missing_pct":
        return (missing_count / total_count) * 100 if total_count else 0.0

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return np.nan

    values = numeric.to_numpy(dtype=float)

    if key == "mean":
        return float(np.mean(values))
    if key == "median":
        return float(np.median(values))
    if key == "std":
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if key == "variance":
        return float(np.var(values))
    if key == "min":
        return float(np.min(values))
    if key == "max":
        return float(np.max(values))
    if key == "range":
        return float(np.max(values) - np.min(values))
    if key == "skew":
        if len(values) <= 2:
            return 0.0
        value = float(skew(values))
        return value if np.isfinite(value) else 0.0
    if key == "kurtosis":
        if len(values) <= 3:
            return 0.0
        value = float(kurtosis(values))
        return value if np.isfinite(value) else 0.0
    if key == "rms":
        return float(np.sqrt(np.mean(values ** 2)))
    if key == "energy":
        return float(np.sum(values ** 2))
    if key == "p2p":
        return float(np.ptp(values))
    if key == "zcr":
        if len(values) < 2:
            return 0.0
        sign_changes = np.count_nonzero(np.signbit(values[1:]) != np.signbit(values[:-1]))
        return float(sign_changes / (len(values) - 1))

    return np.nan


def build_windows(length, window_size_samples, step_samples, keep_partial=False):
    windows = []
    if length <= 0:
        return windows

    start = 0
    while start < length:
        end = start + window_size_samples
        if end <= length:
            windows.append((start, end))
            start += step_samples
            continue

        if not keep_partial:
            break

        # Keep exactly one trailing partial window using remaining samples.
        windows.append((start, length))
        break

    return windows


def build_window_index_table(
    length,
    window_size_samples,
    step_samples,
    keep_partial=False,
    use_time=False,
    sample_rate_hz=1.0,
):
    windows = build_windows(length, window_size_samples, step_samples, keep_partial=keep_partial)
    rows = []

    for index, (start, end) in enumerate(windows, start=1):
        row = {
            "Window Number": index,
            "Start Sample": int(start),
            "End Sample": int(end - 1),
            "Window Samples": int(end - start),
        }
        if use_time:
            row["Start Time (s)"] = float(start / sample_rate_hz)
            row["End Time (s)"] = float((end - 1) / sample_rate_hz)
        rows.append(row)

    return pd.DataFrame(rows)


def extract_windowed_feature_dataset(
    df,
    numeric_columns,
    feature_keys,
    window_size,
    window_unit="samples",
    overlap_pct=0,
    window_type="Fixed Windows",
    keep_partial=False,
    sample_rate_hz=1.0,
):
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")
    if not numeric_columns:
        raise ValueError("Select at least one numeric column for windowed extraction.")
    if window_size <= 0:
        raise ValueError("Window size must be greater than zero.")
    if feature_keys is None or len(feature_keys) == 0:
        raise ValueError("Select at least one feature.")
    if float(overlap_pct) < 0 or float(overlap_pct) > 90:
        raise ValueError("Overlap must be between 0 and 90 percent.")

    unit = str(window_unit).strip().lower()
    use_time = unit == "seconds"
    if use_time:
        if sample_rate_hz <= 0:
            raise ValueError("Sample rate must be greater than zero when window size uses seconds.")
        window_size_samples = max(1, int(round(float(window_size) * float(sample_rate_hz))))
    else:
        window_size_samples = max(1, int(round(float(window_size))))

    if str(window_type).strip().lower().startswith("sliding"):
        step_samples = max(1, int(window_size_samples * (1 - (float(overlap_pct) / 100.0))))
    else:
        step_samples = window_size_samples

    if step_samples <= 0:
        raise ValueError("Step size must be greater than zero.")

    windows = build_windows(len(df), window_size_samples, step_samples, keep_partial=bool(keep_partial))

    rows = []
    for start, end in windows:
        window_slice = df.iloc[start:end]
        if window_slice.empty:
            continue

        row = {
            "Start Sample": int(start),
            "End Sample": int(end - 1),
        }
        if use_time:
            row["Start Time (s)"] = float(start / float(sample_rate_hz))
            row["End Time (s)"] = float((end - 1) / float(sample_rate_hz))

        for col in numeric_columns:
            if col not in window_slice.columns:
                raise ValueError(f"Selected numeric column '{col}' is missing from the dataset.")
            series = window_slice[col]
            for feature_key in feature_keys:
                feature_label = to_feature_label(feature_key)
                value = _compute_feature_value(series, feature_key)
                row[f"{feature_label} {col}"] = value
        rows.append(row)

    for idx, row in enumerate(rows, start=1):
        row["Window Number"] = idx

    ordered_columns = ["Window Number", "Start Sample", "End Sample"]
    if use_time:
        ordered_columns.extend(["Start Time (s)", "End Time (s)"])

    feature_columns = []
    if rows:
        feature_columns = [
            col
            for col in rows[0].keys()
            if col not in ordered_columns
        ]
    dataset = pd.DataFrame(rows)
    if not dataset.empty:
        dataset = dataset[ordered_columns + feature_columns]

    window_table = pd.DataFrame([
        {
            "Window Number": index,
            "Start Sample": int(start),
            "End Sample": int(end - 1),
            "Window Samples": int(end - start),
            **(
                {
                    "Start Time (s)": float(start / sample_rate_hz),
                    "End Time (s)": float((end - 1) / sample_rate_hz),
                }
                if use_time
                else {}
            ),
        }
        for index, (start, end) in enumerate(windows, start=1)
    ])

    metadata = {
        "windowing_enabled": True,
        "window_size": float(window_size),
        "window_unit": "seconds" if use_time else "samples",
        "window_size_samples": int(window_size_samples),
        "step_samples": int(step_samples),
        "overlap_pct": float(overlap_pct),
        "window_type": str(window_type),
        "keep_partial": bool(keep_partial),
        "sample_rate_hz": float(sample_rate_hz),
        "num_windows": int(len(dataset.index)),
        "window_index_table": window_table,
    }
    return dataset, metadata


def display_configuration(ds, cols, label, results):
    print("\n===== CONFIGURATION =====")
    print(f"\nDataset: {ds}")

    print("\nSelected Columns:")
    for col in cols:
        print(f"{col}")

    print(f"\nLabel Column: {label}")
    print("\nFeature Extraction:")
    pprint(results, indent=4)
