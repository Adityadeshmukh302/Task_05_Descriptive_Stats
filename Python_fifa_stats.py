# This script loads results.csv, computes descriptive statistics for each column,
# and highlights numeric columns with extreme outliers (min/max > 3 std from mean).

import pandas as pd
import numpy as np

# Load the CSV, parsing the 'date' column as datetime if present
df = pd.read_csv('Data/results.csv', parse_dates=['date'] if 'date' in pd.read_csv('Data/results.csv', nrows=0).columns else None)

# Separate numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = df.columns.difference(numeric_cols)

# --- Numeric columns statistics ---
numeric_stats = []
for col in numeric_cols:
    series = df[col]
    stats = {
        'Column': col,
        'Count': series.count(),
        'Mean': series.mean(),
        'Min': series.min(),
        'Max': series.max(),
        'Std': series.std()
    }
    numeric_stats.append(stats)
numeric_stats_df = pd.DataFrame(numeric_stats)

print("\n=== Numeric Columns Descriptive Statistics ===")
print(numeric_stats_df.to_string(index=False))

# --- Non-numeric columns statistics (including datetime and bool) ---
non_numeric_stats = []
for col in non_numeric_cols:
    series = df[col]
    stats = {
        'Column': col,
        'Type': str(series.dtype),
        'Non-null Count': series.count(),
        'Unique Values': series.nunique(dropna=True),
        'Most Frequent (Mode)': series.mode().iloc[0] if not series.mode().empty else None,
        'Frequency': series.value_counts(dropna=True).iloc[0] if not series.value_counts(dropna=True).empty else None
    }
    non_numeric_stats.append(stats)
non_numeric_stats_df = pd.DataFrame(non_numeric_stats)

print("\n=== Non-Numeric Columns Descriptive Statistics ===")
print(non_numeric_stats_df.to_string(index=False))

# --- Outlier detection for numeric columns ---
outlier_summary = []
for col in numeric_cols:
    series = df[col].dropna()
    if series.empty:
        continue
    mean = series.mean()
    std = series.std()
    min_val = series.min()
    max_val = series.max()
    # Check if min or max is more than 3 std from mean
    min_outlier = abs(min_val - mean) > 3 * std if std > 0 else False
    max_outlier = abs(max_val - mean) > 3 * std if std > 0 else False
    if min_outlier or max_outlier:
        outlier_summary.append({
            'Column': col,
            'Mean': mean,
            'Std': std,
            'Min': min_val,
            'Max': max_val,
            'Min Outlier': min_outlier,
            'Max Outlier': max_outlier
        })

if outlier_summary:
    print("\n=== Numeric Columns with Extreme Outliers (>|3 std| from mean) ===")
    print(pd.DataFrame(outlier_summary).to_string(index=False))
else:
    print("\nNo extreme outliers (>|3 std| from mean) found in numeric columns.")
