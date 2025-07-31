# python_descriptive_stats.py
# This script loads 'results.csv', computes descriptive statistics for each column, and highlights outliers.
# It uses pandas, numpy, and datetime only.

import pandas as pd
import numpy as np

# Load the CSV file, parsing any date columns automatically
# If you know the date column name, you can specify it in parse_dates
# Example: parse_dates=['date_column']
df = pd.read_csv('Data/results.csv', parse_dates=True, infer_datetime_format=True)

print("\n=== Numeric Columns Descriptive Statistics ===\n")
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_stats = []
for col in numeric_cols:
    series = df[col]
    count = series.count()
    mean = series.mean()
    min_val = series.min()
    max_val = series.max()
    std = series.std()
    numeric_stats.append([col, count, mean, min_val, max_val, std])
numeric_df = pd.DataFrame(numeric_stats, columns=[
    'Column', 'Count', 'Mean', 'Min', 'Max', 'Std'])
print(numeric_df.to_string(index=False))

print("\n=== Non-Numeric Columns Descriptive Statistics ===\n")
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
non_numeric_stats = []
for col in non_numeric_cols:
    series = df[col]
    unique_count = series.nunique(dropna=True)
    mode = series.mode().iloc[0] if not series.mode().empty else None
    mode_freq = series.value_counts().iloc[0] if not series.value_counts().empty else None
    non_numeric_stats.append([col, unique_count, mode, mode_freq])
non_numeric_df = pd.DataFrame(non_numeric_stats, columns=[
    'Column', 'Unique Values', 'Most Frequent Value', 'Frequency'])
print(non_numeric_df.to_string(index=False))

# Outlier detection for numeric columns
print("\n=== Outlier Summary (Min/Max > 3 Std from Mean) ===\n")
outlier_rows = []
for col in numeric_cols:
    series = df[col]
    mean = series.mean()
    std = series.std()
    min_val = series.min()
    max_val = series.max()
    if std > 0:
        min_outlier = (mean - min_val) > 3 * std
        max_outlier = (max_val - mean) > 3 * std
        if min_outlier or max_outlier:
            outlier_rows.append([
                col,
                min_val if min_outlier else '',
                max_val if max_outlier else '',
                mean,
                std
            ])
if outlier_rows:
    outlier_df = pd.DataFrame(outlier_rows, columns=[
        'Column', 'Extreme Min', 'Extreme Max', 'Mean', 'Std'])
    print(outlier_df.to_string(index=False))
else:
    print("No extreme outliers detected (min/max > 3 std from mean).")

# End of script
