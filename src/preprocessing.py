"""
Data Preprocessing Module - From Scratch Implementations
Course: Machine Learning Practical Assignment
Author: Chetna & Team

This module provides from-scratch implementations of fundamental
data preprocessing techniques without relying on ready-made library functions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union


def calculate_stats_from_scratch(data_series: pd.Series) -> Dict[str, float]:
    """
    Calculates basic descriptive statistics from scratch:
    Minimum, Maximum, Mean, Median, Mode, Variance, Standard Deviation, Range.
    """
    clean_vals = [float(x) for x in data_series if pd.notnull(x)]
    n = len(clean_vals)
    if n == 0:
        return {}

    # Mean: \bar{X} = \frac{1}{n} \sum X_i
    mean_val = sum(clean_vals) / n

    # Min & Max
    min_val = clean_vals[0]
    max_val = clean_vals[0]
    for x in clean_vals:
        if x < min_val:
            min_val = x
        if x > max_val:
            max_val = x
    range_val = max_val - min_val

    # Median: Middle value in sorted array
    sorted_vals = sorted(clean_vals)
    if n % 2 == 1:
        median_val = sorted_vals[n // 2]
    else:
        median_val = (sorted_vals[(n // 2) - 1] + sorted_vals[n // 2]) / 2.0

    # Mode: Most frequently occurring value
    freq_dict: Dict[float, int] = {}
    for x in clean_vals:
        freq_dict[x] = freq_dict.get(x, 0) + 1
    max_freq = max(freq_dict.values())
    mode_val = [k for k, v in freq_dict.items() if v == max_freq][0]

    # Variance: \sigma^2 = \frac{1}{n} \sum (x_i - \bar{x})^2
    var_val = sum((x - mean_val) ** 2 for x in clean_vals) / n

    # Standard Deviation: \sigma = \sqrt{\sigma^2}
    std_val = var_val ** 0.5

    return {
        "Minimum": min_val,
        "Maximum": max_val,
        "Mean": round(mean_val, 4),
        "Median": round(median_val, 4),
        "Mode": round(mode_val, 4),
        "Variance": round(var_val, 4),
        "Standard Deviation": round(std_val, 4),
        "Range": round(range_val, 4)
    }


def detect_duplicates_scratch(dataframe: pd.DataFrame) -> Tuple[int, List[int]]:
    """
    Identifies duplicate rows from scratch without using df.duplicated().
    Uses tuple representation of row values.
    """
    seen_rows = set()
    duplicate_indices = []
    
    for idx, row in dataframe.iterrows():
        row_tuple = tuple(row.values)
        if row_tuple in seen_rows:
            duplicate_indices.append(idx)
        else:
            seen_rows.add(row_tuple)
            
    return len(duplicate_indices), duplicate_indices


def check_invalid_data_scratch(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """
    Checks for impossible values, negative values where not permissible,
    and trailing string inconsistencies.
    """
    report = {
        "impossible_ages": 0,
        "negative_incomes": 0,
        "negative_experience": 0,
        "string_whitespace_issues": {}
    }
    
    # Age check (18 <= Age <= 100)
    if "Age" in dataframe.columns:
        report["impossible_ages"] = sum((dataframe["Age"] < 18) | (dataframe["Age"] > 100))
        
    # Negative numerical checks
    if "MonthlyIncome" in dataframe.columns:
        report["negative_incomes"] = sum(dataframe["MonthlyIncome"] < 0)
    if "TotalWorkingYears" in dataframe.columns:
        report["negative_experience"] = sum(dataframe["TotalWorkingYears"] < 0)
        
    # Categorical whitespace / casing check
    for col in dataframe.select_dtypes(include=['object']).columns:
        whitespace_count = sum(dataframe[col].astype(str).str.strip() != dataframe[col].astype(str))
        if whitespace_count > 0:
            report["string_whitespace_issues"][col] = int(whitespace_count)
            
    return report


def manual_train_test_split(dataframe: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divides dataset into Training and Testing sets from scratch without train_test_split().
    Ensures reproducibility and randomized shuffling.
    """
    np.random.seed(seed)
    n = len(dataframe)
    shuffled_indices = np.random.permutation(n)
    test_size = int(n * test_ratio)
    
    test_idx = shuffled_indices[:test_size]
    train_idx = shuffled_indices[test_size:]
    
    train_df = dataframe.iloc[train_idx].copy().reset_index(drop=True)
    test_df = dataframe.iloc[test_idx].copy().reset_index(drop=True)
    
    return train_df, test_df


def calculate_manual_median(series: pd.Series) -> float:
    """
    Computes median from scratch for a series (handling missing values).
    """
    clean = sorted([float(x) for x in series if pd.notnull(x)])
    n = len(clean)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return clean[n // 2]
    else:
        return (clean[(n // 2) - 1] + clean[n // 2]) / 2.0


def calculate_manual_mode(series: pd.Series) -> Any:
    """
    Computes mode from scratch for a series.
    """
    clean = [x for x in series if pd.notnull(x)]
    if not clean:
        return None
    counts: Dict[Any, int] = {}
    for item in clean:
        counts[item] = counts.get(item, 0) + 1
    max_count = max(counts.values())
    for k, v in counts.items():
        if v == max_count:
            return k
    return None


def manual_label_encode(series: pd.Series, mapping: Dict[Any, int]) -> pd.Series:
    """
    Applies label encoding from scratch using explicit dictionary mapping.
    """
    return series.map(mapping)


def manual_one_hot_encode(df_train: pd.DataFrame, df_test: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies One-Hot Encoding from scratch without OneHotEncoder.
    Categories are learned strictly from df_train to prevent data leakage.
    """
    train_encoded = df_train.copy()
    test_encoded = df_test.copy()
    
    for col in columns:
        unique_categories = sorted(df_train[col].dropna().unique())
        for category in unique_categories:
            new_col_name = f"{col}_{category}"
            train_encoded[new_col_name] = (train_encoded[col] == category).astype(int)
            test_encoded[new_col_name] = (test_encoded[col] == category).astype(int)
            
        train_encoded.drop(columns=[col], inplace=True)
        test_encoded.drop(columns=[col], inplace=True)
        
    return train_encoded, test_encoded


def detect_outliers_iqr(series: pd.Series) -> Tuple[float, float, float, float, float, int]:
    """
    Calculates IQR outlier bounds from scratch:
    IQR = Q3 - Q1, Lower = Q1 - 1.5*IQR, Upper = Q3 + 1.5*IQR.
    """
    clean_vals = sorted([float(x) for x in series if pd.notnull(x)])
    n = len(clean_vals)
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q1_idx = int(0.25 * n)
    q3_idx = int(0.75 * n)
    q1 = clean_vals[q1_idx]
    q3 = clean_vals[q3_idx]
    
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    outliers = [x for x in clean_vals if x < lower_bound or x > upper_bound]
    return q1, q3, iqr, lower_bound, upper_bound, len(outliers)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> Tuple[float, float, int]:
    """
    Calculates Z-score outlier detection from scratch:
    Z = (X - \mu) / \sigma, outliers where |Z| > threshold.
    """
    clean_vals = [float(x) for x in series if pd.notnull(x)]
    n = len(clean_vals)
    mean_val = sum(clean_vals) / n
    var_val = sum((x - mean_val) ** 2 for x in clean_vals) / n
    std_val = var_val ** 0.5
    
    outliers = [x for x in clean_vals if abs((x - mean_val) / std_val) > threshold]
    return mean_val, std_val, len(outliers)


def manual_min_max_scale(train_series: pd.Series, test_series: pd.Series) -> Tuple[pd.Series, pd.Series, float, float]:
    """
    Min-Max Normalization from scratch:
    X' = (X - X_min) / (X_max - X_min)
    Learns X_min and X_max strictly from train_series.
    """
    clean_train = [float(x) for x in train_series if pd.notnull(x)]
    min_val = min(clean_train)
    max_val = max(clean_train)
    range_val = max_val - min_val if (max_val - min_val) != 0 else 1.0
    
    train_scaled = (train_series - min_val) / range_val
    test_scaled = (test_series - min_val) / range_val
    
    return train_scaled, test_scaled, min_val, max_val


def manual_standardization(train_series: pd.Series, test_series: pd.Series) -> Tuple[pd.Series, pd.Series, float, float]:
    """
    Standardization (Z-Score Scaling) from scratch:
    Z = (X - \mu) / \sigma
    Learns \mu and \sigma strictly from train_series.
    """
    clean_train = [float(x) for x in train_series if pd.notnull(x)]
    n = len(clean_train)
    mean_val = sum(clean_train) / n
    var_val = sum((x - mean_val) ** 2 for x in clean_train) / n
    std_val = var_val ** 0.5 if var_val > 0 else 1.0
    
    train_scaled = (train_series - mean_val) / std_val
    test_scaled = (test_series - mean_val) / std_val
    
    return train_scaled, test_scaled, mean_val, std_val
