"""
Feature Selection Module - From Scratch Implementations
Course: Machine Learning Practical Assignment
Author: Chetna & Team

This module provides from-scratch implementations of fundamental
filter-based feature selection techniques:
1. Variance Threshold
2. Pearson Correlation
3. Chi-Square Test for Independence
4. ANOVA F-Test
5. Mutual Information (Shannon Entropy & Information Gain)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union


def calculate_feature_variance(series: pd.Series) -> float:
    """
    Computes sample variance from scratch:
    Variance(X) = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2
    """
    clean_vals = [float(x) for x in series if pd.notnull(x)]
    n = len(clean_vals)
    if n <= 1:
        return 0.0
    mean_val = sum(clean_vals) / n
    var_val = sum((x - mean_val) ** 2 for x in clean_vals) / n
    return var_val


def manual_variance_threshold(df_train: pd.DataFrame, df_test: pd.DataFrame, threshold: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, float]]:
    """
    Identifies and drops features with variance <= threshold.
    Variances are learned strictly from df_train.
    """
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    variances = {col: calculate_feature_variance(df_train[col]) for col in numeric_cols}
    
    dropped_cols = [col for col, var in variances.items() if var <= threshold]
    train_filtered = df_train.drop(columns=dropped_cols)
    test_filtered = df_test.drop(columns=dropped_cols, errors='ignore')
    
    return train_filtered, test_filtered, dropped_cols, variances


def manual_pearson_correlation(x_series: pd.Series, y_series: pd.Series) -> float:
    """
    Computes Pearson Correlation Coefficient r from scratch:
    r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}
    """
    df_temp = pd.DataFrame({'x': x_series, 'y': y_series}).dropna()
    x = df_temp['x'].tolist()
    y = df_temp['y'].tolist()
    n = len(x)
    if n <= 1:
        return 0.0
        
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    denom_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    
    denominator = (denom_x * denom_y) ** 0.5
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_correlation_matrix_scratch(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a complete Pearson correlation matrix from scratch for numerical columns.
    """
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols, dtype=float)
    
    for c1 in numeric_cols:
        for c2 in numeric_cols:
            if c1 == c2:
                matrix.loc[c1, c2] = 1.0
            else:
                matrix.loc[c1, c2] = manual_pearson_correlation(dataframe[c1], dataframe[c2])
                
    return matrix


def manual_chi_square(feature_series: pd.Series, target_series: pd.Series) -> Dict[str, Any]:
    """
    Computes Chi-Square Test for Independence from scratch:
    \chi^2 = \sum \frac{(O - E)^2}{E}
    where E = \frac{\text{Row Total} \times \text{Column Total}}{\text{Grand Total}}
    Degrees of Freedom: df = (r - 1) * (c - 1)
    """
    contingency = pd.crosstab(feature_series, target_series)
    observed = contingency.values
    r, c = observed.shape
    
    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    grand_total = observed.sum()
    
    expected = np.zeros((r, c))
    contributions = np.zeros((r, c))
    chi2_stat = 0.0
    
    for i in range(r):
        for j in range(c):
            exp_val = (row_sums[i] * col_sums[j]) / grand_total
            expected[i, j] = exp_val
            if exp_val > 0:
                contrib = ((observed[i, j] - exp_val) ** 2) / exp_val
                contributions[i, j] = contrib
                chi2_stat += contrib
                
    df = (r - 1) * (c - 1)
    
    return {
        "contingency_table": contingency,
        "observed": observed,
        "expected": expected,
        "contributions": contributions,
        "chi2_statistic": round(chi2_stat, 4),
        "degrees_of_freedom": df
    }


def manual_anova_f(numerical_series: pd.Series, target_series: pd.Series) -> Dict[str, Any]:
    """
    Computes One-Way ANOVA F-Test from scratch:
    F = \frac{\text{MSB}}{\text{MSW}}
    where:
    SSB = \sum n_j (\bar{X}_j - \bar{X})^2,  MSB = SSB / (k - 1)
    SSW = \sum \sum (X_{ij} - \bar{X}_j)^2,  MSW = SSW / (N - k)
    """
    df_temp = pd.DataFrame({'num': numerical_series, 'target': target_series}).dropna()
    groups = [df_temp[df_temp['target'] == val]['num'].tolist() for val in df_temp['target'].unique()]
    
    k = len(groups)
    all_values = df_temp['num'].tolist()
    N = len(all_values)
    
    overall_mean = sum(all_values) / N
    group_means = [sum(g) / len(g) for g in groups]
    
    # Between-group sum of squares (SSB)
    ssb = sum(len(g) * ((sum(g) / len(g)) - overall_mean) ** 2 for g in groups)
    df_between = k - 1
    msb = ssb / df_between if df_between > 0 else 0.0
    
    # Within-group sum of squares (SSW)
    ssw = sum(sum((x - (sum(g) / len(g))) ** 2 for x in g) for g in groups)
    df_within = N - k
    msw = ssw / df_within if df_within > 0 else 1.0
    
    f_statistic = msb / msw if msw > 0 else 0.0
    
    return {
        "overall_mean": round(overall_mean, 4),
        "group_means": [round(m, 4) for m in group_means],
        "SSB": round(ssb, 4),
        "SSW": round(ssw, 4),
        "MSB": round(msb, 4),
        "MSW": round(msw, 4),
        "df_between": df_between,
        "df_within": df_within,
        "f_statistic": round(f_statistic, 4)
    }


def calculate_entropy_scratch(target_series: pd.Series) -> float:
    """
    Computes Shannon Entropy H(Y) in bits from scratch:
    H(Y) = - \sum P(y) \log_2 P(y)
    """
    clean_target = target_series.dropna()
    n = len(clean_target)
    if n == 0:
        return 0.0
        
    counts: Dict[Any, int] = {}
    for y in clean_target:
        counts[y] = counts.get(y, 0) + 1
        
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * np.log2(p)
            
    return entropy


def manual_mutual_information(feature_series: pd.Series, target_series: pd.Series) -> Dict[str, float]:
    """
    Computes Mutual Information MI(X; Y) in bits from scratch:
    MI(X; Y) = H(Y) - H(Y|X)
    where H(Y|X) = \sum P(x) H(Y | X=x)
    """
    df_temp = pd.DataFrame({'feature': feature_series, 'target': target_series}).dropna()
    n = len(df_temp)
    if n == 0:
        return {"H_Y": 0.0, "H_Y_given_X": 0.0, "mutual_information": 0.0}
        
    h_y = calculate_entropy_scratch(df_temp['target'])
    
    feature_counts = df_temp['feature'].value_counts()
    h_y_given_x = 0.0
    
    for val, count in feature_counts.items():
        p_x = count / n
        subset = df_temp[df_temp['feature'] == val]['target']
        h_subset = calculate_entropy_scratch(subset)
        h_y_given_x += p_x * h_subset
        
    mi = h_y - h_y_given_x
    
    return {
        "H_Y (Total Entropy)": round(h_y, 4),
        "H_Y_given_X (Conditional Entropy)": round(h_y_given_x, 4),
        "Mutual Information (bits)": round(max(0.0, mi), 4)
    }
