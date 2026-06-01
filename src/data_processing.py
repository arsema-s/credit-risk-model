"""
Data processing utilities for the Credit Risk Probability Model project.

This module contains reusable helper functions used during
data loading and exploratory data analysis.
"""

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Parameters
    ----------
    file_path : str
        Path to CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(file_path)


def get_dataset_shape(df: pd.DataFrame) -> tuple:
    """
    Return dataset dimensions.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple
        (rows, columns)
    """
    return df.shape


def get_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Calculate missing values for all columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.Series
    """
    return df.isnull().sum()


def get_duplicate_count(df: pd.DataFrame) -> int:
    """
    Count duplicated rows.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    int
    """
    return df.duplicated().sum()

def validate_columns(df, required_columns):
    """
    Ensure all expected columns exist.
    """

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print("Column validation passed.")