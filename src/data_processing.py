"""
Data processing utilities for the Credit Risk Probability Model project.

This module contains reusable helper functions used during
data loading and exploratory data analysis.
"""

import pandas as pd

import numpy as np

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


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

def create_aggregate_features(df):
    """
    Create customer-level aggregate features.
    """

    agg_df = (
        df.groupby("CustomerId")
        .agg(
            TotalTransactionAmount=("Amount", "sum"),
            AverageTransactionAmount=("Amount", "mean"),
            TransactionCount=("Amount", "count"),
            StdTransactionAmount=("Amount", "std")
        )
        .reset_index()
    )

    return agg_df

def merge_engineered_features(df):
    """
    Merge aggregate features
    back to transaction dataset.
    """

    agg_df = create_aggregate_features(df)

    df = df.merge(
        agg_df,
        on="CustomerId",
        how="left"
    )

    return df

def extract_time_features(df):
    """
    Extract time-based features.
    """

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["TransactionHour"] = (
        df["TransactionStartTime"]
        .dt.hour
    )

    df["TransactionDay"] = (
        df["TransactionStartTime"]
        .dt.day
    )

    df["TransactionMonth"] = (
        df["TransactionStartTime"]
        .dt.month
    )

    df["TransactionYear"] = (
        df["TransactionStartTime"]
        .dt.year
    )

    return df

def build_feature_pipeline(
    numerical_columns,
    categorical_columns
):
    """
    Build preprocessing pipeline.
    """

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numerical_columns
            ),
            (
                "cat",
                categorical_transformer,
                categorical_columns
            )
        ]
    )

    return preprocessor