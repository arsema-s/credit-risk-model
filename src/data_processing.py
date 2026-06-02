"""
Data processing utilities for the Credit Risk Probability Model project.

This module contains reusable helper functions used during
data loading and exploratory data analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.cluster import KMeans


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

def calculate_rfm(df):
    """
    Calculate Recency,
    Frequency and Monetary
    metrics for each customer.
    """

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    snapshot_date = (
        df["TransactionStartTime"]
        .max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    snapshot_date - x.max()
                ).days
            ),
            Frequency=(
                "TransactionId",
                "count"
            ),
            Monetary=(
                "Amount",
                "sum"
            )
        )
        .reset_index()
    )

    return rfm

def cluster_customers(rfm_df):
    """
    Perform KMeans clustering
    on RFM metrics.
    """

    rfm_df = rfm_df.copy()

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        rfm_df[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    rfm_df["Cluster"] = (
        kmeans.fit_predict(
            scaled_features
        )
    )

    return rfm_df

def identify_high_risk_cluster(
    clustered_rfm
):
    """
    Identify the cluster
    with the weakest customer
    engagement profile.
    """

    cluster_summary = (
        clustered_rfm
        .groupby("Cluster")
        [
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
        .mean()
    )

    cluster_summary["RiskScore"] = (
        cluster_summary["Recency"]
        -
        cluster_summary["Frequency"]
        -
        cluster_summary["Monetary"]
    )

    high_risk_cluster = (
        cluster_summary[
            "RiskScore"
        ].idxmax()
    )

    return high_risk_cluster

def create_high_risk_target(
    clustered_rfm
):
    """
    Create binary
    is_high_risk target.
    """

    clustered_rfm = (
        clustered_rfm.copy()
    )

    high_risk_cluster = (
        identify_high_risk_cluster(
            clustered_rfm
        )
    )

    clustered_rfm[
        "is_high_risk"
    ] = np.where(
        clustered_rfm["Cluster"]
        ==
        high_risk_cluster,
        1,
        0
    )

    return clustered_rfm

def merge_high_risk_target(
    original_df
):
    """
    Merge high risk target
    back into transaction
    dataset.
    """

    rfm = calculate_rfm(
        original_df
    )

    clustered = (
        cluster_customers(
            rfm
        )
    )

    target_df = (
        create_high_risk_target(
            clustered
        )
    )

    merged_df = (
        original_df.merge(
            target_df[
                [
                    "CustomerId",
                    "is_high_risk"
                ]
            ],
            on="CustomerId",
            how="left"
        )
    )

    return merged_df

