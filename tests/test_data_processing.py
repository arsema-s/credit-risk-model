"""
Unit tests for data processing utilities.
"""

import pandas as pd

from src.data_processing import (
    get_dataset_shape,
    get_duplicate_count,
    create_aggregate_features,
    extract_time_features
)

def test_dataset_shape():
    """
    Test shape function.
    """

    df = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [3, 4]
        }
    )

    assert get_dataset_shape(df) == (2, 2)


def test_duplicate_count():
    """
    Test duplicate counting.
    """

    df = pd.DataFrame(
        {
            "A": [1, 1],
            "B": [2, 2]
        }
    )

    assert get_duplicate_count(df) == 1

def test_create_aggregate_features():

    df = pd.DataFrame(
        {
            "CustomerId": [
                "C1",
                "C1",
                "C2"
            ],
            "Amount": [
                100,
                200,
                50
            ]
        }
    )

    result = create_aggregate_features(df)

    assert (
        "TotalTransactionAmount"
        in result.columns
    )

def test_extract_time_features():

    df = pd.DataFrame(
        {
            "TransactionStartTime": [
                "2018-01-01T12:00:00Z"
            ]
        }
    )

    result = extract_time_features(df)

    assert (
        "TransactionHour"
        in result.columns
    )