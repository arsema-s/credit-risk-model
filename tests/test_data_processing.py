"""
Unit tests for data processing utilities.
"""

import pandas as pd

from src.data_processing import (
    get_dataset_shape,
    get_duplicate_count,
    calculate_rfm,
    create_high_risk_target
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

def test_calculate_rfm():

    df = pd.DataFrame(
        {
            "CustomerId": [
                "C1"
            ],
            "TransactionId": [
                "T1"
            ],
            "Amount": [
                100
            ],
            "TransactionStartTime": [
                "2018-01-01"
            ]
        }
    )

    result = calculate_rfm(df)

    assert "Recency" in result.columns

def test_high_risk_column():

    clustered = pd.DataFrame(
        {
            "CustomerId": [
                "A",
                "B",
                "C"
            ],
            "Recency": [
                100,
                10,
                20
            ],
            "Frequency": [
                1,
                10,
                8
            ],
            "Monetary": [
                50,
                1000,
                900
            ],
            "Cluster": [
                0,
                1,
                2
            ]
        }
    )

    result = (
        create_high_risk_target(
            clustered
        )
    )

    assert (
        "is_high_risk"
        in result.columns
    )
