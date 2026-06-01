"""
Unit tests for data processing utilities.
"""

import pandas as pd

from src.data_processing import (
    get_dataset_shape,
    get_duplicate_count
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