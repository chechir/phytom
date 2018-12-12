# pylint: disable=missing-docstring
# pylint: disable=invalid-name
import pandas as pd
import numpy as np
from wutils.features import (
    categorical_to_frequency,
    grouped_lagged_decay,
    days_to_first_event,
)


def test_categorical_to_frequency():
    df = pd.DataFrame({
        'cat': [1, 1, 2, 3, 4, 4, 4],
    })
    result = categorical_to_frequency(df, 'cat')
    expected = [2, 2, 1, 1, 3, 3, 3]
    assert np.allclose(expected, result)


def test_grouped_lagged_decay():
    df = pd.DataFrame({
        'horse_name': ['x', 'x', 'x', 'x'],
        'win_flag': [1, 0, 1, 0]
        })

    result = grouped_lagged_decay(df, 'horse_name', 'win_flag')
    expected = [
        -1, 1,
        (0 + 1*np.e**-1),
        (1 + 0*np.e**-1 + 1*np.e**-2)
    ]
    assert np.allclose(expected, result)


def test_grouped_lagged_decay_with_nans():
    df = pd.DataFrame({
        'horse_name': ['x', 'x', 'x', 'x'],
        'win_flag': [1, np.nan, 0, np.nan]
    })

    result = grouped_lagged_decay(df, 'horse_name', 'win_flag')
    expected = [
        -1, 1,
        (0 + 1*np.e**-1),
        (0 + 0*np.e**-1 + 1*np.e**-2)
    ]
    assert np.allclose(expected, result)


def test_datys_to_first_event():
    df = pd.DataFrame({
        'scheduled_time': np.array(
            ['2011-01-01', '2011-01-02', '2011-01-12']
        ).astype('datetime64[ns]'),
        'horse_name': ['a', 'a', 'a']
    })
    result = days_to_first_event(df, 'horse_name', 'scheduled_time')
    expected = [0, 1, 11]
    assert np.allclose(expected, result)
