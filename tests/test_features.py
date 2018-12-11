# pylint: disable=missing-docstring
# pylint: disable=invalid-name
import pandas as pd
import numpy as np
from features import categorical_to_frequency


def test_categorical_to_frequency():
    df = pd.DataFrame({
        'cat': [1, 1, 2, 3, 4, 4, 4],
    })
    result = categorical_to_frequency(df, 'cat')
    expected = [2, 2, 1, 1, 3, 3, 3]
    assert np.allclose(expected, result)
