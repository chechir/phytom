# pylint: disable=missing-docstring
# pylint: disable=invalid-name

import numpy as np
from wutils.np import get_group_ixs


def categorical_to_numeric(df, column):
    def char_to_numeric(char):
        return str(ord(char))

    def text_to_numeric(text):
        text = str(text).strip()
        text = text[:10]
        text = text.lower()
        numeric_chars = map(char_to_numeric, text)
        result = ''.join(numeric_chars)
        result = float(result)
        return result

    result = map(text_to_numeric, df[column])
    result = np.log(np.array(result))
    return result


def categorical_to_frequency(df, column):
    ixs = get_group_ixs(df[column].values)
    res = np.zeros(len(df))
    for ix in ixs.values():
        res[ix] = len(ix)
    return res.astype(np.int64)
