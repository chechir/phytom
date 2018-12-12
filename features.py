# pylint: disable=invalid-name
from functools import partial
import numpy as np

from wutils import np as wnp


def categorical_to_numeric(df, column):
    """ convert text column into numeric using the character codes """
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
    """ convert categorical column using the frequency of elements """
    ixs = wnp.get_group_ixs(df[column].values)
    res = np.zeros(len(df))
    for ix in ixs.values():
        res[ix] = len(ix)
    return res.astype(np.int64)


def grouped_lagged_decay(df, groupby, col, fillna=-1, decay=1):
    """ Grouped lagged decay """
    values = wnp.fillna(df[col].values, 0)
    f = partial(lagged_decay, decay=decay)
    result = wnp.group_apply(values, df[groupby].values, f)
    result = wnp.fillna(result, fillna)
    return result


def lagged_decay(ordered_values, decay=1):
    """ lagged decay """
    result = np.nan * ordered_values
    previous_value = np.nan
    historic_score = np.nan
    current_score = 0
    for i, value in enumerate(ordered_values):
        if i > 0:
            current_score = previous_value + historic_score * np.exp(-decay)
            result[i] = current_score
        previous_value = value
        historic_score = current_score
    return result


def days_to_first_event(df, groupby, time_col):
    """ Calculate days to the first date for each group, in a Time series """
    dates = df[time_col].astype('datetime64[ns]')
    ids = df[groupby].values
    result = wnp.group_apply(dates, ids, _time_to_min_date)
    result = _convert_ns_to_days(result)
    return result


def _time_to_min_date(v):
    min_date = np.min(v)
    return v - min_date


def _convert_ns_to_days(values):
    return (((values/1000000000)/60)/60)/24
