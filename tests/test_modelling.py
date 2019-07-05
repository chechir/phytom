# pylint: disable=invalid-name
# pylint: disable=missing-docstring

import numpy as np
from wutils import modelling


def test_get_id_fold_ixs():
    np.random.seed(0)
    ids = np.array([1, 1, 1, 2, 2, 3])
    ixs = modelling.get_id_fold_ixs(ids, n_fold=2)
    expected = [
        ([3, 4, 5], [0, 1, 2]),
        ([0, 1, 2], [3, 4, 5]),
    ]

    for i, (trn_ixs, val_ixs) in enumerate(ixs):
        assert np.all(np.array(expected[i][0]) == trn_ixs)
        assert np.all(np.array(expected[i][1]) == val_ixs)
