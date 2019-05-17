from collections import OrderedDict
import copy
import numpy as np
from wutils import inout as io
from scipy.stats import pearsonr, ks_2samp


# pylint: disable=invalid-name
# pylint: disable=missing-docstring

def ks_feat_selection(train, test, threshold=0.05):
    """ the 2 samples are assumed to be continuos """
    pcol = []
    pval = []
    for col in train.columns:
        pcol.append(col)
        ks_result = ks_2samp(train[col].values, test[col].values)
        pval.append(abs(ks_result.pvalue))
    ixs = np.array(pval) > threshold
    selected_feats = np.array(pcol)[ixs]
    contrary = np.array(pcol)[~ixs]
    print(list(zip(np.array(pval)[ixs], selected_feats))[:10])
    print(list(zip(np.array(pval)[~ixs], contrary))[:10])
    return selected_feats


def pearsonr_feat_selection(df, target, threshold=0.05):
    pcol = []
    pcor = []
    pval = []
    for col in df.columns:
        pcol.append(col)
        pearsonr_result = pearsonr(df[col].values, target.values)
        pcor.append(abs(pearsonr_result[0]))
        pval.append(abs(pearsonr_result[1]))
    selected_feats = np.array(pcol)[np.array(pval) < threshold]
    return selected_feats


def get_consecutive_fold_ixs(df, n_fold=6):
    """ obtain consequtive fold ixs. They have to divide exactly by n
        used by earthquake competition
    """
    n = len(df)
    result = []
    fold_length = n/n_fold
    for i in range(n_fold):
        fold = i + 1
        val_ixs = np.array(range(int(i*fold_length), int(fold_length) * fold))
        train_ixs = np.array(list(set(range(0, n)) - set(val_ixs)))
        assert len(val_ixs) == fold_length
        assert len(train_ixs) == fold_length * (n_fold-1)
        result.append([fold, (train_ixs, val_ixs)])
    return result
