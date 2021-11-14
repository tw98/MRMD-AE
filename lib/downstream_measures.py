import numpy as np
import os, sys, glob
from sklearn.model_selection import KFold
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform, cdist
from brainiak.fcma.util import compute_correlation


def ISC(data):
    """
    Intersubject correlation of timeseries responses across features.

    Parameters:
    ----------
    data: array or list of shape (n_subjects, n_samples, n_features)


    Returns:
    --------
    data: array of shape (n_subjects, n_features) where each entry is the
    correlation of the timeseries at subject_i feature_j compared with the
    group average timeseries at feature_j
    """

    n_subjects, n_timepoints, n_features = np.shape(data)
    iscs = np.empty((n_subjects, n_features))
    for test_subject in range(n_subjects):
        train_subjects = np.setdiff1d(np.arange(n_subjects), test_subject)
        test_data = data[test_subject]
        train_data = np.mean(data[train_subjects], axis=0)
        for feature in range(n_features):
            corr = np.corrcoef(train_data[:, feature], test_data[:, feature])[0, 1]
            iscs[test_subject, feature] = corr
    return iscs


# Take in array or list of shape (n_subjects, n_samples, n_features)
# Also specify how big the time segment is to be matched.

def time_segment_matching(data, window_size=10):
    data = np.swapaxes(np.array(data), 2, 1)  # swap to be (n_subjects, n_features, n_samples)
    nsubjs, ndim, nsamples = data.shape
    accuracy = np.zeros(shape=nsubjs)
    nseg = nsamples - window_size
    trn_data = np.zeros((ndim * window_size, nseg), order='f')
    # the training data also include the test data, but will be subtracted when calculating A
    for m in range(nsubjs):
        for w in range(window_size):
            trn_data[w * ndim:(w + 1) * ndim, :] += data[m][:, w:(w + nseg)]
    for tst_subj in range(nsubjs):
        tst_data = np.zeros((ndim * window_size, nseg), order='f')
        for w in range(window_size):
            tst_data[w * ndim:(w + 1) * ndim, :] = data[tst_subj][:, w:(w + nseg)]
        A = np.nan_to_num(zscore((trn_data - tst_data), axis=0, ddof=1))
        B = np.nan_to_num(zscore(tst_data, axis=0, ddof=1))
        # compute correlation matrix
        corr_mtx = compute_correlation(B.T, A.T)
        # The correlation classifier.
        for i in range(nseg):
            for j in range(nseg):
                # exclude segments overlapping with the testing segment
                if abs(i - j) < window_size and i != j:
                    corr_mtx[i, j] = -np.inf
        max_idx = np.argmax(corr_mtx, axis=1)
        accuracy[tst_subj] = sum(max_idx == range(nseg)) / nseg
    return accuracy
