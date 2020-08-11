#!/usr/bin/env python3
"""calculates the maximization step in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """calculates the maximization step in the EM algorithm for a GMM
    @X: np.ndarray shape(n, d) data set
    @g: np.ndarray shape(k, n) posterior probs for each data point in clusters
    Returns: pi, m, S or None, None, None on failure
        @pi: np.ndarray shape(k,) updated priors for each cluster)
        @m: np.ndaray shape(k, d) updated centroid means for each cluster
        @S: np.ndarray of shape(k, d, d) updated covariance matrices for
        each cluster
    """