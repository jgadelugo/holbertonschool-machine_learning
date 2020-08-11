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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    k, ng = g.shape
    if n != ng:
        return None, None, None
    if int(np.sum(np.sum(g, axis=0))) != n:
        return None, None, None
    # placer
    return None, None, None
    