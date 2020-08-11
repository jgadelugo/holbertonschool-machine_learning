#!/usr/bin/env python3
"""Calculates the expectation step in the EM algorithm for a GMM"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM
    @X: np.ndarray shape(n, d) data set
    @pi: np.ndarray shape (k,) the priors for each cluster
    @m: np.ndarray of shape(k, d) centroid means for each cluster
    @S: np.ndarray of shape (k, d, d) covariance matrices for each cluster
    Returns: g, l or None, None on failure
        @g: np.ndarray shape(k, n) posterior probs for each data point
        @l: total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if (not isinstance(pi, np.ndarray) or len(pi.shape) != 1 or
        not np.isclose(1, np.sum(pi))):
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    d = X.shape[1]
    k = pi.shape[0]
    if m.shape[1] != d or S.shape[1] != S.shape[2] or S.shape[1] != d:
        return None, None
    if m.shape[0] != S.shape[0] or m.shape[0] != k:
        return None, None
    g = []
    for i in range(k):
        num = pi[i] * pdf(X, m[i], S[i])
        g.append(num)
    g = np.asarray(g)
    g_sum = np.sum(g, axis=0)
    l = np.sum(np.log(g_sum))
    return g / g_sum, l
