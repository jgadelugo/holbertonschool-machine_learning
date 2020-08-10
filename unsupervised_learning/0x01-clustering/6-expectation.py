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
        @g: np.ndarray shape(k, n) posterior  probs for each data point
        @l: total log likelihood
    """
    return None, None