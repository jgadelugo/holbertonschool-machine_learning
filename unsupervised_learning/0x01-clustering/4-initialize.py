#!/usr/bin/env python3
"""Initializes variables for a Gaussian mixture model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian mixture model
    @X: np.ndarray shape(n, d) - data set
    @k: pos int, number of clusters
    Returns: pi, m, S or None, None, None on failure
        @pi: np.ndarray shape(k,) containing the priors for each cluster
        , initialized evenly
        @m: np.ndarray shape(k, d) - the centroid means for each cluster,
        initialized with K-means
        @S: np.ndarray of shpae(k, d, d) - covariance matrices for each
        cluster, initialized as identity matrices
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return (None, None, None)

    if not isinstance(k, int) or k <= 0:
        return (None, None, None)

    centroids = kmeans(X, k)[0]

    pi = np.ones(k) / k

    d = X.shape[1]
    S = (np.tile(np.identity(d)[None, :], k)).reshape(k, d, d)

    return pi, centroids, S
