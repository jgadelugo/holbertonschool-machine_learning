#!/usr/bin/env python3
"""Initialize cluster centroids for K-mean"""
import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-mean
    @X: np.ndarray shape(n, d) dataset that will be used for K-mean clustering
        @n: number of data points
        @d: number of dimensions for each data point
    @k: positive integer containing the number of clusters
    Return: np.ndarraay shape(k, d) initialized centroids for each cluster,
    or None if failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        # print("Invalid X, must be np.ndarray of shape(n, d)")
        return None
    n, d = X.shape
    if not isinstance(k, int) or k <= 0 or k >= n:
        # print("Invalid k, must be int > 0 and < n")
        return None

    low = X.min(axis=0)
    high = X.max(axis=0)
    return np.random.uniform(low, high, (k, d))
