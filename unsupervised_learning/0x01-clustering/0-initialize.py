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
    return X[np.random.randint(X.shape[0], size=k)]
