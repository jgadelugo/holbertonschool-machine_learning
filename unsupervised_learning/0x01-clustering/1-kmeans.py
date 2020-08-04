#!/usr/bin/env python3
"""performs K-means on a dataset"""
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


def get_distance(X, centroids):
    """Get distance centroids are from each data point
    @X: np.ndarray shape(n, d) dataset that will be used for K-mean clustering
        @n: number of data points
        @d: number of dimensions for each data point
    @centroids: np.ndarraay shape(k, d) initialized centroids for each cluster
        @k: positive integer containing the number of clusters
        @d: number of dimensions for each data point
    Return: np.ndarray shape (n, k) distance of centroids from each data point
    """
    x = X[:, :, np.newaxis]
    c = centroids.T[np.newaxis, :, :]
    # print(x.shape)
    # print(c.shape)
    return np.linalg.norm(x - c, axis=1)


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset
    @X: np.ndarray shape(n, d) dataset
        @n: number of data points
        @d: number of dimensions for each data point
    @k: positive integer containing the number of clusters
    Return: C, clss or None, None on failure
        @C: np.ndaray shape(k, d) centroid means for each cluster
        @clss: np.ndarray shape(n,) the index of the cluster in
        C that each data point belongs to
    """
    centroids = initialize(X, k)
    if centroids is None:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    dist = get_distance(X, centroids)
    # print(dist.shape)
    clss = np.argmin(dist, axis=1)
    d = X.shape[1]
    for i in range(iterations):
        cent = np.copy(centroids)
        for j in range(centroids.shape[0]):
            if (X[clss == j].size == 0):
                low = X.min(axis=0)
                high = X.max(axis=0)
                centroids[j, :] = np.random.uniform(low, high, size=(1, d))
            else:
                centroids[j, :] = np.mean(X[clss == j], 0)
        dist = get_distance(X, centroids)
        clss = np.argmin(dist, axis=1)
        if (cent == centroids).all():
            return (centroids, clss)
    return (centroids, clss)
