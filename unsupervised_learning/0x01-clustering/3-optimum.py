#!/usr/bin/env python3
"""tests for the optimum number of clusters by variance"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ tests for the optimum number of clusters by variance
    @X: np.ndarray shape(n, d) data set
    @kmin: pos int - min number of clusters to check for (inclusive)
    @kmax: pos int - max number of clusters to check for (inclusive)
    @iterations: pos int - max number of iterations for K-means
    *should analyze atleast 2 diff cluster sizes
    Returns: results, d_vars, or None, None on failure
        @results: list containing the outputs of K-means for each cluster size
        @d_vars: list - the diff in variance from smallest cluster size
        for each cluster size
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmin, int) or not isinstance(kmax, int):
        return None, None
    if kmin < 1 or kmax < 1 or kmin >= kmax or iterations < 1:
        return None, None

    results, d_vars = [], []
    # get var from smallest cluster
    centroids, clss = kmeans(X, kmin, iterations)
    first_var = variance(X, centroids)

    for i in range(kmin, kmax + 1):
        centroids, clss = kmeans(X, i, iterations)
        results.append((centroids, clss))
        var = variance(X, centroids)
        # get diff
        d_vars.append(first_var - var)

    return results, d_vars
