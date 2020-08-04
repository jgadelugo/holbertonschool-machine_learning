#!/usr/bin/env python3
"""Calculates the total intra-cluster variance"""
import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance
    @X: np.ndarray shape (n, d) data set
    @c: np.ndarray shape (k, d) centroid means for each cluster
    Return: var, or None
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        # print("Invalid X, must be np.ndarray of shape(n, d)")
        return None
    n, dx = X.shape
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        # print("Invalid X, must be np.ndarray of shape(n, d)")
        return None
    k, dc = C.shape
    if not isinstance(k, int) or k <= 0 or k >= n or dx != dc:
        # print("Invalid k, must be int > 0 and < n")
        return None
    # sqrt((x1 - X2)^2 + (y1 - y2)^2)
    dist = ((X - C[:, np.newaxis]) ** 2).sum(axis=2)
    min_dist = np.min(dist, axis=0)
    # print(min_dist.sum())
    # print(min_dist.shape)
    var = np.sum(min_dist)
    # print(var)
    return var
