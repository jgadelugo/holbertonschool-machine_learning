#!/usr/bin/env python3
""" performs PCA on a dataset"""
import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset
    @X: np.ndarray shape(n, d)
        @n: number of data points
        @d: number of dimensions in each point
    @ndim: new dimensionality of the transformed X
    Return: T, np.ndarray of shape (n, ndim) transformed version of X
    """
    mean = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(mean)

    W = vh.T
    Wr = W[:, 0: ndim]

    T = mean @ Wr
    return T
