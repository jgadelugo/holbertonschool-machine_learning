#!/usr/bin/env python3
""" performs PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset
    @X: np.ndarray shape(n, d)
        @n: number of data points
        @d: number of dimensions in each point
        * all dimensions have a mean of 0 across all data points
    @var: the fraction of the variance that the PCA transformation
    should maintain
    Return: the weights matrix, W, that maintains var fraction
    of X's original variane
        @W: np.ndarray of shape(d, nd) nd is the new dimensionality of
        transformed X
    """
    n, d = X.shape
    u, s, vh = np.linalg.svd(X)
    cumulative = np.cumsum(s)
    threshold = cumulative[-1] * var
    mask = np.where(threshold > cumulative)
    idx = len(cumulative[mask]) + 1
    W = vh.T
    Wr = W[:, 0: idx]
    return Wr
