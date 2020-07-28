#!/usr/bin/env python3
"""Calculate the mean and covariance of a data set"""
import numpy as np


def mean_cov(X):
    """Calculate the mean and covariance of a data set
    @X: np.ndarray shape (n, d) containing the data set
        @n: number of data points
        @d: number of dimensions in each data point
    Return: mean, cov
        @mean: np.ndarray shape (1, d) the mean of the data set
        @cov: np.ndarray shape (d, d) the covariance matrix of the data set
    """
    n, d = X.shape

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')

    if n < 2:
        raise ValueError('X must contain multiple data points')
    # mean
    mean = np.sum(X, axis=0) / n

    # derivative
    dev = X - mean

    # covariance
    cov = np.dot(dev.T, dev) / (n - 1)

    return mean, cov
