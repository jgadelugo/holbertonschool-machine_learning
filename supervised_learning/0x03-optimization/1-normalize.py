#!/usr/bin/env python3


def normalize(X, m, s):
    """ Calculates the normalization constants of a matrix
    @X: numpy.ndarray of shape (m, nx) to normalize
        @m: number of data points
        @nx: number of features
    @m: numpy.ndarray, shape (nx,) mean of all features of X
    @s: numpy.ndarray, shape (nx,) standard deviation of all features of X
    Returns: the normalized X matrix
    """
    return (X - m) / s
