#!/usr/bin/env python3
""" Function that Calculates the normalization constants of a matrix"""
import numpy as np


def normalization_constants(X):
    """ Calculates the normalization constants of a matrix
    @X: numpy.ndarray of shape (m, nx) to normalize
        @m: number of data points
        @nx: number of features
    Returns: the mean and standard deviation of each feature
    """
    return (np.mean(X, axis=0), np.std(X, axis=0))
