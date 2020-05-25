#!/usr/bin/env python3
"""Shuffle X and Y to help against over fitting"""
import numpy as np


def shuffle_data(X, Y):
    """ Shuffles X and Y to help against over fitting
    @X: numpy.ndarray of shape (m, nx) to shuffle
        @m: number of data points
        @nx: number of features in X
    @Y: numpy.ndarray shape (m, ny) to shuffle
        @m: Same number of data points in X
        @ny: number of features in Y
    Returns: the shuffled X and Y matrices
    """
    vector = np.arange(X.shape[0])
    index = np.random.permutation(vector)
    return X[index], Y[index]
