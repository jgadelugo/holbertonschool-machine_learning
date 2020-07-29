#!/usr/bin/env python3
"""calculates the symmetric P affinities of a data set"""
import numpy as np


def Q_affinities(Y):
    """ Calculates the Q affinities
    @Y: np.ndarray shape(n, ndim) low dimensional transformation of X
        @n: number of points
        @ndim: new dimensional representation of X
    Return: Q, num
        @Q: np.ndarray shape(n, n) Q affinities
        @num: np.ndarray shape(n, n) numerator of the Q affinities
    """
    n = Y.shape[0]
    sum_Y = np.sum(np.square(Y), 1)
    num = -2 * np.dot(Y, Y.T)
    num = 1 / (1 + np.add(np.add(num, sum_Y).T, sum_Y))
    # sets diagonals
    num[range(n), range(n)] = 0.
    Q = num / np.sum(num)
    return Q, num
