#!/usr/bin/env python3
"""calculates the gradients of Y"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Calculates the gradient of Y
    @Y: np.ndarray shape(n, ndim) low dimensional transformation of X
    @P: is a np.ndarray of shape(n, n) containing the P affinities of X
    Return: dY, Q
        @dY: np.ndarray shape(n, ndim) gradients of Y
        @Q: np.ndarray shape (n, n) Q affinities of Y
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)

    PQ = P - Q
    dY = np.zeros((n, ndim))

    for i in range(n):
        tiles = np.tile(PQ[:, i] * num[:, i], (ndim, 1))
        dY[i, :] = np.sum(tiles.T * (Y[i, :] - Y), 0)
    return dY, Q
