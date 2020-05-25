#!/usr/bin/env python3
"""Normalize an unactivated output of a neural network
using batch normalization """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ Normalize an unactivated output of a neural network
    using batch normalization
    @Z: numpy.ndarray - shape (m, n) should be normalized
        @m: is the number of data points
        @n: is the number of features in Z
    @gamma: is a numpy.ndarray - shape (1, n) scale used
            for batch normalization
    @beta: numpy.ndarray - shape(1, n) offsets used
            for batch normalization
    @epsilon: small number used to avoid division by zero
    Return: normalized Z matrix
    """
    z_normalize = (Z - np.mean(Z, axis=0)) / np.sqrt(np.var(Z, axis=0) + epsilon)
    return np.multiply(gamma, z_normalize) + beta
