#!/usr/bin/env python3
"""Calculates a correlation matrix"""
import numpy as np


def correlation(C):
    """Calculates a correlation matrix
    @C: np.ndarray shape(d, d) containing a covariance matrix
        @d: number of dimensions
    Return: np.ndarray shape(d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    shape = C.shape
    if (len(shape) != 2) or shape[0] != shape[1]:
        raise ValueError("C must be a 2D square matrix")

    diagonal = np.diag(C)

    # standard deviation
    std = np.sqrt(np.expand_dims(diagonal, axis=0))

    correlation = C / np.matmul(std.T, std)

    return correlation
