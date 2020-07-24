#!/usr/bin/env python3
"""Calculates the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix
    @matrix: np.ndarray of shape (n, n) whose definiteness should be calced
    Return: Positive definite, Positive semi-definite, Negative semi-definite,
    Negative definite, or Indefinite"""

    if isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')

    if len(matrix.shape) != 2 or (matrix.shape[0] != matrix.shape[1]):
        return None

    if not np.all(np.transpose(matrix) == matrix):
        return None

    eign, _ = np.linalg.eign(matrix)

    if all(eign > 0):
        return 'Positive definite'
    elif all(eign >= 0):
        return 'Positive semi-definite'
    elif all(eign < 0):
        return 'Negative definite'
    elif all(eign <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
