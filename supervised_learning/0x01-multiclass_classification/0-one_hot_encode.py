#!/usr/bin/env python3
""" Creates one-hot matrix """
import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix
    @Y: numpy.ndarray w/ shape (m,) - contains numeric class labels
        @m: is the number of examples
    @classes: is the maximum number of classes found in Y
    Returns: one-hot encoding of Y with shape(classes, m) or None if fails
    """
    if type(Y) is not np.ndarray:
        return None
    size = len(Y)
    if size == 0:
        return None
    if type(classes) is not int or classes <= Y.max():
        return None

    new = np.zeros((classes, size))
    for i in range(size):
        new[Y[i]][i] = 1
    return new
