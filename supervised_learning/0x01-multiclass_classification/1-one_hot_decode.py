#!/usr/bin/env python3
"""Converts a one-hot matrix into a vector of labels"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels
    @one_hot is one-hot encoded numpy.ndarray with shape (classes, m)
        @classes: max number of classes
        @m: number of examples
    Returns: numpy.ndarray with (m,) w/ numeric labels or None if fails
    """
    if one_hot is None:
        return None
    if type(one_hot) is not np.ndarray:
        return None
    if len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
