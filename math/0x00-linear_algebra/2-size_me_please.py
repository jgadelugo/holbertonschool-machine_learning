#!/usr/bin/env python3
"""Function that calculates the shape of a matrix"""
import numpy as np


def matrix_shape(matrix, shape=[], count=0):
    a = np.array(matrix)
    return list(a.shape)
