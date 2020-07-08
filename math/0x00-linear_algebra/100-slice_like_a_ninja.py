#!/usr/bin/env python3
"""
    concatenate two matrices along a specific axis
"""


def np_slice(matrix, axes={}):
    """
    that slices a matrix along a specific axes
    """
    slicer = [slice(None)] * len(matrix.shape)
    for key in axes.keys():
        slicer[key] = slice(*axes[key])
    return matrix[tuple(slicer)]
