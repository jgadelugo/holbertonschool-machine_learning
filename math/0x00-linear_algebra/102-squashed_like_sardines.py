#!/usr/bin/env python3
"""Adds two matrices"""


def get_shape(matrix):
    """ gets shape of matrix"""
    shape = ()
    while type(matrix) == list:
        shape += (len(matrix),)
        matrix = matrix[0]
    return shape


def cat_matrices(mat1, mat2, axis=0):
    """Adds two matrices"""
    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    if shape1 != shape2:
        return None

    if len(shape1) == 1:
        return mat1 + mat2

    if len(shape2) == 2:
        new = []
        for i in range(len(mat1)):
            new.append(mat1[i] + mat2[i])
        return new
    return [[]]
