#!/usr/bin/env python3
"""Function that calculates the shape of a matrix"""


def rec_matrix_shape(matrix, shape):
    if type(matrix) == list:
        shape.append(len(matrix))
        rec_matrix_shape(matrix[0], shape)
    return shape


def matrix_shape(matrix):
    return rec_matrix_shape(matrix, [])
