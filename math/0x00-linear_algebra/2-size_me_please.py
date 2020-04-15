#!/usr/bin/env python3
"""Function that calculates the shape of a matrix"""


def rec_matrix_shape(matrix, shape):
    """recursively get shape of matrix"""
    if type(matrix) == list:
        shape.append(len(matrix))
        rec_matrix_shape(matrix[0], shape)
    return shape


def matrix_shape(matrix):
    """call function to recursively get shape"""
    return rec_matrix_shape(matrix, [])
