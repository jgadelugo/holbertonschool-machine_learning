#!/usr/bin/env python3
"""Adds two arrays element wise"""


def rec_matrix_shape(matrix, shape):
    """recursively get shape of matrix"""
    if type(matrix) == list:
        shape.append(len(matrix))
        rec_matrix_shape(matrix[0], shape)
    return shape


def matrix_shape(matrix):
    """call function to recursively get shape"""
    return rec_matrix_shape(matrix, [])


def add_arrays(arr1, arr2):
    """Function that adds two arrays element wise"""
    if matrix_shape(arr1) != matrix_shape(arr2):
        return None
    else:
        return [x + y for x, y in zip(arr1, arr2)]
