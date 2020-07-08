#!/usr/bin/env python3
"""Adds two matrices"""


def get_shape(matrix):
    """ gets shape of matrix"""
    shape = ()
    while type(matrix) == list:
        shape += (len(matrix),)
        matrix = matrix[0]
    return shape


def add_two_lists(list1, list2):
    """adds the elements of two lists together"""
    return [x + y for x, y in zip(list1, list2)]


def add_matrices(mat1, mat2):
    """Adds two matrices"""
    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    if shape1 != shape2:
        return None

    if len(shape1) == 1:
        return add_two_lists(mat1, mat2)

    if len(shape2) == 2:
        new = []
        for i in range(len(mat1)):
            new.append(add_two_lists(mat1[i], mat2[i]))
        return new
    return [[]]
