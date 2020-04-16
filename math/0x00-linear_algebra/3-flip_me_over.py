#!/usr/bin/env python3
"""
Return the transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """function that returns the transpose of a 2D matrix"""
    new_matrix = []
    for i in range(len(matrix[0])):
        new_matrix.append([])
        for x in range(len(matrix)):
                new_matrix[i].append(matrix[x][i])
    return new_matrix
