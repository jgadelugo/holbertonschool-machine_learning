#!/usr/bin/env python3
"""Calculates the minor of a matrix"""


def determinant(matrix):
    """Calculates the determinant of a matrix
    @matrix: list of lists whose determinant should be calculated
    Return: the determinant of matrix
    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')

    if not isinstance(matrix[0], list):
        raise TypeError('matrix must be a list of lists')

    size = len(matrix)

    if size == 1 and len(matrix[0]) == 0:
        return 1

    for l in matrix:
        if not isinstance(l, list):
            raise TypeError('matrix must be a list of lists')
        if len(l) != size:
            raise ValueError('matrix must be a square matrix')

    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    total = 0
    for i in range(size):
        sub_matrix = matrix[1:]

        for j in range(len(sub_matrix)):
            sub_matrix[j] = sub_matrix[j][0:i] + sub_matrix[j][i+1:]
        sign = (-1) ** (i % 2)
        sub_det = determinant(sub_matrix)
        total += sign * matrix[0][i] * sub_det
    return total


def minor(matrix):
    """Calculates the minor of a matrix"""
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')

    if not isinstance(matrix[0], list):
        raise TypeError('matrix must be a list of lists')

    size = len(matrix)

    if size == 1 and len(matrix[0]) == 0:
        return 1

    for l in matrix:
        if not isinstance(l, list):
            raise TypeError('matrix must be a list of lists')
        if len(l) != size:
            raise ValueError('matrix must be a square matrix')

    new = []

    for i in range(size):
        new.append([])
        for j in range(size):
            sub_matrix = []
            for z in range(size):
                if i == z:
                    continue
                sub = matrix[z][0:j] + matrix[z][j+1:]
                sub_matrix.append(sub)

            if sub_matrix == []:
                    sub_matrix = [[]]

            new[i].append([])
            new[i][j] = determinant(sub_matrix)
    return new
