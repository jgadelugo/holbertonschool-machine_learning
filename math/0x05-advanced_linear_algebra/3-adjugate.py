#!/usr/bin/env python3
"""calculate the adjugate matrix of a matrix"""


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
            raise ValueError('matrix must be a non-empty square matrix')

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
    """Calculates the minor of a matrix
    @matrix: list of lists to get minor of
    Return: the minor matrix of matrix
    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')

    if not isinstance(matrix[0], list):
        raise TypeError('matrix must be a list of lists')

    size = len(matrix)

    if size == 1 and len(matrix[0]) == 0:
        raise ValueError('matrix must be a non-empty square matrix')

    for l in matrix:
        if not isinstance(l, list):
            raise TypeError('matrix must be a list of lists')
        if len(l) != size:
            raise ValueError('matrix must be a non-empty square matrix')

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


def cofactor(matrix):
    """ Calculates the cofactor matrix of a matrix
    @matrix: list of lists whose cofactor matrix should be calculated
    Return: cofactor matrix of matrix
    """
    new = minor(matrix)

    size = len(new)

    for i in range(size):
        for j in range(size):
            new[i][j] *= (-1) ** (i + j)
    return new


def adjugate(matrix):
    """calculate the adjugate matrix of a matrix
    @matrix: list of lists whose adjugate matrix should be calculated
    Return: adjugate matrix
    """
    return [list(a) for a in zip(*cofactor(matrix))]
