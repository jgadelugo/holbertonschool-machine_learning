#!/usr/bin/env python3
"""determines if a markov chain is absorbing"""
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing
    @P: square 2D np.ndarray shape(n, n) the transition matrix
        @P[i, j]: probability of transitioning from state i to j
        @n: number of states in the markov chain
    Return: True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    n = P.shape[0]
    if P.shape[1] != n or n < 1:
        return False
    if np.any(np.where(P < 0, 1, 0)):
        return False

    diag = np.diag(P)

    if all(diag == 1):
        return True
    if all(diag != 1):
        return False

    check = [int(i) for i in diag]
    for i in range(n):
        for j in range(n):
            if check[j] == 1 and P[i][j] != 0:
                check[i] = 1

    return all(check)
