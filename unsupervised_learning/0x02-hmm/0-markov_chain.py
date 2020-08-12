#!/usr/bin/env python3

import numpy as np


def markov_chain(P, s, t=1):
    """determines the probability of a markov chain being in a particular
    state after a specified number of iterations
    @P: square 2D np.ndarray of shape (n, n) - transition matrix
        @P[i, j] probability of transitioning from state i to state j
        @n is the number of states in the markov chain
    @s: np.ndarray shape (1, n) probability of starting in each state
    @t: number of iterations that the markov chain has been through
    Returns: np.ndarray shape (1, n)probability of being in a specific state
    after t iterations, or None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n, n1 = P.shape
    if not isinstance(s, np.ndarray) or len(s.shape) != 2:
        return None
    if not isinstance(t, int) or t < 0:
        return None
    if n != n1 or n < 1 or s.shape[0] != 1 or s.shape[1] != n:
        return None
    P = np.linalg.matrix_power(P, t)
    prob = np.matmul(s, P)
    return prob
