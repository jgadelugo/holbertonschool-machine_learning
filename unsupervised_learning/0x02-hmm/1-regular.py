#!/usr/bin/env python3
"""determines the steady state probabilities of a regular markov chain"""
import numpy as np


def regular(P):
    """determines the steady state probabilities of a regular markov chain
    @P: square 2D np.ndarray shape(n, n) the transition matrix
        @P[i, j]: probability of transitioning from state i to j
        @n: number of states in the markov chain
    Return: np.ndarray shape(1,n) steady state probability or None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n = P.shape[0]
    if P.shape[1] != n or n < 1:
        return None
    if np.any(np.where(P < 0, 1, 0)):
        return None
    try:
        q = (P-np.eye(n))
        ones = np.ones(n)
        q = np.c_[q, ones]
        QTQ = np.dot(q, q.T)
        steady = np.linalg.solve(QTQ, ones)
        if np.any(steady < 0):
            return None
        return np.asarray([steady])
    except Exception as e:
        return None
