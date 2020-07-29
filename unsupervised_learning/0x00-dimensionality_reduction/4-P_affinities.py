#!/usr/bin/env python3
"""calculates the symmetric P affinities of a data set"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """calculates the symmetric P affinities of a data set
    @X: np.ndarray shape (n, d) dataset to be transformed by t-SNE
        @n: number of data points
        @d: number of dimensions in each point
    @perplexity: Perplexity all Gaussian distributions should have
    @tol: maximum tolerance allowed (inclusive) for the diff in Shannon
    entropy from perplexity for all Gaussian distribution
    Return: P
        @P: np.ndarray of shape (n, n) containing the symmetric P affinities
    """
    n = X.shape[0]

    D, P, beta, H = P_init(X, perplexity)

    for i in range(n):
        b_min = None
        b_max = None

        Di = D[i]
        Di = np.delete(Di, i, axis=0)

        Hi, Pi = HP(Di, beta[i])

        Hdiff = Hi - H
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                b_min = beta[i]
                if b_max is None:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + b_max) / 2
            else:
                b_max = beta[i]
                if b_min is None:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + b_min) / 2
            Hi, Pi = HP(Di, beta[i])
            Hdiff = Hi - H
        Pi = np.insert(Pi, i, 0)
        P[i] = Pi

    P = (P.T + P) / (2 * n)
    return P
