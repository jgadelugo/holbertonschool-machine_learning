#!/usr/bin/env python3
"""initialize all variables required to calculate the P affinities in t-SNE"""
import numpy as np


def P_init(X, perplexity):
    """
    initialize all variables required to calculate the P affinities in t-SNE
    @X: np.ndarray shape(n, d) containing the dataset
        @n: number of data points
        @d: number of dimensions each point
    @perplexity: perplexity that all Gaussian distributions should have
    Return: (D, P, betas, H)
        @D: np.ndarray shape(n, n) calculates the pairwise distance
        @p: np.ndarray shape(n, n) init to all 0's that will contain the P
        affinities
        @betas: np.ndarray shape(n, 1) init all 1's all of the beta values
        @H: the Shannon entropy for perplexity perplexity
    """
    n, d = X.shape

    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)

    P = np.zeros([n, n])
    betas = np.ones([n, 1])
    H = np.log2(perplexity)
    return (D, P, betas, H)
