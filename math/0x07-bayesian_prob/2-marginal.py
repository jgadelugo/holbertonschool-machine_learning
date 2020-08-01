#!/usr/bin/env python3
"""Calculate the mean and covariance of a data set"""
import numpy as np


def likelihood(x, n, P):
    """Calculate the likelihood of obtaining this data given various
    hypothetical probabilities of developing sever side effects
        @x:number of patients that develop severe side effects
        @n: total number of patients observed
        @P: a 1D np.ndarray: various hypothetical probabilities of developing
        severe side effects
    Return: 1D np.ndarray - the likelihood of obtaining the data, x and n
    for each probability in P, respectively
    """
    ftl = np.math.factorial
    c = ftl(n) / (ftl(x) * ftl(n - x))

    lh = c * P**x * (1 - P)**(n - x)

    return lh


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining this data with the various
    hypothetical probabilities
        @x:number of patients that develop severe side effects
        @n: total number of patients observed
        @P: 1D np.ndarray: various hypothetical probabilities of developing
        severe side effects
        @Pr: 1D np.ndarray Prior beliefs of P
    Return: 1D np.ndarray - intersection of obtaining x and n with each
    probability in P, respectively
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        err_msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(err_msg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    lh = likelihood(x, n, P)
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")
    return lh * Pr


def marginal(x, n, P, Pr):
    """Calculates the marginal probability of obtaining the data
        @x:number of patients that develop severe side effects
        @n: total number of patients observed
        @P: 1D np.ndarray: various hypothetical probabilities of developing
        severe side effects
        @Pr: 1D np.ndarray Prior beliefs of P
    Return: the marginal probability of obtaining x and n
    """
    return np.sum(intersection(x, n, P, Pr))
