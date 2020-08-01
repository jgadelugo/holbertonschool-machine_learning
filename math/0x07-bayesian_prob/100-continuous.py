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
    lh = likelihood(x, n, P)
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


def posterior(x, n, p1, p2):
    """calculates the posterior probability
        @x:number of patients that develop severe side effects
        @n: total number of patients observed
        @P1: lower bound on the range
        @P2: upper bound on the range
    Return: the posterior probability that p is within the range
    [p1, p2] given x and n
    """
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    return intersection(x, n, p1, p2) / marginal(x, n, p1, p2)
