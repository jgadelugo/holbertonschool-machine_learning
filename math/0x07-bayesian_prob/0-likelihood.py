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

    ftl = np.math.factorial
    c = ftl(n) / (ftl(x) * ftl(n - x))

    likelihood = c * P**x * (1 - P)**(n - x)

    return likelihood
