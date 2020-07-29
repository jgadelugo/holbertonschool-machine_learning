#!/usr/bin/env python3
"""calculates the Shannon entropy and P affinities relative to a data point"""
import numpy as np


def HP(Di, beta):
    """
    calculates the Shannon entropy and P affinities relative to a data point
    @Di: np.ndarray of shape(n -1,) containing the pairwise distance between
    a data point and all other points except itself
        @n: the number of data points
    beta: beta value for the Gaussian distribution
    Return: (Hi, Pi)
        @Hi: Shannon entropy of the points
        @Pi: np.ndarray shape (n - 1), the P affinities of the points
    """
    Pi = np.exp(-Di * beta)

    sumP = np.sum(Pi)

    Pi = Pi / sumP
    Hi = -np.sum(Pi * np.log2(Pi))
    return Hi, Pi
