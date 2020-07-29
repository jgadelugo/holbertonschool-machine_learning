#!/usr/bin/env python3
"""Calculates the cost of the t-SNE transformation"""
import numpy as np


def cost(P, Q):
    """Calculates the cost of the t-SNE transformation
    @P: np.ndarray shape(n, n) the P affinities
    @Q: np.ndarray shape(n, n) the Q affinities
    Return: C, the cost of the transformatio
    """
    Q = np.maximum(Q, 1e-12)
    C = np.sum(P * np.log(np.maximum(P / Q, 1e-12)))
    return C
