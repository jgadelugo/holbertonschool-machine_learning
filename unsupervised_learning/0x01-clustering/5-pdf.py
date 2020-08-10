#!/usr/bin/env python3
"""Calculate the probability density function of Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """Calculate the probability density function of Gaussian distribution
    @X: np.ndarray shape(n,d) data points whose PDF should be evaluated
    @m: np.ndarray shape(d,) mean of distribution
    @S: np.ndarray shape(d,d) covariance of the distribution
    Return: P or None on failure
        @P: np.ndarray of shape(n,) the PDF values for each data point
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    d = X.shape[1]
    if m.shape[0] != d or S.shape[0] != S.shape[1] or S.shape[0] != d:
        return None
    det = np.linalg.det(S)
    p1 = 1 / ((2 * np.pi) ** (d / 2) * det ** 0.5)
    Xm = X - m
    X_t = np.linalg.inv(S) @ Xm.T

    p2 = np.exp(-0.5 * np.sum(Xm * X_t.T, axis=1))
    P = p1 * p2.T
    return np.where(P <= 1e-300, 1e-300, P)
