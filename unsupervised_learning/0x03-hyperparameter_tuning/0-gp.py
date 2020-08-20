#!/usr/bin/env python3
"""Represents a noiseless 1D Gaussian process"""
import numpy as np


class GaussianProcess():
    """Represents a noiseless 1D Gaussian process"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Class constructor
        @X_init: np.ndarray shape(t, 1) inputs already sampled with the
        black-box funtion
        @Y_init: np.ndarray shape(t, 1) outputs of the black-box function
        for each input in X_init
        @t: number of initial samples
        @l: length parameter for the kernel
        @sigma_f: standard deviation given to the output of the black-box
        function
        *Sets the public instance attributes X, Y, 1 and sigma_f corresponding
        to the respective constructor inputs
        *Sets the public instance attribute K, current coariance kernel matrix
        for the gaussian process
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """ Calculates the covariance kernel matrix between two matrices
        @X1: np.ndarray shape(m, 1)
        @X2: np.ndarray shape(n, 1)
        * the kernel should use the Radial Basis Function (RBF)
        Returns: covariance kernel matrix as a np.ndarray shape(m, n)
        """
        # (x1-x2)^2
        dist = np.sum(X1 ** 2, 1).reshape(-1, 1)\
            + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        K = self.sigma_f ** 2 * np.exp(-dist / (2 * self.l ** 2))
        return K

