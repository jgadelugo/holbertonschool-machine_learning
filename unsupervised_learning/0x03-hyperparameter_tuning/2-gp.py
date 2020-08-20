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

    def predict(self, X_s):
        """Predicts the mean and standard deviation of points in a Gaussian
        process
        @X_s: np.ndarray shape(s, 1) containing all of the points whose mean
        and standard deviation should be calculated
            @s: number of sample points
        Returns: mu, sigma
            @mu: np.ndarray shape(s,) mean for each point in X_s, respectively
            @sigma: np.ndarray shape(s,) standard deviation for each point in
            X_s, respectively
        """
        # update kernel
        k = self.kernel(self.X, self.X)
        k_inv = np.linalg.inv(k)
        k_1 = self.kernel(self.X, X_s)
        k_2 = self.kernel(X_s, X_s)

        # mu -> mean for each point
        mu = k_1.T.dot(k_inv).dot(self.Y).reshape(-1)

        # sigma -> std for each point? -> variance
        cov = k_2 - k_1.T.dot(k_inv).dot(k_1)
        sigma = np.diagonal(cov)

        return mu, sigma

    def update(self, X_new, Y_new):
        """Updates a Gaussian Process
        @X_new: np.ndarray shape(1,) new sample point
        @Y_new: np.ndarray shape(1,) new sample function value
        *Updates the public instance attributes X, Y and K
        """
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
