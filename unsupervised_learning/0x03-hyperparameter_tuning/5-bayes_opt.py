#!/usr/bin/env python3
"""
Class that performs Bayesian optimization on a noiseless 1D Gaussian process
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess

class BayesianOptimization():
    """
    Class that performs Bayesian optimization on a
    noiseless 1D Gaussian process
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Constructor
        @f: the black-box function to be optimized
        @X_init: np.ndarray shape(t,1) inputs already sampled with the
        black-box function
        @Y_init: np.ndarray shape(t,1) outputs of the black-box function for
        each input in X_init
        @t: number of initial sample*s
        @bounds: tuple(min, max) bounds of the space in which to look for the
        optimal point
        @ac_samples: number of samples that should be analyzed during
        acquisition
        @l: length parameter for the kernel
        @sigma_f: standard deviation given to the output of the black-box
        function
        @xsi: exploration-exploitation factor for acquisition
        @minimize: bool determining whether optimization should be performed
        for minimization (True) or maximization (False)
        * sets the following public instance attributes
            @f: the black-box function
            @gp: an instance of the class GaussianProcess
            @X_s: a np.ndarray shape(ac_samples, 1) all acquisition sample
            points, evenly spaced between min and max
            @xsi: exploration-exploitation factor
            @minimize: a bool for minimization versus maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               num=ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates the next best sample location
        *uses the expected improvement acquisition function
        Returns: X_next, EI
            @X_next: np.ndarray shape(1,) representing the next best point
            @EI: np.ndarray shape(ac_samples,) expected improvement of each
            potential sample
        """
        mu, sigma = GP.predict(self.gp, self.X_s)

        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi

        with np.errstate(divide='warn'):
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        X_next = self.X_s[(np.argmax(ei, 0))]

        return X_next, ei

    def optimize(self, iterations=100):
        """Optimizes the black-box function
        @iterations: maximum number of iterations to perform
        * if the next proposed point is one that has already been sampled,
        optimization should be stopped early
        Returns: X_opt, Y_opt
            @X_opt: np.ndarray shape(1,) optimal point
            @Y_opt: np.ndarray shape(1,) optimal function value
        """
        while (iterations):
            X_next = self.acquisition()[0]
            if X_next in self.gp.X:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
            iterations -= 1

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)
        
        X_opt, Y_opt = self.gp.X[idx], self.gp.Y[idx]

        return X_opt, Y_opt
