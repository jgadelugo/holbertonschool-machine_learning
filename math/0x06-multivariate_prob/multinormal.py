#!/usr/bin/env python3
"""Class for MultiNormal that represents a multivariate Normal distribution"""
import numpy as np


class MultiNormal():
    """Class for MultiNormal that represents a multivariate
    Normal distribution"""
    def __init__(self, data):
        """ constructor
        @data: np.ndarray shape(d, n) with data set
            @n: number of data points
            @d: number of dimensions in each data point
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        d, n = data.shape

        if n < 2:
            raise ValueError('data must contain multiple data points')


        self.mean = np.mean(data, axis=1).reshape(d, 1)
        dev = data - self.mean

        self.cov = np.matmul(dev, dev.T) / (n - 1)
