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

    def pdf(self, x):
        """Calculates the PDF at a data point
        @x: is np.ndarray shape(d, 1) data point whose PDF should be calced
            @d: number of dimensions of the multinomial
        """
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')
        shape = x.shape
        d = self.cov.shape[0]
        if len(shape) != 2:
            raise ValueError("x mush have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        dev = x - self.mean

        const = 1 / np.sqrt(((2 * np.pi) ** d) * det)

        pdf = const * np.exp(np.matmul(np.matmul(-dev.T, inv), dev / 2))
        return pdf.reshape(-1)[0]
