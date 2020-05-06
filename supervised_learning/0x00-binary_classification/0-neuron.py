#!/usr/bin/env python3
"""
Class defines a single neuron performing binary classification
"""
import numpy as np


class Neuron:
    """ class neuron"""
    def __init__(self, nx):
        """ initialize """
        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0

    @property
    def nx(self):
        """ getter for nx"""
        return self.__nx

    @nx.setter
    def nx(self, value):
        """ Setter for nx"""
        if not isinstance(value, int):
            raise TypeError("nx must be an integer")
        elif value < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.__nx = value
