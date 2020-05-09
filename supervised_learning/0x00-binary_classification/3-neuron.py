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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

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

    @property
    def W(self):
        """ getter for W"""
        return self.__W

    @property
    def b(self):
        """ getter for b"""
        return self.__b

    @property
    def A(self):
        """ getter for A"""
        return self.__A

    def forward_prop(self, X):
        """ calculates the forward propagation of the neuron"""
        val = self.__W @ X + self.__b
        # sigmoid formula
        self.__A = 1/(1 + np.exp(-val))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cst = (1 / m) * ((Y * (np.log(A))) + ((1 - Y) * np.log(1.0000001 - A)))

        return cst.sum()
