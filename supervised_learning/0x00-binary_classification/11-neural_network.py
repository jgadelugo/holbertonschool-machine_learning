#!/usr/bin/env python3
"""
Class defines a single neuron performing binary classification
"""
import numpy as np


class NeuralNetwork:
    """ class neuron"""
    def __init__(self, nx, nodes):
        """ initialize """

        # nx is the number of input features
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        # is the number of nodes found in the hidden layer
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer
        # weights vector
        self.__W1 = np.random.normal(size=(nodes, nx))
        # bias
        self.__b1 = np.zeros((nodes, 1))
        # activated output
        self.__A1 = 0

        # Output neuron
        # weights vector
        self.__W2 = np.random.normal(size=(1, nodes))
        # bias
        self.__b2 = 0
        # activated output(prediction)
        self.__A2 = 0

    @property
    def W1(self):
        """ weights vector for Hidden Layer """
        return self.__W1

    @property
    def b1(self):
        """ bias for Hidden Layer """
        return self.__b1

    @property
    def A1(self):
        """ activated output for Hidden Layer """
        return self.__A1

    @property
    def W2(self):
        """ Weight for Output Neuron """
        return self.__W2

    @property
    def b2(self):
        """ Bias for Output Neuron """
        return self.__b2

    @property
    def A2(self):
        """ Activated output(prediction) for Output Neuron"""
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        val1 = self.W1 @ X + self.b1
        # sigmoid formula
        self.__A1 = 1/(1 + np.exp(-val1))

        val2 = self.W2 @ self.A1 + self.b2
        # sigmoid formula
        self.__A2 = 1/(1 + np.exp(-val2))
        return (self.A1, self.A2)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        ct = -(1 / m) * ((Y * (np.log(A))) + ((1 - Y) * np.log(1.0000001 - A)))
        return ct.sum()
