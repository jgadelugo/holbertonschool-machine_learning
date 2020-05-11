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
        self.W1 = np.random.normal(size=(nodes, nx))
        # bias
        self.b1 = np.zeros((nodes, 1))
        # activated output
        self.A1 = 0

        # Output neuron
        # weights vector
        self.W2 = np.random.normal(size=(1, nodes))
        # bias
        self.b2 = 0
        # activated output(prediction)
        self.A2 = 0
