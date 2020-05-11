#!/usr/bin/env python3
"""
Class defines a single neuron performing binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """ class neuron"""
    def __init__(self, nx, layers):
        """ initialize """
        # nx is the number of input features
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        # is the number of layers found in the hidden layer
        if not isinstance(layers, int):
            raise TypeError("layers must be an integer")
        elif layers < 1:
            raise ValueError("layers must be a positive integer")
