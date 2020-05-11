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

    def evaluate(self, X, Y):
        """ Evaluates the neurons predictions """
        A1, A2 = self.forward_prop(X)
        prob = np.where(A2 <= 0.5, 0, 1)
        return (prob, self.cost(Y, A2))

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron """
        dZ2 = A2 - Y
        dW2 = (dZ2 @ A1.T) / X.shape[1]
        db2 = np.sum(dZ2, axis=1, keepdims=True) / X.shape[1]
        dZ1 = (self.W2.T @ dZ2) * (A1 - (A1 ** 2))
        dW1 = (dZ1 @ X.T) / X.shape[1]
        db1 = np.sum(dZ1, axis=1, keepdims=True) / X.shape[1]

        self.__b1 = self.__b1 - alpha * db1
        self.__W1 = self.__W1 - alpha * dW1
        self.__b2 = self.__b2 - alpha * db2
        self.__W2 = self.__W2 - alpha * dW2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the neural network """
        if isinstance(iterations, int) is False:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if isinstance(alpha, float) is False:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        while iterations:
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            iterations -= 1
        return (self.evaluate(X, Y))
