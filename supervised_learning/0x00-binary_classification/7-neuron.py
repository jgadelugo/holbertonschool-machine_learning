#!/usr/bin/env python3
"""
Class defines a single neuron performing binary classification
"""
import numpy as np
import matplotlib.pyplot as plt


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
        ct = -(1 / m) * ((Y * (np.log(A))) + ((1 - Y) * np.log(1.0000001 - A)))
        return ct.sum()

    def evaluate(self, X, Y):
        """ Evaluates the neurons predictions """
        prob = np.where(self.forward_prop(X) < 0.5, 0, 1)
        return (prob, self.cost(Y, self.A))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron """
        dZ = A - Y
        dW = (X @ dZ.T) / X.shape[1]
        db = np.sum(dZ) / X.shape[1]
        self.__W = self.__W - (alpha * dW).T
        self.__b = self.__b - (alpha * db).T

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ method to train the neuron """
        if isinstance(iterations, int) is False:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if isinstance(alpha, float) is False:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if isinstance(step, int) is False:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        count = 0
        while iterations:
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            cost = self.evaluate(X, Y)[1]
            if verbose:
                if count == step or count == 0 or count == iterations:
                    print("Cost after {} iterations: {}".format(count, cost))
            if graph:
                if count == step or count == 0 or count == iterations:
                    plt.xlabel('iteration')
                    plt.ylabel('cost')
                    plt.title('Training Cost')
                    plt.plot(cost, 'b')
                    plt.show()
            count += 1
            iterations -= 1
        return self.evaluate(X, Y)
