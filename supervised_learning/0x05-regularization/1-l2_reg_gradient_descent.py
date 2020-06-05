#!/usr/bin/env python3
"""updates the weights and biases of a neural network using gradient
descent with L2 regularization """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weights and biases of a neural network using
    gradient descent with L2 regularization
    @Y: one-hot np.ndarray - shape (classes, m) - contains the
    correct labels for the data
        @classes: number of classes
        @m: number of data points
    @weights: dictionary of the weights and biases of a neural network
    using gradient descent with L2 regularization
    @cache: dictionary of the outputs of each layer of the neural network
    @alpha: learning rate
    @lambtha: L2 regularization parameter
    @L: number of layers of the network
    """
    w = weights
    m = Y.shape[1]

    for i in range(L, 0, -1):
        if i == L:
            dZ = cache['A' + str(i)] - Y
            dW = np.matmul(dZ, cache["A" + str(i - 1)].T) / m
        else:
            dZ1 = np.matmul(w['W' + str(i + 1)].T, dZ)
            dZ = (1 - cache['A' + str(i)] ** 2) * dZ1
            dW = np.matmul(dZ, cache["A" + str(i - 1)].T) / m

        reg = dW + (lambtha / m) * w['W' + str(i)]
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights['W' + str(i)] = w['W' + str(i)] - (reg * alpha)
        weights['b' + str(i)] = w['b' + str(i)] - (db * alpha)
