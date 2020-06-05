#!/usr/bin/env python3
"""conducts forward propagation using Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout
    @X: np.ndarray - shape (nx, m) - containing the input data
        @nx: number of input features
        @m: number of data points
    @weights: dictionary of the weights and biases of network
    @L: number of layers in the network
    @keep_prob: probability and biases of the eural network
    * all layers but the last should use the tanh activation function
    Return: dict containing the output of each layer and dropout
    mask used on each layer
    """
    cache = {'A0': X}

    for i in range(L):
        Z = np.matmul(weights["W" + str(i + 1)],
                      cache["A" + str(i)]) + weights["b" + str(i + 1)]
        drop = np.random.binomial(1, keep_prob, size=Z.shape)

        if i == L - i:
            t = np.exp(Z)
            cache["A" + str(i + 1)] = (t / np.sum(t, axis=0, keepdims=True))
        else:
            cache["A" + str(i + 1)] = np.tanh(Z)
            cache["D" + str(i + 1)] = drop
            cache_a = cache["A" + str(i + 1) * cache["D" + str(i + 1)]]
            cache["A" + str(i + 1)] = cache_a / keep_prob

    return cache
