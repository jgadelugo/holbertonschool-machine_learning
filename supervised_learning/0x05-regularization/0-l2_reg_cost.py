#!/usr/bin/env python3
"""Calculate the cost of a neural network with L2 regularization """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculate the cost of a neural network with L2 regularization
    @cost: cost of the network without L2 regularization
    @lambtha: is the regularization parameter
    @weights: dictionary of the weights and biasis (np.ndarray)
    of neural network
    @L: number of layers in the neural network
    @m: number of data points used
    Return: cost of the network accounting for L2 regularization
    """
    f = 0
    while (L):
        index = "W{}".format(L)
        weight = weights[index]
        f += np.linalg.norm(weight)
        L -= 1
    return cost + lambtha / (2 * m) * f
