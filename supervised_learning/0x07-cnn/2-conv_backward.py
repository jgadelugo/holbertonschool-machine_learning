#!/usr/bin/env python3
"""performs back propagation over a pooling layer of a neural network"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs back propagation over a pooling layer of a neural network
    """

    # variables of previous layer
    m, h_prev, w_prev, c = A_prev.shape

    return h_prev
