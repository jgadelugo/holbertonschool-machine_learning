#!/usr/bin/env python3
"""Performs forward propagation for a simple RNN"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for simple RNN
    @rnn_cell: instance of RNNCell that will be used for forward propagation
    @X: data to be used, given as a np.ndarray shape(t, m, i)
        @t: maxium number of time steps
        @m: batch size
        @i: dimensionality of the data
    @h_0: initial hidden state given np.ndarray shape(m, h)
        @h: dimensionality of the hidden state
    Return: H, Y
        @H: np.ndarray, all hidden states
        @Y: np.ndarray, all of the outputs
    """
    t = X.shape[0]
    m, h = h_0.shape

    H = np.zeros((t+1, m, h))
    Y = [0] * t
    H[0] = h_0
    for j in range(t):
        H[j+1], Y[j] = rnn_cell.forward(H[j], X[j])

    return H, np.asarray(Y)
