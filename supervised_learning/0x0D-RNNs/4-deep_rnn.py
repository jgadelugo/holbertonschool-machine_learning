#!/usr/bin/env python3
"""Performs forward propagation for a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for deep RNN
    @rnn_cell: list of RNNCell intances of length l that will be used
    for forward propagation
        @l: number of layers
    @X: data to be used, given as a np.ndarray shape(t, m, i)
        @t: maxium number of time steps
        @m: batch size
        @i: dimensionality of the data
    @h_0: initial hidden state given np.ndarray shape(l, m, h)
        @h: dimensionality of the hidden state
    Return: H, Y
        @H: np.ndarray, all hidden states
        @Y: np.ndarray, all of the outputs
    """
    t = X.shape[0]
    l, m, h = h_0.shape

    H = np.zeros((t+1, l, m, h))
    Y = [0] * t
    H[0] = h_0
    for step in range(t):
        # get state at 0 to pass prev hidden state
        h_state, output = rnn_cells[0].forward(H[step, 0], X[step])
        H[step+1, 0, :, :] = h_state
        for lay in range(1, l):
            h_state, output = rnn_cells[lay].forward(H[step, lay], h_state)
            H[step+1, lay, :, :] = h_state
        Y[step] = output
    return H, np.asarray(Y)
