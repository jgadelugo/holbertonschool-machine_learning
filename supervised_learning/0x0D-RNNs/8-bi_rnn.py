#!/usr/bin/env python3
"""Performs forward propagation for a bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Performs forward propagation for bidirectional RNN
    @bi_cell: instance of BidirectionalCell that will be used for the forward
    propagation
    @X: data to be used, given as a np.ndarray shape(t, m, i)
        @t: maxium number of time steps
        @m: batch size
        @i: dimensionality of the data
    @h_0: initial hidden state in the forward direction, np.ndarray
    shape(m, h)
        @h: dimensionality of the hidden state
    @h_t: initial hidden state in the backward direction, np.ndarray
    shape(m, h)
    Return: H, Y
        @H: np.ndarray, all the concatenated hidden states
        @Y: np.ndarray, all of the outputs
    """
    t = X.shape[0]

    H_f_state = [0] * (t + 1)
    H_b_state = [0] * (t + 1)

    H_f_state[0] = h_0
    H_b_state[t] = h_t

    for i in range(t):
        H_f_state[i + 1] = bi_cell.forward(H_f_state[i], X[i])

    for i in range(t - 1, -1, -1):
        H_b_state[i] = bi_cell.backward(H_b_state[i+1], X[i])

    H_f_state = np.array(H_f_state[1:])
    H_b_state = np.array(H_b_state[:-1])

    H = np.concatenate((H_f_state, H_b_state), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
