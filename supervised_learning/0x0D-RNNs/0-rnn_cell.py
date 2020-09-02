#!/usr/bin/env python3
"""Class that represents a cell of a simple RNN"""
import numpy as np


class RNNCell():
    """Represents a cell of a simple RNN"""
    def __init__(self, i, h, o):
        """Constructor
        @i: dimensionality of the data
        @h: dimensionality of the hidden state
        @o: dimensionality of the outputs
        public instance attributes Wh, Wy, bh, by - weights and biases
            @Wh and bh are for the concatenated hidden state and input data
            @wy and by are for the output
        """
        self.Wh = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """performs forward propagation for one time step
        @h_prev: np.ndarray shape(m, h) containing the previous hidden state
            @h_next: next hidden state
            @y: output of the cell
        @x_t: np.ndarray of shape(m, i) containing the data input for the cell
            @m: batche size for the data
        Retrun: h_next, y
            @h_next: the next hidden state
            @y: output of the cell
        """
        con = np.concatenate((h_prev.T, x_t.T), axis=0)
        h_next = np.tanh((np.matmul(con.T, self.Wh)) + self.bh)
        soft = np.matmul(h_next, self.Wy) + self.by
        # softmax
        x_max = np.max(soft, axis=-1, keepdims=True)
        x_exp = np.exp(soft - x_max)
        y = x_exp / x_exp.sum(axis=-1, keepdims=True)
        return h_next, y
