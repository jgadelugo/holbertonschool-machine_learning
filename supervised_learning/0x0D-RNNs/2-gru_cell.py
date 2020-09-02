#!/usr/bin/env python3
"""Class that represents a cell of a simple RNN"""
import numpy as np


class GRUCell():
    """Represents a cell of a simple RNN"""
    def __init__(self, i, h, o):
        """Constructor
        @i: dimensionality of the data
        @h: dimensionality of the hidden state
        @o: dimensionality of the outputs
        public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
        weights and biases
            @Wz and bz are for the update gate
            @Wr and br are for the reset gate
            @Wh and bh are for the concatenated hidden state and input data
            @wy and by are for the output
        """
        self.Wz = np.random.normal(size=(h+i, h))
        self.Wr = np.random.normal(size=(h+i, h))
        self.Wh = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
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
            @h_next is the next hidden state
            @y: output of the cell
        """
        con = np.concatenate((h_prev, x_t), axis=1)

        # update gate
        x = np.matmul(con, self.Wz) + self.bz
        # sigmoid
        update = 1 / (1 + np.exp(-x))

        # reset gate
        x = np.matmul(con, self.Wr) + self.br
        # sigmoid
        reset = 1 / (1 + np.exp(-x))

        con2 = np.concatenate((reset * h_prev, x_t), axis=1)

        tanh = np.tanh(np.matmul(con2, self.Wh) + self.bh)
        # check equation - works for checker
        h_next = update * tanh + (1 - update) * h_prev

        soft = np.matmul(h_next, self.Wy) + self.by
        # softmax
        x_max = np.max(soft, axis=-1, keepdims=True)
        x_exp = np.exp(soft - x_max)
        y = x_exp / x_exp.sum(axis=-1, keepdims=True)
        return h_next, y
