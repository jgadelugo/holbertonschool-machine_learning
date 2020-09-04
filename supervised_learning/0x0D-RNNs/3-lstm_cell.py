#!/usr/bin/env python3
"""Class that represents an LSTM unit"""
import numpy as np


class LSTMCell():
    """epresents an LSTM unit"""
    def __init__(self, i, h, o):
        """Constructor
        @i: dimensionality of the data
        @h: dimensionality of the hidden state
        @o: dimensionality of the outputs
        public instance attributes Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by
        weights and biases
            @Wf and bf are for the forget gate
            @Wu and bu are for the update gate
            @Wc and bc are for the intermediate cell state
            @Wo and bo are for the output gate
            @Wy and by are for the outputs
        """
        self.Wf = np.random.normal(size=(h+i, h))
        self.Wu = np.random.normal(size=(h+i, h))
        self.Wc = np.random.normal(size=(h+i, h))
        self.Wo = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bo = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bf = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """performs forward propagation for one time step
        @h_prev: np.ndarray shape(m, h) containing the previous hidden state
        @x_t: np.ndarray of shape(m, i) containing the data input for the cell
            @m: batche size for the data
        Retrun: h_next, c_next, y
            @h_next is the next hidden state
            @y: output of the cell
        """
        con = np.concatenate((h_prev, x_t), axis=1)

        # forget gate
        x = np.matmul(con, self.Wf) + self.bf
        # sigmoid
        forget = 1 / (1 + np.exp(-x))

        # update gate
        x = np.matmul(con, self.Wu) + self.bu
        # sigmoid
        update = 1 / (1 + np.exp(-x))

        # output gate
        x = np.matmul(con, self.Wo) + self.bo
        # sigmoid
        out = 1 / (1 + np.exp(-x))

        c = np.tanh(np.matmul(con, self.Wc) + self.bc)

        c_next = forget * c_prev + c * update
        h_next = out * np.tanh(c_next)

        soft = np.matmul(h_next, self.Wy) + self.by
        # softmax
        x_max = np.max(soft, axis=-1, keepdims=True)
        x_exp = np.exp(soft - x_max)
        y = x_exp / x_exp.sum(axis=-1, keepdims=True)
        return h_next, c_next, y
