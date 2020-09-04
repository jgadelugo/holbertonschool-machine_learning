#!/usr/bin/env python3
"""Class that represents a bidirectional cell of a RNN"""
import numpy as np


class BidirectionalCell():
    """represents a bidirectional cell of a RNN"""
    def __init__(self, i, h, o):
        """Constructor
        @i: dimensionality of the data
        @h: dimensionality of the hidden state
        @o: dimensionality of the outputs
        public instance attributes Whf, Whb, Wy, bhf, bhb, by
        weights and biases
            @Whf and bhf are for the hidden states in the forward direction
            @Whb and bhb are for the hidden states in the backward direction
            @Wy and by are for the outputs
        """
        self.Whf = np.random.normal(size=(h+i, h))
        self.Whb = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhb = np.zeros((1, h))
        self.bhf = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """performs forward propagation for one time step
        @h_prev: np.ndarray shape(m, h) containing the previous hidden state
        @x_t: np.ndarray of shape(m, i) containing the data input for the cell
            @m: batche size for the data
        Retrun: h_next
            @h_next: the next hidden state
        """
        con = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.matmul(con, self.Whf) + self.bhf)

        return h_next
