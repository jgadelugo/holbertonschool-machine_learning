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

    def backward(self, h_next, x_t):
        """performs backward propagation for one time step
        @h_next: np.ndarray shape(m, h) containing the next hidden state
        @x_t: np.ndarray of shape(m, i) containing the data input for the cell
            @m: batche size for the data
        Retrun: h_pev
            @h_pev: the previous hidden state
        """
        con = np.concatenate((h_next, x_t), axis=1)

        h_pev = np.tanh(np.matmul(con, self.Whb) + self.bhb)

        return h_pev

    def output(self, H):
        """ Calculates all outputs for the RNN
        @H: np.ndarray shape (t, m, 2 * h) contains the concatenated hiddden-
        states from both directions, excluding their initialized states
            @t: number of steps
            @m: batch size for the data
            @h: dimensionality of the hidden states
        Return: Y
            @Y: outputs
        """
        t = H.shape[0]
        print()
        Y = [0] * t

        for i in range(t):
            soft = np.matmul(H[i], self.Wy) + self.by
            # softmax
            x_exp = np.exp(soft)
            Y[i] = x_exp / x_exp.sum(axis=-1, keepdims=True)
        return np.asarray(Y)
