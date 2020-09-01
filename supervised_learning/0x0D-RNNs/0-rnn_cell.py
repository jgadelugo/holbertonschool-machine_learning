#!/usr/bin/env python3
"""Class that represents a cell of a simple RNN"""


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
        pass
    

    def forware(self, h_prev, x_t):
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
        pass
