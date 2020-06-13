#!/usr/bin/env python3
"""performs back propagation over a convolutional layer of a neural network"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """performs back propagation over a convolutional layer of a neural network
    @dZ: np.ndarray, shape (m, h_new, w_new, c_new) containing the partial
    derivatives with respect to the unactivated output of convolutional layer
        @m: number of examples
        @h_new: height of the output
        @w_new: width of the output
        @c_new: number of channels in the output
    @A_prev: np.ndarray, shape (m, h_prev, w_prev, c_new) - containing
    the output of the previous layer
        @m: number of examples
        @h_prev: height of the previous layer
        @w_prev: width of the previous layer
        @c_prev: number of channels in the previous layer
    @W: np.ndarray, shape (kh, kw, c_prev, c_new) containing the kernels
    for the convolution
        @kh: kernel height
        @kw: kernel width
    @b: np.ndarray shape(1, 1, 1, c_new) containing the biases applied to
    the convolution
    @padding: string either 'same' or 'valid', indicating type of padding
    @stride: tuple (sh, sw) containing the strides for the convolution
        @sh: the stride for the height
        @sw: the stride for the width
    @mode: string for either 'max' or 'avg', for max or avg pooling to perform
    Return: output of the pooling layer
    """
    # variables of partial derivatives
    m, h_new, w_new, c_new = dZ.shape

    # variables of previous layer
    m, h_prev, w_prev, c_prev = A_prev.shape

    # variables for the kernel
    kh, kw = W.shape

    # variables for stride
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))
    else:
        ph = 0
        pw = 0

    A_prev = np.pad(A_prev, pad_width=((0, 0),
                                       (ph, ph),
                                       (pw, pw),
                                       (0, 0)),
                    mode='constant', constant_value=0)

    # Initialize dA, dW
    dA = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)

    return A_prev, dA, dW
