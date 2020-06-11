#!/usr/bin/env python3
"""performs forward propagation over a pooling layer of a neural network"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs forward propagation over a pooling layer of a neural network
    @A_prev: np.ndarray, shape (m, h_prev, w_prev, c_new) - containing
    the output of the previous layer
        @m: number of examples
        @h_prev: height of the previous layer
        @w_prev: width of the previous layer
        @c_prev: number of channels in the previous layer
    @kernel_shape: tuple (kh, kw) size of the kernel for pooling
        @kh: kernel height
        @kw: kernel width
    @stride: tuple (sh, sw) containing the strides for the convolution
        @sh: the stride for the height
        @sw: the stride for the width
    @mode: string for either 'max' or 'avg', for max or avg pooling to perform
    Return: output of the pooling layer
    """
    # variables of previous layer
    m, h_prev, w_prev, c_prev = A_prev.shape

    # variables for the kernel
    kh, kw = kernel_shape

    # variables for stride
    sh, sw = stride

    # output dimensions
    oh = int(((h_prev - kh) / sh) + 1)
    ow = int(((w_prev - kw) / sw) + 1)

    # initialize output
    output = np.zeros((m, oh, ow, c_prev))

    for i in range(ow):
        for j in range(oh):
            # elm-wise mult of kernel and image
            x = A_prev[:, j * sh: j * sh + kh, i * sw: i * sw + kw]
            if mode == 'max':
                output[:, j, i, :] = np.max(x, axis=(1, 2))
            else:
                output[:, j, i, :] = np.mean(x, axis=(1, 2))
    return output
