#!/usr/bin/env python3
"""performs forward propagation over a convolutional layer
of a neural network"""
import numpy as np


def conv_forward(A_prev, W, b,
                 activation, padding="same", stride=(1, 1)):
    """ performs forward propagation over a convolutional layer
    of a neural network
    @A_prev: np.ndarray, shape (m, h_prev, w_prev, c_new) - containing
    the output of the previoues layer
        @m: number of examples
        @h_prev: height of the previous layer
        @w_prev: width of the previous layer
        @c_prev: number of channels in the previous layer
    @W: np.ndarray, shape (kh, kw, c_prev, c_new) - containing the kernel
    for the convolution
        @kh: filter height
        @kw: filter width
        @c_prev: number of channels in the previous layer
        @c_new: number of channels in the ouput
    @b: np.ndarray, shape (1, 1, 1, c_new) containing biases applied to
    the convolution
    @activation: activation function applied to the convolution
    @padding: string that is either same or valid, indicating the type of
    padding used
    @stride: tuple (sh, sw) containing the strides for the convolution
        @sh: the stride for the height
        @sw: the stride for the width
    Return: output of the convolutional layer
    """
    # variables of previous layer
    m, h_prev, w_prev, c_prev = A_prev.shape

    # variables for the kernel
    kh, kw, c_prev, c_new = W.shape

    # variables for stride
    sh, sw = stride

    # padding for 'same' or 'valid'
    if padding == 'same':
        ph = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))
    else:
        ph = 0
        pw = 0

    # output dimensions
    oh = int(((h_prev - kh + (2 * ph)) / sh) + 1)
    ow = int(((w_prev - kw + (2 * pw)) / sw) + 1)

    # img padding
    imgs = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                  mode='constant', constant_values=0)

    # initialize output
    output = np.zeros((m, oh, ow, c_new))

    for i in range(oh):
        for j in range(ow):
            for z in range(c_new):
                # elm-wise mult of kernel and image
                img = imgs[:, i * sh: i * sh + kh, j * sw: j * sw + kw, :]
                output[:, i, j, z] = (W[:, :, :, z] * img).sum(axis=(1, 2, 3))

                output[:, i, j, z] = activation(output[:, i, j, z]
                                                + b[0, 0, 0, z])
    return output
